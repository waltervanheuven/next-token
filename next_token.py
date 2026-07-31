#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "torch",
#   "transformers",
#   "sentencepiece",
#   "hf_xet",
# ]
# ///

"""Next-token metrics and optional sampled continuation generation.

The default mode calculates entropy, surprisal, and next-token/next-word
predictions for the words in a supplied sentence.  ``--until-stop`` switches
to continuation generation and stops at the first configured stop character.
"""

from __future__ import annotations

import argparse
import os
import string
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EPSILON = 1e-20
OUTPUT_DECIMAL_PLACES = 4


def format_decimal(value: float) -> str:
    return f"{value:.{OUTPUT_DECIMAL_PLACES}f}"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(parsed) or parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and no greater than 1")
    return parsed


def set_seed(seed: int | None) -> None:
    """Seed NumPy and PyTorch when a seed was supplied."""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_word_start(token_text: str) -> bool:
    stripped = token_text.lstrip()
    return (
        bool(stripped)
        and stripped[0].isalpha()
        and all(char.isalpha() or char in "'-" for char in stripped)
    )


def is_word_continuation(token_text: str) -> bool:
    return (
        bool(token_text)
        and not token_text[0].isspace()
        and all(char.isalpha() or char in "'-" for char in token_text)
    )


def load_causal_lm_model(settings: dict[str, Any]) -> None:
    model_name = settings["CAUSAL_LM_MODEL_NAME"]
    try:
        settings["CAUSAL_LM_MODEL"] = AutoModelForCausalLM.from_pretrained(model_name)
        settings["CAUSAL_LM_TOKENIZER"] = AutoTokenizer.from_pretrained(model_name)
        settings["DEVICE"] = get_device()
        settings["CAUSAL_LM_MODEL"].to(settings["DEVICE"])
        settings["CAUSAL_LM_MODEL"].eval()
    except Exception as exc:
        print(f"Error loading causal language model {model_name}: {exc}", file=sys.stderr)
        sys.exit(1)


def get_model_parts(settings: dict[str, Any]):
    if settings["CAUSAL_LM_MODEL"] is None or settings["CAUSAL_LM_TOKENIZER"] is None:
        load_causal_lm_model(settings)
    return (
        settings["CAUSAL_LM_MODEL"],
        settings["CAUSAL_LM_TOKENIZER"],
        settings["DEVICE"],
    )


def sampling_probabilities(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Return a temperature-scaled, nucleus-filtered probability distribution."""
    scaled_logits = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the first token above the threshold, as in standard nucleus
        # sampling, so the retained set is never empty.
        remove = cumulative_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits.scatter_(0, sorted_indices, sorted_logits)
        return torch.softmax(filtered_logits, dim=-1)
    return torch.softmax(scaled_logits, dim=-1)


def calculate_metrics(
    settings: dict[str, Any],
    context: str,
    target_word: str,
    top_n: int = 5,
    top_words: bool = False,
    beam_width: int = 25,
    max_word_tokens: int = 5,
    sample_mode: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    model, tokenizer, device = get_model_parts(settings)

    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits.float(), dim=-1)

    # Entropy (log2, bits) over the model's full next-token distribution.
    entropy = float(
        -torch.sum(next_token_probs * torch.log2(next_token_probs + EPSILON)).item()
    )

    # Surprisal is the sum over all model tokens making up the target word.
    target_ids = tokenizer.encode(" " + target_word.strip(), add_special_tokens=False)
    if not target_ids:
        surprisal = float("inf")
    else:
        prev_token_id = target_ids[0]
        surprisal = float(
            -torch.log2(next_token_probs[prev_token_id] + EPSILON).item()
        )
        target_input_ids = input_ids
        for target_token_id in target_ids[1:]:
            next_id = torch.tensor([[prev_token_id]], device=device)
            target_input_ids = torch.cat([target_input_ids, next_id], dim=1)
            with torch.inference_mode():
                outputs = model(target_input_ids)
                token_logits = outputs.logits[0, -1, :]
                token_probs = torch.softmax(token_logits.float(), dim=-1)
            surprisal += float(
                -torch.log2(token_probs[target_token_id] + EPSILON).item()
            )
            prev_token_id = target_token_id

    if top_words:
        top_preds, word_entropy = predict_next_words(
            model,
            tokenizer,
            input_ids,
            device,
            top_n,
            beam_width,
            max_word_tokens,
        )
    elif sample_mode:
        # In metric mode, --sample returns independent samples rather than a
        # ranked top-N list. Duplicate sampled tokens are retained deliberately.
        sample_probs = sampling_probabilities(next_token_logits, temperature, top_p)
        sampled_ids = torch.multinomial(sample_probs, top_n, replacement=True)
        top_preds = [
            (tokenizer.decode([int(token_id)]), float(sample_probs[token_id].item()))
            for token_id in sampled_ids
        ]
        word_entropy = None
    else:
        topk_idx = torch.argsort(next_token_probs, descending=True)[:top_n]
        top_preds = [
            (tokenizer.decode([int(idx)]), float(next_token_probs[idx].item()))
            for idx in topk_idx
        ]
        word_entropy = None

    return entropy, surprisal, word_entropy, top_preds


def predict_next_words(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
    top_n: int,
    beam_width: int,
    max_word_tokens: int,
) -> tuple[list[tuple[str, float]], float]:
    beams = [(input_ids, "", 0.0, False)]
    completed: dict[str, float] = {}

    for _ in range(max_word_tokens + 1):
        expanded = []
        for beam_input_ids, word, logprob, started in beams:
            with torch.inference_mode():
                outputs = model(beam_input_ids)
                logits = outputs.logits[0, -1, :]
                token_logprobs = torch.log_softmax(logits.float(), dim=-1)
                top_logprobs, top_indices = torch.topk(token_logprobs, beam_width)

            for token_logprob, token_id in zip(top_logprobs, top_indices):
                token_id_int = int(token_id.item())
                token_text = tokenizer.decode([token_id_int])

                if not started:
                    if not is_word_start(token_text):
                        continue
                    next_id = torch.tensor([[token_id_int]], device=device)
                    next_input_ids = torch.cat([beam_input_ids, next_id], dim=1)
                    expanded.append(
                        (
                            next_input_ids,
                            token_text.lstrip(),
                            logprob + token_logprob.item(),
                            True,
                        )
                    )
                elif is_word_continuation(token_text):
                    next_id = torch.tensor([[token_id_int]], device=device)
                    next_input_ids = torch.cat([beam_input_ids, next_id], dim=1)
                    expanded.append(
                        (
                            next_input_ids,
                            word + token_text,
                            logprob + token_logprob.item(),
                            True,
                        )
                    )
                elif word:
                    completed[word] = max(completed.get(word, float("-inf")), logprob)

        beams = sorted(expanded, key=lambda item: item[2], reverse=True)[:beam_width]
        if not beams:
            break

    for _, word, logprob, started in beams:
        if started and word:
            completed[word] = max(completed.get(word, float("-inf")), logprob)

    if not completed:
        return [], float("nan")

    logprobs = np.array(list(completed.values()))
    max_logprob = np.max(logprobs)
    probs = np.exp(logprobs - max_logprob)
    probs = probs / np.sum(probs)
    word_entropy = float(-np.sum(probs * np.log2(probs + EPSILON)))

    ranked = sorted(completed.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [(word, float(np.exp(logprob))) for word, logprob in ranked], word_entropy


def first_stop_index(text: str, stop_chars: str) -> int | None:
    indices = [text.find(character) for character in stop_chars if text.find(character) >= 0]
    return min(indices) if indices else None


def generate_until_stop(
    settings: dict[str, Any],
    context: str,
    sample_mode: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_chars: str,
) -> tuple[str, bool]:
    """Generate a continuation until a stop character or the token cap."""
    model, tokenizer, device = get_model_parts(settings)
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(input_ids).logits[0, -1, :]

        if sample_mode:
            probs = sampling_probabilities(logits, temperature, top_p)
            token_id = int(torch.multinomial(probs, 1).item())
        else:
            token_id = int(torch.argmax(logits).item())

        generated_ids.append(token_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[token_id]], device=device)], dim=1
        )

        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        stop_index = first_stop_index(continuation, stop_chars)
        if stop_index is not None:
            return continuation[: stop_index + 1], True

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            return continuation, True

    return tokenizer.decode(generated_ids, skip_special_tokens=True), False


def starts_alpha_word(token_text: str) -> bool:
    stripped = token_text.lstrip()
    return bool(stripped) and stripped[0].isalpha()


def complete_word_token(token_text: str) -> bool:
    stripped = token_text.lstrip()
    return starts_alpha_word(token_text) and all(
        character.isalpha() or character in "'-" for character in stripped
    )


def generate_next_words(
    settings: dict[str, Any],
    context: str,
    number_of_words: int,
    sample_mode: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[str, bool]:
    """Generate exactly N lexical words, up to the token safety cap.

    Word boundaries are inferred from decoded tokenizer pieces. Alphabetic
    pieces joined without whitespace are treated as one word; punctuation is
    included in the returned continuation but is not counted as a word.
    """
    model, tokenizer, device = get_model_parts(settings)
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    generated_ids: list[int] = []
    continuation = ""
    words_started = 0

    # A token without leading whitespace can continue the final word of the
    # prompt (e.g. GPT-2's possessive token "'s").
    inside_word = bool(context) and context[-1].isalpha()

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(input_ids).logits[0, -1, :]

        if sample_mode:
            probs = sampling_probabilities(logits, temperature, top_p)
            token_id = int(torch.multinomial(probs, 1).item())
        else:
            token_id = int(torch.argmax(logits).item())

        token_text = tokenizer.decode([token_id], skip_special_tokens=True)

        # A leading space marks the end of the current word. If the requested
        # count has already been reached, do not consume the next word token.
        if inside_word and token_text[:1].isspace():
            if words_started >= number_of_words:
                return continuation, True
            inside_word = False

        if inside_word:
            continuation += token_text
            generated_ids.append(token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token_id]], device=device)], dim=1
            )
            if not is_word_continuation(token_text):
                inside_word = False
                if words_started >= number_of_words:
                    return continuation, True
        elif starts_alpha_word(token_text):
            continuation += token_text
            generated_ids.append(token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token_id]], device=device)], dim=1
            )
            words_started += 1
            inside_word = complete_word_token(token_text)
            if words_started >= number_of_words and not inside_word:
                return continuation, True
        else:
            continuation += token_text
            generated_ids.append(token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token_id]], device=device)], dim=1
            )

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            return continuation, words_started >= number_of_words

    return continuation, words_started >= number_of_words and not inside_word


def read_input_lines(file_path: str, context: str) -> list[str]:
    if not file_path:
        return [context]
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def format_predictions(top_preds: list[tuple[str, float]]) -> str:
    fields: list[str] = []
    for word, probability_value in top_preds:
        fields.extend([repr(word), format_decimal(probability_value)])
    return "\t".join(fields)


def process_sentences(
    settings: dict[str, Any],
    lines: list[str],
    keep_punctuation_and_case: bool,
    top_n: int,
    top_words: bool,
    beam_width: int,
    max_word_tokens: int,
    sample_mode: bool,
    temperature: float,
    top_p: float,
) -> None:
    print(f"Model: {settings['CAUSAL_LM_MODEL_NAME']}")
    print(f"Sampling: {sample_mode}")
    if sample_mode:
        print(f"Temperature: {temperature}")
        print(f"Top-p: {top_p}")
    print()

    header = f"{'WordID'}\t{'SentenceNr'}\t{'WordNr'}\t{'Target'}\t{'Entropy'}\t{'Surprisal'}"
    if top_words:
        header += f"\t{'WordEntropyApprox'}"
    header += "\tPredictions"
    print(header)

    word_id = 0
    for line_number, line in enumerate(lines, start=1):
        words = line.split()
        context = ""
        word_number = 1
        for target in words:
            target = target.strip()
            if not target:
                continue
            if len(target) == 1 and not target.isalnum():
                continue
            if context == "":
                context = target
                word_id += 1
                word_number += 1
                continue

            entropy, surprisal, word_entropy, top_preds = calculate_metrics(
                settings,
                context,
                target,
                top_n,
                top_words,
                beam_width,
                max_word_tokens,
                sample_mode,
                temperature,
                top_p,
            )

            if keep_punctuation_and_case:
                printed_target = repr(target)
            else:
                printed_target = target.lower().strip(string.punctuation)

            output = (
                f"{word_id}\t{line_number}\t{word_number}\t{printed_target}"
                f"\t{format_decimal(entropy)}\t{format_decimal(surprisal)}"
            )
            if top_words:
                output += f"\t{format_decimal(word_entropy)}"
            output += f"\t{format_predictions(top_preds)}"
            print(output)

            context = f"{context} {target}"
            word_number += 1
            word_id += 1


def process_continuations(
    settings: dict[str, Any],
    lines: list[str],
    sample_mode: bool,
    temperature: float,
    top_p: float,
    seed: int | None,
    number_of_samples: int,
    max_new_tokens: int,
    stop_chars: str,
) -> None:
    print(f"Model: {settings['CAUSAL_LM_MODEL_NAME']}")
    print("Mode: continuation until stop")
    print(f"Stop characters: {repr(stop_chars)}")
    print(f"Sampling: {sample_mode}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Seed: {seed}")
    print()
    print("SentenceNr\tSampleNr\tStopped\tContinuation")

    for sentence_number, line in enumerate(lines, start=1):
        for sample_number in range(1, number_of_samples + 1):
            continuation, stopped = generate_until_stop(
                settings,
                line,
                sample_mode,
                temperature,
                top_p,
                max_new_tokens,
                stop_chars,
            )
            print(
                f"{sentence_number}\t{sample_number}\t{stopped}\t"
                f"{continuation.strip()}"
            )


def process_next_words(
    settings: dict[str, Any],
    lines: list[str],
    sample_mode: bool,
    temperature: float,
    top_p: float,
    seed: int | None,
    number_of_words: int,
    number_of_samples: int,
    max_new_tokens: int,
) -> None:
    print(f"Model: {settings['CAUSAL_LM_MODEL_NAME']}")
    print(f"Mode: next {number_of_words} words")
    print(f"Sampling: {sample_mode}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Seed: {seed}")
    print()
    print("SentenceNr\tSampleNr\tCompleted\tContinuation")

    for sentence_number, line in enumerate(lines, start=1):
        for sample_number in range(1, number_of_samples + 1):
            continuation, completed = generate_next_words(
                settings,
                line,
                number_of_words,
                sample_mode,
                temperature,
                top_p,
                max_new_tokens,
            )
            print(
                f"{sentence_number}\t{sample_number}\t{completed}\t"
                f"{continuation.strip()}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate next-token metrics, generate until a sentence stop, "
            "or generate the next N words."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", dest="file_path", help="Input text file")
    input_group.add_argument("-s", "--sentence", dest="sentence", help="Input sentence")

    parser.add_argument(
        "-r",
        "--rawtarget",
        dest="keep_punctuation_and_case",
        action="store_true",
        help="Preserve target punctuation and case in metric-mode output",
    )
    parser.add_argument(
        "-n",
        "--ntop",
        dest="top_n",
        type=positive_int,
        default=5,
        metavar="N",
        help="Number of predictions/samples to show (default: 5)",
    )
    parser.add_argument(
        "--top-words",
        action="store_true",
        help="Show beam-searched next-word predictions in metric mode",
    )
    parser.add_argument(
        "--beam-width",
        type=positive_int,
        default=25,
        metavar="WIDTH",
        help="Beam width for --top-words (default: 25)",
    )
    parser.add_argument(
        "--max-word-tokens",
        type=positive_int,
        default=5,
        metavar="TOKENS",
        help="Maximum model tokens per predicted word (default: 5)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Enable stochastic sampling. In --until-stop mode this samples "
            "the continuation; otherwise it draws N next-token samples."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=1.0,
        help="Sampling temperature, greater than 0 (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=probability,
        default=1.0,
        help="Nucleus-sampling probability, in (0, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=None,
        help="Random seed for reproducible sampling (default: unset)",
    )
    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument(
        "--until-stop",
        action="store_true",
        help="Generate continuation(s) until the first stop character",
    )
    generation_group.add_argument(
        "--next-words",
        type=positive_int,
        default=None,
        metavar="N",
        help="Generate exactly the next N words",
    )
    parser.add_argument(
        "--num-samples",
        type=positive_int,
        default=1,
        metavar="N",
        help="Number of continuations per input in generation modes (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=100,
        metavar="TOKENS",
        help="Safety cap for --until-stop generation (default: 100)",
    )
    parser.add_argument(
        "--stop-chars",
        default=".",
        help="Characters that end --until-stop generation (default: '.')",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        default="openai-community/gpt2",
        help="Causal language model name (default: openai-community/gpt2)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.stop_chars:
        parser.error("--stop-chars must contain at least one character")
    if args.top_words and (args.until_stop or args.next_words is not None):
        parser.error("--top-words is only available in metric mode")
    if args.top_words and args.sample:
        parser.error("--sample cannot be combined with --top-words in metric mode")

    set_seed(args.seed)
    settings = {
        "CAUSAL_LM_MODEL_NAME": args.model_name,
        "CAUSAL_LM_MODEL": None,
        "CAUSAL_LM_TOKENIZER": None,
        "DEVICE": None,
    }

    try:
        lines = read_input_lines(args.file_path or "", args.sentence or "")
    except FileNotFoundError as exc:
        parser.error(f"file not found: {exc.args[0]}")

    if args.until_stop:
        process_continuations(
            settings,
            lines,
            args.sample,
            args.temperature,
            args.top_p,
            args.seed,
            args.num_samples,
            args.max_new_tokens,
            args.stop_chars,
        )
    elif args.next_words is not None:
        process_next_words(
            settings,
            lines,
            args.sample,
            args.temperature,
            args.top_p,
            args.seed,
            args.next_words,
            args.num_samples,
            args.max_new_tokens,
        )
    else:
        process_sentences(
            settings,
            lines,
            args.keep_punctuation_and_case,
            args.top_n,
            args.top_words,
            args.beam_width,
            args.max_word_tokens,
            args.sample,
            args.temperature,
            args.top_p,
        )


if __name__ == "__main__":
    main()
