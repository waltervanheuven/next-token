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

import os
import sys
import argparse
import string
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed

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
    model_name = settings['CAUSAL_LM_MODEL_NAME']
    try:
        settings['CAUSAL_LM_MODEL'] = AutoModelForCausalLM.from_pretrained(model_name)
        settings['CAUSAL_LM_TOKENIZER'] = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading causal language model {model_name}: {e}")
        sys.exit(1)

def calculate_metrics(
    settings: dict[str, Any],
    context: str,
    target_word: str,
    top_n: int = 5,
    top_words: bool = False,
    beam_width: int = 25,
    max_word_tokens: int = 5,
):
    if settings['CAUSAL_LM_MODEL'] is None or settings['CAUSAL_LM_TOKENIZER'] is None:
        load_causal_lm_model(settings)

    model = settings['CAUSAL_LM_MODEL']
    tokenizer = settings['CAUSAL_LM_TOKENIZER']
    device = get_device()
    model.to(device)

    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()

    # ENTROPY (log2, bits): all tokens
    entropy = -np.sum(next_token_probs * np.log2(next_token_probs + 1e-20))

    # SURPRISAL: sum over all tokens in the target word
    target_ids = tokenizer.encode(" " + target_word.strip(), add_special_tokens=False)
    if not target_ids:
        surprisal = float('inf')
    else:
        prev_token_id = target_ids[0]
        surprisal = -np.log2(next_token_probs[prev_token_id] + 1e-20)
        target_input_ids = input_ids
        for target_token_id in target_ids[1:]:
            next_id = torch.tensor([[prev_token_id]], device=device)
            target_input_ids = torch.cat([target_input_ids, next_id], dim=1)
            with torch.no_grad():
                outputs = model(target_input_ids)
                token_logits = outputs.logits[0, -1, :]
                token_probs = torch.softmax(token_logits, dim=-1)
                p = token_probs[target_token_id].item()
            surprisal += -np.log2(p + 1e-20)
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
    else:
        word_entropy = None
        # Top-N predictions (by token prob)
        topk_idx = np.argsort(-next_token_probs)[:top_n]
        top_preds = []
        for idx in topk_idx:
            pred_word = tokenizer.decode([idx])
            top_preds.append((pred_word, next_token_probs[idx]))

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
            with torch.no_grad():
                outputs = model(beam_input_ids)
                logits = outputs.logits[0, -1, :]
                token_logprobs = torch.log_softmax(logits, dim=-1)
                top_logprobs, top_indices = torch.topk(token_logprobs, beam_width)

            for token_logprob, token_id in zip(top_logprobs, top_indices):
                token_id_int = int(token_id.item())
                token_text = tokenizer.decode([token_id_int])

                if not started:
                    if not is_word_start(token_text):
                        continue
                    next_id = torch.tensor([[token_id_int]], device=device)
                    next_input_ids = torch.cat([beam_input_ids, next_id], dim=1)
                    expanded.append((
                        next_input_ids,
                        token_text.lstrip(),
                        logprob + token_logprob.item(),
                        True,
                    ))
                elif is_word_continuation(token_text):
                    next_id = torch.tensor([[token_id_int]], device=device)
                    next_input_ids = torch.cat([beam_input_ids, next_id], dim=1)
                    expanded.append((
                        next_input_ids,
                        word + token_text,
                        logprob + token_logprob.item(),
                        True,
                    ))
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
    word_entropy = -np.sum(probs * np.log2(probs + 1e-20))

    ranked = sorted(completed.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [(word, float(np.exp(logprob))) for word, logprob in ranked], word_entropy

def process_sentences(
    settings: dict[str, Any],
    file_path: str,
    context: str,
    keep_punctuation_and_case: bool,
    top_n: int,
    top_words: bool,
    beam_width: int,
    max_word_tokens: int,
) -> None:
    print(f"Model: {settings['CAUSAL_LM_MODEL_NAME']}")

    # Sentences file
    if len(file_path) > 0:
        print(f"File: {file_path}")
        lines = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
            print(f"Number of sentences: {len(lines)}")
        else:
            print(f"File not found: {file_path}")
            return
    else:
        lines = [context]
    print()

    # Process sentences
    header = f"{'WordID'}\t{'SentenceNr'}\t{'WordNr'}\t{'Target'}\t{'Entropy'}\t{'Surprisal'}"
    if top_words:
        header += f"\t{'WordEntropyApprox'}"
    header += "\tPredictions"
    print(header)

    word_id = 0
    cnt = 1
    line_cnt = 1
    for line in lines:
        words = line.split()
        context = ""
        for target in words:
            target = target.strip()

            if not target:
                print(f"Empty target word at line {line_cnt}, wordNr {cnt}.")
                exit(1)
            if len(target) == 1 and not target.isalnum():
               # skip single punctuation characters, e.g. -
               continue
            if context == "":
                context = target
                word_id += 1
                cnt += 1
                continue

            entropy, surprisal, word_entropy, top_preds = calculate_metrics(
                settings,
                context,
                target,
                top_n,
                top_words,
                beam_width,
                max_word_tokens,
            )

            top = ""
            for w, p in top_preds:
                if len(top) > 0:
                    top += f"\t{repr(w)}\t{p:.3f}"
                else:
                    top += f"{repr(w)}\t{p:.3f}"

            # Remove punctuation from target if specified
            if not keep_punctuation_and_case:
                ptarget = target.lower()
                ptarget = ptarget.strip(string.punctuation)
            else:
                ptarget = repr(target)

            output = f"{word_id}\t{line_cnt}\t{cnt}\t{ptarget}\t{entropy}\t{surprisal}"
            if top_words:
                output += f"\t{word_entropy}"
            output += f"\t{top}"
            print(output)

            # Update context for next iteration
            context = f"{context} {target}"
            cnt += 1
            word_id += 1

        line_cnt += 1
        cnt = 1

def main() -> None:
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Next token prediction. Calculate entropy and surprisal for each word in a given sentence.",
    )

    # Add arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file",
        dest="file_path",
        help="Path to the sentences file",
    )
    input_group.add_argument(
        "-s", "--sentence",
        dest="sentence",
        help="Sentence to process",
    )
    parser.add_argument(
        "-r", "--rawtarget",
        dest="keep_punctuation_and_case",
        action="store_true",
        help="Do not remove punctuation from target and do not change to lower case - for output only (default: False)"
    )
    parser.add_argument(
        "-n", "--ntop",
        dest="top_n",
        type=positive_int,
        default=5,
        metavar="TOP_N",
        help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "--top-words",
        dest="top_words",
        action="store_true",
        help="Show beam-searched next word predictions instead of raw next token predictions"
    )
    parser.add_argument(
        "--beam-width",
        dest="beam_width",
        type=positive_int,
        default=25,
        metavar="WIDTH",
        help="Beam width for --top-words (default: 25)"
    )
    parser.add_argument(
        "--max-word-tokens",
        dest="max_word_tokens",
        type=positive_int,
        default=5,
        metavar="TOKENS",
        help="Maximum number of model tokens per predicted word for --top-words (default: 5)"
    )
    parser.add_argument(
        "-m", "--model",
        dest="model_name",
        default="openai-community/gpt2",
        help="Name of the causal language model to use (default: gpt2)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup settings dictionary
    settings = {
        'CAUSAL_LM_MODEL_NAME': args.model_name,
        'CAUSAL_LM_MODEL': None,
        'CAUSAL_LM_TOKENIZER': None,
    }

    # Process sentences
    if args.file_path is None:
        the_file = ""
    else:
        the_file = args.file_path

    process_sentences(
        settings,
        the_file,
        args.sentence,
        args.keep_punctuation_and_case,
        args.top_n,
        args.top_words,
        args.beam_width,
        args.max_word_tokens,
    )

if __name__ == "__main__":
    main()
