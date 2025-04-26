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

def load_causal_lm_model(settings: dict[str, any]) -> None:
    model_name = settings['CAUSAL_LM_MODEL_NAME']
    try:
        settings['CAUSAL_LM_MODEL'] = AutoModelForCausalLM.from_pretrained(model_name)
        settings['CAUSAL_LM_TOKENIZER'] = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading causal language model {model_name}: {e}")
        sys.exit(1)

def calculate_metrics(
    settings: dict[str, any],
    context: str,
    target_word: str,
    top_n: int = 5,
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

    # SURPRISAL: first token of target word
    target_ids = tokenizer.encode(" " + target_word.strip(), add_special_tokens=False)
    if not target_ids:
        surprisal = float('inf')
    else:
        target_token_id = target_ids[0]
        p = next_token_probs[target_token_id]
        surprisal = -np.log2(p + 1e-20)

    # Top-N predictions (by token prob)
    topk_idx = np.argsort(-next_token_probs)[:top_n]
    top_preds = []
    for idx in topk_idx:
        pred_word = tokenizer.decode([idx])
        top_preds.append((pred_word, next_token_probs[idx]))

    return entropy, surprisal, top_preds

def process_sentences(settings: dict[str, any], file_path: str, context: str, keep_punctuation_and_case: bool, top_n: int) -> None:
    # Sentences file
    if len(file_path) > 0:
        lines = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
        else:
            print(f"File not found: {file_path}")
            return
    else:
        lines = [context]

    # Process sentences
    print(f"Model: {settings['CAUSAL_LM_MODEL_NAME']}\n")
    print(f"Number of sentences: {len(lines)}")
    print(f"{'QID'}\t{'Sentence '}\t{'WordNr '}\t{'Target    '}\t{'Entropy   '}\t{'Surprisal '}\tPredictions")

    qid = 0
    cnt = 1
    line_cnt = 1
    for line in lines:
        words = line.split()
        context = ""
        for n, target in enumerate(words):
            target = target.strip()

            if not target:
                continue
            if n == 0:
                context = target
                qid += 1
                cnt += 1
                continue

            entropy, surprisal, top_preds = calculate_metrics(settings, context, target, top_n)

            top = ""
            for w, p in top_preds:
                if len(top) > 0:
                    top += f"\t{repr(w)}\t{p:.3f}"
                else:
                    top += f"{repr(w)}\t{p:.3f}"

            # Remove punctuation from target if specified
            if not keep_punctuation_and_case:
                ptarget = target.lower()
                ptarget = ptarget.translate(str.maketrans('', '', string.punctuation))
            else:
                ptarget = repr(target)

            print(f"QID{qid:<4}\t{line_cnt:<9}\t{cnt:<7}\t{ptarget:<10}\t{entropy:.7f}\t{surprisal:.7f}\t{top}")

            # Update context for next iteration
            context = f"{context} {target}"
            cnt += 1
            qid += 1

        line_cnt += 1
        cnt = 1

def main() -> None:
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Next token prediction. Calculate entropy and surprisal for each word in a given sentence.",
    )

    # Add arguments
    parser.add_argument(
        "-f", "--file", 
        dest="file_path",
        help="Path to the sentences file",
        required=False
    )
    parser.add_argument(
        "-s", "--sentence", 
        dest="sentence",
        help="Sentence to process",
        required=False
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
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
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

    process_sentences(settings, the_file, args.sentence, args.keep_punctuation_and_case, args.top_n)

if __name__ == "__main__":
    main()
