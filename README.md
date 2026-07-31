# Next-token prediction

Predict next token and calculate entropy and surprisal values of each word in a sentence based on the previous words in the sentence. Approach is based on Cevoli et al. (2022). The script can also generate sampled or greedy continuations.

By default, `Entropy` is token-level entropy: it is calculated over the model's full next-token probability distribution. For words that are split into multiple model tokens, word surprisal is calculated as the sum of the surprisals of those tokens.

By default, the predictions column shows the most likely next model tokens. Use `--top-words` to show beam-searched next word predictions instead. This mode expands likely token continuations until a word boundary is reached, then ranks completed words by the summed log probabilities of the tokens in each word. When `--top-words` is enabled, the output also includes `WordEntropyApprox`, an approximate entropy over the completed words found by the beam search. Because transformer tokenizers predict tokens rather than words, the word boundary is a heuristic based on whitespace and punctuation, and `WordEntropyApprox` depends on `--beam-width` and `--max-word-tokens`.

Default model is [gpt2](https://huggingface.co/openai-community/gpt2).

## Installation

<details>

<summary>Click to expand/collapse</summary>

### macOS

Install first [brew](https://brew.sh), then use brew to install `Python`, and `uv`:

```sh
brew install python@3.12
brew install uv
```

### Windows

Use `winget` which is available on Windows 11 and modern Windows 10. In
PowerShell, install Python 3.12 and `uv`:

```powershell
winget install --id Python.Python.3.12 -e
winget install --id astral-sh.uv -e
```

Restart PowerShell if the commands are not immediately available on `PATH`.

### Clone repository

```sh
git clone https://github.com/waltervanheuven/next-token.git
cd next-token
```

</details>

## Examples

Calculate metrics for a sentence:

```sh
uv run next_token.py -s "The apple fell from the tree"
```

Process a file with one sentence per line:

```sh
uv run next_token.py -f sentences.txt
```

Show next-word predictions instead of raw next token predictions:

```sh
uv run next_token.py -s "The apple fell from the tree" --top-words
```

Adjust the next word beam search.

```sh
uv run next_token.py -s "The apple fell from the tree" --top-words --beam-width 50 --max-word-tokens 6
```

Use a different transformer (base) model.

```sh
uv run next_token.py -f sentences.txt -m "ibm-granite/granite-3.3-2b-base"
```

Generate ten sampled continuations with reproducible settings:

```sh
uv run next_token.py \
  -s "the apple fell from the" \
  --next-words 3 --sample --num-samples 10 \
  --temperature 1.0 --top-p 0.95 --seed 42
```

`--until-stop` and `--next-words` are mutually exclusive. Use
`--max-new-tokens` as a safety limit. Sampling is controlled with
`--sample`, `--temperature`, `--top-p`, and `--seed`. 

Metric values are printed to four decimal places; change `OUTPUT_DECIMAL_PLACES` in the script to alter this.

Show command-line options:

```sh
uv run next_token.py -h
```

Script was improved with help of Codex.

## References

Cevoli, B., Watkins, C., & Rastle, K. (2022). Prediction as a basis for skilled reading: insights from modern language models. *Royal Society Open Science, 9(6)*, 211837. [https://doi.org/10.1098/rsos.211837](https://doi.org/10.1098/rsos.211837)

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI technical report. [https://cdn.openai.com/better-language-models/language-models.pdf](https://cdn.openai.com/better-language-models/language-models.pdf)
