# Next-token prediction

Predict next token and calculate entropy and surprisal values of each word in a sentence based on the previous words in the sentence. Approach is based on Cevoli et al. (2022).

Default model is [gpt2](https://huggingface.co/openai-community/gpt2).

## Installation

<details>

<summary>Click to expand/collapse</summary>

### macOS

Install [brew](https://brew.sh).

Next install `Python` and `uv` using the [Terminal](https://support.apple.com/en-gb/guide/terminal/welcome/mac)

```sh
brew install python@3.12
brew install uv
```

### Windows

Install [scoop](https://scoop.sh).

Next install `Python` and `uv` using the [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.5).

```powershell
scoop bucket add versions
scoop install versions/python312
scoop bucket add main
scoop install main/uv
```

### Clone repository

```sh
git clone https://github.com/waltervanheuven/next-token.git
```

</details>

## Examples

```sh
uv run next_token.py -s "The apple fell from the tree"
```

Process file with sentences.

```sh
uv run next_token.py -f sentences.txt
```

Use a different transformer (base) model.

```sh
uv run next_token.py -f sentences.txt -m "ibm-granite/granite-3.3-2b-base"
```

Show command line options.

```sh
uv run next_token.py -h
```

## References

Cevoli, B., Watkins, C., & Rastle, K. (2022). Prediction as a basis for skilled reading: insights from modern language models. *Royal Society Open Science, 9(6)*, 211837. [https://doi.org/10.1098/rsos.211837](https://doi.org/10.1098/rsos.211837)
