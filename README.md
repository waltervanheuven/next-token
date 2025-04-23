# Next-token prediction

Predict next token and calculate entrophy and surprisal values of each word in a sentence.

Default model is gpt2.

## Installation

<details>

<summary>Click to expand/collapse</summary>

### macOS

Install [brew](https://brew.sh).

Next install `Python` and `uv` in the [Terminal](https://support.apple.com/en-gb/guide/terminal/welcome/mac)

```sh
brew install python@3.12
brew install uv
```

### Windows

Install [scoop](https://scoop.sh).

Next install `Python` and `uv` in the [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.5).

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

Use other transformer (base) model.

```sh
uv run next_token.py -f sentences.txt -m "ibm-granite/granite-3.3-2b-base"
```
