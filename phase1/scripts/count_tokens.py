"""Count tokens in the profile renderings under candidate worker tokenizers.

The worker model is not chosen yet, and tokenizers differ, so this counts under
more than one:
  o200k_base   tiktoken encoding used by the gpt-oss family
  deepseek-v3  DeepSeek-V3 tokenizer from Hugging Face, a proxy for the DeepSeek
               models in the design doc's cost table
  glm-4.5      GLM-4.5 tokenizer from Hugging Face, a proxy for GLM
A tokenizer that fails to load is reported, not silently skipped.

Usage:
  uv run --with tiktoken --with tokenizers --with huggingface_hub \
      python scripts/count_tokens.py runs/profile/<ts>
"""

from __future__ import annotations

import json
import pathlib
import sys


def load_tokenizers() -> tuple[dict, dict]:
    toks, failed = {}, {}
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        toks["o200k_base"] = lambda s, e=enc: len(e.encode(s, disallowed_special=()))
    except Exception as e:  # noqa: BLE001
        failed["o200k_base"] = repr(e)[:200]
    for name, repo in (("deepseek-v3", "deepseek-ai/DeepSeek-V3"), ("glm-4.5", "zai-org/GLM-4.5")):
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
            toks[name] = lambda s, t=t: len(t.encode(s).ids)
        except Exception as e:  # noqa: BLE001
            failed[name] = repr(e)[:200]
    return toks, failed


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    toks, failed = load_tokenizers()
    for k, v in failed.items():
        print(f"!! tokenizer {k} unavailable: {v}")
    files = sorted(p for p in d.glob("*.txt"))
    out = {}
    names = list(toks)
    print(f"{'file':42s} {'bytes':>9s} " + " ".join(f"{n:>12s}" for n in names))
    for p in files:
        s = p.read_text()
        counts = {n: f(s) for n, f in toks.items()}
        out[p.name] = {"bytes": len(s.encode()), **counts}
        print(f"{p.name:42s} {len(s.encode()):>9,} " + " ".join(f"{counts[n]:>12,}" for n in names))
    (d / "tokens.json").write_text(json.dumps({"tokenizers": names, "failed": failed, "files": out}, indent=2))
    print(f"\nwrote {d / 'tokens.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
