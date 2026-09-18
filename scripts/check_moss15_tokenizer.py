#!/usr/bin/env python3
"""Does the "incorrect regex pattern" warning on MOSS v1.5's tokenizer matter for Persian?

Loading MOSS-TTS-Local-Transformer-v1.5 through its processor under the
official transformers 5.0.0 logs that the tokenizer has "an incorrect regex
pattern" and suggests ``fix_mistral_regex=True``. Read in transformers 5.0.0
(``TokenizersBackend._patch_mistral_regex``), the warning fires for any local
tokenizer whose ``config.json`` has no ``transformers_version`` - MOSS v1.5's
has none, its model type is ``moss_tts_local`` and its tokenizer is Qwen2's,
not Mistral's. The suggested flag would swap Qwen2's pre-tokenizer regex for a
Mistral one; under 5.0.0 passing it raises a TypeError anyway.

This compares the two regexes on the benchmark's sentences and a few harder
Persian ones, using only the ``tokenizers`` library and the tokenizer files at
the pinned revision - no model, no remote code:

    .venv/bin/python scripts/check_moss15_tokenizer.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import tokenizers
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent


def constant(script: str, name: str):
    """A module-level constant, read without importing - the worker claims stdout on import."""
    tree = ast.parse((HERE / script).read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


SENTENCES = constant("benchmark.py", "SENTENCES")
MODEL, MODEL_REVISION = constant("moss15_worker.py", "MODEL"), constant("moss15_worker.py", "MODEL_REVISION")

#: What transformers 5.0.0 inserts when fix_mistral_regex=True.
MISTRAL_FIX = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
               r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}|"
               r" ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
HARDER = [
    "می‌خواهم ساعت ۱۰:۳۰ با دکتر رضایی در خیابان ولی‌عصر قرار بگذارم.",
    "قیمت این کتاب ۲۵۰٬۰۰۰ تومان است، نه 250 دلار!",
    "آیا «هوش مصنوعی» می‌تواند صدای انسان را تقلید کند؟",
    "ژیلا و گلاره در پاییز به کوه‌های البرز رفتند.",
]
MIXED_CASE = "The iPhone's GPU runs McDonald's app at 60 FPS."


def main() -> int:
    path = hf_hub_download(MODEL, "tokenizer.json", revision=MODEL_REVISION)
    trained = tokenizers.Tokenizer.from_file(path)
    fixed = tokenizers.Tokenizer.from_file(path)
    fixed.pre_tokenizer[0] = tokenizers.pre_tokenizers.Split(pattern=tokenizers.Regex(MISTRAL_FIX),
                                                             behavior="isolated")
    persian = [text for _, text in SENTENCES] + HARDER
    differ = 0
    for text in persian + [MIXED_CASE]:
        a = trained.encode(text, add_special_tokens=False).ids
        b = fixed.encode(text, add_special_tokens=False).ids
        differ += a != b
        print(f"{'same     ' if a == b else 'DIFFERENT'} {len(a):3d} vs {len(b):3d} tokens  {text[:50]}")
    same_persian = all(trained.encode(t, add_special_tokens=False).ids == fixed.encode(t, add_special_tokens=False).ids
                       for t in persian)
    print(f"Persian: {'identical' if same_persian else 'CHANGED'} in all {len(persian)}; "
          f"{differ} of {len(persian) + 1} sentences differ overall")
    return 0 if same_persian else 1


if __name__ == "__main__":
    raise SystemExit(main())
