#!/usr/bin/env python3
"""Write the English sentences of the LongCat probe (eval/manifests/probe_texts_en_v1.json).

LongCat-AudioDiT reads Chinese and English only, so the probe compares it with
OmniVoice on text both of them read: English, spoken in the voices of Persian
speakers, scored against those speakers' other recordings. Six sentences
composed for this project - short to long, a question, a list, a narrative -
with no digits, abbreviations or quotation marks, so neither engine's number
reading or text normaliser enters the comparison. The longest, at LongCat's
own length estimate and its largest pace ratio (1.5), still fits the 1B's
30 s window beside an 11.4 s reference.

    .venv/bin/python scripts/build_probe_texts.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "eval" / "manifests" / "probe_texts_en_v1.json"

TEXTS = (
    ("e01", "short", "Please close the window."),
    ("e02", "question", "Could you call me back after lunch, when you have a minute?"),
    ("e03", "narrative", "The old bridge was closed for repairs, so we took the ferry across the river."),
    ("e04", "list", "Remember to switch off the lights, lock the back door, and feed the cat."),
    ("e05", "long", "According to the morning report, heavy rain is expected in the northern valleys "
                    "tonight, so drivers should avoid the mountain roads until tomorrow."),
    ("e06", "descriptive", "My grandmother kept a small garden behind the house, where she grew tomatoes, "
                           "mint and bright yellow flowers."),
)


def build() -> dict:
    texts = [{"key": key, "category": category, "text": text, "words": len(text.split())}
             for key, category, text in TEXTS]
    return {
        "version": 1,
        "purpose": "the English sentences every LongCat probe speaker renders, with LongCat and OmniVoice",
        "written_for_this_project": True,
        "language": "en",
        "texts": texts,
        "sha256_of_texts": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def main() -> int:
    manifest = build()
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(manifest['texts'])} texts, sha256 {manifest['sha256_of_texts'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
