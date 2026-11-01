#!/usr/bin/env python3
"""Count the rows benchmark runs render alike: whether they loaded the same weights.

The ezafe voices check's gate on the weights could not pass: it looked for the
model path in each row's metadata, and ``scripts/benchmark.py`` writes none into
its rows. This is the evidence put in its place after the run.

OmniVoice at seed 42 renders the same text from the same prompt bit-identically
on the GPU (FINDINGS section 15), in another session too
(eval/runs/persian-voice-omnivoice-dev.json, "reproduced"), and re-scoring
identical audio gives identical scores. So two runs that loaded the same weights
agree on every row's held-out similarity; runs that disagree on a row rendered
it differently, from other weights.

    .venv/bin/python scripts/same_renders.py eval/runs/evidence/ezafe-voices-v1 \\
        ezafe-voices-A1 ezafe-voices-E1 ezafe-voices-E5 --out eval/runs/evidence/ezafe-voices-v1/same-renders.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

FIELDS = ("held_out_similarity", "reference_similarity", "transcription")


def rows(root: Path, label: str) -> dict[tuple, dict]:
    data = json.loads((root / f"{label}.json").read_text(encoding="utf-8"))
    return {(r["speaker"], r["sentence"], r["seed"]): r for r in data["rows"]}


def compare(left: dict[tuple, dict], right: dict[tuple, dict]) -> dict:
    keys = sorted(set(left) & set(right))
    return {"rows_in_both": len(keys),
            **{f"same_{field}": sum(left[k].get(field) == right[k].get(field) for k in keys) for field in FIELDS}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    runs = {label: rows(args.root, label) for label in args.labels}
    result = {f"{a} vs {b}": compare(runs[a], runs[b]) for a, b in itertools.combinations(args.labels, 2)}
    text = json.dumps(result, ensure_ascii=False, indent=1) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
