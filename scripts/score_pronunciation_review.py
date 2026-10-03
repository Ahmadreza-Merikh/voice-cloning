#!/usr/bin/env python3
"""Tabulate a pronunciation review's answers through its key. It never produces answers.

Written before anyone listened (eval/runs/persian-voice-pronunciation-review-v1.json).
Each answer marks, per render, either "all correct" or the words said wrong,
and per sentence which render read best. Through the key that becomes, per
condition (A plain script, B diacritised twin, C IPA fine-tune):

* renders read without a single marked word, out of those answered
* words marked wrong, out of the words in those renders
* how often a render was picked as reading best
* the same split by the sentence's category, and the marked words themselves

B and C read twins with known wrong diacritics on a few words (the key's
``wrong_in_twins``); a mark on such a sentence is listed with that note.
Fourteen sentences and few listeners make this a diagnostic - which errors,
under which input - not a claim about any system.

    .venv/bin/python scripts/score_pronunciation_review.py eval/runs/evidence/pronunciation-review-v1/key.json answers/*.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def records(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data.get("data", data) if isinstance(data.get("data"), dict) else data
        if "answers" in data:
            out.append(data)
    return out


def tabulate(key: dict, items: dict[str, list[str]], answers: list[dict]) -> dict:
    per = defaultdict(lambda: {"renders": 0, "clean": 0, "words": 0, "wrong": 0, "best": 0})
    by_category = defaultdict(lambda: defaultdict(lambda: {"renders": 0, "clean": 0}))
    marked = []
    for record in answers:
        listener = str(record.get("listener") or "")
        for sid, answer in record["answers"].items():
            sentence = key["sentences"].get(sid)
            if sentence is None:
                continue
            words = items.get(sid, [])
            for slot, condition in sentence["slots"].items():
                given = (answer.get("slots") or {}).get(slot)
                if not given or (given.get("ok") is not True and not given.get("words")):
                    continue
                wrong = sorted({int(i) for i in given.get("words") or [] if 0 <= int(i) < len(words)})
                row = per[condition]
                row["renders"] += 1
                row["words"] += len(words)
                row["wrong"] += len(wrong)
                row["clean"] += 0 if wrong else 1
                cat = by_category[sentence["category"]][condition]
                cat["renders"] += 1
                cat["clean"] += 0 if wrong else 1
                if wrong or given.get("note"):
                    marked.append({"listener": listener, "sentence": sid, "condition": condition,
                                   "words": [words[i] for i in wrong], "note": given.get("note") or "",
                                   "twin_note": key.get("wrong_in_twins", {}).get(sid) if condition in "BC" else None})
            best = answer.get("best")
            if best in sentence["slots"]:
                per[sentence["slots"][best]]["best"] += 1
    summary = {c: {**row, "clean_share": round(row["clean"] / row["renders"], 3) if row["renders"] else None,
                   "wrong_word_share": round(row["wrong"] / row["words"], 3) if row["words"] else None}
               for c, row in sorted(per.items())}
    return {"listeners": sorted({str(r.get("listener") or "") for r in answers}), "per_condition": summary,
            "by_category": {k: dict(v) for k, v in sorted(by_category.items())}, "marked": marked,
            "label": "diagnostic - fourteen sentences, few listeners; not a claim"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2:
        raise SystemExit(__doc__)
    key = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    items_path = Path(args[0]).with_name("items.json")
    items = {i["id"]: i["words"] for i in json.loads(items_path.read_text(encoding="utf-8"))}
    print(json.dumps(tabulate(key, items, records([Path(p) for p in args[1:]])), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
