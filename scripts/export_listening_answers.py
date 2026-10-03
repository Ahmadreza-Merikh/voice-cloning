#!/usr/bin/env python3
"""Turn the listening page's answers into the files score_listening_test.py reads.

The page (eval/runs/persian-voice-listening-v1.json, ``delivery``) interleaves
the trials of two tests and keeps one record per listener: in its database as
``responses/<listener id>``, or, for a listener who cannot save there, as an
answer code they copy and send. Either way a record is
``{"listener": ..., "answers": {trial id: {"same", "natural", "misread"}}}``,
keyed by trial id only - the page never names a test or a system.

This splits each record by the keys (which trial belongs to which test) into
one responses file per listener per test, in the format the scorer checks:
``{"test", "fingerprint", "listener", "answers"}``. It adds nothing: an
unanswered trial stays absent, and an answer to a trial no key knows is
reported and dropped.

    .venv/bin/python scripts/export_listening_answers.py --keys listening/*.key.json \\
        --records responses-db/*.json codes/*.txt --out listening/responses
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_record(path: Path) -> dict:
    """A database document as ArtifactData saves it, or an answer code as pasted."""
    text = path.read_text(encoding="utf-8").strip()
    data = json.loads(text)
    if "data" in data and isinstance(data["data"], dict) and "answers" in data["data"]:
        data = data["data"]
    if "answers" not in data:
        raise SystemExit(f"{path}: no answers in it")
    return data


def split(record: dict, keys: list[dict]) -> tuple[dict[str, dict], list[str]]:
    owner = {tid: key for key in keys for tid in key["trials"]}
    by_test: dict[str, dict] = {}
    unknown = []
    for tid, answer in record["answers"].items():
        key = owner.get(tid)
        if key is None:
            unknown.append(tid)
            continue
        entry = by_test.setdefault(key["name"], {"test": key["name"], "fingerprint": key["fingerprint"],
                                                 "listener": str(record.get("listener") or ""), "answers": {}})
        entry["answers"][tid] = answer
    return by_test, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keys", nargs="+", type=Path, required=True)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    keys = [json.loads(p.read_text(encoding="utf-8")) for p in args.keys]
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.records:
        record = read_record(path)
        by_test, unknown = split(record, keys)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", path.stem).strip("-") or "record"
        for name, body in by_test.items():
            target = args.out / name / f"{stem}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{path.name}: listener {record.get('listener')!r}, "
              + ", ".join(f"{name} {len(body['answers'])}" for name, body in sorted(by_test.items()))
              + (f"; {len(unknown)} answers to trials no key knows, dropped" if unknown else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
