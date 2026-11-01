#!/usr/bin/env python3
"""Write Neyshekar's validation speakers as a benchmark corpus, for a check that may choose.

The sealed test sets (``scripts/prepare_test_corpus.py``) are scored once per
final candidate and never used to choose. A check that chooses between exports
on many voices needs speakers of its own. Neyshekar v6's validation split has
26: in no training split of the adaptation (their clips gave only its
validation loss), not on OmniVoice's published list of training data, and never
used to choose anything here.

The roles are the test sets' rule, decided on durations alone: the reference is
the speaker's longest clip of 6.5 to 11.4 s, and the four longest of the rest,
of at least 3.5 s, are held out. A speaker without them, or with a clip missing
from the parquet or unreadable, is left out and listed; there is no reserve.

Output as the test sets': ``<out>/manifest.json`` in the benchmark's format
(``nev01_000.wav`` is the reference, ``_001`` to ``_004`` the held-out clips),
each clip's bytes as the parquet stores them, and ``<out>/roster.json`` with who
is in, who was left out and why, and every file's sha256. Only validation
records are looked up.

    .venv/bin/python scripts/prepare_valid_corpus.py --splits eval/manifests/adaptation_splits_v1.jsonl.gz \\
        --parquet-dir /tmp/neyshekar-validation --out /tmp/valid-ney
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_test_corpus import ROLE_RULE, load_records, read_audio, roster, whole_reference_roles  # noqa: E402

SOURCE = "neyshekar"
SPLIT = "valid"
PREFIX = "nev"
CORPUS = "neyshekar_v6_validation"


def validation(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[dict]]]:
    clips: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["source"] == SOURCE and record["split"] == SPLIT:
            clips[record["speaker"]].append(record)
    return sorted(clips), dict(clips)


def write(result: dict[str, Any], audio: dict[str, bytes], out: Path, splits_sha256: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    entries, files, speakers = [], {}, []
    for index, row in enumerate(result["chosen"], start=1):
        label = f"{PREFIX}{index:02d}"
        reference, held = row["roles"]
        for position, record in enumerate((reference, *held)):
            name = f"{label}_{position:03d}.wav"
            data = audio[record["clip"]]
            (out / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
            entries.append({
                "speaker": label, "file": name, "role": "reference" if position == 0 else "held_out",
                "transcript": record["text"].strip(), "duration": record["duration"],
                "gender": record["gender"], "corpus": CORPUS,
                "session": record["clip"], "speaker_hash": row["speaker"], "source_ref": record.get("source_ref"),
            })
        speakers.append({"label": label, "speaker": row["speaker"], "gender": reference["gender"],
                         "reference": reference["clip"], "reference_seconds": reference["duration"],
                         "held_out": [c["clip"] for c in held]})
    manifest = {
        "version": 1,
        "purpose": "Neyshekar's validation speakers: a set that may choose, kept apart from the sealed test sets",
        "speakers": len(speakers),
        "clips": len(entries),
        "sources": [CORPUS],
        "notes": ROLE_RULE,
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "source": SOURCE,
        "split": SPLIT,
        "splits_sha256": splits_sha256,
        "role_rule": ROLE_RULE,
        "speakers": speakers,
        "left_out": result["left_out"],
        "files": files,
        "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
    }
    (out / "roster.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = load_records(args.splits)
    speakers, clips = validation(records)
    wanted = {}
    for rows in clips.values():
        roles = whole_reference_roles(rows)
        if roles is not None:
            wanted.update({r["clip"]: r for r in (roles[0], *roles[1])})
    audio = read_audio(SOURCE, args.parquet_dir, wanted)
    result = roster(SOURCE, speakers, {}, clips, audio)
    record = write(result, audio, args.out, hashlib.sha256(args.splits.read_bytes()).hexdigest())
    print(f"{len(record['speakers'])} of {len(speakers)} validation speakers written to {args.out}; "
          f"left out: {record['left_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
