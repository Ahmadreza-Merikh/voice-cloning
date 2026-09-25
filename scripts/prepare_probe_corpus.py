#!/usr/bin/env python3
"""Write the LongCat probe's speakers as a benchmark corpus: Neyshekar's valid split.

The probe (``eval/runs/persian-voice-longcat-probe.json``) asks whether
LongCat-AudioDiT carries a voice better than OmniVoice before anyone trains it
on Persian. Its speakers are none of the ones this project already uses:

* not the development set - eight of its nine speakers are Common Voice, which
  OmniVoice very probably trained on and LongCat did not
* not the sealed test sets - they are rendered once per final candidate and
  never used to choose anything
* but Neyshekar v6's official validation speakers: speaker-disjoint from its
  test split and from training, and not in OmniVoice's published training
  list. The adaptation reads them only as validation loss; nothing has been
  chosen on them

Roles are the sealed test's (``prepare_test_corpus.py``): the reference is the
speaker's longest clip of 6.5-11.4 s, which the reference pipeline keeps whole,
and the four longest of the rest, of at least 3.5 s, are held out. A speaker
without them, or with a clip that fails the mechanical checks, is left out and
listed; there is no reserve. Only records of the valid split are looked up, so
no training or test clip can be written.

    .venv/bin/python scripts/prepare_probe_corpus.py --splits eval/manifests/adaptation_splits_v1.jsonl.gz \\
        --parquet-dir /tmp/neyshekar-valid --out /tmp/probe-ney
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_adaptation_splits import HELD_OUT, MIN_REFERENCE_SECONDS  # noqa: E402
from prepare_test_corpus import (  # noqa: E402
    ROLE_RULE,
    WHOLE_REFERENCE_SECONDS,
    load_records,
    problems_of,
    read_audio,
    whole_reference_roles,
)

SOURCE = "neyshekar"
SPLIT = "valid"
PREFIX = "nvl"
CORPUS = "neyshekar_v6_valid"


def valid_speakers(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["source"] == SOURCE and record["split"] == SPLIT:
            clips[record["speaker"]].append(record)
    return dict(clips)


def roster(clips: dict[str, list[dict[str, Any]]], audio: dict[str, bytes]) -> dict[str, Any]:
    chosen: list[dict[str, Any]] = []
    left_out: list[dict[str, str]] = []
    for speaker in sorted(clips):
        roles = whole_reference_roles(clips[speaker])
        if roles is None:
            left_out.append({"speaker": speaker, "reason": f"no clip of {MIN_REFERENCE_SECONDS}-"
                             f"{WHOLE_REFERENCE_SECONDS} s with {HELD_OUT} more to hold out"})
            continue
        problems = problems_of(roles, audio, SOURCE)
        if problems:
            left_out.append({"speaker": speaker, "reason": "; ".join(problems)})
            continue
        chosen.append({"speaker": speaker, "roles": roles})
    for index, row in enumerate(chosen, start=1):
        row["label"] = f"{PREFIX}{index:02d}"
    return {"chosen": chosen, "left_out": left_out}


def write(result: dict[str, Any], audio: dict[str, bytes], out: Path, splits_sha256: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    entries, files, speakers = [], {}, []
    for row in result["chosen"]:
        reference, held = row["roles"]
        for position, record in enumerate((reference, *held)):
            name = f"{row['label']}_{position:03d}.wav"
            data = audio[record["clip"]]
            (out / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
            entries.append({
                "speaker": row["label"], "file": name, "role": "reference" if position == 0 else "held_out",
                "transcript": record["text"].strip(), "duration": record["duration"],
                "gender": record["gender"], "corpus": CORPUS, "session": record["clip"],
                "speaker_hash": row["speaker"], "source_ref": record.get("source_ref"),
            })
        speakers.append({"label": row["label"], "speaker": row["speaker"], "reference": reference["clip"],
                         "reference_seconds": reference["duration"], "held_out": [c["clip"] for c in held]})
    manifest = {
        "version": 1,
        "purpose": "LongCat probe: Neyshekar's valid speakers, never a development, test or training clip",
        "speakers": len(result["chosen"]),
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
    if args.splits.suffix == ".gz":
        with gzip.open(args.splits) as handle:
            raw = handle.read()
    else:
        raw = args.splits.read_bytes()
    clips = valid_speakers(records)
    wanted = {}
    for speaker_clips in clips.values():
        roles = whole_reference_roles(speaker_clips)
        if roles:
            wanted.update({r["clip"]: r for r in (roles[0], *roles[1])})
    audio = read_audio(SOURCE, args.parquet_dir, wanted)
    record = write(roster(clips, audio), audio, args.out, hashlib.sha256(raw).hexdigest())
    print(json.dumps({"speakers": len(record["speakers"]), "left_out": record["left_out"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
