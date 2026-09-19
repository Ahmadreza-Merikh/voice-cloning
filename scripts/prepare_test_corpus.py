#!/usr/bin/env python3
"""Write one sealed test set as a benchmark corpus - in the session that scores it, nowhere else.

The speakers are the split's (``scripts/build_adaptation_splits.py``):
Neyshekar's official test split, the primary set, and the thirty Common Voice
test speakers, the secondary one. Nothing here draws them again.

**Roles, amended on 2026-09-19 before any test clip was read or rendered.**
The split gave each test speaker's longest clip the reference role. The
reference pipeline keeps a clip whole only up to 11.5 s (the 10 s target times
1.15, ``audio/preprocess.py``); a longer one is cut to a window while the
transcript given with it is not, so the engine would be told words its prompt
may no longer contain. Thirteen of the 28 Neyshekar references were longer.
So the reference is the speaker's longest clip of 6.5 to 11.4 s - 0.1 s under
the limit, twice the 50 ms by which a file may differ from the split's
duration - and the four longest of the rest, of at least 3.5 s, are held out.
Decided on durations alone. It changes no Common Voice speaker's roles and
leaves the same 28 Neyshekar speakers usable.

**Mechanical failures** - a clip missing from the parquet, unreadable, not
mono, not the source's sample rate, or more than 50 ms from the split's
duration - are found before anything is written. A Common Voice test speaker
with one is replaced by the next reserve speaker of the same self-reported
gender, in the order the split builder drew them: the split's own rule. That
order is recomputed from the split file and must reproduce its assignment
exactly. A Neyshekar speaker has no reserve and is left out. Either way it is
listed.

Output: ``<out>/manifest.json`` in the benchmark's format (``<label>_000.wav``
is the reference, ``_001`` to ``_004`` the held-out clips), each clip's bytes
exactly as the parquet stores them, and ``<out>/roster.json``: who is in, who
was replaced or left out and why, and every file's sha256.

Only records of the sealed split are looked up, so no training or valid clip
can be written.

    .venv/bin/python scripts/prepare_test_corpus.py --splits eval/manifests/adaptation_splits_v1.jsonl.gz \\
        --source neyshekar --parquet-dir /tmp/neyshekar-test --out /tmp/test-ney
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

from build_adaptation_splits import (  # noqa: E402
    GENDERS,
    HELD_OUT,
    MIN_HELD_OUT_SECONDS,
    MIN_REFERENCE_SECONDS,
    SEED,
    draw_common_voice,
)
from prepare_adaptation_data import SAMPLE_RATE, check, parquet_key  # noqa: E402

WHOLE_REFERENCE_SECONDS = 11.4
PREFIX = {"neyshekar": "ney", "cv26": "cvt"}
CORPUS = {"neyshekar": "neyshekar_v6_test", "cv26": "common_voice_26_fa_clean_test"}
ROLE_RULE = (f"reference: the longest clip of {MIN_REFERENCE_SECONDS}-{WHOLE_REFERENCE_SECONDS} s; held out: "
             f"the {HELD_OUT} longest of the rest of at least {MIN_HELD_OUT_SECONDS} s")


def load_records(path: Path) -> list[dict[str, Any]]:
    # Kaggle unpacks a .gz inside a dataset, so the plain file is accepted too.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def by_length(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(clips, key=lambda r: (-r["duration"], r["clip"]))


def whole_reference_roles(clips: list[dict[str, Any]]) -> tuple[dict, list[dict]] | None:
    fitting = [c for c in by_length(clips) if MIN_REFERENCE_SECONDS <= c["duration"] <= WHOLE_REFERENCE_SECONDS]
    if not fitting:
        return None
    reference = fitting[0]
    held = [c for c in by_length(clips) if c is not reference and c["duration"] >= MIN_HELD_OUT_SECONDS]
    if len(held) < HELD_OUT:
        return None
    return reference, held[:HELD_OUT]


def reserve_order(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The Common Voice reserve per gender, in the order the split builder drew it."""
    speakers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["source"] == "cv26":
            label = "" if record["gender"] == "unknown" else record["gender"]
            speakers[record["speaker"]].append({**record, "gender_label": label})
    excluded = {s for s, clips in speakers.items() if clips[0]["split"] == "excluded-development"}
    drawn = draw_common_voice(speakers, excluded, SEED)
    recorded = {s: clips[0]["split"] for s, clips in speakers.items()
                if clips[0]["split"] in {"test", "test-reserve", "valid"}}
    if drawn != recorded:
        raise SystemExit("the Common Voice draw does not reproduce from the split file - refusing to use its reserve")
    gender = {s: clips[0]["gender"] for s, clips in speakers.items()}
    order: dict[str, list[str]] = {label: [] for label in GENDERS}
    for speaker, split in drawn.items():
        if split == "test-reserve":
            order[gender[speaker]].append(speaker)
    return order


def sealed(records: list[dict[str, Any]], source: str) -> tuple[list[str], dict[str, list[str]], dict[str, list[dict]]]:
    """Test speakers, the reserve (Common Voice only) and every clip of both."""
    clips: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["source"] == source and record["split"] in {"test", "test-reserve"}:
            clips[record["speaker"]].append(record)
    test = sorted(s for s, rows in clips.items() if rows[0]["split"] == "test")
    reserve = reserve_order(records) if source == "cv26" else {}
    return test, reserve, dict(clips)


def read_audio(source: str, parquet_dir: Path, wanted: dict[str, dict]) -> dict[str, bytes]:
    import pyarrow.parquet as pq

    keys = {(r["clip"] if source == "cv26" else r["clip"].removeprefix("neyshekar-")): r["clip"]
            for r in wanted.values()}
    found: dict[str, bytes] = {}
    columns = ["audio"] if source == "cv26" else ["id", "audio"]
    for path in sorted(parquet_dir.rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                clip = keys.get(parquet_key(source, row))
                if clip is not None and clip not in found:
                    found[clip] = (row.get("audio") or {}).get("bytes") or b""
    return found


def problems_of(roles: tuple[dict, list[dict]], audio: dict[str, bytes], source: str) -> list[str]:
    reference, held = roles
    out = []
    for record in (reference, *held):
        data = audio.get(record["clip"])
        problem = "not in the parquet files" if data is None else (
            "no audio bytes" if not data else check(data, SAMPLE_RATE[source], record["duration"]))
        if problem:
            out.append(f"{record['clip']}: {problem}")
    return out


def roster(source: str, test: list[str], reserve: dict[str, list[str]], clips: dict[str, list[dict]],
           audio: dict[str, bytes]) -> dict[str, Any]:
    chosen: list[dict[str, Any]] = []
    left_out: list[dict[str, str]] = []
    replaced: list[dict[str, str]] = []
    spare = {gender: list(order) for gender, order in reserve.items()}

    def usable(speaker: str) -> tuple[tuple[dict, list[dict]] | None, str]:
        roles = whole_reference_roles(clips[speaker])
        if roles is None:
            return None, f"no clip of {MIN_REFERENCE_SECONDS}-{WHOLE_REFERENCE_SECONDS} s with {HELD_OUT} more to hold out"
        problems = problems_of(roles, audio, source)
        return (None, "; ".join(problems)) if problems else (roles, "")

    for speaker in test:
        roles, reason = usable(speaker)
        if roles is not None:
            chosen.append({"speaker": speaker, "roles": roles})
            continue
        mechanical = not reason.startswith("no clip")
        gender = clips[speaker][0]["gender"]
        substitute = None
        while mechanical and spare.get(gender):
            candidate = spare[gender].pop(0)
            candidate_roles, candidate_reason = usable(candidate)
            if candidate_roles is not None:
                substitute = candidate
                chosen.append({"speaker": candidate, "roles": candidate_roles, "replaces": speaker})
                break
            left_out.append({"speaker": candidate, "reason": f"reserve, unusable: {candidate_reason}"})
        if substitute is not None:
            replaced.append({"speaker": speaker, "reason": reason, "by": substitute})
        else:
            left_out.append({"speaker": speaker, "reason": reason})
    chosen.sort(key=lambda row: row["speaker"])
    for index, row in enumerate(chosen, start=1):
        row["label"] = f"{PREFIX[source]}{index:02d}"
    return {"chosen": chosen, "left_out": left_out, "replaced": replaced}


def write(source: str, result: dict[str, Any], audio: dict[str, bytes], clips: dict[str, list[dict]],
          out: Path, splits_sha256: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    entries, files, speakers, changed = [], {}, [], []
    for row in result["chosen"]:
        reference, held = row["roles"]
        recorded = [c["clip"] for c in by_length(clips[row["speaker"]]) if c.get("role") == "reference"]
        recorded += sorted(c["clip"] for c in clips[row["speaker"]] if c.get("role") == "held_out")
        if recorded != [reference["clip"], *sorted(c["clip"] for c in held)]:
            changed.append(row["speaker"])
        for position, record in enumerate((reference, *held)):
            name = f"{row['label']}_{position:03d}.wav"
            data = audio[record["clip"]]
            (out / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
            entries.append({
                "speaker": row["label"], "file": name, "role": "reference" if position == 0 else "held_out",
                "transcript": record["text"].strip(), "duration": record["duration"],
                "gender": record["gender"], "corpus": CORPUS[source],
                # Each clip is its own recording in both sources, so the clip id is the session.
                "session": record["clip"], "speaker_hash": row["speaker"], "source_ref": record.get("source_ref"),
            })
        speakers.append({"label": row["label"], "speaker": row["speaker"], "gender": reference["gender"],
                         "reference": reference["clip"], "reference_seconds": reference["duration"],
                         "held_out": [c["clip"] for c in held], "replaces": row.get("replaces")})
    manifest = {
        "version": 1,
        "purpose": f"sealed test set, {source}: rendered and scored once per final candidate",
        "speakers": len(result["chosen"]),
        "clips": len(entries),
        "sources": [CORPUS[source]],
        "notes": ROLE_RULE,
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "source": source,
        "splits_sha256": splits_sha256,
        "role_rule": ROLE_RULE,
        "speakers": speakers,
        "roles_differ_from_the_split": changed,
        "replaced": result["replaced"],
        "left_out": result["left_out"],
        "files": files,
        "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
    }
    (out / "roster.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--source", choices=sorted(PREFIX), required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    records = load_records(args.splits)
    if args.splits.suffix == ".gz":
        with gzip.open(args.splits) as handle:
            raw = handle.read()
    else:
        raw = args.splits.read_bytes()
    test, reserve, clips = sealed(records, args.source)
    wanted = {}
    for speaker in [*test, *(s for order in reserve.values() for s in order)]:
        roles = whole_reference_roles(clips[speaker])
        if roles:
            wanted.update({r["clip"]: r for r in (roles[0], *roles[1])})
    audio = read_audio(args.source, args.parquet_dir, wanted)
    result = roster(args.source, test, reserve, clips, audio)
    record = write(args.source, result, audio, clips, args.out, hashlib.sha256(raw).hexdigest())
    print(json.dumps({"source": args.source, "speakers": len(record["speakers"]),
                      "roles_differ_from_the_split": len(record["roles_differ_from_the_split"]),
                      "replaced": record["replaced"], "left_out": record["left_out"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
