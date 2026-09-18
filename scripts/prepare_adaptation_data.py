#!/usr/bin/env python3
"""Write the audio and the JSONL that OmniVoice's official tokeniser reads.

Input is the split decided by ``scripts/build_adaptation_splits.py`` and the
two sources' parquet files at their pinned revisions. Output is one WAV per
clip - the bytes stored in the parquet, unchanged - and, per split, the JSONL
``run_finetune.sh`` expects: ``{"id", "audio_path", "text", "language_id"}``.

Only ``train`` and ``valid`` can be written. Test and reserve clips are refused
by name, so a sealed split cannot be read into a training run by accident.

Every file is checked before it is listed: it must parse as audio, be mono,
have the source's sample rate, and last what the split file says to within
50 ms. What does not is counted and left out, and the summary says so.

``--limit`` takes the first N clips per source in the order the parquet files
are read - the probe's slice. Without it every clip of the split must be found,
or the run stops.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WRITABLE = {"train", "valid"}
SAMPLE_RATE = {"cv26": 24000, "neyshekar": 16000}
TOLERANCE = 0.05


def load_split(path: Path, wanted: set[str]) -> dict[str, dict[str, dict]]:
    """Records of the wanted splits, keyed by source and by the key each parquet carries."""
    refused = wanted - WRITABLE
    if refused:
        raise SystemExit(f"refused: {sorted(refused)} - only {sorted(WRITABLE)} may be written")
    records: dict[str, dict[str, dict]] = defaultdict(dict)
    # Kaggle unpacks a .gz inside a dataset, so the plain file is accepted too.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] not in wanted:
                continue
            key = record["clip"] if record["source"] == "cv26" else record["clip"].removeprefix("neyshekar-")
            records[record["source"]][key] = record
    return records


def parquet_key(source: str, row: dict) -> str | None:
    if source == "cv26":
        audio = row.get("audio") or {}
        return audio.get("path")
    return str(row["id"])


def check(data: bytes, expected_rate: int, expected_seconds: float) -> str | None:
    import soundfile

    try:
        info = soundfile.info(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - reported per clip
        return f"unreadable: {type(exc).__name__}"
    if info.channels != 1:
        return f"{info.channels} channels"
    if info.samplerate != expected_rate:
        return f"{info.samplerate} Hz"
    if abs(info.frames / info.samplerate - expected_seconds) > TOLERANCE:
        return f"lasts {info.frames / info.samplerate:.3f} s, split says {expected_seconds:.3f} s"
    return None


def write(
    records: dict[str, dict[str, dict]],
    parquet_dirs: dict[str, Path],
    out: Path,
    limit: int | None,
) -> dict:
    import pyarrow.parquet as pq

    lines: dict[str, list[str]] = defaultdict(list)
    written: Counter = Counter()
    seconds: Counter = Counter()
    problems: dict[str, list[str]] = defaultdict(list)
    found: dict[str, set[str]] = defaultdict(set)
    for source, folder in parquet_dirs.items():
        wanted = records.get(source, {})
        columns = ["audio"] if source == "cv26" else ["id", "audio"]
        for path in sorted(folder.rglob("*.parquet")):
            if limit is not None and len(found[source]) >= limit:
                break
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    key = parquet_key(source, row)
                    record = wanted.get(key)
                    if record is None or key in found[source]:
                        continue
                    if limit is not None and len(found[source]) >= limit:
                        break
                    found[source].add(key)
                    data = (row.get("audio") or {}).get("bytes")
                    problem = "no audio bytes" if not data else check(data, SAMPLE_RATE[source], record["duration"])
                    if problem:
                        problems[source].append(f"{record['clip']}: {problem}")
                        continue
                    # WebDataset keys end at the first dot, so the id has none.
                    name = Path(record["clip"]).stem
                    target = out / "audio" / source / f"{name}.wav"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    lines[record["split"]].append(json.dumps(
                        {"id": name, "audio_path": str(target.resolve()),
                         "text": record["text"], "language_id": "fa"},
                        ensure_ascii=False,
                    ))
                    written[(record["split"], source)] += 1
                    seconds[(record["split"], source)] += record["duration"]
    missing = {source: len(set(wanted) - found[source]) for source, wanted in records.items()}
    if limit is None and any(missing.values()):
        raise SystemExit(f"clips of the split not found in the parquet files: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    for split, entries in lines.items():
        (out / f"{split}.jsonl").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return {
        "written": {f"{split}/{source}": n for (split, source), n in sorted(written.items())},
        "hours": {f"{split}/{source}": round(s / 3600, 3) for (split, source), s in sorted(seconds.items())},
        "left_out": {source: len(items) for source, items in problems.items()},
        "left_out_examples": {source: items[:20] for source, items in problems.items()},
        "not_found": missing,
        "limit": limit,
        "jsonl": {split: str((out / f"{split}.jsonl").resolve()) for split in lines},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True, choices=sorted(WRITABLE | {"test", "test-reserve"}))
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--neyshekar-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="First N clips per source: the probe's slice")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    records = load_split(args.splits, set(args.split))
    dirs = {name: path for name, path in (("cv26", args.cv_dir), ("neyshekar", args.neyshekar_dir)) if path}
    summary = write(records, dirs, args.out, args.limit)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary:
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
