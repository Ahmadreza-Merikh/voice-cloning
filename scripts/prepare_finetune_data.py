#!/usr/bin/env python3
"""Build an F5-TTS training manifest from multi-speaker Persian audio.

Why multi-speaker
-----------------
Measurement says the Persian checkpoint in use pronounces Persian better than
the corpus it is scored against (2.9% CER against a 7.8% recogniser floor) and
transfers identity poorly (0.520 held-out against a 0.689 human ceiling). Every
inference knob is inert. That pattern - excellent language, weak voice - is what
a model trained on a lot of Persian from few speakers looks like: it learned the
language and lost the ability to generalise across voices.

So the training set here is selected for *speaker diversity* rather than hours.
Common Voice Persian contributes thousands of speakers at a few clips each,
which is the opposite of what the existing checkpoint most likely saw.

This script only prepares data. It does not train: a full-quality run does not
belong on a 16 GB laptop, and `docs/REMOTE-TRAINING.md` carries the recipe.

Output layout (what F5-TTS's finetune CLI expects)
--------------------------------------------------
    <out>/wavs/<id>.wav
    <out>/metadata.csv          id|transcript
    <out>/dataset_info.json     provenance, speaker counts, hours

Usage
-----
    .venv/bin/python scripts/prepare_finetune_data.py --out data/finetune-fa
    .venv/bin/python scripts/prepare_finetune_data.py --max-per-speaker 8 --hours 25
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"

#: Below this a clip is too short to teach anything about a voice.
MIN_SECONDS = 2.5
MAX_SECONDS = 20.0


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def load_eval_speakers(corpus_dir: Path) -> set[str]:
    """Speakers held out for evaluation, which must never be trained on.

    Without this the benchmark measures memorisation. The manifest records a
    truncated ``client_id`` for exactly this check.
    """
    manifest = corpus_dir / "manifest.json"
    if not manifest.is_file():
        print(f"  no eval corpus at {manifest}; nothing to exclude", file=sys.stderr)
        return set()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {e["speaker_hash"] for e in data["entries"] if e.get("speaker_hash")}


def select(
    kept: list[dict[str, str]],
    present: set[str],
    excluded: set[str],
    max_per_speaker: int,
    hours: float,
) -> list[dict[str, str]]:
    """Round-robin across speakers so no voice dominates the set."""
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped_eval = 0
    for row in kept:
        if row["output_audio"] not in present:
            continue
        if not MIN_SECONDS <= float(row["output_duration"]) <= MAX_SECONDS:
            continue
        if float(row["clipping_ratio"]) > 0.001 or float(row["rms_dbfs"]) < -34.0:
            continue
        if row["client_id"][:16] in excluded:
            skipped_eval += 1
            continue
        by_speaker[row["client_id"]].append(row)

    if skipped_eval:
        print(f"  held back {skipped_eval} clips from evaluation speakers")

    # Longest clips first within a speaker: more voice per sample.
    for rows in by_speaker.values():
        rows.sort(key=lambda r: -float(r["output_duration"]))

    budget = hours * 3600
    chosen: list[dict[str, str]] = []
    total = 0.0
    for depth in range(max_per_speaker):
        for speaker in sorted(by_speaker):
            rows = by_speaker[speaker]
            if depth >= len(rows):
                continue
            row = rows[depth]
            chosen.append(row)
            total += float(row["output_duration"])
            if total >= budget:
                return chosen
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "finetune-fa")
    parser.add_argument("--eval-corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    parser.add_argument("--max-per-speaker", type=int, default=6)
    parser.add_argument("--hours", type=float, default=25.0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/persian-voice-cloning/eval-data").expanduser(),
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    _rule("Selecting multi-speaker Persian audio")
    kept_path = hf_hub_download(
        repo_id=CV_REPO, filename="reports/kept.csv", repo_type="dataset", cache_dir=str(args.cache)
    )
    with open(kept_path, encoding="utf-8") as handle:
        kept = list(csv.DictReader(handle))
    present = {f for f in list_repo_files(CV_REPO, repo_type="dataset") if f.startswith("wavs/")}
    excluded = load_eval_speakers(args.eval_corpus)
    print(f"  {len(kept)} clips listed, {len(present)} published, {len(excluded)} eval speakers excluded")

    chosen = select(kept, present, excluded, args.max_per_speaker, args.hours)
    speakers = {r["client_id"] for r in chosen}
    total = sum(float(r["output_duration"]) for r in chosen)
    print(f"  selected {len(chosen)} clips, {len(speakers)} speakers, {total / 3600:.1f} hours")
    if not chosen:
        print("Nothing selected.", file=sys.stderr)
        return 1

    wavs = args.out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    _rule("Fetching audio")
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(chosen, start=1):
        if index % 250 == 0:
            print(f"  {index}/{len(chosen)}", flush=True)
        source = hf_hub_download(
            repo_id=CV_REPO,
            filename=row["output_audio"],
            repo_type="dataset",
            cache_dir=str(args.cache),
        )
        name = Path(row["output_audio"]).stem
        shutil.copyfile(source, wavs / f"{name}.wav")
        rows.append((name, " ".join(row["text"].split())))

    with (args.out / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for name, text in rows:
            writer.writerow([name, text])

    info: dict[str, Any] = {
        "source": CV_REPO,
        "clips": len(rows),
        "speakers": len(speakers),
        "hours": round(total / 3600, 2),
        "max_per_speaker": args.max_per_speaker,
        "excluded_eval_speakers": sorted(excluded),
        "note": (
            "Selected for speaker diversity, not hours: the checkpoint being improved "
            "already reads Persian well and transfers identity poorly, which is what "
            "training on many hours of few speakers produces. Evaluation speakers are "
            "excluded so the benchmark cannot measure memorisation."
        ),
    }
    (args.out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _rule("Ready")
    print(f"  {args.out}")
    print(f"  {len(rows)} clips | {len(speakers)} speakers | {total / 3600:.1f} h")
    print("\n  Next: docs/REMOTE-TRAINING.md - this does not train locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
