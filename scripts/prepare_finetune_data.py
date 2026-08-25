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

Surviving the download
----------------------
Nine thousand files over anonymous Hub requests will hit a rate limit, and an
earlier serial version of this script died partway with 1157 wavs on disk and no
`metadata.csv` - the transcripts are written last, so a crash leaves audio that
cannot be trained on at all. Every fetch now retries with backoff, files already
on disk are skipped so a rerun resumes, and the manifest is written from what
actually landed rather than from what was planned.

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
import random
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"

#: Below this a clip is too short to teach anything about a voice.
MIN_SECONDS = 2.5
MAX_SECONDS = 20.0

T = TypeVar("T")


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def retry(call: Callable[[], T], *, what: str, tries: int = 8, base: float = 2.0) -> T:
    """Retry with jittered backoff, waiting longer when the Hub says slow down.

    Anonymous Hub traffic is throttled, and a throttled request looks like an
    ordinary failure. Backing off is the difference between finishing and losing
    the whole run to one bad minute.
    """
    delay = base
    failure: Exception | None = None
    for attempt in range(tries):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - any failure is worth a retry here
            failure = error
            message = str(error)
            wait = delay + random.random() * 2
            if "429" in message or "rate limit" in message.lower():
                wait = max(wait, 15.0)
            print(
                f"  retry {what} {attempt + 1}/{tries}: "
                f"{type(error).__name__}: {message[:160]} (sleep {wait:.1f}s)",
                flush=True,
            )
            time.sleep(wait)
            delay = min(delay * 1.7, 90.0)
    assert failure is not None
    raise failure


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


def load_present(repo_files: Path | None, cache: Path) -> set[str]:
    """Which wavs the dataset actually publishes.

    ``kept.csv`` lists 34134 clips; the repository holds 9985. Selecting from the
    listing rather than the repository is a guaranteed 404 partway through, so
    the two are intersected. Listing the repository is also the single most
    failure-prone call in the script, hence the file escape hatch.
    """
    if repo_files and repo_files.is_file():
        present = {
            line.strip()
            for line in repo_files.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("wavs/")
        }
        print(f"  published wavs read from {repo_files}: {len(present)}")
        return present

    from huggingface_hub import list_repo_files

    del cache  # listing is not cached; the parameter keeps call sites uniform
    return retry(
        lambda: {
            name
            for name in list_repo_files(CV_REPO, repo_type="dataset")
            if name.startswith("wavs/")
        },
        what="list_repo_files",
        tries=10,
        base=3.0,
    )


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


def fetch(
    chosen: list[dict[str, str]], wavs: Path, cache: Path, workers: int
) -> tuple[dict[str, str], int]:
    """Download the selection, skipping whatever a previous run already got."""
    from huggingface_hub import hf_hub_download

    def one(row: dict[str, str]) -> tuple[str, str, str]:
        name = Path(row["output_audio"]).stem
        text = " ".join(row["text"].split())
        destination = wavs / f"{name}.wav"
        if destination.exists() and destination.stat().st_size > 1000:
            return name, text, "skip"

        def download() -> tuple[str, str, str]:
            source = hf_hub_download(
                repo_id=CV_REPO,
                filename=row["output_audio"],
                repo_type="dataset",
                cache_dir=str(cache),
            )
            shutil.copyfile(source, destination)
            return name, text, "new"

        return retry(download, what=name, tries=6)

    landed: dict[str, str] = {}
    new = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in chosen]
        for future in as_completed(futures):
            try:
                name, text, status = future.result()
                landed[name] = text
                new += status == "new"
                skipped += status == "skip"
            except Exception as error:  # noqa: BLE001 - one clip is not the run
                failed += 1
                if failed <= 20:
                    print(f"  gave up on a clip: {error}", flush=True)
            done = new + skipped + failed
            if done % 250 == 0 or done == len(chosen):
                print(
                    f"  {done}/{len(chosen)}  new={new} resumed={skipped} lost={failed}",
                    flush=True,
                )
    return landed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "finetune-fa")
    parser.add_argument("--eval-corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    parser.add_argument("--max-per-speaker", type=int, default=6)
    parser.add_argument("--hours", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--repo-files",
        type=Path,
        default=None,
        help="Text file of published wav paths, one per line, instead of listing the Hub.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/persian-voice-cloning/eval-data").expanduser(),
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    _rule("Selecting multi-speaker Persian audio")
    kept_path = retry(
        lambda: hf_hub_download(
            repo_id=CV_REPO,
            filename="reports/kept.csv",
            repo_type="dataset",
            cache_dir=str(args.cache),
        ),
        what="kept.csv",
    )
    with open(kept_path, encoding="utf-8") as handle:
        kept = list(csv.DictReader(handle))
    present = load_present(args.repo_files, args.cache)
    excluded = load_eval_speakers(args.eval_corpus)
    print(
        f"  {len(kept)} clips listed, {len(present)} published, "
        f"{len(excluded)} eval speakers excluded"
    )

    chosen = select(kept, present, excluded, args.max_per_speaker, args.hours)
    total = sum(float(r["output_duration"]) for r in chosen)
    print(
        f"  selected {len(chosen)} clips, "
        f"{len({r['client_id'] for r in chosen})} speakers, {total / 3600:.1f} hours"
    )
    if not chosen:
        print("Nothing selected.", file=sys.stderr)
        return 1

    wavs = args.out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    _rule("Fetching audio")
    landed, failed = fetch(chosen, wavs, args.cache, args.workers)

    # Write the manifest from what is on disk. A row pointing at a missing wav
    # fails the training run much later and much less clearly than it fails here.
    rows = [
        (name, landed[name])
        for name in (Path(r["output_audio"]).stem for r in chosen)
        if name in landed and (wavs / f"{name}.wav").is_file()
    ]
    with (args.out / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for name, text in rows:
            writer.writerow([name, text])

    got = {name for name, _ in rows}
    kept_rows = [r for r in chosen if Path(r["output_audio"]).stem in got]
    speakers = {r["client_id"] for r in kept_rows}
    hours = sum(float(r["output_duration"]) for r in kept_rows) / 3600

    info: dict[str, Any] = {
        "source": CV_REPO,
        "clips": len(rows),
        "speakers": len(speakers),
        "hours": round(hours, 2),
        "max_per_speaker": args.max_per_speaker,
        "excluded_eval_speakers": sorted(excluded),
        "published_files": len(present),
        "selected": len(chosen),
        "download_failures": failed,
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
    print(f"  {len(rows)} clips | {len(speakers)} speakers | {hours:.1f} h | lost {failed}")
    if failed:
        print("  Rerun to retry the losses; existing files are skipped.")
    print("\n  Next: docs/REMOTE-TRAINING.md - this does not train locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
