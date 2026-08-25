#!/usr/bin/env python3
"""Robust prepare for F5 finetune data.

Same selection rules as prepare_finetune_data.py, but:
- retries list_repo_files / downloads
- only present files (no 404)
- parallel download with resume
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"
MIN_SECONDS = 2.5
MAX_SECONDS = 20.0


def load_eval_speakers(corpus_dir: Path) -> set[str]:
    manifest = corpus_dir / "manifest.json"
    if not manifest.is_file():
        return set()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {e["speaker_hash"] for e in data["entries"] if e.get("speaker_hash")}


def retry_call(fn, *, tries=8, base=2.0, what="op"):
    delay = base
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            wait = delay + random.random() * 2
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                wait = max(wait, 15)
            print(
                f"  retry {what} {i+1}/{tries}: {type(e).__name__}: {msg[:160]} (sleep {wait:.1f}s)",
                flush=True,
            )
            time.sleep(wait)
            delay = min(delay * 1.7, 90)
    raise last


def select(kept, present, excluded, max_per_speaker, hours):
    by_speaker = defaultdict(list)
    skipped_eval = 0
    for row in kept:
        audio = row.get("output_audio") or ""
        if audio not in present:
            continue
        try:
            dur = float(row["output_duration"])
        except Exception:
            continue
        if not MIN_SECONDS <= dur <= MAX_SECONDS:
            continue
        try:
            if float(row.get("clipping_ratio", 0)) > 0.001 or float(row.get("rms_dbfs", -20)) < -34.0:
                continue
        except Exception:
            pass
        if (row.get("client_id") or "")[:16] in excluded:
            skipped_eval += 1
            continue
        by_speaker[row["client_id"]].append(row)
    print(f"  held back {skipped_eval} eval clips; speakers pool {len(by_speaker)}", flush=True)
    for rows in by_speaker.values():
        rows.sort(key=lambda r: -float(r["output_duration"]))
    budget = hours * 3600
    chosen, total = [], 0.0
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "finetune-fa")
    ap.add_argument("--eval-corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    ap.add_argument("--max-per-speaker", type=int, default=6)
    ap.add_argument("--hours", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--present-file",
        type=Path,
        default=None,
        help="Optional text file with present wav paths (one per line). Skips list_repo_files.",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/persian-voice-cloning/eval-data").expanduser(),
    )
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    print("\nSelecting multi-speaker Persian audio (robust)\n" + "-" * 46, flush=True)
    kept_path = retry_call(
        lambda: hf_hub_download(
            CV_REPO, "reports/kept.csv", repo_type="dataset", cache_dir=str(args.cache)
        ),
        what="kept.csv",
    )
    with open(kept_path, encoding="utf-8") as f:
        kept = list(csv.DictReader(f))
    if args.present_file and args.present_file.is_file():
        present = {
            line.strip()
            for line in args.present_file.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("wavs/")
        }
        print(f"  present from file {args.present_file}: {len(present)}", flush=True)
    else:
        present = retry_call(
            lambda: {
                f
                for f in list_repo_files(CV_REPO, repo_type="dataset")
                if f.startswith("wavs/")
            },
            what="list_repo_files",
            tries=10,
            base=3.0,
        )
    excluded = load_eval_speakers(args.eval_corpus)
    print(
        f"  listed={len(kept)} present={len(present)} eval_speakers={len(excluded)}",
        flush=True,
    )

    chosen = select(kept, present, excluded, args.max_per_speaker, args.hours)
    speakers = {r["client_id"] for r in chosen}
    total = sum(float(r["output_duration"]) for r in chosen)
    print(
        f"  selected {len(chosen)} clips / {len(speakers)} speakers / {total/3600:.1f} h",
        flush=True,
    )
    if not chosen:
        print("Nothing selected.", file=sys.stderr)
        return 1

    wavs = args.out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    def one(row):
        name = Path(row["output_audio"]).stem
        dest = wavs / f"{name}.wav"
        text = " ".join(row["text"].split())
        if dest.exists() and dest.stat().st_size > 1000:
            return name, text, "skip"

        def dl():
            src = hf_hub_download(
                CV_REPO,
                row["output_audio"],
                repo_type="dataset",
                cache_dir=str(args.cache),
            )
            shutil.copyfile(src, dest)
            return name, text, "ok"

        return retry_call(dl, what=name, tries=6, base=2.0)

    print(f"\nFetching audio with {args.workers} workers\n" + "-" * 34, flush=True)
    rows = []
    done = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, r): r for r in chosen}
        for fut in as_completed(futs):
            try:
                name, text, status = fut.result()
                rows.append((name, text))
                if status == "skip":
                    skip += 1
                else:
                    done += 1
            except Exception as e:
                fail += 1
                if fail <= 20:
                    print(f"  fail: {e}", flush=True)
            n = done + skip + fail
            if n % 100 == 0 or n == len(chosen):
                print(
                    f"  progress {n}/{len(chosen)} new={done} skip={skip} fail={fail}",
                    flush=True,
                )

    got = {n: t for n, t in rows}
    meta_rows = []
    for row in chosen:
        name = Path(row["output_audio"]).stem
        if name in got and (wavs / f"{name}.wav").exists():
            meta_rows.append((name, got[name]))

    with (args.out / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for name, text in meta_rows:
            w.writerow([name, text])

    ok_names = {n for n, _ in meta_rows}
    total_ok = sum(
        float(r["output_duration"])
        for r in chosen
        if Path(r["output_audio"]).stem in ok_names
    )
    speakers_ok = {
        r["client_id"] for r in chosen if Path(r["output_audio"]).stem in ok_names
    }
    info: dict[str, Any] = {
        "source": CV_REPO,
        "clips": len(meta_rows),
        "speakers": len(speakers_ok),
        "hours": round(total_ok / 3600, 2),
        "max_per_speaker": args.max_per_speaker,
        "excluded_eval_speakers": sorted(excluded),
        "present_files": len(present),
        "selected_requested": len(chosen),
        "download_fail": fail,
        "note": "Present-filtered multi-speaker set for remote F5 finetune.",
    }
    (args.out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nReady\n-----", flush=True)
    print(f"  {args.out}", flush=True)
    print(
        f"  {len(meta_rows)} clips | {len(speakers_ok)} speakers | {total_ok/3600:.1f} h | fail={fail}",
        flush=True,
    )
    if len(meta_rows) < 1000:
        print("Too few clips.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
