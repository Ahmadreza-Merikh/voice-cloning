#!/usr/bin/env python3
"""Parallel resume-friendly prepare for F5 finetune data."""
from __future__ import annotations
import argparse, csv, json, random, shutil, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"
MIN_SECONDS, MAX_SECONDS = 2.5, 20.0

def load_eval_speakers(corpus_dir: Path) -> set[str]:
    manifest = corpus_dir / "manifest.json"
    if not manifest.is_file():
        return set()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {e["speaker_hash"] for e in data["entries"] if e.get("speaker_hash")}

def select(kept, excluded, max_per_speaker, hours):
    by_speaker = defaultdict(list)
    skipped_eval = 0
    for row in kept:
        audio = row.get("output_audio") or ""
        if not audio.startswith("wavs/"):
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
        if row.get("client_id", "")[:16] in excluded:
            skipped_eval += 1
            continue
        by_speaker[row["client_id"]].append(row)
    print(f"  held back {skipped_eval} clips from evaluation speakers", flush=True)
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

def hf_download(filename: str, cache: Path, retries: int = 8):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    delay = 5.0
    for attempt in range(retries):
        try:
            return hf_hub_download(
                repo_id=CV_REPO,
                filename=filename,
                repo_type="dataset",
                cache_dir=str(cache),
            )
        except HfHubHTTPError as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                wait = delay + random.uniform(0, 3)
                # parse retry-after if present
                if "Retry after" in msg:
                    try:
                        wait = max(wait, float(msg.split("Retry after")[1].split("seconds")[0].strip()) + 1)
                    except Exception:
                        pass
                print(f"  rate-limit on {filename}: sleep {wait:.0f}s (try {attempt+1}/{retries})", flush=True)
                time.sleep(wait)
                delay = min(delay * 1.7, 120)
                continue
            raise
        except Exception as e:
            if attempt + 1 >= retries:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  err {filename}: {e}; sleep {wait:.0f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 1.5, 90)
    raise RuntimeError(f"failed {filename}")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "finetune-fa")
    p.add_argument("--eval-corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    p.add_argument("--max-per-speaker", type=int, default=6)
    p.add_argument("--hours", type=float, default=25.0)
    p.add_argument("--cache", type=Path, default=Path("~/.cache/persian-voice-cloning/eval-data").expanduser())
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    print("Selecting multi-speaker Persian audio", flush=True)
    kept_path = hf_download("reports/kept.csv", args.cache)
    with open(kept_path, encoding="utf-8") as handle:
        kept = list(csv.DictReader(handle))
    excluded = load_eval_speakers(args.eval_corpus)
    print(f"  {len(kept)} clips listed, {len(excluded)} eval speakers excluded", flush=True)
    chosen = select(kept, excluded, args.max_per_speaker, args.hours)
    speakers = {r["client_id"] for r in chosen}
    total = sum(float(r["output_duration"]) for r in chosen)
    print(f"  selected {len(chosen)} clips, {len(speakers)} speakers, {total/3600:.1f} hours", flush=True)

    wavs = args.out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    todo = []
    have = 0
    for row in chosen:
        name = Path(row["output_audio"]).stem
        dest = wavs / f"{name}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            have += 1
        else:
            todo.append(row)
    print(f"  already have {have}; downloading {len(todo)} with {args.workers} workers", flush=True)

    def one(row):
        source = hf_download(row["output_audio"], args.cache)
        name = Path(row["output_audio"]).stem
        dest = wavs / f"{name}.wav"
        if not dest.exists() or dest.stat().st_size < 1000:
            shutil.copyfile(source, dest)
        return name, " ".join(row["text"].split()), float(row["output_duration"])

    rows_map = {}
    for row in chosen:
        name = Path(row["output_audio"]).stem
        dest = wavs / f"{name}.wav"
        if dest.exists() and dest.stat().st_size > 1000:
            rows_map[name] = (" ".join(row["text"].split()), float(row["output_duration"]))

    done = 0
    fails = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, row): row for row in todo}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                name, text, dur = fut.result()
                rows_map[name] = (text, dur)
                done += 1
            except Exception as e:
                fails += 1
                print(f"  FAIL {row.get(output_audio)}: {e}", flush=True)
            if (done + fails) % 50 == 0 or (done + fails) == len(todo):
                print(f"  progress {done}/{len(todo)} ok, {fails} fail, total_on_disk={len(rows_map)}", flush=True)

    rows = []
    for row in chosen:
        name = Path(row["output_audio"]).stem
        if name not in rows_map:
            continue
        text, _ = rows_map[name]
        rows.append((name, text))

    with (args.out / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for name, text in rows:
            writer.writerow([name, text])

    hours = sum(rows_map[n][1] for n,_ in rows) / 3600 if rows else 0
    info = {
        "source": CV_REPO,
        "clips": len(rows),
        "speakers": len({r["client_id"] for r in chosen if Path(r["output_audio"]).stem in rows_map}),
        "hours": round(hours, 2),
        "max_per_speaker": args.max_per_speaker,
        "excluded_eval_speakers": sorted(excluded),
        "failed_downloads": fails,
        "note": "Speaker-diversity selection; eval speakers excluded; resume/backoff downloads.",
    }
    (args.out / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ready: {args.out} | {len(rows)} clips | {info[speakers]} speakers | {info[hours]} h | fails={fails}", flush=True)
    return 0 if len(rows) > 500 else 2

if __name__ == "__main__":
    raise SystemExit(main())
