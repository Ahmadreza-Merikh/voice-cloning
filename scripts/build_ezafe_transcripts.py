#!/usr/bin/env python3
"""Write the app's ezafe into every train and valid transcript of the adaptation. It trains nothing.

Registered before it ran (eval/runs/persian-voice-omnivoice-ezafe-train.json).
Each distinct transcript of adaptation_splits_v1's train and valid splits goes
through the app's own writer (EzafeWriter: hazm's ParsBERT tagger, no ezafe on
روی, none joining مدیر to عامل) and nothing else: not normalised, its leading
and trailing spacing kept. Every word of the result is checked to be the
original word, or it with a kasra, a hamza above or a ی added at its end -
anything else stops the run. The map goes to
eval/manifests/adaptation_ezafe_texts_v1.jsonl.gz, original text to marked
text; the training sessions replace each shard row's text through it.
It stops if available memory falls under 3.4 GB or pressure reaches warning.

    .venv/bin/python scripts/build_ezafe_transcripts.py
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone import memory  # noqa: E402

SPLITS = ROOT / "eval" / "manifests" / "adaptation_splits_v1.jsonl.gz"
OUT = ROOT / "eval" / "manifests" / "adaptation_ezafe_texts_v1.jsonl.gz"
SUMMARY = ROOT / "eval" / "manifests" / "adaptation_ezafe_texts_v1.summary.json"
BATCH = 4000
STOP_GB = 3.4
KASRA, HAMZA_ABOVE = "ِ", "ٔ"


def watch() -> None:
    while True:
        level, free = memory.pressure_level(), memory.available_gb()
        if (level is not None and level >= memory.WARNING) or (free is not None and free < STOP_GB):
            print(f"stopped: {free:.2f} GB available, pressure {memory.describe(level)}", flush=True)
            os._exit(3)
        time.sleep(0.5)


def only_ezafe_added(original: str, marked: str) -> bool:
    before, after = original.split(), marked.split()
    if len(before) != len(after):
        return False
    return all(b == a or a in (b + KASRA, b + HAMZA_ABOVE, b + "ی") for b, a in zip(before, after, strict=True))


def main() -> int:
    from voiceclone.config import AppConfig
    from voiceclone.text.ezafe_fa import EzafeWriter

    threading.Thread(target=watch, daemon=True).start()
    with gzip.open(SPLITS, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    texts = sorted({r["text"] for r in rows if r["split"] in ("train", "valid")})
    writer = EzafeWriter(AppConfig.load(None).text.ezafe)
    pairs, marked_words, started = [], 0, time.time()
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        cores = [t.strip() for t in batch]
        marked, written, failed = writer.write(cores)
        if failed:
            raise SystemExit(f"the writer did not run: {failed}")
        for original, done in zip(batch, marked, strict=True):
            lead = original[: len(original) - len(original.lstrip())]
            trail = original[len(original.rstrip()):]
            full = lead + done + trail
            if not only_ezafe_added(original, full):
                raise SystemExit(f"more than the ezafe changed:\n{original!r}\n{full!r}")
            pairs.append({"text": original, "ezafe": full})
        marked_words += len(written)
        done_count = start + len(batch)
        rate = done_count / (time.time() - started)
        print(f"  {done_count}/{len(texts)} texts, {marked_words} ezafes written, "
              f"{(len(texts) - done_count) / rate / 60:.0f} min left; {memory.available_gb():.2f} GB available", flush=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    changed = sum(1 for p in pairs if p["ezafe"] != p["text"])
    summary = {"texts": len(pairs), "texts_with_an_ezafe": changed, "ezafes_written": marked_words,
               "clips_covered": sum(1 for r in rows if r["split"] in ("train", "valid")),
               "writer": "voiceclone.text.ezafe_fa.EzafeWriter at " + commit, "minutes": round((time.time() - started) / 60, 1)}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
