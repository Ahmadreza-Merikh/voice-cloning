#!/usr/bin/env python3
"""Write the app's ezafe into every train and valid transcript of the adaptation. It trains nothing.

Registered before it ran (eval/runs/persian-voice-omnivoice-ezafe-train.json).
Each distinct transcript of adaptation_splits_v1's train and valid splits goes
through the app's own text path for OmniVoice, less the homograph vowels: the
app's normaliser with the options it uses for an engine that reads marks, the
comma after a greeting, then the app's ezafe writer (EzafeWriter: hazm's
ParsBERT tagger, no ezafe on روی, none joining مدیر to عامل). Amended before any
session (the registration's amendment_2026_10_27_normalise): the raw
transcripts split verb prefixes off (می مانند), which the tagger reads as a noun
taking an ezafe, and the app never tags such text. Every word of the result is
checked against the normalised text to be the same word, or it with a kasra, a
hamza above or a ی added at the end of its letters - anything else stops the
run. The map goes to
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
from voiceclone.text.marks_fa import PUNCT  # noqa: E402

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


def parts(word: str) -> tuple[str, str, str]:
    core = word.strip(PUNCT)
    if not core:
        return word, "", ""
    start = word.index(core)
    return word[:start], core, word[start + len(core):]


def only_ezafe_added(normalised: str, marked: str) -> bool:
    before, after = normalised.split(), marked.split()
    if len(before) != len(after):
        return False
    for b, a in zip(before, after, strict=True):
        if b == a:
            continue
        (lead_b, core_b, trail_b), (lead_a, core_a, trail_a) = parts(b), parts(a)
        if (lead_b, trail_b) != (lead_a, trail_a) or core_a not in (core_b + KASRA, core_b + HAMZA_ABOVE, core_b + "ی"):
            return False
    return True


def main() -> int:
    from voiceclone.config import AppConfig
    from voiceclone.text.ezafe_fa import EzafeWriter
    from voiceclone.text.normalize_fa import NormalizationOptions, normalize
    from voiceclone.text.prosody_fa import break_vocative_clauses

    threading.Thread(target=watch, daemon=True).start()
    with gzip.open(SPLITS, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    texts = sorted({r["text"] for r in rows if r["split"] in ("train", "valid")})
    config = AppConfig.load(None).text
    options = NormalizationOptions(expand_numbers=bool(config.expand_numbers),
                                   strip_bracketed_tags=bool(config.strip_bracketed_tags), remove_diacritics=False,
                                   keep_hamza_alef=True, ezafe_heh_as_hamza=True)
    writer = EzafeWriter(config.ezafe)
    pairs, marked_words, normalised_changed, started = [], 0, 0, time.time()
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        normalised = [break_vocative_clauses(normalize(t, options)) or t.strip() for t in batch]
        marked, written, failed = writer.write(normalised)
        if failed:
            raise SystemExit(f"the writer did not run: {failed}")
        for original, plain, done in zip(batch, normalised, marked, strict=True):
            if not only_ezafe_added(plain, done):
                raise SystemExit(f"more than the ezafe changed:\n{plain!r}\n{done!r}")
            normalised_changed += plain != original.strip()
            pairs.append({"text": original, "normalised": plain, "ezafe": done})
        marked_words += len(written)
        done_count = start + len(batch)
        rate = done_count / (time.time() - started)
        print(f"  {done_count}/{len(texts)} texts, {marked_words} ezafes written, "
              f"{(len(texts) - done_count) / rate / 60:.0f} min left; {memory.available_gb():.2f} GB available", flush=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    changed = sum(1 for p in pairs if p["ezafe"] != p["normalised"])
    summary = {"texts": len(pairs), "texts_changed_by_the_normaliser": normalised_changed,
               "texts_with_an_ezafe": changed, "ezafes_written": marked_words,
               "clips_covered": sum(1 for r in rows if r["split"] in ("train", "valid")),
               "writer": "voiceclone.text.ezafe_fa.EzafeWriter at " + commit, "minutes": round((time.time() - started) / 60, 1)}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
