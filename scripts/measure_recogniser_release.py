#!/usr/bin/env python3
"""Measure whether unloading the recogniser gives the memory back to the system.

The memory guard makes a 16 GB Mac choose: either the recogniser is loaded or a
guarded engine is, never both (`src/voiceclone/pipelines/studio.py`). That is
only the right rule if the two really must overlap. If `release_recogniser()`
returns what faster-whisper took, the reference can be transcribed *first* and
the recogniser dropped before the engine loads, and nobody has to type the
transcript by hand.

This loads the recogniser, transcribes a clip, releases it, and reads the
machine's available memory and this process's resident size at every step -
twice, with a different clip each time, because the transcript cache would
otherwise skip the second load. It synthesises nothing and loads no TTS engine.

    .venv/bin/python scripts/measure_recogniser_release.py \
        --clip eval/corpus/cv02_000.wav --clip eval/corpus/mana_000.wav \
        --report eval/runs/evidence/recogniser-release/measurement.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from voiceclone import memory  # noqa: E402
from voiceclone.audio.io import load_audio  # noqa: E402
from voiceclone.pipelines.studio import Studio  # noqa: E402


def resident_gb() -> float | None:
    """This process's resident size, the part of the question the OS answers."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**30


def settle() -> None:
    """Give the allocators their chance before reading, so a reading is not just lag."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    time.sleep(1.0)


def sample(stage: str) -> dict:
    settle()
    available, resident = memory.available_gb(), resident_gb()
    reading = {
        "stage": stage,
        "available_gb": None if available is None else round(available, 2),
        "resident_gb": None if resident is None else round(resident, 2),
        "pressure": memory.describe(memory.pressure_level()),
    }
    print(json.dumps(reading), flush=True)
    return reading


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", required=True, help="Reference clip; pass twice")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    readings = [sample("start")]
    studio = Studio()
    readings.append(sample("studio built"))

    rounds = []
    for index, clip in enumerate(args.clip, start=1):
        audio = load_audio(Path(clip))
        started = time.time()
        transcript = studio.transcribe_reference(audio)
        seconds = round(time.time() - started, 1)
        loaded = sample(f"round {index}: recogniser loaded and {Path(clip).name} transcribed")
        studio.release_recogniser()
        released = sample(f"round {index}: recogniser released")
        rounds.append({
            "clip": clip,
            "seconds": seconds,
            "backend": studio.config.engines.asr.backend,
            "model": studio.config.engines.asr.model,
            "characters": len(transcript),
            "resident_taken_gb": None if loaded["resident_gb"] is None or readings[-1]["resident_gb"] is None
            else round(loaded["resident_gb"] - readings[-1]["resident_gb"], 2),
            "resident_returned_gb": None if loaded["resident_gb"] is None or released["resident_gb"] is None
            else round(loaded["resident_gb"] - released["resident_gb"], 2),
        })
        readings += [loaded, released]
        if memory.pressure_level() == memory.WARNING:
            print("memory pressure reached warning: stopping", flush=True)
            break

    report = {
        "measured": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "machine": {"ram_gb": 16, "readings_are": "system available memory and this process's resident size"},
        "readings": readings,
        "rounds": rounds,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
