#!/usr/bin/env python3
"""Bring the XTTS check's two forms together and score both with one recogniser on this Mac.

Registered before it ran (eval/runs/persian-voice-xtts-check-v1.json): O was
rendered on Kaggle, X on this Mac, and CER for both comes from the app's own
recogniser here, after both are collected, so one recogniser scores both. O's
Kaggle CER is kept beside it. The wavs go to ``eval/runs/xtts-check-v1/wavs``
and one manifest holds both forms for the similarity script and the page. It
stops if available memory falls under 3.4 GB or pressure reaches warning.

    .venv/bin/python kaggle/fetch_output.py marsmanam/persian-voice-xtts-check-run \\
        remote/xtts-check-v1 xtts-check-v1/ logs/
    ~/.cache/persian-voice-cloning/xtts/.venv/bin/python scripts/render_xtts.py
    .venv/bin/python scripts/collect_xtts_check.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone import memory  # noqa: E402

FETCHED = ROOT / "remote" / "xtts-check-v1" / "xtts-check-v1"
RUN = ROOT / "eval" / "runs" / "xtts-check-v1"
EVIDENCE = ROOT / "eval" / "runs" / "evidence" / "xtts-check-v1"
EXPECTED = 10
STOP_GB = 3.4


def watch() -> None:
    while True:
        level, free = memory.pressure_level(), memory.available_gb()
        if (level is not None and level >= memory.WARNING) or (free is not None and free < STOP_GB):
            print(f"stopped: {free:.2f} GB available, pressure {memory.describe(level)}", flush=True)
            os._exit(3)
        time.sleep(0.3)


def main() -> int:
    from voiceclone.audio.io import load_audio
    from voiceclone.config import AppConfig
    from voiceclone.evaluation.similarity import character_error_rate
    from voiceclone.pipelines.studio import Studio

    texts = json.loads((RUN / "texts.json").read_text(encoding="utf-8"))
    o = json.loads((FETCHED / "A" / "manifest.json").read_text(encoding="utf-8"))
    x = json.loads((RUN / "X" / "manifest.json").read_text(encoding="utf-8"))
    for part, form in ((o, "O"), (x, "X")):
        if part["texts"]["sentences_sha256"] != texts["sentences_sha256"]:
            raise SystemExit(f"{form}: not the registered texts")
        if len([r for r in part["renders"] if r.get("wav")]) != EXPECTED:
            raise SystemExit(f"{form}: {EXPECTED} renders expected")
    (RUN / "wavs").mkdir(exist_ok=True)
    renders = []
    for r in o["renders"]:
        name = Path(r["wav"]).name
        shutil.copy2(FETCHED / "A" / "wavs" / name, RUN / "wavs" / name)
        renders.append({**r, "wav": str((RUN / "wavs" / name).relative_to(ROOT)), "kaggle_wav": r["wav"],
                        "cer_kaggle": r.get("cer")})
    for r in x["renders"]:
        name = Path(r["wav"]).name
        shutil.copy2(ROOT / r["wav"], RUN / "wavs" / name)
        renders.append({**r, "wav": str((RUN / "wavs" / name).relative_to(ROOT))})

    threading.Thread(target=watch, daemon=True).start()
    config = AppConfig.load(None)
    studio = Studio(config)
    plain = {s["id"]: s["text"] for s in texts["sentences"]}
    for r in renders:
        heard = studio.asr.transcribe(load_audio(str(ROOT / r["wav"]))).text.strip()
        r["asr_text"], r["cer"] = heard, character_error_rate(plain[r["id"]], heard)
        print(f"  [{r['condition']}] {r['id']} CER {r['cer'] * 100:.1f}%", flush=True)
    studio.release()

    manifest = {"version": 1, "purpose": "the XTTS check: the app (O, Kaggle) and ParsVoice-XTTS (X, this Mac)",
                "texts": {"file": str((RUN / "texts.json").relative_to(ROOT)),
                          "sentences_sha256": texts["sentences_sha256"]},
                "settings": {"O": o["settings"], "X": x["settings"]}, "reference": o["reference"],
                "conditions": texts["conditions"],
                "sentences": [{"id": s["id"], "category": s["category"], "text": s["text"], "texts": s["texts"]}
                              for s in texts["sentences"]],
                "cer": f"both forms on this Mac by the app's recogniser ({config.engines.asr.backend} "
                       f"{config.engines.asr.model}); O's Kaggle CER kept as cer_kaggle",
                "events": [*({**e, "form": "O"} for e in o["events"]),
                           {"event": "collected", "at": datetime.now(UTC).isoformat(timespec="seconds")}],
                "renders": renders}
    (RUN / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for name in ("gates.json", "runs.json", "result.json", "environment.json"):
        shutil.copy2(FETCHED / name, EVIDENCE / name)
    print(f"{len(renders)} renders -> {(RUN / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
