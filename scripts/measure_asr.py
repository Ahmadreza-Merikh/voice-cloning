#!/usr/bin/env python3
"""Score a recogniser on this project's human-speech corpus.

The gate for changing the Studio default: mean CER on ``eval/corpus`` must
beat the 7.8% FINDINGS.md measured with large-v3-turbo. Decode flags default
to the product Studio settings (no second VAD, Persian prompt). Pass
``--scoring`` to replay the historical block instead.

    .venv/bin/python scripts/measure_asr.py
    .venv/bin/python scripts/measure_asr.py --model nezamisafa/whisper-persian-v4
    .venv/bin/python scripts/measure_asr.py --scoring
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CORPUS = REPO_ROOT / "eval" / "corpus"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score ASR on eval/corpus")
    parser.add_argument("--model", default=None, help="Override engines.asr.model")
    parser.add_argument("--scoring", action="store_true", help="Pin the FINDINGS.md recogniser")
    parser.add_argument("--limit", type=int, default=0, help="First N clips only (0 = all)")
    parser.add_argument("--report", type=Path, default=None, help="Write JSON here")
    args = parser.parse_args(argv)

    from voiceclone.audio.io import load_audio
    from voiceclone.config import AppConfig
    from voiceclone.device import get_device_spec
    from voiceclone.engines.asr import ASREngine, apply_scoring_asr
    from voiceclone.evaluation.similarity import character_error_rate, word_error_rate

    config = AppConfig.load()
    if args.scoring:
        config = apply_scoring_asr(config)
    if args.model:
        config = config.copy_with(**{"engines.asr.model": args.model})

    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e.get("transcript")]
    if args.limit:
        entries = entries[: args.limit]

    engine = ASREngine(config, get_device_spec())
    rows: list[dict[str, Any]] = []
    started = time.time()
    for entry in entries:
        audio = load_audio(CORPUS / entry["file"], mono=True)
        heard = engine.transcribe(audio).text.strip()
        cer = character_error_rate(entry["transcript"], heard)
        wer = word_error_rate(entry["transcript"], heard)
        ref_words = len(entry["transcript"].split())
        hyp_words = len(heard.split())
        rows.append(
            {
                "file": entry["file"],
                "seconds": round(audio.duration, 1),
                "cer": None if cer is None else round(cer, 4),
                "wer": None if wer is None else round(wer, 4),
                "reference_words": ref_words,
                "heard_words": hyp_words,
                "words_dropped": max(0, ref_words - hyp_words),
                "heard": heard,
            }
        )
        print(
            f"  {entry['file']:16s} CER {0 if cer is None else cer * 100:5.1f}%  "
            f"{ref_words}->{hyp_words}  {heard[:60]}",
            flush=True,
        )
    engine.unload()

    good = [r for r in rows if r["cer"] is not None]
    cers = [r["cer"] for r in good]
    wers = [r["wer"] for r in good if r["wer"] is not None]
    summary = {
        "model": config.engines.asr.model,
        "vad_filter": config.engines.asr.vad_filter,
        "no_speech_threshold": config.engines.asr.no_speech_threshold,
        "initial_prompt": config.engines.asr.initial_prompt,
        "backend": engine.backend,
        "clips": len(good),
        "seconds": round(time.time() - started, 1),
        "cer_mean": None if not cers else round(statistics.mean(cers), 4),
        "wer_mean": None if not wers else round(statistics.mean(wers), 4),
        "words_dropped": sum(r["words_dropped"] for r in good),
        "rows": rows,
    }
    print(
        f"\n{summary['model']}: CER {summary['cer_mean']} WER {summary['wer_mean']} "
        f"words short {summary['words_dropped']} in {summary['seconds']}s "
        f"(vad_filter={summary['vad_filter']})",
        flush=True,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
