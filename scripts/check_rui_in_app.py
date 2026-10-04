#!/usr/bin/env python3
"""Check that the app now sends OmniVoice the R0 text of the روی check.

The روی check (eval/runs/persian-voice-rui-check-v1.json) heard R1, the app's
text with the kasra the tagger writes on روی, against R0, the same text
without it, and its rule chose R0. The app now leaves روی plain. Each of the 10
sentences goes through the app's own text pipeline at its default settings,
with the real tagger and the real Homo-GE2PE, and what OmniVoice would be fed
is compared with the R0 that was heard.

    .venv/bin/python scripts/check_rui_in_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

TEXTS = ROOT / "eval" / "runs" / "rui-check-v1" / "texts.json"
OUT = ROOT / "eval" / "runs" / "evidence" / "rui-check-v1" / "app_vs_heard.json"


def main() -> int:
    config = AppConfig.load(None)
    pipeline = TextPipeline(config.text)
    rows = []
    for s in json.loads(TEXTS.read_text(encoding="utf-8"))["sentences"]:
        start = time.time()
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        fed = " ".join(chunk.synth_text for chunk in prepared.chunks)
        rows.append({"id": s["id"], "heard_R0": s["texts"]["R0"], "app": fed, "same": fed == s["texts"]["R0"],
                     "seconds": round(time.time() - start, 1), "notes": prepared.notes})
    result = {"settings": {"ezafe": vars(config.text.ezafe), "homographs": vars(config.text.homographs),
                           "marks_engines": config.text.marks_engines},
              "same": sum(r["same"] for r in rows), "of": len(rows), "sentences": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(r["id"], f"{r['seconds']:5.1f}s", "same" if r["same"] else f"DIFFERS\n  R0  {r['heard_R0']}\n  app {r['app']}")
    print(f"{result['same']} of {result['of']} as heard -> {OUT.relative_to(ROOT)}")
    return 0 if result["same"] == result["of"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
