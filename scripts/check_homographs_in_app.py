#!/usr/bin/env python3
"""Check that the app now sends OmniVoice the EH text the listener heard in step 3.

Step 3's EH (eval/runs/frontend-listening-v2/texts.json) was written by
scripts/build_frontend_texts_v2.py. The app now writes the same marks itself:
the tagger's ezafe, then Homo-GE2PE's first vowel on HomoRich's homographs -
less به, which every mark broke. Each of the 12 sentences goes through the
app's own text pipeline at its default settings, with the real tagger and the
real Homo-GE2PE, and what OmniVoice would be fed is compared with EH with its
بِه put back to به.

    .venv/bin/python scripts/check_homographs_in_app.py
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

TEXTS = ROOT / "eval" / "runs" / "frontend-listening-v2" / "texts.json"
OUT = ROOT / "eval" / "runs" / "evidence" / "frontend-listening-v2" / "app_vs_heard.json"


def main() -> int:
    config = AppConfig.load(None)
    pipeline = TextPipeline(config.text)
    rows = []
    for s in json.loads(TEXTS.read_text(encoding="utf-8"))["sentences"]:
        start = time.time()
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        fed = " ".join(chunk.synth_text for chunk in prepared.chunks)
        meant = s["texts"]["EH"].replace("بِه ", "به ")
        rows.append({"id": s["id"], "heard_EH": s["texts"]["EH"], "meant": meant, "app": fed, "same": fed == meant,
                     "seconds": round(time.time() - start, 1), "notes": prepared.notes})
    result = {"settings": {"ezafe": vars(config.text.ezafe), "homographs": vars(config.text.homographs),
                           "marks_engines": config.text.marks_engines},
              "same": sum(r["same"] for r in rows), "of": len(rows), "sentences": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(r["id"], f"{r['seconds']:5.1f}s", "same" if r["same"] else f"DIFFERS\n  meant {r['meant']}\n  app   {r['app']}")
    print(f"{result['same']} of {result['of']} as heard (به unmarked) -> {OUT.relative_to(ROOT)}")
    return 0 if result["same"] == result["of"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
