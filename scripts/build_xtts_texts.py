#!/usr/bin/env python3
"""Write the XTTS check's two texts once. It renders nothing.

Registered before it ran (eval/runs/persian-voice-xtts-check-v1.json):

* O  what the app sends OmniVoice today, at its default settings
* X  the app's normalised text with no marks - numbers already words - which
     ParsVoice-XTTS reads; its own punctuation mapping is applied when it
     renders, as its notebook does

O renders on Kaggle, so it gets a texts.json of its own in ``A/``;
``texts.json`` beside it holds both for the scoring and the page.

    .venv/bin/python scripts/build_xtts_texts.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.text.marks_fa import HARAKAT  # noqa: E402
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

SENTENCES = ROOT / "eval" / "manifests" / "xtts_news_v1.json"
OUT = ROOT / "eval" / "runs" / "xtts-check-v1"
DESCRIBED = {"O": "the app today: OmniVoice with A1 reading the app's text",
             "X": "ParsVoice-XTTS reading the app's normalised text with no marks"}


def main() -> int:
    config = AppConfig.load(None)
    pipeline = TextPipeline(config.text)
    plain_pipeline = TextPipeline(config.copy_with(**{"text.ezafe.enabled": False,
                                                      "text.homographs.enabled": False}).text)
    built = []
    for s in json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]:
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        if len(prepared.chunks) != 1 or any(" not written" in n for n in prepared.notes):
            raise SystemExit(f"{s['id']}: the app did not give one chunk with its marks: {prepared.notes}")
        plain = plain_pipeline.prepare(s["text"], engine_name="omnivoice").chunks
        if len(plain) != 1 or any(c in HARAKAT for c in plain[0].synth_text):
            raise SystemExit(f"{s['id']}: the plain text is not one unmarked chunk")
        built.append({"id": s["id"], "category": "news", "text": s["text"],
                      "texts": {"O": prepared.chunks[0].synth_text, "X": plain[0].synth_text},
                      "app_notes": prepared.notes})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    common = {"sentences_file": str(SENTENCES.relative_to(ROOT)),
              "sentences_sha256": hashlib.sha256(SENTENCES.read_bytes()).hexdigest(), "code_commit": commit}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "texts.json").write_text(json.dumps({**common, "conditions": DESCRIBED, "sentences": built},
                                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT / "A").mkdir(exist_ok=True)
    only_o = [{**s, "texts": {"O": s["texts"]["O"]}} for s in built]
    (OUT / "A" / "texts.json").write_text(json.dumps({**common, "conditions": {"O": DESCRIBED["O"]}, "sentences": only_o},
                                                     ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for s in built:
        print(s["id"], "\n  O", s["texts"]["O"], "\n  X", s["texts"]["X"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
