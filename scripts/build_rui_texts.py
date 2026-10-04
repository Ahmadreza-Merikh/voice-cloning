#!/usr/bin/env python3
"""Write each روی-check sentence two ways. It renders nothing.

Registered before it ran (eval/runs/persian-voice-rui-check-v1.json):

* R1  what the app sends OmniVoice today, at its default settings
* R0  R1 with the kasra taken off every روی that carries one

    .venv/bin/python scripts/build_rui_texts.py
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
from voiceclone.text.marks_fa import KASRA, PUNCT  # noqa: E402
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

SENTENCES = ROOT / "eval" / "manifests" / "rui_check_v1.json"
OUT = ROOT / "eval" / "runs" / "rui-check-v1" / "texts.json"
RUI = "روی"


def without_kasra_on_rui(text: str) -> tuple[str, int]:
    parts, taken = text.split(" "), 0
    for i, raw in enumerate(parts):
        core = raw.strip(PUNCT)
        if core == RUI + KASRA:
            parts[i] = raw.replace(core, RUI)
            taken += 1
    return " ".join(parts), taken


def main() -> int:
    pipeline = TextPipeline(AppConfig.load(None).text)
    built = []
    for s in json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]:
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        if len(prepared.chunks) != 1 or any(" not written" in n for n in prepared.notes):
            raise SystemExit(f"{s['id']}: the app did not give one chunk with its marks: {prepared.notes}")
        r1 = prepared.chunks[0].synth_text
        r0, taken = without_kasra_on_rui(r1)
        built.append({"id": s["id"], "category": "rui", "text": s["text"], "rui": s["rui"], "kasra_taken": taken,
                      "texts": {"R1": r1, "R0": r0}, "app_notes": prepared.notes})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    body = {"sentences_file": str(SENTENCES.relative_to(ROOT)),
            "sentences_sha256": hashlib.sha256(SENTENCES.read_bytes()).hexdigest(), "code_commit": commit,
            "conditions": {"R1": "the app today (رویِ)", "R0": "the same without the kasra on روی"},
            "sentences": built}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for s in built:
        print(s["id"], f"kasra taken {s['kasra_taken']} of {s['rui']}")
        for c, t in s["texts"].items():
            print(f"  {c} {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
