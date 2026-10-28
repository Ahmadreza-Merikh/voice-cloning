#!/usr/bin/env python3
"""Write the adapted reading check's text once, for all three forms. It renders nothing.

Registered before it ran (eval/runs/persian-voice-adapted-reading-v1.json):
every form reads what the app sends OmniVoice today at its default settings,
and only the weights differ. The Kaggle session renders one form at a time, so
each form gets a texts.json of its own in its own folder, the same text in
each; ``texts.json`` beside them holds all three for the scoring and the page.
The ezafe check (eval/runs/persian-voice-omnivoice-ezafe-train.json) is built
the same way, its sentences keeping their own categories.

    .venv/bin/python scripts/build_adapted_texts.py
    .venv/bin/python scripts/build_adapted_texts.py --run ezafe-check-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

CHECKS = {
    "adapted-reading-v1": {
        "sentences": "adapted_reading_v1.json",
        "forms": {"P": "the published OmniVoice", "A1": "the adapted export at step 1,000",
                  "A5": "the adapted export at step 5,000"}},
    "ezafe-check-v1": {
        "sentences": "ezafe_check_v1.json",
        "forms": {"A1": "the app's weights: the adapted export at step 1,000",
                  "E1": "the ezafe-marked run's export at step 1,000",
                  "E5": "the ezafe-marked run's export at step 5,000"}},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", choices=sorted(CHECKS), default="adapted-reading-v1")
    run = parser.parse_args(argv).run
    sentences_file = ROOT / "eval" / "manifests" / CHECKS[run]["sentences"]
    forms = CHECKS[run]["forms"]
    out = ROOT / "eval" / "runs" / run
    pipeline = TextPipeline(AppConfig.load(None).text)
    built = []
    for s in json.loads(sentences_file.read_text(encoding="utf-8"))["sentences"]:
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        if len(prepared.chunks) != 1 or any(" not written" in n for n in prepared.notes):
            raise SystemExit(f"{s['id']}: the app did not give one chunk with its marks: {prepared.notes}")
        text = prepared.chunks[0].synth_text
        built.append({"id": s["id"], "category": s.get("category", "adapted"), "text": s["text"], "app_text": text,
                      "app_notes": prepared.notes})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    common = {"sentences_file": str(sentences_file.relative_to(ROOT)),
              "sentences_sha256": hashlib.sha256(sentences_file.read_bytes()).hexdigest(), "code_commit": commit}
    every = [{**{k: v for k, v in s.items() if k != "app_text"}, "texts": dict.fromkeys(forms, s["app_text"])}
             for s in built]
    out.mkdir(parents=True, exist_ok=True)
    (out / "texts.json").write_text(json.dumps({**common, "conditions": forms, "sentences": every},
                                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for form, described in forms.items():
        one = [{**{k: v for k, v in s.items() if k != "app_text"}, "texts": {form: s["app_text"]}} for s in built]
        (out / form).mkdir(exist_ok=True)
        (out / form / "texts.json").write_text(json.dumps({**common, "conditions": {form: described}, "sentences": one},
                                                          ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for s in built:
        print(s["id"], s["app_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
