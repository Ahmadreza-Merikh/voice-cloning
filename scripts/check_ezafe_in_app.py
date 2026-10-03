#!/usr/bin/env python3
"""Check that the app now sends OmniVoice the E text the listener heard.

The listening test's E condition (eval/runs/frontend-listening-v1/texts.json)
was written by scripts/build_frontend_texts.py - the spaCy tagger, KaamelDict
readings for a final h - before the app's normaliser ran on it. The app now
writes the ezafe itself: the tagger without spaCy, the h decided without a
lexicon, and the normaliser keeping أ for OmniVoice. Each of the 12 plain
sentences goes through the app's own text pipeline at its default settings,
and what OmniVoice would be fed is compared with E. One difference is meant:
no ezafe joins a pair of ``ezafe_fa.NO_EZAFE_BEFORE`` (مدیر عامل, heard right
only without it), so a word differing only there counts as intended.

    .venv/bin/python scripts/check_ezafe_in_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.text.ezafe_fa import NO_EZAFE_BEFORE  # noqa: E402
from voiceclone.text.marks_fa import PUNCT, bare  # noqa: E402
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

TEXTS = ROOT / "eval" / "runs" / "frontend-listening-v1" / "texts.json"
OUT = ROOT / "eval" / "runs" / "evidence" / "ezafe-tagger-v1" / "app_vs_heard.json"


def intended(heard: str, app: str) -> list[str] | None:
    """The words differing only by an ezafe a kept-apart pair refuses, or None if any other differs."""
    hw, aw = heard.split(" "), app.split(" ")
    if len(hw) != len(aw):
        return None
    out = []
    for i, (h, a) in enumerate(zip(hw, aw, strict=True)):
        if h == a:
            continue
        following = bare(aw[i + 1].strip(PUNCT)) if i + 1 < len(aw) else ""
        if bare(h) != bare(a) or (bare(a.strip(PUNCT)), following) not in NO_EZAFE_BEFORE:
            return None
        out.append(a.strip(PUNCT))
    return out


def main() -> int:
    config = AppConfig.load(None)
    pipeline = TextPipeline(config.text)
    rows = []
    for s in json.loads(TEXTS.read_text(encoding="utf-8"))["sentences"]:
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        fed = " ".join(chunk.synth_text for chunk in prepared.chunks)
        rows.append({"id": s["id"], "heard_E": s["texts"]["E"], "app": fed, "same": fed == s["texts"]["E"],
                     "differs_only_on_kept_apart_pairs": intended(s["texts"]["E"], fed) if fed != s["texts"]["E"]
                     else None, "notes": prepared.notes})
    as_meant = sum(r["same"] or bool(r["differs_only_on_kept_apart_pairs"]) for r in rows)
    result = {"settings": {"ezafe": vars(config.text.ezafe), "marks_engines": config.text.marks_engines},
              "same": sum(r["same"] for r in rows), "as_meant": as_meant, "of": len(rows), "sentences": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        if r["same"]:
            print(r["id"], "same")
        else:
            print(r["id"], "only", r["differs_only_on_kept_apart_pairs"] or "DIFFERS",
                  f"\n  heard {r['heard_E']}\n  app   {r['app']}")
    print(f"{result['same']} of {result['of']} identical, {as_meant} as meant -> {OUT.relative_to(ROOT)}")
    return 0 if as_meant == result["of"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
