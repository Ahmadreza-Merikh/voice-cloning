#!/usr/bin/env python3
"""Write the news vowels check's two texts once. It renders nothing.

Registered before it ran (eval/runs/persian-voice-news-vowels-v1.json):

* B  what the app sends OmniVoice today, at its default settings
* V  B with the first short vowel of every word of three or more letters
     written in, as Homo-GE2PE reads the sentence - through the app's own
     homograph writer, given every such word instead of the homograph list -
     except به, اپل and کند, whose marks are known to hurt

Each set of weights renders in its own run, so each gets a texts.json of its
own in its own folder (P: PB and PV, A: AB and AV); ``texts.json`` beside them
holds all four for the scoring and the page. The marks V writes are checked
against the sentences' hand-written first vowels.

    .venv/bin/python scripts/build_news_texts.py
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
from voiceclone.text.homographs_fa import write_first_vowels  # noqa: E402
from voiceclone.text.marks_fa import HARAKAT, PUNCT, bare  # noqa: E402
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

SENTENCES = ROOT / "eval" / "manifests" / "news_vowels_v1.json"
OUT = ROOT / "eval" / "runs" / "news-vowels-v1"
NEVER_MARKED = frozenset({"به", "اپل", "کند"})
SHORT = {"َ": "a", "ِ": "e", "ُ": "o"}
MODELS = {"P": {"PB": "B", "PV": "V"}, "A": {"AB": "B", "AV": "V"}}
DESCRIBED = {"PB": "the published weights reading the app's text", "PV": "the published weights, every first vowel",
             "AB": "the adapted weights reading the app's text", "AV": "the adapted weights, every first vowel"}


def first_mark(word: str) -> str | None:
    return SHORT.get(word[1]) if len(word) > 1 and word[1] in HARAKAT else None


def main() -> int:
    config = AppConfig.load(None)
    pipeline = TextPipeline(config.text)
    plain_pipeline = TextPipeline(config.copy_with(**{"text.ezafe.enabled": False,
                                                      "text.homographs.enabled": False}).text)
    built, checks = [], []
    for s in json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]:
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        if len(prepared.chunks) != 1 or any(" not written" in n for n in prepared.notes):
            raise SystemExit(f"{s['id']}: the app did not give one chunk with its marks: {prepared.notes}")
        b = prepared.chunks[0].synth_text
        plain = plain_pipeline.prepare(s["text"], engine_name="omnivoice").chunks[0].synth_text
        words = {bare(w.strip(PUNCT)) for w in plain.split() if len(bare(w.strip(PUNCT))) >= 3} - NEVER_MARKED
        compact = list(pipeline.g2p.phonemize(plain).compact_words)
        v, written = write_first_vowels(plain, b, compact, homographs=frozenset(words))
        marked = {bare(w.strip(PUNCT)): first_mark(w.strip(PUNCT)) for w in v.split()}
        for word, vowel in s["first_vowels"].items():
            got = marked.get(bare(word))
            checks.append({"sentence": s["id"], "word": word, "gold": vowel, "marked": got,
                           "verdict": "right" if got == vowel else "missing" if got is None else "wrong"})
        built.append({"id": s["id"], "category": "news", "text": s["text"], "B": b, "V": v, "marked_in_V": written,
                      "app_notes": prepared.notes})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    common = {"sentences_file": str(SENTENCES.relative_to(ROOT)),
              "sentences_sha256": hashlib.sha256(SENTENCES.read_bytes()).hexdigest(), "code_commit": commit}
    keep = ("id", "category", "text", "marked_in_V", "app_notes")
    every = [{**{k: s[k] for k in keep}, "texts": {f: s[t] for forms in MODELS.values() for f, t in forms.items()}}
             for s in built]
    OUT.mkdir(parents=True, exist_ok=True)
    tally = {k: sum(1 for c in checks if c["verdict"] == k) for k in ("right", "wrong", "missing")}
    (OUT / "texts.json").write_text(json.dumps({**common, "conditions": DESCRIBED, "sentences": every,
                                                "first_vowels_against_gold": {"tally": tally, "words": checks}},
                                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for model, forms in MODELS.items():
        one = [{**{k: s[k] for k in keep}, "texts": {f: s[t] for f, t in forms.items()}} for s in built]
        (OUT / model).mkdir(exist_ok=True)
        (OUT / model / "texts.json").write_text(
            json.dumps({**common, "conditions": {f: DESCRIBED[f] for f in forms}, "sentences": one},
                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for s in built:
        print(s["id"], "\n  B", s["B"], "\n  V", s["V"])
    print("V's first vowels against the gold:", tally)
    for c in checks:
        if c["verdict"] != "right":
            print("  ", c["sentence"], c["word"], "gold", c["gold"], "V", c["marked"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
