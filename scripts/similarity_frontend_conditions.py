#!/usr/bin/env python3
"""Did writing the marks in change how much the voice resembles the speaker?

Exploratory, asked by the user after hearing the ezafe in the app (2026-10-03):
"the cloning accuracy dropped". The step 2 renders answer it for the marks
alone - 12 sentences in four forms, one voice (mana_000.wav), seed 42, one
GPU session, so text is the only difference.

Fixed before any score was computed: the project's measure - held-out speaker
similarity, the mean speechbrain ECAPA cosine between a render and the
speaker's four other clips (mana_001-004), every clip through the same
CachedEmbedder - per render; per form against A, the mean paired difference
over the 12 sentences with a 95% bootstrap interval (10,000 resamples of
sentences, seed 20261003). An interval wholly below zero means the marks lower
similarity on this voice. One voice and 12 sentences: a diagnostic, not a
claim about voices in general.

    .venv/bin/python scripts/similarity_frontend_conditions.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "eval" / "runs" / "frontend-listening-v1" / "manifest.json"
CORPUS = ROOT / "eval" / "corpus"
OUT = ROOT / "eval" / "runs" / "evidence" / "frontend-listening-v1" / "similarity.json"
CONDITIONS = ("A", "E", "EN", "ENV")
SEED = 20261003
RESAMPLES = 10_000


def interval(diffs: list[float]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES))
    return sum(diffs) / n, means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def main() -> int:
    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.embedding import CachedEmbedder

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corpus = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    mana = [e for e in corpus["entries"] if e["speaker"] == "mana"]
    reference = next(e["file"] for e in mana if e["role"] == "reference")
    held_out = sorted(e["file"] for e in mana if e["role"] == "held_out")
    embedder = CachedEmbedder()
    clips = {name: load_audio(str(CORPUS / name)) for name in [reference, *held_out]}

    scores: dict[str, dict[str, dict[str, float]]] = {}
    for entry in manifest["renders"]:
        audio = load_audio(str(ROOT / entry["wav"]))
        key = f"{entry['id']}_{entry['condition']}"
        against = [embedder.similarity(audio, key, clips[h], h) for h in held_out]
        scores.setdefault(entry["id"], {})[entry["condition"]] = {
            "held_out": round(sum(against) / len(against), 4),
            "prompt": round(embedder.similarity(audio, key, clips[reference], reference), 4)}

    sentences = sorted(scores)
    summary = {}
    for c in CONDITIONS:
        held = [scores[s][c]["held_out"] for s in sentences]
        row = {"held_out_mean": round(sum(held) / len(held), 4)}
        if c != "A":
            mean, low, high = interval([scores[s][c]["held_out"] - scores[s]["A"]["held_out"] for s in sentences])
            row["minus_A"] = [round(mean, 4), round(low, 4), round(high, 4)]
        summary[c] = row
    floor = [embedder.similarity(clips[a], a, clips[b], b) for i, a in enumerate(held_out) for b in held_out[i + 1:]]
    result = {"measure": "held-out ECAPA similarity, speechbrain via voiceclone.evaluation.embedding.CachedEmbedder",
              "backend": embedder.backend, "reference": reference, "held_out": held_out,
              "speaker_own_clips_mean": round(sum(floor) / len(floor), 4),
              "summary": summary, "per_sentence": scores}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("backend", "speaker_own_clips_mean", "summary")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
