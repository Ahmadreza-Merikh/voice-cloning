#!/usr/bin/env python3
"""Look for something that says in advance which reference clip will clone well.

Reference choice is worth more than anything else measured on this corpus - an
oracle picking each speaker's best clip gains +0.028, as much as the finetune
and best-of-4 together. The oracle is not available in production. This asks
whether any property of a clip predicts the answer.

Correlations are computed **within speaker** and then averaged. Between speakers
the voices differ by as much as the references do, so a pooled correlation would
mostly measure which speaker a clip belongs to. The question is narrower and more
useful: given this person's recordings, which one should be the reference?

    .venv/bin/python scripts/predict_reference_quality.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

LABELS = {
    0: "finetuned",
    1: "finetuned-ref1",
    2: "finetuned-ref2",
    3: "finetuned-ref3",
    4: "finetuned-ref4",
}


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    a, b = rank(np.asarray(xs, float)), rank(np.asarray(ys, float))
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def features(path: Path, transcript: str, embedder, others: list[np.ndarray]) -> dict[str, float]:
    """Everything measurable about a clip before anything is rendered from it."""
    import librosa

    from voiceclone.audio.analysis import analyze
    from voiceclone.audio.io import load_audio
    from voiceclone.audio.vad import detect_speech, speech_duration

    audio = load_audio(path)
    samples = np.asarray(audio.to_mono().samples, dtype=np.float32)
    rate = audio.sample_rate

    report = analyze(audio)
    speech = speech_duration(detect_speech(audio))

    f0, voiced, _ = librosa.pyin(
        samples, fmin=60, fmax=400, sr=rate, frame_length=2048
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    # How typical this clip is of the *rest* of the speaker's recordings. The
    # most motivated candidate: an outlying take should be a poor description of
    # the voice. Computed against the other clips only, never itself.
    vector = embedder.embed(audio, key=str(path))
    unit = vector / (np.linalg.norm(vector) or 1.0)
    centre = np.mean([o / (np.linalg.norm(o) or 1.0) for o in others], axis=0)
    centre = centre / (np.linalg.norm(centre) or 1.0)

    return {
        "duration": float(len(samples) / rate),
        "speech_seconds": float(speech),
        "speech_ratio": float(speech / (len(samples) / rate)) if len(samples) else 0.0,
        "snr_db": float(report.estimated_snr_db),
        "quality_score": float(report.score),
        "bandwidth_hz": float(report.effective_bandwidth_hz),
        "longest_silence": float(report.longest_silence),
        "chars_per_second": len(transcript) / (speech or 1.0),
        "f0_median": float(np.median(voiced_f0)) if voiced_f0.size else 0.0,
        "f0_spread": float(np.std(voiced_f0)) if voiced_f0.size else 0.0,
        "voiced_fraction": float(voiced_f0.size / len(f0)) if f0 is not None and len(f0) else 0.0,
        "rms_dbfs": float(report.rms_dbfs),
        "typicality": float(np.dot(unit, centre)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "eval" / "results")
    args = parser.parse_args()

    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.embedding import CachedEmbedder

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    clips: dict[str, list[dict]] = {}
    for entry in manifest["entries"]:
        clips.setdefault(entry["speaker"], []).append(entry)
    for group in clips.values():
        group.sort(key=lambda e: e["file"])
    speakers = sorted(clips)

    # Measured outcome: mean held-out similarity when that clip was the reference.
    measured: dict[tuple[str, int], float] = {}
    for index, label in LABELS.items():
        path = args.results / f"{label}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        per: dict[str, list[float]] = {}
        for row in data["rows"]:
            value = row.get("held_out_similarity")
            if value is not None:
                per.setdefault(row["speaker"], []).append(float(value))
        for speaker, values in per.items():
            if len(values) == 3:
                measured[(speaker, index)] = float(np.mean(values))

    embedder = CachedEmbedder()
    _rule("Measuring the clips")
    table: dict[tuple[str, int], dict[str, float]] = {}
    for speaker in speakers:
        vectors = [
            embedder.embed(load_audio(args.corpus / e["file"]), key=str(args.corpus / e["file"]))
            for e in clips[speaker]
        ]
        for index, entry in enumerate(clips[speaker]):
            if (speaker, index) not in measured:
                continue
            others = [v for j, v in enumerate(vectors) if j != index]
            table[(speaker, index)] = features(
                args.corpus / entry["file"], entry["transcript"], embedder, others
            )
        print(f"  {speaker}: {sum(1 for i in range(5) if (speaker, i) in table)} usable clips")

    names = sorted(next(iter(table.values())))
    _rule("Within-speaker rank correlation with the measured outcome")
    print("  Averaged over speakers with at least three usable clips. Positive")
    print("  means more of it clones better.\n")
    print(f"  {'feature':18s} {'mean rho':>9s} {'speakers':>9s} {'agree':>7s}")

    summary = []
    for name in names:
        rhos = []
        for speaker in speakers:
            indices = [i for i in range(5) if (speaker, i) in table]
            if len(indices) < 3:
                continue
            xs = [table[(speaker, i)][name] for i in indices]
            ys = [measured[(speaker, i)] for i in indices]
            rhos.append(spearman(xs, ys))
        if not rhos:
            continue
        mean = float(np.mean(rhos))
        agree = sum(1 for r in rhos if (r > 0) == (mean > 0))
        summary.append((abs(mean), name, mean, len(rhos), agree))

    for _, name, mean, count, agree in sorted(summary, reverse=True):
        print(f"  {name:18s} {mean:+9.3f} {count:9d} {agree:4d}/{count}")

    # A correlation is not a rule you can ship. This is: for each speaker, pick
    # the clip the feature ranks highest, and see what that actually scores
    # against the default clip and against the oracle that already knows.
    _rule("What a rule based on each feature would actually buy")
    print("  Per speaker: pick the top-ranked clip, compare to the manifest's")
    print("  default (index 0) and to picking the true best.\n")
    print(f"  {'rule':18s} {'vs default':>11s} {'of oracle':>10s} {'picks best':>11s}")

    usable = [s for s in speakers if sum(1 for i in range(5) if (s, i) in table) >= 3]
    default = {s: measured[(s, 0)] for s in usable if (s, 0) in measured}
    oracle = {s: max(measured[(s, i)] for i in range(5) if (s, i) in table) for s in usable}
    headroom = float(np.mean([oracle[s] - default[s] for s in default]))

    ranked = []
    for name in names:
        gains, exact = [], 0
        for speaker in default:
            indices = [i for i in range(5) if (speaker, i) in table]
            choice = max(indices, key=lambda i: table[(speaker, i)][name])
            gains.append(measured[(speaker, choice)] - default[speaker])
            exact += measured[(speaker, choice)] == oracle[speaker]
        gain = float(np.mean(gains))
        ranked.append((gain, name, exact, len(gains)))

    for gain, name, exact, count in sorted(ranked, reverse=True)[:6]:
        share = gain / headroom * 100 if headroom else 0.0
        print(f"  {name:18s} {gain:+11.4f} {share:9.0f}% {exact:6d}/{count}")

    print(f"\n  oracle headroom over the default: {headroom:+.4f}")

    _rule("Reading it")
    best = max(summary)
    top_gain, top_rule, _, _ = max(ranked)
    print(f"  Strongest correlation: {best[1]} at rho {best[2]:+.3f}, consistent")
    print(f"  in {best[4]} of {best[3]} speakers.")
    print(f"  Best realised gain:    {top_rule} at {top_gain:+.4f}.")

    # The verdict keys off the gain, not the correlation. Ranking well and
    # picking the maximum are different problems: rho near 0.6 over four clips
    # still puts the wrong one first most of the time, and only the top pick
    # is what a rule actually ships.
    if top_gain < 0.0205:
        print("\n  Below the encoder's own noise. The headroom is real but none of")
        print("  these properties reaches it - a correlation strong enough to rank")
        print("  is still too weak to reliably pick the single best clip.")
        print("  Selecting a reference by measurement means rendering and")
        print("  comparing, the same trade best-of-N makes on the seed axis.")
    else:
        print("\n  Worth acting on, with the usual caution that this is a handful")
        print("  of clips per speaker and the correlation is a direction, not a rule.")
    print("\n  Caveat on typicality: it is measured against the same recordings the")
    print("  render is later scored against, so part of any signal there reflects")
    print("  the evaluation setup rather than a property of the clip alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
