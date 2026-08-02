#!/usr/bin/env python3
"""Score voice cloning honestly, over many speakers, reproducibly.

What it measures
----------------
**Held-out speaker similarity.** Condition the engine on one clip, then score
the render against *other recordings of that speaker that it never saw*. This
is the number that describes voice cloning. Scoring against the conditioning
clip instead - which is what most of this project reported until recently -
also rewards reproducing that clip's room, microphone and noise floor, and runs
about 0.26 high.

**A per-speaker human baseline.** For every speaker the corpus also yields the
score two genuine recordings of that person get against each other. That is the
ceiling: a clone cannot be expected to beat it, and if it does, something is
being measured other than identity.

**A floor.** Pairs of *different* speakers. Without it "0.46" means nothing -
a stranger might score 0.05 or 0.55. With it, every clone score becomes a
position on a measured scale.

**Round-trip WER.** ASR re-transcribes the render and it is scored against the
text that was asked for. Similarity without intelligibility is worthless: a
render can match a voice closely and still be mush.

Design notes
------------
*Everything gets identical audio treatment before embedding* - resample to
16 kHz, peak-normalise - so the clone score and the human baseline are on one
scale. This is the flaw that made the earlier same-reference numbers
uninterpretable, and it is not worth reintroducing for convenience.

*Reference transcripts come from the corpus, not from ASR*, unless
``--ref-transcript asr`` asks otherwise. F5-TTS conditions on a transcript of
the reference clip, so measuring with ASR transcripts confounds the engine with
the recogniser. Having both settings makes that confound measurable instead of
invisible.

*Every run writes JSON* to ``eval/results/<label>.json``, so before/after
comparisons are mechanical rather than a matter of scrolling back.

Usage
-----
    .venv/bin/python scripts/benchmark.py --label baseline
    .venv/bin/python scripts/benchmark.py --label nfe64 --engine-option nfe_step=64
    .venv/bin/python scripts/benchmark.py --label asr-text --ref-transcript asr
    .venv/bin/python scripts/benchmark.py --compare baseline nfe64
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

#: ECAPA's own spread across identity-preserving perturbations of one render,
#: measured by this project. Differences below it are not differences.
NOISE = 0.0205

#: Held-out Persian test sentences. Chosen to exercise what Persian TTS gets
#: wrong - unwritten short vowels, invisible ezafe, homographs, numerals,
#: ZWNJ-joined verb forms - rather than to flatter the engine. None of them
#: appears in any reference clip, so the engine is never asked to repeat text
#: it was just conditioned on.
SENTENCES: list[tuple[str, str]] = [
    ("s1", "دیروز در کتابخانه شهر، کتابی درباره تاریخ ایران خواندم."),
    ("s2", "او گفت که فردا صبح برای دیدن دوستانش به شمال سفر می‌کند."),
    ("s3", "گل زیبای باغ ما امروز شکفت و بهار از راه رسید."),
    ("s4", "قیمت این کتاب ۲۳۰۰۰ تومان است و تا پایان هفته تخفیف دارد."),
    ("s5", "پژوهشگران دانشگاه تهران روش تازه‌ای برای تشخیص این بیماری ارائه کردند."),
]


def _rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}", flush=True)


def _fmt(value: float | None, width: int = 6, digits: int = 3) -> str:
    return "n/a".rjust(width) if value is None else f"{value:.{digits}f}".rjust(width)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _spread(values: list[float]) -> str:
    if not values:
        return "no data"
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1)"
    return (
        f"{statistics.mean(values):.3f} +/- {statistics.stdev(values):.3f} "
        f"[{min(values):.3f}, {max(values):.3f}] n={len(values)}"
    )


# ----------------------------------------------------------------------------
# corpus
# ----------------------------------------------------------------------------
def load_corpus(corpus_dir: Path) -> dict[str, list[dict[str, Any]]]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"No corpus at {corpus_dir}.\n"
            "Build one first:  .venv/bin/python scripts/build_eval_corpus.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["entries"]:
        entry = dict(entry)
        entry["path"] = corpus_dir / entry["file"]
        by_speaker[entry["speaker"]].append(entry)
    return dict(by_speaker)


# ----------------------------------------------------------------------------
# embeddings - one treatment for every piece of audio, real or synthetic
# ----------------------------------------------------------------------------
class Embedder:
    """Cached ECAPA embeddings with a single, uniform preprocessing path.

    The comparison only means something if both sides are treated the same. A
    real clip straight off disk and a render that has been loudness-normalised
    to -18 LUFS are not the same treatment, and the difference lands in the
    score. So everything that reaches the encoder here goes through the same
    two steps and nothing else.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._backend = "unknown"
        self._reliable = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def reliable(self) -> bool:
        return self._reliable

    def _prepare(self, audio: Any) -> Any:
        import numpy as np

        from voiceclone.audio.io import Audio

        mono = audio.to_mono()
        samples = np.asarray(mono.samples, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = samples * (0.95 / peak)
        return Audio(samples, mono.sample_rate, source="benchmark")

    def embed(self, audio: Any, key: str) -> Any:
        import numpy as np

        if key in self._cache:
            return self._cache[key]

        from voiceclone.evaluation.similarity import (
            EMBEDDING_SAMPLE_RATE,
            _embed_speechbrain,
            _load_speechbrain,
        )

        model = _load_speechbrain()
        if model is None:
            raise SystemExit(
                "speechbrain is not installed, so every similarity number would be a\n"
                "spectral fallback rather than a speaker embedding:  pip install speechbrain"
            )
        self._backend, self._reliable = "speechbrain-ecapa", True

        prepared = self._prepare(audio).resample(EMBEDDING_SAMPLE_RATE)
        vector = _embed_speechbrain(model, prepared)
        if vector is None:
            raise RuntimeError(f"Could not embed {key}")
        vector = np.asarray(vector, dtype=np.float64)
        self._cache[key] = vector
        return vector

    def similarity(self, left: Any, left_key: str, right: Any, right_key: str) -> float:
        from voiceclone.evaluation.similarity import cosine_similarity

        return float(cosine_similarity(self.embed(left, left_key), self.embed(right, right_key)))


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------
def calibrate(
    corpus: dict[str, list[dict[str, Any]]], embedder: Embedder
) -> dict[str, Any]:
    """Bound the scale: what the same person scores, and what a stranger scores."""
    from itertools import combinations

    from voiceclone.audio.io import load_audio

    _rule("Calibration - real speech only, no synthesis")

    audio_of: dict[str, Any] = {}
    for entries in corpus.values():
        for entry in entries:
            audio_of[entry["file"]] = load_audio(str(entry["path"]))

    per_speaker: dict[str, list[float]] = {}
    same: list[float] = []
    for speaker, entries in sorted(corpus.items()):
        scores = [
            embedder.similarity(audio_of[a["file"]], a["file"], audio_of[b["file"]], b["file"])
            for a, b in combinations(entries, 2)
        ]
        if scores:
            per_speaker[speaker] = scores
            same.extend(scores)
            print(f"  {speaker:6s} same-speaker  {_spread(scores)}")

    speakers = sorted(corpus)
    cross: list[float] = []
    for left, right in combinations(speakers, 2):
        for a in corpus[left]:
            for b in corpus[right]:
                cross.append(
                    embedder.similarity(
                        audio_of[a["file"]], a["file"], audio_of[b["file"]], b["file"]
                    )
                )

    ceiling = _mean(same)
    floor = _mean(cross)
    print(f"\n  same speaker, different recordings  {_spread(same)}")
    print(f"  different speakers                  {_spread(cross)}")
    if ceiling is not None and floor is not None:
        span = ceiling - floor
        print(
            f"\n  => Scale: {floor:.3f} (stranger) to {ceiling:.3f} (same person), "
            f"{span:.3f} wide, about {span / NOISE:.0f} distinguishable steps."
        )
        if cross:
            print(f"     Worst case: the highest different-speaker pair scored {max(cross):.3f}.")

    return {
        "ceiling": ceiling,
        "floor": floor,
        "cross_max": max(cross) if cross else None,
        "same_pairs": len(same),
        "cross_pairs": len(cross),
        "per_speaker_ceiling": {k: statistics.mean(v) for k, v in per_speaker.items()},
    }


# ----------------------------------------------------------------------------
# the benchmark itself
# ----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    from voiceclone.audio.io import load_audio
    from voiceclone.config import AppConfig
    from voiceclone.errors import VoiceCloneError
    from voiceclone.pipelines import Studio

    corpus = load_corpus(args.corpus_dir)
    if args.speakers:
        corpus = {k: v for k, v in corpus.items() if k in set(args.speakers)}
    if args.limit_speakers:
        corpus = dict(sorted(corpus.items())[: args.limit_speakers])

    embedder = Embedder()
    calibration = calibrate(corpus, embedder)

    sentences = [s for s in SENTENCES if not args.sentences or s[0] in set(args.sentences)]
    sentences = sentences[: args.max_sentences]
    seeds = list(args.seeds)

    config = AppConfig.load()
    if args.device:
        config.runtime.device = args.device
    config.evaluation.enabled = True
    config.evaluation.speaker_similarity = False  # scored here, uniformly
    config.evaluation.transcribe_check = True
    config.evaluation.candidates = args.candidates

    engine_options: dict[str, Any] = {}
    for item in args.engine_option or []:
        key, _, value = item.partition("=")
        try:
            engine_options[key] = int(value)
        except ValueError:
            try:
                engine_options[key] = float(value)
            except ValueError:
                engine_options[key] = value

    _rule(f"Benchmark - engine {args.engine}, label '{args.label}'")
    print(f"Speakers  : {len(corpus)}  ({', '.join(sorted(corpus))})")
    print(f"Sentences : {len(sentences)}   Seeds: {seeds}   Takes: {args.candidates}")
    print(f"Reference transcripts: {args.ref_transcript}")
    print(f"Engine options: {engine_options or 'config defaults'}")
    print(f"Device    : {config.runtime.device}")
    total = len(corpus) * len(sentences) * len(seeds)
    print(f"{total} renders\n")

    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    studio = Studio(config)

    rows: list[dict[str, Any]] = []
    started = time.time()
    done = 0

    for speaker, entries in sorted(corpus.items()):
        reference = next((e for e in entries if e["role"] == "reference"), entries[0])
        held_out = [e for e in entries if e is not reference]
        held_audio = {e["file"]: load_audio(str(e["path"])) for e in held_out}
        reference_audio = load_audio(str(reference["path"]))
        reference_text = None if args.ref_transcript == "asr" else reference["transcript"]

        for key, sentence in sentences:
            for seed in seeds:
                done += 1
                elapsed = time.time() - started
                rate = elapsed / max(done - 1, 1)
                print(
                    f"  [{done:>3}/{total}] {speaker} {key} seed {seed}"
                    f"   (~{rate * (total - done + 1) / 60:.0f} min left)",
                    flush=True,
                )
                try:
                    result = studio.synthesize(
                        sentence,
                        str(reference["path"]),
                        engine=args.engine,
                        reference_text=reference_text,
                        seed=seed,
                        use_phonemes=False,
                        engine_options=engine_options or None,
                        output_path=out_dir / f"{speaker}_{key}_seed{seed}.wav",
                    )
                except (VoiceCloneError, Exception) as exc:  # noqa: BLE001 - report, continue
                    print(f"      failed: {type(exc).__name__}: {exc}")
                    rows.append(
                        {
                            "speaker": speaker,
                            "sentence": key,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                render_key = f"render::{speaker}::{key}::{seed}::{args.label}"
                against = [
                    embedder.similarity(
                        held_audio[e["file"]], e["file"], result.audio, render_key
                    )
                    for e in held_out
                ]
                own = embedder.similarity(
                    reference_audio, reference["file"], result.audio, render_key
                )
                rows.append(
                    {
                        "speaker": speaker,
                        "sentence": key,
                        "seed": seed,
                        "held_out_similarity": round(_mean(against) or 0.0, 4),
                        "held_out_scores": [round(v, 4) for v in against],
                        "reference_similarity": round(own, 4),
                        "word_error_rate": result.scores.word_error_rate,
                        "transcription": result.scores.transcription,
                        "expected": sentence,
                        "output": str(result.output_path),
                        "seconds": round(result.audio.duration, 2),
                    }
                )
                last = rows[-1]
                wer = last["word_error_rate"]
                print(
                    f"      held-out {_fmt(last['held_out_similarity'])}"
                    f"   own-ref {_fmt(last['reference_similarity'])}"
                    f"   WER {'n/a' if wer is None else f'{wer * 100:.1f}%'}",
                    flush=True,
                )

    studio.release()
    return summarise(args, rows, calibration, corpus, sentences, seeds, embedder, started)


# ----------------------------------------------------------------------------
def summarise(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    corpus: dict[str, list[dict[str, Any]]],
    sentences: list[tuple[str, str]],
    seeds: list[int],
    embedder: Embedder,
    started: float,
) -> dict[str, Any]:
    good = [r for r in rows if "error" not in r]
    held = [r["held_out_similarity"] for r in good]
    own = [r["reference_similarity"] for r in good]
    wers = [r["word_error_rate"] for r in good if r["word_error_rate"] is not None]

    ceiling = calibration.get("ceiling")
    floor = calibration.get("floor")
    per_ceiling = calibration.get("per_speaker_ceiling", {})

    # Position on the measured scale, per speaker against that speaker's own
    # ceiling - speakers differ in how distinctive they are, and averaging them
    # against one global ceiling hides that.
    fractions: list[float] = []
    for row in good:
        speaker_ceiling = per_ceiling.get(row["speaker"])
        if speaker_ceiling is None or floor is None or speaker_ceiling <= floor:
            continue
        fractions.append((row["held_out_similarity"] - floor) / (speaker_ceiling - floor))

    _rule(f"Results - {args.label}")
    print(f"  held-out similarity   {_spread(held)}")
    print(f"  reference similarity  {_spread(own)}")
    if held and own:
        print(f"  channel inflation     {statistics.mean(own) - statistics.mean(held):+.3f}")
    if wers:
        print(f"  round-trip WER        {statistics.mean(wers) * 100:.1f}%  (median {statistics.median(wers) * 100:.1f}%)")
    if fractions:
        print(f"  position on scale     {statistics.mean(fractions):.1%} of stranger -> same person")
    if rows and len(good) < len(rows):
        print(f"  failures              {len(rows) - len(good)} of {len(rows)}")

    print(f"\n  {'speaker':8s} {'held-out':>9s} {'ceiling':>8s} {'gap':>7s} {'own ref':>8s} {'WER':>7s}")
    per_speaker: dict[str, dict[str, Any]] = {}
    for speaker in sorted({r["speaker"] for r in good}):
        subset = [r for r in good if r["speaker"] == speaker]
        h = _mean([r["held_out_similarity"] for r in subset])
        o = _mean([r["reference_similarity"] for r in subset])
        w = _mean([r["word_error_rate"] for r in subset if r["word_error_rate"] is not None])
        c = per_ceiling.get(speaker)
        gap = None if (h is None or c is None) else h - c
        print(
            f"  {speaker:8s} {_fmt(h, 9)} {_fmt(c, 8)} {_fmt(gap, 7)} {_fmt(o, 8)} "
            f"{'    n/a' if w is None else f'{w * 100:6.1f}%'}"
        )
        per_speaker[speaker] = {
            "held_out": h,
            "ceiling": c,
            "gap": gap,
            "reference": o,
            "wer": w,
            "renders": len(subset),
        }

    if ceiling is not None and floor is not None and held:
        mean_held = statistics.mean(held)
        cross_max = calibration.get("cross_max")
        print(
            f"\n  Overall {mean_held:.3f} against a {floor:.3f}-{ceiling:.3f} scale."
        )
        if cross_max is not None and mean_held < cross_max:
            print(
                f"  Below the highest different-speaker pair ({cross_max:.3f}): on this "
                "corpus\n  the clone is not reliably distinguishable from a different person."
            )
        elif mean_held < ceiling - NOISE:
            print(f"  Below the human ceiling by {ceiling - mean_held:.3f} ({(ceiling - mean_held) / NOISE:.0f}x the encoder's noise).")
        else:
            print("  At the human ceiling.")

    payload = {
        "label": args.label,
        "engine": args.engine,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_minutes": round((time.time() - started) / 60, 1),
        "config": {
            "ref_transcript": args.ref_transcript,
            "engine_option": args.engine_option or [],
            "candidates": args.candidates,
            "seeds": seeds,
            "sentences": [k for k, _ in sentences],
            "device": args.device or "auto",
            "speakers": sorted(corpus),
            "similarity_backend": embedder.backend,
        },
        "calibration": calibration,
        "summary": {
            "held_out_mean": _mean(held),
            "held_out_sd": statistics.stdev(held) if len(held) > 1 else None,
            "reference_mean": _mean(own),
            "wer_mean": _mean(wers),
            "wer_median": statistics.median(wers) if wers else None,
            "scale_position": _mean(fractions),
            "renders": len(good),
            "failures": len(rows) - len(good),
        },
        "per_speaker": per_speaker,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.label}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {path}")
    return payload


# ----------------------------------------------------------------------------
def compare(labels: list[str]) -> int:
    """Put two or more runs side by side, with the noise floor in view."""
    runs = []
    for label in labels:
        path = RESULTS_DIR / f"{label}.json"
        if not path.is_file():
            print(f"No result named '{label}' in {RESULTS_DIR}", file=sys.stderr)
            return 1
        runs.append(json.loads(path.read_text(encoding="utf-8")))

    _rule("Comparison")
    print(f"  {'label':22s} {'held-out':>9s} {'vs first':>9s} {'own ref':>8s} {'WER':>8s} {'scale':>7s}")
    base = runs[0]["summary"]["held_out_mean"]
    for run_data in runs:
        summary = run_data["summary"]
        delta = None if (base is None or summary["held_out_mean"] is None) else summary["held_out_mean"] - base
        wer = summary["wer_mean"]
        scale = summary["scale_position"]
        print(
            f"  {run_data['label']:22s} {_fmt(summary['held_out_mean'], 9)} "
            f"{'        -' if delta is None else f'{delta:+9.3f}'} "
            f"{_fmt(summary['reference_mean'], 8)} "
            f"{'     n/a' if wer is None else f'{wer * 100:7.1f}%'} "
            f"{'    n/a' if scale is None else f'{scale:6.1%}'}"
        )
    print(
        f"\n  A held-out difference smaller than {NOISE} is inside the encoder's own\n"
        "  noise and is not a difference."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", default="run", help="Name for this run's results file")
    parser.add_argument("--engine", default="f5")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--sentences", nargs="+", default=None, help="Sentence ids, e.g. s1 s3")
    parser.add_argument("--speakers", nargs="+", default=None)
    parser.add_argument("--limit-speakers", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=1, help="Takes per render (best-of-N)")
    parser.add_argument(
        "--ref-transcript",
        choices=["corpus", "asr"],
        default="corpus",
        help="Where the reference transcript comes from (F5 conditions on it)",
    )
    parser.add_argument(
        "--engine-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Forwarded to the engine, e.g. nfe_step=64",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--compare", nargs="+", default=None, metavar="LABEL")
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare)

    if args.calibrate_only:
        corpus = load_corpus(args.corpus_dir)
        if args.limit_speakers:
            corpus = dict(sorted(corpus.items())[: args.limit_speakers])
        calibrate(corpus, Embedder())
        return 0

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
