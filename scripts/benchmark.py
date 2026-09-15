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
def Embedder() -> Any:  # noqa: N802 - reads as a constructor at every call site
    """The library's uniformly-preprocessing embedder.

    Kept as a local name so this script reads the same as it did when the class
    lived here; the implementation belongs in the library because the corpus
    builder needs the identical treatment to verify speakers.
    """
    from voiceclone.evaluation import CachedEmbedder

    return CachedEmbedder()


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------
def calibrate(
    corpus: dict[str, list[dict[str, Any]]], embedder: Any
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
# the recogniser's own error floor
# ----------------------------------------------------------------------------
def asr_floor(corpus: dict[str, list[dict[str, Any]]], config: Any) -> dict[str, Any]:
    """Transcribe *real human speech* and score it against its own transcript.

    A render's CER is only interpretable next to this. Whisper does not reach
    zero on Persian - crowdsourced audio, ambiguous spelling, no diacritics -
    so a clone scoring 6% CER against a recogniser that scores 5% on genuine
    speech is not making four times the errors of a human, it is making
    roughly none. Without this number every intelligibility claim in this
    project is uncalibrated.
    """
    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.similarity import character_error_rate, word_error_rate
    from voiceclone.pipelines import Studio

    _rule("Recogniser floor - Whisper against real human speech")
    studio = Studio(config)
    cers: list[float] = []
    wers: list[float] = []
    for speaker, entries in sorted(corpus.items()):
        for entry in entries:
            heard = studio.asr.transcribe(load_audio(str(entry["path"]))).text.strip()
            cer = character_error_rate(entry["transcript"], heard)
            wer = word_error_rate(entry["transcript"], heard)
            if cer is not None:
                cers.append(cer)
            if wer is not None:
                wers.append(wer)
        print(f"  {speaker:6s} done", flush=True)

    print(f"\n  CER on real speech  {_spread(cers)}")
    print(f"  WER on real speech  {_spread(wers)}")
    print(
        "\n  A render cannot be expected to beat these. Anything at or below is\n"
        "  as intelligible as the corpus itself."
    )
    return {
        "cer_mean": _mean(cers),
        "wer_mean": _mean(wers),
        "cer_median": statistics.median(cers) if cers else None,
        "clips": len(cers),
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

    if args.sentence_text:
        # An ad-hoc sentence, for probing something the standard set cannot -
        # e.g. giving an engine text in a language it can pronounce, to
        # separate its speaker transfer from its reading of Persian.
        sentences = [("custom", args.sentence_text)]
    else:
        sentences = [s for s in SENTENCES if not args.sentences or s[0] in set(args.sentences)]
        sentences = sentences[: args.max_sentences]
    seeds = list(args.seeds)

    config = AppConfig.load()
    if args.device:
        config.runtime.device = args.device

    floor_path = RESULTS_DIR / "_asr_floor.json"
    if args.asr_floor or not floor_path.is_file():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        measured = asr_floor(corpus, config)
        floor_path.write_text(json.dumps(measured, indent=2), encoding="utf-8")
    recogniser_floor = json.loads(floor_path.read_text(encoding="utf-8"))
    if args.asr_floor_only:
        return {"asr_floor": recogniser_floor}

    config = AppConfig.load()
    if args.device:
        config.runtime.device = args.device
    config.evaluation.enabled = True
    config.evaluation.speaker_similarity = False  # scored here, uniformly
    config.evaluation.transcribe_check = True
    config.evaluation.candidates = args.candidates

    def parse_options(items: list[str] | None) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for item in items or []:
            key, _, raw = item.partition("=")
            if raw.lower() in {"true", "false"}:
                parsed[key] = raw.lower() == "true"
                continue
            if raw.lower() in {"none", "null"}:
                parsed[key] = None
                continue
            try:
                parsed[key] = int(raw)
            except ValueError:
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = raw
        return parsed

    engine_options = parse_options(args.engine_option)
    preprocess_overrides = parse_options(args.preprocess_option)

    _rule(f"Benchmark - engine {args.engine}, label '{args.label}'")
    print(f"Speakers  : {len(corpus)}  ({', '.join(sorted(corpus))})")
    print(f"Sentences : {len(sentences)}   Seeds: {seeds}   Takes: {args.candidates}")
    print(f"Reference transcripts: {args.ref_transcript}")
    print(f"Engine options: {engine_options or 'config defaults'}")
    print(f"Preprocess overrides: {preprocess_overrides or 'config defaults'}")
    print(f"Device    : {config.runtime.device}")
    total = len(corpus) * len(sentences) * len(seeds)
    print(f"{total} renders\n")

    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.out_dir / (args.replay_from or args.label)
    if args.score_only:
        print(f"Scoring existing renders in {replay_dir} - nothing is synthesised.\n")
    studio = Studio(config)

    rows: list[dict[str, Any]] = []
    started = time.time()
    done = 0

    for speaker, entries in sorted(corpus.items()):
        # Index 0 is the clip the manifest tags as the reference, so the default
        # reproduces earlier runs exactly. Any other index makes a different clip
        # the one the model is conditioned on and scores against the rest, which
        # is what separates a hard voice from an unlucky sample of it.
        ordered = sorted(entries, key=lambda e: e["file"])
        reference = ordered[args.reference_index % len(ordered)]
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
                    if args.score_only:
                        result = _score_existing(
                            studio,
                            replay_dir / f"{speaker}_{key}_seed{seed}.wav",
                            sentence,
                        )
                    else:
                        result = studio.synthesize(
                            sentence,
                            str(reference["path"]),
                            engine=args.engine,
                            reference_text=reference_text,
                            seed=seed,
                            use_phonemes=False,
                            engine_options=engine_options or None,
                            preprocess_overrides=preprocess_overrides or None,
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
                        "character_error_rate": result.scores.character_error_rate,
                        "transcription": result.scores.transcription,
                        "expected": sentence,
                        "output": str(result.output_path),
                        "seconds": round(result.audio.duration, 2),
                    }
                )
                last = rows[-1]
                cer, wer = last["character_error_rate"], last["word_error_rate"]
                print(
                    f"      held-out {_fmt(last['held_out_similarity'])}"
                    f"   own-ref {_fmt(last['reference_similarity'])}"
                    f"   CER {'n/a' if cer is None else f'{cer * 100:.1f}%'}"
                    f"   WER {'n/a' if wer is None else f'{wer * 100:.1f}%'}",
                    flush=True,
                )

    studio.release()
    return summarise(
        args, rows, calibration, corpus, sentences, seeds, embedder, started, recogniser_floor
    )


def _score_existing(studio: Any, path: Path, expected: str) -> Any:
    """Score a render that already exists, loading no synthesis engine.

    The recogniser and the two error-rate functions are the ones Studio itself
    uses when it scores a fresh render, so a replayed run stays comparable to
    the run that produced the audio. Similarity is computed by the caller from
    the same embedder every other row uses.

    This is what makes a score measured on one machine checkable on another:
    re-score the same wavs elsewhere and the difference is the scale, not the
    engine.
    """
    from types import SimpleNamespace

    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.similarity import character_error_rate, word_error_rate

    if not path.is_file():
        raise FileNotFoundError(f"no render at {path}")
    audio = load_audio(str(path))
    heard = studio.asr.transcribe(audio).text.strip()
    return SimpleNamespace(
        audio=audio,
        output_path=path,
        scores=SimpleNamespace(
            transcription=heard,
            character_error_rate=character_error_rate(expected, heard),
            word_error_rate=word_error_rate(expected, heard),
        ),
    )


# ----------------------------------------------------------------------------
def summarise(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    corpus: dict[str, list[dict[str, Any]]],
    sentences: list[tuple[str, str]],
    seeds: list[int],
    embedder: Any,
    started: float,
    recogniser_floor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    good = [r for r in rows if "error" not in r]
    held = [r["held_out_similarity"] for r in good]
    own = [r["reference_similarity"] for r in good]
    wers = [r["word_error_rate"] for r in good if r["word_error_rate"] is not None]
    cers = [r["character_error_rate"] for r in good if r.get("character_error_rate") is not None]

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
    if cers:
        line = f"  round-trip CER        {statistics.mean(cers) * 100:.1f}%  (median {statistics.median(cers) * 100:.1f}%)"
        base = (recogniser_floor or {}).get("cer_mean")
        if base is not None:
            line += f"   vs {base * 100:.1f}% on real speech"
        print(line)
    if wers:
        print(f"  round-trip WER        {statistics.mean(wers) * 100:.1f}%  (spacing-sensitive)")
    if fractions:
        print(f"  position on scale     {statistics.mean(fractions):.1%} of stranger -> same person")
    if rows and len(good) < len(rows):
        print(f"  failures              {len(rows) - len(good)} of {len(rows)}")
    # A render the recogniser never transcribed is not a failure - its
    # similarity is real - but it silently drops out of CER and WER. A MOSS run
    # lost the recogniser on every cell and still reported zero failures, so
    # this is stated plainly rather than left to a missing line.
    untranscribed = sum(1 for r in good if r.get("character_error_rate") is None)
    if good and untranscribed == len(good):
        print(f"  NOT TRANSCRIBED       none of the {len(good)} renders - this run has no CER or WER")
    elif untranscribed:
        print(f"  NOT TRANSCRIBED       {untranscribed} of {len(good)} renders - CER and WER cover only the rest")

    print(f"\n  {'speaker':8s} {'held-out':>9s} {'ceiling':>8s} {'gap':>7s} {'own ref':>8s} {'CER':>7s} {'WER':>7s}")
    per_speaker: dict[str, dict[str, Any]] = {}
    for speaker in sorted({r["speaker"] for r in good}):
        subset = [r for r in good if r["speaker"] == speaker]
        h = _mean([r["held_out_similarity"] for r in subset])
        o = _mean([r["reference_similarity"] for r in subset])
        w = _mean([r["word_error_rate"] for r in subset if r["word_error_rate"] is not None])
        c_err = _mean([r["character_error_rate"] for r in subset if r.get("character_error_rate") is not None])
        c = per_ceiling.get(speaker)
        gap = None if (h is None or c is None) else h - c
        print(
            f"  {speaker:8s} {_fmt(h, 9)} {_fmt(c, 8)} {_fmt(gap, 7)} {_fmt(o, 8)} "
            f"{'    n/a' if c_err is None else f'{c_err * 100:6.1f}%'} "
            f"{'    n/a' if w is None else f'{w * 100:6.1f}%'}"
        )
        per_speaker[speaker] = {
            "held_out": h,
            "ceiling": c,
            "gap": gap,
            "reference": o,
            "wer": w,
            "cer": c_err,
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
            "preprocess_option": args.preprocess_option or [],
            "candidates": args.candidates,
            "reference_index": args.reference_index,
            "score_only": args.score_only,
            "replay_from": args.replay_from,
            "seeds": seeds,
            "sentences": [k for k, _ in sentences],
            "device": args.device or "auto",
            "speakers": sorted(corpus),
            "similarity_backend": embedder.backend,
        },
        "calibration": calibration,
        "recogniser_floor": recogniser_floor,
        "summary": {
            "held_out_mean": _mean(held),
            "held_out_sd": statistics.stdev(held) if len(held) > 1 else None,
            "reference_mean": _mean(own),
            "wer_mean": _mean(wers),
            "cer_mean": _mean(cers),
            "cer_median": statistics.median(cers) if cers else None,
            "scale_position": _mean(fractions),
            "renders": len(good),
            "failures": len(rows) - len(good),
            # Scored for similarity but never transcribed - the NOT TRANSCRIBED
            # line above. Results written before this field existed lack it.
            "untranscribed": untranscribed,
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
def _paired(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Compare two runs cell by cell rather than mean against mean.

    Two runs over the same speakers, sentences and seeds are *paired*, and
    treating them as independent samples throws away most of the power. It also
    misreads what a selector does: best-of-N keeps the baseline take whenever
    the baseline take was already best, so a large share of cells are exact
    ties and the mean difference is diluted by them. What matters is how often
    it changed the pick and how often that change was an improvement.
    """

    def cells(run: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
        return {
            (r["speaker"], r["sentence"], r["seed"]): r
            for r in run["rows"]
            if "error" not in r
        }

    left, right = cells(before), cells(after)
    shared = sorted(set(left) & set(right))
    if len(shared) < 3:
        return

    deltas = [right[k]["held_out_similarity"] - left[k]["held_out_similarity"] for k in shared]
    ties = [d for d in deltas if abs(d) < 1e-9]
    wins = [d for d in deltas if d > 1e-9]
    losses = [d for d in deltas if d < -1e-9]

    print(f"\n  Paired over {len(shared)} cells (same speaker, sentence and seed):")
    mean = statistics.mean(deltas)
    if len(deltas) > 1 and statistics.stdev(deltas) > 0:
        error = statistics.stdev(deltas) / len(deltas) ** 0.5
        print(f"    mean difference  {mean:+.4f}   ({abs(mean) / error:.1f} standard errors from zero)")
    else:
        print(f"    mean difference  {mean:+.4f}")
    print(f"    unchanged        {len(ties):>3d}")
    if wins:
        print(f"    better           {len(wins):>3d}   mean {statistics.mean(wins):+.3f}, best {max(wins):+.3f}")
    if losses:
        print(f"    worse            {len(losses):>3d}   mean {statistics.mean(losses):+.3f}, worst {min(losses):+.3f}")
    changed = len(wins) + len(losses)
    if changed:
        print(f"    -> changed the outcome in {changed} of {len(shared)} cells, improving {len(wins)}")


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

    # A run over four speakers and a run over nine are not comparable: the mean
    # is over different voices, and `scale` is normalised against a floor and
    # ceiling computed from whichever speakers were present. Saying so is
    # cheaper than someone reading a speaker-set difference as an improvement.
    speaker_sets = {tuple(r["config"].get("speakers", [])) for r in runs}
    mixed = len(speaker_sets) > 1
    if mixed:
        print("  WARNING: these runs do not cover the same speakers, so the means are")
        print("  over different voices and the scale positions have different origins.")
        for run_data in runs:
            speakers = run_data["config"].get("speakers", [])
            print(f"    {run_data['label']:22s} {len(speakers)} speakers: {', '.join(speakers)}")
        print()

    print(f"  {'label':22s} {'held-out':>9s} {'vs first':>9s} {'own ref':>8s} {'CER':>8s} {'scale':>7s}")
    base = runs[0]["summary"]["held_out_mean"]
    for run_data in runs:
        summary = run_data["summary"]
        delta = None if (base is None or summary["held_out_mean"] is None) else summary["held_out_mean"] - base
        wer = summary.get("cer_mean")
        scale = summary["scale_position"]
        print(
            f"  {run_data['label']:22s} {_fmt(summary['held_out_mean'], 9)} "
            f"{'        -' if delta is None else f'{delta:+9.3f}'} "
            f"{_fmt(summary['reference_mean'], 8)} "
            f"{'     n/a' if wer is None else f'{wer * 100:7.1f}%'} "
            f"{'    n/a' if scale is None else f'{scale:6.1%}'}"
        )
    if not mixed and len(runs) == 2:
        _paired(runs[0], runs[1])

    print(
        f"\n  A held-out difference smaller than {NOISE} is inside the encoder's own\n"
        "  noise for a *single* render. A paired mean over many cells can resolve\n"
        "  finer than that - see the breakdown above when there is one."
    )
    if mixed:
        print(
            "  The 'vs first' column is not meaningful above: rerun the shorter run\n"
            "  over the same speakers before drawing a conclusion from it."
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
    parser.add_argument(
        "--sentence-text",
        default=None,
        help="Synthesise this text instead of the standard set. CER is then only "
             "meaningful if the engine is meant to read it.",
    )
    parser.add_argument("--speakers", nargs="+", default=None)
    parser.add_argument("--limit-speakers", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=1, help="Takes per render (best-of-N)")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score renders that already exist instead of making new ones. No synthesis "
             "engine is loaded, so a run can be re-scored anywhere - which is how a "
             "score measured on one machine is checked against another.",
    )
    parser.add_argument(
        "--replay-from",
        default=None,
        metavar="LABEL",
        help="With --score-only, read renders from this label's directory while writing "
             "results under --label, so the original run's results are left alone.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Which of a speaker's clips to condition on (0 = the manifest's reference)",
    )
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
    parser.add_argument(
        "--preprocess-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Reference preprocessing override, e.g. denoise=none",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--asr-floor", action="store_true", help="Re-measure the recogniser floor")
    parser.add_argument("--asr-floor-only", action="store_true")
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
