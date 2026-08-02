#!/usr/bin/env python3
"""Calibrate the speaker-similarity metric, then score the clone honestly.

The first run of this script found something that inverts how the project's
headline number should be read:

    two genuine recordings of the same human   ->  0.662  (mean of 6 pairs)
    a clone scored against its own reference   ->  0.82

The clone beats real human speech, which is impossible if the score measures
"is this the same person". It is not measuring only that. A zero-shot clone is
conditioned on one specific clip and reproduces its *channel* - room, mic
colouration, noise floor - alongside the speaker, and ECAPA embeddings are
sensitive to channel as well as identity. So "clone vs its own reference" is an
easier comparison than "recording A vs recording B", and the two numbers are
not on the same scale.

This script measures all three quantities needed to read the metric:

1. **Ceiling** - real speech vs real speech, same speaker, different clips.
   The best a cross-recording comparison can do. Measured both on raw audio and
   through the pipeline's own preprocessing, because the first version of this
   script compared raw clips while `Studio._evaluate` compares the *processed*
   reference, and denoising plus loudness normalisation removes exactly the
   channel differences under discussion.

2. **Floor** - real speech vs a *different* speaker. Without it the ceiling has
   no scale: 0.662 means nothing until you know whether a stranger scores 0.05
   or 0.55. Needs clips from a second speaker (see ``--voices-dir``).

3. **The honest clone score** - synthesise from clip A, then score the output
   against clips B, C, D rather than against A. Now the clone and the human
   baseline are both cross-recording comparisons and can be compared directly.
   The same-reference score is printed alongside it, so the size of the channel
   inflation is visible.

Naming: clips are grouped by filename prefix, ``mana_102-61.wav`` -> ``mana``.
To measure the floor, add clips from another speaker under a different prefix,
e.g. ``ali_1.wav``.

Usage
-----
    .venv/bin/python scripts/measure_ceiling.py --calibrate-only   # 1 + 2, seconds
    .venv/bin/python scripts/measure_ceiling.py                    # everything
    .venv/bin/python scripts/measure_ceiling.py --engines f5 moss --device cpu

Part 3 renders one sentence per engine and is as slow as ordinary synthesis.
Renders go to ``outputs/ceiling/``; nothing is written to the voice library.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: Neutral Persian sentence: unmarked ezafe, a homograph, ordinary vocabulary.
#: Deliberately not a sentence an earlier measurement used, so the engine
#: comparison is not tuned to a string someone already optimised against.
DEFAULT_SENTENCE = "گل زیبای باغ ما امروز شکفت و بهار از راه رسید."

#: ECAPA's own spread across identity-preserving perturbations of one render,
#: measured by this project. Differences below it are not differences.
NOISE = 0.0205


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _speaker_of(path: Path) -> str:
    return path.stem.split("_")[0]


def _fmt(value: float | None, width: int = 6) -> str:
    return "n/a".rjust(width) if value is None else f"{value:.3f}".rjust(width)


def _describe(scores: list[float]) -> str:
    if not scores:
        return "no pairs"
    if len(scores) == 1:
        return f"{scores[0]:.3f} (1 pair)"
    return (
        f"mean {statistics.mean(scores):.3f}  "
        f"min {min(scores):.3f}  max {max(scores):.3f}  "
        f"sd {statistics.stdev(scores):.4f}  ({len(scores)} pairs)"
    )


def _clips(voices_dir: Path) -> list[Path]:
    return sorted(p for p in voices_dir.glob("*.wav") if p.is_file())


def _by_speaker(clips: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for clip in clips:
        grouped.setdefault(_speaker_of(clip), []).append(clip)
    return grouped


def _score(left: Any, right: Any) -> tuple[float, str, bool] | None:
    from voiceclone.evaluation.similarity import speaker_similarity

    result = speaker_similarity(left, right)
    if result is None:
        return None
    return result.score, result.backend, result.reliable


# ----------------------------------------------------------------------------
# parts 1 and 2 - calibrate the scale
# ----------------------------------------------------------------------------
def calibrate(voices_dir: Path, processed: bool) -> dict[str, float]:
    """Score real recordings against each other, within and across speakers."""
    from voiceclone.audio.io import load_audio
    from voiceclone.audio.preprocess import preprocess_reference
    from voiceclone.config import AppConfig
    from voiceclone.evaluation.similarity import available_backends

    if not any(available_backends().values()):
        print(
            "No speaker-verification backend installed, so every number would be a\n"
            "spectral fallback rather than a speaker embedding:  pip install speechbrain",
            file=sys.stderr,
        )
        return {}

    clips = _clips(voices_dir)
    if len(clips) < 2:
        print(f"Need at least two clips in {voices_dir}; found {len(clips)}.", file=sys.stderr)
        return {}

    grouped = _by_speaker(clips)
    label = "through the pipeline's preprocessing" if processed else "raw audio"

    _rule(f"Calibration - real speech vs real speech ({label})")
    print(f"{len(clips)} clips, {len(grouped)} speaker(s): ", end="")
    print(", ".join(f"{name} x{len(paths)}" for name, paths in sorted(grouped.items())))
    print("\nNo synthesis here. These are the bounds the clone score has to be read against.\n")

    config = AppConfig.load()
    cache: dict[Path, Any] = {}
    for clip in clips:
        audio = load_audio(str(clip))
        if processed:
            # Match Studio._evaluate, which scores the clone against the
            # *processed* reference rather than the file on disk.
            audio = preprocess_reference(audio, config.audio).audio
        cache[clip] = audio

    same: list[float] = []
    cross: list[float] = []
    backend = "none"
    warned = False

    for left, right in combinations(clips, 2):
        scored = _score(cache[left], cache[right])
        if scored is None:
            continue
        score, backend, reliable = scored
        if not reliable and not warned:
            warned = True
            print(f"  warning: {backend} is a fallback, not a speaker model.")
        is_same = _speaker_of(left) == _speaker_of(right)
        print(f"  [{'same ' if is_same else 'cross'}] {left.name:22s} vs {right.name:22s}  {score:.3f}")
        (same if is_same else cross).append(score)

    print(f"\nBackend: {backend}")
    out: dict[str, float] = {}
    if same:
        print(f"  same speaker   {_describe(same)}")
        out["ceiling"] = statistics.mean(same)
    if cross:
        print(f"  cross speaker  {_describe(cross)}")
        out["floor"] = statistics.mean(cross)
    else:
        print(
            "  cross speaker  none - only one speaker present.\n"
            "    Until clips from a second speaker are added, the ceiling has no scale.\n"
            "    A stranger scoring 0.55 and a stranger scoring 0.05 imply completely\n"
            "    different readings of every other number in this project."
        )

    if "ceiling" in out and "floor" in out:
        span = out["ceiling"] - out["floor"]
        print(
            f"\n  => Usable range: {out['floor']:.3f} (different person) to "
            f"{out['ceiling']:.3f} (same person), {span:.3f} wide."
        )
        print(
            f"     At ECAPA's own noise of {NOISE}, that holds about "
            f"{span / NOISE:.0f} distinguishable steps."
        )
    elif "ceiling" in out:
        print(f"\n  => Ceiling: {out['ceiling']:.3f}. Floor unknown, so the scale is open-ended.")

    return out


# ----------------------------------------------------------------------------
# part 3 - the honest clone score
# ----------------------------------------------------------------------------
def score_clone(
    voices_dir: Path,
    reference: Path,
    sentence: str,
    engines: list[str],
    device: str | None,
    out_dir: Path,
    bounds: dict[str, float],
) -> None:
    """Render from one clip, score against the *other* clips of that speaker."""
    from voiceclone.audio.io import load_audio
    from voiceclone.config import AppConfig
    from voiceclone.errors import VoiceCloneError
    from voiceclone.pipelines import Studio

    speaker = _speaker_of(reference)
    held_out = [c for c in _clips(voices_dir) if _speaker_of(c) == speaker and c != reference]

    config = AppConfig.load()
    if device:
        config.runtime.device = device
    config.evaluation.enabled = True
    config.evaluation.speaker_similarity = True
    config.evaluation.transcribe_check = True
    # One take each: best-of-N would measure the selector, not the engine.
    config.evaluation.candidates = 1

    _rule("The honest clone score - scored against held-out recordings")
    print(f"Reference : {reference.name}  (what the clone is conditioned on)")
    print(f"Held out  : {', '.join(c.name for c in held_out) or 'none'}")
    print(f"Sentence  : {sentence}")
    print(f"Device    : {config.runtime.device}")
    print("Seed 42, single take, phonemes off - the only variable is the engine.\n")

    if not held_out:
        print(
            f"  Only one clip for '{speaker}', so there is nothing to hold out and this\n"
            "  part cannot run. Add a second recording of the same speaker - a different\n"
            "  session, not a second slice of the same file - and rerun.",
            file=sys.stderr,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    studio = Studio(config)
    held_audio = {c: load_audio(str(c)) for c in held_out}
    rows: list[tuple[str, float | None, list[float], float | None]] = []

    for engine in engines:
        print(f"  rendering {engine} ...", flush=True)
        try:
            result = studio.synthesize(
                sentence,
                str(reference),
                engine=engine,
                seed=42,
                use_phonemes=False,
                output_path=out_dir / f"{engine}.wav",
            )
        except VoiceCloneError as exc:
            print(f"    failed: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"    failed: {type(exc).__name__}: {exc}")
            continue

        own = result.scores.speaker_similarity
        wer = result.scores.word_error_rate
        against: list[float] = []
        for clip, audio in held_audio.items():
            scored = _score(audio, result.audio)
            if scored is not None:
                against.append(scored[0])
                print(f"    vs {clip.name:22s}  {scored[0]:.3f}")
        print(f"    vs its own reference     {_fmt(own)}   (WER {_fmt(wer)})")
        rows.append((engine, own, against, wer))
        studio.release()

    if not rows:
        return

    print(f"\n  {'engine':8s} {'held-out':>9s} {'own ref':>9s} {'inflation':>10s} {'WER':>8s}")
    for engine, own, against, wer in rows:
        mean_out = statistics.mean(against) if against else None
        inflation = None if (mean_out is None or own is None) else own - mean_out
        wer_text = "n/a".rjust(8) if wer is None else f"{wer * 100:.1f}%".rjust(8)
        print(
            f"  {engine:8s} {_fmt(mean_out, 9)} {_fmt(own, 9)} "
            f"{_fmt(inflation, 10)} {wer_text}"
        )

    print(
        "\n  'held-out' is the number that describes voice cloning: the clone measured\n"
        "  against recordings it was never conditioned on. 'own ref' is what the project\n"
        "  reports today. The gap is how much of that figure was channel reproduction\n"
        "  rather than speaker identity."
    )

    ceiling = bounds.get("ceiling")
    floor = bounds.get("floor")
    if ceiling:
        print(f"\n  Against the human baseline of {ceiling:.3f} for the same comparison:")
        for engine, _, against, _ in rows:
            if not against:
                continue
            mean_out = statistics.mean(against)
            verdict = (
                "indistinguishable from a real second recording"
                if abs(mean_out - ceiling) <= NOISE
                else ("above it - suspect a shared channel" if mean_out > ceiling else "below it")
            )
            print(f"    {engine:8s} {mean_out:.3f} vs {ceiling:.3f}  ->  {verdict}")
        if floor:
            print(f"\n  Placed on the {floor:.3f}-{ceiling:.3f} scale:")
            for engine, _, against, _ in rows:
                if not against:
                    continue
                fraction = (statistics.mean(against) - floor) / (ceiling - floor)
                print(f"    {engine:8s} {fraction:6.1%} of the way from stranger to same person")
        else:
            print("\n  No floor measured, so these cannot be turned into a percentage.")


# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the similarity metric, then score the clone against held-out audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--voices-dir", type=Path, default=REPO_ROOT / "voices")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Clip to clone from (default: the first in --voices-dir)",
    )
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--engines", nargs="+", default=["f5", "moss"])
    parser.add_argument("--device", default=None, help="cpu / mps / cuda")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "ceiling")
    parser.add_argument(
        "--calibrate-only", action="store_true", help="Skip synthesis; just bound the scale"
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Skip the preprocessed pass (faster, but not the pipeline's comparison)",
    )
    args = parser.parse_args()

    if not args.voices_dir.is_dir():
        print(f"No such directory: {args.voices_dir}", file=sys.stderr)
        return 1

    calibrate(args.voices_dir, processed=False)
    bounds: dict[str, float] = {}
    if not args.raw_only:
        # The pipeline scores the clone against the processed reference, so this
        # is the pass whose ceiling is comparable to the clone's number.
        bounds = calibrate(args.voices_dir, processed=True)
        if bounds:
            print(
                "\nThe preprocessed pass is the comparable one: Studio scores the clone\n"
                "against the processed reference, so its ceiling belongs on the same scale."
            )

    if args.calibrate_only:
        return 0

    reference = args.reference
    if reference is None:
        clips = _clips(args.voices_dir)
        if not clips:
            print(f"No .wav files in {args.voices_dir}.", file=sys.stderr)
            return 1
        reference = clips[0]
    if not reference.is_file():
        print(f"No such file: {reference}", file=sys.stderr)
        return 1

    score_clone(
        args.voices_dir,
        reference,
        args.sentence,
        args.engines,
        args.device,
        args.out_dir,
        bounds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
