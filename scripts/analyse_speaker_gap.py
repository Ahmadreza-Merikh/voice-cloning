#!/usr/bin/env python3
"""Ask why the gap to the human ceiling is three times wider for some voices.

The headline number hides a spread: at 0.546 overall, cv09 lands 0.071 from its
own ceiling while mana is 0.217 away. Averaging that is what makes it look like
one problem.

The hypothesis under test is **mean-voice regression**: that the model pulls
every render toward the centre of the voices it was trained on, so a distinctive
speaker loses more than an average one. It is directly measurable rather than
arguable - embed the renders and the real recordings, and compare how far each
sits from the centroid of the corpus.

If it holds, two things should be true together:

1. renders sit closer to the centroid than the references they were cloned from
2. how distinctive a speaker is predicts how much they lose

Failing (1) refutes it outright. Passing (1) but not (2) would mean the pull is
real but uniform, which is a different problem with a different fix.

    .venv/bin/python scripts/analyse_speaker_gap.py
    .venv/bin/python scripts/analyse_speaker_gap.py --label baseline
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


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, computed here to avoid a scipy dependency.

    Rank rather than Pearson because nine points cannot support an assumption
    about the shape of the relationship, only about its direction.
    """

    def rank(values: list[float]) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    a, b = rank(xs), rank(ys)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", default="finetuned-bestof4")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "eval" / "results")
    parser.add_argument("--renders", type=Path, default=REPO_ROOT / "outputs" / "benchmark")
    args = parser.parse_args()

    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.embedding import CachedEmbedder

    results = json.loads((args.results / f"{args.label}.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))

    embedder = CachedEmbedder()
    _rule(f"Embedding {args.label}")

    def embed_file(path: Path) -> np.ndarray:
        return embedder.embed(load_audio(path), key=str(path))

    # Real recordings, kept apart by role: the reference is what the model was
    # conditioned on, the held-out clips are what it is scored against.
    references: dict[str, np.ndarray] = {}
    held_out: dict[str, list[np.ndarray]] = {}
    for entry in manifest["entries"]:
        path = args.corpus / "audio" / entry["file"]
        if not path.is_file():
            path = args.corpus / entry["file"]
        if not path.is_file():
            print(f"  missing {entry['file']}", file=sys.stderr)
            continue
        vector = embed_file(path)
        if entry["role"] == "reference":
            references[entry["speaker"]] = vector
        else:
            held_out.setdefault(entry["speaker"], []).append(vector)

    renders: dict[str, list[np.ndarray]] = {}
    for row in results["rows"]:
        path = Path(row["output"])
        if not path.is_file():
            path = args.renders / args.label / path.name
        if not path.is_file():
            continue
        renders.setdefault(row["speaker"], []).append(embed_file(path))

    # Printed after the work, not before: the backend name is set on first load,
    # so reading it earlier reports "none" for a model that loads perfectly well.
    print(f"  backend: {embedder.backend}")
    if embedder.backend != "speechbrain-ecapa":
        print("  Not a speaker embedding. Stopping.", file=sys.stderr)
        return 1

    speakers = sorted(set(references) & set(held_out) & set(renders))
    if not speakers:
        print("Nothing to compare.", file=sys.stderr)
        return 1

    # The centroid of the real voices. Built from held-out recordings only, so
    # it does not inherit whatever the reference clips share with the renders.
    identity = {s: unit(np.mean([unit(v) for v in held_out[s]], axis=0)) for s in speakers}
    centroid = unit(np.mean([identity[s] for s in speakers], axis=0))

    _rule("Is the render pulled toward the centre")
    print("  Cosine to the centroid of all nine voices. A render that sits closer")
    print("  to it than its own reference does has been pulled inward.\n")
    print(f"  {'speaker':9s} {'reference':>10s} {'render':>8s} {'pull':>8s} {'gap':>8s}")

    pulls: list[float] = []
    gaps: list[float] = []
    distinctiveness: list[float] = []
    rows: list[tuple[str, float, float, float, float, float]] = []

    per_speaker = results["per_speaker"]
    ceilings = results["calibration"]["per_speaker_ceiling"]

    for speaker in speakers:
        reference_to_centre = cosine(references[speaker], centroid)
        render_to_centre = float(np.mean([cosine(v, centroid) for v in renders[speaker]]))
        pull = render_to_centre - reference_to_centre

        measured = per_speaker[speaker]
        held = measured["held_out"] if isinstance(measured, dict) else float(measured)
        ceiling = float(ceilings[speaker])
        gap = ceiling - held

        # How far this voice sits from the middle of the set.
        apart = 1.0 - cosine(identity[speaker], centroid)

        pulls.append(pull)
        gaps.append(gap)
        distinctiveness.append(apart)
        rows.append((speaker, reference_to_centre, render_to_centre, pull, gap, apart))
        print(
            f"  {speaker:9s} {reference_to_centre:10.3f} {render_to_centre:8.3f} "
            f"{pull:+8.3f} {gap:8.3f}"
        )

    inward = sum(1 for p in pulls if p > 0)
    print(f"\n  pulled toward the centre: {inward} of {len(pulls)}")
    print(f"  mean pull: {np.mean(pulls):+.4f}")

    _rule("Does distinctiveness predict the loss")
    print("  Distance from the centroid against the gap to that speaker's own")
    print("  ceiling. A positive correlation means distinctive voices lose more.\n")
    print(f"  {'speaker':9s} {'apart':>8s} {'gap':>8s}")
    for speaker, _, _, _, gap, apart in sorted(rows, key=lambda r: -r[5]):
        print(f"  {speaker:9s} {apart:8.3f} {gap:8.3f}")

    rho = spearman(distinctiveness, gaps)
    print(f"\n  Spearman rho: {rho:+.3f} over {len(rows)} speakers")
    if abs(rho) < 0.4:
        verdict = "no relationship worth acting on"
    elif rho > 0:
        verdict = "distinctive voices do lose more"
    else:
        verdict = "the relationship runs the other way"
    print(f"  -> {verdict}")

    print(
        "\n  Nine points is few. Treat rho as a direction to test, not a"
        "\n  measurement - the sign is worth more here than the magnitude."
    )

    # If renders move away from the centre of real speech rather than toward it,
    # the deficit may not be identity at all: it may be one shared "this is
    # synthetic" direction that every render carries regardless of speaker.
    _rule("Is there one shared direction every render carries")
    deltas = {s: unit(np.mean([unit(v) for v in renders[s]], axis=0)) - identity[s] for s in speakers}
    matrix = np.stack([unit(deltas[s]) for s in speakers])
    alignments = [
        float(np.dot(matrix[i], matrix[j]))
        for i in range(len(speakers))
        for j in range(i + 1, len(speakers))
    ]
    print("  Cosine between each pair of (render - real) offsets. Near zero means")
    print("  each voice fails in its own way; high means one common artefact.\n")
    print(f"  pairs        {len(alignments)}")
    print(f"  mean cosine  {np.mean(alignments):+.3f}")
    print(f"  range        {min(alignments):+.3f} to {max(alignments):+.3f}")

    # The decisive test, and the one easiest to fake: removing a direction fitted
    # on the same speaker would improve that speaker by construction. Each
    # speaker's direction is therefore fitted on the other eight only.
    _rule("Does removing it recover anything (leave-one-speaker-out)")
    print("  Similarity to that speaker's own held-out recordings, before and")
    print("  after projecting out a direction fitted without them.\n")
    print(f"  {'speaker':9s} {'before':>8s} {'after':>8s} {'change':>8s}")

    changes: list[float] = []
    for speaker in speakers:
        others = [unit(deltas[s]) for s in speakers if s != speaker]
        direction = unit(np.mean(others, axis=0))
        before, after = [], []
        for vector in renders[speaker]:
            v = unit(vector)
            before.append(cosine(v, identity[speaker]))
            after.append(cosine(v - np.dot(v, direction) * direction, identity[speaker]))
        change = float(np.mean(after) - np.mean(before))
        changes.append(change)
        print(
            f"  {speaker:9s} {np.mean(before):8.3f} {np.mean(after):8.3f} {change:+8.3f}"
        )

    print(f"\n  mean change {np.mean(changes):+.4f} over {len(changes)} speakers")
    print(f"  improved    {sum(1 for c in changes if c > 0)} of {len(changes)}")
    if np.mean(changes) > 0.02:
        print(
            "\n  A single direction, fitted without the speaker it is applied to,"
            "\n  recovers part of the gap. That is a systematic artefact rather"
            "\n  than lost identity - and it is a property of the renders, not of"
            "\n  the scoring, so it will not fix itself by measuring differently."
        )
    else:
        print(
            "\n  Removing it recovers little. Whatever the offset is, it does not"
            "\n  reduce to one direction shared across voices."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
