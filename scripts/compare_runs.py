#!/usr/bin/env python3
"""Compare benchmark runs honestly: paired, clustered by speaker, with intervals.

Every engine is scored on the same 27 cells - 9 speakers x 3 sentences, one seed
- against the same reference clips, so runs are compared cell by cell rather
than mean against mean. That much this project already did. What it did wrong
was treat those 27 paired differences as 27 independent observations.

They are not. Three cells share a speaker, a reference clip and a voice; nine
share a sentence. A difference that is really "this engine suits cv09" shows up
three times and looks three times as certain as it is. The standard error over
27 cells is therefore too small, and it is the denominator of every claim here.

So the headline statistic is the **speaker-level** one: average the three cells
of a speaker into one number, then work with the nine speaker differences. That
is n=9, a t interval with 8 degrees of freedom, and intervals roughly twice as
wide. The cell-level figure is printed beside it, because the gap between the
two is itself worth seeing, but the speaker-level interval is what a claim may
rest on.

Decision rule, fixed before looking (the same one FINDINGS has used):

* a difference is **resolved** only if the 95% interval excludes zero *and*
  lies entirely beyond +/-0.0205
* an interval that excludes zero but not that floor is **direction only**: the
  sign is established, the size is not
* anything else is **not separable**, which is not the same as equal - with
  nine speakers this test cannot see a difference smaller than about 0.04

0.0205 is this encoder's spread on re-scoring, an instrument floor. It is not a
threshold of human perception: nobody has listened to a pair 0.02 apart and
reported whether they can hear it. Both comparisons are printed - against zero
and against that floor - so a reader can apply their own bar.

A note on what the inputs are. ``benchmark.py`` writes every per-cell
similarity rounded to four decimals, and a run's mean is the mean of those
rounded values, so everything here is computed from four-decimal numbers. And
the encoder is loaded with ``run_opts={"device": "cpu"}`` in
``evaluation/similarity.py``, so ``--device cuda`` never reaches the scoring
path: scores do not depend on which accelerator rendered the audio.

    .venv/bin/python scripts/compare_runs.py omnivoice higgs moss baseline
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"

#: The encoder's own noise on repeated scoring, pre-registered in FINDINGS.
NOISE = 0.0205

#: Two-sided 95% t critical values by degrees of freedom.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def load(label: str) -> dict:
    path = RESULTS / f"{label}.json"
    if not path.is_file():
        raise SystemExit(f"no results for {label} at {path}")
    return json.loads(path.read_text())


def cells(data: dict, field: str) -> dict[tuple, float]:
    return {
        (row["speaker"], row["sentence"], row["seed"]): row[field]
        for row in data["rows"]
        if row.get(field) is not None
    }


def interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% t interval of a sample."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, mean, mean
    half = T95.get(n - 1, 1.96) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def compare(left: dict, right: dict, field: str) -> dict | None:
    a, b = cells(left, field), cells(right, field)
    shared = sorted(set(a) & set(b))
    if len(shared) < 6:
        return None

    diffs = [a[k] - b[k] for k in shared]
    by_speaker: dict[str, list[float]] = {}
    for key in shared:
        by_speaker.setdefault(key[0], []).append(a[key] - b[key])
    per_speaker = [statistics.mean(v) for v in by_speaker.values()]

    cell_mean = statistics.mean(diffs)
    cell_se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    mean, low, high = interval(per_speaker)
    return {
        "cells": len(diffs),
        "cell_mean": cell_mean,
        "cell_t": cell_mean / cell_se if cell_se else math.inf,
        "speakers": len(per_speaker),
        "mean": mean,
        "low": low,
        "high": high,
        "speakers_favouring": sum(1 for d in per_speaker if d > 0),
    }


def verdict(stat: dict, noise: float) -> str:
    excludes_zero = stat["low"] > 0 or stat["high"] < 0
    beyond_noise = stat["low"] > noise or stat["high"] < -noise
    if excludes_zero and beyond_noise:
        return "resolved"
    if excludes_zero:
        return "direction only"
    return "not separable"


def bars(stat: dict, noise: float) -> str:
    """The two questions kept apart: is it non-zero, and is it above the floor."""
    zero = "yes" if (stat["low"] > 0 or stat["high"] < 0) else "no "
    floor = "yes" if (stat["low"] > noise or stat["high"] < -noise) else "no "
    return f"non-zero {zero}  above {noise:.4f} {floor}"


def table(runs: dict[str, dict], field: str, noise: float, scale: float, unit: str) -> None:
    print(f"  {'comparison':34s} {'cells':>8s} {'t':>5s} | {'speakers':>9s} "
          f"{'95% interval':>20s}  verdict")
    for left, right in combinations(runs, 2):
        stat = compare(runs[left], runs[right], field)
        if stat is None:
            print(f"  {left + ' - ' + right:34s}   too few shared cells")
            continue
        print(
            f"  {left + ' - ' + right:34s} {stat['cell_mean'] * scale:+8.3f} "
            f"{abs(stat['cell_t']):5.1f} | {stat['mean'] * scale:+9.3f} "
            f"[{stat['low'] * scale:+7.3f},{stat['high'] * scale:+7.3f}]  "
            f"{verdict(stat, noise):15s}"
            f" {bars(stat, noise)}"
            f"  {stat['speakers_favouring']}/{stat['speakers']} speakers"
        )
    print(f"  (differences in {unit}; the interval is over speakers, not cells)")


def main() -> int:
    labels = sys.argv[1:]
    if not labels:
        raise SystemExit(__doc__)
    runs = {label: load(label) for label in labels}

    print("\nRuns")
    print("=" * 96)
    print(f"  {'label':26s} {'engine':12s} {'held-out':>9s} {'ref':>7s} {'CER':>7s} "
          f"{'renders':>8s} {'fails':>6s} {'no CER':>7s}")
    for label, data in runs.items():
        s = data["summary"]
        print(
            f"  {label:26s} {str(data.get('engine')):12s} "
            f"{s['held_out_mean']:9.4f}" if s.get("held_out_mean") is not None else
            f"  {label:26s} {str(data.get('engine')):12s} {'-':>9s}",
            end="",
        )
        cer = f"{s['cer_mean'] * 100:6.1f}%" if s.get("cer_mean") is not None else "      -"
        ref = f"{s['reference_mean']:7.3f}" if s.get("reference_mean") is not None else "      -"
        print(f" {ref} {cer} {s['renders']:8d} {s['failures']:6d} {s.get('untranscribed', 0):7d}")

    print("\nHeld-out speaker similarity")
    print("=" * 96)
    table(runs, "held_out_similarity", NOISE, 1.0, "cosine")

    print("\nCharacter error rate - lower is better, so a positive difference is worse")
    print("=" * 96)
    # No pre-registered instrument noise for CER, so the only bar is the interval.
    table(runs, "character_error_rate", 0.0, 100.0, "percentage points")

    print("\nShare of each speaker's own ceiling")
    print("=" * 96)
    print("  A speaker's own recordings score 0.595 to 0.781 against each other, so a raw")
    print("  mean mixes cloning quality with how consistent that speaker's corpus is. This")
    print("  normalises it. It is a ratio of cosines, not a percentage of human likeness.")
    for label, data in runs.items():
        ceilings = data["calibration"]["per_speaker_ceiling"]
        shares = [
            row["held_out"] / ceilings[speaker]
            for speaker, row in data["per_speaker"].items()
            if ceilings.get(speaker) and row.get("held_out") is not None
        ]
        if shares:
            mean, low, high = interval(shares)
            print(f"  {label:26s} {mean * 100:5.1f}%  [{low * 100:.1f}, {high * 100:.1f}]"
                  f"  over {len(shares)} speakers")

    controls = sorted(p.stem for p in RESULTS.glob("baseline-*-env.json"))
    if controls:
        reference = load("baseline")["summary"]["held_out_mean"]
        print("\nScale controls - the same renders re-scored in another environment")
        print("=" * 96)
        print("  Per-cell scores are written rounded to four decimals, so an identical mean")
        print("  says the rounded cells matched - agreement within 5e-5 a cell, not a claim")
        print("  about the embedder's raw output. The encoder runs on CPU on every machine.")
        for name in controls:
            value = load(name)["summary"]["held_out_mean"]
            drift = value - reference
            state = "same to 4 dp in every cell" if value == reference else f"{drift:+.10f}"
            print(f"  {name:26s} {value!r}   {state}")
        print(f"  {'baseline (this machine)':26s} {reference!r}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
