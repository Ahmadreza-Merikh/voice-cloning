#!/usr/bin/env python3
"""Size a paired, speaker-level test for a superiority claim - with the margin
and the true effect kept apart.

The claim being planned is one-sided: candidate minus comparator, averaged per
speaker, is above a margin ``m``. The null hypothesis is ``delta <= m``. The
effect the test is sized to detect is a separate number, the true difference
``delta`` that the design assumes when it asks "how often would the interval's
lower end clear m?". With nine-speaker spreads, the two get mixed easily, and
mixing them breaks the plan:

* ``m = 0`` asks "is the candidate better at all";
* ``m = 0.0205`` asks "is it better by more than the practical reference line";
* the power depends on ``delta - m``. A design with ``delta == m`` has power
  equal to alpha whatever its size - sizing "margin 0.03" for "an effect of
  0.03", as the roadmap once did, plans a test that cannot succeed.

Power is exact: under a true difference ``delta`` the statistic
``(mean - m) / (sd / sqrt(n))`` follows a noncentral t with ``n - 1`` degrees
of freedom and noncentrality ``(delta - m) * sqrt(n) / sd``. The test rejects
when the lower end of the one-sided ``1 - alpha`` interval clears ``m``; with
``alpha = 0.025`` that is the lower end of the usual two-sided 95% interval.

The spread ``sd`` is the speaker-level standard deviation of paired
differences. It is estimated from the development results named below, and
because nine speakers estimate it loosely, the table is also given at the upper
end of its 95% interval.

Usage
-----
    .venv/bin/python scripts/power_design.py
    .venv/bin/python scripts/power_design.py --json eval/manifests/power_design_v1.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The development comparisons the spread is estimated from: OmniVoice, the
#: candidate being adapted, against every other engine measured on all 27 cells.
SPREAD_PAIRS: list[tuple[str, str]] = [
    ("omnivoice", "baseline"),
    ("omnivoice", "fine" + "tuned-bestof4"),
    ("omnivoice", "moss"),
    ("omnivoice", "fish"),
    ("omnivoice", "higgs"),
]

MARGINS = (0.0, 0.0205, 0.03)
EFFECTS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
ALPHA = 0.025
POWER = 0.80
MAX_SPEAKERS = 2000


def power(n: int, delta: float, margin: float, sd: float, alpha: float = ALPHA) -> float:
    """Chance that the one-sided interval's lower end clears ``margin``."""
    from scipy import stats

    if n < 2:
        return 0.0
    df = n - 1
    critical = stats.t.ppf(1 - alpha, df)
    noncentrality = (delta - margin) * math.sqrt(n) / sd
    return float(stats.nct.sf(critical, df, noncentrality))


def speakers_needed(
    delta: float, margin: float, sd: float, alpha: float = ALPHA, target: float = POWER
) -> int | None:
    """Smallest number of speakers reaching ``target`` power, or None if none can."""
    if delta <= margin:
        return None
    for n in range(2, MAX_SPEAKERS + 1):
        if power(n, delta, margin, sd, alpha) >= target:
            return n
    return None


def spread_interval(sd: float, df: int, level: float = 0.95) -> tuple[float, float]:
    """Chi-square interval for a standard deviation estimated with ``df`` degrees of freedom."""
    from scipy import stats

    tail = (1 - level) / 2
    return (
        sd * math.sqrt(df / stats.chi2.ppf(1 - tail, df)),
        sd * math.sqrt(df / stats.chi2.ppf(tail, df)),
    )


def _compare_runs():
    spec = importlib.util.spec_from_file_location("compare_runs", REPO_ROOT / "scripts" / "compare_runs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_runs"] = module
    spec.loader.exec_module(module)
    return module


def pilot_spread(pairs: list[tuple[str, str]] = SPREAD_PAIRS) -> dict:
    """Speaker-level spread of paired held-out similarity, per pair and pooled."""
    compare_runs = _compare_runs()
    per_pair = {}
    variances = []
    speakers = None
    for left, right in pairs:
        pair = compare_runs.compare(
            left, compare_runs.load(left), right, compare_runs.load(right), "held_out_similarity"
        )
        values = list(pair.per_speaker.values())
        sd = statistics.stdev(values)
        per_pair[f"{left} - {right}"] = round(sd, 4)
        variances.append(sd**2)
        speakers = len(values)
    pooled = math.sqrt(statistics.mean(variances))
    df = speakers - 1
    low, high = spread_interval(pooled, df)
    return {
        "pairs": per_pair,
        "pooled_sd": round(pooled, 4),
        "speakers": speakers,
        "interval_df": df,
        "interval_95": [round(low, 4), round(high, 4)],
        "note": (
            "The pairs share their nine speakers, so they are not independent "
            "estimates; the interval uses the degrees of freedom of one pair."
        ),
    }


def table(sds: list[float]) -> list[dict]:
    rows = []
    for margin in MARGINS:
        for delta in EFFECTS:
            row = {"margin": margin, "true_effect": delta}
            for sd in sds:
                row[f"speakers_at_sd_{sd:.4f}"] = speakers_needed(delta, margin, sd)
            rows.append(row)
    return rows


def power_at(speakers: int, sds: list[float]) -> list[dict]:
    rows = []
    for margin in MARGINS:
        for delta in EFFECTS:
            if delta <= margin:
                continue
            row = {"margin": margin, "true_effect": delta, "speakers": speakers}
            for sd in sds:
                row[f"power_at_sd_{sd:.4f}"] = round(power(speakers, delta, margin, sd), 3)
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=Path, help="Also write the design to this file")
    parser.add_argument("--speakers", type=int, default=30, help="Also give the power at this many speakers")
    args = parser.parse_args(argv)

    spread = pilot_spread()
    sds = [spread["pooled_sd"], spread["interval_95"][1]]
    rows = table(sds)
    at_n = power_at(args.speakers, sds)

    print("Speaker-level spread of paired held-out similarity (development set)")
    for name, sd in spread["pairs"].items():
        print(f"  {name:40s} {sd:.4f}")
    print(
        f"  pooled {spread['pooled_sd']:.4f}, 95% interval {spread['interval_95'][0]:.4f}-"
        f"{spread['interval_95'][1]:.4f} ({spread['interval_df']} df)\n"
    )
    print(f"Speakers for {POWER:.0%} power, one-sided alpha {ALPHA} (the lower end of a two-sided 95% interval)")
    print(f"  {'margin m':>9s} {'true effect':>12s} " + " ".join(f"{'sd ' + format(sd, '.4f'):>11s}" for sd in sds))
    for row in rows:
        cells = []
        for sd in sds:
            value = row[f"speakers_at_sd_{sd:.4f}"]
            cells.append(f"{'impossible' if value is None else value:>11}")
        print(f"  {row['margin']:>9.4f} {row['true_effect']:>12.3f} " + " ".join(cells))
    print("\n'impossible': the true effect does not exceed the margin, so power stays at alpha.")

    print(f"\nPower with {args.speakers} speakers")
    print(f"  {'margin m':>9s} {'true effect':>12s} " + " ".join(f"{'sd ' + format(sd, '.4f'):>11s}" for sd in sds))
    for row in at_n:
        cells = " ".join(f"{row[f'power_at_sd_{sd:.4f}']:>11.0%}" for sd in sds)
        print(f"  {row['margin']:>9.4f} {row['true_effect']:>12.3f} {cells}")

    if args.json:
        design = {
            "method": "exact noncentral t, one-sided, paired speaker-level means",
            "alpha_one_sided": ALPHA,
            "target_power": POWER,
            "spread": spread,
            "table": rows,
            "power_at_speakers": at_n,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
