#!/usr/bin/env python3
"""Compare benchmark runs: paired, clustered by speaker, and checked before counted.

Engines are scored on the same cells - speaker x sentence x seed - against the
same reference clips, so two runs are compared cell by cell. The cells are not
independent: every cell of a speaker shares that speaker's voice and reference
clip. So a comparison is summarised per speaker (the mean of that speaker's
paired differences) and the interval is a t interval over speakers.

Before anything is averaged, each run is checked, and every problem is printed
rather than stepped around:

* a key that appears twice is an error - there is no right answer to which
  value to use, so the comparison is refused
* a missing value, a failed render or a non-finite score makes that speaker
  *incomplete* for the pair, and incomplete speakers are left out **and listed**.
  Dropping a failed cell quietly would compare an engine only where it worked,
  which flatters exactly the engine that failed
* an interval needs at least MIN_SPEAKERS complete speakers, and at least some
  spread between them. Fewer, or none, and the tool says so instead of printing
  a zero-width interval that looks like certainty

How to read the output. Each pair gets a 95% t interval over speakers, with the
exact t quantile for its degrees of freedom. The interval is then described
against two reference points, kept separate because they are different kinds
of thing:

* **0** - the statistical question
* **+/-0.0205** - a practical reference, not an instrument floor and not a
  perceptual threshold. It is the largest change in ECAPA score measured
  across identity-preserving perturbations of *one* render - 85-95% crops,
  +/-1 dB, dither at -60 dBFS - on 2026-07-30 (commit dee1ace, sd 0.0055),
  four days before the benchmark produced its first result. It measures the
  encoder's sensitivity to edits that should not matter. It does not measure
  re-scoring repeatability (re-scoring identical audio reproduces every cell
  here), seed-to-seed generation variance, or clip-to-clip variance, and
  nobody has tested whether a listener hears a difference of that size.

The speaker-level interval rule was adopted on 2026-09-18, after these results
had been seen. It is a correction, not a pre-registration.

Inputs are what ``benchmark.py`` writes: per-cell similarities rounded to four
decimals. Two scores equal at four decimals can differ by up to 1e-4 before
rounding.

    .venv/bin/python scripts/compare_runs.py omnivoice higgs fish baseline
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"

#: See the module docstring for what this is and, as importantly, what it is not.
PRACTICAL_REFERENCE = 0.0205

#: Fewest complete speakers an interval is printed for. A policy, not a law:
#: a t interval is defined from two, but with so few independent units it says
#: more about the two speakers than about the engines.
MIN_SPEAKERS = 5

SIMILARITY = "held_out_similarity"
CER = "character_error_rate"


class DuplicateKeyError(ValueError):
    """A run lists the same (speaker, sentence, seed) twice."""


def t_quantile(df: int, level: float = 0.95) -> float:
    """Exact two-sided t quantile. No table and no normal fallback."""
    from scipy.stats import t

    return float(t.ppf(0.5 + level / 2, df))


def load(label: str, root: Path = RESULTS) -> dict:
    path = root / f"{label}.json"
    if not path.is_file():
        raise SystemExit(f"no results for {label} at {path}")
    return json.loads(path.read_text())


@dataclass
class Cells:
    """One run's values for one field, with everything that went wrong."""

    values: dict[tuple, float] = field(default_factory=dict)
    missing: list[tuple] = field(default_factory=list)
    failed: list[tuple] = field(default_factory=list)
    non_finite: list[tuple] = field(default_factory=list)
    design: set[tuple] = field(default_factory=set)


def cells(data: dict, name: str) -> Cells:
    """Read one field from every row, refusing duplicates and keeping the losses."""
    out = Cells()
    for row in data["rows"]:
        key = (row["speaker"], row["sentence"], row.get("seed"))
        if key in out.design:
            raise DuplicateKeyError(f"{data.get('label', '?')}: {key} appears twice")
        out.design.add(key)
        value = row.get(name)
        if row.get("error"):
            out.failed.append(key)
        elif value is None:
            out.missing.append(key)
        elif not math.isfinite(float(value)):
            out.non_finite.append(key)
        else:
            out.values[key] = float(value)
    return out


def interval(values: list[float], level: float = 0.95) -> tuple[float, float, float] | None:
    """Mean and t interval, or None when there is nothing an interval could honestly say."""
    n = len(values)
    if n < 2:
        return None
    spread = statistics.stdev(values)
    if spread == 0:
        return None
    mean = statistics.mean(values)
    half = t_quantile(n - 1, level) * spread / math.sqrt(n)
    return mean, mean - half, mean + half


@dataclass
class Pair:
    left: str
    right: str
    design_cells: int
    paired_cells: int
    speakers_total: int
    speakers_complete: list[str]
    speakers_excluded: dict[str, str]
    per_speaker: dict[str, float]
    mean: float | None = None
    low: float | None = None
    high: float | None = None
    sd: float | None = None
    note: str = ""


def compare(left_label: str, left: dict, right_label: str, right: dict, name: str,
            min_speakers: int = MIN_SPEAKERS) -> Pair:
    a, b = cells(left, name), cells(right, name)
    design = a.design | b.design
    speakers = sorted({key[0] for key in design})

    problems: dict[tuple, str] = {}
    for label, run in ((left_label, a), (right_label, b)):
        for key in run.failed:
            problems.setdefault(key, f"{label} failed")
        for key in run.missing:
            problems.setdefault(key, f"{label} has no value")
        for key in run.non_finite:
            problems.setdefault(key, f"{label} non-finite")
        for key in design - run.design:
            problems.setdefault(key, f"{label} lacks the cell")

    per_speaker: dict[str, float] = {}
    excluded: dict[str, str] = {}
    paired = 0
    for speaker in speakers:
        keys = sorted(k for k in design if k[0] == speaker)
        bad = [problems[k] for k in keys if k in problems]
        if bad:
            excluded[speaker] = f"{len(bad)} of {len(keys)} cells unusable ({bad[0]})"
            continue
        diffs = [a.values[k] - b.values[k] for k in keys]
        per_speaker[speaker] = statistics.mean(diffs)
        paired += len(diffs)

    result = Pair(left_label, right_label, len(design), paired, len(speakers),
                  sorted(per_speaker), excluded, per_speaker)
    values = list(per_speaker.values())
    if len(values) < min_speakers:
        result.note = f"no interval: {len(values)} complete speakers, {min_speakers} needed"
        if values:
            result.mean = statistics.mean(values)
        return result
    stats = interval(values)
    result.mean = statistics.mean(values)
    result.sd = statistics.stdev(values)
    if stats is None:
        result.note = "no interval: speakers do not vary, so the interval would have zero width"
        return result
    _, result.low, result.high = stats
    return result


def describe(pair: Pair, reference: float) -> str:
    """Where the interval sits, in words that make no claim beyond it."""
    if pair.low is None or pair.high is None:
        return pair.note
    clear_zero = pair.low > 0 or pair.high < 0
    clear_reference = pair.low > reference or pair.high < -reference
    if not clear_zero:
        return "interval includes 0"
    if reference and clear_reference:
        return f"interval clear of 0 and of +/-{reference:g}"
    if reference:
        return f"interval clear of 0, overlaps +/-{reference:g}"
    return "interval clear of 0"


def _provenance(root: Path = RESULTS) -> dict:
    path = root / "_provenance.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("runs", {})


def scoring_environment(data: dict, root: Path = RESULTS) -> str | None:
    """Where a run was *scored*, or None when nothing on record says.

    Newer files carry the environment ``benchmark.py`` recorded. Older ones are
    looked up in ``eval/results/_provenance.json``, which maps each label to a
    Mac or to one specific Kaggle kernel session with the evidence for it. The
    device alone is not enough: every Kaggle kernel says ``cuda``, and each one
    is a separate install. A run found in neither place has no environment, and
    nothing that needs one - a CER comparison - is judged for it.
    """
    recorded = data.get("scoring_environment")
    if recorded and recorded.get("id"):
        return str(recorded["id"])
    entry = _provenance(root).get(str(data.get("label")))
    return entry["environment"] if entry else None


def scale_control(reference: dict, control: dict) -> dict:
    """Compare every cell of a re-scoring against the original, by key.

    Equal means prove nothing: opposite differences cancel. So keys are matched
    and each cell is compared, and anything unmatched is reported.
    """
    a, b = cells(reference, SIMILARITY), cells(control, SIMILARITY)
    only_reference = sorted(a.design - b.design)
    only_control = sorted(b.design - a.design)
    shared = sorted(set(a.values) & set(b.values))
    differences = {k: b.values[k] - a.values[k] for k in shared}
    unequal = {k: d for k, d in differences.items() if d != 0}
    return {
        "shared": len(shared),
        "only_reference": only_reference,
        "only_control": only_control,
        "unusable": len(a.design) - len(a.values) + len(b.design) - len(b.values),
        "unequal": unequal,
        "max_abs": max((abs(d) for d in differences.values()), default=0.0),
    }


def _source(data: dict) -> str:
    replay = (data.get("config") or {}).get("replay_from")
    return f"renders of {replay}" if replay else str(data.get("engine"))


def main(argv: list[str]) -> int:
    labels = argv[1:]
    if not labels:
        raise SystemExit(__doc__)
    runs = {label: load(label) for label in labels}

    print("\nRuns")
    print("=" * 100)
    for label, data in runs.items():
        s = data["summary"]
        sim, cer = cells(data, SIMILARITY), cells(data, CER)
        held = f"{s['held_out_mean']:.4f}" if s.get("held_out_mean") is not None else "-"
        mean_cer = f"{s['cer_mean'] * 100:.1f}%" if s.get("cer_mean") is not None else "-"
        print(f"  {label:26s} {_source(data):22s} held-out {held:>7s}  CER {mean_cer:>6s}  "
              f"cells {len(sim.design):2d}  failed {len(sim.failed)}  "
              f"no similarity {len(sim.missing) + len(sim.non_finite)}  "
              f"no CER {len(cer.missing) + len(cer.non_finite)}")
        print(f"  {'':26s} scored in {scoring_environment(data) or 'an environment nothing records'}")

    print("\nHeld-out speaker similarity - 95% t interval over speakers")
    print("=" * 100)
    for left, right in combinations(runs, 2):
        pair = compare(left, runs[left], right, runs[right], SIMILARITY)
        _print_pair(pair, 1.0, PRACTICAL_REFERENCE)

    print("\nCharacter error rate - positive means the first run is worse")
    print("=" * 100)
    print("  FINDINGS section 9: CER may only be compared between runs scored in the")
    print("  same environment. Pairs scored in different places are shown, not judged.")
    for left, right in combinations(runs, 2):
        pair = compare(left, runs[left], right, runs[right], CER)
        place = scoring_environment(runs[left])
        same_place = place is not None and place == scoring_environment(runs[right])
        _print_pair(pair, 100.0, 0.0, judged=same_place)

    controls = sorted(p.stem for p in RESULTS.glob("baseline-*-env.json"))
    if controls and (RESULTS / "baseline.json").is_file():
        reference = load("baseline")
        print("\nScale controls - the baseline renders re-scored elsewhere, cell by cell")
        print("=" * 100)
        for name in controls:
            check = scale_control(reference, load(name))
            if check["only_reference"] or check["only_control"] or check["unusable"]:
                state = (f"keys differ: {len(check['only_reference'])} only in baseline, "
                         f"{len(check['only_control'])} only here, {check['unusable']} unusable")
            elif check["unequal"]:
                state = (f"{len(check['unequal'])} of {check['shared']} cells differ, "
                         f"largest {check['max_abs']:.4f}")
            else:
                state = f"all {check['shared']} cells equal at 4 dp (raw scores within 1e-4)"
            print(f"  {name:26s} {state}")
    print()
    return 0


def _print_pair(pair: Pair, scale: float, reference: float, judged: bool = True) -> None:
    head = f"  {pair.left + ' - ' + pair.right:34s}"
    coverage = (f"{len(pair.speakers_complete)}/{pair.speakers_total} speakers, "
                f"{pair.paired_cells}/{pair.design_cells} cells")
    if pair.low is None:
        mean = "" if pair.mean is None else f"mean {pair.mean * scale:+.3f}  "
        print(f"{head} {mean}{pair.note}  ({coverage})")
    else:
        verdict = describe(pair, reference) if judged else "not the same scoring environment - not judged"
        print(f"{head} {pair.mean * scale:+7.3f} [{pair.low * scale:+7.3f}, {pair.high * scale:+7.3f}]"
              f"  sd {pair.sd * scale:.3f}  {verdict}  ({coverage})")
    for speaker, reason in pair.speakers_excluded.items():
        print(f"  {'':34s}   left out: {speaker} - {reason}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
