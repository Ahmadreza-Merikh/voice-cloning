#!/usr/bin/env python3
"""Read the sealed test by the rule fixed before any test render (SOTA_ROADMAP.md section 3.1).

**Primary**: the Persian adaptation of OmniVoice minus OmniVoice as
published, on held-out similarity, paired per speaker, on the Neyshekar test
set. H0 is a difference of at most 0, tested one-sided at alpha 0.025, which
is the lower end of ``compare_runs``' two-sided 95% t interval over speakers:
H0 is rejected only if that lower end is above 0. The design was sized for a
true difference of 0.03 - 87% power at the pilot's spread of 0.050, 36% at the
upper end of its interval, 0.096.

Fixed with it:

* whether the lower end also clears 0.0205, the practical reference line -
  secondary, and reported as secondary
* CER is a guard, not a target: the adaptation's CER minus the published
  model's, both scored in one session, with its interval. An interval above
  zero counts against the adaptation whatever its similarity
* the Common Voice set is a secondary test set - the same numbers, labelled
  so, because OmniVoice has very probably heard those speakers
* anything else is exploratory and Holm-corrected; here, the text categories
  (seven labels: the roadmap's sixth category, very short or long, is two)
* a speaker missing a cell for either model is left out of the pair and
  listed (``compare_runs``' rule), no speaker is added after looking, and a
  spread above the planning value is reported as it is

The two runs must have been scored in one environment, or nothing is judged.

    .venv/bin/python scripts/read_sealed_test.py --candidate omnivoice-fa-3000-test-ney \\
        --published omnivoice-published-test-ney --role primary --out reading.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import (  # noqa: E402
    CER,
    PRACTICAL_REFERENCE,
    RESULTS,
    SIMILARITY,
    compare,
    load,
    scoring_environment,
)

ALPHA_ONE_SIDED = 0.025
PLANNED_EFFECT = 0.03
PLANNED_SPREAD = 0.050
SPREAD_UPPER = 0.096
TEXTS = Path(__file__).resolve().parents[1] / "eval" / "manifests" / "test_texts_v1.json"


def t_statistic(values: list[float]) -> tuple[float, int] | None:
    if len(values) < 2 or statistics.stdev(values) == 0:
        return None
    return statistics.mean(values) / (statistics.stdev(values) / math.sqrt(len(values))), len(values) - 1


def one_sided_p(values: list[float]) -> float | None:
    """P of a mean difference at least this large if the true difference were 0."""
    from scipy.stats import t

    found = t_statistic(values)
    return None if found is None else float(t.sf(found[0], found[1]))


def two_sided_p(values: list[float]) -> float | None:
    from scipy.stats import t

    found = t_statistic(values)
    return None if found is None else float(2 * t.sf(abs(found[0]), found[1]))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def pair_facts(pair: Any, scale: float = 1.0) -> dict[str, Any]:
    return {
        "mean": None if pair.mean is None else pair.mean * scale,
        "low": None if pair.low is None else pair.low * scale,
        "high": None if pair.high is None else pair.high * scale,
        "sd": None if pair.sd is None else pair.sd * scale,
        "speakers": len(pair.speakers_complete),
        "speakers_total": pair.speakers_total,
        "cells": pair.paired_cells,
        "design_cells": pair.design_cells,
        "left_out": pair.speakers_excluded,
        "note": pair.note,
    }


def verdict(role: str, same_place: bool, similarity: Any, cer: Any) -> str:
    prefix = "" if role == "primary" else "secondary set: "
    if not same_place:
        return prefix + "not judged - the two runs were not scored in one environment"
    if similarity.low is None:
        return prefix + f"not judged - {similarity.note}"
    if similarity.high < 0:
        return prefix + "the adaptation scores lower held-out similarity than the published model (the whole interval is below 0)"
    if similarity.low <= 0:
        return prefix + "no detectable improvement in held-out similarity: the interval's lower end is not above 0"
    if cer.low is not None and cer.low > 0:
        return (prefix + "higher held-out similarity (H0 rejected), but the CER interval lies above zero, which "
                "counts against the adaptation by the rule fixed before the run")
    return (prefix + "higher held-out similarity than the published model (H0 of no improvement rejected, one-sided "
            "alpha 0.025), with a CER interval that does not lie above zero")


def by_category(candidate_label: str, candidate: dict, published_label: str, published: dict,
                categories: dict[str, str]) -> dict[str, Any]:
    rows = {}
    for category in sorted(set(categories.values())):
        keys = {k for k, c in categories.items() if c == category}
        left = {"label": candidate_label, "rows": [r for r in candidate["rows"] if r["sentence"] in keys]}
        right = {"label": published_label, "rows": [r for r in published["rows"] if r["sentence"] in keys]}
        pair = compare(candidate_label, left, published_label, right, SIMILARITY)
        rows[category] = {**pair_facts(pair), "p_two_sided": two_sided_p(list(pair.per_speaker.values()))}
    tested = {name: row["p_two_sided"] for name, row in rows.items() if row["p_two_sided"] is not None}
    for name, p in holm(tested).items():
        rows[name]["p_holm"] = p
    return {"label": "exploratory - Holm-corrected across the categories, not a claim", "categories": rows}


def reading(candidate_label: str, candidate: dict, published_label: str, published: dict,
            role: str, categories: dict[str, str]) -> dict[str, Any]:
    similarity = compare(candidate_label, candidate, published_label, published, SIMILARITY)
    cer = compare(candidate_label, candidate, published_label, published, CER)
    place = scoring_environment(candidate)
    same_place = place is not None and place == scoring_environment(published)
    judged = same_place and similarity.low is not None
    return {
        "role": role,
        "candidate": candidate_label,
        "published": published_label,
        "scoring_environment": place if same_place else [place, scoring_environment(published)],
        "held_out_similarity": {**pair_facts(similarity),
                                "p_one_sided": one_sided_p(list(similarity.per_speaker.values()))},
        "cer_percentage_points": pair_facts(cer, 100.0),
        "h0_rejected": judged and similarity.low > 0,
        "clears_practical_reference": judged and similarity.low > PRACTICAL_REFERENCE,
        "cer_counts_against": same_place and cer.low is not None and cer.low > 0,
        "verdict": verdict(role, same_place, similarity, cer),
        "planning": {"true_effect": PLANNED_EFFECT, "spread": PLANNED_SPREAD, "spread_upper": SPREAD_UPPER,
                     "spread_observed": similarity.sd, "alpha_one_sided": ALPHA_ONE_SIDED,
                     "note": "a spread above the planning value lowers the power; the interval stands as it is"},
        "exploratory": by_category(candidate_label, candidate, published_label, published, categories),
    }


def categories_of(path: Path = TEXTS) -> dict[str, str]:
    return {t["key"]: t["category"] for t in json.loads(path.read_text(encoding="utf-8"))["texts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--published", required=True)
    parser.add_argument("--role", choices=["primary", "secondary"], required=True)
    parser.add_argument("--texts", type=Path, default=TEXTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    result = reading(args.candidate, load(args.candidate, args.results), args.published,
                     load(args.published, args.results), args.role, categories_of(args.texts))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
