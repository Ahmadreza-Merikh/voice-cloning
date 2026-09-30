#!/usr/bin/env python3
"""The arms session's reading, as registered before anything rendered.

``eval/runs/persian-voice-omnivoice-test.json`` ->
``amendment_2026_09_20_other_engines_on_the_same_test`` fixed it on 2026-09-20:
on each sealed set, held-out speaker similarity only, the candidate minus each
arm, paired per speaker, with the two-sided 95% t interval over speakers that
every comparison here uses. Three outcomes, per arm:

* the interval above zero: the candidate beats that arm on unseen speakers
* the interval below zero: that arm beats the candidate - the adaptation is not
  the strongest thing measured here, and the project says so
* the interval across zero: nothing is claimed about the base; unresolved

It never promotes or demotes the primary result. Similarity is compared across
sessions, which every check here allowed; CER only between the two arms, which
share their session. The published model minus each arm is printed too,
labelled as not registered.

The candidate's cells come from the sessions that scored them: the Neyshekar
set from the primary rerun, the Common Voice set from the test's version 1.

    .venv/bin/python scripts/read_arms.py remote/omnivoice-arms-v2/arms
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_runs import compare  # noqa: E402

EVIDENCE = ROOT / "eval" / "runs" / "evidence"
CANDIDATE = {"ney": EVIDENCE / "omnivoice-test-primary-v1" / "omnivoice-fa-1000-test-ney.json",
             "cv": EVIDENCE / "omnivoice-test-v1" / "omnivoice-fa-1000-test-cv.json"}
PUBLISHED = {"ney": EVIDENCE / "omnivoice-test-primary-v1" / "omnivoice-published-test-ney.json",
             "cv": EVIDENCE / "omnivoice-test-v1" / "omnivoice-published-test-cv.json"}
ROLES = {"ney": "primary", "cv": "secondary"}
ARMS = ("f5", "pocket")
SIMILARITY, CER = "held_out_similarity", "character_error_rate"


def facts(pair: Any, scale: float = 1.0) -> dict[str, Any]:
    def scaled(value):
        return None if value is None else round(value * scale, 4)
    return {"mean": scaled(pair.mean), "low": scaled(pair.low), "high": scaled(pair.high),
            "speakers": len(pair.speakers_complete), "speakers_total": pair.speakers_total,
            "cells": pair.paired_cells, "left_out": pair.speakers_excluded, "note": pair.note}


def outcome(arm: str, pair: dict[str, Any]) -> str:
    if pair["low"] is None:
        return f"not judged - {pair['note'] or 'no interval'}"
    if pair["low"] > 0:
        return f"the candidate is above {arm} on these speakers (interval above zero)"
    if pair["high"] < 0:
        return f"{arm} is above the candidate (interval below zero): the adaptation is not the strongest thing measured here"
    return f"the interval crosses zero: nothing is claimed about the base against {arm}; unresolved"


def reading(candidate: dict, published: dict, arms: dict[str, dict], role: str) -> dict[str, Any]:
    out: dict[str, Any] = {"role": role, "arms": {}}
    for arm, data in arms.items():
        pair = facts(compare("candidate", candidate, arm, data, SIMILARITY))
        out["arms"][arm] = {"summary": {k: data.get("summary", {}).get(k)
                                        for k in ("held_out_mean", "cer_mean", "renders", "failures")},
                            "candidate_minus_arm": pair, "outcome": outcome(arm, pair),
                            "published_minus_arm_not_registered": facts(compare("published", published, arm,
                                                                                data, SIMILARITY))}
    if set(arms) == set(ARMS):
        out["cer_points_f5_minus_pocket"] = facts(compare("f5", arms["f5"], "pocket", arms["pocket"], CER), 100.0)
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit(__doc__)
    folder = Path(args[0])
    result = {}
    for name, role in ROLES.items():
        arms = {arm: json.loads((folder / f"{arm}-test-{name}.json").read_text(encoding="utf-8"))
                for arm in ARMS if (folder / f"{arm}-test-{name}.json").is_file()}
        candidate = json.loads(CANDIDATE[name].read_text(encoding="utf-8"))
        published = json.loads(PUBLISHED[name].read_text(encoding="utf-8"))
        result[name] = reading(candidate, published, arms, role)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
