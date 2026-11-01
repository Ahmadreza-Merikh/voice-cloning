#!/usr/bin/env python3
"""Read the ezafe voices check through its registered rule.

eval/runs/persian-voice-ezafe-voices-v1.json fixed, before any render: over
Neyshekar's validation voices, the app's OmniVoice weights become E5 if the 95%
interval of E5's held-out similarity minus E1's does not lie wholly below zero
and E5's mean CER is not more than 2 points above E1's; otherwise they stay E1.
The interval is ``scripts/compare_runs.py``'s: the mean of each speaker's paired
differences, a t interval over speakers. An interval it cannot give - too few
complete speakers - keeps E1.

Beside the rule, not deciding anything: E1 and E5 against A1 the same way. If
either lies wholly below zero it is said, and whether to go back to A1 is the
user's call; the listening check heard A1 say 16 words wrong to E1's 6.

    .venv/bin/python scripts/read_ezafe_voices.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "remote" / "ezafe-voices-v1" / "ezafe-voices-v1" / "results"
OUT = ROOT / "eval" / "runs" / "evidence" / "ezafe-voices-v1" / "reading.json"
LABELS = {form: f"ezafe-voices-{form}" for form in ("A1", "E1", "E5")}
PAIRS = (("E5", "E1"), ("E1", "A1"), ("E5", "A1"))

spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts" / "compare_runs.py")
compare_runs = importlib.util.module_from_spec(spec)
sys.modules["compare_runs"] = compare_runs
spec.loader.exec_module(compare_runs)


def pair(runs: dict, left: str, right: str, name: str) -> dict:
    p = compare_runs.compare(LABELS[left], runs[left], LABELS[right], runs[right], name)
    return {"mean": p.mean, "low": p.low, "high": p.high, "speakers": len(p.speakers_complete),
            "of": p.speakers_total, "left_out": p.speakers_excluded,
            "wholly_below_zero": p.high is not None and p.high < 0}


def main(results: Path = RESULTS, out: Path = OUT) -> int:
    runs = {form: compare_runs.load(label, root=results) for form, label in LABELS.items()}
    similarity = {f"{left}-{right}": pair(runs, left, right, compare_runs.SIMILARITY) for left, right in PAIRS}
    cer = {f"{left}-{right}": pair(runs, left, right, compare_runs.CER) for left, right in PAIRS}
    summary = {form: {key: runs[form]["summary"].get(key) for key in ("held_out_mean", "cer_mean", "renders",
                                                                       "failures")} for form in runs}
    primary = similarity["E5-E1"]
    cer_rise = (summary["E5"]["cer_mean"] - summary["E1"]["cer_mean"]) * 100
    if primary["low"] is None:
        choice, why = "E1", f"no interval for E5 - E1 ({primary['speakers']} complete speakers)"
    elif primary["wholly_below_zero"]:
        choice, why = "E1", "E5's similarity against E1 lies wholly below zero"
    elif cer_rise > 2:
        choice, why = "E1", f"E5's CER is {cer_rise:.2f} points above E1's"
    else:
        choice, why = "E5", "E5's similarity against E1 does not lie wholly below zero and its CER is within 2 points"
    flags = [name for name in ("E1-A1", "E5-A1") if similarity[name]["wholly_below_zero"]]
    result = {"summary": summary, "held_out_similarity": similarity, "character_error_rate": cer,
              "e5_cer_rise_points": round(cer_rise, 2), "choice": choice, "why": why,
              "read_beside_it": {"below_A1": flags, "practical_reference": compare_runs.PRACTICAL_REFERENCE}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("summary", "held_out_similarity", "e5_cer_rise_points", "choice",
                                                   "why")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
