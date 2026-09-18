#!/usr/bin/env python3
"""Score real responses to a blind listening test. It never produces responses.

Reads the key written by ``build_listening_test.py`` and one or more
``responses.json`` files that listeners saved from the page, and reports, per
question, how often each system was preferred.

The unit that matters is again the speaker, not the answer. One listener
hearing three sentences from one speaker is not three independent judgements,
and neither are ten listeners hearing the same pair. So the headline is a t
interval over speakers of each speaker's mean preference (+1 for system a, -1
for b, 0 for "no difference"), printed only when there are enough speakers and
some spread - the same rule as ``compare_runs.py``. Raw counts are printed
beside it. Listener agreement is not modelled; with few listeners, treat the
result as exploratory unless a plan was written down before the test was run.

    .venv/bin/python scripts/score_listening_test.py listening/<name>.key.json responses/*.json
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_runs_for_listening", ROOT / "scripts" / "compare_runs.py")
compare_runs = importlib.util.module_from_spec(_spec)
sys.modules["compare_runs_for_listening"] = compare_runs
_spec.loader.exec_module(compare_runs)

QUESTIONS = ("same", "natural")


def preference(answer: str | None, trial: dict) -> int | None:
    """+1 if system a was chosen, -1 for b, 0 for no difference, None if unanswered."""
    if answer is None:
        return None
    if answer == "0":
        return 0
    chosen = trial["sample_1"] if answer == "1" else trial["sample_2"]
    return 1 if chosen == "a" else -1


def score(key: dict, responses: list[dict]) -> dict:
    per_speaker: dict[str, dict[str, list[int]]] = {q: defaultdict(list) for q in QUESTIONS}
    counts = {q: {"a": 0, "b": 0, "none": 0, "unanswered": 0} for q in QUESTIONS}
    misread = {"a": 0, "b": 0, "trials": 0}
    for response in responses:
        for trial_id, answer in response.get("answers", {}).items():
            trial = key["trials"].get(trial_id)
            if trial is None:
                continue
            for question in QUESTIONS:
                value = preference(answer.get(question), trial)
                if value is None:
                    counts[question]["unanswered"] += 1
                    continue
                counts[question][{1: "a", -1: "b", 0: "none"}[value]] += 1
                per_speaker[question][trial["speaker"]].append(value)
            misread["trials"] += 1
            for flagged in answer.get("misread") or []:
                misread[trial["sample_1"] if flagged == "1" else trial["sample_2"]] += 1

    report = {"listeners": len(responses), "counts": counts, "misread": misread, "intervals": {}}
    for question in QUESTIONS:
        means = [statistics.mean(v) for v in per_speaker[question].values() if v]
        if len(means) < compare_runs.MIN_SPEAKERS:
            report["intervals"][question] = f"no interval: {len(means)} speakers, {compare_runs.MIN_SPEAKERS} needed"
            continue
        stats = compare_runs.interval(means)
        report["intervals"][question] = (
            "no interval: speakers do not vary" if stats is None
            else {"mean": stats[0], "low": stats[1], "high": stats[2], "speakers": len(means)}
        )
    return report


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    key = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    responses = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:]]
    report = score(key, responses)
    print(f"{report['listeners']} listener file(s); system a = {key['systems']['a']}, b = {key['systems']['b']}")
    for question in QUESTIONS:
        print(f"  {question:8s} counts {report['counts'][question]}  interval {report['intervals'][question]}")
    print(f"  misread flags over {report['misread']['trials']} answered trials: "
          f"a {report['misread']['a']}, b {report['misread']['b']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
