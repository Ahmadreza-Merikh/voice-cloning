#!/usr/bin/env python3
"""Score real responses to a blind listening test. It never produces responses.

Reads the key written by ``build_listening_test.py`` and the ``responses.json``
files listeners saved from the page. Everything that could inflate or distort a
count is checked and reported rather than absorbed:

* a file whose test name or key fingerprint does not match the key is rejected -
  it answers a different test, or an older build of this one
* byte-identical files are counted once, and a listener id that appears in more
  than one file is counted once (the first file) with the rest reported. What
  is reported as listeners is distinct ids; files without an id are counted as
  files and flagged, because nothing shows they are different people
* a preference other than "1", "2" or "0" (no difference) is invalid, counted
  as such, and never turned into a vote
* "which misread a word" needs an explicit answer - sample 1, sample 2, both, or
  "none". Nothing ticked is unanswered, and only answered trials enter the
  denominator; "none" ticked together with a sample is a contradiction and is
  invalid

Two intervals, labelled for what they are:

* over **speakers** (each speaker's mean preference, t interval, the same rule
  as ``compare_runs.py``): conditional on *these* listeners, and exploratory
* over **speakers and listeners together**, a two-way cluster bootstrap that
  resamples both - the analysis declared for a final test, because shared
  listeners make answers dependent just as shared speakers do. It needs at
  least five of each; with fewer it is not printed

Neither is a result until real listeners have answered, and a result without a
plan written down beforehand is exploratory whatever it shows.

    .venv/bin/python scripts/score_listening_test.py listening/<name>.key.json responses/*.json
"""

from __future__ import annotations

import hashlib
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
VALID = {"0", "1", "2"}
MIN_LISTENERS = 5
BOOTSTRAP_DRAWS = 2000


def preference(answer: object, trial: dict) -> int | str | None:
    """+1 system a, -1 system b, 0 no difference; None unanswered; "invalid" otherwise."""
    if answer is None:
        return None
    if answer not in VALID:
        return "invalid"
    if answer == "0":
        return 0
    chosen = trial["sample_1"] if answer == "1" else trial["sample_2"]
    return 1 if chosen == "a" else -1


def misread_answer(boxes: object) -> str | set[str] | None:
    """The set of systems flagged, an empty set for an explicit "none", or None/"invalid"."""
    if not boxes:
        return None
    if not isinstance(boxes, list) or any(b not in {"1", "2", "none"} for b in boxes):
        return "invalid"
    if "none" in boxes:
        return set() if boxes == ["none"] else "invalid"
    return set(boxes)


def accept(key: dict, files: list[tuple[str, bytes]]) -> tuple[list[dict], dict]:
    """Keep one response per distinct content and per listener id; report the rest."""
    seen_content, seen_listener, kept = set(), {}, []
    report = {"files": len(files), "rejected": {}, "duplicate_files": [], "duplicate_listeners": [],
              "unnamed_files": 0}
    for name, raw in files:
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_content:
            report["duplicate_files"].append(name)
            continue
        seen_content.add(digest)
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            report["rejected"][name] = f"not JSON: {exc}"
            continue
        if response.get("test") != key["name"] or response.get("fingerprint") != key["fingerprint"]:
            report["rejected"][name] = "answers a different test or an older build of this one"
            continue
        listener = (response.get("listener") or "").strip()
        if not listener:
            report["unnamed_files"] += 1
        elif listener in seen_listener:
            report["duplicate_listeners"].append(f"{name} repeats {listener!r} from {seen_listener[listener]}")
            continue
        else:
            seen_listener[listener] = name
        response["_listener"] = listener or f"unnamed:{name}"
        kept.append(response)
    report["distinct_listener_ids"] = len(seen_listener)
    return kept, report


def crossed_bootstrap(values: dict[tuple[str, str], list[int]], draws: int = BOOTSTRAP_DRAWS,
                      seed: int = 0) -> tuple[float, float, float] | None:
    """Mean and 95% percentile interval, resampling speakers and listeners independently."""
    import numpy as np

    speakers = sorted({s for s, _ in values})
    listeners = sorted({who for _, who in values})
    if len(speakers) < compare_runs.MIN_SPEAKERS or len(listeners) < MIN_LISTENERS:
        return None
    rng = np.random.default_rng(seed)
    cells = {k: (sum(v), len(v)) for k, v in values.items()}
    estimates = []
    for _ in range(draws):
        s_pick = rng.choice(len(speakers), len(speakers))
        l_pick = rng.choice(len(listeners), len(listeners))
        total = count = 0
        for i in s_pick:
            for j in l_pick:
                cell = cells.get((speakers[i], listeners[j]))
                if cell:
                    total += cell[0]
                    count += cell[1]
        if count:
            estimates.append(total / count)
    everything = [x for v in values.values() for x in v]
    low, high = np.percentile(estimates, [2.5, 97.5])
    return statistics.mean(everything), float(low), float(high)


def score(key: dict, files: list[tuple[str, bytes]]) -> dict:
    responses, intake = accept(key, files)
    counts = {q: {"a": 0, "b": 0, "none": 0, "unanswered": 0, "invalid": 0} for q in QUESTIONS}
    misread = {"a": 0, "b": 0, "answered": 0, "unanswered": 0, "invalid": 0}
    unknown_trials = 0
    per_speaker: dict[str, dict[str, list[int]]] = {q: defaultdict(list) for q in QUESTIONS}
    crossed: dict[str, dict[tuple[str, str], list[int]]] = {q: defaultdict(list) for q in QUESTIONS}

    for response in responses:
        for trial_id, answer in (response.get("answers") or {}).items():
            trial = key["trials"].get(trial_id)
            if trial is None:
                unknown_trials += 1
                continue
            for question in QUESTIONS:
                value = preference(answer.get(question), trial)
                if value is None:
                    counts[question]["unanswered"] += 1
                elif value == "invalid":
                    counts[question]["invalid"] += 1
                else:
                    counts[question][{1: "a", -1: "b", 0: "none"}[value]] += 1
                    per_speaker[question][trial["speaker"]].append(value)
                    crossed[question][(trial["speaker"], response["_listener"])].append(value)
            flagged = misread_answer(answer.get("misread"))
            if flagged is None:
                misread["unanswered"] += 1
            elif flagged == "invalid":
                misread["invalid"] += 1
            else:
                misread["answered"] += 1
                for sample in flagged:
                    misread[trial["sample_1"] if sample == "1" else trial["sample_2"]] += 1

    report = {"intake": intake, "unknown_trials": unknown_trials, "counts": counts, "misread": misread,
              "speaker_interval": {}, "crossed_interval": {},
              "label": "exploratory: no analysis plan was registered before these answers"}
    for question in QUESTIONS:
        means = [statistics.mean(v) for v in per_speaker[question].values() if v]
        if len(means) < compare_runs.MIN_SPEAKERS:
            report["speaker_interval"][question] = f"no interval: {len(means)} speakers"
        else:
            stats = compare_runs.interval(means)
            report["speaker_interval"][question] = (
                "no interval: speakers do not vary" if stats is None else
                {"mean": stats[0], "low": stats[1], "high": stats[2], "speakers": len(means),
                 "conditional_on": "these listeners"})
        boot = crossed_bootstrap(dict(crossed[question]))
        report["crossed_interval"][question] = (
            f"no interval: needs {compare_runs.MIN_SPEAKERS}+ speakers and {MIN_LISTENERS}+ listeners"
            if boot is None else {"mean": boot[0], "low": boot[1], "high": boot[2]})
    return report


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    key = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    files = [(p, Path(p).read_bytes()) for p in sys.argv[2:]]
    report = score(key, files)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
