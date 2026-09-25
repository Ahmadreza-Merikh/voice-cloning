#!/usr/bin/env python3
"""Read the LongCat probe by the rule fixed before any render (eval/runs/persian-voice-longcat-probe.json).

The question: does LongCat-AudioDiT carry a Persian speaker's voice better than
OmniVoice, when both read text they know? If it does not, adapting it to
Persian - no training code, a new language learned from nothing, A100-class
hardware for its fp32 recipe - is not justified by its cloning.

**Primary**: LongCat-1B minus OmniVoice on held-out similarity, paired per
speaker (``compare_runs``: each speaker's mean over its cells, a speaker with
any unusable cell left out and listed), with the two-sided 95% t interval over
speakers. The bar is the practical reference line, 0.0205:

* the lower end above it - better by more than the line: an adaptation is
  worth costing
* the upper end below it - not better by the line: its cloning does not
  justify adapting it (below 0 as a whole: worse than OmniVoice)
* otherwise not resolved - which does not justify the cost either

At 25 speakers that bar is cleared with 81% power if the true advantage is
0.05 and the spread is the pilot's 0.050, 31% at its upper end 0.096; an
advantage of 0.03 would very probably not clear it, and would not pay for an
adaptation.

**Guard**: similarity of a render that is not the requested speech says
little about the voice. A render the recogniser - forced to English - could
not read, or read at a WER above 0.5, is counted as not the text; an arm with
more than 10% of those is inconclusive and cannot clear the bar.

**Secondary**, labelled so: LongCat-3.5B minus OmniVoice by the same rule, and
3.5B minus 1B. **Exploratory**: how much more each engine scores against its
own prompt than against the speaker's other clips - a larger gap for LongCat
would fit the reading that its Seed-TTS-eval lead is partly the prompt
recording copied - and, if those runs are given, LongCat reading Persian
against OmniVoice on CER.

Every run must have been scored in one environment, or nothing is judged.

    .venv/bin/python scripts/read_longcat_probe.py --omnivoice omnivoice-probe-en \\
        --longcat longcat-1b-probe-en --secondary longcat-3.5b-probe-en --out reading.json
"""

from __future__ import annotations

import argparse
import json
import re
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
from read_sealed_test import one_sided_p, pair_facts  # noqa: E402

WER_LIMIT = 0.5
SHARE_LIMIT = 0.10
PROMPT_GAP = "prompt_minus_held_out"


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", (text or "").lower()).split()


def english_wer(expected: str, heard: str) -> float | None:
    reference, hypothesis = words(expected), words(heard)
    if not reference:
        return None
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, other in enumerate(hypothesis, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other))
        previous = current
    return previous[-1] / len(reference)


def guard(data: dict) -> dict[str, Any]:
    """How many of a run's renders were not the requested English."""
    bad = []
    for row in data["rows"]:
        key = f"{row['speaker']}/{row['sentence']}/{row.get('seed')}"
        if row.get("error"):
            bad.append({"cell": key, "why": "failed"})
            continue
        wer = english_wer(row.get("expected", ""), row.get("transcription") or "")
        if not (row.get("transcription") or "").strip():
            bad.append({"cell": key, "why": "the recogniser heard nothing"})
        elif wer is not None and wer > WER_LIMIT:
            bad.append({"cell": key, "why": f"WER {wer:.2f}"})
    renders = len(data["rows"])
    share = len(bad) / renders if renders else 1.0
    return {"renders": renders, "not_the_text": len(bad), "share": share, "holds": share <= SHARE_LIMIT,
            "examples": bad[:10]}


def wer_rows(data: dict) -> dict:
    """The run with each row's English WER as a field ``compare`` can pair."""
    rows = []
    for row in data["rows"]:
        wer = None if row.get("error") else english_wer(row.get("expected", ""), row.get("transcription") or "")
        rows.append({**row, "english_wer": wer})
    return {**data, "rows": rows}


def gap_rows(data: dict) -> dict:
    rows = []
    for row in data["rows"]:
        prompt, held = row.get("reference_similarity"), row.get(SIMILARITY)
        rows.append({**row, PROMPT_GAP: None if prompt is None or held is None else prompt - held})
    return {**data, "rows": rows}


def verdict(same_place: bool, guards: dict[str, dict], similarity: Any) -> str:
    if not same_place:
        return "not judged - the runs were not scored in one environment"
    broken = [label for label, facts in guards.items() if not facts["holds"]]
    if broken:
        return (f"inconclusive - more than {SHARE_LIMIT:.0%} of {', '.join(broken)}'s renders were not the "
                "requested English, so their similarity says little about the voice")
    if similarity.low is None:
        return f"not judged - {similarity.note}"
    if similarity.low > PRACTICAL_REFERENCE:
        return (f"better by more than the practical line: the whole interval is above {PRACTICAL_REFERENCE}, "
                "so an adaptation is worth costing")
    if similarity.high < 0:
        return "worse than the other: the whole interval is below 0"
    if similarity.high < PRACTICAL_REFERENCE:
        return (f"not better by the practical line: the whole interval is below {PRACTICAL_REFERENCE}, so its "
                "cloning does not justify adapting it")
    return (f"not resolved: the interval straddles {PRACTICAL_REFERENCE}, which does not show an advantage "
            "worth an adaptation")


def paired(left_label: str, left: dict, right_label: str, right: dict) -> dict[str, Any]:
    similarity = compare(left_label, left, right_label, right, SIMILARITY)
    wer = compare(left_label, wer_rows(left), right_label, wer_rows(right), "english_wer")
    place = scoring_environment(left)
    same_place = place is not None and place == scoring_environment(right)
    guards = {left_label: guard(left), right_label: guard(right)}
    judged = same_place and all(g["holds"] for g in guards.values()) and similarity.low is not None
    return {
        "left": left_label,
        "right": right_label,
        "scoring_environment": place if same_place else [place, scoring_environment(right)],
        "held_out_similarity": {**pair_facts(similarity),
                                "p_one_sided_above_0": one_sided_p(list(similarity.per_speaker.values()))},
        "english_wer_percentage_points": pair_facts(wer, 100.0),
        "means": {left_label: left["summary"].get("held_out_mean"), right_label: right["summary"].get("held_out_mean")},
        "guard": guards,
        "clears_practical_line": judged and similarity.low > PRACTICAL_REFERENCE,
        "verdict": verdict(same_place, guards, similarity),
    }


def prompt_copying(labelled: dict[str, dict], reference: str) -> dict[str, Any]:
    """Exploratory: prompt similarity minus held-out similarity, each engine against the reference engine."""
    gaps = {label: statistics.mean(v for r in gap_rows(data)["rows"] if (v := r.get(PROMPT_GAP)) is not None)
            for label, data in labelled.items()}
    pairs = {label: pair_facts(compare(label, gap_rows(data), reference, gap_rows(labelled[reference]), PROMPT_GAP))
             for label, data in labelled.items() if label != reference}
    return {"label": "exploratory - not a claim", "mean_gap": gaps, f"minus_{reference}": pairs}


def persian(omnivoice_label: str, omnivoice: dict, longcat_label: str, longcat: dict) -> dict[str, Any]:
    """Exploratory: LongCat reading Persian script, beside OmniVoice in the same session."""
    cer = compare(longcat_label, longcat, omnivoice_label, omnivoice, CER)
    return {"label": "exploratory - LongCat was never trained on Persian",
            "cer_mean": {omnivoice_label: omnivoice["summary"].get("cer_mean"),
                         longcat_label: longcat["summary"].get("cer_mean")},
            "untranscribed": {omnivoice_label: omnivoice["summary"].get("untranscribed"),
                              longcat_label: longcat["summary"].get("untranscribed")},
            f"{longcat_label}_minus_{omnivoice_label}_cer_percentage_points": pair_facts(cer, 100.0)}


def reading(runs: dict[str, dict], omnivoice: str, longcat: str, secondary: str | None,
            persian_runs: tuple[str, str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question": "does LongCat-AudioDiT carry a Persian speaker's voice better than OmniVoice, when both read English?",
        "primary": {"role": "primary", **paired(longcat, runs[longcat], omnivoice, runs[omnivoice])},
    }
    if secondary:
        result["secondary"] = {
            "role": "secondary - reported, not the decision",
            "against_omnivoice": paired(secondary, runs[secondary], omnivoice, runs[omnivoice]),
            "against_the_1b": paired(secondary, runs[secondary], longcat, runs[longcat]),
        }
    engines = {label: runs[label] for label in (omnivoice, longcat, secondary) if label}
    result["exploratory_prompt_copying"] = prompt_copying(engines, omnivoice)
    if persian_runs:
        omnivoice_fa, longcat_fa = persian_runs
        result["exploratory_persian"] = persian(omnivoice_fa, runs[omnivoice_fa], longcat_fa, runs[longcat_fa])
    result["decision"] = {
        "adaptation_worth_costing": result["primary"]["clears_practical_line"],
        "rule": "only if LongCat-1B's held-out similarity minus OmniVoice's has a 95% interval over speakers "
                f"entirely above {PRACTICAL_REFERENCE}, with both arms' renders the requested English",
        "secondary_3_5b_clears_the_line": (result.get("secondary", {}).get("against_omnivoice", {})
                                           .get("clears_practical_line")),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--omnivoice", required=True)
    parser.add_argument("--longcat", required=True)
    parser.add_argument("--secondary")
    parser.add_argument("--omnivoice-fa")
    parser.add_argument("--longcat-fa")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    labels = [args.omnivoice, args.longcat, args.secondary, args.omnivoice_fa, args.longcat_fa]
    runs = {label: load(label, args.results) for label in labels if label}
    both = args.omnivoice_fa and args.longcat_fa
    result = reading(runs, args.omnivoice, args.longcat, args.secondary,
                     (args.omnivoice_fa, args.longcat_fa) if both else None)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
