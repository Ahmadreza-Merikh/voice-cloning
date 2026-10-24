#!/usr/bin/env python3
"""Read the adapted reading check through its registered rule.

eval/runs/persian-voice-adapted-reading-v1.json fixed, before anyone listened:
the app's OmniVoice weights become an adapted export (A1 at step 1,000, A5 at
step 5,000) if it has fewer words marked wrong than the published model (P) and
is not flagged; if both qualify, the one with fewer words wrong, and on a tie
between them the one with the higher held-out similarity. A tie with P keeps P.
An export is flagged - and not adopted without the user's say - if its mean CER
is more than 2 points above P's, or if its paired held-out similarity
difference against P has a 95% interval wholly below zero. The words marked
wrong are summed over every listener who answered all 12 sentences.

Beside the rule, not deciding anything: per form, the marks on the words the
sentences were written to test - ezafe traps, vowel traps, the silent-h ezafes
- the picked readings and the notes.

    .venv/bin/python scripts/read_adapted_reading.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "eval" / "runs" / "evidence" / "adapted-review-v1"
RUN = ROOT / "eval" / "runs" / "evidence" / "adapted-reading-v1"
MANIFEST = ROOT / "eval" / "runs" / "adapted-reading-v1" / "manifest.json"
SENTENCES = ROOT / "eval" / "manifests" / "adapted_reading_v1.json"
FORMS = ("P", "A1", "A5")
EXPORTS = ("A1", "A5")
TRAPS = ("ezafe_traps", "vowel_traps", "silent_h_ezafe")


def slot_done(given: dict | None) -> bool:
    return bool(given) and (given.get("ok") is True or bool(given.get("words")))


def complete(answers: dict, key: dict) -> bool:
    return all(sid in answers and answers[sid].get("best")
               and all(slot_done(answers[sid].get("slots", {}).get(slot)) for slot in sentence["slots"])
               for sid, sentence in key["sentences"].items())


def plain(word: str) -> str:
    return re.sub(r"[.,،؛:!؟?«»()\"]", "", word)


def main() -> int:
    key = json.loads((REVIEW / "key.json").read_text(encoding="utf-8"))
    items = {i["id"]: i["words"] for i in json.loads((REVIEW / "items.json").read_text(encoding="utf-8"))}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    similarity = json.loads((RUN / "similarity.json").read_text(encoding="utf-8"))
    sentences = {s["id"]: s for s in json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]}

    listeners, left_out = [], []
    for path in sorted(REVIEW.glob("answers-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        (listeners if complete(record["answers"], key) else left_out).append((path.name, record))

    wrong: dict[tuple[str, str, str], set[int]] = {}
    notes, best = [], {}
    for name, record in listeners:
        for sid, answer in record["answers"].items():
            for slot, form in key["sentences"][sid]["slots"].items():
                given = answer["slots"][slot]
                wrong[(name, sid, form)] = set(given.get("words") or [])
                if (given.get("note") or "").strip():
                    notes.append({"listener": record.get("listener"), "sentence": sid, "form": form,
                                  "note": given["note"].strip()})
            picked = key["sentences"][sid]["slots"].get(answer.get("best"), "no difference")
            best[picked] = best.get(picked, 0) + 1

    words_wrong = {f: sum(len(wrong[(n, s, f)]) for n, _ in listeners for s in items) for f in FORMS}
    cer: dict[str, list[float]] = {}
    for r in manifest["renders"]:
        cer.setdefault(r["condition"], []).append(r["cer"])
    mean_cer = {f: sum(v) / len(v) for f, v in cer.items()}
    held_out = {f: similarity["summary"][f]["held_out_mean"] for f in FORMS}

    exports = {}
    for f in EXPORTS:
        contrast = similarity["contrasts"][f]
        cer_rise = (mean_cer[f] - mean_cer["P"]) * 100
        flagged = cer_rise > 2 or contrast["wholly_below_zero"]
        fewer = words_wrong[f] < words_wrong["P"]
        exports[f] = {"words_marked_wrong": words_wrong[f], "fewer_than_P": fewer, "cer_rise_points": round(cer_rise, 2),
                      "similarity_minus_P": contrast["mean_low_high"], "flagged": flagged,
                      "qualifies": fewer and not flagged, "fewer_but_flagged": fewer and flagged}
    qualifying = [f for f in EXPORTS if exports[f]["qualifies"]]
    choice = min(qualifying, key=lambda f: (words_wrong[f], -held_out[f])) if qualifying else "P"

    trap_marks = {}
    for f in FORMS:
        counts = dict.fromkeys(TRAPS, 0)
        for n, _ in listeners:
            for sid, words in items.items():
                for t in TRAPS:
                    for trap in sentences[sid][t]:
                        at = [i for i, w in enumerate(words) if plain(w) == trap]
                        counts[t] += sum(1 for i in at[:1] if i in wrong[(n, sid, f)])
        trap_marks[f] = counts
    of_traps = {t: sum(len(s[t]) for s in sentences.values()) * len(listeners) for t in TRAPS}

    result = {"listeners": [r.get("listener") for _, r in listeners],
              "left_out_incomplete": [r.get("listener") for _, r in left_out],
              "words_marked_wrong": words_wrong, "of_words": sum(len(v) for v in items.values()) * len(listeners),
              "mean_cer": {f: round(v, 4) for f, v in mean_cer.items()}, "held_out_similarity": held_out,
              "exports": exports, "choice": choice,
              "read_beside_it": {"trap_words_marked": trap_marks, "of_trap_words": of_traps, "best": best,
                                 "notes": notes,
                                 "marked": {f"{n}:{s}:{f}": [items[s][i] for i in sorted(wrong[(n, s, f)])]
                                            for n, _ in listeners for s in items for f in FORMS if wrong[(n, s, f)]}}}
    (REVIEW / "reading.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("listeners", "words_marked_wrong", "mean_cer", "held_out_similarity",
                                             "exports", "choice")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
