#!/usr/bin/env python3
"""Read the XTTS check through its registered rule.

eval/runs/persian-voice-xtts-check-v1.json fixed, before anyone listened:
ParsVoice-XTTS (X) replaces OmniVoice as the app's Persian engine if it has
fewer words marked wrong than the app (O) and is not flagged - flagged if its
mean CER is more than 2 points above O's (both scored on this Mac by one
recogniser) or if its paired held-out similarity difference against O has a
95% interval wholly below zero. A tie keeps O. Fewer words wrong but flagged
goes to the user. Words marked wrong are summed over every listener who
answered all 10 sentences.

    .venv/bin/python scripts/read_xtts_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "eval" / "runs" / "evidence" / "xtts-review-v1"
RUN = ROOT / "eval" / "runs" / "evidence" / "xtts-check-v1"
MANIFEST = ROOT / "eval" / "runs" / "xtts-check-v1" / "manifest.json"
FORMS = ("O", "X")


def slot_done(given: dict | None) -> bool:
    return bool(given) and (given.get("ok") is True or bool(given.get("words")))


def complete(answers: dict, key: dict) -> bool:
    return all(sid in answers and answers[sid].get("best")
               and all(slot_done(answers[sid].get("slots", {}).get(slot)) for slot in sentence["slots"])
               for sid, sentence in key["sentences"].items())


def main() -> int:
    key = json.loads((REVIEW / "key.json").read_text(encoding="utf-8"))
    items = {i["id"]: i["words"] for i in json.loads((REVIEW / "items.json").read_text(encoding="utf-8"))}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    similarity = json.loads((RUN / "similarity.json").read_text(encoding="utf-8"))

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
    cer_rise = (mean_cer["X"] - mean_cer["O"]) * 100
    contrast = similarity["contrasts"]["X"]
    flagged = cer_rise > 2 or contrast["wholly_below_zero"]
    fewer = words_wrong["X"] < words_wrong["O"]
    result = {"listeners": [r.get("listener") for _, r in listeners],
              "left_out_incomplete": [r.get("listener") for _, r in left_out],
              "words_marked_wrong": words_wrong, "of_words": sum(len(v) for v in items.values()) * len(listeners),
              "mean_cer": {f: round(v, 4) for f, v in mean_cer.items()}, "cer_rise_points": round(cer_rise, 2),
              "held_out_similarity": {f: similarity["summary"][f]["held_out_mean"] for f in FORMS},
              "similarity_X_minus_O": contrast["mean_low_high"], "flagged": flagged,
              "adopt_X": fewer and not flagged, "fewer_but_flagged": fewer and flagged,
              "read_beside_it": {"best": best, "notes": notes,
                                 "marked": {f"{n}:{s}:{f}": [items[s][i] for i in sorted(wrong[(n, s, f)])]
                                            for n, _ in listeners for s in items for f in FORMS
                                            if wrong[(n, s, f)]}}}
    (REVIEW / "reading.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "read_beside_it"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
