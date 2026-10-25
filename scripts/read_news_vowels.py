#!/usr/bin/env python3
"""Read the news vowels check through its registered rules.

eval/runs/persian-voice-news-vowels-v1.json fixed, before anyone listened:

* V: the app writes the first short vowel of every word if PV + AV have fewer
  words marked wrong than PB + AB and V is not flagged - flagged if the mean
  CER of PV and AV is more than 2 points above that of PB and AB, or if the
  held-out similarity contrast V = PV + AV - PB - AB has a 95% interval wholly
  below zero. A tie does not adopt it.
* weights: the app goes back to the published weights if PB + PV have fewer
  words marked wrong than AB + AV, and otherwise keeps A1.

Words marked wrong are summed over every listener who answered all 8
sentences. Beside the rules, deciding nothing: V's effect under each set of
weights, per word what V fixed and broke - split by whether V put a mark on
the word - V's marks against the hand-written first vowels, the picked
readings and the notes.

    .venv/bin/python scripts/read_news_vowels.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "eval" / "runs" / "evidence" / "news-review-v1"
RUN = ROOT / "eval" / "runs" / "evidence" / "news-vowels-v1"
MANIFEST = ROOT / "eval" / "runs" / "news-vowels-v1" / "manifest.json"
TEXTS = ROOT / "eval" / "runs" / "news-vowels-v1" / "texts.json"
FORMS = ("PB", "PV", "AB", "AV")


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
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

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

    with_v, without_v = words_wrong["PV"] + words_wrong["AV"], words_wrong["PB"] + words_wrong["AB"]
    cer_rise = ((mean_cer["PV"] + mean_cer["AV"]) - (mean_cer["PB"] + mean_cer["AB"])) / 2 * 100
    contrast = similarity["contrasts"]["V"]
    flagged = cer_rise > 2 or contrast["wholly_below_zero"]
    published, adapted = words_wrong["PB"] + words_wrong["PV"], words_wrong["AB"] + words_wrong["AV"]

    changed = {}
    for s in texts["sentences"]:
        b, v = s["texts"]["PB"].split(), s["texts"]["PV"].split()
        changed[s["id"]] = ({i for i, (x, y) in enumerate(zip(b, v, strict=True)) if x != y}
                            if len(b) == len(v) == len(items[s["id"]]) else None)
    per_word = {}
    for weights in ("P", "A"):
        tally = {"fixed_marked": 0, "broke_marked": 0, "fixed_unmarked": 0, "broke_unmarked": 0}
        for n, _ in listeners:
            for sid in items:
                before, after = wrong[(n, sid, weights + "B")], wrong[(n, sid, weights + "V")]
                marked = changed[sid] or set()
                for i in before - after:
                    tally["fixed_marked" if i in marked else "fixed_unmarked"] += 1
                for i in after - before:
                    tally["broke_marked" if i in marked else "broke_unmarked"] += 1
        per_word[weights] = tally

    result = {"listeners": [r.get("listener") for _, r in listeners],
              "left_out_incomplete": [r.get("listener") for _, r in left_out],
              "words_marked_wrong": words_wrong, "of_words": sum(len(v) for v in items.values()) * len(listeners),
              "mean_cer": {f: round(v, 4) for f, v in mean_cer.items()},
              "V": {"with_V": with_v, "without_V": without_v, "fewer": with_v < without_v,
                    "cer_rise_points": round(cer_rise, 2), "similarity_V": contrast["mean_low_high"],
                    "flagged": flagged, "adopt": with_v < without_v and not flagged,
                    "fewer_but_flagged": with_v < without_v and flagged},
              "weights": {"published": published, "adapted": adapted,
                          "choice": "published" if published < adapted else "adapted (A1, kept)"},
              "read_beside_it": {
                  "V_under_each_weights": {"P": words_wrong["PB"] - words_wrong["PV"],
                                           "A": words_wrong["AB"] - words_wrong["AV"]},
                  "per_word": per_word,
                  "V_marks_against_gold": texts.get("first_vowels_against_gold", {}).get("tally"),
                  "best": best, "notes": notes,
                  "marked": {f"{n}:{s}:{f}": [items[s][i] for i in sorted(wrong[(n, s, f)])]
                             for n, _ in listeners for s in items for f in FORMS if wrong[(n, s, f)]}}}
    (REVIEW / "reading.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("listeners", "words_marked_wrong", "mean_cer", "V", "weights")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
