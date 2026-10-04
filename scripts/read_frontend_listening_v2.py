#!/usr/bin/env python3
"""Read step 3's listening through its registered rules.

eval/runs/persian-voice-frontend-listening-v2.json fixed, before anyone
listened: each fix is kept if the words marked wrong in its two forms together
are fewer than in the two without it; a fix is flagged if its two forms' mean
CER is more than 2 points above the two without it, or if its paired held-out
similarity difference has a 95% interval wholly below zero. This applies them,
counts the 23 homographs per form, and splits each fix's changes into words it
marked and words it left alone (the rendering re-rolls the whole sentence).

    .venv/bin/python scripts/read_frontend_listening_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.text.marks_fa import PUNCT, bare  # noqa: E402

REVIEW = ROOT / "eval" / "runs" / "evidence" / "frontend-review-v2"
RUN = ROOT / "eval" / "runs" / "evidence" / "frontend-listening-v2"
TEXTS = ROOT / "eval" / "runs" / "frontend-listening-v2" / "texts.json"
MANIFEST = ROOT / "eval" / "runs" / "frontend-listening-v2" / "manifest.json"
SENTENCES = ROOT / "eval" / "manifests" / "frontend_listening_v2.json"
OUT = REVIEW / "reading.json"
FIXES = {"H": (("EH", "EHS"), ("E", "ES")), "S": (("ES", "EHS"), ("E", "EH"))}


def main() -> int:
    key = json.loads((REVIEW / "key.json").read_text(encoding="utf-8"))
    items = {i["id"]: i["words"] for i in json.loads((REVIEW / "items.json").read_text(encoding="utf-8"))}
    answers = json.loads((REVIEW / "answers-listener-1.json").read_text(encoding="utf-8"))["answers"]
    texts = {s["id"]: s for s in json.loads(TEXTS.read_text(encoding="utf-8"))["sentences"]}
    expected = {s["id"]: s["homographs_expected"] for s in json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    similarity = json.loads((RUN / "similarity.json").read_text(encoding="utf-8"))

    wrong: dict[tuple[str, str], set[int]] = {}
    notes: list[dict] = []
    best: dict[str, int] = {}
    for sid, answer in answers.items():
        for slot, condition in key["sentences"][sid]["slots"].items():
            given = answer["slots"][slot]
            wrong[(sid, condition)] = set(given.get("words") or [])
            if (given.get("note") or "").strip():
                notes.append({"sentence": sid, "form": condition, "note": given["note"].strip()})
        picked = key["sentences"][sid]["slots"].get(answer.get("best"), "no difference")
        best[picked] = best.get(picked, 0) + 1
    forms = ("E", "EH", "ES", "EHS")
    words_wrong = {c: sum(len(wrong[(s, c)]) for s in items) for c in forms}

    # The 23 homographs, by position: the n-th occurrence of each expected word.
    homograph_at: dict[str, list[tuple[int, str, str]]] = {}
    for sid, wanted in expected.items():
        seen: dict[str, int] = {}
        places = []
        for i, w in enumerate(items[sid]):
            b = bare(w.strip(PUNCT))
            seen[b] = seen.get(b, 0) + 1
            for label, reading in wanted.items():
                word, _, nth = label.partition("#")
                if b == word and seen[b] == int(nth or 1):
                    places.append((i, label, reading))
        homograph_at[sid] = places
    homographs = sum(len(v) for v in homograph_at.values())
    homographs_wrong = {c: sum(1 for s, places in homograph_at.items() for i, _, _ in places if i in wrong[(s, c)])
                        for c in forms}

    cer = {}
    for r in manifest["renders"]:
        cer.setdefault(r["condition"], []).append(r["cer"])
    mean_cer = {c: sum(v) / len(v) for c, v in cer.items()}

    decisions = {}
    for fix, (with_it, without) in FIXES.items():
        a = sum(words_wrong[c] for c in with_it)
        b = sum(words_wrong[c] for c in without)
        cer_rise = (sum(mean_cer[c] for c in with_it) - sum(mean_cer[c] for c in without)) / 2 * 100
        sim = similarity["contrasts"][fix]
        flagged = cer_rise > 2 or sim["wholly_below_zero"]
        # Which words moved, and whether the fix had marked them.
        local_fixed = local_broke = other_fixed = other_broke = 0
        moved = []
        for sid in items:
            parts_with = {c: texts[sid]["texts"][c].split(" ") for c in forms}
            for x, y in zip(without, with_it, strict=True):
                for i in range(len(items[sid])):
                    bx, by = i in wrong[(sid, x)], i in wrong[(sid, y)]
                    if bx == by:
                        continue
                    marked = parts_with[x][i] != parts_with[y][i]
                    kind = ("its own word" if marked else "a word it left") + (", fixed" if bx else ", broke")
                    moved.append({"sentence": sid, "word": items[sid][i], "from": x, "to": y, "kind": kind})
                    if marked and bx:
                        local_fixed += 1
                    elif marked:
                        local_broke += 1
                    elif bx:
                        other_fixed += 1
                    else:
                        other_broke += 1
        decisions[fix] = {"words_wrong_with": a, "words_wrong_without": b, "kept": a < b,
                          "cer_rise_points": round(cer_rise, 2),
                          "similarity_minus": sim["mean_low_high"], "flagged": flagged,
                          "adopted": a < b and not flagged,
                          "its_own_words": {"fixed": local_fixed, "broke": local_broke},
                          "words_it_left": {"fixed": other_fixed, "broke": other_broke}, "moved": moved}
    result = {"words_wrong": words_wrong, "of_words": sum(len(v) for v in items.values()), "best": best,
              "homographs": {"count": homographs, "marked_wrong": homographs_wrong},
              "mean_cer": {c: round(v, 4) for c, v in mean_cer.items()},
              "similarity": {c: similarity["summary"][c]["held_out_mean"] for c in forms},
              "decisions": decisions, "notes": notes,
              "marked": {f"{s}:{c}": [items[s][i] for i in sorted(wrong[(s, c)])] for s in items for c in forms
                         if wrong[(s, c)]}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("marked",)}, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
