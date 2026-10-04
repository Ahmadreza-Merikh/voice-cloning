#!/usr/bin/env python3
"""Read the روی check through its registered rule.

eval/runs/persian-voice-rui-check-v1.json fixed, before anyone listened: the
app stops writing the kasra on روی if R0 has روی marked wrong fewer times than
R1 and its words marked wrong in all are not more; a tie keeps it. R0 is
flagged if its mean CER is more than 2 points above R1's or its paired
held-out similarity difference has a 95% interval wholly below zero.

    .venv/bin/python scripts/read_rui_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.text.marks_fa import PUNCT, bare  # noqa: E402

REVIEW = ROOT / "eval" / "runs" / "evidence" / "rui-review-v1"
RUN = ROOT / "eval" / "runs" / "evidence" / "rui-check-v1"
MANIFEST = ROOT / "eval" / "runs" / "rui-check-v1" / "manifest.json"
RUI = "روی"


def main() -> int:
    key = json.loads((REVIEW / "key.json").read_text(encoding="utf-8"))
    items = {i["id"]: i["words"] for i in json.loads((REVIEW / "items.json").read_text(encoding="utf-8"))}
    answers = json.loads((REVIEW / "answers-listener-1.json").read_text(encoding="utf-8"))["answers"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    similarity = json.loads((RUN / "similarity.json").read_text(encoding="utf-8"))

    wrong, notes, best = {}, [], {}
    for sid, answer in answers.items():
        for slot, form in key["sentences"][sid]["slots"].items():
            given = answer["slots"][slot]
            wrong[(sid, form)] = set(given.get("words") or [])
            if (given.get("note") or "").strip():
                notes.append({"sentence": sid, "form": form, "note": given["note"].strip()})
        picked = key["sentences"][sid]["slots"].get(answer.get("best"), "no difference")
        best[picked] = best.get(picked, 0) + 1
    forms = ("R1", "R0")
    rui_at = {sid: [i for i, w in enumerate(words) if bare(w.strip(PUNCT)) == RUI] for sid, words in items.items()}
    rui_wrong = {f: sum(1 for s, places in rui_at.items() for i in places if i in wrong[(s, f)]) for f in forms}
    words_wrong = {f: sum(len(wrong[(s, f)]) for s in items) for f in forms}
    cer = {}
    for r in manifest["renders"]:
        cer.setdefault(r["condition"], []).append(r["cer"])
    mean_cer = {f: sum(v) / len(v) for f, v in cer.items()}
    contrast = similarity["contrasts"]["R0"]
    cer_rise = (mean_cer["R0"] - mean_cer["R1"]) * 100
    flagged = cer_rise > 2 or contrast["wholly_below_zero"]
    drop_kasra = rui_wrong["R0"] < rui_wrong["R1"] and words_wrong["R0"] <= words_wrong["R1"]
    result = {"rui": sum(len(v) for v in rui_at.values()), "rui_marked_wrong": rui_wrong,
              "words_marked_wrong": words_wrong, "of_words": sum(len(v) for v in items.values()), "best": best,
              "mean_cer": {f: round(v, 4) for f, v in mean_cer.items()}, "cer_rise_points": round(cer_rise, 2),
              "similarity_R0_minus_R1": contrast["mean_low_high"], "flagged": flagged,
              "drop_the_kasra_on_rui": drop_kasra and not flagged, "notes": notes,
              "marked": {f"{s}:{f}": [items[s][i] for i in sorted(wrong[(s, f)])] for s in items for f in forms
                         if wrong[(s, f)]}}
    (REVIEW / "reading.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
