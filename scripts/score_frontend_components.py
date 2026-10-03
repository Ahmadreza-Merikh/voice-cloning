#!/usr/bin/env python3
"""Score Persian text front-end components against gold readings. It never runs a component.

Registered before any component ran on either set
(eval/runs/persian-voice-frontend-components-v1.json). The pronunciation review
found that OmniVoice's errors on plain script are mostly what the script leaves
out: the ezafe and short vowels. This scores the components that could supply
them, on plain script, two ways:

* ezafe - per word, does it take the linking -e/-ye after it
* readings - per word, its vowels in order (the vowel skeleton, ezafe left out),
  which is what diacritics would have to carry

Gold comes from SentenceBench (400 sentences with phoneme annotations; a word's
ezafe is read off the ending of its annotated pronunciation) and from the 14
sentences of the pronunciation review (eval/manifests/frontend_review14_gold_v1.json).
Predictions are the files scripts/measure_frontend_components.py writes.

    .venv/bin/python scripts/score_frontend_components.py eval/runs/evidence/frontend-components-v1 SentenceBench.csv [HomoRich.parquet]

With HomoRich (the training data named for Homo-GE2PE and the ALBERT model) it
also reports which SentenceBench sentences appear there.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_GOLD = ROOT / "eval" / "manifests" / "frontend_review14_gold_v1.json"
SEED = 20261003
RESAMPLES = 10000

HARAKAT = "ًٌٍَُِّْٰٕٔ"
ZWNJ = "‌"
PUNCT = "،؛؟!?.,:;«»\"'()[]{}…–—-*/"
VOWELS = "Aaeoiu"
SOUNDS = set("bptdkgqfszSZxCjmnlr")
LETTER = {"ب": "b", "پ": "p", "ت": "t", "ط": "t", "ث": "s", "س": "s", "ص": "s", "ج": "j", "چ": "C",
          "خ": "x", "د": "d", "ذ": "z", "ز": "z", "ض": "z", "ظ": "z", "ر": "r", "ژ": "Z", "ش": "S",
          "غ": "q", "ق": "q", "ف": "f", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n"}
GE2PE = {"/": "a", "a": "A", "$": "S", ";": "Z", "c": "C", "@": "?", "'": "?"}
MERGE, SKIP, LEADING_AND = 0.2, 0.6, 0.1


def clean(text: str) -> str:
    for a, b in (("ي", "ی"), ("ى", "ی"), ("ك", "ک"), ("ۀ", "هٔ"), ("ة", "ه"), ("ـ", ""),
                 (" ", " "), ("‏", ""), ("‎", "")):
        text = text.replace(a, b)
    return text


def words(sentence: str) -> list[str]:
    return [w for w in (t.strip(PUNCT) for t in clean(sentence).split()) if w]


def letters(word: str) -> str:
    return "".join(c for c in word if c not in HARAKAT and c != ZWNJ)


def from_ge2pe(token: str) -> str:
    return "".join(GE2PE.get(c, c) for c in token)


def ezafe_split(word: str, token: str) -> tuple[str, bool]:
    t = token.strip(".,،؟!")
    marked = t.endswith("1")
    t = t.rstrip("1")
    if t.endswith("-ye"):
        return t[:-3], True
    if t.endswith("-e"):
        return t[:-2], True
    last = letters(word)[-1:] if letters(word) else ""
    if last == "ه":
        if len(t) > 3 and t.endswith("eye"):
            return t[:-2], True
        if len(t) > 2 and t.endswith("he"):
            return t[:-1], True
        return t, marked
    if last in ("ا", "آ", "و", "ی") and len(t) > 2 and t.endswith("ye"):
        return t[:-2], True
    if len(t) > 1 and t.endswith("e"):
        return t[:-1], True
    return t, marked


def vowels(reading: str) -> str:
    return "".join(c for c in reading if c in VOWELS)


def distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def align(ws: list[str], ts: list[str]) -> list[tuple[list[int], list[int]]]:
    gw = ["".join(LETTER.get(c, "") for c in letters(w)) for w in ws]
    gt = ["".join(c for c in t if c in SOUNDS) for t in ts]
    n, m = len(ws), len(ts)
    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            here = cost[i][j]
            if here == float("inf"):
                continue
            moves = [(a, b, distance("".join(gw[i:i + a]), "".join(gt[j:j + b])) + extra
                      + (LEADING_AND if a > 1 and letters(ws[i]) == "و" else 0.0))
                     for a, b, extra in ((1, 1, 0.0), (2, 1, MERGE), (3, 1, 2 * MERGE), (1, 2, MERGE))
                     if i + a <= n and j + b <= m]
            if i < n:
                moves.append((1, 0, SKIP))
            if j < m:
                moves.append((0, 1, SKIP))
            for a, b, d in moves:
                if here + d < cost[i + a][j + b]:
                    cost[i + a][j + b] = here + d
                    back[i + a][j + b] = (a, b)
    groups = []
    i, j = n, m
    while i or j:
        a, b = back[i][j]
        groups.append((list(range(i - a, i)), list(range(j - b, j))))
        i, j = i - a, j - b
    return groups[::-1]


def groups(ws: list[str], ts: list[str]) -> list[tuple[list[int], str]]:
    out = []
    for wi, ti in align(ws, ts):
        if not wi or not ti:
            continue
        token = "".join(ts[k] for k in ti)
        if len(wi) == 2 and letters(ws[wi[1]]) == "و":
            token = token[:-2] if token.endswith("-o") else token[:-1] if token.endswith("o") else token
            out.append(([wi[0]], token))
        elif len(wi) == 3 and letters(ws[wi[1]]) == "و" and "-o-" in token:
            left, right = token.split("-o-", 1)
            out += [([wi[0]], left), ([wi[2]], right)]
        else:
            out.append((wi, token))
    return out


def gold_readings(ws: list[str], ts: list[str]) -> list[dict | None]:
    out: list[dict | None] = [None] * len(ws)
    for wi, token in groups(ws, ts):
        if len(wi) == 1:
            base, ez = ezafe_split(ws[wi[0]], token)
            out[wi[0]] = {"ezafe": ez, "vowels": [vowels(base)], "reading": token}
    return out


def predicted_readings(ws: list[str], ts: list[str], gold: list[dict | None]) -> list[dict]:
    out = [{"ezafe": False, "vowels": ""} for _ in ws]
    for wi, token in groups(ws, ts):
        base, ez = ezafe_split(ws[wi[-1]], token)
        skeleton = vowels(base)
        for k in wi[:-1]:
            share = len(gold[k]["vowels"][0]) + (1 if gold[k]["ezafe"] else 0) if gold[k] else 0
            out[k] = {"ezafe": False, "vowels": skeleton[:share]}
            skeleton = skeleton[share:]
        out[wi[-1]] = {"ezafe": ez, "vowels": skeleton}
    return out


def sentencebench(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for n, row in enumerate(csv.DictReader(handle)):
            ws = words(row["grapheme"])
            ts = [t for t in (x.strip(".,،؟!()") for x in row["phoneme"].split()) if t]
            gold = gold_readings(ws, ts)
            homograph = None
            if row.get("homograph word") and row.get("pronunciation"):
                target = letters(clean(row["homograph word"]).strip())
                where = [i for i, w in enumerate(ws) if letters(w) == target]
                if where:
                    homograph = {"index": where[0], "vowels": vowels(row["pronunciation"].strip())}
            out.append({"id": f"sb{n:03d}", "subset": row["dataset"], "text": row["grapheme"], "words": ws,
                        "gold": gold, "homograph": homograph})
    return out


def review14(path: Path = REVIEW_GOLD) -> list[dict]:
    out = []
    for s in json.loads(path.read_text(encoding="utf-8"))["sentences"]:
        ws = words(s["text"])
        if [w["w"] for w in s["words"]] != ws:
            raise SystemExit(f"{s['id']}: gold words do not match the text")
        gold = [{"ezafe": w["ezafe"], "vowels": [vowels(p) for p in w["pron"]], "reading": w["pron"][0],
                 "from": w["from"]} for w in s["words"]]
        out.append({"id": s["id"], "subset": "review14", "text": s["text"], "words": ws, "gold": gold,
                    "homograph": None, "heard": s.get("heard", [])})
    return out


def lexicon_vowels(options: list[str]) -> str | None:
    found = {vowels(o) for o in options}
    return found.pop() if len(found) == 1 else None


def per_word(sentence: dict, pred: dict) -> list[dict]:
    if pred.get("tokens") is not None:
        tokens = [from_ge2pe(t) for t in pred["tokens"]] if pred.get("scheme") == "ge2pe" else pred["tokens"]
        return predicted_readings(sentence["words"], tokens, sentence["gold"])
    if pred.get("lexicon") is not None:
        return [{"ezafe": None, "vowels": lexicon_vowels(o or [])} for o in pred["lexicon"]]
    return [{"ezafe": bool(e) if e is not None else False, "vowels": None} for e in pred["ezafe"]]


def tally(sentences: list[dict], preds: dict[str, dict], kind: str) -> dict:
    row = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "sentences": 0, "clean_sentences": 0, "per_sentence": []}
    if kind == "vowels":
        row = {"words": 0, "answered": 0, "correct": 0, "homographs": 0, "homographs_answered": 0,
               "homographs_correct": 0, "per_sentence": []}
    for s in sentences:
        got = per_word(s, preds[s["id"]]) if s["id"] in preds else None
        errors = scored = 0
        for i, g in enumerate(s["gold"]):
            if g is None or got is None:
                continue
            if kind == "ezafe":
                if g["ezafe"] is None or got[i]["ezafe"] is None:
                    continue
                scored += 1
                key = ("t" if got[i]["ezafe"] == g["ezafe"] else "f") + ("p" if got[i]["ezafe"] else "n")
                row[key] += 1
                errors += key[0] == "f"
            else:
                row["words"] += 1
                scored += 1
                if got[i]["vowels"] is not None:
                    row["answered"] += 1
                    ok = got[i]["vowels"] in g["vowels"]
                    row["correct"] += ok
                    errors += not ok
        if kind == "vowels" and s.get("homograph") and got is not None:
            h = s["homograph"]
            row["homographs"] += 1
            if got[h["index"]]["vowels"] is not None:
                row["homographs_answered"] += 1
                row["homographs_correct"] += got[h["index"]]["vowels"] == h["vowels"]
        if kind == "ezafe" and scored:
            row["sentences"] += 1
            row["clean_sentences"] += errors == 0
        row["per_sentence"].append((errors, scored))
    if kind == "ezafe":
        words_n = row["tp"] + row["fp"] + row["fn"] + row["tn"]
        p = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        r = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        row.update({"words": words_n, "errors": row["fp"] + row["fn"],
                    "error_rate": round((row["fp"] + row["fn"]) / words_n, 4) if words_n else None,
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0})
    else:
        row.update({"accuracy_answered": round(row["correct"] / row["answered"], 4) if row["answered"] else None,
                    "coverage": round(row["answered"] / row["words"], 4) if row["words"] else None,
                    "homograph_accuracy": round(row["homographs_correct"] / row["homographs_answered"], 4)
                    if row["homographs_answered"] else None})
    return row


def interval(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> dict:
    rng = random.Random(SEED)
    n = len(a)
    diffs = []
    for _ in range(RESAMPLES):
        pick = [rng.randrange(n) for _ in range(n)]
        words_n = sum(a[k][1] for k in pick)
        if words_n:
            diffs.append((sum(a[k][0] for k in pick) - sum(b[k][0] for k in pick)) / words_n)
    diffs.sort()
    point = (sum(x[0] for x in a) - sum(x[0] for x in b)) / max(sum(x[1] for x in a), 1)
    return {"difference": round(point, 4), "low": round(diffs[int(0.025 * len(diffs))], 4),
            "high": round(diffs[int(0.975 * len(diffs)) - 1], 4), "resamples": len(diffs)}


def heard(sentences: list[dict], preds: dict[str, dict]) -> list[dict]:
    out = []
    for s in sentences:
        got = per_word(s, preds[s["id"]]) if s["id"] in preds else None
        for h in s.get("heard", []):
            i, g = h["index"], s["gold"][h["index"]]
            if got is None:
                ok = None
            elif h["kind"] == "ezafe":
                ok = got[i]["ezafe"] == g["ezafe"] if got[i]["ezafe"] is not None else None
            else:
                ok = got[i]["vowels"] in g["vowels"] if got[i]["vowels"] is not None else None
            out.append({"sentence": s["id"], "word": s["words"][i], "kind": h["kind"], "right": ok})
    return out


def score(sets: dict[str, list[dict]], predictions: dict[str, dict[str, dict]]) -> dict:
    result = {}
    for name, sentences in sets.items():
        table = {"ezafe": {}, "vowels": {}, "heard": {}}
        never = {s["id"]: {"ezafe": [False] * len(s["words"])} for s in sentences}
        for component, by_set in {"never": {name: never}, **predictions}.items():
            preds = by_set.get(name)
            if not preds:
                continue
            sample = next(iter(preds.values()))
            if sample.get("lexicon") is None:
                table["ezafe"][component] = tally(sentences, preds, "ezafe")
            if sample.get("tokens") is not None or sample.get("lexicon") is not None:
                table["vowels"][component] = tally(sentences, preds, "vowels")
            if name == "review14" and component != "never":
                table["heard"][component] = heard(sentences, preds)
        if "kaamel" in table["vowels"] and "homo-ge2pe" in table["vowels"]:
            joined = {}
            for s in sentences:
                lex = per_word(s, predictions["kaamel"][name][s["id"]])
                ge = per_word(s, predictions["homo-ge2pe"][name][s["id"]])
                joined[s["id"]] = {"lexicon": [[x["vowels"]] if x["vowels"] is not None else [g["vowels"] or ""]
                                               for x, g in zip(lex, ge, strict=True)]}
            table["vowels"]["kaamel-then-homo-ge2pe"] = tally(sentences, joined, "vowels")
        subsets = sorted({s["subset"] for s in sentences})
        if len(subsets) > 1:
            table["ezafe_by_subset"] = {}
            for component, by_set in {"never": {name: never}, **predictions}.items():
                preds = by_set.get(name)
                if preds and next(iter(preds.values())).get("lexicon") is None:
                    table["ezafe_by_subset"][component] = {}
                    for sub in subsets:
                        row = tally([s for s in sentences if s["subset"] == sub], preds, "ezafe")
                        table["ezafe_by_subset"][component][sub] = {k: row[k] for k in ("words", "errors", "error_rate", "f1")}
        ranked = sorted((c for c in table["ezafe"] if c != "never"), key=lambda c: table["ezafe"][c]["error_rate"])
        if len(ranked) >= 2:
            best, second = ranked[0], ranked[1]
            table["ezafe_best_minus_runner_up"] = {"best": best, "runner_up": second, **interval(
                table["ezafe"][best]["per_sentence"], table["ezafe"][second]["per_sentence"])}
        for kind in ("ezafe", "vowels"):
            for row in table[kind].values():
                row.pop("per_sentence", None)
        result[name] = table
    return result


def squash(text: str) -> str:
    return "".join(c for c in clean(text) if c.isalpha() and c not in HARAKAT)


def overlap(sentences: list[dict], others: list[str], width: int = 12) -> dict:
    keys = {s["id"]: squash(s["text"]) for s in sentences}
    shingles: dict[str, set[str]] = {}
    for sid, k in keys.items():
        for i in range(max(len(k) - width + 1, 1)):
            shingles.setdefault(k[i:i + width], set()).add(sid)
    hits: dict[str, str] = {}
    for text in others:
        t = squash(text)
        seen: set[str] = set()
        for i in range(max(len(t) - width + 1, 1)):
            seen |= shingles.get(t[i:i + width], set())
        for sid in seen - hits.keys():
            short, longer = sorted((keys[sid], t), key=len)
            if keys[sid] == t:
                hits[sid] = "equal"
            elif len(short) >= 20 and len(short) >= len(longer) / 2 and short in longer:
                hits[sid] = "contained"
    by: dict[str, dict] = {}
    for s in sentences:
        row = by.setdefault(s["subset"], {"sentences": 0, "equal": 0, "contained": 0})
        row["sentences"] += 1
        if s["id"] in hits:
            row[hits[s["id"]]] += 1
    return by


def load_predictions(folder: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(folder.glob("predictions-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault(data["component"], {})[data["set"]] = {s["id"]: {**s, "scheme": data.get("scheme")}
                                                               for s in data["sentences"]}
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        raise SystemExit(__doc__)
    sets = {"sentencebench": sentencebench(Path(args[1])), "review14": review14()}
    result = score(sets, load_predictions(Path(args[0])))
    result["gold"] = {name: {"sentences": len(ss), "words": sum(len(s["words"]) for s in ss),
                             "scorable": sum(1 for s in ss for g in s["gold"] if g),
                             "ezafe_words": sum(1 for s in ss for g in s["gold"] if g and g["ezafe"]),
                             "homographs_located": sum(1 for s in ss if s.get("homograph"))}
                      for name, ss in sets.items()}
    if len(args) == 3:
        import pandas
        others = pandas.read_parquet(args[2], columns=["Grapheme"])["Grapheme"].astype(str).tolist()
        result["overlap_with_homorich"] = {"sentences_checked_against": len(others),
                                           "rule": "letters only; equal, or one inside the other with at least 20 letters and half the longer",
                                           "by_subset": overlap(sets["sentencebench"], others)}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
