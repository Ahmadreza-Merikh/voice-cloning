#!/usr/bin/env python3
"""Write each listening sentence four ways, from the front-end's decisions. It renders nothing.

Registered before it ran (eval/runs/persian-voice-frontend-listening-v1.json).
The sentences are eval/manifests/frontend_listening_v1.json; for each:

* A    the plain sentence - what the app sends today
* E    + the ezafe hazm's ParsBERT tagger predicts, written in
* EN   + a sukun closing a noun, adjective or number the tagger gives no ezafe,
       when a content word follows (where OmniVoice added spurious ones)
* ENV  + the first short vowel of each noun and adjective, where every
       KaamelDict reading of the word agrees on it

The marks are placed by voiceclone.text.marks_fa; this script only gathers the
decisions. It runs where hazm does:

    ~/.cache/persian-voice-cloning/frontend/.venv/bin/python scripts/build_frontend_texts.py \\
        --kaamel ~/.cache/persian-voice-cloning/eval-data/frontend/KaamelDict.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = ROOT / "eval" / "manifests" / "frontend_listening_v1.json"
OUT = ROOT / "eval" / "runs" / "frontend-listening-v1" / "texts.json"
TAGGER = ("roshan-research/hazm-parsbert-postagger", "eba1d2b30cc48206c0144b36d2b663652d44faf9")
CLITICS = ("هایم", "هایت", "هایش", "هایمان", "هایتان", "هایشان", "های", "ها", "مان", "تان", "شان", "ام", "ات", "اش",
           "م", "ت", "ش")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


marks = load("marks_fa", ROOT / "src" / "voiceclone" / "text" / "marks_fa.py")
mfc = load("measure_frontend_components", ROOT / "scripts" / "measure_frontend_components.py")


def key(word: str) -> str:
    bare = "".join(c for c in mfc.sfc.clean(word) if c not in marks.HARAKAT)
    while marks.ZWNJ * 2 in bare:
        bare = bare.replace(marks.ZWNJ * 2, marks.ZWNJ)
    return bare.strip(marks.ZWNJ)


def lexicon(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(key(row["grapheme"]), []).extend("".join(p) for p in ast.literal_eval(row["phoneme"]))
    return out


def readings(word: str, lex: dict[str, list[str]]) -> tuple[str, list[str]]:
    k = key(word)
    if k in lex:
        return k, lex[k]
    stem = k.split(marks.ZWNJ)[0]
    if stem != k and stem in lex:
        return stem, lex[stem]
    for clitic in CLITICS:
        if stem.endswith(clitic) and len(stem) - len(clitic) >= 2 and stem[:-len(clitic)] in lex:
            return stem[:-len(clitic)], lex[stem[:-len(clitic)]]
    return k, []


def agreed(word: str, stem: str, options: list[str]) -> str | None:
    """A reading to place marks from, if every listed reading agrees on what the marks need."""
    if not options:
        return None
    firsts = {marks.first_short_vowel(stem, o) for o in options}
    finals = {o.rstrip("-").endswith("h") for o in options}
    return options[0] if len(firsts) == 1 and len(finals) == 1 else None


def build(sentences: list[dict], lex: dict[str, list[str]], tagger) -> list[dict]:
    out = []
    for s in sentences:
        ws = mfc.sfc.words(s["text"])
        tokens, where = mfc.tagger_tokens(s["text"], len(ws))
        tags = [t for _, t in tagger.tag(tokens, universal_tag=False)]
        info = []
        for k, word in enumerate(ws):
            at = where.index(k)
            tag = tags[at]
            nxt = tags[at + 1].split(",")[0] if at + 1 < len(tags) else None
            stem, options = readings(word, lex)
            info.append({"word": word, "tag": tag, "pos": tag.split(",")[0], "ezafe": ",EZ" in tag, "next_pos": nxt,
                         "lexicon_key": stem, "lexicon": sorted(set(options)), "reading": agreed(word, stem, options)})
        words = [marks.Word(i["word"], ezafe=i["ezafe"], pos=i["pos"], reading=i["reading"], next_pos=i["next_pos"])
                 for i in info]
        texts = {"A": s["text"],
                 "E": marks.rebuild(s["text"], marks.mark_words(words, ezafe=True)),
                 "EN": marks.rebuild(s["text"], marks.mark_words(words, ezafe=True, sukun=True)),
                 "ENV": marks.rebuild(s["text"], marks.mark_words(words, ezafe=True, sukun=True, vowels=True))}
        out.append({"id": s["id"], "category": s["category"], "text": s["text"], "words": info, "texts": texts})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaamel", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    from hazm import SpacyPOSTagger
    from huggingface_hub import snapshot_download

    tagger = SpacyPOSTagger(model_path=snapshot_download(TAGGER[0], revision=TAGGER[1]))
    sentences = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    built = build(sentences, lexicon(args.kaamel), tagger)
    body = {"sentences_file": str(SENTENCES.relative_to(ROOT)),
            "sentences_sha256": hashlib.sha256(SENTENCES.read_bytes()).hexdigest(),
            "tagger": {"repo": TAGGER[0], "revision": TAGGER[1]},
            "kaamel_sha256": hashlib.sha256(args.kaamel.read_bytes()).hexdigest(),
            "marks_sha256": hashlib.sha256((ROOT / "src/voiceclone/text/marks_fa.py").read_bytes()).hexdigest(),
            "packages": {p: importlib.metadata.version(p) for p in ("hazm", "spacy", "spacy-transformers", "transformers")},
            "sentences": built}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for s in built:
        print(s["id"])
        for c, t in s["texts"].items():
            print(f"  {c:3s} {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
