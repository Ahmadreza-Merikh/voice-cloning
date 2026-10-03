#!/usr/bin/env python3
"""Write each step 3 sentence four ways. It renders nothing.

Registered before it ran (eval/runs/persian-voice-frontend-listening-v2.json).
The sentences are eval/manifests/frontend_listening_v2.json; for each:

* E    what the app sends OmniVoice today: its own text pipeline at its
       default settings (the normaliser keeping أ and ۀ, the tagger's ezafe)
* EH   + the first short vowel of each word that is one of HomoRich's
       homographs told apart by it, from Homo-GE2PE's reading of the sentence
* ES   + a sukun on a noun, adjective or number the tagger gives no ezafe
       when the next word is an adverb
* EHS  both

Every decision is kept per word in eval/runs/frontend-listening-v2/texts.json.

    .venv/bin/python scripts/build_frontend_texts_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("score_frontend_components",
                                              ROOT / "scripts" / "score_frontend_components.py")
sfc = importlib.util.module_from_spec(spec)
sys.modules["score_frontend_components"] = sfc
spec.loader.exec_module(sfc)

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.text import ezafe_tagger  # noqa: E402
from voiceclone.text.ezafe_fa import plan, tokens_of  # noqa: E402
from voiceclone.text.marks_fa import (  # noqa: E402
    PUNCT,
    bare,
    first_short_vowel,
    with_first_vowel,
    with_sukun,
)
from voiceclone.text.pipeline import TextPipeline  # noqa: E402

SENTENCES = ROOT / "eval" / "manifests" / "frontend_listening_v2.json"
HOMOGRAPHS = ROOT / "eval" / "manifests" / "homorich_homographs_v1.json"
OUT = ROOT / "eval" / "runs" / "frontend-listening-v2" / "texts.json"
SUKUN_HEADS = ("NOUN", "ADJ", "NUM")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_parts(text: str) -> tuple[list[str], list[list[int]], list[int]]:
    """The text's parts, its sentences, and the part index of every word in order."""
    parts, groups = plan(text)
    order = [i for g in groups for i in g if parts[i].strip(PUNCT)]
    return parts, groups, order


def readings_from_ge2pe(text: str, tokens: list[str]) -> list[str | None]:
    """Homo-GE2PE's reading of each word, without its ezafe, where a word maps to a token of its own."""
    ws = sfc.words(text)
    out: list[str | None] = [None] * len(ws)
    for wi, token in sfc.groups(ws, [sfc.from_ge2pe(t) for t in tokens]):
        if len(wi) == 1:
            base, _ = sfc.ezafe_split(ws[wi[0]], token)
            out[wi[0]] = base
    return out


def build(sentences: list[dict], homographs: set[str], tagger, g2p) -> list[dict]:
    pipeline = TextPipeline(AppConfig.load(None).text)
    built = []
    for s in sentences:
        prepared = pipeline.prepare(s["text"], engine_name="omnivoice")
        if len(prepared.chunks) != 1 or any(n.startswith("Ezafe not written") for n in prepared.notes):
            raise SystemExit(f"{s['id']}: the app did not give one chunk with its ezafe: {prepared.notes}")
        e_text = prepared.chunks[0].synth_text
        parts, groups, order = word_parts(prepared.normalized)
        e_parts, _, e_order = word_parts(e_text)
        if order != e_order:
            raise SystemExit(f"{s['id']}: the ezafe moved a word")

        tag_of: dict[int, str] = {}
        next_of: dict[int, str | None] = {}
        for g in groups:
            tokens, where = tokens_of(parts, g)
            tags = tagger.tag(tokens)
            words = [(i, t) for i, t in zip(where, tags, strict=True) if i is not None]
            for n, (i, t) in enumerate(words):
                tag_of[i] = t
                next_of[i] = words[n + 1][1] if n + 1 < len(words) else None
        result = g2p.phonemize(s["text"])
        tokens = list(result.compact_words)
        readings = readings_from_ge2pe(s["text"], tokens)
        if len(readings) != len(order):
            raise SystemExit(f"{s['id']}: {len(readings)} words for G2P, {len(order)} in the text")

        h_parts, s_parts, info = list(e_parts), list(e_parts), []
        for k, i in enumerate(order):
            raw = e_parts[i]
            core = raw.strip(PUNCT)
            start = raw.index(core)
            tag = tag_of[i]
            pos, nxt = tag.split(",")[0], (next_of[i] or "").split(",")[0] or None
            homograph = bare(core) in homographs
            vowel = first_short_vowel(core, readings[k]) if homograph else None
            sukun = pos in SUKUN_HEADS and ",EZ" not in tag and nxt == "ADV"
            if vowel:
                h_parts[i] = raw[:start] + with_first_vowel(core, readings[k]) + raw[start + len(core):]
            marked = with_sukun(core) if sukun else core
            if marked != core:
                s_parts[i] = raw[:start] + marked + raw[start + len(core):]
            info.append({"word": core, "tag": tag, "next": next_of[i], "ge2pe": readings[k], "homograph": homograph,
                         "first_vowel": vowel, "sukun": marked != core})
        both = list(h_parts)
        for i in order:
            if s_parts[i] != e_parts[i]:
                raw, core = h_parts[i], h_parts[i].strip(PUNCT)
                start = raw.index(core)
                both[i] = raw[:start] + with_sukun(core) + raw[start + len(core):]
        texts = {"E": e_text, "EH": "".join(h_parts), "ES": "".join(s_parts), "EHS": "".join(both)}
        built.append({"id": s["id"], "category": s["category"], "text": s["text"], "normalized": prepared.normalized,
                      "ge2pe": tokens, "words": info, "texts": texts, "app_notes": prepared.notes})
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    from voiceclone.text.g2p import PersianG2P

    config = AppConfig.load(None)
    export = Path(config.text.ezafe.export_dir).expanduser() / ezafe_tagger.REVISION[:12]
    record = json.loads(HOMOGRAPHS.read_text(encoding="utf-8"))
    homographs = {w for w, v in record["words"].items() if v["first_vowel_tells_apart"]}
    sentences = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    tagger = ezafe_tagger.ParsBertTagger(export)
    g2p = PersianG2P()
    built = build(sentences, homographs, tagger, g2p)
    weights = Path(g2p.repo_path) / "third_party" / "Homo-GE2PE-Persian" / "homo-ge2pe" / "model.safetensors"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    body = {"sentences_file": str(SENTENCES.relative_to(ROOT)), "sentences_sha256": sha256(SENTENCES),
            "homographs_sha256": sha256(HOMOGRAPHS), "code_commit": commit,
            "tagger": {"repo": ezafe_tagger.REPO, "revision": ezafe_tagger.REVISION},
            "g2p": {"backend": g2p.backend, "weights_sha256": sha256(weights)},
            "conditions": {"E": "the app today", "EH": "+ homograph first vowels", "ES": "+ sukun before an adverb",
                           "EHS": "+ both"},
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
