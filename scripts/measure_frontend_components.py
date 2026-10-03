#!/usr/bin/env python3
"""Run one Persian text front-end component over a gold set and save what it says. It never scores.

Registered before any component ran on either set
(eval/runs/persian-voice-frontend-components-v1.json). Every component reads
plain script, as the app would hand it text, cleaned the same way for all
(Arabic letter forms to Persian, tatweel dropped, spacing left alone so the
words stay the gold's words):

* hazm-crf     hazm's CRF tagger (roshan-research/hazm-postagger); a tag with ,EZ is an ezafe
* hazm-spacy   hazm's ParsBERT tagger (roshan-research/hazm-parsbert-postagger), asked for
               its full tags, since the universal ones drop the ,EZ
* albert       abreza/persian-ezafe-albert; the label on each word's first piece, as its authors read it
* homo-ge2pe   Homo-GE2PE through the app's own bridge (voiceclone.text.g2p); its words, unaligned
* kaamel       KaamelDict looked up word by word; every pronunciation it lists

The taggers get punctuation as separate tokens, ALBERT gets it attached as its
authors fed it, Homo-GE2PE gets the sentence. hazm needs Python 3.12 and the
spaCy transformer layer an older transformers, so the two taggers run in
~/.cache/persian-voice-cloning/frontend/.venv and the rest in the project's .venv.

    ~/.cache/persian-voice-cloning/frontend/.venv/bin/python scripts/measure_frontend_components.py hazm-crf sentencebench OUT --sentencebench SentenceBench.csv
    .venv/bin/python scripts/measure_frontend_components.py homo-ge2pe review14 OUT
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_frontend_components", ROOT / "scripts" / "score_frontend_components.py")
sfc = importlib.util.module_from_spec(spec)
sys.modules["score_frontend_components"] = sfc
spec.loader.exec_module(sfc)

REVISIONS = {
    "hazm-crf": ("roshan-research/hazm-postagger", "0a50b839a732ae10d600b69de67349079c3268e9"),
    "hazm-spacy": ("roshan-research/hazm-parsbert-postagger", "eba1d2b30cc48206c0144b36d2b663652d44faf9"),
    "albert": ("abreza/persian-ezafe-albert", "e6614371c6cd93fca2ad2e192a978613a14d92d6"),
}
PACKAGES = ("hazm", "spacy", "spacy-transformers", "transformers", "torch", "tokenizers", "huggingface-hub",
            "python-crfsuite", "sentencepiece")


def tagger_tokens(text: str, n_words: int) -> tuple[list[str], list[int | None]]:
    tokens: list[str] = []
    where: list[int | None] = []
    k = 0
    for raw in sfc.clean(text).split():
        core = raw.strip(sfc.PUNCT)
        if not core:
            tokens += list(raw)
            where += [None] * len(raw)
            continue
        start = raw.index(core)
        lead, trail = raw[:start], raw[start + len(core):]
        tokens += list(lead) + [core] + list(trail)
        where += [None] * len(lead) + [k] + [None] * len(trail)
        k += 1
    if k != n_words:
        raise SystemExit(f"tokens do not match the gold words: {text}")
    return tokens, where


def ezafe_from_tags(tags: list[str], where: list[int | None], n_words: int) -> list[bool]:
    out = [False] * n_words
    for tag, k in zip(tags, where, strict=True):
        if k is not None:
            out[k] = ",EZ" in tag
    return out


def run_hazm_crf(sentences: list[dict]) -> tuple[list[dict], dict]:
    from hazm import POSTagger
    from huggingface_hub import hf_hub_download

    repo, sha = REVISIONS["hazm-crf"]
    tagger = POSTagger(model=hf_hub_download(repo, "pos_tagger.model", revision=sha))
    out = []
    for s in sentences:
        tokens, where = tagger_tokens(s["text"], len(s["words"]))
        tags = [t for _, t in tagger.tag(tokens)]
        out.append({"id": s["id"], "ezafe": ezafe_from_tags(tags, where, len(s["words"]))})
    return out, {"repo": repo, "file": "pos_tagger.model", "revision": sha}


def run_hazm_spacy(sentences: list[dict]) -> tuple[list[dict], dict]:
    from hazm import SpacyPOSTagger
    from huggingface_hub import snapshot_download

    repo, sha = REVISIONS["hazm-spacy"]
    tagger = SpacyPOSTagger(model_path=snapshot_download(repo, revision=sha))
    out = []
    for s in sentences:
        tokens, where = tagger_tokens(s["text"], len(s["words"]))
        tags = [t for _, t in tagger.tag(tokens, universal_tag=False)]
        out.append({"id": s["id"], "ezafe": ezafe_from_tags(tags, where, len(s["words"]))})
    return out, {"repo": repo, "revision": sha}


def run_albert(sentences: list[dict]) -> tuple[list[dict], dict]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    repo, sha = REVISIONS["albert"]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha)
    model = AutoModelForTokenClassification.from_pretrained(repo, revision=sha).eval()
    out = []
    for s in sentences:
        raws = sfc.clean(s["text"]).split()
        enc = tokenizer(raws, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            labels = model(**enc).logits.argmax(-1)[0].tolist()
        first: dict[int, int] = {}
        for i, w in enumerate(enc.word_ids()):
            if w is not None and w not in first:
                first[w] = labels[i]
        ezafe, k = [False] * len(s["words"]), 0
        for j, raw in enumerate(raws):
            if raw.strip(sfc.PUNCT):
                ezafe[k] = bool(first.get(j, 0))
                k += 1
        out.append({"id": s["id"], "ezafe": ezafe, "truncated": len(first) < len(raws)})
    return out, {"repo": repo, "revision": sha, "max_length": 128}


def run_homo_ge2pe(sentences: list[dict]) -> tuple[list[dict], dict]:
    from voiceclone.text.g2p import PersianG2P

    g2p = PersianG2P()
    out = []
    for s in sentences:
        result = g2p.phonemize(s["text"])
        out.append({"id": s["id"], "tokens": list(result.compact_words), "backend": result.backend})
    weights = Path(g2p.repo_path) / "third_party" / "Homo-GE2PE-Persian" / "homo-ge2pe" / "model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    return out, {"repo": str(g2p.repo_path), "backend": g2p.backend, "weights_sha256": digest,
                 "path": "voiceclone.text.g2p.PersianG2P().phonemize - normalize_for_g2p, then GE2PE with its rules"}


def run_kaamel(sentences: list[dict], path: Path) -> tuple[list[dict], dict]:
    def key(word: str) -> str:
        bare = "".join(c for c in sfc.clean(word) if c not in sfc.HARAKAT)
        while sfc.ZWNJ * 2 in bare:
            bare = bare.replace(sfc.ZWNJ * 2, sfc.ZWNJ)
        return bare

    lexicon: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lexicon.setdefault(key(row["grapheme"]), []).extend("".join(p) for p in ast.literal_eval(row["phoneme"]))
    out = [{"id": s["id"], "lexicon": [sorted(set(lexicon.get(key(w), []))) for w in s["words"]]} for s in sentences]
    return out, {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "entries": len(lexicon)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("component", choices=["hazm-crf", "hazm-spacy", "albert", "homo-ge2pe", "kaamel"])
    parser.add_argument("set", choices=["sentencebench", "review14"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--sentencebench", type=Path)
    parser.add_argument("--kaamel", type=Path)
    args = parser.parse_args(argv)

    sentences = sfc.sentencebench(args.sentencebench) if args.set == "sentencebench" else sfc.review14()
    started = time.perf_counter()
    if args.component == "kaamel":
        predictions, source = run_kaamel(sentences, args.kaamel)
    else:
        predictions, source = {"hazm-crf": run_hazm_crf, "hazm-spacy": run_hazm_spacy, "albert": run_albert,
                               "homo-ge2pe": run_homo_ge2pe}[args.component](sentences)
    seconds = time.perf_counter() - started
    versions = {}
    for name in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    body = {"component": args.component, "set": args.set, "scheme": "ge2pe" if args.component == "homo-ge2pe" else None,
            "source": source, "python": platform.python_version(), "packages": versions,
            "seconds": round(seconds, 1), "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
            "sentences": predictions}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"predictions-{args.component}-{args.set}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.component} on {args.set}: {len(predictions)} sentences, {seconds:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
