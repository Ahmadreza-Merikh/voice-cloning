#!/usr/bin/env python3
"""Check that the tagger run without spaCy gives the spaCy pipeline's decisions.

Registered before it ran (eval/runs/persian-voice-ezafe-tagger-v1.json). The
reference is what the spaCy pipeline recorded - nothing heavy runs again:

* the ezafe decision of every word of SentenceBench (400 sentences) and the
  review's 14 sentences, from the step 1 predictions, on the same tokens
* every tag of the 12 listening sentences, from their texts.json

Writes the comparison, the export's file hashes and the light tagger's cost to
``eval/runs/evidence/ezafe-tagger-v1/check.json``.

    .venv/bin/python scripts/check_ezafe_tagger.py --sentencebench ~/.cache/persian-voice-cloning/eval-data/frontend/SentenceBench.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("measure_frontend_components",
                                              ROOT / "scripts" / "measure_frontend_components.py")
mfc = importlib.util.module_from_spec(spec)
sys.modules["measure_frontend_components"] = mfc
spec.loader.exec_module(mfc)

EVIDENCE = ROOT / "eval" / "runs" / "evidence" / "frontend-components-v1"
TEXTS = ROOT / "eval" / "runs" / "frontend-listening-v1" / "texts.json"
OUT = ROOT / "eval" / "runs" / "evidence" / "ezafe-tagger-v1" / "check.json"
EXPORT = Path("~/.cache/persian-voice-cloning/ezafe-tagger").expanduser()


def peak_gb() -> float:
    # ru_maxrss is in bytes on macOS, kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30, 2)


def compare_ezafe(tagger, sentences: list[dict], recorded: dict[str, list[bool]]) -> dict:
    differ = []
    words = 0
    for s in sentences:
        tokens, where = mfc.tagger_tokens(s["text"], len(s["words"]))
        flags = mfc.ezafe_from_tags(tagger.tag(tokens), where, len(s["words"]))
        words += len(flags)
        for i, (mine, theirs) in enumerate(zip(flags, recorded[s["id"]], strict=True)):
            if mine != theirs:
                differ.append({"sentence": s["id"], "word": s["words"][i], "light": mine, "spacy": theirs})
    return {"sentences": len(sentences), "words": words, "differ": differ}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sentencebench", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=EXPORT)
    args = parser.parse_args(argv)

    from voiceclone.text import ezafe_tagger as et

    out_dir = args.export_dir / et.REVISION[:12]
    start = time.time()
    record = None
    if not et.exported(out_dir):
        record = et.export(et.snapshot(), out_dir)
    export_seconds = round(time.time() - start, 1)
    export_peak = peak_gb()
    record = record or json.loads((out_dir / "export.json").read_text(encoding="utf-8"))

    start = time.time()
    tagger = et.ParsBertTagger(out_dir)
    load_seconds = round(time.time() - start, 2)

    recorded = {}
    for name in ("sentencebench", "review14"):
        data = json.loads((EVIDENCE / f"predictions-hazm-spacy-{name}.json").read_text(encoding="utf-8"))
        recorded.update({s["id"]: s["ezafe"] for s in data["sentences"]})
    start = time.time()
    bench = compare_ezafe(tagger, mfc.sfc.sentencebench(args.sentencebench), recorded)
    review = compare_ezafe(tagger, mfc.sfc.review14(), recorded)
    tag_seconds = round(time.time() - start, 1)

    listening = json.loads(TEXTS.read_text(encoding="utf-8"))["sentences"]
    tag_differ, tags = [], 0
    for s in listening:
        ws = mfc.sfc.words(s["text"])
        tokens, where = mfc.tagger_tokens(s["text"], len(ws))
        mine = tagger.tag(tokens)
        for k, info in enumerate(s["words"]):
            tags += 1
            light = mine[where.index(k)]
            if light != info["tag"]:
                tag_differ.append({"sentence": s["id"], "word": info["word"], "light": light, "spacy": info["tag"]})

    agree = not bench["differ"] and not review["differ"] and not tag_differ
    result = {
        "export": {"dir": str(out_dir).replace(str(Path.home()), "~"), "files_sha256": record["files"],
                   "labels": record["labels"], "seconds": export_seconds, "peak_gb": export_peak},
        "light_tagger": {"load_seconds": load_seconds, "tagging_seconds_426_sentences": tag_seconds,
                         "peak_gb_whole_process": peak_gb()},
        "ezafe": {"sentencebench": bench, "review14": review},
        "tags": {"listening12": {"tokens": tags, "differ": tag_differ}},
        "agree": agree,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "export"}, ensure_ascii=False, indent=1)[:3000])
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
