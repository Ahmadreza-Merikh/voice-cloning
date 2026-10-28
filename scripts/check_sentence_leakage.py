#!/usr/bin/env python3
"""Check a listening check's new sentences against every text they must not come from.

Step 2's rule, as every check since has recorded it: letters only, a sentence
leaks if it equals a text or one is inside the other with at least 20 letters
and half the longer (``found_in`` in score_frontend_components.py). The texts:
HomoRich's sentences, SentenceBench's, every transcript in the adaptation's
splits, the sentences of every earlier listening check and the sealed test's
texts. It prints the counts the registrations record and stops on any hit.

    .venv/bin/python scripts/check_sentence_leakage.py eval/manifests/ezafe_check_v1.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / ".cache" / "persian-voice-cloning" / "eval-data" / "frontend"
MANIFESTS = ROOT / "eval" / "manifests"
EARLIER = ("frontend_listening_v1.json", "frontend_review14_gold_v1.json", "frontend_listening_v2.json",
           "rui_check_v1.json", "adapted_reading_v1.json", "news_vowels_v1.json", "xtts_news_v1.json")

spec = importlib.util.spec_from_file_location("score_frontend_components", ROOT / "scripts" / "score_frontend_components.py")
components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components)


def sources(checked: Path) -> dict[str, list[str]]:
    import pyarrow.parquet as pq

    homorich = pq.read_table(DATA / "HomoRich-train-01.parquet", columns=["Grapheme"]).column(0).to_pylist()
    with (DATA / "SentenceBench.csv").open(encoding="utf-8") as handle:
        bench = [row["grapheme"] for row in csv.DictReader(handle)]
    with gzip.open(MANIFESTS / "adaptation_splits_v1.jsonl.gz", "rt", encoding="utf-8") as handle:
        splits = [json.loads(line)["text"] for line in handle]
    earlier = [s["text"] for name in EARLIER if (MANIFESTS / name).resolve() != checked.resolve()
               for s in json.loads((MANIFESTS / name).read_text(encoding="utf-8"))["sentences"]]
    sealed = [t["text"] for t in json.loads((MANIFESTS / "test_texts_v1.json").read_text(encoding="utf-8"))["texts"]]
    return {"homorich": [t for t in homorich if t], "sentencebench": bench, "adaptation_splits": splits,
            "earlier_tests": earlier, "sealed_test_texts": sealed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sentences", type=Path)
    args = parser.parse_args(argv)
    sentences = json.loads(args.sentences.read_text(encoding="utf-8"))["sentences"]
    report, hits = {}, {}
    for name, texts in sources(args.sentences).items():
        found = components.found_in(sentences, texts)
        report[name] = {"checked": len(texts), "hits": len(found)}
        hits.update({f"{name}:{sid}": how for sid, how in found.items()})
    print(json.dumps({"leakage_check": report, "hits": hits}, ensure_ascii=False, indent=1))
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
