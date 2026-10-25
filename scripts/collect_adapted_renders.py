#!/usr/bin/env python3
"""Bring the adapted reading check's renders from the fetched Kaggle output into the repository.

The session rendered each set of weights in its own run, so its output holds
one manifest per set. This checks each against the registered texts, copies
the wavs into ``eval/runs/<run>/wavs``, and writes one manifest with every
render - the paths made the repository's, the Kaggle ones kept - for the
similarity script and the review page. The session's gates, runs, result and
environment go to the run's evidence folder. The news vowels check renders two
texts under each of its two sets of weights the same way.

    .venv/bin/python kaggle/fetch_output.py marsmanam/persian-voice-adapted-reading-run \\
        remote/adapted-reading-v1 adapted-reading-v1/ logs/
    .venv/bin/python scripts/collect_adapted_renders.py
    .venv/bin/python scripts/collect_adapted_renders.py --run news-vowels-v1 \\
        --session marsmanam/persian-voice-news-vowels-run --weights P A --expected 16
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="adapted-reading-v1")
    parser.add_argument("--session", default="marsmanam/persian-voice-adapted-reading-run")
    parser.add_argument("--weights", nargs="+", default=["P", "A1", "A5"])
    parser.add_argument("--expected", type=int, default=12, help="renders per set of weights")
    args = parser.parse_args(argv)
    fetched = ROOT / "remote" / args.run / args.run
    run = ROOT / "eval" / "runs" / args.run
    evidence = ROOT / "eval" / "runs" / "evidence" / args.run
    texts = json.loads((run / "texts.json").read_text(encoding="utf-8"))
    (run / "wavs").mkdir(exist_ok=True)
    renders, settings, events = [], {}, []
    for form in args.weights:
        part = json.loads((fetched / form / "manifest.json").read_text(encoding="utf-8"))
        meant = json.loads((run / form / "texts.json").read_text(encoding="utf-8"))["conditions"]
        if part["texts"]["sentences_sha256"] != texts["sentences_sha256"] or part["conditions"] != meant:
            raise SystemExit(f"{form}: the manifest is not for the registered texts and this set of weights")
        done = [r for r in part["renders"] if r.get("wav") and r.get("cer") is not None]
        if len(done) != args.expected:
            raise SystemExit(f"{form}: {len(done)} renders scored, {args.expected} expected")
        for r in done:
            name = Path(r["wav"]).name
            shutil.copy2(fetched / form / "wavs" / name, run / "wavs" / name)
            renders.append({**r, "wav": str((run / "wavs" / name).relative_to(ROOT)), "kaggle_wav": r["wav"]})
        settings[form] = part["settings"]
        events += [{**e, "form": form} for e in part["events"]]
    events.append({"event": "fetched_from_kaggle", "session": args.session,
                   "collected": datetime.now(UTC).isoformat(timespec="seconds")})
    first = json.loads((fetched / args.weights[0] / "manifest.json").read_text(encoding="utf-8"))
    purpose = ("the adapted reading check: the same text read by three sets of OmniVoice weights"
               if args.run == "adapted-reading-v1" else f"{args.run}: its texts read by each set of OmniVoice weights")
    manifest = {"version": 1, "purpose": purpose,
                "texts": {"file": str((run / "texts.json").relative_to(ROOT)),
                          "sentences_sha256": texts["sentences_sha256"]},
                "settings": settings, "reference": first["reference"], "conditions": texts["conditions"],
                "sentences": [{"id": s["id"], "category": s["category"], "text": s["text"], "texts": s["texts"]}
                              for s in texts["sentences"]],
                "events": events, "renders": renders}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("gates.json", "runs.json", "result.json", "environment.json"):
        shutil.copy2(fetched / name, evidence / name)
    print(f"{len(renders)} renders -> {(run / 'manifest.json').relative_to(ROOT)}; evidence -> {evidence.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
