#!/usr/bin/env python3
"""Bring the adapted reading check's renders from the fetched Kaggle output into the repository.

The session rendered each form in its own run, so its output holds three
manifests, one per set of weights. This checks each against the registered
texts, copies the wavs into ``eval/runs/adapted-reading-v1/wavs``, and writes
one manifest with all 36 renders - the paths made the repository's, the Kaggle
ones kept - for the similarity script and the review page. The session's
gates, runs, result and environment go to the run's evidence folder.

    .venv/bin/python kaggle/fetch_output.py marsmanam/persian-voice-adapted-reading-run \\
        remote/adapted-reading-v1 adapted-reading-v1/ logs/
    .venv/bin/python scripts/collect_adapted_renders.py
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FETCHED = ROOT / "remote" / "adapted-reading-v1" / "adapted-reading-v1"
RUN = ROOT / "eval" / "runs" / "adapted-reading-v1"
EVIDENCE = ROOT / "eval" / "runs" / "evidence" / "adapted-reading-v1"
FORMS = ("P", "A1", "A5")
EXPECTED = 12


def main() -> int:
    texts = json.loads((RUN / "texts.json").read_text(encoding="utf-8"))
    (RUN / "wavs").mkdir(exist_ok=True)
    renders, settings, events = [], {}, []
    for form in FORMS:
        part = json.loads((FETCHED / form / "manifest.json").read_text(encoding="utf-8"))
        if part["texts"]["sentences_sha256"] != texts["sentences_sha256"] or tuple(part["conditions"]) != (form,):
            raise SystemExit(f"{form}: the manifest is not for the registered texts and this form")
        done = [r for r in part["renders"] if r.get("wav") and r.get("cer") is not None]
        if len(done) != EXPECTED:
            raise SystemExit(f"{form}: {len(done)} renders scored, {EXPECTED} expected")
        for r in done:
            name = Path(r["wav"]).name
            shutil.copy2(FETCHED / form / "wavs" / name, RUN / "wavs" / name)
            renders.append({**r, "wav": str((RUN / "wavs" / name).relative_to(ROOT)), "kaggle_wav": r["wav"]})
        settings[form] = part["settings"]
        events += [{**e, "form": form} for e in part["events"]]
    events.append({"event": "fetched_from_kaggle", "session": "marsmanam/persian-voice-adapted-reading-run",
                   "collected": datetime.now(UTC).isoformat(timespec="seconds")})
    first = json.loads((FETCHED / "P" / "manifest.json").read_text(encoding="utf-8"))
    manifest = {"version": 1, "purpose": "the adapted reading check: the same text read by three sets of OmniVoice weights",
                "texts": {"file": str((RUN / "texts.json").relative_to(ROOT)),
                          "sentences_sha256": texts["sentences_sha256"]},
                "settings": settings, "reference": first["reference"], "conditions": texts["conditions"],
                "sentences": [{"id": s["id"], "category": s["category"], "text": s["text"], "texts": s["texts"]}
                              for s in texts["sentences"]],
                "events": events, "renders": renders}
    (RUN / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for name in ("gates.json", "runs.json", "result.json", "environment.json"):
        shutil.copy2(FETCHED / name, EVIDENCE / name)
    print(f"{len(renders)} renders -> {(RUN / 'manifest.json').relative_to(ROOT)}; evidence -> {EVIDENCE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
