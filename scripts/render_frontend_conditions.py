#!/usr/bin/env python3
"""Render each listening sentence in the front-end's four forms with OmniVoice.

Registered before it ran (eval/runs/persian-voice-frontend-listening-v1.json).
Reads eval/runs/frontend-listening-v1/texts.json (scripts/build_frontend_texts.py)
and renders A, E, EN and ENV with the same voice, seed and settings the A/B/C
run used, so the text is the only difference. Before any engine loads, every
text goes through the app's own text pipeline, and the run stops if a mark
would not reach the model.

``--cer`` transcribes every render after the engine is unloaded, against the
plain sentence with diacritics stripped. It is a harm check only: the script
writes neither the ezafe nor short vowels, so CER cannot see them.

Step 3 (eval/runs/persian-voice-frontend-listening-v2.json) renders its own
texts.json the same way: its forms are read from the file, and
``--app-normaliser`` gives OmniVoice the normalisation the app now gives it
(أ and ۀ kept) while the app still adds no ezafe of its own - E holds it.

    .venv/bin/python scripts/render_frontend_conditions.py --cer
    .venv/bin/python scripts/render_frontend_conditions.py --cer --app-normaliser \
        --texts eval/runs/frontend-listening-v2/texts.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ab_omnivoice_ipa", ROOT / "scripts" / "ab_omnivoice_ipa.py")
ab = importlib.util.module_from_spec(spec)
sys.modules["ab_omnivoice_ipa"] = ab
spec.loader.exec_module(ab)

from voiceclone.config import AppConfig  # noqa: E402

TEXTS = ROOT / "eval" / "runs" / "frontend-listening-v1" / "texts.json"
CONDITIONS = ("A", "E", "EN", "ENV")
#: Step 2's forms, as its manifest described them; a later texts.json names its own.
STEP2_CONDITIONS = {"A": "plain script", "E": "+ ezafe marks", "EN": "+ sukun where no ezafe",
                    "ENV": "+ first short vowels of nouns and adjectives"}
ENGINE = "omnivoice"


def jobs(sentences: list[dict], seed: int, conditions=CONDITIONS) -> list:
    return [ab.Job(s["id"], c, ENGINE, False, s["texts"][c], seed) for s in sentences for c in conditions]


def marks_of(text: str) -> list[tuple[int, str]]:
    from voiceclone.text.marks_fa import HARAKAT

    return [(w, c) for w, word in enumerate(text.split()) for c in word if c in HARAKAT]


def check_marks_survive(config: AppConfig, all_jobs: list) -> tuple[list[str], list[str]]:
    """Marks lost or moved on the way to the model, and every other change the app makes."""
    from voiceclone.text.pipeline import TextPipeline

    pipeline = TextPipeline(config.text)
    lost, changed = [], []
    for job in all_jobs:
        fed = " ".join(chunk.synth_text for chunk in pipeline.prepare(job.input_text, engine_name=ENGINE).chunks)
        if marks_of(fed) != marks_of(job.input_text):
            lost.append(f"{job.id} {job.condition}: {job.input_text!r} -> {fed!r}")
        elif " ".join(fed.split()) != " ".join(job.input_text.split()):
            changed.append(f"{job.id} {job.condition}: {job.input_text!r} -> {fed!r}")
    return lost, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--texts", type=Path, default=TEXTS)
    parser.add_argument("--out", type=Path, default=None, help="default: the texts' folder")
    parser.add_argument("--seed", type=int, default=ab.DEFAULT_SEED)
    parser.add_argument("--speaker", default="mana")
    parser.add_argument("--cer", action="store_true")
    parser.add_argument("--app-normaliser", action="store_true",
                        help="normalise for OmniVoice as the app now does (أ and ۀ kept); no ezafe added")
    args = parser.parse_args(argv)
    # Paths are recorded relative to the repository, so they are made absolute first.
    args.texts = args.texts.resolve()
    args.out = (args.out or args.texts.parent).resolve()

    config = ab.render_config(AppConfig.load(None))
    if args.app_normaliser:
        config = config.copy_with(**{"text.marks_engines": [ENGINE]})
    built = json.loads(args.texts.read_text(encoding="utf-8"))
    sentences = built["sentences"]
    described = built.get("conditions") or STEP2_CONDITIONS
    conditions = tuple(described)
    all_jobs = jobs(sentences, args.seed, conditions)
    lost, changed = check_marks_survive(config, all_jobs)
    if lost:
        raise SystemExit("the text pipeline drops or moves a mark in these inputs:\n" + "\n".join(lost))

    corpus = json.loads((ROOT / "eval" / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    reference = ab.reference_entry(corpus, args.speaker)
    manifest_path = args.out / "manifest.json"
    settings = {field: getattr(config.engines.omnivoice, field) for field in ab._SETTINGS_FIELDS}
    manifest = {
        "version": 1,
        "purpose": ("front-end listening test: OmniVoice on plain script and on three marked forms of it"
                    if conditions == CONDITIONS else
                    f"front-end listening test: OmniVoice on {len(conditions)} forms of each sentence"),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "texts": {"file": str(args.texts.relative_to(ROOT)), "sentences_sha256": built["sentences_sha256"]},
        "settings": {"seed": args.seed, "candidates": 1, "evaluation_enabled": False, "cer_after_unload": args.cer,
                     "omnivoice": settings, "omnivoice_model_id": config.engines.omnivoice.model_id,
                     "text": {"ezafe_enabled": config.text.ezafe.enabled, "marks_engines": config.text.marks_engines}},
        "reference": {"speaker": args.speaker, "file": reference["file"], "transcript": reference["transcript"]},
        "conditions": dict(described),
        "sentences": [{"id": s["id"], "category": s["category"], "text": s["text"], "texts": s["texts"]} for s in sentences],
        "events": [{"event": "marks_checked_through_the_text_pipeline", "inputs": len(all_jobs),
                    "other_changes_by_the_app": changed}],
        "renders": [],
    }
    ab.write_manifest(manifest_path, manifest)
    ab._check_engines_available([ENGINE])

    from voiceclone.pipelines.studio import Studio

    studio = Studio(config)
    ab.render_group(studio, all_jobs, out_dir=args.out, reference_path=ROOT / "eval" / "corpus" / reference["file"],
                    reference_text=reference["transcript"], reference_file=reference["file"], manifest=manifest,
                    manifest_path=manifest_path)
    studio.release()
    manifest["events"].append({"event": "released", "engine": ENGINE})
    ab.write_manifest(manifest_path, manifest)

    if args.cer:
        from voiceclone.audio.io import load_audio
        from voiceclone.evaluation.similarity import character_error_rate

        plain = {s["id"]: s["text"] for s in sentences}
        manifest["events"].append({"event": "cer_started", "engines_unloaded": True})
        for entry in manifest["renders"]:
            if not entry.get("wav"):
                continue
            heard = studio.asr.transcribe(load_audio(entry["wav"])).text.strip()
            entry["asr_text"] = heard
            entry["cer"] = character_error_rate(plain[entry["id"]], heard)
            print(f"  [{entry['condition']}] {entry['id']} CER {entry['cer'] * 100:.1f}%")
            ab.write_manifest(manifest_path, manifest)
        manifest["asr_model"] = f"{config.engines.asr.backend} {config.engines.asr.model}"
        studio.release()
        ab.write_manifest(manifest_path, manifest)
    rendered = sum(1 for entry in manifest["renders"] if entry.get("wav"))
    print(f"{rendered} rendered -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
