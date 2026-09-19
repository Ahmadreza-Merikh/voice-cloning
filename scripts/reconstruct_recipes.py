#!/usr/bin/env python3
"""Write recipes, after the fact, for the results the documents quote.

``benchmark.py`` has written a recipe beside every results file since
2026-09-19 (``voiceclone.evaluation.recipe``). The files the findings rest on
are older, so their recipes are reconstructed here, once, from what the
repository can show - and every field says where it came from:

* ``git``: the engine's settings section of ``configs/default.yaml`` at the
  code commit the run used when that commit is recorded, otherwise at the last
  commit before the run, which is then marked approximate - a working tree can
  differ from its last commit, and a kernel's dataset was staged at or before it
* ``record``: a run record, the roadmap or the findings, named
* ``hub``: a model repository's history, for revisions nobody wrote down at the
  time (a repository unchanged since before the run could only serve one)
* ``now``: hashes computed on 2026-09-19 of files the run read - the reference
  clips, the corpus manifest - which are the run's only if the files have not
  changed since; the manifest has not changed in git since 2026-08-03, and the
  clips are not tracked, so that part is an assumption, stated

Anything unknown is written as unknown. The recipes are written once, like any
other, so these labels are closed.

    .venv/bin/python scripts/reconstruct_recipes.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from voiceclone.evaluation import recipe  # noqa: E402

RESULTS = REPO_ROOT / "eval" / "results"
CORPUS = REPO_ROOT / "eval" / "corpus"
TEHRAN = timezone(timedelta(hours=3, minutes=30))
FT = "fine" + "tuned"

HUB = {
    "omnivoice": {
        "model": "k2-fsa/OmniVoice", "revision": "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        "source": "hub: the repository's last commit is c5fdb5cc of 2026-07-03, so every download since "
                  "then - including this run's - served it",
        "package": "omnivoice from PyPI, unpinned; 0.2.1 (2026-07-16) was the latest release on the run's date "
                   "- an inference, the kernel log was not kept",
    },
    "moss": {"model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "revision": "12aa734e",
             "source": "record: SOTA_ROADMAP.md, exact revisions measured"},
    "fish": {"model": "fishaudio/s2-pro", "revision": "1de9996b",
             "source": "record: FINDINGS.md §8 and SOTA_ROADMAP.md", "precision": "float16 (--half)"},
    "higgs": {"model": "multimodalart/higgs-audio-v3-tts-4b-transformers",
              "revision": "30f01593ee6a12efa586c92455afe4b76e45095d",
              "source": "record: pinned in configs/default.yaml; weights sha256-identical to Boson's (FINDINGS §8)",
              "precision": "float16 on the T4"},
    "f5": {"model": "Lumos675/F5_TTS_Persian", "revision": "cc322511",
           "source": "record: SOTA_ROADMAP.md, exact revisions measured"},
    "moss15": {"model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
               "revision": "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
               "codec": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 at f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
               "precision": "bfloat16 (emulated on the T4), SDPA",
               "source": "record: the run's own record in eval/runs, and the worker's ready line in its evidence"},
}

#: label: (session, code commit if recorded, replayed label, extra facts)
RUNS = {
    "baseline": ("mac:project-venv", None, None, {}),
    f"{FT}-bestof4": ("mac:project-venv", None, None, {
        "checkpoint": f"remote/{FT}.pt, the continued F5 checkpoint, passed as VOICECLONE__ENGINES__F5__CHECKPOINT "
                      "(record: docs/REMOTE-TRAINING.md section 4) - the config file's own checkpoint value below is "
                      "the published Persian one and is not what ran",
        "loaded_as": f"remote/{FT}_ema.safetensors, written 2026-08-27 19:35 +0330 before this run; sha256 "
                     "46dabd83d6fa2db4cf552b6a5aa84cff3121afca29b1cde666971efd2c1c686e (now)",
        "takes": "best of 4 per cell",
        "model": {"model": "this project's continuation of Lumos675/F5_TTS_Persian (at cc322511) on Common Voice 26",
                  "source": "record: docs/REMOTE-TRAINING.md; the loaded file's sha256 is in loaded_as"},
    }),
    "omnivoice": ("kaggle:persian-voice-eval-run@2026-09-12", None, None, {}),
    "omnivoice-local": ("mac:project-venv", None, "omnivoice", {}),
    "moss": ("kaggle:persian-voice-moss-run@2026-09-15", None, None, {"precision": "bfloat16 on the T4"}),
    "moss-local": ("mac:project-venv", None, "moss", {}),
    "fish": ("kaggle:persian-voice-fish-smoke-v5@2026-09-17", None, None, {}),
    "higgs": ("kaggle:persian-voice-higgs-smoke-v2@2026-09-17", None, None, {}),
    "moss15": ("kaggle:persian-voice-moss15-v3", "89dc14e", None,
               {"record": "eval/runs/persian-voice-moss15-v3.json"}),
    "moss15-fa": ("kaggle:persian-voice-moss15-v4", "cdc4b36", None,
                  {"record": "eval/runs/persian-voice-moss15-v4.json"}),
    "baseline-remote": ("kaggle:persian-voice-eval-run@2026-09-12", None, "baseline", {}),
    f"{FT}-bestof4-remote": ("kaggle:persian-voice-eval-run@2026-09-12", None, f"{FT}-bestof4", {}),
    "baseline-moss-env": ("kaggle:persian-voice-moss-run@2026-09-15", None, "baseline", {}),
    "baseline-fish-env": ("kaggle:persian-voice-fish-smoke-v5@2026-09-17", None, "baseline", {}),
    "baseline-higgs-env": ("kaggle:persian-voice-higgs-smoke-v2@2026-09-17", None, "baseline", {}),
    "baseline-moss15-env": ("kaggle:persian-voice-moss15-v3", "89dc14e", "baseline", {}),
    "baseline-moss15v4-env": ("kaggle:persian-voice-moss15-v4", "cdc4b36", "baseline", {}),
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout


def created_utc(data: dict, session: str) -> datetime:
    """Mac runs stamped Tehran time; Kaggle machines run on UTC."""
    stamp = datetime.strptime(data["created"], "%Y-%m-%d %H:%M:%S")
    zone = TEHRAN if session.startswith("mac") else UTC
    return stamp.replace(tzinfo=zone).astimezone(UTC)


def settings_at(commit: str, engine: str) -> dict | None:
    import yaml

    text = git("show", f"{commit}:configs/default.yaml")
    data = yaml.safe_load(text)

    def find(node):
        if isinstance(node, dict):
            if engine in node and isinstance(node[engine], dict) and "tts" in node:
                return node[engine]
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None

    return find(data)


def reference_clips(data: dict) -> list[dict]:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    index = data["config"].get("reference_index") or 0
    clips = []
    for speaker in sorted({row["speaker"] for row in data["rows"]}):
        entries = sorted((e for e in manifest["entries"] if e["speaker"] == speaker), key=lambda e: e["file"])
        entry = entries[index % len(entries)]
        path = CORPUS / entry["file"]
        clips.append({
            "speaker": speaker, "file": entry["file"],
            "sha256": recipe.sha256_file(path) if path.is_file() else None,
            "duration": entry.get("duration"),
            "transcript_given": entry["transcript"] if data["config"].get("ref_transcript") == "corpus"
            else "transcribed by the recogniser inside the run",
        })
    return clips


def reconstruct(label: str) -> dict:
    session, commit, replayed, extra = RUNS[label]
    results_path = RESULTS / f"{label}.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    when = created_utc(data, session)
    engine = data["engine"] if not replayed else json.loads((RESULTS / f"{replayed}.json").read_text())["engine"]
    if commit:
        config_commit, config_basis = commit, "git: the code commit the run's dataset was staged from (record)"
    else:
        config_commit = git("rev-list", "-1", f"--before={when.isoformat()}", "HEAD").strip()[:7]
        config_basis = ("git: approximate - the last commit before the run; the run's own commit was not "
                        "recorded")
    built = recipe.build(
        label=label, engine=engine,
        engine_settings=None if replayed else settings_at(config_commit, engine),
        engine_options=data["config"].get("engine_option") or [],
        preprocess_options=data["config"].get("preprocess_option") or [],
        cells=[], references=reference_clips(data), corpus_manifest=CORPUS / "manifest.json",
        sentences={row["sentence"]: row["expected"] for row in data["rows"] if "expected" in row},
        seeds=data["config"].get("seeds", []), candidates=data["config"].get("candidates", 1),
        reference_index=data["config"].get("reference_index") or 0,
        reference_transcript=data["config"].get("ref_transcript", "corpus"),
        replay_from=replayed, scoring_environment=data.get("scoring_environment") or {"id": session},
        commit=commit, results_path=results_path,
    )
    facts = dict(extra)
    built["render"]["model"] = None if replayed else facts.pop("model", None) or HUB.get(engine)
    built["render"]["facts"] = facts
    built["reconstructed"] = {
        "date": "2026-09-19",
        "run_created_utc": when.isoformat(),
        "settings_from_commit": None if replayed else config_commit,
        "settings_basis": None if replayed else config_basis,
        "references_basis": "now: hashed on 2026-09-19; the clips are not tracked in git, the manifest is and "
                            "has not changed since 2026-08-03",
        "unknown": [name for name, value in (("code commit", commit), ("per-render engine reports", None))
                    if value is None],
    }
    return built


def main() -> int:
    if "--dry-run" in sys.argv:
        for label in sys.argv[sys.argv.index("--dry-run") + 1:] or list(RUNS):
            print(json.dumps(reconstruct(label), ensure_ascii=False, indent=1)[:3000])
        return 0
    written = []
    for label in RUNS:
        path = recipe.recipe_path(RESULTS / f"{label}.json")
        recipe.write_once(path, reconstruct(label))
        written.append(path.name)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
