#!/usr/bin/env python3
"""Stage the evaluation dataset every GPU kernel mounts, and say how to push it.

Kaggle notebooks cannot clone this repository - it is not published - so the
measurement code, the corpus and the renders worth re-scoring travel as a
private dataset that the kernels attach. That dataset had been assembled by
hand, which is fine until a kernel fails on a machine because the copy it
mounted predates the engine it is supposed to run. This script makes the
contents reproducible and prints what changed against the version now on
Kaggle.

What goes in, and why each part:

* ``src``, ``scripts``, ``configs``, ``pyproject.toml`` - the measurement code
* ``eval/corpus`` - the 45 clips every score is computed against
* ``eval/results`` - previous runs, so a new environment can re-score them and
  prove it lands on the same scale
* ``outputs/benchmark/{baseline,finetuned,finetuned-bestof4}`` - the renders
  those results describe, which is what ``--score-only --replay-from`` reads

Deliberately excluded: model weights, caches, the virtualenv, and anything
under ``outputs`` that no results file refers to.

    .venv/bin/python kaggle/build_dataset.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle datasets version \\
        -p kaggle/dataset -m "message" --dir-mode zip
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = HERE / "dataset"
STAGE = OUT_DIR / "repo"

USER = "marsmanam"
DATASET = "persian-voice-eval"

#: Whole directories, copied as they are.
TREES = (
    "src/voiceclone",
    "configs",
    "eval/corpus",
    "eval/results",
    "outputs/benchmark/baseline",
    "outputs/benchmark/finetuned",
    "outputs/benchmark/finetuned-bestof4",
)

#: Single files, and the scripts a kernel actually invokes.
FILES = ("pyproject.toml",)
SCRIPTS = ("benchmark.py", "compare_runs.py", "omnivoice_worker.py", "chatterbox_worker.py",
           "fish_worker.py", "higgs_worker.py", "moss15_worker.py")

PROVENANCE = """\
# Provenance

Private evaluation harness for Persian zero-shot voice cloning.

- `repo/eval/corpus`: 40 clips from a re-upload of Common Voice 26 Persian
  (`pymmdrza/Common-Voice-Speech-26.0-Persian-Clean`, which declares no licence on
  Hugging Face; Mozilla Common Voice upstream is CC0) and 5 clips from Mana-TTS
  (`MahtaFetrat/Mana-TTS`, CC0-1.0, verified).
- `repo/outputs/benchmark/*`: renders produced locally by this project, uploaded so
  every configuration can be re-scored by one stack on one device.
- `repo/src`, `repo/scripts`, `repo/configs`: the measurement code itself.

Staged by `kaggle/build_dataset.py` from commit {commit}.
"""


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    STAGE.mkdir(parents=True)

    for tree in TREES:
        source = ROOT / tree
        if not source.is_dir():
            raise SystemExit(f"missing {source}")
        shutil.copytree(
            source, STAGE / tree,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    for name in FILES:
        shutil.copy2(ROOT / name, STAGE / name)
    (STAGE / "scripts").mkdir()
    for name in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, STAGE / "scripts" / name)

    (OUT_DIR / "PROVENANCE.md").write_text(PROVENANCE.format(commit=commit()), encoding="utf-8")
    (OUT_DIR / "dataset-metadata.json").write_text(
        json.dumps({"title": "Persian Voice Eval Harness", "id": f"{USER}/{DATASET}"}, indent=2) + "\n",
        encoding="utf-8",
    )

    staged = sorted(p for p in OUT_DIR.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in staged)
    print(f"staged {len(staged)} files, {total / 1e6:.1f} MB, in {OUT_DIR}")
    for name in SCRIPTS:
        print(f"  scripts/{name}")
    print()
    print("push it with:")
    print("  KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle datasets version"
          " -p kaggle/dataset -m 'your message' --dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
