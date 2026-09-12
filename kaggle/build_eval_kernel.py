#!/usr/bin/env python3
"""Generate the Kaggle notebook that measures OmniVoice against F5.

Why remotely: two local attempts were stopped by this project's own memory
policy - the second before its first render - because a 16 GB Mac already
carrying ~3.9 GB in the compressor cannot hold the worker and the scoring stack
at once. The published recipe does not change; only where it runs does.

Why every configuration is re-scored here rather than comparing against the
numbers measured on the Mac: a score produced by a different device is not
automatically on the same scale as one produced by MPS. Re-scoring the F5
renders on the same machine, with the same embedder and recogniser, keeps the
comparison paired - and the gap between each F5 run's remote score and its known
local score is itself the control that says whether the two scales agree.

    .venv/bin/python kaggle/build_eval_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/eval
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "eval"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-eval-run"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: Runs scored from renders already in the dataset. F5 cannot be re-rendered
#: here - its checkpoint is not uploaded - and it does not need to be: scoring
#: reads the wavs.
REPLAY_LABELS = ("baseline", "finetuned", "finetuned-bestof4")

SETUP = '''\
# OmniVoice against F5 on the same nine speakers, sentences, seed and reference
# clips - every configuration scored by one embedder and one recogniser here.
!pip -q install omnivoice speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime
'''

ENVIRONMENT = '''\
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

INPUT = Path("/kaggle/input")
print("mounted   :", sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else "nothing")

# Two runs died here, both on an assumed depth. The dataset does not mount at
# /kaggle/input/<slug> as version 1 assumed, nor one level down as version 2's
# glob assumed - it lands at /kaggle/input/datasets/<owner>/<slug>/. Searching
# for a file that has to be inside the repo makes the depth irrelevant.
# Version 2's failure listing was 40 corpus wavs and said nothing, so list
# directories instead.
found = sorted(INPUT.rglob("repo/scripts/benchmark.py"))
if not found:
    for path in sorted(p for p in INPUT.rglob("*") if p.is_dir())[:40]:
        print("   ", path)
    raise SystemExit("no repo under /kaggle/input - see the directories above")
SOURCE = found[0].parent.parent
print("source    :", SOURCE)

REPO = Path("/kaggle/working/repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SOURCE, REPO)

# The worker runs out-of-process so it can hold its own pins on the laptop.
# Here one interpreter has everything, so point it at this one. The name is the
# flat one find_omnivoice_python() actually reads - version 3 set a nested
# VOICECLONE__ENGINES__OMNIVOICE__PYTHON, which nothing consumes, and every one
# of the 27 cells failed on an unavailable engine. It went unnoticed locally
# because the Mac has a prepared venv the lookup falls back to.
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable

print("gpus      :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("corpus    :", len(list((REPO / "eval/corpus").glob("*.wav"))), "clips")
for label in ("baseline", "finetuned", "finetuned-bestof4"):
    renders = list((REPO / "outputs/benchmark" / label).glob("*.wav"))
    print(f"renders   : {label:20s} {len(renders)}")
assert torch.cuda.device_count() >= 1, "no GPU - set the accelerator"
'''

RENDER = '''\
# Renders and scores OmniVoice. Same flags as the laptop run apart from the
# device, so the cells line up with every earlier run.
#
# This runs after the control on purpose. Version 3 failed here, and had the
# check below existed then it would have stopped the notebook before the
# control was written - which was the only part of that run worth keeping.
assert run("--label", "omnivoice", "--engine", "omnivoice", "--device", "cuda") == 0

# benchmark.py records failures in its results file, not in its exit code, so
# version 3 exited 0 with all 27 cells failed and the kernel reported COMPLETE.
# The file is the evidence; the exit code is not.
import json

summary = json.loads((REPO / "eval/results/omnivoice.json").read_text())["summary"]
print("omnivoice summary:", summary, flush=True)
failures = summary.get("failures") or 0
assert failures == 0, f"OmniVoice failed {failures} cells - see the traceback above"
assert (summary.get("renders") or 0) > 0, "OmniVoice produced no renders"
'''

REPLAY = '''\
# The control: score the F5 renders that were made on the laptop, with the
# embedder and recogniser running here. Nothing is re-rendered.
#
# It runs before OmniVoice so that a failing engine cannot cost us the control.
def run(*arguments):
    command = [sys.executable, "scripts/benchmark.py", *arguments]
    print(" ".join(command), flush=True)
    outcome = subprocess.run(command, cwd=REPO, text=True)
    print("exit code:", outcome.returncode, flush=True)
    return outcome.returncode

for label in ("baseline", "finetuned", "finetuned-bestof4"):
    assert run("--label", label + "-remote", "--score-only", "--replay-from", label,
               "--device", "cuda") == 0
'''

COMPARE = '''\
outcome = subprocess.run(
    [sys.executable, "scripts/benchmark.py", "--compare",
     "baseline-remote", "finetuned-remote", "finetuned-bestof4-remote", "omnivoice"],
    cwd=REPO, text=True, capture_output=True,
)
print(outcome.stdout)
print(outcome.stderr[-2000:] if outcome.returncode else "", flush=True)
'''

COLLECT = '''\
# Everything worth bringing home: the results files and the renders.
import json

destination = Path("/kaggle/working/results")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)

summary = {}
for name in ("omnivoice", "baseline-remote", "finetuned-remote", "finetuned-bestof4-remote"):
    path = destination / f"{name}.json"
    if path.is_file():
        data = json.loads(path.read_text())["summary"]
        summary[name] = {k: data.get(k) for k in ("held_out_mean", "cer_mean", "renders", "failures")}
print(json.dumps(summary, indent=2))

renders = Path("/kaggle/working/omnivoice_renders")
renders.mkdir(exist_ok=True)
for path in (REPO / "outputs/benchmark/omnivoice").glob("*.wav"):
    shutil.copy2(path, renders / path.name)
print("renders saved:", len(list(renders.glob("*.wav"))))
'''


def cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    # The control runs before OmniVoice: version 3's engine failed every cell,
    # and the re-scored F5 runs were the only part of it worth keeping.
    sources = [SETUP, ENVIRONMENT, REPLAY, RENDER, COMPARE, COLLECT]
    for index, source in enumerate(sources):
        compile(
            "".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
            f"<cell {index}>",
            "exec",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [cell(source) for source in sources],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT_DIR / f"{SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "id": f"{USER}/{SLUG}",
        "title": SLUG,
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": MACHINE_SHAPE,
        "enable_internet": True,
        "dataset_sources": [f"{USER}/{DATASET}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {path} ({len(sources)} cells, all compile)")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {metadata['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
