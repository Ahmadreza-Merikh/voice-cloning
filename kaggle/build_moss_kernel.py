#!/usr/bin/env python3
"""Generate the Kaggle notebook that measures MOSS-TTS.

Why a second kernel rather than another label in the OmniVoice one: MOSS is
pinned to transformers 5.0.x - later 5.x releases removed the private
generation APIs its remote code calls, and upstream's own card asks for 5.0.0.
The rest of this project runs on transformers 5.14. Those two cannot share an
interpreter, and unlike OmniVoice and Chatterbox the MOSS engine runs
in-process, so it cannot be isolated behind a worker without writing one.

The consequence is that MOSS's renders are also *scored* inside that pinned
environment, which is a different dependency world from the one every number on
record came from. So the notebook proves the scale before it measures anything:
it re-scores a set of renders whose answer is already established and stops if
the answer comes back different. The engine runs only after that passes.

    .venv/bin/python kaggle/build_moss_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/moss
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "moss"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-moss-run"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: Held-out similarity the baseline renders are already known to score. The
#: control re-measures exactly this, on the same wav files.
KNOWN_BASELINE = 0.5196

SETUP = '''\
# MOSS-TTS against the same nine speakers, sentences, seed and reference clips.
# transformers is pinned because MOSS's remote code calls generation internals
# that 5.1 removed; upstream's card asks for 5.0.0 for the same reason.
!pip -q install "transformers>=5.0,<5.1" accelerate speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime
'''

ENVIRONMENT = '''\
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import transformers

INPUT = Path("/kaggle/input")
print("mounted     :", sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else "nothing")

# The dataset mounts at /kaggle/input/datasets/<owner>/<slug>/, so the repo is
# found by a file that has to be inside it rather than by an assumed depth.
found = sorted(INPUT.rglob("repo/scripts/benchmark.py"))
if not found:
    for path in sorted(p for p in INPUT.rglob("*") if p.is_dir())[:40]:
        print("   ", path)
    raise SystemExit("no repo under /kaggle/input - see the directories above")
SOURCE = found[0].parent.parent
print("source      :", SOURCE)

REPO = Path("/kaggle/working/repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SOURCE, REPO)

version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__)
print("gpus        :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# T4 is Turing and has no native bfloat16, so the engine falls back to float16.
# Printed rather than assumed, because it decides what precision MOSS runs at.
print("bf16 native :", torch.cuda.is_bf16_supported())

assert (5, 0) <= version < (5, 1), f"MOSS needs transformers 5.0.x, found {transformers.__version__}"
assert torch.cuda.device_count() >= 1, "no GPU - set the accelerator"


def run(*arguments):
    command = [sys.executable, "scripts/benchmark.py", *arguments]
    print(" ".join(command), flush=True)
    outcome = subprocess.run(command, cwd=REPO, text=True)
    print("exit code:", outcome.returncode, flush=True)
    return outcome.returncode
'''

CONTROL = f'''\
# Prove the scale before measuring anything on it.
#
# Every number on record was produced with transformers 5.14. This kernel runs
# 5.0.x, so its embedder and recogniser are a different build. Re-score renders
# whose answer is already established and require that answer back. If the
# scale has moved, stop here - a MOSS number from a shifted scale would be
# worse than no MOSS number, because it would look comparable and not be.
import json

assert run("--label", "baseline-moss-env", "--score-only", "--replay-from", "baseline",
           "--device", "cuda") == 0

summary = json.loads((REPO / "eval/results/baseline-moss-env.json").read_text())["summary"]
print("control held-out:", summary["held_out_mean"], " CER:", summary["cer_mean"], flush=True)
assert abs(summary["held_out_mean"] - {KNOWN_BASELINE}) < 0.0005, (
    f"this environment scores the same audio at {{summary['held_out_mean']:.4f}}, "
    "not {KNOWN_BASELINE} - its numbers are not on the established scale"
)
print("scale confirmed", flush=True)
'''

RENDER = '''\
# Same flags as every other engine apart from the device, so the cells line up.
assert run("--label", "moss", "--engine", "moss", "--device", "cuda") == 0

# benchmark.py records failures in its results file, not in its exit code: an
# earlier run exited 0 with all 27 cells failed and the kernel said COMPLETE.
summary = json.loads((REPO / "eval/results/moss.json").read_text())["summary"]
print("moss summary:", summary, flush=True)
failures = summary.get("failures") or 0
assert failures == 0, f"MOSS failed {failures} cells - see the traceback above"
assert (summary.get("renders") or 0) > 0, "MOSS produced no renders"
'''

COLLECT = '''\
destination = Path("/kaggle/working/results")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)

renders = Path("/kaggle/working/moss_renders")
renders.mkdir(exist_ok=True)
for path in (REPO / "outputs/benchmark/moss").glob("*.wav"):
    shutil.copy2(path, renders / path.name)
print("renders saved:", len(list(renders.glob("*.wav"))))

for name in ("moss", "baseline-moss-env"):
    path = destination / f"{name}.json"
    if path.is_file():
        data = json.loads(path.read_text())["summary"]
        print(name, {k: data.get(k) for k in ("held_out_mean", "cer_mean", "renders", "failures")})
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
    sources = [SETUP, ENVIRONMENT, CONTROL, RENDER, COLLECT]
    for index, source in enumerate(sources):
        compile(
            "".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
            f"<cell {index}>",
            "exec",
        )

    # run() is defined in the environment cell and used by the two after it;
    # a NameError there would only show up at run time, so check it here.
    defines = [i for i, s in enumerate(sources) if "def run(" in s]
    uses = [i for i, s in enumerate(sources) if "--label" in s]
    assert defines and min(defines) <= min(uses), "run() is used before it is defined"

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

    print(f"wrote {path} ({len(sources)} cells, all compile, run() defined before use)")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {metadata['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
