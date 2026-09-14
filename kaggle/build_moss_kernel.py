#!/usr/bin/env python3
"""Generate the Kaggle notebook that measures MOSS-TTS over the full benchmark.

Why a second kernel rather than another label in the OmniVoice one: MOSS is
pinned to transformers 5.0.x - later 5.x releases removed the private
generation APIs its remote code calls, and upstream's own card asks for 5.0.0.
The rest of this project runs on transformers 5.14. Those two cannot share an
interpreter, and unlike OmniVoice and Chatterbox the MOSS engine runs
in-process, so it cannot be isolated behind a worker without writing one.

The first attempt at this ran for about seven hours and was cancelled having
produced nothing. A single-cell timing run then measured 2.6 minutes end to
end - download and model load included - which extrapolates to roughly 1.2
hours for 27 cells, so the engine is not inherently slow enough to explain
that. Three things changed as a result:

* **the render runs before the scale control**, so a control that trips cannot
  cost us the measurement, which is what the first attempt's ordering did;
* **the render is bounded.** A hard subprocess timeout means a pathological run
  costs two hours rather than a whole GPU session. The bound is generous
  against the 1.2 hour estimate on purpose - it is a backstop, not a budget;
* **the control reports rather than asserts.** It runs last, and stopping the
  notebook there would skip collecting the results the render just produced.

One known inefficiency is left alone deliberately. With MOSS resident on the
card, faster-whisper cannot allocate and falls back to a Transformers Whisper
implementation, which is slower. The fallback is correct - scoring the baseline
renders through it reproduced the established 0.5196 exactly - so this is a
speed cost, not a measurement problem, and changing the scoring path would
change the instrument mid-comparison.

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

#: Backstop for the whole 27-cell render. The measured pace predicts ~1.2 h.
RENDER_TIMEOUT = 7200

#: Held-out similarity the baseline renders are already known to score.
KNOWN_BASELINE = 0.5196

SETUP = '''\
# MOSS-TTS against the same nine speakers, sentences, seed and reference clips.
# transformers is pinned because MOSS's remote code calls generation internals
# that 5.1 removed; upstream's card asks for 5.0.0 for the same reason.
!pip -q install "transformers>=5.0,<5.1" accelerate speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime
'''

ENVIRONMENT = '''\
import json
import os
import shutil
import subprocess
import sys
import time
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
print("bf16        :", torch.cuda.is_bf16_supported())

assert (5, 0) <= version < (5, 1), f"MOSS needs transformers 5.0.x, found {transformers.__version__}"
assert torch.cuda.device_count() >= 1, "no GPU - set the accelerator"


def run(*arguments, timeout=None):
    """Run the benchmark and return (exit code, wall-clock seconds)."""
    command = [sys.executable, "scripts/benchmark.py", *arguments]
    print(" ".join(command), flush=True)
    start = time.time()
    try:
        outcome = subprocess.run(command, cwd=REPO, text=True, timeout=timeout)
        code = outcome.returncode
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {timeout}s - called off", flush=True)
        code = -1
    elapsed = time.time() - start
    print(f"exit {code} in {elapsed / 60:.1f} min", flush=True)
    return code, elapsed
'''

RENDER = f'''\
# Same flags as every other engine apart from the device, so the cells line up
# with every earlier run. Bounded: the single-cell timing run predicts about
# 1.2 h, and this stops at two.
code, elapsed = run("--label", "moss", "--engine", "moss", "--device", "cuda",
                    timeout={RENDER_TIMEOUT})
print(f"render wall clock: {{elapsed / 60:.1f}} min")

# benchmark.py records failures in its results file, not in its exit code: an
# earlier run exited 0 with all 27 cells failed and the kernel said COMPLETE.
path = REPO / "eval/results/moss.json"
if path.is_file():
    summary = json.loads(path.read_text())["summary"]
    print("moss summary:", summary, flush=True)
    print("failures:", summary.get("failures"), " renders:", summary.get("renders"))
else:
    print("no results file - the render did not get far enough to write one")
'''

CONTROL = f'''\
# The scale check, after the measurement is safely on disk.
#
# Every number on record was produced with transformers 5.14; this kernel runs
# 5.0.x, so its embedder and recogniser are a different build. Re-score renders
# whose answer is already established and compare. This reports rather than
# asserts, because stopping here would skip collecting what the render made.
code, elapsed = run("--label", "baseline-moss-env", "--score-only", "--replay-from", "baseline",
                    "--device", "cuda", timeout=1800)
path = REPO / "eval/results/baseline-moss-env.json"
if path.is_file():
    value = json.loads(path.read_text())["summary"]["held_out_mean"]
    drift = value - {KNOWN_BASELINE}
    print(f"control: {{value:.4f}} vs {KNOWN_BASELINE} known ({{drift:+.4f}})")
    print("ON the established scale" if abs(drift) < 0.0005 else "OFF THE ESTABLISHED SCALE")
else:
    print("control did not produce a results file")
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
    # Render before control: the first attempt put the control first, and when
    # the run was cancelled it had produced nothing worth keeping.
    sources = [SETUP, ENVIRONMENT, RENDER, CONTROL, COLLECT]
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
