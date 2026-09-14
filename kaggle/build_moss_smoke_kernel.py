#!/usr/bin/env python3
"""Generate a Kaggle notebook that times ONE MOSS render, and stops if it drags.

The full MOSS run was cancelled after about seven hours without producing
anything. This repository already records roughly 2 minutes for a single MOSS
take on an M1 Pro, so seven hours for 27 cells is not what the engine is
supposed to cost - something on this hardware is wrong, and the cheapest way to
learn what is to render one cell and measure it.

Three deliberate choices:

* **the timed render runs first.** If the scale control ran first and tripped,
  the run would stop before producing the timing, which is the whole point.
  That is the same ordering mistake an earlier kernel made in reverse.
* **the render is bounded.** ``subprocess.run`` gets a hard timeout, so a cell
  that is pathologically slow costs fifteen minutes instead of consuming the
  session's GPU budget.
* **wall clock is measured here**, not read from the results file. The
  ``seconds`` field in a row is the *duration of the rendered audio*, not how
  long it took to produce - a comparison of the two is what says whether the
  engine is running at a sane rate.

    .venv/bin/python kaggle/build_moss_smoke_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/moss-smoke
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "moss-smoke"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-moss-smoke"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: One cell may take this long before the experiment is called off.
RENDER_TIMEOUT = 900

#: What the baseline renders are already known to score, for the scale check.
KNOWN_BASELINE = 0.5196

SETUP = '''\
# One MOSS cell, timed. transformers is pinned because MOSS's remote code calls
# generation internals that 5.1 removed.
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
found = sorted(INPUT.rglob("repo/scripts/benchmark.py"))
if not found:
    for path in sorted(p for p in INPUT.rglob("*") if p.is_dir())[:40]:
        print("   ", path)
    raise SystemExit("no repo under /kaggle/input - see the directories above")
SOURCE = found[0].parent.parent

REPO = Path("/kaggle/working/repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SOURCE, REPO)

version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__)
print("gpus        :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# T4 is Turing: no native bfloat16, so the engine drops to float16. MOSS is the
# only engine here that takes that path, which makes it a suspect for the
# slowness this notebook is measuring.
print("bf16 native :", torch.cuda.is_bf16_supported())
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

SMOKE = f'''\
# One cell: one speaker, one sentence. The wall clock includes downloading
# roughly 6.1 GB of weights and loading them, so it is an upper bound on what a
# steady-state render costs - which is enough to decide whether 27 cells is
# hours or minutes.
code, elapsed = run("--label", "moss-smoke", "--engine", "moss", "--device", "cuda",
                    "--speakers", "cv02", "--sentences", "s1", timeout={RENDER_TIMEOUT})

print()
print(f"one cell, end to end: {{elapsed / 60:.1f}} min")
print(f"that pace over 27 cells: {{elapsed * 27 / 3600:.1f}} h (before subtracting the one-time download)")

path = REPO / "eval/results/moss-smoke.json"
if path.is_file():
    data = json.loads(path.read_text())
    print("summary  :", data["summary"])
    print("benchmark's own duration_minutes:", data.get("duration_minutes"))
    for row in data["rows"]:
        audio = row.get("seconds")
        print("row      :", {{k: row.get(k) for k in
              ("speaker", "sentence", "seconds", "held_out_similarity",
               "character_error_rate", "error")}})
        if audio:
            # seconds is the length of the rendered audio, not the time spent.
            # Their ratio is the real-time factor, which is the number that says
            # whether generation is running at a sane rate.
            print(f"rendered {{audio:.2f}} s of audio; "
                  f"{{elapsed / audio:.0f}}x slower than real time, load included")
else:
    print("no results file - the render did not get far enough to write one")
'''

CONTROL = f'''\
# Now the scale check, after the timing is safely printed. This environment is
# transformers 5.0.x, a different dependency world from every number on record,
# so re-score renders whose answer is already established.
code, elapsed = run("--label", "baseline-moss-env", "--score-only", "--replay-from", "baseline",
                    "--device", "cuda", timeout=1800)
path = REPO / "eval/results/baseline-moss-env.json"
if path.is_file():
    value = json.loads(path.read_text())["summary"]["held_out_mean"]
    drift = value - {KNOWN_BASELINE}
    print(f"control: {{value:.4f}} vs {KNOWN_BASELINE} known ({{drift:+.4f}})")
    print("on the established scale" if abs(drift) < 0.0005 else "OFF THE ESTABLISHED SCALE")
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
for path in (REPO / "outputs/benchmark/moss-smoke").glob("*.wav"):
    shutil.copy2(path, renders / path.name)
print("saved:", len(list(renders.glob("*.wav"))), "renders")
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
    sources = [SETUP, ENVIRONMENT, SMOKE, CONTROL, COLLECT]
    for index, source in enumerate(sources):
        compile(
            "".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
            f"<cell {index}>",
            "exec",
        )

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
