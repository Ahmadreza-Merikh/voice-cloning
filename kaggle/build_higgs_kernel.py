#!/usr/bin/env python3
"""Generate a Kaggle notebook that proves out Higgs TTS 3 on hardware it disowns.

Boson ships Higgs TTS 3 for SGLang-Omni or vLLM-Omni. Both want compute
capability 8.0; the free T4 is 7.5, vLLM refuses the checkpoint's bfloat16 below
8.0, and its default attention backend is gated there too. The one public
attempt at forcing that path returned WAV files whose every sample was -32768 -
a complete failure that reports success.

So this runs the weights through a transformers port instead, and the questions
are in the order that makes a failure cheap to interpret:

1. does the port load in float16 inside 15 GB
2. does one render come back as *audio* - finite, varying, the right length
3. does a cell scored through the engine land on the established scale

Question 2 is where the Turing risk actually lives, so the render is inspected
rather than trusted, and the worker refuses a near-constant render itself.

    .venv/bin/python kaggle/build_higgs_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/higgs
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "higgs"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-higgs-smoke"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: Loading 9.3 GB of weights plus a 0.8 GB codec, then one render.
WORKER_TIMEOUT = 1800

#: The same again through the harness, with scoring models on top.
BENCH_TIMEOUT = 1800

#: What the baseline renders are already known to score, for the scale check.
KNOWN_BASELINE = 0.5196

MODEL = "multimodalart/higgs-audio-v3-tts-4b-transformers"
REVISION = "30f01593ee6a12efa586c92455afe4b76e45095d"

SETUP = '''\
# The port needs transformers 5.5 or newer, which is the opposite pin from the
# Fish notebook - hence a separate kernel. The rest is the harness's own scoring
# stack, and the scale control at the end says whether it minds the version.
!pip -q install "transformers>=5.5" speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime 2>&1 | tail -5
'''

ENVIRONMENT = f'''\
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
assert (REPO / "scripts/higgs_worker.py").is_file(), "the dataset predates the Higgs worker"

# The engine starts the worker as a subprocess, which inherits this environment.
os.environ["VOICECLONE_HIGGS_PYTHON"] = sys.executable
os.environ["PYTHONPATH"] = str(REPO / "src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("transformers:", transformers.__version__)
print("torch       :", torch.__version__)
print("gpus        :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# This is the whole problem: no bfloat16 on Turing, so the LM loads float16.
# The codec runs fp32 regardless, which is what the vendor's own stack did not
# do and is the likeliest reason its Turing renders came out constant.
print("bf16 native :", torch.cuda.is_bf16_supported())
free, total = torch.cuda.mem_get_info()
print(f"vram        : {{free / 2**30:.1f}} GB free of {{total / 2**30:.1f}} GB")
version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
assert version >= (5, 5), f"the port needs transformers >= 5.5, found {{transformers.__version__}}"
assert torch.cuda.device_count() >= 1, "no GPU - set the accelerator"


def run(command, timeout=None, cwd=None):
    """Run a command, echoing it, and return (exit code, wall-clock seconds)."""
    print("$", " ".join(str(part) for part in command), flush=True)
    start = time.time()
    try:
        code = subprocess.run(command, cwd=cwd, text=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {{timeout}}s - called off", flush=True)
        code = -1
    elapsed = time.time() - start
    print(f"exit {{code}} in {{elapsed / 60:.1f}} min", flush=True)
    return code, elapsed
'''

DOWNLOAD = f'''\
# 9.3 GB of weights at a pinned revision - the commit whose modeling code was
# read - plus the 806 MB codec the port loads separately.
from huggingface_hub import snapshot_download

start = time.time()
CHECKPOINT = Path(snapshot_download(repo_id="{MODEL}", revision="{REVISION}"))
print(f"downloaded in {{(time.time() - start) / 60:.1f}} min -> {{CHECKPOINT}}")

start = time.time()
CODEC = Path(snapshot_download(repo_id="bosonai/higgs-audio-v2-tokenizer"))
print(f"codec in {{(time.time() - start) / 60:.1f}} min -> {{CODEC}}")

# The weights are the vendor's own: this hash was checked against
# bosonai/higgs-tts-3-4b before the port was trusted with anything.
import hashlib

weights = CHECKPOINT / "model.safetensors"
digest = hashlib.sha256()
with open(weights, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 24), b""):
        digest.update(block)
print("model.safetensors sha256:", digest.hexdigest())
print("matches bosonai/higgs-tts-3-4b:",
      digest.hexdigest() == "2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a")
'''

DRIVER = f'''\
# A driver that knows only the worker protocol: one job in, one JSON line out.
# If this fails the problem is the port or the new worker, not the engine, the
# config or the scoring.
driver = Path("/kaggle/working/drive_higgs.py")
driver.write_text("""
import json, subprocess, sys, time

reference, reference_text, text, output, cwd = sys.argv[1:6]
command = [sys.executable, "scripts/higgs_worker.py", "--model", "{MODEL}",
           "--revision", "{REVISION}", "--device", "cuda", "--dtype", "bfloat16"]
start = time.time()
worker = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)

def reply():
    while True:
        line = worker.stdout.readline()
        if not line:
            raise SystemExit(f"worker exited with {{worker.poll()}}")
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            continue

ready = reply()
print("READY", json.dumps(ready), flush=True)
print(f"LOAD {{time.time() - start:.1f}}s", flush=True)
if not ready.get("ready"):
    raise SystemExit("worker did not load")

job = {{"text": text, "reference": reference, "reference_text": reference_text,
       "output": output, "seed": 42}}
start = time.time()
worker.stdin.write(json.dumps(job, ensure_ascii=False) + chr(10))
worker.stdin.flush()
answer = reply()
print(f"RENDER {{time.time() - start:.1f}}s", flush=True)
print("RESULT", json.dumps(answer), flush=True)
worker.stdin.write(json.dumps({{"op": "quit"}}) + chr(10))
worker.stdin.flush()
""", encoding="utf-8")

manifest = json.loads((REPO / "eval/corpus/manifest.json").read_text(encoding="utf-8"))
entry = next(e for e in manifest["entries"] if e["speaker"] == "cv02" and e["role"] == "reference")
REFERENCE = REPO / "eval/corpus" / entry["file"]
REFERENCE_TEXT = entry["transcript"]
SENTENCE = "دیروز در کتابخانه شهر، کتابی درباره تاریخ ایران خواندم."
RENDER = Path("/kaggle/working/higgs_worker.wav")
print("reference:", REFERENCE.name, f"{{entry['duration']:.1f}}s")
'''

WORKER = f'''\
worker_code, worker_seconds = run(
    [sys.executable, "/kaggle/working/drive_higgs.py", str(REFERENCE), REFERENCE_TEXT,
     SENTENCE, str(RENDER), str(REPO)],
    timeout={WORKER_TIMEOUT},
)

# The failure this hardware is known for writes a perfectly well-formed file, so
# the render is inspected. The worker refuses a near-constant render itself; this
# is the same check from the outside, and it also reports the pace.
if RENDER.is_file():
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(RENDER)
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    distinct = len(np.unique(samples))
    rms = float(np.sqrt((samples ** 2).mean())) if samples.size else 0.0
    print()
    print(f"render      : {{samples.size / rate:.2f}} s at {{rate}} Hz")
    print(f"peak / rms  : {{float(np.abs(samples).max()):.3f}} / {{rms:.4f}}")
    print(f"distinct    : {{distinct}} sample values")
    verdict = []
    if not np.isfinite(samples).all():
        verdict.append("NOT FINITE")
    if distinct < 1000:
        verdict.append("CONSTANT OR NEARLY SO - this is not speech")
    if rms < 1e-3:
        verdict.append("SILENT")
    print("VERDICT     :", " | ".join(verdict) if verdict else "looks like audio")
    if worker_seconds and samples.size:
        print(f"real-time factor, load included: {{worker_seconds / (samples.size / rate):.0f}}x")
else:
    print("no render was written")
'''

BENCH = f'''\
# The same render through the engine, the config and the scoring: one speaker,
# one sentence. Its held-out similarity is the first number this model has on
# this scale.
code, elapsed = run([sys.executable, "scripts/benchmark.py", "--label", "higgs-smoke",
                     "--engine", "higgs", "--device", "cuda",
                     "--speakers", "cv02", "--sentences", "s1"],
                    timeout={BENCH_TIMEOUT}, cwd=str(REPO))

path = REPO / "eval/results/higgs-smoke.json"
if path.is_file():
    data = json.loads(path.read_text())
    print("summary:", json.dumps(data["summary"], ensure_ascii=False))
    for row in data["rows"]:
        print("row    :", json.dumps({{k: row.get(k) for k in
              ("speaker", "sentence", "seconds", "held_out_similarity",
               "character_error_rate", "error")}}, ensure_ascii=False))
    print()
    print(f"that pace over 27 cells: {{elapsed * 27 / 3600:.1f}} h, load counted once")
else:
    print("no results file - the cell did not get far enough to write one")
'''

CONTROL = f'''\
# The scale check, last because it is the cheapest to lose.
code, elapsed = run([sys.executable, "scripts/benchmark.py", "--label", "baseline-higgs-env",
                     "--score-only", "--replay-from", "baseline", "--device", "cuda"],
                    timeout={BENCH_TIMEOUT}, cwd=str(REPO))
path = REPO / "eval/results/baseline-higgs-env.json"
if path.is_file():
    value = json.loads(path.read_text())["summary"]["held_out_mean"]
    drift = value - {KNOWN_BASELINE}
    print(f"control: {{value:.4f}} vs {KNOWN_BASELINE} known ({{drift:+.4f}})")
    print("on the established scale" if abs(drift) < 0.0005 else "OFF THE ESTABLISHED SCALE")
else:
    print("control did not produce a results file")
'''

FULL = f'''\
# The full 27 cells, but only if the single scored cell above came back with a
# number. A broken engine that reached this point would otherwise spend the
# session rendering 27 failures, which is what the smoke cell exists to prevent.
smoke = REPO / "eval/results/higgs-smoke.json"
ready = False
if smoke.is_file():
    data = json.loads(smoke.read_text())
    row = (data["rows"] or [{{}}])[0]
    ready = data["summary"]["renders"] == 1 and row.get("held_out_similarity") is not None
    print("smoke cell:", json.dumps(data["summary"], ensure_ascii=False))
if not ready:
    print("no scored render from the smoke cell - not starting the full run")
else:
    code, elapsed = run([sys.executable, "scripts/benchmark.py", "--label", "higgs",
                         "--engine", "higgs", "--device", "cuda"],
                        timeout=7200, cwd=str(REPO))
    path = REPO / "eval/results/higgs.json"
    if path.is_file():
        data = json.loads(path.read_text())
        print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    else:
        print("no results file for the full run")
'''

COLLECT = '''\
destination = Path("/kaggle/working/results")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)
renders = Path("/kaggle/working/higgs_renders")
renders.mkdir(exist_ok=True)
for path in list((REPO / "outputs/benchmark/higgs-smoke").glob("*.wav")
                    + list((REPO / "outputs/benchmark/higgs").glob("*.wav"))) + [RENDER]:
    if path.is_file():
        shutil.copy2(path, renders / path.name)
print("saved:", sorted(p.name for p in renders.glob("*.wav")))
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
    sources = [SETUP, ENVIRONMENT, DOWNLOAD, DRIVER, WORKER, BENCH, CONTROL, FULL, COLLECT]
    for index, source in enumerate(sources):
        compile(
            "".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
            f"<cell {index}>",
            "exec",
        )

    defines = [i for i, s in enumerate(sources) if "def run(" in s]
    uses = [i for i, s in enumerate(sources) if "run(" in s and "def run(" not in s]
    assert defines and min(defines) < min(uses), "run() is used before it is defined"
    assert sources.index(DRIVER) < sources.index(WORKER), "the driver is written after it is used"

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
