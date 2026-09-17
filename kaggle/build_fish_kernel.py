#!/usr/bin/env python3
"""Generate a Kaggle notebook that proves out Fish Audio S2 Pro, cheaply first.

S2 Pro is the strongest claim left unmeasured, and it is also the most
expensive thing this project has tried to run: 11 GB of weights, a pinned
dependency world, and upstream's own documentation asking for 24 GB of VRAM
against the 15 GB a T4 offers. So this notebook does not attempt the full 27
cells. It answers, in order, the questions that decide whether the full run is
worth a session:

1. does the dependency world install at all next to the harness's own stack
2. does the model load inside 15 GB once the KV cache is bounded
3. does one render come back as *audio* - finite, varying, the right length
4. does a cell scored through the engine land on the established scale

Question 3 is asked with a deliberate paranoia. The one public attempt at
running Higgs TTS 3 on a T4 produced perfectly formed WAV files in which every
sample was -32768: a serving stack that reports success and returns silence is
a real failure mode on this hardware, and a benchmark that only checks "a file
appeared" would have scored it as a very strange speaker.

Three deliberate choices, all learned from the MOSS run:

* **the worker is tested before the harness.** If the engine, the config and
  the scoring all run at once, a failure is ambiguous. The first render goes
  through a driver that only knows about the worker protocol.
* **upstream's own CLI is the control**, and it only runs if the worker fails -
  it costs another 9 GB load, and its whole purpose is to separate "this
  project's new code is wrong" from "the model does not work here".
* **everything is bounded.** Each step gets a hard timeout, so a pathological
  load costs minutes instead of the session's GPU budget.

    .venv/bin/python kaggle/build_fish_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/fish
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "fish"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-fish-smoke"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: Fetching 11 GB may be slow; anything past this is not a download problem.
DOWNLOAD_TIMEOUT = 2400

#: Loading 9.1 GB of weights plus a 1.9 GB codec, then one render.
WORKER_TIMEOUT = 1500

#: The same again through the harness, with scoring models on top.
BENCH_TIMEOUT = 1800

#: What the baseline renders are already known to score, for the scale check.
KNOWN_BASELINE = 0.5196

SETUP = '''\
# fish-speech is used from a checkout, not installed: nothing here needs its
# console scripts, and `pip install -e .` would drag in its pinned torch on top
# of the image's working CUDA build. Only the imports the inference path
# actually reaches are installed. transformers is pinned to what the checkpoint
# was saved with - fish-speech declares <=4.57.3 - and the scale control at the
# end says whether the scoring stack minds.
!pip -q install descript-audio-codec loralib loguru hydra-core omegaconf einops "transformers==4.57.1" speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime 2>&1 | tail -5
!git clone --depth 1 https://github.com/fishaudio/fish-speech /kaggle/working/fish-speech 2>&1 | tail -2
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

FISH = Path("/kaggle/working/fish-speech")
assert (FISH / "fish_speech" / "models" / "text2semantic" / "inference.py").is_file(), FISH

# The engine starts the worker as a subprocess, which inherits this environment:
# one interpreter here, so point the engine at it, and put fish-speech on the
# path for the worker's own imports.
os.environ["VOICECLONE_FISH_PYTHON"] = sys.executable
os.environ["PYTHONPATH"] = f"{FISH}:{REPO / 'src'}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# The run that first reached a render died 54 MiB short of the card, and this
# is what the allocator's own message asks for when that happens.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("torch       :", torch.__version__)
print("gpus        :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# Turing has no bfloat16 unit, so the worker drops to float16 - upstream's own
# --half. Both are 16-bit; which one ran is recorded with the results.
print("bf16 native :", torch.cuda.is_bf16_supported())
free, total = torch.cuda.mem_get_info()
print(f"vram        : {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")
assert torch.cuda.device_count() >= 1, "no GPU - set the accelerator"


def run(command, timeout=None, cwd=None):
    """Run a command, echoing it, and return (exit code, wall-clock seconds)."""
    print("$", " ".join(str(part) for part in command), flush=True)
    start = time.time()
    try:
        code = subprocess.run(command, cwd=cwd, text=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {timeout}s - called off", flush=True)
        code = -1
    elapsed = time.time() - start
    print(f"exit {code} in {elapsed / 60:.1f} min", flush=True)
    return code, elapsed
'''

IMPORTS = '''\
# fish-speech is imported from a checkout with a hand-picked dependency list, and
# the first run of this notebook died on one name that list had missed - after
# paying for the install and the 11 GB download. So the import chain is walked
# here, before anything expensive, and a module it turns out to need is fetched
# by name. Each install is printed: the point is to finish the run *and* to end
# up with the list this notebook should have shipped with.
sys.path.insert(0, str(FISH))
for attempt in range(5):
    try:
        from fish_speech.models.text2semantic.inference import (  # noqa: F401
            decode_to_audio, encode_audio, generate_long, init_model, load_codec_model,
        )
        print("import chain complete" + (" after installing" if attempt else ""))
        break
    except ModuleNotFoundError as exc:
        missing = exc.name.split(".")[0]
        print(f"missing module: {missing} - installing", flush=True)
        outcome = subprocess.run([sys.executable, "-m", "pip", "install", "-q", missing],
                                 text=True, capture_output=True)
        if outcome.returncode != 0:
            print(outcome.stderr[-2000:])
            raise SystemExit(f"could not install {missing}; the PyPI name may differ")
else:
    raise SystemExit("still missing modules after five installs")
'''

DOWNLOAD = f'''\
# 11 GB: two 16-bit shards of the dual-AR transformer and a 1.9 GB codec. It
# lands in the HF cache rather than /kaggle/working, which has 20 GB and has to
# hold the renders.
from huggingface_hub import snapshot_download

start = time.time()
CHECKPOINT = Path(snapshot_download(repo_id="fishaudio/s2-pro"))
print(f"downloaded in {{(time.time() - start) / 60:.1f}} min -> {{CHECKPOINT}}")
for path in sorted(CHECKPOINT.iterdir()):
    if path.is_file():
        print(f"   {{path.stat().st_size / 2**30:6.2f}} GB  {{path.name}}")
'''

DRIVER = '''\
# A driver that knows only the worker protocol: one job in, one JSON line out.
# If this fails the problem is fish-speech or the new worker, not the engine,
# the config or the scoring - which is exactly what the next cell needs to know.
driver = Path("/kaggle/working/drive_fish.py")
driver.write_text("""
import json, subprocess, sys, time
from pathlib import Path

checkpoint, reference, reference_text, text, output = sys.argv[1:6]
command = [sys.executable, "scripts/fish_worker.py", "--checkpoint", checkpoint,
           "--device", "cuda", "--precision", "bfloat16", "--max-seq-len", "4096"]
start = time.time()
worker = subprocess.Popen(command, cwd=sys.argv[6], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)

def reply():
    while True:
        line = worker.stdout.readline()
        if not line:
            raise SystemExit(f"worker exited with {worker.poll()}")
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            continue

ready = reply()
print("READY", json.dumps(ready), flush=True)
print(f"LOAD {time.time() - start:.1f}s", flush=True)
if not ready.get("ready"):
    raise SystemExit("worker did not load")

job = {"text": text, "reference": reference, "reference_text": reference_text,
       "output": output, "seed": 42}
start = time.time()
worker.stdin.write(json.dumps(job, ensure_ascii=False) + chr(10))
worker.stdin.flush()
answer = reply()
print(f"RENDER {time.time() - start:.1f}s", flush=True)
print("RESULT", json.dumps(answer), flush=True)
worker.stdin.write(json.dumps({"op": "quit"}) + chr(10))
worker.stdin.flush()
""", encoding="utf-8")

manifest = json.loads((REPO / "eval/corpus/manifest.json").read_text(encoding="utf-8"))
entry = next(e for e in manifest["entries"] if e["speaker"] == "cv02" and e["role"] == "reference")
REFERENCE = REPO / "eval/corpus" / entry["file"]
REFERENCE_TEXT = entry["transcript"]
SENTENCE = "دیروز در کتابخانه شهر، کتابی درباره تاریخ ایران خواندم."
RENDER = Path("/kaggle/working/fish_worker.wav")
print("reference:", REFERENCE.name, f"{entry['duration']:.1f}s")
print("transcript:", REFERENCE_TEXT)
'''

WORKER = f'''\
worker_code, worker_seconds = run(
    [sys.executable, "/kaggle/working/drive_fish.py", str(CHECKPOINT), str(REFERENCE),
     REFERENCE_TEXT, SENTENCE, str(RENDER), str(REPO)],
    timeout={WORKER_TIMEOUT},
)

# "A file appeared" is not the same as "audio came back". The one public attempt
# at Higgs TTS 3 on a T4 returned well-formed WAVs in which every sample was
# -32768, so the render is inspected rather than trusted.
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

CLI_CONTROL = f'''\
# Only if the worker failed: upstream's own one-shot CLI, which loads the same
# 9 GB again. It separates "this project's new code is wrong" from "the model
# does not work on this hardware", and there is no reason to pay for it
# otherwise.
if worker_code == 0 and RENDER.is_file():
    print("worker path worked - skipping the upstream CLI control")
else:
    run([sys.executable, "fish_speech/models/text2semantic/inference.py",
         "--text", SENTENCE, "--prompt-audio", str(REFERENCE), "--prompt-text", REFERENCE_TEXT,
         "--checkpoint-path", str(CHECKPOINT), "--half", "--seed", "42",
         "--output", "/kaggle/working/fish_cli.wav", "--output-dir", "/kaggle/working/fish_cli"],
        timeout={WORKER_TIMEOUT}, cwd=str(FISH))
    print("cli render exists:", Path("/kaggle/working/fish_cli.wav").is_file())
'''

BENCH = f'''\
# Now the same render through the engine, the config and the scoring: one
# speaker, one sentence. This is what says the plumbing is right, and its
# held-out similarity is the first number this model has on this scale.
code, elapsed = run([sys.executable, "scripts/benchmark.py", "--label", "fish-smoke",
                     "--engine", "fish", "--device", "cuda",
                     "--speakers", "cv02", "--sentences", "s1"],
                    timeout={BENCH_TIMEOUT}, cwd=str(REPO))

path = REPO / "eval/results/fish-smoke.json"
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
# The scale check, last because it is the cheapest to lose. This environment
# pins transformers 4.57, a different dependency world from the numbers on
# record, so re-score renders whose answer is already established.
code, elapsed = run([sys.executable, "scripts/benchmark.py", "--label", "baseline-fish-env",
                     "--score-only", "--replay-from", "baseline", "--device", "cuda"],
                    timeout={BENCH_TIMEOUT}, cwd=str(REPO))
path = REPO / "eval/results/baseline-fish-env.json"
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
smoke = REPO / "eval/results/fish-smoke.json"
ready = False
if smoke.is_file():
    data = json.loads(smoke.read_text())
    row = (data["rows"] or [{{}}])[0]
    ready = data["summary"]["renders"] == 1 and row.get("held_out_similarity") is not None
    print("smoke cell:", json.dumps(data["summary"], ensure_ascii=False))
if not ready:
    print("no scored render from the smoke cell - not starting the full run")
else:
    code, elapsed = run([sys.executable, "scripts/benchmark.py", "--label", "fish",
                         "--engine", "fish", "--device", "cuda"],
                        timeout=7200, cwd=str(REPO))
    path = REPO / "eval/results/fish.json"
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
renders = Path("/kaggle/working/fish_renders")
renders.mkdir(exist_ok=True)
takes = [RENDER]
for folder in ("fish-smoke", "fish"):
    takes.extend(sorted((REPO / "outputs/benchmark" / folder).glob("*.wav")))
for path in takes:
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
    sources = [SETUP, ENVIRONMENT, IMPORTS, DOWNLOAD, DRIVER, WORKER, CLI_CONTROL, BENCH, CONTROL, FULL, COLLECT]
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
