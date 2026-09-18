#!/usr/bin/env python3
"""Generate a Kaggle notebook that smoke-tests MOSS-TTS v1.5 at its official recipe.

The order is the one that makes a failure cheap to read, and nothing expensive
starts until the step before it has shown it works:

1. the official stack installs: transformers 5.0.0 on the image's torch, which
   must be the pinned 2.9.1+cu128 - checked, not assumed
2. the runner is checked on this notebook's own display with a model-free job
   shaped like the one that stalled (scripts/runner_check.py); nothing is
   downloaded unless it passes
3. the pinned model (be7766a6) and codec (f6e20e54) download
4. one render through a driver that knows only the worker protocol, with load
   time, render time and memory on each card recorded, and the audio inspected
   - 48 kHz, two channels, finite, not a constant
5. the same cell through the engine and the scorer
6. the baseline renders re-scored here, compared cell by cell
7. only if every gate above passed and the measured cost fits: the 27-cell
   development benchmark

Precision is the card's: bf16 on CUDA, which a T4 runs emulated. If it cannot,
the run stops at step 4 and the log says why; it is never switched to fp16.

Every command's output goes to its own file under /kaggle/working/logs, and
the notebook shows a bounded view of it: the earlier runner coupled the two,
and a stalled notebook display stopped the worker (see kernel_gates.py and
eval/runs/persian-voice-moss15-probe-v2.json). For the same reason no cell
lets a subprocess write to the notebook's raw stdout - output is captured and
printed.

    .venv/bin/python kaggle/build_dataset.py            # the kernel mounts this
    .venv/bin/python kaggle/build_moss15_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/moss15
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "moss15"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-moss15"
MACHINE_SHAPE = "NvidiaTeslaT4"
SESSION = "kaggle:persian-voice-moss15-v3"

MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"

#: v1 and v2 went silent a second into loading and used their whole budgets -
#: most likely behind a stalled display, which the probe measured but they,
#: keeping no log file, could not show. Load plus one render needs far less;
#: the monitor says where time goes.
WORKER_TIMEOUT = 2700
BENCH_TIMEOUT = 3600
FULL_TIMEOUT = 4 * 3600

#: Seconds per cell for scoring on top of generation, for the budget estimate.
SCORING_ALLOWANCE = 20

SETUP = '''\
# The card's runtime stack is torch 2.9.1+cu128 (checked in the next cell) with
# transformers 5.0.0. torchcodec is part of that stack only for torchaudio.load,
# which the worker does not use - it reads references with soundfile and hands
# them to the processor's own encoder - so it is not installed.
!pip -q install "transformers==5.0.0" "accelerate>=1.10.1" speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime 2>&1 | tail -5
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


def captured(command):
    # Output is captured and printed - never written straight to the notebook's
    # raw stdout, which is what stalled before.
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


# An earlier kernel's transformers install pulled a torchvision built for
# another torch, and transformers' lazy import of PreTrainedModel died on it.
# If that happens here, torchvision goes - nothing in this run loads a vision
# model - and the removal is printed, not hidden.
probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"],
                       text=True, capture_output=True)
if probe.returncode != 0 and "torchvision" in probe.stderr:
    print("transformers cannot import beside this torchvision - removing torchvision")
    print(captured([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"])[1])
    probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"],
                           text=True, capture_output=True)
assert probe.returncode == 0, probe.stderr[-2000:]

import transformers

print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
assert torch.__version__.startswith("2.9.1"), "the official stack pins torch 2.9.1+cu128"
assert transformers.__version__ == "5.0.0", "the official stack pins transformers 5.0.0"
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("gpus        :", gpus)
print("capability  :", [torch.cuda.get_device_capability(i) for i in range(len(gpus))])
for i in range(len(gpus)):
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{{i}}      : {{free / 2**30:.1f}} GiB free of {{total / 2**30:.1f}}")
print("disk        :", f"{{shutil.disk_usage('/root').free / 2**30:.0f}} GiB free for the HF cache")
# The model (9.1 GB bf16) and the codec (about 4.3 GB bf16) together, plus the
# KV cache, do not fit one 15 GiB T4 with room to spare. The second card holds
# the codec. That is placement; precision is untouched.
assert len(gpus) >= 2, "this recipe places the codec on a second GPU - use T4 x2"

INPUT = Path("/kaggle/input")
found = sorted(INPUT.rglob("repo/scripts/benchmark.py"))
if not found:
    for path in sorted(p for p in INPUT.rglob("*") if p.is_dir())[:40]:
        print("   ", path)
    raise SystemExit("no repo under /kaggle/input - see the directories above")
REPO = Path("/kaggle/working/repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(found[0].parent.parent, REPO)
assert (REPO / "scripts/moss15_worker.py").is_file(), "the dataset predates the MOSS v1.5 worker"

os.environ["VOICECLONE_MOSS15_PYTHON"] = sys.executable
os.environ["PYTHONPATH"] = str(REPO / "src")
os.environ["VOICECLONE_SCORING_ENV"] = "{SESSION}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Sequential weight materialisation. The v1.0 run hung twice inside
# transformers' threaded loader and finished once this was set. It changes how
# weights are loaded, not what is computed.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"


# Every stage asks the gates before it starts; every command runs in its own
# process group with its output in its own log. See scripts/kernel_gates.py.
sys.path.insert(0, str(REPO / "scripts"))
from kernel_gates import (Gates, bench_verdict, budget_verdict, control_verdict,  # noqa: E402
                          report_group, run_bounded, worker_verdict)

GATES = Gates()
RUNS = {{}}
LOGS = Path("/kaggle/working/logs")
LOGS.mkdir(exist_ok=True)
CAPACITY = {{f"cuda:{{i}}": torch.cuda.get_device_properties(i).total_memory / 2**30
            for i in range(torch.cuda.device_count())}}
'''

RUNNER = '''\
# The job that stalled the last probe, without a model: a fake worker through the
# real worker's stdout claim and protocol, 3,000 lines and 20,000 progress
# updates, on this notebook's own display. Nothing is downloaded unless it passes.
from runner_check import check, diagnostics  # noqa: E402

ok, reason, run = check(run_bounded, Path("/kaggle/working/runner_check"), REPO / "scripts/moss15_worker.py")
RUNS["runner_check"] = run.record()
print(run.summary())
seen, line = diagnostics(report_group)
print("descriptor report:", "works" if seen else "DOES NOT WORK", "-", line)
GATES.record("runner", ok, reason)
'''

DOWNLOAD = f'''\
from huggingface_hub import snapshot_download

if not GATES.skip("download", ("runner",)):
    try:
        start = time.time()
        MODEL_DIR = Path(snapshot_download(repo_id="{MODEL}", revision="{MODEL_REVISION}"))
        CODEC_DIR = Path(snapshot_download(repo_id="{CODEC}", revision="{CODEC_REVISION}"))
        sizes = {{d.name: sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 2**30
                 for d in (MODEL_DIR, CODEC_DIR)}}
        GATES.record("download", True, f"{{(time.time() - start) / 60:.1f}} min, " +
                     ", ".join(f"{{n}} {{s:.2f}} GiB" for n, s in sizes.items()))
    except Exception as exc:
        GATES.record("download", False, f"{{type(exc).__name__}}: {{exc}}")
'''

DRIVER = '''\
driver = Path("/kaggle/working/drive_moss15.py")
driver.write_text("""
import json, subprocess, sys, time

model_dir, codec_dir, reference, text, output, cwd = sys.argv[1:7]
command = [sys.executable, "scripts/moss15_worker.py", "--model-dir", model_dir,
           "--codec-dir", codec_dir, "--device", "cuda", "--codec-device", "auto"]
start_load = time.time()
worker = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1)

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
ready_at = time.time()
print("READY", json.dumps(ready), flush=True)
print(f"LOAD {ready_at - start_load:.1f}s", flush=True)
if not ready.get("ready"):
    raise SystemExit("worker did not load")
job = {"text": text, "reference": reference, "output": output, "seed": 42, "language": "Persian",
       "audio_temperature": 1.7, "audio_top_p": 0.8, "audio_top_k": 25,
       "audio_repetition_penalty": 1.0, "max_new_tokens": 4096}
start = time.time()
worker.stdin.write(json.dumps(job, ensure_ascii=False) + chr(10))
worker.stdin.flush()
answer = reply()
render_seconds = time.time() - start
print(f"RENDER {render_seconds:.1f}s", flush=True)
print("RESULT", json.dumps(answer), flush=True)
worker.stdin.write(json.dumps({"op": "quit"}) + chr(10))
worker.stdin.flush()
worker.wait(timeout=120)
with open("/kaggle/working/moss15_smoke.json", "w") as handle:
    json.dump({"ready": ready, "load_seconds": ready_at - start_load, "result": answer,
               "render_seconds": render_seconds}, handle)
if not answer.get("ok"):
    raise SystemExit("the render failed")
""", encoding="utf-8")

manifest = json.loads((REPO / "eval/corpus/manifest.json").read_text(encoding="utf-8"))
entry = next(e for e in manifest["entries"] if e["speaker"] == "cv02" and e["role"] == "reference")
REFERENCE = REPO / "eval/corpus" / entry["file"]
SENTENCE = "دیروز در کتابخانه شهر، کتابی درباره تاریخ ایران خواندم."
RENDER = Path("/kaggle/working/moss15_worker.wav")
SMOKE = Path("/kaggle/working/moss15_smoke.json")
'''

WORKER = f'''\
FACTS = {{}}
if not GATES.skip("worker", ("download",)):
    run = run_bounded(
        [sys.executable, "/kaggle/working/drive_moss15.py", str(MODEL_DIR), str(CODEC_DIR),
         str(REFERENCE), SENTENCE, str(RENDER), str(REPO)],
        timeout={WORKER_TIMEOUT}, monitor_every=120, log=LOGS / "worker.log")
    RUNS["worker"] = run.record()
    print(run.summary())
    print(captured(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv"])[1])
    ok, reason, FACTS = worker_verdict(run, RENDER, SMOKE, expect_rate=48000, expect_channels=2)
    GATES.record("worker", ok, reason)
    if FACTS:
        print("load s:", round(FACTS.get("load_seconds", 0), 1),
              "| render s:", round(FACTS.get("render_seconds", 0), 1),
              "| peaks after load:", FACTS.get("ready", {{}}).get("memory_gib_after_load"),
              "| peaks during render:", FACTS.get("result", {{}}).get("memory_gib"))
'''

BENCH = f'''\
if not GATES.skip("bench", ("worker",)):
    run = run_bounded(
        [sys.executable, "scripts/benchmark.py", "--label", "moss15-smoke", "--engine", "moss15",
         "--device", "cuda", "--speakers", "cv02", "--sentences", "s1"],
        timeout={BENCH_TIMEOUT}, cwd=str(REPO), monitor_every=300, log=LOGS / "bench.log")
    RUNS["bench"] = run.record()
    print(run.summary())
    ok, reason = bench_verdict(run, REPO / "eval/results/moss15-smoke.json")
    GATES.record("bench", ok, reason)
'''

CONTROL = '''\
if not GATES.skip("control", ("bench",)):
    run = run_bounded(
        [sys.executable, "scripts/benchmark.py", "--label", "baseline-moss15-env",
         "--score-only", "--replay-from", "baseline", "--device", "cuda"],
        timeout=3600, cwd=str(REPO), monitor_every=300, log=LOGS / "control.log")
    RUNS["control"] = run.record()
    print(run.summary())
    check = None
    if run.ok and (REPO / "eval/results/baseline-moss15-env.json").is_file():
        import compare_runs

        results = REPO / "eval/results"
        check = compare_runs.scale_control(compare_runs.load("baseline", results),
                                           compare_runs.load("baseline-moss15-env", results))
    ok, reason = control_verdict(run, check)
    GATES.record("control", ok, reason)
'''

FULL = f'''\
# The full run is priced from what the smoke run measured, not guessed: the
# worker's load time, one render's generation time plus a scoring allowance,
# and the peak memory each card reached, against the time budget and each
# card's capacity.
if not GATES.skip("budget", ("worker", "bench", "control")):
    peaks = {{}}
    for snapshot in (FACTS.get("ready", {{}}).get("memory_gib_after_load", {{}}),
                     FACTS.get("result", {{}}).get("memory_gib", {{}})):
        for card, stats in snapshot.items():
            peaks[card] = max(peaks.get(card, 0.0), stats.get("peak_reserved", 0.0))
    ok, reason = budget_verdict(load_seconds=FACTS.get("load_seconds", 0.0),
                                cell_seconds=FACTS["result"]["generate_seconds"] + {SCORING_ALLOWANCE},
                                cells=27, timeout={FULL_TIMEOUT}, peaks_gib=peaks, capacity_gib=CAPACITY)
    GATES.record("budget", ok, reason)

if not GATES.skip("full", ("budget",)):
    run = run_bounded(
        [sys.executable, "scripts/benchmark.py", "--label", "moss15", "--engine", "moss15", "--device", "cuda"],
        timeout={FULL_TIMEOUT}, cwd=str(REPO), monitor_every=600, log=LOGS / "full.log")
    RUNS["full"] = run.record()
    print(run.summary())
    path = REPO / "eval/results/moss15.json"
    summary = json.loads(path.read_text())["summary"] if path.is_file() else {{}}
    complete = (run.ok and summary.get("renders") == 27
                and not summary.get("failures") and not summary.get("untranscribed"))
    GATES.record("full", complete, json.dumps(summary) if summary else f"{{run.outcome}}, no results file")
'''

COLLECT = '''\
destination = Path("/kaggle/working/results")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)
native = REPO / "outputs/native/moss15"
if native.is_dir():
    shutil.copytree(native, Path("/kaggle/working/moss15_native"), dirs_exist_ok=True)
renders = Path("/kaggle/working/moss15_renders")
renders.mkdir(exist_ok=True)
for folder in ("moss15-smoke", "moss15"):
    for path in sorted((REPO / "outputs/benchmark" / folder).glob("*.wav")):
        shutil.copy2(path, renders / f"{folder}__{path.name}")
if RENDER.is_file():
    shutil.copy2(RENDER, renders / RENDER.name)
for extra in (SMOKE,):
    if extra.is_file():
        shutil.copy2(extra, destination / extra.name)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
print("gates:", json.dumps(GATES.summary(), indent=1))
print("saved:", len(list(renders.glob("*.wav"))), "harness renders,",
      len(list(Path("/kaggle/working/moss15_native").glob("*.wav"))) if native.is_dir() else 0, "native")
'''


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main() -> int:
    sources = [SETUP, ENVIRONMENT, RUNNER, DOWNLOAD, DRIVER, WORKER, BENCH, CONTROL, FULL, COLLECT]
    for index, source in enumerate(sources):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [cell(s) for s in sources],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = OUT_DIR / f"{SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{USER}/{SLUG}", "title": SLUG, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(sources)} cells, all compile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
