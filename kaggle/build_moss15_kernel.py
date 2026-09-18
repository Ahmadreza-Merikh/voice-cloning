#!/usr/bin/env python3
"""Generate a Kaggle notebook that smoke-tests MOSS-TTS v1.5 at its official recipe.

The order is the one that makes a failure cheap to read, and nothing expensive
starts until the step before it has shown it works:

1. the official runtime is installed, not assumed: the code repository's
   pyproject at 934d6826 pins torch 2.9.1+cu128, torchaudio 2.9.1+cu128,
   transformers 5.0.0 and safetensors 0.6.2 - everything the model's code
   imports. Kaggle images move - the probe kernel got torch 2.10.0+cu128 on
   2026-09-18 while this kernel's image had 2.9.1 - so these are installed and
   checked, and every installed version is written to environment.json beside
   the official one
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

Version 4 changes one thing from version 3: the language tag, from "Persian"
(an analogy with the card's examples) to "Persian (Farsi)", the official
string (the card's table and the official v1.5 app at 934d6826). Readings
decided before it runs:

* CER is compared against this session's own baseline control. Against
  version 3's run directly only if the two sessions' baseline controls agree
  cell for cell in CER as well - then the recogniser behaved identically
* the tag string matters if the speaker-level interval of (v4 - v3) excludes
  0; with nine speakers an interval that includes 0 is weak evidence of no
  effect, not evidence of none
* the renders differ between the runs through sampling as well as the tag;
  that variation is inside the interval, not separated from it

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
SESSION = "kaggle:persian-voice-moss15-v4"
#: Result labels; version 3's were moss15, moss15-smoke and baseline-moss15-env.
LABEL = "moss15-fa"
SMOKE_LABEL = "moss15-fa-smoke"
CONTROL_LABEL = "baseline-moss15v4-env"
#: The official Persian tag - see OFFICIAL_PERSIAN_TAG in the engine.
LANGUAGE = "Persian (Farsi)"

MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"

#: The official stack, from pyproject.toml at 934d6826 (read 2026-09-18): its
#: ``torch-runtime`` extra and its core dependencies.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
OFFICIAL = {"torch": "2.9.1+cu128", "torchaudio": "2.9.1+cu128", "torchcodec": "0.8.1",
            "transformers": "5.0.0", "accelerate": ">=1.10.1", "safetensors": "0.6.2",
            "numpy": "2.1.0", "tqdm": "4.67.1", "einops": "0.8.1", "scipy": "1.16.2",
            "librosa": "0.11.0", "tiktoken": "0.12.0"}
#: What the model's remote code at the pinned revisions imports - so what can
#: change what is computed - and accelerate, which transformers loads through.
#: These are installed at the official versions and checked; a mismatch stops
#: the run. The rest of OFFICIAL is the repository's own tooling (app, training,
#: audio utilities) and torchcodec, which serves only torchaudio.load, which the
#: worker does not call. Those keep the image's versions, so the scorer's stack
#: is not disturbed, and every difference is written to environment.json.
PINNED = ("torch", "torchaudio", "transformers", "safetensors", "accelerate")

#: v1 and v2 went silent a second into loading and used their whole budgets -
#: most likely behind a stalled display, which the probe measured but they,
#: keeping no log file, could not show. Load plus one render needs far less;
#: the monitor says where time goes.
WORKER_TIMEOUT = 2700
BENCH_TIMEOUT = 3600
FULL_TIMEOUT = 4 * 3600

#: Seconds per cell for scoring on top of generation, for the budget estimate.
SCORING_ALLOWANCE = 20

SETUP = f'''\
# The official runtime stack first (see OFFICIAL in the builder), from PyTorch's
# cu128 index; torchvision at the version that belongs to that torch, so nothing
# on the image is left half-matched. Then the scorer's own packages.
!pip -q install "torch==2.9.1" "torchaudio==2.9.1" "torchvision==0.24.1" --index-url {TORCH_INDEX} 2>&1 | tail -3
!pip -q install "transformers==5.0.0" "accelerate>=1.10.1" "safetensors==0.6.2" speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime 2>&1 | tail -3
'''

ENVIRONMENT = f'''\
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def captured(command):
    # Output is captured and printed - never written straight to the notebook's
    # raw stdout, which is what stalled before.
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


# An earlier kernel's transformers install pulled a torchvision built for
# another torch, and transformers' lazy import of PreTrainedModel died on it.
# The matching torchvision is installed above; if the import still breaks on
# it, torchvision goes - nothing in this run loads a vision model - and the
# removal is printed, not hidden.
probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"],
                       text=True, capture_output=True)
if probe.returncode != 0 and "torchvision" in probe.stderr:
    print("transformers cannot import beside this torchvision - removing torchvision")
    print(captured([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"])[1])
    probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"],
                           text=True, capture_output=True)
assert probe.returncode == 0, probe.stderr[-2000:]

import torch

OFFICIAL = {OFFICIAL!r}
PINNED = {PINNED!r}


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def matches(have, want):
    if have is None:
        return False
    if want.startswith(">="):
        from packaging.version import Version
        return Version(have.split("+")[0]) >= Version(want[2:])
    return have == want or (have.split("+")[0] == want.split("+")[0] and "+" not in want)


stack = {{}}
for name, want in OFFICIAL.items():
    have = installed(name)
    stack[name] = {{"official": want, "installed": have, "matches": matches(have, want),
                   "pinned_here": name in PINNED}}
others = ("tokenizers", "huggingface-hub", "soundfile", "speechbrain", "faster-whisper", "ctranslate2",
          "jiwer", "torchvision", "ipykernel", "jupyter-client", "nbclient", "nbconvert", "papermill")
code, driver = captured(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
ENVIRONMENT_RECORD = {{
    "python": platform.python_version(),
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpus": [{{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}}
             for i in range(torch.cuda.device_count())],
    "nvidia_smi": driver,
    "official_stack": stack,
    "other_packages": {{name: installed(name) for name in others}},
}}
Path("/kaggle/working/environment.json").write_text(json.dumps(ENVIRONMENT_RECORD, indent=2))
for name, row in stack.items():
    mark = "ok" if row["matches"] else ("DIFFERS" if row["pinned_here"] else "differs - image's version kept")
    print(f"{{name:13}} official {{row['official']:12}} installed {{str(row['installed']):16}} {{mark}}")
print("cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version(), "|", driver)
for name in PINNED:
    assert stack[name]["matches"], f"{{name}} is not the official {{OFFICIAL[name]}}: {{stack[name]['installed']}}"

gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
for i in range(len(gpus)):
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{{i}}: {{gpus[i]}}, capability {{torch.cuda.get_device_capability(i)}}, "
          f"{{free / 2**30:.1f}} GiB free of {{total / 2**30:.1f}}")
print("disk:", f"{{shutil.disk_usage('/root').free / 2**30:.0f}} GiB free for the HF cache")
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
# Sequential weight materialisation, as in every earlier MOSS run. It changes
# how weights are loaded, not what is computed.
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
job = {"text": text, "reference": reference, "output": output, "seed": 42, "language": __LANGUAGE__,
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

DRIVER = DRIVER.replace("__LANGUAGE__", json.dumps(LANGUAGE))

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
        [sys.executable, "scripts/benchmark.py", "--label", "{SMOKE_LABEL}", "--engine", "moss15",
         "--device", "cuda", "--speakers", "cv02", "--sentences", "s1"],
        timeout={BENCH_TIMEOUT}, cwd=str(REPO), monitor_every=300, log=LOGS / "bench.log")
    RUNS["bench"] = run.record()
    print(run.summary())
    ok, reason = bench_verdict(run, REPO / "eval/results/{SMOKE_LABEL}.json")
    GATES.record("bench", ok, reason)
'''

CONTROL = f'''\
if not GATES.skip("control", ("bench",)):
    run = run_bounded(
        [sys.executable, "scripts/benchmark.py", "--label", "{CONTROL_LABEL}",
         "--score-only", "--replay-from", "baseline", "--device", "cuda"],
        timeout=3600, cwd=str(REPO), monitor_every=300, log=LOGS / "control.log")
    RUNS["control"] = run.record()
    print(run.summary())
    check = None
    if run.ok and (REPO / "eval/results/{CONTROL_LABEL}.json").is_file():
        import compare_runs

        results = REPO / "eval/results"
        check = compare_runs.scale_control(compare_runs.load("baseline", results),
                                           compare_runs.load("{CONTROL_LABEL}", results))
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
        [sys.executable, "scripts/benchmark.py", "--label", "{LABEL}", "--engine", "moss15", "--device", "cuda"],
        timeout={FULL_TIMEOUT}, cwd=str(REPO), monitor_every=600, log=LOGS / "full.log")
    RUNS["full"] = run.record()
    print(run.summary())
    path = REPO / "eval/results/{LABEL}.json"
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
for folder in __FOLDERS__:
    for path in sorted((REPO / "outputs/benchmark" / folder).glob("*.wav")):
        shutil.copy2(path, renders / f"{folder}__{path.name}")
if RENDER.is_file():
    shutil.copy2(RENDER, renders / RENDER.name)
for extra in (SMOKE, Path("/kaggle/working/environment.json")):
    if extra.is_file():
        shutil.copy2(extra, destination / extra.name)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
print("gates:", json.dumps(GATES.summary(), indent=1))
print("saved:", len(list(renders.glob("*.wav"))), "harness renders,",
      len(list(Path("/kaggle/working/moss15_native").rglob("*.wav"))) if native.is_dir() else 0, "native")
'''

COLLECT = COLLECT.replace("__FOLDERS__", repr((SMOKE_LABEL, LABEL)))


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
