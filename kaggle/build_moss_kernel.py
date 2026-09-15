#!/usr/bin/env python3
"""Generate the Kaggle notebook that measures MOSS-TTS over the full benchmark.

Why a kernel of its own: MOSS is pinned to transformers 5.0.x - later releases
removed generation internals its remote code calls, and upstream's card asks for
5.0.0 - while the rest of this project runs 5.14, and the MOSS engine runs
in-process, so it cannot share an interpreter with anything else here.

What happened before this version, because it decides the design:

* The first run went about seven hours and was cancelled having produced
  nothing. The second, bounded at two hours, froze on its very first cell: its
  log shows transformers materialising the MOSS audio tokenizer's weights on
  the CPU and stopping at parameter 866 of 1600, followed by silence until the
  timeout. A single-cell run in an identical environment - same transformers,
  torch and hardware - loaded the same weights without trouble.
* transformers 5.0.0 loads weights through a thread pool by default and skips
  it when ``HF_DEACTIVATE_ASYNC_LOAD`` is set (read from the v5.0.0 source). A
  controlled load test - four loads with the pool off, three with it on, one
  session - came back inconclusive: nothing hung either way, so it could not say
  whether the pool is the cause.

So this run does not bet on the cause. It is built to finish either way:

* **threaded weight loading is off.** The load test measured no speed cost.
* **a silence watchdog replaces the total timeout.** Every byte of output
  counts as activity, a progress bar redrawing with a carriage return included,
  so a load or download that is moving is never mistaken for a hang. A run that
  prints nothing for ten minutes has its whole process group killed and is
  retried. Only silence is retried: a run that exits with an error is reported,
  because repeating a deterministic crash would just spend the budget. The
  render stage is capped at an hour, and no attempt starts with less than twenty
  minutes of that hour left.
* **the weights are downloaded first**, into the engine's own cache, so the
  load no longer begins a second after its download finished - the one timing
  difference left between the run that hung and the test that did not.
* **the scale control runs last.** It has already reproduced the established
  0.5196 exactly in this environment, and in the second run it hung straight
  after a killed render, so it goes where it cannot cost the measurement.
* **the repo copy lives in /tmp**, so the kernel's output is only what this run
  measured. An earlier fetch of an output made mostly of that copy broke off.

One comparability problem is recorded rather than hidden. With MOSS on the
card, faster-whisper cannot allocate and the pipeline falls back to a
Transformers Whisper implementation. That fallback scored the baseline renders'
*similarity* identically, but MOSS's CER then comes from a different recogniser
than every other run's. The notebook detects the fallback and says so.

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

#: A run that prints nothing at all for this long is treated as hung.
SILENCE = 600
#: Hard cap on the whole render stage, retries included.
RENDER_BUDGET = 60 * 60
#: No attempt starts with less than this left - it could not finish.
MIN_ATTEMPT = 20 * 60
MAX_ATTEMPTS = 4
#: Held-out similarity the baseline renders are already known to score.
KNOWN_BASELINE = 0.5196

DOWNLOAD_SCRIPT = '''\
import os
import time

from huggingface_hub import snapshot_download

cache = os.path.expanduser("~/.cache/persian-voice-cloning")
for repo in ("OpenMOSS-Team/MOSS-TTS-Local-Transformer", "OpenMOSS-Team/MOSS-Audio-Tokenizer"):
    start = time.time()
    snapshot_download(repo, cache_dir=cache)
    print(repo, f"cached in {time.time() - start:.0f}s", flush=True)
'''

SETUP = '''\
# MOSS-TTS against the same nine speakers, sentences, seed and reference clips.
# transformers is pinned because MOSS's remote code calls generation internals
# that 5.1 removed; upstream's card asks for 5.0.0 for the same reason.
!pip -q install "transformers>=5.0,<5.1" accelerate speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime
'''

ENV_HEAD = r'''import json
import os
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
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

# /tmp rather than /kaggle/working: everything under /kaggle/working becomes
# kernel output, and a fetch of an output that was mostly this copy broke off.
REPO = Path("/tmp/repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(found[0].parent.parent, REPO)

version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__)
print("gpus        :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert (5, 0) <= version < (5, 1), f"MOSS needs transformers 5.0.x, found {transformers.__version__}"
assert torch.cuda.device_count() >= 1, "no GPU - set the accelerator"

'''

ENV_FUNCS = r'''
#: Output worth knowing about even after it has scrolled out of the tail.
SIGNALS = {"asr_fallback": rb"could not load 'large-v3-turbo'"}


def benchmark(*arguments):
    return [sys.executable, "-u", "scripts/benchmark.py", *arguments]


def watched(command, label, *, silence, cap, env_extra=None, cwd=None, poll=5.0):
    """Run a command, echo its output, and kill its process group if it goes quiet.

    Any byte counts as activity - a progress bar redrawing with a carriage return
    included - so a download or a weight load that is moving is never taken for a
    hang, and one that has frozen is caught within `silence` seconds.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.update(env_extra or {})
    print(f"[{label}] {' '.join(command)}", flush=True)
    start = last = time.time()
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    chunks = queue.Queue()

    def pump():
        while True:
            data = os.read(proc.stdout.fileno(), 65536)
            if not data:
                chunks.put(None)
                return
            chunks.put(data)

    threading.Thread(target=pump, daemon=True).start()
    tail, cells, seen, stopped, exited = b"", {}, set(), None, None
    while True:
        try:
            data = chunks.get(timeout=poll)
        except queue.Empty:
            data = b""
        now = time.time()
        if data is None:
            break
        if data:
            last = now
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()
            window = tail[-128:] + data
            tail = (tail + data)[-8000:]
            for match in re.finditer(rb"\[\s*(\d+)/(\d+)\]", window):
                cells.setdefault(int(match.group(1)), round(now - start, 1))
            for name, pattern in SIGNALS.items():
                if re.search(pattern, window):
                    seen.add(name)
        elif proc.poll() is not None:
            # Exited. Give the pipe a few seconds to hand over its last output -
            # the pump thread can lag the exit - then stop waiting on anything
            # that may still be holding it open.
            exited = exited or now
            if now - exited > 3:
                break
        if now - last > silence:
            stopped = "silent"
            break
        if now - start > cap:
            stopped = "capped"
            break
    if stopped:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        code = None
    seconds = time.time() - start
    status = stopped or ("ok" if code == 0 else "error")
    reason = {"silent": f"printed nothing for {silence / 60:.0f} min",
              "capped": f"ran past its {cap / 60:.0f} min cap"}
    print(f"\n[{label}] {reason.get(status, f'exit {code}')} - {seconds / 60:.1f} min", flush=True)
    text = tail.decode(errors="replace").replace("\r", "\n")
    return {"label": label, "status": status, "code": code, "seconds": round(seconds, 1),
            "last_output": [line for line in text.splitlines() if line.strip()][-3:],
            "cells": cells, "seen": sorted(seen)}


def retrying(command, label, *, silence, budget, min_attempt, max_attempts,
             env_extra=None, cwd=None, poll=5.0):
    """Retry a run only when it went silent - the failure seen twice before.

    A run that exits with an error is reported and not repeated: re-running a
    deterministic crash would only spend the budget. Each attempt is capped at
    the budget that remains, and none starts with less than `min_attempt` left.
    """
    attempts = []
    deadline = time.time() + budget
    for number in range(1, max_attempts + 1):
        remaining = deadline - time.time()
        if remaining < min_attempt:
            print(f"[{label}] not starting attempt {number}: "
                  f"{remaining / 60:.0f} min of budget left", flush=True)
            break
        result = watched(command, f"{label} #{number}", silence=silence, cap=remaining,
                         env_extra=env_extra, cwd=cwd, poll=poll)
        attempts.append(result)
        if result["status"] != "silent":
            break
        print(f"[{label}] went silent after: {result['last_output']}", flush=True)
    return attempts
'''

ENVIRONMENT = (
    ENV_HEAD
    + f"SILENCE = {SILENCE}\n"
    + f"RENDER_BUDGET = {RENDER_BUDGET}\n"
    + f"MIN_ATTEMPT = {MIN_ATTEMPT}\n"
    + f"MAX_ATTEMPTS = {MAX_ATTEMPTS}\n"
    + f"KNOWN_BASELINE = {KNOWN_BASELINE}\n"
    + f"DOWNLOAD_SCRIPT = {DOWNLOAD_SCRIPT!r}\n\n"
    + ENV_FUNCS
)

DOWNLOAD = '''\
# MOSS's weights go into the engine's own cache before the render starts, so the
# load no longer begins a second after its download finished, and a stalled
# download is caught here on its own rather than inside the render.
downloads = retrying([sys.executable, "-u", "-c", DOWNLOAD_SCRIPT], "download",
                     silence=240, budget=15 * 60, min_attempt=3 * 60, max_attempts=2)
print("MOSS weights cached:", bool(downloads) and downloads[-1]["status"] == "ok", flush=True)
'''

RENDER = '''\
# The measurement. Threaded weight loading is off - it cost nothing in the load
# test - and a run that prints nothing for ten minutes is killed and retried, so
# the hang seen twice costs minutes instead of hours even if the thread pool
# turns out not to be what causes it.
render = retrying(benchmark("--label", "moss", "--engine", "moss", "--device", "cuda"), "moss",
                  silence=SILENCE, budget=RENDER_BUDGET, min_attempt=MIN_ATTEMPT,
                  max_attempts=MAX_ATTEMPTS, env_extra={"HF_DEACTIVATE_ASYNC_LOAD": "1"}, cwd=REPO)
'''

REPORT = '''\
print(f"{'attempt':10s} {'status':8s} {'minutes':>7s}  cells reached")
for attempt in render:
    reached = max(attempt["cells"]) if attempt["cells"] else 0
    print(f"{attempt['label']:10s} {attempt['status']:8s} {attempt['seconds'] / 60:7.1f}  {reached}/27")
    if attempt["status"] != "ok":
        for line in attempt["last_output"]:
            print("           last output:", line[-140:])

# Fixed before the run: what these outcomes do and do not show about the cause.
print()
if any(attempt["status"] == "silent" for attempt in render):
    print("A load went silent with threaded weight loading disabled: the thread pool is not the cause,"
          " or not the only one.")
elif render and render[0]["status"] == "ok":
    print("No hang in this session. That is consistent with the thread pool being the cause but does"
          " not show it - a run without the flag also succeeded once.")

finished = [attempt for attempt in render if attempt["status"] == "ok"]
if finished:
    marks = finished[-1]["cells"]
    gaps = [marks[k + 1] - marks[k] for k in sorted(marks) if k >= 2 and k + 1 in marks]
    if 1 in marks and 2 in marks:
        print(f"first cell, loading MOSS included: {marks[2] - marks[1]:.0f}s")
    if gaps:
        print(f"each later cell, render and score: median {statistics.median(gaps):.0f}s,"
              f" {min(gaps):.0f}-{max(gaps):.0f}s over {len(gaps)} cells")
    if "asr_fallback" in finished[-1]["seen"]:
        print("MOSS was transcribed by the Transformers Whisper fallback, not faster-whisper, because"
              " MOSS occupied the card. Its CER comes from a different recogniser than every other run"
              " and is not comparable with theirs. Similarity is unaffected.")

path = REPO / "eval/results/moss.json"
if path.is_file():
    summary = json.loads(path.read_text())["summary"]
    print("moss summary:", {k: summary.get(k) for k in ("held_out_mean", "cer_mean", "renders", "failures")})
else:
    print("no MOSS results file")
'''

CONTROL = '''\
# The scale check, last. It has reproduced 0.5196 exactly in this environment
# once already, and in an earlier run it hung straight after a killed render, so
# it goes where it cannot cost the measurement.
control = retrying(benchmark("--label", "baseline-moss-env", "--score-only", "--replay-from",
                             "baseline", "--device", "cuda"), "control",
                   silence=SILENCE, budget=15 * 60, min_attempt=5 * 60, max_attempts=1, cwd=REPO)
path = REPO / "eval/results/baseline-moss-env.json"
if path.is_file():
    value = json.loads(path.read_text())["summary"]["held_out_mean"]
    print(f"control: {value:.4f} vs {KNOWN_BASELINE} known ({value - KNOWN_BASELINE:+.4f}) ->",
          "on the established scale" if abs(value - KNOWN_BASELINE) < 0.0005 else "OFF THE ESTABLISHED SCALE")
else:
    print("control produced no results file")
'''

COLLECT = '''\
# Only what this kernel produced: the repo copy carries every other run's
# results too, and shipping those would let stale copies ride along.
working = Path("/kaggle/working")
(working / "results").mkdir(exist_ok=True)
for name in ("moss.json", "baseline-moss-env.json"):
    source = REPO / "eval/results" / name
    if source.is_file():
        shutil.copy2(source, working / "results" / name)
renders = working / "moss_renders"
renders.mkdir(exist_ok=True)
for wav in (REPO / "outputs/benchmark/moss").glob("*.wav"):
    shutil.copy2(wav, renders / wav.name)
record = {"download": downloads, "render": render, "control": control}
(working / "moss_attempts.json").write_text(json.dumps(record, indent=2, default=str))
print("saved", len(list(renders.glob("*.wav"))), "renders and",
      sorted(p.name for p in (working / "results").glob("*.json")))
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
    compile(DOWNLOAD_SCRIPT, "<download script>", "exec")

    sources = [SETUP, ENVIRONMENT, DOWNLOAD, RENDER, REPORT, CONTROL, COLLECT]
    for index, source in enumerate(sources):
        compile(
            "".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
            f"<cell {index}>",
            "exec",
        )

    # Helpers are defined in the environment cell and used by the cells after
    # it; a NameError there would only surface at run time, so check here.
    for name in ("benchmark", "watched", "retrying"):
        defines = [i for i, s in enumerate(sources) if f"def {name}(" in s]
        uses = [i for i, s in enumerate(sources) if f"{name}(" in s and f"def {name}(" not in s]
        assert defines and (not uses or min(defines) < min(uses)), f"{name}() is used before it is defined"

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

    print(f"wrote {path} ({len(sources)} cells and the download script, all compile)")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {metadata['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
