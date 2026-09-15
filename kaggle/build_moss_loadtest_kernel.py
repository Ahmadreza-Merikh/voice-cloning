#!/usr/bin/env python3
"""Generate a Kaggle notebook that tests the MOSS load hang directly.

Two full MOSS runs produced nothing. The second one's log locates the stall
exactly: transformers was materialising the MOSS audio tokenizer's weights on
the CPU and stopped at parameter 866 of 1600, then printed nothing for the rest
of its two-hour bound. A single-cell run in an identical environment - the same
transformers 5.0.0, torch 2.10.0+cu128, two T4s - loaded the same weights
without stalling. Same versions, different outcome: that is what a race looks
like, not a slow or broken model.

transformers 5.0.0 loads weights through a thread pool by default
(``ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)`` in core_model_loading.py)
and skips the pool when ``HF_DEACTIVATE_ASYNC_LOAD`` is set. Both facts were
read from the v5.0.0 source, not recalled. If the race lives in that pool,
loading sequentially should never hang.

That is a hypothesis, and one successful load would not test it - the threaded
load also succeeded once. So this notebook loads MOSS repeatedly in a single
session, each load in its own process under a hard timeout:

* four loads with the pool disabled, first, so that a threaded load that hangs
  cannot leave the session in a state that makes them fail for another reason;
* then three loads with transformers' default threading, as the control.

The weights are downloaded once beforehand and every load runs with the Hub
offline, so a network stall cannot be mistaken for a load that hangs.

The verdict rule is fixed here, before the run, and the notebook prints it with
the results. The flag is the fix only if every sequential load succeeds *and*
at least one threaded load hangs in the same session. If nothing hangs, the
session did not reproduce the race and the sequential successes prove nothing.

Loading order does not change the weights that end up in memory, so disabling
the pool can cost load time and nothing else. It is not a change to the model
or to what it produces.

    .venv/bin/python kaggle/build_moss_loadtest_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/moss-loadtest
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "moss-loadtest"

USER = "marsmanam"
SLUG = "persian-voice-moss-loadtest"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: A healthy load of processor plus model is well inside this; a hang is not.
TRIAL_TIMEOUT = 300
#: The one-time download of both repositories.
WARMUP_TIMEOUT = 1500
SEQUENTIAL_TRIALS = 4
THREADED_TRIALS = 3

#: One load, in its own process. It first checks that transformers itself read
#: the flag the way this trial intends - otherwise a "sequential" trial could
#: silently be a threaded one and the comparison would be meaningless.
TRIAL_SCRIPT = '''\
import os
import sys
import time

import torch
import transformers.core_model_loading as loading

deactivated = loading.is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD")
print(f"FLAG {deactivated}", flush=True)
if deactivated != (os.environ.get("EXPECT_SEQUENTIAL") == "1"):
    print("FLAG MISMATCH - transformers did not read the variable as intended", flush=True)
    sys.exit(3)

from transformers import AutoModel, AutoProcessor

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
start = time.time()
AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
middle = time.time()
print(f"PROCESSOR {middle - start:.1f}", flush=True)
AutoModel.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
print(f"MODEL {time.time() - middle:.1f}", flush=True)
'''

SETUP = '''\
# The same installs as the MOSS runs that hung, so the loader, safetensors and
# huggingface_hub versions match the session being diagnosed.
!pip -q install "transformers>=5.0,<5.1" accelerate speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime
'''

ENV_HEAD = '''\
import os
import re
import statistics
import subprocess
import sys
import time

import torch
import transformers
import transformers.core_model_loading as loading

print("transformers  :", transformers.__version__)
print("torch         :", torch.__version__)
print("cpu count     :", os.cpu_count())
print("loader workers:", loading.GLOBAL_WORKERS)
assert transformers.__version__.startswith("5.0."), "the hang was seen on 5.0.x - test that version"

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
RESULTS = []
'''

ENV_TAIL = '''\
def _text(stream):
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode(errors="replace")


def trial(sequential, index):
    env = dict(os.environ)
    env.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
    env["EXPECT_SEQUENTIAL"] = "1" if sequential else "0"
    env["HF_HUB_OFFLINE"] = "1"
    if sequential:
        env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    name = f"{'sequential' if sequential else 'threaded'} #{index}"
    start = time.time()
    try:
        done = subprocess.run([sys.executable, "-c", TRIAL_SCRIPT], env=env, text=True,
                              capture_output=True, timeout=TRIAL_TIMEOUT)
        out, err = done.stdout, done.stderr
        status = "ok" if done.returncode == 0 else f"exit {done.returncode}"
    except subprocess.TimeoutExpired as exc:
        out, err = _text(exc.stdout), _text(exc.stderr)
        status = "HUNG"
    elapsed = time.time() - start
    processor = re.search(r"PROCESSOR ([\\d.]+)", out)
    model = re.search(r"MODEL ([\\d.]+)", out)
    flag = re.search(r"FLAG (\\w+)", out)
    print(f"{name:14s} {status:8s} {elapsed:6.1f}s   "
          f"processor {processor.group(1) + 's' if processor else '-':>7s}   "
          f"model {model.group(1) + 's' if model else '-':>7s}   "
          f"flag read as {flag.group(1) if flag else '?'}", flush=True)
    if status != "ok":
        # The last progress bar says where it stopped; tqdm redraws with \\r.
        for line in err.strip().splitlines()[-3:]:
            print("        ", line.split("\\r")[-1][-150:], flush=True)
    RESULTS.append({"sequential": sequential, "status": status, "elapsed": elapsed})
'''

ENVIRONMENT = (
    ENV_HEAD
    + f"TRIAL_TIMEOUT = {TRIAL_TIMEOUT}\n"
    + f"WARMUP_TIMEOUT = {WARMUP_TIMEOUT}\n"
    + f"SEQUENTIAL_TRIALS = {SEQUENTIAL_TRIALS}\n"
    + f"THREADED_TRIALS = {THREADED_TRIALS}\n"
    + f"TRIAL_SCRIPT = {TRIAL_SCRIPT!r}\n\n\n"
    + ENV_TAIL
)

WARMUP = '''\
# Download once, outside the timed loads, so each trial measures loading and
# nothing else. Bounded, because a stalled download must not pass for a hang.
download = (
    "import time\\n"
    "from huggingface_hub import snapshot_download\\n"
    f"for repo in ({MODEL_ID!r}, {CODEC_ID!r}):\\n"
    "    t = time.time()\\n"
    "    snapshot_download(repo)\\n"
    "    print(repo, f'{time.time() - t:.0f}s', flush=True)\\n"
)
try:
    subprocess.run([sys.executable, "-c", download], text=True, timeout=WARMUP_TIMEOUT, check=True)
except subprocess.TimeoutExpired:
    raise SystemExit(f"the download did not finish within {WARMUP_TIMEOUT}s - a network stall, "
                     "not the load race, so nothing was tested")
'''

SEQUENTIAL = '''\
# The pool disabled, first - before any threaded load has had the chance to
# hang and leave something behind in this session.
for index in range(1, SEQUENTIAL_TRIALS + 1):
    trial(sequential=True, index=index)
'''

THREADED = '''\
# transformers' default, with the thread pool on: the control for the cell above.
for index in range(1, THREADED_TRIALS + 1):
    trial(sequential=False, index=index)
'''

VERDICT = '''\
def tally(sequential):
    rows = [r for r in RESULTS if r["sequential"] == sequential]
    loaded = [r["elapsed"] for r in rows if r["status"] == "ok"]
    hung = sum(r["status"] == "HUNG" for r in rows)
    return rows, loaded, hung, len(rows) - len(loaded) - hung


seq_rows, seq_ok, seq_hung, seq_err = tally(True)
thr_rows, thr_ok, thr_hung, thr_err = tally(False)
for name, rows, loaded, hung, err in (("sequential", seq_rows, seq_ok, seq_hung, seq_err),
                                      ("threaded", thr_rows, thr_ok, thr_hung, thr_err)):
    median = f", median {statistics.median(loaded):.0f}s" if loaded else ""
    print(f"{name:10s}: {len(loaded)}/{len(rows)} loaded, {hung} hung, {err} errored{median}")

# The rule was fixed before the run, in the notebook's generator.
if seq_hung:
    verdict = "a sequential load hung - the thread pool is not the cause, or not the only one"
elif seq_err:
    verdict = ("sequential loads errored rather than hung - the test itself is broken; "
               "read the error lines above before concluding anything")
elif thr_hung:
    verdict = ("every sequential load succeeded and threaded loading hung in the same session - "
               "HF_DEACTIVATE_ASYNC_LOAD is the fix")
elif thr_err:
    verdict = "threaded loads errored rather than hung - the race was not reproduced; read the errors above"
else:
    verdict = ("inconclusive - nothing hung, so this session did not reproduce the race "
               "and the sequential successes show nothing")
print()
print("VERDICT:", verdict)
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
    compile(TRIAL_SCRIPT, "<trial script>", "exec")

    sources = [SETUP, ENVIRONMENT, WARMUP, SEQUENTIAL, THREADED, VERDICT]
    for index, source in enumerate(sources):
        compile(
            "".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
            f"<cell {index}>",
            "exec",
        )

    defines = [i for i, s in enumerate(sources) if "def trial(" in s]
    uses = [i for i, s in enumerate(sources) if "trial(sequential=" in s]
    assert defines and uses and min(defines) < min(uses), "trial() is used before it is defined"

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
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {path} ({len(sources)} cells and the trial script, all compile)")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {metadata['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
