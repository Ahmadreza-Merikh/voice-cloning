#!/usr/bin/env python3
"""Historical: the notebook behind the two MOSS v1.5 load probes of 2026-09-18.

Kept as the record of what ran, not as something to run again. It was built
against ``scripts/kernel_gates.py`` as of commit fa7578f - whose ``tee`` and
``drain`` arguments no longer exist - and its results are in
eval/runs/persian-voice-moss15-probe-v1.json and -v2.json. Two statements
below turned out wrong and are corrected here rather than silently edited:

* v2 did not get "the T4 image's 2.9.1+cu128". It got torch 2.10.0+cu128; the
  main kernel's v1 and v2 had 2.9.1+cu128. The two kernels ran on different
  images, so v2 did not reproduce v1/v2's environment exactly
* the readings decided in advance did not include the cause v2 found: the
  worker path did not hang in loading - the runner's reader was stuck writing
  to the notebook display, and the worker was blocked writing progress output
  behind it (see the v2 record)

Version 1 of this probe ran on the CPU image and could not reproduce the hang:
the codec loaded in 20 s drained and 13 s inherited (eval/runs/). Version 2
ran the same two loads on a T4 image with short budgets, then the worker's own
full load path, drained and monitored. What each outcome meant was written in
the VERDICT cell before running. The original rationale follows.

Two T4 runs went silent a second into loading the codec and stayed silent for
their whole budget, and in the second one not even faulthandler - a C thread
that dumps stacks without needing the GIL - got a line out. A Python deadlock
would not stop that thread; a process blocked on writing to a full, undrained
pipe would. The one MOSS run that ever completed read its child's output
through a watcher; the hung ones wrote straight into the notebook's stderr
while printing a long progress bar.

The hang happened before anything touched a GPU, so this needs none and spends
no GPU quota. It loads only the codec, exactly as the processor does, twice:

1. **drained** - output piped to a reader that empties it continuously -
   first, on a cold disk cache
2. **inherited** - output straight into the notebook's stderr, the way both hung
   runs did - second, with the cache warm, so it has every advantage

Every minute each process's state and the kernel function it is waiting in
(``wchan``) are printed, with free memory. Readings, decided before running:

* drained finishes, inherited hangs with a writer waiting in pipe_write -
  the hypothesis holds, and the fix is already in ``run_bounded``
* both finish - not reproduced on this box; the T4 run gets the drained path
  anyway, plus the monitor, and the question moves there
* both hang - something else; the wait channels and memory say what

    .venv/bin/python kaggle/build_moss15_probe_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/moss15-probe
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "moss15-probe"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-moss15-probe"

#: v1 ran on the CPU image (torch 2.10.0+cpu) and loaded the codec fine both ways.
#: v2 ran on a T4 image - which had torch 2.10.0+cu128, not the 2.9.1+cu128 this
#: comment first assumed - and added the worker's own load path.
ON_GPU = True
MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"

SETUP = '''\
!pip -q install "transformers==5.0.0" "accelerate>=1.10.1" 2>&1 | tail -3
'''

GUARD = '''\
import subprocess, sys
probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"], text=True, capture_output=True)
if probe.returncode != 0 and "torchvision" in probe.stderr:
    print("removing a torchvision built for another torch")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"])
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

print("torch", torch.__version__, "| transformers", transformers.__version__, "| cpus", os.cpu_count())
print(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)

found = sorted(Path("/kaggle/input").rglob("repo/scripts/kernel_gates.py"))
assert found, "the dataset has no kernel_gates.py"
sys.path.insert(0, str(found[0].parent))
from kernel_gates import run_bounded  # noqa: E402

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"  # as in both T4 runs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROBE = Path("/kaggle/working/codec_probe.py")
PROBE.write_text("""
import faulthandler, resource, sys, time
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
started = time.time()
from transformers import AutoModel
codec = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, codec_weight_dtype="bf16")
faulthandler.cancel_dump_traceback_later()
params = sum(p.numel() for p in codec.parameters())
print(f"LOADED in {time.time() - started:.0f}s, {params / 1e9:.2f}B parameters, "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)
""")
RESULTS = {}
'''

DOWNLOAD = f'''\
from huggingface_hub import snapshot_download

start = time.time()
CODEC_DIR = Path(snapshot_download(repo_id="{CODEC}", revision="{CODEC_REVISION}"))
MODEL_DIR = Path(snapshot_download(repo_id="{MODEL}", revision="{MODEL_REVISION}"))
print(f"codec and model in {{(time.time() - start) / 60:.1f}} min")
print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''

WORKER_LOAD = '''\
# The worker's own path: processor (tokenizer + codec), codec to cuda:1, model
# onto cuda:0 in bf16 - then quit as soon as it reports ready. Drained and
# monitored, with a budget that is generous for a load and short for a hang.
scripts = found[0].parent
driver = Path("/kaggle/working/ready_then_quit.py")
driver.write_text(f"""
import json, subprocess, sys, time
start = time.time()
worker = subprocess.Popen([sys.executable, "{scripts}/moss15_worker.py", "--model-dir", sys.argv[1],
                           "--codec-dir", sys.argv[2], "--device", "cuda", "--codec-device", "auto"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
line = worker.stdout.readline()
print("WORKER SAID", line.strip()[:2000], flush=True)
print(f"READY AFTER {{time.time() - start:.0f}}s", flush=True)
worker.stdin.write(json.dumps({{"op": "quit"}}) + chr(10))
worker.stdin.flush()
worker.wait(timeout=120)
sys.exit(0 if json.loads(line).get("ready") else 1)
""")
code, seconds, timed_out = run_bounded([sys.executable, str(driver), str(MODEL_DIR), str(CODEC_DIR)],
                                       timeout=1200, drain=True, monitor_every=60,
                                       tee=Path("/kaggle/working/worker_load.log"))
RESULTS["worker_path_drained"] = {"exit": code, "seconds": round(seconds), "timed_out": timed_out}
'''

DRAINED = '''\
code, seconds, timed_out = run_bounded([sys.executable, str(PROBE), str(CODEC_DIR)], timeout=600,
                                       drain=True, monitor_every=60, tee=Path("/kaggle/working/drained.log"))
RESULTS["drained_cold_cache"] = {"exit": code, "seconds": round(seconds), "timed_out": timed_out}
'''

INHERITED = '''\
code, seconds, timed_out = run_bounded([sys.executable, str(PROBE), str(CODEC_DIR)], timeout=600,
                                       drain=False, monitor_every=60)
RESULTS["inherited_warm_cache"] = {"exit": code, "seconds": round(seconds), "timed_out": timed_out}
'''

VERDICT = '''\
print(json.dumps(RESULTS, indent=2))
ok = {name: (not r["timed_out"] and r["exit"] == 0) for name, r in RESULTS.items()}
if not ok["drained_cold_cache"]:
    reading = "the codec alone hangs on this image even drained - image-specific; see the wait channels"
elif not ok["inherited_warm_cache"]:
    reading = "codec loads drained but hangs inherited on this image - consistent with an undrained pipe"
elif not ok["worker_path_drained"]:
    reading = "the codec loads alone but the worker's full path hangs - the cause is in that path"
else:
    reading = "every load finished on this image - the T4 kernel's drained, monitored run should get past loading"
print("READING:", reading)
RESULTS["reading"] = reading
Path("/kaggle/working/probe.json").write_text(json.dumps(RESULTS, indent=2))
'''


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main() -> int:
    sources = [SETUP, GUARD, ENVIRONMENT, DOWNLOAD, DRAINED, INHERITED, WORKER_LOAD, VERDICT]
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
                "kernel_type": "notebook", "is_private": True, "enable_gpu": ON_GPU, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(sources)} cells, all compile) - {'T4 image' if ON_GPU else 'CPU only'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
