#!/usr/bin/env python3
"""Generate a CPU-only Kaggle notebook that tests why MOSS v1.5's codec load goes silent.

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
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"

SETUP = '''\
!pip -q install "transformers==5.0.0" "accelerate>=1.10.1" 2>&1 | tail -3
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
print(f"codec in {{(time.time() - start) / 60:.1f}} min")
'''

DRAINED = '''\
code, seconds, timed_out = run_bounded([sys.executable, str(PROBE), str(CODEC_DIR)], timeout=1500,
                                       drain=True, monitor_every=60, tee=Path("/kaggle/working/drained.log"))
RESULTS["drained_cold_cache"] = {"exit": code, "seconds": round(seconds), "timed_out": timed_out}
'''

INHERITED = '''\
code, seconds, timed_out = run_bounded([sys.executable, str(PROBE), str(CODEC_DIR)], timeout=900,
                                       drain=False, monitor_every=60)
RESULTS["inherited_warm_cache"] = {"exit": code, "seconds": round(seconds), "timed_out": timed_out}
'''

VERDICT = '''\
print(json.dumps(RESULTS, indent=2))
drained, inherited = RESULTS["drained_cold_cache"], RESULTS["inherited_warm_cache"]
if not drained["timed_out"] and drained["exit"] == 0 and inherited["timed_out"]:
    reading = "drained finished, inherited hung: consistent with a writer blocked on an undrained pipe"
elif not drained["timed_out"] and not inherited["timed_out"]:
    reading = "both finished: not reproduced on a CPU box"
else:
    reading = "the drained load also failed: something other than output - see the wait channels above"
print("READING:", reading)
RESULTS["reading"] = reading
Path("/kaggle/working/probe.json").write_text(json.dumps(RESULTS, indent=2))
'''


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main() -> int:
    sources = [SETUP, ENVIRONMENT, DOWNLOAD, DRAINED, INHERITED, VERDICT]
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
                "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
                "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(sources)} cells, all compile) - CPU only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
