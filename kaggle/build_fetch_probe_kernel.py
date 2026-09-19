#!/usr/bin/env python3
"""Generate a CPU-only session that downloads what training session a needs, twice, and times it.

Session a v1 stalled 138 s into its downloads and held a T4 pair until it was
cancelled. Before another GPU session is spent, this runs session a's exact
download plan (``build_omnivoice_train_kernel.fetch_plan("a")``: the SDPA
config, the base, the audio tokenizer, Common Voice and Neyshekar's train and
valid shards, about 21 GiB) through ``scripts/fetch_inputs.py`` - embedded
here byte for byte - with the pinned huggingface-hub, on a machine with no
accelerator, so no GPU quota is used. It runs once with Hugging Face's Xet
client and once over plain HTTP, each bounded at 30 minutes, clearing the
downloads in between, and keeps each run's log and report. What it measures
decides ``FETCH_FLAGS`` for the GPU sessions.

    .venv/bin/python kaggle/build_fetch_probe_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/fetch-probe
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


probe = module("probe", "build_omnivoice_adapt_kernel.py")
train = module("train", "build_omnivoice_train_kernel.py")

SLUG = "persian-voice-fetch-probe"
MINUTES = 30
HUB = "1.7.2"

SETUP = f'''\
!pip -q install "huggingface-hub=={HUB}" 2>&1 | tail -3
'''

RUN = f'''\
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/fetch_probe")
OUT.mkdir(parents=True, exist_ok=True)
SCRIPT = OUT / "fetch_inputs.py"
SCRIPT.write_text({(ROOT / "scripts" / "fetch_inputs.py").read_text(encoding="utf-8")!r})
PLAN = json.loads({json.dumps(json.dumps(train.fetch_plan("a")))})
FOLDERS = sorted({{step["local_dir"] for step in PLAN["snapshots"]}} | {{"/tmp/adapt"}})


def versions():
    found = {{}}
    for name in ("huggingface-hub", "hf-xet", "requests", "httpx"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def clear():
    for folder in FOLDERS:
        shutil.rmtree(folder, ignore_errors=True)
    # Only on Kaggle: anywhere else this cache holds models someone downloaded on purpose.
    if Path("/kaggle/working").is_dir():
        shutil.rmtree(Path.home() / ".cache/huggingface", ignore_errors=True)


RESULT = {{"versions": versions(), "disk_free_gib": round(shutil.disk_usage("/tmp").free / 2**30, 1),
           "plan": [step["name"] for step in PLAN["snapshots"]], "runs": {{}}}}
print(json.dumps(RESULT, indent=1), flush=True)
(OUT / "plan.json").write_text(json.dumps(PLAN, indent=1))
for mode, flags in (("xet", []), ("http", ["--no-xet"])):
    clear()
    started = time.time()
    log = OUT / f"{{mode}}.log"
    report = OUT / f"{{mode}}.json"
    with log.open("wb") as sink:
        process = subprocess.Popen([sys.executable, str(SCRIPT), "--plan", str(OUT / "plan.json"),
                                    "--report", str(report), *flags], stdout=sink, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout={MINUTES} * 60)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            code, timed_out = process.wait(), True
    finished = json.loads(report.read_text()) if report.is_file() else {{}}
    RESULT["runs"][mode] = {{
        "exit": code, "timed_out": timed_out, "minutes": round((time.time() - started) / 60, 2),
        "ok": finished.get("ok"),
        "steps": {{s["name"]: {{k: s.get(k) for k in ("gib", "seconds", "mib_per_second", "tries", "failed_check")}}
                  for s in finished.get("steps", [])}},
    }}
    print(mode, json.dumps(RESULT["runs"][mode], indent=1), flush=True)
clear()
(OUT / "result.json").write_text(json.dumps(RESULT, indent=2))
print(json.dumps(RESULT, indent=2))
'''


def main() -> int:
    cells = [SETUP, RUN]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / "fetch-probe"
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in cells],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{probe.USER}/{SLUG}", "title": SLUG, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); CPU only, no dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
