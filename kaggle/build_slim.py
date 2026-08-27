#!/usr/bin/env python3
"""Generate a CPU-only Kaggle notebook that slims the trained checkpoint.

The trainer writes 5.4 GB: model weights, the EMA copy, both Adam moments and
the scheduler. Inference needs only the EMA weights, which is 1.35 GB - and the
link to Kaggle here breaks partway through large transfers, so dropping three
quarters of the bytes before they cross it is worth one short CPU run.

It also sidesteps the reason the training kernel ended in error: /kaggle/working
filled up at 20 GB while saving, which truncated model_15000.pt to 2.05 GB.
model_last.pt is the intact one.

    .venv/bin/python kaggle/build_slim.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/slim
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "slim"

USER = "marsmanam"
SOURCE_KERNEL = "persian-f5-multispeaker-train"
SLUG = "persian-f5-slim"

SLIM = '''\
import shutil
from pathlib import Path

import torch

# The training kernel's output is mounted read-only under /kaggle/input.
candidates = sorted(Path("/kaggle/input").rglob("model_last.pt"))
assert candidates, f"no model_last.pt under {[p.name for p in Path('/kaggle/input').glob('*')]}"
source = candidates[0]
print("source     :", source)
print("size       : %.3f GB" % (source.stat().st_size / 1e9))

state = torch.load(str(source), map_location="cpu", weights_only=True)
print("keys       :", list(state.keys()))
print("update     :", state.get("update"))

ema = state["ema_model_state_dict"]
print("ema tensors:", len(ema))

# Keep initted/step: the EMA wrapper's own load_state_dict expects them, and
# they cost nothing. What matters is that no top-level "update" survives, so
# anything loading this treats it as weights rather than a resumable run.
out = Path("/kaggle/working/finetuned_ema.pt")
torch.save({"ema_model_state_dict": ema}, out)
del state, ema

check = torch.load(str(out), map_location="cpu", weights_only=True)
assert "update" not in check and "optimizer_state_dict" not in check
print("wrote      :", out, "%.3f GB" % (out.stat().st_size / 1e9))
print("tensors    :", len(check["ema_model_state_dict"]))
del check

# Nothing else should ride along and eat the output quota.
for stray in Path("/kaggle/working").iterdir():
    if stray != out:
        shutil.rmtree(stray) if stray.is_dir() else stray.unlink()
print("working    :", [p.name for p in Path("/kaggle/working").iterdir()])
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
    compile(SLIM, "<slim cell>", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": [cell(SLIM)],
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
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{USER}/{SOURCE_KERNEL}"],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {path}")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {metadata['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
