#!/usr/bin/env python3
"""Higgs TTS 3 synthesis worker. Runs in its own interpreter, not this project's.

Higgs TTS 3 (Boson AI) is a Qwen3-4B backbone with a fused 8-codebook audio
head, and it claims Persian at production quality - inside the 85-language tier
its model card calls "WER/CER under 5", not the 17-language tier below it.

Boson ships it for SGLang-Omni or vLLM-Omni, both of which want an Ampere card:
vLLM gates its default attention backend at compute 8.0 and refuses the
checkpoint's bfloat16 below it. The one public attempt to force the Turing path
produced well-formed WAV files in which every sample was -32768 - and the likely
reason is visible in the port used here, which notes that the codec "is unstable
in bf16" and therefore runs it in fp32.

So this worker uses ``multimodalart/higgs-audio-v3-tts-4b-transformers``: the
same checkpoint with a small modeling pair added so it loads with plain
transformers. Two things were checked rather than assumed, because the code runs
with ``trust_remote_code``:

* ``model.safetensors`` and ``tokenizer.json`` have byte-identical sha256 hashes
  to ``bosonai/higgs-tts-3-4b``. The weights are the vendor's own.
* the 391 lines of modeling code were read. They import torch and transformers
  and nothing else - no filesystem, network, shell or pickle - and the revision
  is pinned below, so what runs is what was read.

What is *not* the vendor's own is the generation loop: the delay-pattern ramp,
the sampler and the prompt assembly are this port's re-implementation. If it
diverges from upstream, this measures the port rather than the model, and a
Persian CER far from the model card's claim is the signal to distrust it.

The protocol is the OmniVoice worker's: one JSON job per line on stdin, one JSON
result per line on stdout, the model kept resident in between.

Protocol
--------
in   {"text": "...", "reference": "ref.wav", "reference_text": "...",
      "output": "out.wav", "seed": 42, "temperature": 0.8}
out  {"ok": true, "output": "out.wav", "sample_rate": 24000, "seconds": 3.4}
     {"ok": false, "error": "..."}

``{"op": "quit"}`` shuts it down. stderr is diagnostic; the parent parses only
stdout.

Settings
--------
Defaults are the values in Boson's own serving example - temperature 0.8,
top_k 50, at most 1024 new frames.

Startup
-------
    higgs_worker.py --device cuda [--dtype float16]

The first line on stdout is ``{"ready": true, ...}`` once the model and the
codec are both loaded, and names the dtype that actually loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Any

#: The port's revision, pinned to the commit whose modeling code was read.
DEFAULT_MODEL = "multimodalart/higgs-audio-v3-tts-4b-transformers"
DEFAULT_REVISION = "30f01593ee6a12efa586c92455afe4b76e45095d"


def _claim_stdout() -> Any:
    """Take sole ownership of stdout, and point everything else at stderr."""
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


_PROTOCOL = _claim_stdout()

#: Generation settings a job may override. Anything not named here is not
#: forwarded, so a typo in a job cannot silently change what runs.
GENERATION_FIELDS = ("temperature", "top_p", "top_k", "max_new_tokens")


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def choose_dtype(requested: str, device: str) -> str:
    """Drop bfloat16 to float16 where the GPU has no bfloat16 at all.

    Turing - the T4 this is benchmarked on - has no bfloat16 unit. Both are
    16-bit; bfloat16 trades mantissa for range. The codec is loaded in fp32 by
    the port either way, which is the part that was unstable in 16-bit.
    """
    import torch

    if requested != "bfloat16" or not device.startswith("cuda"):
        return requested
    if not torch.cuda.is_available() or torch.cuda.is_bf16_supported():
        return requested
    _log(f"higgs: {torch.cuda.get_device_name(0)} has no bfloat16 - loading float16")
    return "float16"


def load_model(model_id: str, revision: str | None, device: str, dtype_name: str) -> tuple[Any, Any]:
    """Load the LM, then force the codec, so a codec failure surfaces now.

    The codec is fetched lazily on the first render otherwise, which would
    charge a 806 MB download and a fp32 load to whatever job happened to be
    first and report it as a synthesis failure.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, dtype=_dtype(dtype_name)
    )
    model = model.to(device).eval()
    model.get_audio_codec()
    return model, tokenizer


def _seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with contextlib.suppress(Exception):
        torch.mps.manual_seed(seed)


def _release_memory() -> None:
    """Hand cached accelerator memory back after every job."""
    import gc

    import torch

    gc.collect()
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def synthesize(model: Any, tokenizer: Any, job: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    if job.get("seed") is not None:
        _seed(int(job["seed"]))

    reference = None
    reference_rate = None
    if job.get("reference"):
        data, reference_rate = sf.read(job["reference"], dtype="float32", always_2d=True)
        reference = torch.from_numpy(data.mean(axis=1))

    kwargs = {name: job[name] for name in GENERATION_FIELDS if job.get(name) is not None}
    waveform = model.generate_speech(
        job["text"],
        tokenizer,
        reference_audio=reference,
        reference_sample_rate=reference_rate,
        reference_text=(job.get("reference_text") or "").strip() or None,
        **kwargs,
    )

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("Higgs returned no audio")
    if not np.isfinite(samples).all():
        raise RuntimeError("non-finite samples in the render; reload with --dtype float32")
    if np.unique(samples).size < 1000:
        # The failure this hardware is known for: a full-scale constant, or near
        # enough, written out as a perfectly well-formed file. Speech at 24 kHz
        # has thousands of distinct values in even a short clip.
        raise RuntimeError(
            f"the render has only {np.unique(samples).size} distinct sample values - "
            "this is not speech"
        )

    rate = int(model.config.sample_rate)
    sf.write(job["output"], samples, rate)
    return {
        "ok": True,
        "output": job["output"],
        "sample_rate": rate,
        "seconds": round(samples.size / rate, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()

    try:
        dtype = choose_dtype(args.dtype, args.device)
        model, tokenizer = load_model(args.model, args.revision or None, args.device, dtype)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit(
        {
            "ready": True,
            "sample_rate": int(model.config.sample_rate),
            "device": args.device,
            "dtype": dtype,
            "model": args.model,
            "revision": args.revision,
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"bad job: {exc}"})
            continue
        if job.get("op") == "quit":
            break
        try:
            _emit(synthesize(model, tokenizer, job))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
        finally:
            _release_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
