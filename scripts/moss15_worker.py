#!/usr/bin/env python3
"""MOSS-TTS-Local-Transformer-v1.5 worker. Runs in its own interpreter.

Every choice here is the model card's own, at the revisions pinned below,
except the two stated under "Departures". That is the point of a separate
worker rather than a new checkpoint name in the v1.0 engine: v1.5 changes the
codec (MOSS-Audio-Tokenizer-v2, 48 kHz stereo, a fixed 12-codebook depth),
adds a language tag the card recommends setting, and ships its own processor.

Official, from the card at ``be7766a6`` and ``pyproject.toml`` at ``934d6826``:

* transformers 5.0.0 with ``trust_remote_code=True``; torch 2.9.1+cu128
* ``torch.bfloat16`` on CUDA, ``torch.float32`` on CPU
* FlashAttention 2 only when installed and the card is compute 8.0+, else SDPA
  on CUDA and eager on CPU; the cuDNN SDPA backend disabled, the others kept
* ``audio_temperature`` 1.7, ``audio_top_p`` 0.8, ``audio_top_k`` 25,
  ``audio_repetition_penalty`` 1.0, ``max_new_tokens`` 4096, sampling on
* the codec loaded through the processor, bf16 weights by default
* output saved as the processor returns it: ``[channels, samples]`` at 48 kHz

Departures, both about plumbing and neither about computation:

* the reference is read with soundfile and handed to the processor's own
  ``encode_audios_from_wav`` - the same mono-to-stereo, resample and loudness
  path the card's file loader ends in - because that loader is
  ``torchaudio.load``, which needs torchcodec and FFmpeg libraries that an
  earlier engine on the same Kaggle image could not find
* the codec may sit on a second GPU. Placement, not precision

The language tag is not documented for Persian. The card's examples use English
language names ("French", "Chinese"); "Persian" follows them and is recorded in
every result as an inference, not a fact.

Nothing here downgrades bf16. A T4 has no bf16 unit and runs it emulated; this
model family's v1.0 did that on the same card to completion. If it cannot, the
worker fails and says so - it does not quietly load float16.

Protocol: the other workers'. One JSON job per line in, one result per line
out, ``{"op": "quit"}`` to stop; stderr is diagnostics only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

#: The revisions whose code and card were read. trust_remote_code runs them.
MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"

GENERATION_FIELDS = (
    "audio_temperature",
    "audio_top_p",
    "audio_top_k",
    "audio_repetition_penalty",
    "max_new_tokens",
)


def _claim_stdout() -> Any:
    """Take sole ownership of stdout; everything else goes to stderr."""
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


_PROTOCOL = _claim_stdout()


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def memory() -> dict[str, float]:
    """GiB allocated on each card, for the record rather than for decisions."""
    import torch

    if not torch.cuda.is_available():
        return {}
    return {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**30, 2)
            for i in range(torch.cuda.device_count())}


def official_dtype(device: str) -> Any:
    import torch

    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def official_attention(device: str, dtype: Any) -> str:
    """The card's resolve_attn_implementation, verbatim in behaviour."""
    import importlib.util

    import torch

    if (device.startswith("cuda") and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa" if device.startswith("cuda") else "eager"


def snapshot(repo: str, revision: str) -> Path:
    """A local copy of a repository at exactly the pinned revision."""
    from huggingface_hub import snapshot_download

    _log(f"moss15: fetching {repo} at {revision[:8]}")
    return Path(snapshot_download(repo_id=repo, revision=revision))


def codec_device_for(device: str, requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        return "cuda:1"
    return device


def load(model_dir: Path, codec_dir: Path, device: str, codec_device: str) -> tuple[Any, Any, dict]:
    import torch
    from transformers import AutoModel, AutoProcessor

    # The card's own SDPA settings.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_grad_enabled(False)

    dtype = official_dtype(device)
    attention = official_attention(device, dtype)
    started = time.time()
    # Local, pinned snapshots: passing revision= through would also reach the
    # codec's from_pretrained, where the model's sha does not exist.
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True,
                                              codec_path=str(codec_dir))
    processor.audio_tokenizer = processor.audio_tokenizer.to(codec_device)
    codec_loaded = time.time()
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                      attn_implementation=attention, torch_dtype=dtype).to(device)
    model.eval()
    facts = {
        "dtype": str(dtype).replace("torch.", ""),
        "attention": attention,
        "device": device,
        "codec_device": codec_device,
        "codec_weight_dtype": str(next(processor.audio_tokenizer.parameters()).dtype).replace("torch.", ""),
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "codec_load_seconds": round(codec_loaded - started, 1),
        "model_load_seconds": round(time.time() - codec_loaded, 1),
        "memory_gib_after_load": memory(),
        "sampling_rate": int(processor.model_config.sampling_rate),
        "n_vq": int(processor.model_config.n_vq),
    }
    return processor, model, facts


def _seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(processor: Any, model: Any, device: str, job: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    if job.get("seed") is not None:
        _seed(int(job["seed"]))

    reference = None
    if job.get("reference"):
        data, rate = sf.read(job["reference"], dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))  # [channels, samples]
        reference = processor.encode_audios_from_wav([wav], int(rate))[0]

    message = processor.build_user_message(
        text=job["text"],
        reference=[reference] if reference is not None else None,
        language=job.get("language") or None,
    )
    batch = processor([[message]], mode="generation")
    kwargs = {name: job[name] for name in GENERATION_FIELDS if job.get(name) is not None}
    started = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            do_sample=True,
            **kwargs,
        )
    decoded = [m for m in processor.decode(outputs) if m is not None]
    if not decoded:
        raise RuntimeError("the processor decoded no message")
    audio = decoded[0].audio_codes_list[0]
    samples = np.asarray(audio.float().cpu().numpy(), dtype=np.float32)  # [channels, samples]
    if samples.size == 0:
        raise RuntimeError("MOSS v1.5 returned no audio")
    if not np.isfinite(samples).all():
        raise RuntimeError("non-finite samples in the render")
    if np.unique(samples).size < 1000:
        raise RuntimeError(f"only {np.unique(samples).size} distinct sample values - not speech")

    rate = int(processor.model_config.sampling_rate)
    sf.write(job["output"], samples.T, rate)  # native 48 kHz, native channels
    return {
        "ok": True,
        "output": job["output"],
        "sample_rate": rate,
        "channels": int(samples.shape[0]),
        "seconds": round(samples.shape[-1] / rate, 3),
        "generate_seconds": round(time.time() - started, 2),
        "memory_gib": memory(),
    }


def _release() -> None:
    import gc

    import torch

    gc.collect()
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", default=None,
                        help="snapshot of MODEL at MODEL_REVISION; fetched if not given")
    parser.add_argument("--codec-dir", default=None,
                        help="snapshot of CODEC at CODEC_REVISION; fetched if not given")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codec-device", default="auto")
    args = parser.parse_args()

    try:
        model_dir = Path(args.model_dir) if args.model_dir else snapshot(MODEL, MODEL_REVISION)
        codec_dir = Path(args.codec_dir) if args.codec_dir else snapshot(CODEC, CODEC_REVISION)
        codec_device = codec_device_for(args.device, args.codec_device)
        processor, model, facts = load(model_dir, codec_dir, args.device, codec_device)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit({"ready": True, "model": MODEL, "model_revision": MODEL_REVISION,
           "codec": CODEC, "codec_revision": CODEC_REVISION, **facts})

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
            _emit(synthesize(processor, model, args.device, job))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
        finally:
            _release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
