#!/usr/bin/env python3
"""Fish Audio S2 Pro synthesis worker. Runs in its own interpreter, not this project's.

S2 Pro (fishaudio, arXiv 2603.08823) is a dual-autoregressive TTS trained on
10M+ hours across 80+ languages, Persian among them. It pins ``torch==2.8.0``
and ``transformers<=4.57.3``, which is a different dependency world from the
one every existing result was measured in, so it lives in its own environment
and is driven out-of-process - the arrangement Chatterbox, Seed-VC and
OmniVoice already use.

The protocol is the OmniVoice worker's: one JSON job per line on stdin, one
JSON result per line on stdout, both models kept resident in between. Upstream's
own three-step CLI reloads 9 GB of weights for every sentence; keeping them
resident is the difference between a benchmark that finishes and one that does
not.

Protocol
--------
in   {"text": "...", "reference": "ref.wav", "reference_text": "...",
      "output": "out.wav", "seed": 42, "temperature": 1.0}
out  {"ok": true, "output": "out.wav", "sample_rate": 44100, "seconds": 3.4}
     {"ok": false, "error": "..."}

``{"op": "quit"}`` shuts it down. stderr is diagnostic; the parent parses only
stdout.

Settings
--------
Generation defaults are upstream's own CLI values - temperature 1.0, top_p 0.9,
top_k 30, repetition penalty 1.1, chunk length 300 - so a run with nothing
overridden is the configuration upstream ships.

Startup
-------
    fish_worker.py --checkpoint fishaudio/s2-pro --device cuda --precision float16

A repository id is downloaded through ``huggingface_hub``; a path is used as
given. The first line on stdout is ``{"ready": true, ...}`` once both models are
loaded, and names the precision that actually loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _claim_stdout() -> Any:
    """Take sole ownership of stdout, and point everything else at stderr.

    The protocol is one JSON object per line, and third-party libraries do not
    know that. fish-speech logs through loguru and draws tqdm bars, and hydra
    prints on instantiation; some libraries write to file descriptor 1 from C,
    which a ``sys.stdout`` swap does not catch. So the descriptor itself is
    duplicated away and then replaced by stderr, exactly as the other workers do.
    """
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


_PROTOCOL = _claim_stdout()

#: Generation settings a job may override. Anything not named here is not
#: forwarded, so a typo in a job cannot silently change what upstream runs.
GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "chunk_length",
    "max_new_tokens",
)


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


def resolve_checkpoint(checkpoint: str) -> Path:
    """Return a local checkpoint directory, downloading a repository id."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    _log(f"fish: downloading {checkpoint} (about 11 GB on a cold cache)")
    return Path(snapshot_download(repo_id=checkpoint))


def choose_precision(requested: str, device: str) -> str:
    """Drop bfloat16 to float16 where the GPU has no bfloat16 at all.

    Turing cards - the T4 this benchmark runs on - have no bfloat16 unit, and
    upstream's own answer is the ``--half`` flag. Both are 16-bit; bfloat16
    trades mantissa for range. Which one ran is reported back to the parent and
    ends up in the results file, because it is a difference between machines
    and not something to discover later from a log.

    The test is the compute capability, not ``torch.cuda.is_bf16_supported()``:
    on a T4 that returns True, because torch counts the emulated path. Emulation
    is exactly what upstream's flag exists to avoid.
    """
    import torch

    if requested != "bfloat16" or not device.startswith("cuda"):
        return requested
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 8:
        return requested
    name = torch.cuda.get_device_name(0)
    _log(f"fish: {name} has no bfloat16 unit - loading float16 instead (upstream's --half)")
    return "float16"


def _report_memory(stage: str) -> None:
    """Say what is on the card after each stage, rather than leaving it to guesswork.

    The first run to get this far died out of memory during the reference
    encode, with 14.40 GiB of a 14.56 GiB card in use, and nothing in the log
    said which of the transformer, its buffers and the codec had taken it.
    """
    import torch

    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(index) / 2**30
        reserved = torch.cuda.memory_reserved(index) / 2**30
        if allocated or reserved:
            _log(f"fish: {stage:28s} cuda:{index}  {allocated:5.2f} GiB allocated, {reserved:5.2f} reserved")


def choose_codec_device(device: str) -> str:
    """Put the codec on a second GPU when there is one.

    The transformer alone is 8.5 GiB of weights before its caches, and the codec
    is another gigabyte-plus; on a 15 GiB card that is close enough to the edge
    that the reference encode's own activations tipped it over. Kaggle's T4
    session has two cards and the second sits idle, so the codec goes there.
    Nothing is split across devices - only the small code tensors cross.
    """
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return device
    if torch.cuda.device_count() < 2:
        return device
    _log("fish: putting the codec on cuda:1, which leaves the transformer a card to itself")
    return "cuda:1"


def load_models(
    checkpoint: Path, device: str, precision_name: str, max_seq_len: int
) -> tuple[Any, Any, Any, str]:
    """Load the dual-AR transformer and the DAC codec, both resident.

    ``max_seq_len`` is passed to ``from_pretrained`` rather than set afterwards,
    because the model allocates from it at construction: a causal mask of
    ``max_seq_len`` squared booleans, which at the checkpoint's declared 32768
    is a full gibibyte of nothing but a triangle, plus rope tables and then the
    KV cache. At 4096 the mask is 16 MiB. Nothing about the model changes: at
    roughly 21 frames per second a 4096-token window is about three minutes of
    speech, and upstream refuses a prompt longer than ``max_seq_len - 2048`` in
    any case, so a benchmark sentence of a few seconds is nowhere near it. A
    render that did approach the bound would raise rather than come back
    quietly truncated.
    """
    import torch
    from fish_speech.models.text2semantic.inference import decode_one_token_ar, load_codec_model
    from fish_speech.models.text2semantic.llama import DualARTransformer

    # Upstream loads the codec under @torch.inference_mode(), which makes its
    # parameters inference tensors. Anything that then touches them with grad
    # still enabled fails inside the codec's TorchScript snake activation with
    # "Inference tensors cannot be saved for backward" - which is how one render
    # died after the model had loaded perfectly.
    torch.set_grad_enabled(False)
    dtype = _dtype(precision_name)

    # This is upstream's init_model, with max_length passed through - the one
    # thing its signature does not expose.
    model = DualARTransformer.from_pretrained(
        str(checkpoint), load_weights=True, max_length=max_seq_len
    )
    if not isinstance(model, DualARTransformer):
        # upstream's init_model refuses anything else, and the sampling loop
        # below is the dual-AR one.
        raise RuntimeError(f"{checkpoint} is not a dual-AR checkpoint: {type(model).__name__}")
    model = model.to(device=device, dtype=dtype)
    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)
    model._cache_setup_done = False
    model = model.eval()
    _report_memory("transformer")

    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _report_memory("transformer + kv cache")

    codec_device = choose_codec_device(device)
    codec = load_codec_model(checkpoint / "codec.pth", codec_device, dtype)
    _report_memory("everything loaded")
    return model, decode_one_token_ar, codec, codec_device


def encode_reference(path: str, codec: Any, device: str) -> Any:
    """Reference waveform -> VQ codes, without going through torchaudio.load.

    This is upstream's ``encode_audio`` with one line changed. Upstream reads
    the clip with ``torchaudio.load``, which in torchaudio 2.9 delegates to
    torchcodec and needs FFmpeg's shared libraries; on the image this runs on,
    every version of libavutil torchcodec looks for is missing, so a model that
    had loaded perfectly failed on its first render. soundfile reads a WAV
    without any of that. Everything after the read - the mono mean, the
    resample, the dtype, the length tensor - is upstream's own.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    with torch.inference_mode():  # upstream's decorator on encode_audio
        wav = torch.from_numpy(np.asarray(data, dtype=np.float32).mean(axis=1)).to(device)
        if rate != codec.sample_rate:
            wav = torchaudio.functional.resample(wav, rate, codec.sample_rate)

        model_dtype = next(codec.parameters()).dtype
        audios = wav[None, None].to(dtype=model_dtype)
        audio_lengths = torch.tensor([wav.shape[-1]], device=device, dtype=torch.long)
        indices, feature_lengths = codec.encode(audios, audio_lengths)
        return indices[0, :, : feature_lengths[0]]


def _seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release_memory() -> None:
    """Hand cached accelerator memory back after every job."""
    import gc

    import torch

    gc.collect()
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def synthesize(state: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.inference import decode_to_audio, generate_long

    if job.get("seed") is not None:
        _seed(int(job["seed"]))

    device = state["device"]
    codec = state["codec"]
    reference_text = (job.get("reference_text") or "").strip()
    prompt_tokens = None
    prompt_text = None
    if job.get("reference") and reference_text:
        prompt_tokens = [encode_reference(job["reference"], codec, state["codec_device"]).cpu()]
        prompt_text = [reference_text]

    kwargs = {name: job[name] for name in GENERATION_FIELDS if job.get(name) is not None}
    generator = generate_long(
        model=state["model"],
        device=device,
        decode_one_token=state["decode_one_token"],
        text=job["text"],
        num_samples=1,
        compile=False,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        **kwargs,
    )

    codes = [response.codes for response in generator if response.action == "sample"]
    if not codes:
        raise RuntimeError("Fish S2 produced no semantic codes")

    merged = torch.cat(codes, dim=1)
    samples = decode_to_audio(merged.to(state["codec_device"]), codec).float().cpu().numpy().reshape(-1)
    if samples.size == 0:
        raise RuntimeError("the codec returned no audio")
    if not np.isfinite(samples).all():
        # float16 overflow in the codec produces NaNs rather than an exception,
        # and NaN audio scores as a very strange speaker instead of as an error.
        raise RuntimeError("non-finite samples in the render; reload with --precision float32")

    rate = int(codec.sample_rate)
    sf.write(job["output"], samples, rate)
    return {
        "ok": True,
        "output": job["output"],
        "sample_rate": rate,
        "seconds": round(samples.size / rate, 3),
        "frames": int(merged.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="fishaudio/s2-pro")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", default="bfloat16", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        precision = choose_precision(args.precision, args.device)
        model, decode_one_token, codec, codec_device = load_models(
            checkpoint, args.device, precision, args.max_seq_len
        )
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    state = {
        "model": model,
        "decode_one_token": decode_one_token,
        "codec": codec,
        "device": args.device,
        "codec_device": codec_device,
    }
    _emit(
        {
            "ready": True,
            "sample_rate": int(codec.sample_rate),
            "device": args.device,
            "codec_device": codec_device,
            "precision": precision,
            "max_seq_len": args.max_seq_len,
            "checkpoint": str(checkpoint),
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
            _emit(synthesize(state, job))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
        finally:
            _release_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
