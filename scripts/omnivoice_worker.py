#!/usr/bin/env python3
"""OmniVoice synthesis worker. Runs in its own interpreter, not this project's.

OmniVoice (k2-fsa, arXiv 2604.00688) is a diffusion-language-model TTS trained
on 581k hours across 600+ languages, Persian among them. It brings its own
torch and transformers, and adding those to this project's environment would
put the F5-TTS baseline - which every existing result is measured against - at
risk. So it lives in a separate virtualenv and is driven out-of-process, the
arrangement Chatterbox and Seed-VC already use.

The protocol is the Chatterbox worker's: one JSON job per line on stdin, one
JSON result per line on stdout, the model kept resident in between.

Protocol
--------
in   {"text": "...", "reference": "ref.wav", "reference_text": "...",
      "output": "out.wav", "language": "fa", "seed": 42, "num_step": 32}
out  {"ok": true, "output": "out.wav", "sample_rate": 24000, "seconds": 3.4}
     {"ok": false, "error": "..."}

``{"op": "quit"}`` shuts it down. stderr is diagnostic; the parent parses only
stdout.

Settings
--------
Generation defaults are upstream's own ``omnivoice-infer`` values - 32 steps,
guidance 2.0, t_shift 0.1, layer penalty 5.0, position temperature 5.0 - which
are what the paper's numbers were measured with. A job overrides only the
fields it names; anything it leaves out falls through to upstream.

Startup
-------
    omnivoice_worker.py --device mps [--dtype float16] [--model k2-fsa/OmniVoice]

The first line on stdout is ``{"ready": true, ...}`` once the model is loaded,
and names the device and dtype that actually loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Any


def _claim_stdout() -> Any:
    """Take sole ownership of stdout, and point everything else at stderr.

    The protocol is one JSON object per line, and third-party libraries do not
    know that. Some of them write to file descriptor 1 from C, which a
    ``sys.stdout`` swap does not catch - so the descriptor itself is duplicated
    away and then replaced by stderr, exactly as the Chatterbox worker does.
    """
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


_PROTOCOL = _claim_stdout()

#: Generation settings a job may override. Anything not named here is not
#: forwarded, so a typo in a job cannot silently change what upstream runs.
GENERATION_FIELDS = (
    "num_step",
    "guidance_scale",
    "t_shift",
    "denoise",
    "postprocess_output",
    "layer_penalty_factor",
    "position_temperature",
    "class_temperature",
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


def _model_class() -> Any:
    try:
        from omnivoice.models.omnivoice import OmniVoice
    except ImportError:
        from omnivoice import OmniVoice
    return OmniVoice


def _load_once(model_class: Any, model_id: str, device: str, dtype_name: str) -> Any:
    """One placement attempt.

    On Apple Silicon a float16 load is cast on CPU and then moved, because
    casting during the load itself kills the process. transformers 5.x
    materialises weights in a thread pool (``GLOBAL_WORKERS``, hard-coded, no
    environment switch), and asking it for float16 on MPS aborts inside
    ``_materialize_copy``: macOS's crash report puts the fault in torch's
    ``copy_cast_kernel_mps`` -> ``MetalShaderLibrary::exec_unary_kernel``. It is a
    native crash, so no ``except`` here can recover from it.

    Probed on this machine, each placement in its own process, loading and then
    rendering one Persian sentence:

        MPS, float16                     aborts during weight loading
        MPS, float32                     loads and renders
        CPU float16, then moved to MPS   loads and renders (fastest of the three)
        CPU, float32                     loads and renders, about twice as slow

    Casting on CPU and moving the finished model keeps upstream's float16 - the
    dtype its published numbers were measured with - on Metal. The checkpoint is
    stored in float32, so this is the same cast, just on the other side of the
    copy. The audio tokenizer stays on CPU, as upstream already arranges for MPS.

    A minimal threaded cast-copy did not reproduce the crash in isolation, so the
    exact trigger is unconfirmed; the placement above is what measurably works.
    """
    import torch

    dtype = _dtype(dtype_name)
    if device.startswith("mps") and dtype != torch.float32:
        model = model_class.from_pretrained(model_id, device_map="cpu", dtype=dtype)
        tokenizer = model.audio_tokenizer
        model = model.to(device)
        model.audio_tokenizer = tokenizer.to("cpu")
        return model
    return model_class.from_pretrained(model_id, device_map=device, dtype=dtype)


def load_model(model_id: str, device: str, dtype_name: str) -> tuple[Any, str, str]:
    """Load on the requested device and dtype, stepping aside only on failure.

    Every step is reported back to the parent. float32 is higher precision than
    the float16 upstream uses, and CPU is only slower than Metal, so neither
    fallback trades away quality - but a run that quietly landed somewhere other
    than where it was asked to would still be measuring a configuration nobody
    chose, and that should be visible in the result.

    These fallbacks only catch Python-level failures. A native crash ends the
    process before any of them run, which is why :func:`_load_once` avoids the
    known one up front instead of relying on this loop.
    """
    import torch

    torch.set_grad_enabled(False)
    model_class = _model_class()

    attempts = [(device, dtype_name)]
    if dtype_name != "float32":
        attempts.append((device, "float32"))
    if device != "cpu":
        attempts.append(("cpu", "float32"))

    last_error: Exception | None = None
    for attempt_device, attempt_dtype in attempts:
        try:
            model = _load_once(model_class, model_id, attempt_device, attempt_dtype)
            model.eval()
            if (attempt_device, attempt_dtype) != (device, dtype_name):
                _log(
                    f"omnivoice: loaded on {attempt_device}/{attempt_dtype} "
                    f"instead of {device}/{dtype_name} ({last_error})"
                )
            return model, attempt_device, attempt_dtype
        except Exception as exc:  # noqa: BLE001 - try the next placement
            last_error = exc
            _log(
                f"omnivoice: load on {attempt_device}/{attempt_dtype} failed: "
                f"{type(exc).__name__}: {exc}"
            )
    raise RuntimeError(f"could not load {model_id}: {last_error}")


def _seed(seed: int) -> None:
    """Seed every generator OmniVoice could be drawing from.

    Covering all of them means reproducibility does not depend on knowing which
    one its sampling loop happens to use in this release.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.suppress(Exception):
        torch.mps.manual_seed(seed)


def _release_memory() -> None:
    """Hand cached accelerator memory back to the system after every job.

    The first full run on this 16 GB Mac had to be stopped at cell 20 of 27 when
    system memory pressure reached warning. Sampled every 30 s, this worker's
    resident size had risen from 1,855 to 2,021 MB in one minute of rendering -
    roughly 55 MB per cell. Emptying Metal's allocator cache and collecting
    garbage changes no computation, only what stays reserved between renders.
    Whether it removes the growth is for the next run's samples to show.
    """
    import gc

    import torch

    gc.collect()
    with contextlib.suppress(Exception):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def synthesize(model: Any, job: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    if job.get("seed") is not None:
        _seed(int(job["seed"]))

    kwargs = {name: job[name] for name in GENERATION_FIELDS if job.get(name) is not None}
    speed = job.get("speed")
    if speed is not None and abs(float(speed) - 1.0) > 1e-6:
        # Left unset at 1.0 so the model's own duration estimate applies, which
        # is what generate() does when speed is not given.
        kwargs["speed"] = float(speed)

    with torch.inference_mode():
        audios = model.generate(
            text=job["text"],
            language=job.get("language") or None,
            ref_audio=job["reference"],
            ref_text=job.get("reference_text") or None,
            **kwargs,
        )

    samples = np.asarray(audios[0], dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("OmniVoice returned no audio")
    if not np.isfinite(samples).all():
        # float16 overflow on Metal produces NaNs rather than an exception, and
        # NaN audio scores as a very strange speaker instead of as an error.
        raise RuntimeError("non-finite samples in the render; reload with --dtype float32")

    rate = int(model.sampling_rate)
    sf.write(job["output"], samples, rate)
    return {
        "ok": True,
        "output": job["output"],
        "sample_rate": rate,
        "seconds": round(samples.size / rate, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()

    try:
        model, device, dtype = load_model(args.model, args.device, args.dtype)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit(
        {
            "ready": True,
            "sample_rate": int(model.sampling_rate),
            "device": device,
            "dtype": dtype,
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
            _emit(synthesize(model, job))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
        finally:
            _release_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
