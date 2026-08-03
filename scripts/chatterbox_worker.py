#!/usr/bin/env python3
"""Chatterbox synthesis worker. Runs in its own interpreter, not this project's.

Chatterbox pins ``torch==2.6``, ``transformers==5.2`` and ``numpy<2``; F5-TTS
needs versions that cannot coexist with those. So Chatterbox lives in a
separate virtualenv and is driven out-of-process, the same arrangement Seed-VC
already uses here.

Unlike Seed-VC this worker is *persistent*: loading the model costs about half
a minute, and a benchmark run asks for dozens of renders. It reads one JSON
job per line on stdin and writes one JSON result per line on stdout, keeping
the model resident in between.

Protocol
--------
in   {"text": "...", "reference": "ref.wav", "output": "out.wav",
      "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "seed": 42}
out  {"ok": true, "output": "out.wav", "sample_rate": 24000, "seconds": 3.4}
     {"ok": false, "error": "..."}

A line of ``{"op": "quit"}`` shuts it down. Anything written to stderr is
diagnostic; the parent only parses stdout.

Startup
-------
    chatterbox_worker.py --device mps [--t3-checkpoint t3_fa.safetensors]

``--t3-checkpoint`` swaps the text-to-speech-token transformer for a
language-specific finetune, which is how Persian works: the base model's
speaker path (voice encoder and S3Gen) is language-agnostic and stays, only
the module that turns text into speech tokens is replaced. The first line on
stdout is ``{"ready": true}`` once the model is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def load_model(device: str, t3_checkpoint: str | None) -> Any:
    import torch
    from chatterbox.tts import ChatterboxTTS

    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception as exc:
        if device == "cpu":
            raise
        # Metal has gaps; falling back is better than failing the whole run,
        # and the parent is told which device actually ran.
        _log(f"chatterbox: {device} failed ({exc}); retrying on cpu")
        model = ChatterboxTTS.from_pretrained(device="cpu")

    if t3_checkpoint:
        from safetensors.torch import load_file

        state = load_file(t3_checkpoint, device="cpu")
        missing, unexpected = model.t3.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Not fatal - finetunes often omit buffers - but a large mismatch
            # means the wrong checkpoint, which would otherwise show up only
            # as mysteriously bad audio.
            _log(
                f"chatterbox: t3 loaded with {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys"
            )
        model.t3.to(next(model.t3.parameters()).device)
        model.t3.eval()

    torch.set_grad_enabled(False)
    return model


def synthesize(model: Any, job: dict[str, Any]) -> dict[str, Any]:
    import soundfile as sf
    import torch

    seed = job.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        with contextlib.suppress(Exception):
            torch.mps.manual_seed(int(seed))

    kwargs: dict[str, Any] = {}
    for name in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p"):
        if job.get(name) is not None:
            kwargs[name] = job[name]

    with torch.inference_mode():
        wav = model.generate(
            text=job["text"],
            audio_prompt_path=job["reference"],
            **kwargs,
        )

    samples = wav.detach().cpu().numpy().reshape(-1)
    sample_rate = int(getattr(model, "sr", 24000))
    sf.write(job["output"], samples, sample_rate)
    return {
        "ok": True,
        "output": job["output"],
        "sample_rate": sample_rate,
        "seconds": round(len(samples) / sample_rate, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t3-checkpoint", default=None)
    args = parser.parse_args()

    try:
        model = load_model(args.device, args.t3_checkpoint)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit({"ready": True, "sample_rate": int(getattr(model, "sr", 24000))})

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
