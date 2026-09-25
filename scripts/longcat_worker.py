#!/usr/bin/env python3
"""LongCat-AudioDiT synthesis worker. Runs the repository's own inference, out of process.

LongCat-AudioDiT (Meituan, arXiv 2603.29339) is a diffusion transformer in the
latent space of a waveform VAE, with a UMT5 text encoder. It is not a package:
its code is ``audiodit/`` plus ``utils.py`` at a pinned commit, put on the path
from ``--code-dir``. Every step below is ``inference.py``'s at that commit, in
its order, with its helpers imported rather than copied:

* ``AudioDiTModel.from_pretrained(...).to(device)`` - fp32 - then
  ``model.vae.to_half()``
* ``normalize_text`` on the prompt transcript and on the text, joined as
  ``f"{prompt_text} {text}"`` (the README's form; ``batch_inference.py`` drops
  the space unless the prompt ends in a full stop)
* the prompt encoded once to count its frames, ``approx_duration_from_text``
  for both texts, the ratio clipped to 1.0-1.5, capped at the model's
  ``max_wav_duration``
* ``model(..., steps, cfg_strength, guidance_method)`` - the README's
  voice-cloning call is 16 steps, strength 4.0, APG

The one thing that is not upstream's: ``--transformer-device``. The 3.5B's
fp32 weights (15.3 GB) do not fit one 15 GB T4 beside its text encoder, so the
transformer can sit on a second GPU. Its inputs are moved to it and its
outputs back, and nothing else changes - where it runs, not how it computes.

Protocol
--------
in   {"text": "...", "reference": "ref.wav", "reference_text": "...",
      "output": "out.wav", "seed": 42, "nfe": 16, "guidance_strength": 4.0,
      "guidance_method": "apg"}
out  {"ok": true, "output": "out.wav", "sample_rate": 24000, "seconds": 3.4,
      "frames": 138, "prompt_frames": 96, "full_text": "..."}
     {"ok": false, "error": "..."}

``{"op": "quit"}`` shuts it down. The first line on stdout is ``{"ready": true,
...}`` once the model is loaded; stderr is diagnostic only.

    longcat_worker.py --model-dir /tmp/longcat-1b --code-dir /tmp/longcat-code \\
        --tokenizer-dir /tmp/umt5-base --device cuda [--transformer-device cuda:1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

#: Frames of padding the prompt is encoded with and then cut, as upstream.
PROMPT_PAD_FRAMES = 3

_PROTOCOL: Any = None


def _claim_stdout() -> Any:
    """Take sole ownership of stdout: the protocol is one JSON object per line."""
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def joined_text(prompt_text: str, text: str) -> str:
    """What the text encoder reads when a prompt is given: its transcript, a space, the text."""
    return f"{prompt_text} {text}"


def frames(prompt_frames: int, text: str, prompt_text: str, approx: Callable[..., float],
           sample_rate: int, hop: int, max_duration: float) -> int:
    """``inference.py``'s duration, in latent frames, prompt included."""
    import numpy as np

    prompt_time = prompt_frames * hop / sample_rate
    seconds = approx(text, max_duration=max_duration - prompt_time)
    prompt_estimate = approx(prompt_text, max_duration=max_duration)
    seconds = seconds * np.clip(prompt_time / prompt_estimate, 1.0, 1.5)
    duration = int(seconds * sample_rate // hop)
    return min(duration + prompt_frames, int(max_duration * sample_rate // hop))


def moved(value: Any, device: Any) -> Any:
    """Every tensor inside ``value`` on ``device``; everything else as it was."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: moved(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(moved(item, device) for item in value)
    return value


def place(module: Any, device: Any, home: Any) -> None:
    """Run ``module`` on ``device`` while its callers keep their tensors on ``home``."""

    def inputs(_: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        return moved(args, device), moved(kwargs, device)

    def outputs(_: Any, args: tuple, kwargs: dict, result: Any) -> Any:
        return moved(result, home)

    module.to(device)
    module.register_forward_pre_hook(inputs, with_kwargs=True)
    module.register_forward_hook(outputs, with_kwargs=True)


def load(model_dir: str, code_dir: str, tokenizer_dir: str, device: str,
         transformer_device: str) -> tuple[Any, Any, Any]:
    sys.path.insert(0, code_dir)
    import audiodit  # noqa: F401 - registers AudioDiTConfig and AudioDiTModel with transformers
    import torch
    import utils as official
    from audiodit import AudioDiTModel
    from transformers import AutoTokenizer

    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    if transformer_device:
        model = AudioDiTModel.from_pretrained(model_dir)
        model.text_encoder.to(device)
        model.vae.to(device)
        place(model.transformer, transformer_device, device)
    else:
        model = AudioDiTModel.from_pretrained(model_dir).to(device)
    model.vae.to_half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer, official


def describe(model: Any, transformer_device: str) -> dict[str, Any]:
    config = model.config
    return {
        "device": str(model.device),
        "transformer_device": transformer_device or str(model.device),
        "dtype": {name: str(next(getattr(model, name).parameters()).dtype).removeprefix("torch.")
                  for name in ("text_encoder", "transformer", "vae")},
        "parameters": sum(p.numel() for p in model.parameters()),
        "dit_depth": config.dit_depth,
        "dit_dim": config.dit_dim,
        "max_wav_duration": config.max_wav_duration,
    }


def synthesise(model: Any, tokenizer: Any, official: Any, job: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    seed = job.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    config = model.config
    rate, hop, limit = config.sampling_rate, config.latent_hop, config.max_wav_duration
    prompt_text = (job.get("reference_text") or "").strip()
    if not prompt_text:
        raise ValueError("LongCat clones from a prompt and its transcript; the transcript is empty")
    text = official.normalize_text(job["text"])
    prompt_text = official.normalize_text(prompt_text)
    full = joined_text(prompt_text, text)
    inputs = tokenizer([full], padding="longest", return_tensors="pt")

    prompt = official.load_audio(job["reference"], rate)
    padded = prompt
    if padded.shape[-1] % hop != 0:
        padded = F.pad(padded, (0, hop - padded.shape[-1] % hop))
    padded = F.pad(padded, (0, hop * PROMPT_PAD_FRAMES))
    latent = model.vae.encode(padded.unsqueeze(0).to(model.device))[..., :-PROMPT_PAD_FRAMES]
    prompt_frames = int(latent.shape[-1])
    duration = frames(prompt_frames, text, prompt_text, official.approx_duration_from_text, rate, hop, limit)

    output = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_audio=prompt.unsqueeze(0),
        duration=duration,
        steps=int(job["nfe"]),
        cfg_strength=float(job["guidance_strength"]),
        guidance_method=str(job["guidance_method"]),
    )
    samples = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("LongCat returned no audio")
    if not np.isfinite(samples).all():
        raise RuntimeError("non-finite samples in the render")
    sf.write(job["output"], samples, rate)
    return {"ok": True, "output": job["output"], "sample_rate": int(rate),
            "seconds": round(samples.size / rate, 3), "frames": duration,
            "prompt_frames": prompt_frames, "full_text": full}


def main() -> int:
    global _PROTOCOL
    _PROTOCOL = _claim_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transformer-device", default="")
    args = parser.parse_args()

    try:
        model, tokenizer, official = load(args.model_dir, args.code_dir, args.tokenizer_dir, args.device,
                                          args.transformer_device)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit({"ready": True, "sample_rate": int(model.config.sampling_rate), "model_dir": args.model_dir,
           **describe(model, args.transformer_device)})

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            job = json.loads(stripped)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"bad job: {exc}"})
            continue
        if job.get("op") == "quit":
            break
        try:
            _emit(synthesise(model, tokenizer, official, job))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
