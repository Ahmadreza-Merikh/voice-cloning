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


def _text_vocab_size(state: dict[str, Any]) -> int | None:
    """Rows of the T3 text embedding, i.e. how many characters it knows."""
    for key, tensor in state.items():
        if key.endswith("text_emb.weight"):
            return int(tensor.shape[0])
    return None


def _build(kind: str, device: str) -> Any:
    """Instantiate one of the two Chatterbox stacks, falling back off Metal."""
    if kind == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
    else:
        from chatterbox.tts import ChatterboxTTS as Model

    try:
        return Model.from_pretrained(device=device)
    except Exception as exc:
        if device == "cpu":
            raise
        # Metal has gaps; falling back beats failing the run, and the parent
        # is told which device actually ran.
        _log(f"chatterbox: {kind} on {device} failed ({exc}); retrying on cpu")
        return Model.from_pretrained(device="cpu")


def load_model(device: str, t3_checkpoint: str | None) -> tuple[Any, str]:
    """Load the stack the checkpoint actually belongs to.

    Chatterbox ships two: an English model whose character tokenizer holds 704
    tokens and no Arabic script at all, and a multilingual one whose tokenizer
    holds 2352 and covers every Persian letter. A Persian T3 is only meaningful
    on the second, but the finetunes distribute a bare ``t3_*.safetensors`` with
    no note of which - and loading it onto the wrong stack does not raise. The
    text embedding would simply be the wrong size, the weights would be dropped
    by ``strict=False``, and the result would be fluent-sounding gibberish that
    looks like a model quality problem.

    So the checkpoint picks the stack: whichever base has a text embedding of
    matching height is the one it was trained against.
    """
    import torch

    torch.set_grad_enabled(False)

    if not t3_checkpoint:
        return _build("multilingual", device), "multilingual"

    from safetensors.torch import load_file

    state = load_file(t3_checkpoint, device="cpu")
    wanted = _text_vocab_size(state)
    _log(f"chatterbox: checkpoint text vocabulary is {wanted}")

    last_error: Exception | None = None
    for kind in ("multilingual", "english"):
        try:
            model = _build(kind, device)
        except Exception as exc:  # noqa: BLE001 - try the other stack
            last_error = exc
            _log(f"chatterbox: could not build the {kind} stack: {exc}")
            continue

        have = int(model.t3.text_emb.weight.shape[0])
        if wanted is not None and wanted != have:
            _log(f"chatterbox: {kind} stack has {have} text tokens, checkpoint wants {wanted}")
            del model
            continue

        missing, unexpected = model.t3.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Finetunes legitimately omit buffers, but a large mismatch means
            # the wrong checkpoint - which otherwise shows up only as
            # mysteriously bad audio.
            _log(
                f"chatterbox: t3 loaded with {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys"
            )
        model.t3.to(next(model.t3.parameters()).device)
        model.t3.eval()
        _log(f"chatterbox: using the {kind} stack")
        return model, kind

    raise RuntimeError(
        f"No Chatterbox stack matches this T3 (text vocabulary {wanted}). "
        f"Last error: {last_error}"
    )


def synthesize(model: Any, kind: str, language_id: str, job: dict[str, Any]) -> dict[str, Any]:
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
    if kind == "multilingual":
        # Required by the multilingual signature. Persian is not one of the 23
        # languages the base model was released with, so a Persian T3 has to be
        # driven under some other tag; which one is a property of how it was
        # trained, hence configurable rather than assumed.
        kwargs["language_id"] = job.get("language_id") or language_id

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
    parser.add_argument(
        "--language-id",
        default="ar",
        help="Tag for the multilingual stack. Persian is not one of its 23 languages; "
             "Arabic is the nearest script and the default until measurement says otherwise.",
    )
    args = parser.parse_args()

    try:
        model, kind = load_model(args.device, args.t3_checkpoint)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit({"ready": True, "sample_rate": int(getattr(model, "sr", 24000)), "stack": kind})

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
            _emit(synthesize(model, kind, args.language_id, job))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
