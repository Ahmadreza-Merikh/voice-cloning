#!/usr/bin/env python3
"""Pocket-TTS Farsi synthesis worker. Runs in its own interpreter, not this project's.

`mehdi-hf/pocket-tts-farsi-v2` is a 109.5 M-parameter Persian model trained on
973 hours from 2,978 speakers, distilled from Kyutai's pocket-tts and small
enough to clone a voice on a CPU faster than real time. It is here to be
measured against the engines this project already scores.

Two things make it unlike every other engine here, and both come from its own
model card rather than from a guess:

* **It reads phonemes, not Persian script.** Persian text handed to it maps
  entirely to the unknown token and it emits end-of-speech immediately - silence,
  with no error. So this worker phonemises first, with the G2P the card names
  (`mehdi-hf/Homo-GE2PE-Persian-HF`), the normaliser shipped beside the weights,
  and the card's own symbol translation. That G2P is Homo-GE2PE, the same family
  as the Persian G2P research this project already bridges to.
* **Chunk length is the constraint.** Its training utterances averaged about 11
  tokens; the card puts the clean band at 9-16, calls 18 a safe budget and says
  generations stop terminating at 21 and above. So text is split into sentences
  before phonemisation - punctuation does not survive the G2P - and the library's
  own chunker is given the card's budget.

A generation that never emits end-of-speech runs to the cap and repeats itself.
The card's test is the duration: compare against ``tokens / 3 + 2`` seconds and
generate again. That is done here, up to ``--retries`` times, and reported.

Protocol
--------
in   {"text": "...", "reference": "ref.wav", "output": "out.wav", "seed": 42}
out  {"ok": true, "output": "out.wav", "sample_rate": 24000, "seconds": 3.4,
      "phonemes": ["salAm ..."], "retried": 0}
     {"ok": false, "error": "..."}

A line of ``{"op": "quit"}`` shuts it down. The first line on stdout is
``{"ready": true}`` once both models are loaded; stderr is diagnostic only.

    pocket_worker.py --model-dir ~/.cache/persian-voice-cloning/pocket/model \\
                     --g2p-dir ~/.cache/persian-voice-cloning/pocket/g2p
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


def _claim_stdout() -> Any:
    """Take sole ownership of stdout: the protocol is one JSON object per line."""
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


_PROTOCOL = _claim_stdout()

#: The card's table: the G2P's compact output is not quite the model's alphabet.
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})
#: The G2P marks the ezafe with a trailing 1. It keeps a bound phrase together
#: when splitting, and the model must never see it.
EZAFE = "1"
SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def local_config(model_dir: Path) -> Path:
    """A copy of the shipped ``model.yaml`` pointing at the local files.

    The shipped one names ``hf://`` paths, which would reach the network on
    every load. Its three flags are kept exactly as they are: without
    ``capitalize_first_letter: false`` the frontend upper-cases the first
    letter of every chunk, and since these Latin letters are phonemes that
    either drops the first word or turns `salAm` into /shalaam/.
    """
    source = (model_dir / "model.yaml").read_text(encoding="utf-8")
    target = model_dir / "model.local.yaml"
    target.write_text(
        source.replace("hf://mehdi-hf/pocket-tts-farsi-v2/model.safetensors",
                       str(model_dir / "model.safetensors"))
              .replace("hf://mehdi-hf/pocket-tts-farsi-v2/tokenizer_ph.model",
                       str(model_dir / "tokenizer_ph.model")),
        encoding="utf-8")
    return target


class Phonemiser:
    """Persian script -> the romanised phonemes this model was trained on."""

    def __init__(self, model_dir: Path, g2p_dir: Path) -> None:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        sys.path.insert(0, str(model_dir))
        from normalize_fa import normalize_for_model

        self._torch = torch
        self._normalise = normalize_for_model
        self._tokeniser = AutoTokenizer.from_pretrained(str(g2p_dir))
        self._model = T5ForConditionalGeneration.from_pretrained(str(g2p_dir)).eval()

    def __call__(self, text: str) -> str:
        # Normalise first: the G2P has no reading for digits and drops them
        # without a word, so "تا سال ۲۰۳۰" would lose the year silently.
        cleaned = self._normalise(text)
        # The Persian question mark shares its symbol with the glottal stop in
        # this notation; their own corpus work found 6.6% of utterances carrying
        # a glottal stop nobody uttered until it was stripped before phonemising.
        cleaned = cleaned.replace("؟", "").replace("?", "")
        encoded = self._tokeniser([cleaned], add_special_tokens=False, return_tensors="pt")
        with self._torch.no_grad():
            generated = self._model.generate(**encoded, num_beams=5, max_length=512,
                                             early_stopping=True)
        raw = self._tokeniser.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return raw.translate(TO_PHONEMES)


def sentences(text: str) -> list[str]:
    """Split before phonemising: the G2P discards punctuation, boundaries and all."""
    found = [part.strip() for part in SENTENCE_END.split(text.strip()) if part.strip()]
    return found or [text.strip()]


def load_models(model_dir: Path, g2p_dir: Path) -> tuple[Any, Phonemiser, Any]:
    import sentencepiece
    from pocket_tts import TTSModel

    phonemiser = Phonemiser(model_dir, g2p_dir)
    model = TTSModel.load_model(config=str(local_config(model_dir)))
    tokeniser = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "tokenizer_ph.model"))
    return model, phonemiser, tokeniser


def synthesise(model: Any, phonemiser: Phonemiser, tokeniser: Any, job: dict[str, Any],
               max_tokens: int, retries: int) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    seed = job.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    state = model.get_state_for_audio_prompt(job["reference"], truncate=True)
    gap = np.zeros(int(0.25 * model.sample_rate), dtype=np.float32)

    pieces: list[np.ndarray] = []
    spoken: list[str] = []
    retried = 0
    for sentence in sentences(job["text"]):
        phonemes = phonemiser(sentence)
        for_model = phonemes.replace(EZAFE, "")
        if not for_model.strip():
            continue
        # The card's runaway test: a generation that never terminates repeats
        # itself to the length cap, and a second attempt usually ends properly.
        cap = len(tokeniser.encode(for_model)) / 3.0 + 2.0
        for attempt in range(retries + 1):
            audio = model.generate_audio(state, for_model, max_tokens=max_tokens)
            samples = audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if len(samples) / model.sample_rate <= cap or attempt == retries:
                break
            retried += 1
            _log(f"pocket: runaway ({len(samples) / model.sample_rate:.1f}s over {cap:.1f}s cap), retrying")
        pieces.append(samples)
        spoken.append(phonemes)

    if not pieces:
        raise RuntimeError("nothing to synthesise after phonemisation")

    joined = pieces[0]
    for piece in pieces[1:]:
        joined = np.concatenate([joined, gap, piece])
    sf.write(job["output"], joined, model.sample_rate)
    return {"ok": True, "output": job["output"], "sample_rate": int(model.sample_rate),
            "seconds": round(len(joined) / model.sample_rate, 3), "phonemes": spoken,
            "retried": retried}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--g2p-dir", required=True)
    #: The card's safe budget; 21 and above stop terminating.
    parser.add_argument("--max-tokens", type=int, default=18)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    try:
        model, phonemiser, tokeniser = load_models(Path(args.model_dir).expanduser(),
                                                   Path(args.g2p_dir).expanduser())
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        _log(traceback.format_exc())
        return 1

    _emit({"ready": True, "sample_rate": int(model.sample_rate)})

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
            _emit(synthesise(model, phonemiser, tokeniser, job, args.max_tokens, args.retries))
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            _log(traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
