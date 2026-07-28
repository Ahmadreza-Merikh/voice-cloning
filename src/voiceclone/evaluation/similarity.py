"""Objective quality measurement of a clone.

Speaker similarity is the number that actually answers "did this work?".
It is the cosine similarity between speaker embeddings of the reference and
the render, computed with a model trained for speaker verification.

Backends, best first:

* ``speechbrain`` ECAPA-TDNN - the standard for speaker verification
* ``resemblyzer``            - lighter GE2E encoder
* spectral fallback          - long-term average spectrum correlation; a
  coarse signal, clearly labelled as such

All of it is optional; missing backends simply mean no score.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..audio.io import Audio
from ..logging_utils import get_logger

log = get_logger("evaluation")

#: Speaker-verification models expect 16 kHz.
EMBEDDING_SAMPLE_RATE = 16000


@dataclass
class SimilarityResult:
    score: float
    backend: str
    #: True when the score comes from a real speaker-verification model.
    reliable: bool = True


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_speechbrain() -> Any | None:
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
    except Exception:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        except Exception:
            log.debug("speechbrain unavailable", exc_info=True)
            return None

    try:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=None,
            run_opts={"device": "cpu"},
        )
    except Exception:
        log.debug("Could not load ECAPA-TDNN", exc_info=True)
        return None


@lru_cache(maxsize=1)
def _load_resemblyzer() -> Any | None:
    try:
        from resemblyzer import VoiceEncoder  # type: ignore

        return VoiceEncoder("cpu")
    except Exception:
        log.debug("resemblyzer unavailable", exc_info=True)
        return None


def _embed_speechbrain(model: Any, audio: Audio) -> np.ndarray | None:
    try:
        import torch

        work = audio.to_mono().resample(EMBEDDING_SAMPLE_RATE)
        tensor = torch.from_numpy(np.ascontiguousarray(work.samples)).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_batch(tensor)
        return embedding.squeeze().detach().cpu().numpy().reshape(-1)
    except Exception:
        log.debug("ECAPA embedding failed", exc_info=True)
        return None


def _embed_resemblyzer(model: Any, audio: Audio) -> np.ndarray | None:
    try:
        from resemblyzer import preprocess_wav  # type: ignore

        work = audio.to_mono().resample(EMBEDDING_SAMPLE_RATE)
        return np.asarray(model.embed_utterance(preprocess_wav(work.samples, EMBEDDING_SAMPLE_RATE)))
    except Exception:
        log.debug("Resemblyzer embedding failed", exc_info=True)
        return None


def _spectral_fingerprint(audio: Audio) -> np.ndarray:
    """Long-term average spectrum - a crude proxy for vocal timbre.

    Correlates with speaker identity because it captures formant structure,
    but it also captures recording conditions, so treat it as a hint only.
    """
    work = audio.to_mono().resample(EMBEDDING_SAMPLE_RATE)
    samples = work.samples
    if samples.size < 1024:
        return np.zeros(64, dtype=np.float64)

    n_fft = 1024
    hop = n_fft // 2
    window = np.hanning(n_fft)
    frames = 1 + (samples.size - n_fft) // hop
    if frames < 1:
        return np.zeros(64, dtype=np.float64)

    accumulator = np.zeros(n_fft // 2 + 1)
    for index in range(min(frames, 400)):
        start = index * hop
        accumulator += np.abs(np.fft.rfft(samples[start : start + n_fft] * window))
    spectrum = accumulator / min(frames, 400)

    # Compress into mel-ish bands and normalise out overall level.
    bands = np.array_split(spectrum, 64)
    profile = np.array([float(np.mean(band)) for band in bands])
    profile = np.log(profile + 1e-10)
    return (profile - profile.mean()) / (profile.std() + 1e-9)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def available_backends() -> dict[str, bool]:
    return {
        "speechbrain": importlib.util.find_spec("speechbrain") is not None,
        "resemblyzer": importlib.util.find_spec("resemblyzer") is not None,
    }


def speaker_similarity(reference: Audio, generated: Audio) -> SimilarityResult | None:
    """Cosine similarity between the two voices, in ``[-1, 1]``.

    Rough reading for zero-shot cloning: >0.80 excellent, 0.70-0.80 strong,
    0.60-0.70 fair, below 0.60 the clone did not take.
    """
    if reference.num_frames == 0 or generated.num_frames == 0:
        return None

    model = _load_speechbrain()
    if model is not None:
        left = _embed_speechbrain(model, reference)
        right = _embed_speechbrain(model, generated)
        if left is not None and right is not None:
            return SimilarityResult(cosine_similarity(left, right), "ecapa-tdnn", True)

    model = _load_resemblyzer()
    if model is not None:
        left = _embed_resemblyzer(model, reference)
        right = _embed_resemblyzer(model, generated)
        if left is not None and right is not None:
            return SimilarityResult(cosine_similarity(left, right), "resemblyzer", True)

    try:
        score = cosine_similarity(_spectral_fingerprint(reference), _spectral_fingerprint(generated))
        return SimilarityResult(score, "spectral (approximate)", False)
    except Exception:
        log.debug("Spectral similarity failed", exc_info=True)
        return None


#: Punctuation to discard before scoring, Persian and Latin.
_WER_PUNCTUATION = "،؛؟!.:«»\"'()[]{}-–—…,;?"


def normalize_for_wer(text: str) -> str:
    """Strip everything ASR cannot be expected to reproduce.

    Whisper returns unpunctuated text and does not emit ZWNJ, so comparing
    raw strings scores a *perfect* render at ~35% WER. Levelling both sides
    first is what makes the number mean "did it say the right words".
    """
    import re
    import unicodedata

    cleaned = unicodedata.normalize("NFC", text or "")
    # ZWNJ joins parts of one word; ASR writes them as separate words, so
    # treat it as a space on both sides rather than guessing.
    cleaned = cleaned.replace("‌", " ")
    cleaned = "".join(" " if char in _WER_PUNCTUATION else char for char in cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def word_error_rate(reference_text: str, hypothesis: str) -> float | None:
    """WER between the requested text and what ASR heard back.

    Both sides are punctuation- and ZWNJ-normalised first, so the score
    reflects pronunciation and intelligibility rather than typography.
    """
    if not reference_text or not hypothesis:
        return None

    reference_text = normalize_for_wer(reference_text)
    hypothesis = normalize_for_wer(hypothesis)
    if not reference_text:
        return None

    try:
        import jiwer  # type: ignore

        return float(jiwer.wer(reference_text, hypothesis))
    except Exception:
        log.debug("jiwer unavailable; using the built-in edit distance", exc_info=True)

    reference_words = reference_text.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return None

    # Standard Levenshtein over words, two-row rolling table.
    previous = list(range(len(hypothesis_words) + 1))
    for i, ref_word in enumerate(reference_words, start=1):
        current = [i] + [0] * len(hypothesis_words)
        for j, hyp_word in enumerate(hypothesis_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current

    return previous[-1] / len(reference_words)


__all__ = [
    "SimilarityResult",
    "available_backends",
    "cosine_similarity",
    "normalize_for_wer",
    "speaker_similarity",
    "word_error_rate",
]
