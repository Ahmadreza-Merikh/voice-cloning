"""Speaker embeddings with one preprocessing path and a cache.

``speaker_similarity`` in :mod:`.similarity` answers "how close are these two
clips" for a single render, which is what the runtime needs. Measurement work
needs something different: hundreds of comparisons over the same few dozen
clips, and - more importantly - a *guarantee* that every clip reaching the
encoder was treated identically.

That guarantee is the point. A real recording straight off disk and a render
that has been loudness-normalised to -18 LUFS are not the same treatment, and
the difference lands in the cosine score rather than in anything about who is
speaking. Comparing a clone against a human baseline measured the other way
produced this project's worst measurement error: a synthetic voice appearing to
resemble a speaker more closely than her own second recording did.

So everything here is peak-normalised and resampled the same way, and the
embeddings are cached by key because a calibration pass compares each clip
against every other one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..audio.io import Audio

log = get_logger("evaluation.embedding")

#: Peak the encoder sees for every clip. Below clipping, above the encoder's
#: own quiet-input behaviour, and identical for real and synthetic audio.
TARGET_PEAK = 0.95


class EmbeddingUnavailable(RuntimeError):
    """No speaker-verification model is installed."""


class CachedEmbedder:
    """ECAPA-TDNN embeddings, uniformly preprocessed and memoised by key.

    Keys are caller-chosen and must be unique per distinct audio: a filename
    for corpus clips, something carrying the render parameters for synthesis.
    A colliding key silently returns the wrong vector, so prefer keys that are
    obviously unique.
    """

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._model: Any = None
        self._backend = "none"

    @property
    def backend(self) -> str:
        return self._backend

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from .similarity import _load_speechbrain

        model = _load_speechbrain()
        if model is None:
            raise EmbeddingUnavailable(
                "No speaker-verification model available. Every similarity number "
                "would be a spectral fallback rather than a speaker embedding. "
                "Install one with:  pip install speechbrain"
            )
        self._model = model
        self._backend = "speechbrain-ecapa"
        return model

    @staticmethod
    def preprocess(audio: Audio) -> Audio:
        """Mono, peak-normalised. The only treatment any clip receives."""
        from ..audio.io import Audio as AudioType

        mono = audio.to_mono()
        samples = np.asarray(mono.samples, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = samples * (TARGET_PEAK / peak)
        return AudioType(samples, mono.sample_rate, source=mono.source)

    def embed(self, audio: Audio, key: str) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        from .similarity import _embed_speechbrain

        vector = _embed_speechbrain(self._load(), self.preprocess(audio))
        if vector is None:
            raise RuntimeError(f"Could not embed {key!r}")
        result = np.asarray(vector, dtype=np.float64)
        self._cache[key] = result
        return result

    def similarity(self, left: Audio, left_key: str, right: Audio, right_key: str) -> float:
        from .similarity import cosine_similarity

        return float(cosine_similarity(self.embed(left, left_key), self.embed(right, right_key)))


__all__ = ["CachedEmbedder", "EmbeddingUnavailable", "TARGET_PEAK"]
