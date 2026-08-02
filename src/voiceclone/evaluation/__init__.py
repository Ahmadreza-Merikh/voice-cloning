"""Objective evaluation of cloning quality."""

from .embedding import CachedEmbedder, EmbeddingUnavailable
from .similarity import SimilarityResult, available_backends, speaker_similarity, word_error_rate

__all__ = [
    "CachedEmbedder",
    "EmbeddingUnavailable",
    "SimilarityResult",
    "available_backends",
    "speaker_similarity",
    "word_error_rate",
]
