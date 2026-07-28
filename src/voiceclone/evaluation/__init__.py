"""Objective evaluation of cloning quality."""

from .similarity import SimilarityResult, available_backends, speaker_similarity, word_error_rate

__all__ = ["SimilarityResult", "available_backends", "speaker_similarity", "word_error_rate"]
