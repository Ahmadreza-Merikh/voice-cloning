"""Persian-first text processing: normalisation, numbers, G2P and chunking."""

from .chunker import TextChunk, chunk_text, estimate_duration, split_sentences
from .g2p import G2PResult, PersianG2P, find_g2p_repo, get_g2p
from .lang import detect_language, has_persian, is_mixed_script
from .normalize_fa import NormalizationOptions, normalize, normalize_for_g2p
from .numbers_fa import expand_numbers, number_to_words, ordinal_to_words
from .pipeline import PreparedText, TextPipeline
from .prosody_fa import break_vocative_clauses

__all__ = [
    "G2PResult",
    "NormalizationOptions",
    "PersianG2P",
    "PreparedText",
    "TextChunk",
    "TextPipeline",
    "break_vocative_clauses",
    "chunk_text",
    "detect_language",
    "estimate_duration",
    "expand_numbers",
    "find_g2p_repo",
    "get_g2p",
    "has_persian",
    "is_mixed_script",
    "normalize",
    "normalize_for_g2p",
    "number_to_words",
    "ordinal_to_words",
    "split_sentences",
]
