"""Sentence segmentation and chunking for long-form synthesis.

Zero-shot TTS models degrade past roughly 20-30 seconds per generation:
timbre drifts, pacing wanders, and some engines simply truncate. Splitting on
sentence boundaries and rendering piece by piece keeps every second at full
quality, so long text sounds the same at minute ten as at minute one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..logging_utils import get_logger

log = get_logger("text.chunker")

#: Sentence terminators, Persian and Latin.
_TERMINATORS = ".؟?!…"
_SENTENCE_SPLIT_RE = re.compile(rf"(?<=[{re.escape(_TERMINATORS)}])\s+")
#: Secondary break points used only when a sentence is still too long.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[،؛:])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

#: Characters per second of Persian speech at a natural pace. Used to predict
#: chunk duration before synthesis; measured, not guessed, on Persian corpora
#: where a syllable averages ~2.6 characters and speakers hit ~5.5 syl/s.
CHARS_PER_SECOND = 14.0


@dataclass
class TextChunk:
    """One unit of synthesis."""

    index: int
    text: str
    #: Silence to insert before this chunk, in milliseconds.
    gap_before_ms: int = 0
    is_paragraph_start: bool = False
    estimated_seconds: float = 0.0
    #: Populated by the text pipeline when phonemes are used.
    phonemes: str = ""

    @property
    def synth_text(self) -> str:
        """What the engine should actually receive."""
        return self.phonemes or self.text

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.text)


def estimate_duration(text: str, chars_per_second: float = CHARS_PER_SECOND) -> float:
    """Rough spoken duration of a string, in seconds."""
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped:
        return 0.0
    # Punctuation costs pause time rather than speaking time.
    pauses = sum(0.25 for char in stripped if char in "،؛:") + sum(
        0.4 for char in stripped if char in _TERMINATORS
    )
    return len(stripped) / max(chars_per_second, 1.0) + pauses


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminators attached."""
    if not text or not text.strip():
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text.strip())]
    return [part for part in parts if part]


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(text.strip()) if part.strip()]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence at clause boundaries, then at spaces."""
    if len(sentence) <= max_chars:
        return [sentence]

    pieces: list[str] = []
    buffer = ""
    for clause in _CLAUSE_SPLIT_RE.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        candidate = f"{buffer} {clause}".strip() if buffer else clause
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer)
            buffer = clause
    if buffer:
        pieces.append(buffer)

    # Still too long (a single clause without punctuation): break on words.
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            result.append(piece)
            continue
        words = piece.split()
        buffer = ""
        for word in words:
            candidate = f"{buffer} {word}".strip()
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)
                buffer = word
        if buffer:
            result.append(buffer)

    if any(len(piece) > max_chars for piece in result):
        log.debug("Sentence still exceeds %d chars after splitting", max_chars)
    return result or [sentence]


def chunk_text(
    text: str,
    *,
    max_chars: int = 220,
    max_seconds: float = 22.0,
    min_chars: int = 24,
    sentence_gap_ms: int = 180,
    paragraph_gap_ms: int = 380,
    chars_per_second: float = CHARS_PER_SECOND,
) -> list[TextChunk]:
    """Split text into synthesis-sized chunks.

    Sentences are packed greedily up to whichever limit bites first
    (``max_chars`` or ``max_seconds``). Very short trailing fragments are
    merged backwards - a two-word chunk gets odd, clipped prosody.
    """
    if not text or not text.strip():
        return []

    chunks: list[TextChunk] = []
    paragraphs = split_paragraphs(text) or [text.strip()]

    for para_index, paragraph in enumerate(paragraphs):
        sentences: list[str] = []
        for sentence in split_sentences(paragraph) or [paragraph]:
            sentences.extend(_split_long_sentence(sentence, max_chars))

        buffer = ""
        # Only the first chunk of a *subsequent* paragraph earns the longer gap.
        pending_paragraph_break = para_index > 0

        def emit(content: str, paragraph_start: bool) -> None:
            content = content.strip()
            if not content:
                return
            gap = 0
            if chunks:
                gap = paragraph_gap_ms if paragraph_start else sentence_gap_ms
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=content,
                    gap_before_ms=gap,
                    is_paragraph_start=paragraph_start,
                    estimated_seconds=estimate_duration(content, chars_per_second),
                )
            )

        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            too_long = len(candidate) > max_chars
            too_slow = estimate_duration(candidate, chars_per_second) > max_seconds

            if buffer and (too_long or too_slow):
                emit(buffer, pending_paragraph_break)
                pending_paragraph_break = False
                buffer = sentence
            else:
                buffer = candidate

        if buffer:
            emit(buffer, pending_paragraph_break)

    # Merge a runt final chunk into its predecessor - but never across a
    # paragraph boundary, which is deliberate structure the author asked for.
    if (
        len(chunks) >= 2
        and len(chunks[-1].text) < min_chars
        and not chunks[-1].is_paragraph_start
    ):
        tail = chunks.pop()
        previous = chunks[-1]
        previous.text = f"{previous.text} {tail.text}".strip()
        previous.estimated_seconds = estimate_duration(previous.text, chars_per_second)

    for position, chunk in enumerate(chunks):
        chunk.index = position

    total = sum(chunk.estimated_seconds for chunk in chunks)
    log.debug("Chunked %d chars -> %d chunk(s), ~%.1fs total", len(text), len(chunks), total)
    return chunks


__all__ = [
    "CHARS_PER_SECOND",
    "TextChunk",
    "chunk_text",
    "estimate_duration",
    "split_paragraphs",
    "split_sentences",
]
