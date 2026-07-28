"""Assembling and mastering synthesised audio.

Long-form text is synthesised chunk by chunk. Joining those chunks badly is
one of the most audible failure modes in TTS pipelines - clicks at the seams,
inconsistent loudness, or unnatural pacing. This module handles the join and
the final master.
"""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .io import Audio
from .loudness import fade_edges, normalize_loudness
from .vad import detect_speech

log = get_logger("audio.postprocess")


def equal_power_crossfade(left: np.ndarray, right: np.ndarray, n_fade: int) -> np.ndarray:
    """Join two blocks with a constant-power crossfade.

    Equal-power (cosine/sine) rather than linear: a linear crossfade dips in
    perceived loudness at the midpoint when the two sides are uncorrelated,
    which synthesised chunks always are.
    """
    n_fade = int(min(n_fade, left.size, right.size))
    if n_fade <= 1:
        return np.concatenate([left, right])

    angle = np.linspace(0.0, np.pi / 2.0, n_fade, dtype=np.float32)
    fade_out = np.cos(angle)
    fade_in = np.sin(angle)
    blended = left[-n_fade:] * fade_out + right[:n_fade] * fade_in
    return np.concatenate([left[:-n_fade], blended, right[n_fade:]])


def trim_silence_edges(audio: Audio, *, threshold_db: float = -45.0, keep_ms: int = 40) -> Audio:
    """Trim near-silence from the head and tail of a render.

    Engines commonly emit a little padding; removing it keeps chunk joins
    tight and stops gaps from compounding.
    """
    if audio.num_frames == 0:
        return audio

    mono = audio.to_mono().samples
    threshold = 10.0 ** (threshold_db / 20.0)
    loud = np.where(np.abs(mono) > threshold)[0]
    if loud.size == 0:
        return audio

    keep = int(round(keep_ms * audio.sample_rate / 1000.0))
    start = max(0, int(loud[0]) - keep)
    end = min(audio.num_frames, int(loud[-1]) + keep)
    return audio.with_samples(audio.samples[..., start:end])


def strip_reference_leakage(audio: Audio, *, max_lead_s: float = 0.6, vad_backend: str = "auto") -> Audio:
    """Drop a leading silence/artefact block before the first speech.

    Some zero-shot engines echo a fragment of the reference prompt or emit a
    breath before the target text starts.
    """
    if audio.duration < 0.5:
        return audio
    try:
        segments = detect_speech(audio, backend=vad_backend)
    except Exception:
        return audio
    if not segments:
        return audio

    first_start = segments[0][0]
    if 0.05 < first_start <= max_lead_s:
        return audio.slice_seconds(max(0.0, first_start - 0.05), None)
    return audio


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sample_rate))), dtype=np.float32)


def join_chunks(
    chunks: list[Audio],
    *,
    sample_rate: int,
    crossfade_ms: int = 40,
    gaps_ms: list[int] | None = None,
    trim_each: bool = True,
) -> Audio:
    """Stitch synthesised chunks into one continuous take.

    Args:
        chunks: rendered pieces, in order.
        gaps_ms: silence to insert *before* chunk ``i`` (``len == len(chunks)``
            or ``None`` for a uniform gap). Element 0 is ignored.
        crossfade_ms: applied only where the gap is zero; with a real pause
            there is nothing to crossfade.
    """
    usable = [c for c in chunks if c is not None and c.num_frames > 0]
    if not usable:
        return Audio(np.zeros(0, dtype=np.float32), sample_rate)
    if len(usable) == 1:
        single = usable[0].resample(sample_rate).to_mono()
        return trim_silence_edges(single) if trim_each else single

    prepared: list[np.ndarray] = []
    for chunk in usable:
        piece = chunk.resample(sample_rate).to_mono()
        if trim_each:
            piece = trim_silence_edges(piece)
        prepared.append(piece.samples)

    n_fade = int(round(crossfade_ms * sample_rate / 1000.0))
    result = prepared[0]

    for index, piece in enumerate(prepared[1:], start=1):
        gap_ms = 0
        if gaps_ms is not None and index < len(gaps_ms):
            gap_ms = max(0, int(gaps_ms[index]))

        if gap_ms > 0:
            result = np.concatenate([result, silence(gap_ms / 1000.0, sample_rate), piece])
        else:
            result = equal_power_crossfade(result, piece, n_fade)

    log.debug("Joined %d chunks -> %.2fs", len(prepared), result.size / sample_rate)
    return Audio(result, sample_rate)


def master(
    audio: Audio,
    *,
    target_lufs: float | None = -18.0,
    peak_dbfs: float = -1.0,
    fade_ms: int = 12,
    trim_edges: bool = True,
) -> Audio:
    """Final delivery chain: trim -> loudness -> peak ceiling -> de-click."""
    result = audio
    if trim_edges:
        result = trim_silence_edges(result)
    result = normalize_loudness(result, target_lufs, peak_ceiling_dbfs=peak_dbfs)
    return fade_edges(result, fade_ms)


def concatenate(audios: list[Audio], sample_rate: int, gap_ms: int = 0) -> Audio:
    """Plain concatenation with an optional uniform gap."""
    gaps = [0] + [gap_ms] * (len(audios) - 1) if len(audios) > 1 else None
    return join_chunks(audios, sample_rate=sample_rate, crossfade_ms=0, gaps_ms=gaps, trim_each=False)


__all__ = [
    "concatenate",
    "equal_power_crossfade",
    "join_chunks",
    "master",
    "silence",
    "strip_reference_leakage",
    "trim_silence_edges",
]
