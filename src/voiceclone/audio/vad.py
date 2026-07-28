"""Voice activity detection and silence handling.

Two backends with the same interface:

* ``silero``  - neural VAD, accurate in noise (optional dependency)
* ``energy``  - adaptive energy + zero-crossing gate, always available

Both return speech regions as ``(start_seconds, end_seconds)`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from ..logging_utils import get_logger
from .io import Audio

log = get_logger("audio.vad")

Segment = tuple[float, float]

#: Silero is trained at 8/16 kHz only; input is resampled to this rate.
_SILERO_RATE = 16000


class VADBackend(Protocol):
    name: str

    def detect(self, audio: Audio, threshold: float) -> list[Segment]: ...


# --------------------------------------------------------------------------
# Silero
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_silero() -> object | None:
    """Load the Silero VAD model once, trying pip package then torch.hub."""
    try:
        from silero_vad import load_silero_vad  # type: ignore

        log.debug("Using silero-vad package")
        return load_silero_vad(onnx=False)
    except Exception:
        log.debug("silero_vad package unavailable", exc_info=True)

    try:
        import torch

        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        log.debug("Using silero VAD from torch.hub")
        return model
    except Exception:
        log.debug("torch.hub silero unavailable", exc_info=True)

    return None


class SileroVAD:
    name = "silero"

    def __init__(self, model: object) -> None:
        self._model = model

    def detect(self, audio: Audio, threshold: float = 0.5) -> list[Segment]:
        import torch

        work_rate = _SILERO_RATE
        work = audio.to_mono().resample(work_rate)
        tensor = torch.from_numpy(np.ascontiguousarray(work.samples))

        try:
            from silero_vad import get_speech_timestamps  # type: ignore

            stamps = get_speech_timestamps(
                tensor,
                self._model,
                sampling_rate=work_rate,
                threshold=float(threshold),
                min_speech_duration_ms=120,
                min_silence_duration_ms=180,
                speech_pad_ms=0,
            )
        except Exception:
            # torch.hub variant exposes the helper through the model bundle.
            from torch.hub import load as hub_load  # type: ignore

            _, utils = hub_load("snakers4/silero-vad", "silero_vad", trust_repo=True)
            stamps = utils[0](
                tensor, self._model, sampling_rate=work_rate, threshold=float(threshold)
            )

        return [(s["start"] / work_rate, s["end"] / work_rate) for s in stamps]


# --------------------------------------------------------------------------
# Energy fallback
# --------------------------------------------------------------------------
class EnergyVAD:
    """Adaptive energy gate. No dependencies, decent on clean recordings."""

    name = "energy"

    def __init__(self, frame_ms: float = 30.0, hop_ms: float = 10.0) -> None:
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms

    def detect(self, audio: Audio, threshold: float = 0.5) -> list[Segment]:
        mono = audio.to_mono()
        samples = mono.samples
        sample_rate = mono.sample_rate

        frame = max(1, int(sample_rate * self.frame_ms / 1000.0))
        hop = max(1, int(sample_rate * self.hop_ms / 1000.0))
        if samples.size < frame:
            return [(0.0, mono.duration)] if mono.rms > 1e-4 else []

        n_frames = 1 + (samples.size - frame) // hop
        strided = np.lib.stride_tricks.as_strided(
            samples,
            shape=(n_frames, frame),
            strides=(samples.strides[0] * hop, samples.strides[0]),
            writeable=False,
        )
        energy_db = 20.0 * np.log10(
            np.maximum(np.sqrt(np.mean(np.square(strided.astype(np.float64)), axis=1)), 1e-12)
        )

        noise_floor = float(np.percentile(energy_db, 15))
        speech_level = float(np.percentile(energy_db, 95))
        span = max(speech_level - noise_floor, 6.0)
        # `threshold` in 0..1 slides the gate between noise floor and speech.
        gate = noise_floor + span * (0.25 + 0.45 * float(np.clip(threshold, 0.0, 1.0)))

        voiced = energy_db > gate

        # Zero-crossing rate rejects steady tones (hum, hiss) that pass the
        # energy gate but are not speech.
        zcr = np.mean(np.abs(np.diff(np.signbit(strided), axis=1)), axis=1)
        voiced &= (zcr > 0.005) & (zcr < 0.45)

        return _mask_to_segments(voiced, hop / sample_rate, frame / sample_rate, min_duration=0.12)


def _mask_to_segments(
    mask: np.ndarray, hop_s: float, frame_s: float, *, min_duration: float = 0.1
) -> list[Segment]:
    """Convert a per-frame boolean mask into merged time segments."""
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    segments: list[Segment] = []
    for start, end in zip(starts, ends, strict=False):
        begin = start * hop_s
        finish = (end - 1) * hop_s + frame_s
        if finish - begin >= min_duration:
            segments.append((float(begin), float(finish)))
    return segments


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_vad(backend: str = "auto") -> VADBackend:
    """Resolve a VAD backend by name, falling back gracefully."""
    backend = (backend or "auto").lower()

    if backend in {"auto", "silero"}:
        model = _load_silero()
        if model is not None:
            return SileroVAD(model)
        if backend == "silero":
            log.warning("Silero VAD unavailable; using the energy detector instead.")

    return EnergyVAD()


def detect_speech(audio: Audio, *, backend: str = "auto", threshold: float = 0.5) -> list[Segment]:
    """Return speech regions, or the whole clip if detection fails."""
    if audio.num_frames == 0:
        return []
    try:
        segments = get_vad(backend).detect(audio, threshold)
    except Exception:
        log.warning("VAD failed; treating the whole clip as speech.", exc_info=True)
        return [(0.0, audio.duration)]

    if not segments:
        log.debug("VAD found no speech; keeping the clip intact.")
        return [(0.0, audio.duration)]
    return segments


@dataclass
class TrimResult:
    audio: Audio
    segments: list[Segment]
    removed_seconds: float


def trim_and_compact(
    audio: Audio,
    *,
    backend: str = "auto",
    threshold: float = 0.5,
    pad_ms: int = 120,
    max_internal_silence: float = 0.35,
    crossfade_ms: int = 8,
) -> TrimResult:
    """Trim leading/trailing silence and shorten long internal pauses.

    Long pauses inside a reference prompt confuse zero-shot cloners: they get
    modelled as part of the speaker's rhythm. Capping them at
    ``max_internal_silence`` keeps delivery natural without sounding spliced.
    """
    if audio.num_frames == 0:
        return TrimResult(audio, [], 0.0)

    segments = detect_speech(audio, backend=backend, threshold=threshold)
    if not segments:
        return TrimResult(audio, [], 0.0)

    pad = max(0.0, pad_ms / 1000.0)
    sample_rate = audio.sample_rate
    gap_samples = max(0, int(round(max_internal_silence * sample_rate)))

    pieces: list[np.ndarray] = []
    previous_end: float | None = None

    for start, end in segments:
        begin = max(0.0, start - pad)
        finish = min(audio.duration, end + pad)
        if previous_end is not None:
            gap = begin - previous_end
            if gap > 0:
                keep = min(int(round(gap * sample_rate)), gap_samples)
                if keep > 0:
                    pieces.append(np.zeros(keep, dtype=np.float32))
        chunk = audio.slice_seconds(begin, finish).to_mono().samples
        if chunk.size:
            pieces.append(chunk)
        previous_end = finish

    if not pieces:
        return TrimResult(audio, segments, 0.0)

    joined = _concat_with_micro_crossfade(pieces, sample_rate, crossfade_ms)
    trimmed = audio.with_samples(joined)

    # A VAD that discards most of the clip is far more likely to be wrong than
    # the recording is to be mostly silence - quiet speakers and heavily
    # modulated delivery both trigger this. Keep the original rather than
    # hand a shredded prompt to the cloner.
    if trimmed.duration < audio.duration * 0.35:
        log.warning(
            "Silence trimming would discard %.0f%% of the audio; keeping it untrimmed.",
            100.0 * (1.0 - trimmed.duration / max(audio.duration, 1e-6)),
        )
        return TrimResult(audio, segments, 0.0)

    removed = audio.duration - trimmed.duration
    if removed > 0.05:
        log.debug("Trimmed %.2fs of silence (%d speech regions)", removed, len(segments))
    return TrimResult(trimmed, segments, max(0.0, removed))


def _concat_with_micro_crossfade(pieces: list[np.ndarray], sample_rate: int, crossfade_ms: int) -> np.ndarray:
    """Join blocks with a very short crossfade to avoid edge discontinuities."""
    n_fade = int(round(crossfade_ms * sample_rate / 1000.0))
    if n_fade <= 1:
        return np.concatenate(pieces)

    result = pieces[0]
    for piece in pieces[1:]:
        fade = min(n_fade, result.size, piece.size)
        if fade <= 1:
            result = np.concatenate([result, piece])
            continue
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        blended = result[-fade:] * (1.0 - ramp) + piece[:fade] * ramp
        result = np.concatenate([result[:-fade], blended, piece[fade:]])
    return result


def speech_duration(segments: list[Segment]) -> float:
    return float(sum(end - start for start, end in segments))


__all__ = [
    "EnergyVAD",
    "Segment",
    "SileroVAD",
    "TrimResult",
    "detect_speech",
    "get_vad",
    "speech_duration",
    "trim_and_compact",
]
