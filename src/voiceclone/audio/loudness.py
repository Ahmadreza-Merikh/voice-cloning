"""Loudness measurement and normalisation (ITU-R BS.1770 / EBU R128)."""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .io import Audio

log = get_logger("audio.loudness")

#: Below this RMS the signal is effectively silence; skip gain maths.
_SILENCE_FLOOR = 1e-6


def amplitude_to_db(value: float) -> float:
    """Linear amplitude -> dBFS, with a floor instead of ``-inf``."""
    return 20.0 * np.log10(max(float(value), 1e-12))


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def measure_lufs(audio: Audio) -> float | None:
    """Integrated loudness in LUFS, or ``None`` if it cannot be measured.

    ``pyloudnorm`` needs at least ~400 ms (one BS.1770 block) of signal.
    """
    if audio.duration < 0.45 or audio.rms < _SILENCE_FLOOR:
        return None
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(audio.sample_rate)
        value = float(meter.integrated_loudness(audio.to_mono().samples.astype(np.float64)))
        return value if np.isfinite(value) else None
    except Exception:
        log.debug("pyloudnorm unavailable; falling back to RMS estimate", exc_info=True)
        # Rough stand-in: RMS dBFS is typically a few dB above LUFS for speech.
        return amplitude_to_db(audio.rms) - 3.0


def peak_dbfs(audio: Audio) -> float:
    return amplitude_to_db(audio.peak)


def apply_gain(audio: Audio, gain_db: float) -> Audio:
    if abs(gain_db) < 1e-6:
        return audio
    return audio.with_samples(audio.samples * db_to_amplitude(gain_db))


def peak_limit(audio: Audio, ceiling_dbfs: float = -1.0) -> Audio:
    """Bring peaks under a ceiling.

    Uses plain gain reduction when the signal only slightly exceeds the
    ceiling, and a soft-knee tanh curve when it is badly over - this avoids
    the harsh distortion of hard clipping while keeping transients intact.
    """
    if audio.num_frames == 0:
        return audio

    ceiling = db_to_amplitude(ceiling_dbfs)
    peak = audio.peak
    if peak <= ceiling or peak < _SILENCE_FLOOR:
        return audio

    overshoot_db = amplitude_to_db(peak) - ceiling_dbfs
    if overshoot_db <= 6.0:
        return audio.with_samples(audio.samples * (ceiling / peak))

    # Heavily over: normalise to the ceiling, then soft-saturate the residual.
    scaled = audio.samples * (ceiling / peak)
    softened = ceiling * np.tanh(scaled / max(ceiling, 1e-9))
    log.debug("Soft-limited signal that was %.1f dB over the ceiling", overshoot_db)
    return audio.with_samples(softened)


def normalize_loudness(
    audio: Audio,
    target_lufs: float | None = -23.0,
    *,
    peak_ceiling_dbfs: float = -1.0,
    max_gain_db: float = 30.0,
) -> Audio:
    """Normalise to a loudness target, then guarantee the peak ceiling.

    ``max_gain_db`` stops a near-silent take from having its noise floor
    amplified into a hiss wall.
    """
    if audio.num_frames == 0:
        return audio

    result = audio
    if target_lufs is not None and audio.rms >= _SILENCE_FLOOR:
        current = measure_lufs(audio)
        if current is not None and np.isfinite(current):
            gain_db = float(target_lufs) - current
            if gain_db > max_gain_db:
                log.debug("Clamping loudness gain %.1f dB -> %.1f dB", gain_db, max_gain_db)
                gain_db = max_gain_db
            result = apply_gain(result, gain_db)
        else:
            log.debug("Loudness unmeasurable; leaving level untouched.")

    return peak_limit(result, peak_ceiling_dbfs)


def remove_dc_offset(audio: Audio) -> Audio:
    """Subtract the mean. A DC bias eats headroom and upsets the vocoder."""
    if audio.num_frames == 0:
        return audio
    axis = -1
    mean = np.mean(audio.samples, axis=axis, keepdims=True, dtype=np.float64)
    if float(np.max(np.abs(mean))) < 1e-5:
        return audio
    return audio.with_samples(audio.samples - mean.astype(np.float32))


def fade_edges(audio: Audio, fade_ms: int = 10) -> Audio:
    """Apply short raised-cosine fades so renders never start/end with a click."""
    n = int(round(fade_ms * audio.sample_rate / 1000.0))
    if n <= 1 or audio.num_frames < 2 * n:
        return audio

    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    samples = audio.samples.copy()
    samples[..., :n] *= ramp
    samples[..., -n:] *= ramp[::-1]
    return audio.with_samples(samples)


__all__ = [
    "amplitude_to_db",
    "apply_gain",
    "db_to_amplitude",
    "fade_edges",
    "measure_lufs",
    "normalize_loudness",
    "peak_dbfs",
    "peak_limit",
    "remove_dc_offset",
]
