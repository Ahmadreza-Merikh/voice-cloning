"""Noise reduction, filtering and source separation.

Backends, best first:

* ``deepfilternet`` - DeepFilterNet3, a real-time speech enhancer that also
  suppresses reverb. Best perceptual quality, optional dependency.
* ``noisereduce``   - spectral gating. Portable, decent, can sound "phasey"
  if pushed hard, so the mix is capped.
* ``none``          - passthrough.

Demucs vocal separation is exposed separately: it is for pulling a voice out
of music, which is a different job from denoising.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..logging_utils import get_logger
from .io import Audio, resample_array

log = get_logger("audio.denoise")

#: DeepFilterNet operates at 48 kHz.
_DFN_RATE = 48000


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
def butter_filter(audio: Audio, *, highpass_hz: float | None = None, lowpass_hz: float | None = None, order: int = 4) -> Audio:
    """Zero-phase Butterworth high/low/band-pass.

    A high-pass around 50-70 Hz removes rumble, HVAC and handling noise that
    sit below the human F0 range and otherwise waste headroom.
    """
    if audio.num_frames == 0 or (not highpass_hz and not lowpass_hz):
        return audio

    try:
        from scipy.signal import butter, sosfiltfilt  # type: ignore
    except Exception:
        log.debug("scipy unavailable; skipping filtering", exc_info=True)
        return audio

    nyquist = audio.sample_rate / 2.0
    low = float(highpass_hz) / nyquist if highpass_hz else None
    high = float(lowpass_hz) / nyquist if lowpass_hz else None

    # Guard against invalid normalised frequencies.
    if low is not None and not (0.0 < low < 1.0):
        low = None
    if high is not None and not (0.0 < high < 1.0):
        high = None
    if low is None and high is None:
        return audio

    if low is not None and high is not None:
        sos = butter(order, [low, high], btype="bandpass", output="sos")
    elif low is not None:
        sos = butter(order, low, btype="highpass", output="sos")
    else:
        sos = butter(order, high, btype="lowpass", output="sos")

    # filtfilt needs a few times the filter length; skip very short clips.
    if audio.num_frames < 3 * (order * 2 + 1):
        return audio

    try:
        filtered = sosfiltfilt(sos, audio.samples.astype(np.float64), axis=-1)
    except Exception:
        log.debug("Filtering failed; returning input", exc_info=True)
        return audio
    return audio.with_samples(filtered.astype(np.float32))


# --------------------------------------------------------------------------
# DeepFilterNet
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_deepfilternet() -> tuple[object, object, object] | None:
    try:
        from df.enhance import init_df  # type: ignore

        model, state, _ = init_df(log_level="ERROR")
        log.info("DeepFilterNet ready (48 kHz enhancement).")
        return model, state, None
    except Exception:
        log.debug("DeepFilterNet unavailable", exc_info=True)
        return None


def _denoise_deepfilternet(audio: Audio, *, dereverb: bool = False) -> Audio | None:
    bundle = _load_deepfilternet()
    if bundle is None:
        return None

    model, state, _ = bundle
    try:
        import torch
        from df.enhance import enhance  # type: ignore

        mono = audio.to_mono()
        work = mono.resample(_DFN_RATE)
        tensor = torch.from_numpy(np.ascontiguousarray(work.samples)).unsqueeze(0)

        # atten_lim_db caps how much the noise is attenuated; unlimited
        # suppression sounds unnatural and can gate quiet consonants.
        enhanced = enhance(model, state, tensor, atten_lim_db=None if dereverb else 30.0)

        out = enhanced.squeeze(0).detach().cpu().numpy().astype(np.float32)
        restored = resample_array(out, _DFN_RATE, audio.sample_rate)
        return audio.with_samples(restored)
    except Exception:
        log.warning("DeepFilterNet enhancement failed; falling back.", exc_info=True)
        return None


# --------------------------------------------------------------------------
# noisereduce
# --------------------------------------------------------------------------
def _denoise_noisereduce(audio: Audio, strength: float = 0.8) -> Audio | None:
    try:
        import noisereduce as nr  # type: ignore
    except Exception:
        log.debug("noisereduce unavailable", exc_info=True)
        return None

    try:
        mono = audio.to_mono()
        # prop_decrease above ~0.85 starts to sound processed on speech.
        prop = float(np.clip(strength, 0.0, 1.0)) * 0.85
        reduced = nr.reduce_noise(
            y=mono.samples.astype(np.float32),
            sr=mono.sample_rate,
            stationary=False,
            prop_decrease=prop,
            n_fft=1024,
            time_mask_smooth_ms=64,
            freq_mask_smooth_hz=500,
        )
        return audio.with_samples(np.asarray(reduced, dtype=np.float32))
    except Exception:
        log.warning("noisereduce failed; leaving audio unprocessed.", exc_info=True)
        return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def denoise(
    audio: Audio,
    backend: str = "auto",
    *,
    strength: float = 0.8,
    dereverb: bool = False,
) -> tuple[Audio, str]:
    """Reduce background noise.

    Returns ``(audio, backend_actually_used)`` so callers can report what ran.
    """
    backend = (backend or "auto").lower()
    if backend == "none" or audio.num_frames == 0:
        return audio, "none"

    if backend in {"auto", "deepfilternet", "dfn"}:
        result = _denoise_deepfilternet(audio, dereverb=dereverb)
        if result is not None:
            return result, "deepfilternet"
        if backend != "auto":
            log.warning("DeepFilterNet requested but unavailable; trying noisereduce.")

    if backend in {"auto", "noisereduce", "spectral"}:
        result = _denoise_noisereduce(audio, strength)
        if result is not None:
            return result, "noisereduce"

    log.info("No denoiser available; audio left as-is.")
    return audio, "none"


def separate_vocals(audio: Audio, *, model_name: str = "htdemucs") -> tuple[Audio, bool]:
    """Isolate the vocal stem with Demucs.

    Use this when the reference comes from a song, a film or anything with
    music underneath. Returns ``(audio, succeeded)``.
    """
    try:
        import torch
        from demucs.apply import apply_model  # type: ignore
        from demucs.pretrained import get_model  # type: ignore
    except Exception:
        log.warning("Demucs is not installed; skipping vocal separation.")
        return audio, False

    try:
        model = get_model(model_name)
        model.eval()
        model_rate = int(getattr(model, "samplerate", 44100))

        work = audio.resample(model_rate)
        samples = work.samples
        if samples.ndim == 1:  # Demucs expects stereo
            samples = np.stack([samples, samples])

        tensor = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
        with torch.no_grad():
            stems = apply_model(model, tensor, device="cpu", progress=False)[0]

        sources = list(getattr(model, "sources", []))
        if "vocals" not in sources:
            log.warning("Demucs model %s has no vocal stem.", model_name)
            return audio, False

        vocals = stems[sources.index("vocals")].mean(dim=0).cpu().numpy().astype(np.float32)
        restored = resample_array(vocals, model_rate, audio.sample_rate)
        log.info("Isolated vocal stem with Demucs (%s).", model_name)
        return audio.with_samples(restored), True
    except Exception:
        log.warning("Vocal separation failed; using the original audio.", exc_info=True)
        return audio, False


def available_backends() -> dict[str, bool]:
    """Which enhancement backends this installation can actually use."""
    import importlib.util as util

    return {
        "deepfilternet": util.find_spec("df") is not None,
        "noisereduce": util.find_spec("noisereduce") is not None,
        "demucs": util.find_spec("demucs") is not None,
        "scipy": util.find_spec("scipy") is not None,
    }


__all__ = ["available_backends", "butter_filter", "denoise", "separate_vocals"]
