"""The reference-audio preprocessing pipeline.

Order matters. Each stage assumes the previous ones have run:

    decode -> mono -> resample -> DC removal -> high/low-pass
      -> [vocal separation] -> [denoise] -> [window selection]
      -> [silence trim] -> loudness normalise -> peak ceiling

Window selection deliberately comes *after* denoising so the quality scores
that drive it reflect the audio the model will actually see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AudioConfig
from ..errors import ReferenceQualityError
from ..logging_utils import get_logger
from .analysis import QualityReport, analyze
from .denoise import butter_filter, denoise, separate_vocals
from .io import Audio, load_audio
from .loudness import normalize_loudness, remove_dc_offset
from .segment import select_reference_window
from .vad import trim_and_compact

log = get_logger("audio.preprocess")


@dataclass
class PreprocessResult:
    """Processed audio plus a full account of what happened to it."""

    audio: Audio
    report_before: QualityReport
    report_after: QualityReport
    steps: list[str] = field(default_factory=list)
    window_start: float = 0.0
    window_end: float = 0.0
    denoiser: str = "none"

    @property
    def improved_snr_db(self) -> float:
        return self.report_after.estimated_snr_db - self.report_before.estimated_snr_db

    def summary(self) -> str:
        lines = [
            "--- Reference audio ---",
            self.report_after.summary(),
            "",
            "Processing applied:",
        ]
        lines.extend(f"  {i + 1}. {step}" for i, step in enumerate(self.steps))
        if abs(self.improved_snr_db) >= 1.0:
            lines.append(f"\nSNR change: {self.improved_snr_db:+.1f} dB")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "denoiser": self.denoiser,
            "window": [round(self.window_start, 2), round(self.window_end, 2)],
            "before": self.report_before.to_dict(),
            "after": self.report_after.to_dict(),
        }


def preprocess_reference(
    source: str | Path | Audio | tuple[int, np.ndarray],
    config: AudioConfig,
    *,
    strict: bool = False,
    override: dict[str, Any] | None = None,
) -> PreprocessResult:
    """Turn an arbitrary recording into a clean voice prompt.

    Args:
        source: file path, ``Audio``, or Gradio ``(sample_rate, samples)``.
        config: the ``audio`` config section.
        strict: raise if the result scores below ``reference.min_quality_score``
            instead of warning.
        override: per-call overrides of ``preprocess`` fields, e.g.
            ``{"denoise": "none", "separate_vocals": True}``.
    """
    settings = config.preprocess
    overrides = override or {}

    def option(name: str) -> Any:
        """Value of a ``preprocess`` field, honouring per-call overrides."""
        return overrides.get(name, getattr(settings, name))

    def reference_option(name: str) -> Any:
        """Same, for ``reference`` fields (namespaced to avoid collisions)."""
        return overrides.get(f"reference.{name}", getattr(config.reference, name))

    steps: list[str] = []

    # -- decode ------------------------------------------------------------
    audio = load_audio(source, mono=bool(option("to_mono")))
    original_rate = audio.sample_rate
    steps.append(f"Decoded {audio.duration:.1f}s @ {original_rate} Hz")

    report_before = analyze(audio, min_seconds=config.reference.min_seconds)
    log.info(
        "Reference in: %.1fs, SNR ~%.0f dB, score %.0f (%s)",
        report_before.duration,
        report_before.estimated_snr_db,
        report_before.score,
        report_before.verdict,
    )

    # -- resample to the working rate --------------------------------------
    if audio.sample_rate != config.sample_rate:
        audio = audio.resample(config.sample_rate)
        steps.append(f"Resampled {original_rate} Hz -> {config.sample_rate} Hz")

    # -- DC offset ---------------------------------------------------------
    if option("remove_dc"):
        before = float(np.mean(audio.samples, dtype=np.float64))
        audio = remove_dc_offset(audio)
        if abs(before) > 1e-5:
            steps.append(f"Removed DC offset ({before:+.4f})")

    # -- band limiting -----------------------------------------------------
    highpass = option("highpass_hz")
    lowpass = option("lowpass_hz")
    if highpass or lowpass:
        audio = butter_filter(audio, highpass_hz=highpass, lowpass_hz=lowpass)
        described = []
        if highpass:
            described.append(f"high-pass {highpass:.0f} Hz")
        if lowpass:
            described.append(f"low-pass {lowpass:.0f} Hz")
        steps.append("Filtered: " + ", ".join(described))

    # -- vocal separation --------------------------------------------------
    if option("separate_vocals"):
        audio, ok = separate_vocals(audio)
        steps.append("Isolated vocals with Demucs" if ok else "Vocal separation unavailable (skipped)")

    # -- denoise -----------------------------------------------------------
    denoiser = "none"
    denoise_mode = str(option("denoise")).lower()
    if denoise_mode != "none":
        # Clean recordings get worse, not better, from aggressive processing.
        if denoise_mode == "auto" and report_before.estimated_snr_db >= 32.0:
            steps.append(f"Denoising skipped - already clean (~{report_before.estimated_snr_db:.0f} dB SNR)")
        else:
            audio, denoiser = denoise(
                audio,
                denoise_mode,
                strength=float(option("denoise_strength")),
                dereverb=bool(option("dereverb")),
            )
            steps.append(
                f"Denoised with {denoiser}" if denoiser != "none" else "No denoiser available (skipped)"
            )

    # -- reference window --------------------------------------------------
    window_start, window_end = 0.0, audio.duration
    if reference_option("auto_select_window") and audio.duration > config.reference.target_seconds * 1.15:
        choice = select_reference_window(
            audio,
            target_seconds=config.reference.target_seconds,
            min_seconds=config.reference.min_seconds,
            max_seconds=config.reference.max_seconds,
            vad_backend=str(option("vad_backend")),
        )
        audio = choice.audio
        window_start, window_end = choice.start, choice.end
        steps.append(f"Selected {choice.duration:.1f}s window at {choice.start:.1f}s - {choice.reason}")
    elif audio.duration > config.reference.max_seconds:
        audio = audio.slice_seconds(0.0, config.reference.max_seconds)
        window_end = config.reference.max_seconds
        steps.append(f"Truncated to {config.reference.max_seconds:.0f}s")

    # -- silence handling --------------------------------------------------
    if option("trim_silence"):
        trimmed = trim_and_compact(
            audio,
            backend=str(option("vad_backend")),
            threshold=float(option("vad_threshold")),
            pad_ms=int(option("vad_pad_ms")),
            max_internal_silence=float(option("max_internal_silence")),
        )
        if trimmed.removed_seconds > 0.05:
            audio = trimmed.audio
            steps.append(
                f"Trimmed {trimmed.removed_seconds:.2f}s of silence "
                f"({len(trimmed.segments)} speech regions kept)"
            )

    # -- level -------------------------------------------------------------
    target_lufs = option("loudness_lufs")
    audio = normalize_loudness(audio, target_lufs, peak_ceiling_dbfs=float(option("peak_dbfs")))
    if target_lufs is not None:
        steps.append(f"Normalised to {target_lufs:.0f} LUFS, peak ceiling {option('peak_dbfs'):.1f} dBFS")

    # -- verdict -----------------------------------------------------------
    report_after = analyze(audio, min_seconds=config.reference.min_seconds)
    result = PreprocessResult(
        audio=audio,
        report_before=report_before,
        report_after=report_after,
        steps=steps,
        window_start=window_start,
        window_end=window_end,
        denoiser=denoiser,
    )

    log.info(
        "Reference out: %.1fs, SNR ~%.0f dB, score %.0f (%s)",
        report_after.duration,
        report_after.estimated_snr_db,
        report_after.score,
        report_after.verdict,
    )

    minimum = config.reference.min_quality_score
    if report_after.score < minimum:
        message = (
            f"Reference quality is {report_after.verdict} (score {report_after.score:.0f}/100, "
            f"minimum {minimum:.0f})."
        )
        hint = " ".join(report_after.issues) or "Try a longer, cleaner recording."
        if strict:
            raise ReferenceQualityError(message, hint=hint)
        log.warning("%s %s", message, hint)

    if report_after.duration < config.reference.min_seconds:
        raise ReferenceQualityError(
            f"Only {report_after.duration:.1f}s of usable speech remains after processing "
            f"(minimum {config.reference.min_seconds:.0f}s).",
            hint="Provide a longer recording, or disable silence trimming if the clip is mostly speech.",
        )

    return result


def preprocess_source_audio(
    source: str | Path | Audio | tuple[int, np.ndarray],
    config: AudioConfig,
    *,
    denoise_mode: str | None = None,
) -> Audio:
    """Lighter pass for speech-to-speech *source* audio.

    The source supplies content and prosody, not timbre, so it is left as
    intact as possible: no window selection, no silence compaction, no
    loudness target that would distort dynamics.
    """
    audio = load_audio(source, mono=True, target_rate=config.sample_rate)
    audio = remove_dc_offset(audio)
    if config.preprocess.highpass_hz:
        audio = butter_filter(audio, highpass_hz=config.preprocess.highpass_hz)

    mode = denoise_mode if denoise_mode is not None else config.preprocess.denoise
    if str(mode).lower() not in {"none", "auto"}:
        audio, used = denoise(audio, str(mode), strength=config.preprocess.denoise_strength)
        log.info("Source audio denoised with %s", used)

    # Peak-limit only - preserve the original dynamics.
    return normalize_loudness(audio, None, peak_ceiling_dbfs=config.preprocess.peak_dbfs)


__all__ = ["PreprocessResult", "preprocess_reference", "preprocess_source_audio"]
