"""Objective quality analysis of a reference recording.

Cloning quality is bounded by the reference. Rather than let a noisy, clipped
or band-limited prompt quietly ruin the output, we measure it, score it, and
tell the user exactly what is wrong and how to fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..logging_utils import get_logger
from .io import Audio
from .loudness import amplitude_to_db, measure_lufs, peak_dbfs

log = get_logger("audio.analysis")

#: Speech energy lives here; used for the band-limiting check.
_SPEECH_BAND = (80.0, 8000.0)


@dataclass
class QualityReport:
    """Measured properties of a recording plus a 0-100 usability score."""

    duration: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    lufs: float | None
    dc_offset: float
    clipping_ratio: float
    estimated_snr_db: float
    effective_bandwidth_hz: float
    speech_ratio: float
    longest_silence: float
    score: float = 0.0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.score >= 80:
            return "excellent"
        if self.score >= 60:
            return "good"
        if self.score >= 40:
            return "usable"
        if self.score >= 25:
            return "poor"
        return "unusable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration, 2),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "peak_dbfs": round(self.peak_dbfs, 2),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "lufs": round(self.lufs, 2) if self.lufs is not None else None,
            "dc_offset": round(self.dc_offset, 5),
            "clipping_pct": round(self.clipping_ratio * 100, 3),
            "estimated_snr_db": round(self.estimated_snr_db, 1),
            "effective_bandwidth_hz": int(self.effective_bandwidth_hz),
            "speech_ratio": round(self.speech_ratio, 3),
            "longest_silence_s": round(self.longest_silence, 2),
            "score": round(self.score, 1),
            "verdict": self.verdict,
            "issues": self.issues,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        """Compact multi-line summary for the UI."""
        lines = [
            f"Verdict: {self.verdict.upper()}  (score {self.score:.0f}/100)",
            f"Duration {self.duration:.1f}s | {self.sample_rate} Hz | {self.channels}ch",
            f"Peak {self.peak_dbfs:.1f} dBFS | RMS {self.rms_dbfs:.1f} dBFS"
            + (f" | {self.lufs:.1f} LUFS" if self.lufs is not None else ""),
            f"Estimated SNR {self.estimated_snr_db:.0f} dB | "
            f"bandwidth ~{self.effective_bandwidth_hz / 1000:.1f} kHz | "
            f"speech {self.speech_ratio * 100:.0f}%",
        ]
        for issue in self.issues:
            lines.append(f"  [!] {issue}")
        for warning in self.warnings:
            lines.append(f"  [~] {warning}")
        return "\n".join(lines)


def _frame_energies(samples: np.ndarray, sample_rate: int, frame_ms: float = 25.0) -> np.ndarray:
    """RMS energy per frame, in dB."""
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, frame // 2)
    if samples.size < frame:
        return np.array([amplitude_to_db(float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0)])

    n_frames = 1 + (samples.size - frame) // hop
    strided = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(np.square(strided.astype(np.float64)), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def estimate_snr(samples: np.ndarray, sample_rate: int) -> tuple[float, float, np.ndarray]:
    """Estimate SNR from the spread of frame energies.

    Speech recordings are bimodal: quiet frames are noise floor, loud frames
    are speech. The gap between the 10th and 95th percentile approximates SNR
    without needing a VAD.

    Returns ``(snr_db, speech_ratio, frame_energies_db)``.
    """
    energies = _frame_energies(samples, sample_rate)
    if energies.size == 0:
        return 0.0, 0.0, energies

    noise_floor = float(np.percentile(energies, 10))
    speech_level = float(np.percentile(energies, 95))
    snr = max(0.0, speech_level - noise_floor)

    # Frames within 20 dB of the speech level count as speech.
    threshold = speech_level - 20.0
    speech_ratio = float(np.mean(energies > threshold))
    return snr, speech_ratio, energies


def longest_silence_seconds(energies_db: np.ndarray, sample_rate: int, frame_ms: float = 25.0) -> float:
    """Longest consecutive run of low-energy frames, in seconds."""
    if energies_db.size == 0:
        return 0.0
    threshold = float(np.percentile(energies_db, 95)) - 30.0
    quiet = energies_db < threshold
    if not quiet.any():
        return 0.0

    best = current = 0
    for is_quiet in quiet:
        current = current + 1 if is_quiet else 0
        best = max(best, current)
    hop_s = (frame_ms / 2.0) / 1000.0
    return best * hop_s


def effective_bandwidth(samples: np.ndarray, sample_rate: int, drop_db: float = 40.0) -> float:
    """Highest frequency still carrying meaningful energy.

    Catches telephone-band, aggressively compressed or upsampled sources -
    all of which cap achievable clone fidelity.
    """
    if samples.size < 1024:
        return sample_rate / 2.0

    n_fft = 2048
    window = np.hanning(n_fft).astype(np.float32)
    hop = n_fft // 2
    n_frames = max(1, min(200, 1 + (samples.size - n_fft) // hop))
    if samples.size < n_fft:
        padded = np.pad(samples, (0, n_fft - samples.size))
        spectrum = np.abs(np.fft.rfft(padded * window))
    else:
        # Average the magnitude spectrum over frames spread across the file.
        starts = np.linspace(0, samples.size - n_fft, n_frames).astype(int)
        accum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        for start in starts:
            accum += np.abs(np.fft.rfft(samples[start : start + n_fft] * window))
        spectrum = accum / len(starts)

    power_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # Reference level from the speech band, not from DC rumble.
    band = (freqs >= _SPEECH_BAND[0]) & (freqs <= min(_SPEECH_BAND[1], sample_rate / 2.0))
    if not band.any():
        return sample_rate / 2.0
    reference = float(np.max(power_db[band]))

    above = np.where(power_db >= reference - drop_db)[0]
    return float(freqs[above[-1]]) if above.size else sample_rate / 2.0


def analyze(audio: Audio, *, min_seconds: float = 3.0) -> QualityReport:
    """Measure a recording and score its fitness as a cloning reference."""
    mono = audio.to_mono()
    samples = mono.samples

    peak = peak_dbfs(mono)
    rms = amplitude_to_db(mono.rms)
    dc = float(np.mean(samples, dtype=np.float64)) if samples.size else 0.0
    clipping = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
    snr, speech_ratio, energies = estimate_snr(samples, mono.sample_rate)
    bandwidth = effective_bandwidth(samples, mono.sample_rate)
    silence = longest_silence_seconds(energies, mono.sample_rate)

    report = QualityReport(
        duration=mono.duration,
        sample_rate=mono.sample_rate,
        channels=audio.channels,
        peak_dbfs=peak,
        rms_dbfs=rms,
        lufs=measure_lufs(mono),
        dc_offset=dc,
        clipping_ratio=clipping,
        estimated_snr_db=snr,
        effective_bandwidth_hz=bandwidth,
        speech_ratio=speech_ratio,
        longest_silence=silence,
    )

    _score(report, min_seconds=min_seconds)
    return report


def _score(report: QualityReport, *, min_seconds: float) -> None:
    """Populate ``score``/``issues``/``warnings`` in place.

    Starts at 100 and deducts for each defect, weighted by how much that
    defect actually degrades a zero-shot clone.
    """
    score = 100.0

    # -- duration ----------------------------------------------------------
    if report.duration < min_seconds:
        penalty = min(45.0, (min_seconds - report.duration) / max(min_seconds, 1e-6) * 60.0)
        score -= penalty
        report.issues.append(
            f"Only {report.duration:.1f}s of audio - at least {min_seconds:.0f}s is needed "
            "for a stable voice match."
        )
    elif report.duration < 5.0:
        score -= 8.0
        report.warnings.append("Short reference; 8-12s of continuous speech clones more reliably.")

    # -- signal-to-noise ---------------------------------------------------
    if report.estimated_snr_db < 12.0:
        score -= 30.0
        report.issues.append(
            f"Very noisy (~{report.estimated_snr_db:.0f} dB SNR). Enable denoising, or "
            "re-record somewhere quieter."
        )
    elif report.estimated_snr_db < 20.0:
        score -= 15.0
        report.warnings.append(f"Moderate background noise (~{report.estimated_snr_db:.0f} dB SNR).")
    elif report.estimated_snr_db < 30.0:
        score -= 5.0

    # -- clipping ----------------------------------------------------------
    if report.clipping_ratio > 0.005:
        score -= 25.0
        report.issues.append(
            f"{report.clipping_ratio * 100:.1f}% of samples are clipped - the distortion "
            "will be copied into the clone."
        )
    elif report.clipping_ratio > 0.0005:
        score -= 10.0
        report.warnings.append("Slight clipping detected; lower the input gain when recording.")

    # -- bandwidth ---------------------------------------------------------
    if report.effective_bandwidth_hz < 5000.0:
        score -= 25.0
        report.issues.append(
            f"Band-limited to ~{report.effective_bandwidth_hz / 1000:.1f} kHz (telephone quality). "
            "The clone will sound muffled."
        )
    elif report.effective_bandwidth_hz < 8000.0:
        score -= 12.0
        report.warnings.append(
            f"Limited bandwidth (~{report.effective_bandwidth_hz / 1000:.1f} kHz); "
            "high-frequency detail is missing."
        )

    # -- level -------------------------------------------------------------
    if report.rms_dbfs < -45.0:
        score -= 15.0
        report.issues.append("Recording is extremely quiet; normalisation will amplify its noise floor.")
    elif report.rms_dbfs < -35.0:
        score -= 5.0
        report.warnings.append("Low recording level.")

    if report.peak_dbfs > -0.5 and report.clipping_ratio <= 0.0005:
        score -= 3.0
        report.warnings.append("Peaks are right at 0 dBFS; almost no headroom.")

    # -- content -----------------------------------------------------------
    if report.speech_ratio < 0.25:
        score -= 15.0
        report.issues.append(
            f"Only {report.speech_ratio * 100:.0f}% of the file looks like speech - "
            "mostly silence or noise."
        )
    if report.longest_silence > 2.0:
        score -= 5.0
        report.warnings.append(
            f"Contains a {report.longest_silence:.1f}s pause; silence trimming is recommended."
        )

    if abs(report.dc_offset) > 0.01:
        score -= 5.0
        report.warnings.append("Noticeable DC offset; it will be removed automatically.")

    report.score = float(max(0.0, min(100.0, score)))


__all__ = ["QualityReport", "analyze", "effective_bandwidth", "estimate_snr"]
