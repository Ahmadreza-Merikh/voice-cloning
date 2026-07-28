"""Reference-window selection.

Given a long recording, which N seconds should condition the clone? Taking
the first N is a common and costly mistake - openings are often noisy, quiet,
or start mid-breath. This module scores every candidate window and picks the
one that will actually clone well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_utils import get_logger
from .analysis import estimate_snr
from .io import Audio
from .vad import Segment, detect_speech

log = get_logger("audio.segment")


@dataclass
class WindowChoice:
    """The selected reference window and why it won."""

    audio: Audio
    start: float
    end: float
    score: float
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _score_window(samples: np.ndarray, sample_rate: int) -> tuple[float, dict[str, float]]:
    """Score a candidate window on the properties that drive clone quality."""
    if samples.size == 0:
        return 0.0, {}

    snr, speech_ratio, energies = estimate_snr(samples, sample_rate)
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    clipping = float(np.mean(np.abs(samples) >= 0.999))

    # Consistent level across the window means one speaker at one distance,
    # rather than a splice or a fade.
    level_stability = float(np.std(energies)) if energies.size > 1 else 0.0

    score = 0.0
    # Clean signal matters most.
    score += min(snr, 40.0) / 40.0 * 40.0
    # Dense speech: we want voice, not pauses.
    score += speech_ratio * 30.0
    # Healthy level, centred on -25 dBFS.
    score += max(0.0, 15.0 - abs(rms_db + 25.0) * 0.75)
    # Even delivery.
    score += max(0.0, 15.0 - level_stability * 0.8)
    # Distortion is disqualifying.
    score -= min(clipping * 4000.0, 40.0)

    metrics = {
        "snr": snr,
        "speech_ratio": speech_ratio,
        "rms_db": rms_db,
        "clipping": clipping,
        "level_stability": level_stability,
    }
    return float(max(0.0, score)), metrics


def _merge_close(segments: list[Segment], max_gap: float = 0.35) -> list[Segment]:
    """Merge speech regions separated by short pauses into continuous runs."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]


def select_reference_window(
    audio: Audio,
    *,
    target_seconds: float = 10.0,
    min_seconds: float = 3.0,
    max_seconds: float = 20.0,
    vad_backend: str = "auto",
    hop_seconds: float = 0.5,
) -> WindowChoice:
    """Pick the best window of ``target_seconds`` to use as the voice prompt.

    Strategy:
      1. Find continuous speech runs (short pauses merged).
      2. Prefer a run that is already close to the target length - starting
         and ending on natural phrase boundaries sounds best.
      3. Otherwise slide a window across the speech and score each position.
    """
    duration = audio.duration
    target = float(np.clip(target_seconds, min_seconds, max_seconds))

    if duration <= target * 1.15:
        # Already about the right size - use everything.
        score, _ = _score_window(audio.to_mono().samples, audio.sample_rate)
        return WindowChoice(audio, 0.0, duration, score, "clip already at target length")

    segments = _merge_close(detect_speech(audio, backend=vad_backend))
    mono = audio.to_mono()

    # -- 1. a single speech run of about the right length ------------------
    best: WindowChoice | None = None
    for start, end in segments:
        run = end - start
        if run < min_seconds:
            continue
        take_end = min(end, start + max_seconds)
        candidate = mono.slice_seconds(start, take_end)
        score, metrics = _score_window(candidate.samples, candidate.sample_rate)
        # Reward runs near the target length; penalise very short ones.
        length_fit = 1.0 - min(abs(candidate.duration - target) / target, 1.0)
        total = score + length_fit * 12.0
        if best is None or total > best.score:
            best = WindowChoice(
                candidate,
                start,
                take_end,
                total,
                f"continuous speech run ({candidate.duration:.1f}s, SNR ~{metrics.get('snr', 0):.0f} dB)",
            )

    # -- 2. sliding window over the whole recording ------------------------
    window_samples = int(round(target * audio.sample_rate))
    hop_samples = max(1, int(round(hop_seconds * audio.sample_rate)))
    total_samples = mono.num_frames

    if total_samples > window_samples:
        for start_idx in range(0, total_samples - window_samples + 1, hop_samples):
            chunk = mono.samples[start_idx : start_idx + window_samples]
            score, metrics = _score_window(chunk, audio.sample_rate)
            if best is None or score > best.score:
                start_s = start_idx / audio.sample_rate
                best = WindowChoice(
                    mono.with_samples(chunk),
                    start_s,
                    start_s + target,
                    score,
                    f"best {target:.0f}s window at {start_s:.1f}s (SNR ~{metrics.get('snr', 0):.0f} dB)",
                )

    if best is None:
        fallback = mono.slice_seconds(0.0, target)
        return WindowChoice(fallback, 0.0, target, 0.0, "fallback: first window")

    log.info(
        "Reference window: %.1fs-%.1fs (%.1fs) - %s",
        best.start,
        best.end,
        best.duration,
        best.reason,
    )
    return best


def fit_to_duration(
    audio: Audio, *, target_seconds: float, min_seconds: float = 3.0, vad_backend: str = "auto"
) -> Audio:
    """Convenience wrapper returning just the trimmed audio."""
    return select_reference_window(
        audio,
        target_seconds=target_seconds,
        min_seconds=min_seconds,
        vad_backend=vad_backend,
    ).audio


__all__ = ["WindowChoice", "fit_to_duration", "select_reference_window"]
