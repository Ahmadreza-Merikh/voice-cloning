"""Result objects returned by the synthesis pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audio.io import Audio
from ..audio.preprocess import PreprocessResult
from ..text.pipeline import PreparedText

#: Where the speaker encoder actually sits, measured on this project's own
#: audio and pipeline (ECAPA-TDNN, clips through `preprocess_reference`):
#:
#:     different speaker, same gender, same corpus   0.288   (39 pairs, max 0.544)
#:     same speaker, *different* recording session   0.647   (6 pairs)
#:     same speaker, *same* recording session        0.800   (17 pairs)
#:
#: The last two are the important pair. A single speaker scores 0.800 against
#: themselves within one session and 0.647 across sessions; the gap is shared
#: channel - room, microphone, noise floor - not identity. A zero-shot clone
#: reproduces its prompt's channel, so scoring it against that prompt lands it
#: in the same-session band and flatters it by roughly 0.26.
SIMILARITY_DIFFERENT_SPEAKER = 0.288
SIMILARITY_DIFFERENT_SPEAKER_MAX = 0.544
SIMILARITY_SAME_SPEAKER_CROSS_SESSION = 0.647
SIMILARITY_SAME_SPEAKER_SAME_SESSION = 0.800


@dataclass
class QualityScores:
    """Objective measurements of a render."""

    #: Cosine similarity between the render and *the clip it was conditioned
    #: on*. Shares that clip's channel, so it is not a measure of identity
    #: transfer - see `held_out_similarity` for that. Useful for spotting a
    #: collapsed render, and valid for comparing engines or settings on one
    #: fixed reference, where the channel term is constant.
    reference_similarity: float | None = None
    #: Cosine similarity against recordings of the same speaker that the
    #: render was *not* conditioned on. This is the number that describes
    #: voice cloning. Only available when held-out audio was supplied.
    held_out_similarity: float | None = None
    transcription: str | None = None
    word_error_rate: float | None = None

    @property
    def reference_verdict(self) -> str:
        """Deliberately not phrased as identity - this comparison cannot show it."""
        value = self.reference_similarity
        if value is None:
            return "not measured"
        if value >= SIMILARITY_SAME_SPEAKER_SAME_SESSION:
            return "as close to the prompt as two takes from one session"
        if value >= 0.65:
            return "close to the prompt"
        if value >= 0.50:
            return "moderate"
        return "weak - the render may have come apart"

    @property
    def held_out_verdict(self) -> str:
        """The identity question, on a scale that was actually measured."""
        value = self.held_out_similarity
        if value is None:
            return "not measured"
        if value >= SIMILARITY_SAME_SPEAKER_CROSS_SESSION:
            return "as close as another recording of the same speaker"
        if value >= SIMILARITY_DIFFERENT_SPEAKER_MAX:
            return "above every different-speaker pair measured, below same-speaker"
        if value >= SIMILARITY_DIFFERENT_SPEAKER:
            return "inside the range different speakers occupy - not distinguishable"
        return "no better than an unrelated speaker"

    def summary(self) -> str:
        lines: list[str] = []
        if self.held_out_similarity is not None:
            lines.append(
                f"Speaker similarity (held-out): {self.held_out_similarity:.3f} "
                f"({self.held_out_verdict})"
            )
        if self.reference_similarity is not None:
            label = "Similarity to the reference clip"
            if self.held_out_similarity is None:
                label += " (shares its channel; not identity transfer)"
            lines.append(
                f"{label}: {self.reference_similarity:.3f} ({self.reference_verdict})"
            )
        if self.word_error_rate is not None:
            lines.append(f"Round-trip WER: {self.word_error_rate * 100:.1f}%")
        if self.transcription:
            lines.append(f"Heard back: {self.transcription}")
        return "\n".join(lines) or "No objective metrics were computed."


@dataclass
class CloneResult:
    """Everything produced by one synthesis run."""

    audio: Audio
    engine: str
    mode: str  # "tts" | "s2s-cascade" | "s2s-vc"
    seed: int | None = None
    output_path: Path | None = None
    reference: PreprocessResult | None = None
    text: PreparedText | None = None
    source_transcript: str | None = None
    reference_transcript: str | None = None
    scores: QualityScores = field(default_factory=QualityScores)
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.audio.duration

    @property
    def total_seconds(self) -> float:
        return float(sum(self.timings.values()))

    @property
    def realtime_factor(self) -> float | None:
        """Seconds of compute per second of audio. Lower is faster."""
        if self.duration <= 0:
            return None
        return self.total_seconds / self.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "mode": self.mode,
            "seed": self.seed,
            "duration_s": round(self.duration, 2),
            "output_path": str(self.output_path) if self.output_path else None,
            "timings": {key: round(value, 2) for key, value in self.timings.items()},
            "realtime_factor": round(self.realtime_factor, 2) if self.realtime_factor else None,
            "reference_similarity": self.scores.reference_similarity,
            "held_out_similarity": self.scores.held_out_similarity,
            "text": self.text.to_dict() if self.text else None,
            "reference": self.reference.to_dict() if self.reference else None,
            "warnings": self.warnings,
        }

    def report(self) -> str:
        """Human-readable summary rendered next to the audio in the UI."""
        lines = [
            f"Engine: {self.engine}   Mode: {self.mode}"
            + (f"   Seed: {self.seed}" if self.seed is not None else ""),
            f"Output: {self.duration:.1f}s"
            + (
                f"   Generated in {self.total_seconds:.1f}s "
                f"({self.realtime_factor:.1f}x realtime)"
                if self.realtime_factor
                else ""
            ),
        ]

        if self.text is not None:
            lines += ["", self.text.summary()]

        if self.reference is not None:
            lines += ["", self.reference.summary()]

        if self.reference_transcript:
            lines += ["", f"Reference transcript: {self.reference_transcript}"]
        if self.source_transcript:
            lines += ["", f"Source transcript: {self.source_transcript}"]

        scores = self.scores.summary()
        if scores and "No objective" not in scores:
            lines += ["", "--- Quality ---", scores]

        if self.timings:
            breakdown = "  ".join(f"{key} {value:.1f}s" for key, value in self.timings.items())
            lines += ["", f"Timing: {breakdown}"]

        if self.warnings:
            lines += ["", "Warnings:"]
            lines += [f"  [!] {warning}" for warning in self.warnings]

        return "\n".join(lines)


__all__ = ["CloneResult", "QualityScores"]
