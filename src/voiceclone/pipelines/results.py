"""Result objects returned by the synthesis pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audio.io import Audio
from ..audio.preprocess import PreprocessResult
from ..text.pipeline import PreparedText


@dataclass
class QualityScores:
    """Objective measurements of a render."""

    speaker_similarity: float | None = None
    transcription: str | None = None
    word_error_rate: float | None = None

    @property
    def similarity_verdict(self) -> str:
        if self.speaker_similarity is None:
            return "not measured"
        value = self.speaker_similarity
        if value >= 0.80:
            return "excellent match"
        if value >= 0.70:
            return "strong match"
        if value >= 0.60:
            return "fair match"
        if value >= 0.45:
            return "weak match"
        return "poor match"

    def summary(self) -> str:
        lines: list[str] = []
        if self.speaker_similarity is not None:
            lines.append(
                f"Speaker similarity: {self.speaker_similarity:.3f} ({self.similarity_verdict})"
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
            "speaker_similarity": self.scores.speaker_similarity,
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
