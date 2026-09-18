"""Result objects returned by the synthesis pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audio.io import Audio
from ..audio.preprocess import PreprocessResult
from ..text.pipeline import PreparedText

#: Where this encoder sits on *this* corpus. Measured by `scripts/benchmark.py`
#: on 45 clips from 9 speakers, every clip through the same peak-normalising
#: path in `evaluation/embedding.py`, ECAPA-TDNN on CPU:
#:
#:     different speakers                     0.132 mean, 0.508 highest (900 pairs)
#:     same speaker, two different clips      0.689 mean                 (90 pairs)
#:
#: Three things this is not.
#:
#: It is not a general ECAPA calibration. Nine speakers of Persian read speech,
#: recorded by crowdsourcing, is what it describes.
#:
#: It is not an unselected sample. `scripts/build_eval_corpus.py` drops a
#: speaker whose own clips score worse against each other than against someone
#: else's - using this same encoder. That keeps the ground truth usable and it
#: also pushes this ceiling up and this floor down relative to a random sample.
#: Comparing engines on this set is still well-defined, since every engine
#: meets the same speakers, but the result is conditional on the set: a
#: selection made by this encoder can favour whichever engine best preserves
#: what this encoder listens to, so neither a ranking nor its generalisation to
#: other speakers is protected from the bias. Absolute statements about "the
#: human ceiling" are affected directly.
#:
#: And it is not a session calibration. An earlier version of this block put
#: same-speaker pairs into "same session 0.800" and "different session 0.647"
#: bands. Common Voice publishes no session field, so for those speakers the
#: evidenced statement is only "a different clip", and the bands are gone.
#:
#: What is measured directly is a gap: across five engines a render scores 0.11
#: to 0.16 higher against the clip it was conditioned on than against other
#: recordings of the same speaker, so scoring a clone against its own prompt
#: flatters it by about that much. *Why* is not established. Shared recording
#: channel is one explanation; the render continuing the prompt's prosody,
#: level or content is another, and neither has been separated from the other,
#: nor shown to be the same size for every engine.
SIMILARITY_DIFFERENT_SPEAKER = 0.132
SIMILARITY_DIFFERENT_SPEAKER_MAX = 0.508
SIMILARITY_SAME_SPEAKER = 0.689

#: The lowest per-engine mean of render-against-its-own-prompt measured here
#: (Fish S2 Pro, 0.651; the highest is OmniVoice at 0.741). Used only to tell
#: "this render sits where working engines sit" from "something came apart".
REFERENCE_SIMILARITY_TYPICAL = 0.65


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
    #: Word error rate of the round trip. Persian compounds have no fixed
    #: spacing, so this partly scores where the recogniser put its spaces:
    #: a flawless render measured 55.6% here purely because Whisper wrote
    #: `کتاب خانه` for `کتابخانه`. Kept for continuity; read `character_error_rate`.
    word_error_rate: float | None = None
    #: Character error rate, whitespace-insensitive. The intelligibility
    #: number to trust for Persian.
    character_error_rate: float | None = None

    @property
    def reference_verdict(self) -> str:
        """Deliberately not phrased as identity - this comparison cannot show it."""
        value = self.reference_similarity
        if value is None:
            return "not measured"
        if value >= REFERENCE_SIMILARITY_TYPICAL:
            return "as close to the prompt as a working engine gets here"
        if value >= 0.50:
            return "moderate"
        return "weak - the render may have come apart"

    @property
    def held_out_verdict(self) -> str:
        """The identity question, on a scale that was actually measured."""
        value = self.held_out_similarity
        if value is None:
            return "not measured"
        if value >= SIMILARITY_SAME_SPEAKER:
            return "as close as another recording of the same speaker on this corpus"
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
        if self.character_error_rate is not None:
            lines.append(f"Round-trip CER: {self.character_error_rate * 100:.1f}%")
        if self.word_error_rate is not None:
            lines.append(
                f"Round-trip WER: {self.word_error_rate * 100:.1f}% "
                "(counts the recogniser's word spacing too)"
            )
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
