"""An engine's reference limit is applied before the reference is transcribed.

F5-TTS clips a prompt longer than 12 s inside its own preprocessing but keeps
the transcript it was given. For a 12.4 s reference that turned a four-word
greeting into 1.1 s of garble, because the transcript described speech the
clipped audio no longer contained. The Studio now cuts the window first and
transcribes the cut, so the two cannot disagree.

Nothing here loads a model: a recording engine stands in for the TTS engine and
the recogniser is replaced, so the tests see exactly what an engine is handed.
"""

from __future__ import annotations

from typing import Any

import pytest

from voiceclone.audio import analyze
from voiceclone.audio.io import Audio
from voiceclone.audio.preprocess import PreprocessResult
from voiceclone.audio.segment import fit_reference
from voiceclone.audio.vad import detect_speech
from voiceclone.config import AppConfig
from voiceclone.engines import list_tts_engines
from voiceclone.engines.base import EngineCapabilities, SynthesisResult
from voiceclone.pipelines.studio import Studio

TRANSCRIPT_OF_THE_CUT = "رونوشت همین کلیپ"


class RecordingEngine:
    """Stands in for a TTS engine and records what it was conditioned on."""

    display_name = "Recording engine"

    def __init__(self, limit: float | None, output: Audio) -> None:
        self.capabilities = EngineCapabilities(requires_reference_text=True, max_reference_seconds=limit)
        self.output = output
        self.calls: list[tuple[float, str | None]] = []

    def load(self) -> None:
        pass

    def synthesize(self, text: str, reference: Audio, *, reference_text: str | None = None,
                   speed: float = 1.0, seed: int | None = None, **kwargs: Any) -> SynthesisResult:
        self.calls.append((reference.duration, reference_text))
        return SynthesisResult(audio=self.output, engine="recording", seed=seed or 0, metadata={})


@pytest.fixture
def quiet_config(config: AppConfig) -> AppConfig:
    # No speaker model or recogniser is loaded for scoring: these tests are about
    # what the engine is handed, not about how its output scores.
    return config.copy_with(**{"evaluation.enabled": False, "evaluation.candidates": 1})


def studio_with(config: AppConfig, engine: RecordingEngine, reference: Audio,
                transcribed: list[float]) -> Studio:
    studio = Studio(config)
    report = analyze(reference)
    studio.tts_engine = lambda name=None: engine
    studio.prepare_reference = lambda source, **_: PreprocessResult(
        audio=reference, report_before=report, report_after=report
    )

    def transcribe(audio: Audio, *, language: str | None = None) -> str:
        transcribed.append(audio.duration)
        return TRANSCRIPT_OF_THE_CUT

    studio.transcribe_reference = transcribe
    return studio


class TestFitReference:
    def test_a_clip_that_fits_comes_back_unchanged(self, speech_audio: Audio) -> None:
        assert fit_reference(speech_audio, 11.5, vad_backend="energy") is speech_audio

    def test_the_result_never_exceeds_the_limit(self, long_audio: Audio) -> None:
        fitted = fit_reference(long_audio, 11.5, vad_backend="energy")
        assert 3.0 <= fitted.duration <= 11.5

    def test_it_ends_in_a_pause_rather_than_mid_word(self, long_audio: Audio) -> None:
        # The synthetic speech falls silent for 0.4 s every 3 s and only dips
        # between syllables. A first version cut at a syllable dip right at the
        # limit; a cut inside a real pause ends on silence.
        fitted = fit_reference(long_audio, 11.5, vad_backend="energy")
        assert fitted.duration < 11.4, "cut at the limit instead of at a pause"

        def rms(samples) -> float:
            return float((samples.astype("float64") ** 2).mean() ** 0.5)

        tail = fitted.samples[-int(0.05 * fitted.sample_rate):]
        assert rms(tail) < 0.25 * rms(fitted.samples), "the clip ends on speech"

    def test_a_syllable_dip_is_not_taken_for_a_pause(self, long_audio: Audio) -> None:
        # Every boundary the VAD reports is kept honest by the pause length it
        # needs: the cut must follow a gap of at least MIN_PAUSE_SECONDS.
        from voiceclone.audio.segment import MIN_PAUSE_SECONDS

        segments = detect_speech(long_audio, backend="energy")
        following = [start for start, _ in segments[1:]] + [long_audio.duration]
        real_pauses = [
            end for (_, end), nxt in zip(segments, following, strict=True) if nxt - end >= MIN_PAUSE_SECONDS
        ]
        fitted = fit_reference(long_audio, 11.5, vad_backend="energy")
        # 1 ms of slack for rounding the cut to a whole sample.
        assert any(end < fitted.duration <= end + 0.1 + 1e-3 for end in real_pauses)


class TestEngineReferenceLimit:
    def test_f5_declares_a_limit_under_its_own_12_second_clip(self) -> None:
        limit = list_tts_engines()["f5"].capabilities.max_reference_seconds
        assert limit is not None and limit < 12.0

    def test_a_long_reference_is_cut_and_transcribed_again(
        self, quiet_config: AppConfig, long_audio: Audio, speech_audio: Audio
    ) -> None:
        engine = RecordingEngine(limit=11.5, output=speech_audio.slice_seconds(0.0, 1.5))
        transcribed: list[float] = []
        studio = studio_with(quiet_config, engine, long_audio, transcribed)

        result = studio.synthesize("سلام، حالت خوبه؟", "unused.wav", engine="f5",
                                   reference_text="متنی که برای کلیپ بلند نوشته شده")

        conditioned, handed = engine.calls[0]
        assert conditioned <= 11.5
        # Transcribed once, and it was the cut that was transcribed, not the long clip.
        assert transcribed == [pytest.approx(conditioned)]
        # The transcript written for the long clip was not passed on.
        assert handed == TRANSCRIPT_OF_THE_CUT
        assert any("cut to" in warning for warning in result.warnings)

    def test_an_engine_without_a_limit_gets_the_whole_reference(
        self, quiet_config: AppConfig, long_audio: Audio, speech_audio: Audio
    ) -> None:
        engine = RecordingEngine(limit=None, output=speech_audio.slice_seconds(0.0, 1.5))
        transcribed: list[float] = []
        studio = studio_with(quiet_config, engine, long_audio, transcribed)

        result = studio.synthesize("سلام، حالت خوبه؟", "unused.wav", engine="f5",
                                   reference_text="متن کامل کلیپ")

        assert engine.calls[0] == (pytest.approx(long_audio.duration), "متن کامل کلیپ")
        assert transcribed == []
        assert not any("cut to" in warning for warning in result.warnings)

    def test_a_reference_under_the_limit_keeps_its_transcript(
        self, quiet_config: AppConfig, speech_audio: Audio
    ) -> None:
        engine = RecordingEngine(limit=11.5, output=speech_audio.slice_seconds(0.0, 1.5))
        transcribed: list[float] = []
        studio = studio_with(quiet_config, engine, speech_audio, transcribed)

        studio.synthesize("سلام، حالت خوبه؟", "unused.wav", engine="f5", reference_text="متن کامل کلیپ")

        assert engine.calls[0] == (pytest.approx(speech_audio.duration), "متن کامل کلیپ")
        assert transcribed == []
