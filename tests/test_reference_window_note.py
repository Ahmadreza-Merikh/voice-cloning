"""A windowed reference says so beside its transcript.

The engine is conditioned on the window the preprocessing keeps, not on the
file that was uploaded, so the transcript has to describe the window - which
looks exactly like a transcription that stopped early. The user reported it as
one on 2026-09-21. Nothing about the pairing changes here; what changes is that
the cut is stated where the transcript is read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voiceclone.audio import analyze
from voiceclone.audio.io import Audio
from voiceclone.audio.preprocess import PreprocessResult
from voiceclone.config import AppConfig
from voiceclone.ui.app import REFERENCE_UPLOAD, VoiceCloneUI


class Studio:
    def __init__(self, whole: Audio, kept: Audio, window_start: float | None) -> None:
        self.result = PreprocessResult(audio=kept, report_before=analyze(whole), report_after=analyze(kept),
                                       window_start=window_start,
                                       window_end=None if window_start is None else window_start + kept.duration)

    def prepare_reference(self, reference: Any, overrides: Any = None) -> PreprocessResult:
        return self.result

    def transcribe_reference(self, audio: Audio, **_: Any) -> str:
        return "رونوشت خودکار"

    def release_recogniser(self) -> None:
        pass


def ui_with(config: AppConfig, tmp_path: Path, studio: Studio) -> VoiceCloneUI:
    made = VoiceCloneUI(config.copy_with(**{"paths.voices_dir": str(tmp_path / "voices")}))
    made._studio = studio
    return made


@pytest.fixture
def analysed(config: AppConfig, tmp_path: Path, speech_audio: Audio, long_audio: Audio):
    def run(window_start: float | None):
        studio = Studio(long_audio, speech_audio, window_start)
        ui = ui_with(config, tmp_path, studio)
        _, summary, _, transcript = ui.on_analyse(REFERENCE_UPLOAD, "ref.wav", None, "auto", False, True, True,
                                                  "f5", progress=None)
        return summary, transcript, ui
    return run


def test_a_cut_reference_says_which_seconds_the_transcript_covers(analysed) -> None:
    summary, transcript, ui = analysed(0.5)
    assert transcript == "رونوشت خودکار"
    # The numbers matter: "why is it short" is answered by seeing both durations.
    assert "0.5" in summary and ui.strings["transcript_is_window"].split("{")[0] in summary


def test_an_untouched_reference_says_nothing_extra(analysed) -> None:
    summary, transcript, ui = analysed(None)
    assert transcript == "رونوشت خودکار"
    assert ui.strings["transcript_is_window"].split("{")[0] not in summary
