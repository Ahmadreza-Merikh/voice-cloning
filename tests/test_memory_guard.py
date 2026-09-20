"""OmniVoice is the default, and the recogniser is never loaded beside it.

Measured on this 16 GB Mac (2026-09-17): an OmniVoice render given its
reference transcript peaked at 3.4 GB with no memory pressure, while one that
transcribed *while the engine was loaded* reached warning pressure and was
stopped. Measured again (2026-09-20,
`eval/runs/evidence/recogniser-release/`): releasing the recogniser gives back
the 1.3 GB faster-whisper took, twice in a row, with the process returning to
0.9 GB both times. So the rule is not "type it yourself" but "one at a time":
the reference is transcribed first, the recogniser is released, and only then
does a guarded engine load. What is still refused is best-of-N, which
transcribes each take while the engine holds memory. No model is loaded here:
stand-ins record what they were asked to do.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from voiceclone.audio import analyze
from voiceclone.audio.io import Audio
from voiceclone.audio.preprocess import PreprocessResult
from voiceclone.config import AppConfig
from voiceclone.engines import list_tts_engines
from voiceclone.engines.base import EngineCapabilities, SynthesisResult
from voiceclone.errors import SynthesisError, VoiceCloneError
from voiceclone.pipelines.studio import Studio
from voiceclone.ui.app import REFERENCE_SAVED, REFERENCE_UPLOAD, VoiceCloneUI
from voiceclone.ui.strings import DEFAULT_PRESET

ROOT = Path(__file__).resolve().parents[1]


class Engine:
    display_name = "OmniVoice (stand-in)"

    def __init__(self, output: Audio, log: list[str]) -> None:
        self.capabilities = EngineCapabilities(requires_reference_text=True)
        self.output = output
        self.log = log
        self.calls: list[str | None] = []

    def load(self) -> None:
        self.log.append("loaded")

    def synthesize(self, text: str, reference: Audio, *, reference_text: str | None = None,
                   speed: float = 1.0, seed: int | None = None, **kwargs: Any) -> SynthesisResult:
        self.calls.append(reference_text)
        return SynthesisResult(audio=self.output, engine="omnivoice", seed=seed or 0, metadata={})


class Recogniser:
    """Stands in for a loaded ASREngine, and says when it is unloaded."""

    def __init__(self, log: list[str], text: str = "متن مبدأ") -> None:
        self.log = log
        self.text = text

    def transcribe(self, audio: Audio, **kwargs: Any) -> Any:
        self.log.append("source transcribed")
        from voiceclone.engines.asr import Transcript

        return Transcript(text=self.text)

    def unload(self) -> None:
        self.log.append("released")


@pytest.fixture
def quiet(config: AppConfig, monkeypatch) -> AppConfig:
    from voiceclone import memory

    # The machine's own pressure must not decide what these tests see.
    monkeypatch.setattr(memory, "pressure_level", lambda: 1)
    return config.copy_with(**{"evaluation.enabled": False, "evaluation.candidates": 1})


def studio_with(config: AppConfig, engine: Engine, reference: Audio, log: list[str]) -> Studio:
    studio = Studio(config)
    report = analyze(reference)
    studio.tts_engine = lambda name=None: engine

    def prepare(source, **_):
        log.append("prepared")
        return PreprocessResult(audio=reference, report_before=report, report_after=report)

    def transcribe(audio: Audio, *, language: str | None = None) -> str:
        log.append("transcribed")
        # What the real call leaves behind: a loaded recogniser.
        studio._asr = Recogniser(log)
        return "رونوشت خودکار"

    studio.prepare_reference = prepare
    studio.transcribe_reference = transcribe
    return studio


class TestTheDefault:
    def test_omnivoice_is_the_default_and_is_guarded(self, config: AppConfig) -> None:
        assert config.engines.tts == "omnivoice"
        assert Studio(config).memory_guarded() is True
        assert Studio(config).memory_guarded("f5") is False

    def test_the_benchmark_turns_the_guard_off(self) -> None:
        spec = importlib.util.spec_from_file_location("benchmark_guard", ROOT / "scripts" / "benchmark.py")
        benchmark = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_guard"] = benchmark
        spec.loader.exec_module(benchmark)
        config = benchmark.harness_config(argparse.Namespace(device=None, candidates=1))
        assert config.engines.omnivoice.memory_guard is False
        assert config.evaluation.transcribe_check is True
        assert config.evaluation.speaker_similarity is False


class TestTheStudio:
    def test_the_reference_is_transcribed_and_the_recogniser_released_before_the_engine_loads(
        self, quiet: AppConfig, speech_audio: Audio
    ) -> None:
        log: list[str] = []
        engine = Engine(speech_audio, log)
        studio = studio_with(quiet, engine, speech_audio, log)
        studio.synthesize("سلام", "unused.wav", engine="omnivoice")
        assert log == ["prepared", "transcribed", "released", "loaded"]
        assert engine.calls == ["رونوشت خودکار"] and studio._asr is None

    def test_a_typed_transcript_is_used_and_a_leftover_recogniser_still_goes_first(
        self, quiet: AppConfig, speech_audio: Audio
    ) -> None:
        log: list[str] = []
        engine = Engine(speech_audio, log)
        studio = studio_with(quiet, engine, speech_audio, log)
        studio._asr = Recogniser(log)  # left loaded by an earlier F5 render
        studio.synthesize("سلام", "unused.wav", engine="omnivoice", reference_text="متن مرجع")
        assert log == ["prepared", "released", "loaded"]
        assert engine.calls == ["متن مرجع"]

    def test_best_of_n_is_refused(self, quiet: AppConfig, speech_audio: Audio) -> None:
        log: list[str] = []
        engine = Engine(speech_audio, log)
        studio = studio_with(quiet, engine, speech_audio, log)
        with pytest.raises(SynthesisError, match="one take"):
            studio.synthesize("سلام", "unused.wav", engine="omnivoice", reference_text="متن", candidates=2)
        assert engine.calls == [] and log == []

    def test_with_the_guard_off_the_recogniser_stays_loaded(self, quiet: AppConfig, speech_audio: Audio) -> None:
        log: list[str] = []
        engine = Engine(speech_audio, log)
        studio = studio_with(quiet.copy_with(**{"engines.omnivoice.memory_guard": False}), engine,
                             speech_audio, log)
        studio.synthesize("سلام", "unused.wav", engine="omnivoice")
        assert log == ["prepared", "transcribed", "loaded"]
        assert engine.calls == ["رونوشت خودکار"] and studio._asr is not None

    def test_speech_to_speech_transcribes_both_before_the_engine_loads(
        self, quiet: AppConfig, speech_audio: Audio
    ) -> None:
        log: list[str] = []
        engine = Engine(speech_audio, log)
        studio = studio_with(quiet, engine, speech_audio, log)
        studio._asr = Recogniser(log)
        studio.convert(speech_audio, speech_audio, mode="cascade", engine="omnivoice")
        assert log == ["source transcribed", "prepared", "transcribed", "released", "loaded"]
        assert engine.calls == ["رونوشت خودکار"] and studio._asr is None

    def test_releasing_the_recogniser_unloads_it(self, quiet: AppConfig) -> None:
        log: list[str] = []
        studio = Studio(quiet)
        studio._asr = Recogniser(log)
        studio.release_recogniser()
        assert log == ["released"] and studio._asr is None


class FakeStudio:
    def __init__(self, reference: Audio) -> None:
        report = analyze(reference)
        self.result = PreprocessResult(audio=reference, report_before=report, report_after=report)
        self.events: list[Any] = []

    def prepare_reference(self, reference, overrides=None):
        return self.result

    def transcribe_reference(self, audio: Audio, **_) -> str:
        self.events.append("transcribed")
        return "رونوشت خودکار"

    def release_recogniser(self) -> None:
        self.events.append("released")

    def synthesize(self, text, reference, **kwargs):
        self.events.append(("synthesize", kwargs["engine"], kwargs["reference_text"]))
        raise VoiceCloneError("stop here")


@pytest.fixture
def ui(config: AppConfig, tmp_path: Path, speech_audio: Audio) -> VoiceCloneUI:
    made = VoiceCloneUI(config.copy_with(**{"paths.voices_dir": str(tmp_path / "voices")}))
    made._studio = FakeStudio(speech_audio)
    return made


class TestTheApp:
    def test_it_starts_on_omnivoice_where_omnivoice_runs(self, ui: VoiceCloneUI, monkeypatch) -> None:
        monkeypatch.setattr(list_tts_engines()["omnivoice"], "is_available", classmethod(lambda cls: (True, "")))
        assert ui._default_engine() == "omnivoice"

    def test_it_falls_back_to_f5_where_omnivoice_is_not_installed(self, ui: VoiceCloneUI, monkeypatch) -> None:
        monkeypatch.setattr(list_tts_engines()["omnivoice"], "is_available",
                            classmethod(lambda cls: (False, "no environment")))
        assert ui._default_engine() == "f5"

    def test_analysis_fills_the_transcript_and_then_releases_with_omnivoice(
        self, ui: VoiceCloneUI, speech_audio: Audio
    ) -> None:
        *_, transcript = ui.on_analyse(REFERENCE_UPLOAD, "ref.wav", None, "auto", False, True, True,
                                       "omnivoice", progress=None)
        assert transcript == "رونوشت خودکار"
        assert ui.studio.events == ["transcribed", "released"]

    def test_analysis_keeps_the_recogniser_for_f5(self, ui: VoiceCloneUI) -> None:
        *_, transcript = ui.on_analyse(REFERENCE_UPLOAD, "ref.wav", None, "auto", False, True, True, "f5",
                                       progress=None)
        assert transcript == "رونوشت خودکار"
        assert ui.studio.events == ["transcribed"]

    def test_a_saved_voice_brings_its_transcript_and_nothing_transcribes(
        self, ui: VoiceCloneUI, speech_audio: Audio
    ) -> None:
        slug = ui.library.save("Narrator", speech_audio, transcript="متن ذخیره‌شده").slug
        with pytest.raises(gr.Error):
            ui.on_generate_tts("سلام", REFERENCE_SAVED, None, slug, "", "omnivoice", DEFAULT_PRESET,
                               1.0, 0, 1, False, "auto", False, True, True, progress=None)
        assert ui.studio.events == [("synthesize", "omnivoice", "متن ذخیره‌شده")]

    def test_saving_a_voice_transcribes_and_releases_under_a_guarded_default(
        self, ui: VoiceCloneUI, speech_audio: Audio, monkeypatch
    ) -> None:
        monkeypatch.setattr(ui, "_resolve_reference", lambda *args: speech_audio)
        ui.on_save_voice("Narrator", REFERENCE_UPLOAD, "ref.wav", None, "", "", "auto", False, True, True,
                         progress=None)
        assert ui.studio.events == ["transcribed", "released"]
        assert ui.library.list()[0].transcript == "رونوشت خودکار"


class TestTheCommandLine:
    def run(self, argv: list[str], monkeypatch) -> dict:
        from voiceclone import cli

        seen: dict = {}

        def convert(self, source, reference, **kwargs):
            seen.update(kwargs)
            raise VoiceCloneError("stop here")

        monkeypatch.setattr(Studio, "convert", convert)
        args = cli.build_parser().parse_args(argv)
        with pytest.raises(VoiceCloneError):
            cli.command_s2s(args)
        return seen

    def test_speech_to_speech_uses_the_configured_engine(self, monkeypatch) -> None:
        assert self.run(["s2s", "source.wav", "-r", "ref.wav"], monkeypatch)["engine"] == "omnivoice"

    def test_an_engine_asked_for_is_still_used(self, monkeypatch) -> None:
        assert self.run(["s2s", "source.wav", "-r", "ref.wav", "-e", "f5"], monkeypatch)["engine"] == "f5"
