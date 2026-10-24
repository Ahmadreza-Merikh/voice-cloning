"""OmniVoice reads with the adapted export the check chose, only once it is here and verified.

Nothing here loads a model: the weights are a few bytes with a record beside
them, and the studio test hands it an engine that says what it read with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voiceclone.audio import analyze
from voiceclone.audio.io import Audio
from voiceclone.audio.preprocess import PreprocessResult
from voiceclone.config import AppConfig
from voiceclone.device import get_device_spec
from voiceclone.engines.base import EngineCapabilities, SynthesisResult
from voiceclone.engines.omnivoice_tts import PUBLISHED_MODEL_ID, OmniVoiceEngine, adapted_weights
from voiceclone.errors import EngineNotAvailableError
from voiceclone.pipelines.studio import Studio

SHA = "9c453a74e9eab8324c27e039b9bdc052cd8d6721cecbb20df53c919a0e6ba39a"


def export(folder: Path, sha: str = SHA, verified: bool = True, size: int = 7) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(b"weights")
    (folder / "SOURCE.json").write_text(json.dumps({"model_sha256": sha, "model_bytes": size, "verified": verified}))
    return folder


def engine(config: AppConfig, tmp_path: Path, **settings: Any):
    overrides = {"engines.omnivoice.adapted_dir": str(tmp_path / "adapted"),
                 **{f"engines.omnivoice.{name}": value for name, value in settings.items()}}
    configured = config.copy_with(**overrides)
    return OmniVoiceEngine(configured, get_device_spec("cpu")), configured.engines.omnivoice


def test_the_default_is_the_export_the_check_chose(config: AppConfig) -> None:
    settings = config.engines.omnivoice
    assert (settings.weights, settings.adapted_sha256, settings.model_id) == ("auto", SHA, PUBLISHED_MODEL_ID)


def test_only_a_folder_whose_record_names_the_weights_is_taken(config: AppConfig, tmp_path: Path) -> None:
    _, settings = engine(config, tmp_path)
    assert adapted_weights(settings) is None
    export(tmp_path / "adapted", sha="0" * 64)
    assert adapted_weights(settings) is None
    export(tmp_path / "adapted", verified=False)
    assert adapted_weights(settings) is None
    export(tmp_path / "adapted", size=8)
    assert adapted_weights(settings) is None
    export(tmp_path / "adapted")
    assert adapted_weights(settings) == tmp_path / "adapted"


def test_auto_reads_with_the_export_when_it_is_here(config: AppConfig, tmp_path: Path) -> None:
    export(tmp_path / "adapted")
    omnivoice, settings = engine(config, tmp_path)
    assert omnivoice._model_source(settings) == str(tmp_path / "adapted")
    assert omnivoice._weights_note is None


def test_without_it_auto_reads_the_published_model_and_says_how_to_fetch_it(config: AppConfig, tmp_path: Path) -> None:
    omnivoice, settings = engine(config, tmp_path)
    assert omnivoice._model_source(settings) == PUBLISHED_MODEL_ID
    assert "fetch_adapted_weights.py" in omnivoice._weights_note


def test_published_or_another_model_named_outright_is_read_as_named(config: AppConfig, tmp_path: Path) -> None:
    export(tmp_path / "adapted")
    omnivoice, settings = engine(config, tmp_path, weights="published")
    assert omnivoice._model_source(settings) == PUBLISHED_MODEL_ID
    omnivoice, settings = engine(config, tmp_path, model_id="/tmp/omnivoice-base")
    assert omnivoice._model_source(settings) == "/tmp/omnivoice-base"
    assert omnivoice._weights_note is None


def test_adapted_refuses_to_start_without_the_export(config: AppConfig, tmp_path: Path) -> None:
    omnivoice, settings = engine(config, tmp_path, weights="adapted")
    with pytest.raises(EngineNotAvailableError):
        omnivoice._model_source(settings)


class NotingEngine:
    display_name = "Noting engine"

    def __init__(self, output: Audio) -> None:
        self.capabilities = EngineCapabilities(requires_reference_text=True)
        self.output = output

    def load(self) -> None:
        pass

    def synthesize(self, text: str, reference: Audio, **kwargs: Any) -> SynthesisResult:
        return SynthesisResult(audio=self.output, engine="noting", seed=0,
                               metadata={"weights_note": "read with the published weights"})


def test_the_note_reaches_the_output_once(config: AppConfig, speech_audio: Audio) -> None:
    quiet = config.copy_with(**{"evaluation.enabled": False, "evaluation.candidates": 1})
    studio = Studio(quiet)
    report = analyze(speech_audio)
    studio.tts_engine = lambda name=None: NotingEngine(speech_audio.slice_seconds(0.0, 1.0))
    studio.prepare_reference = lambda source, **_: PreprocessResult(audio=speech_audio, report_before=report,
                                                                   report_after=report)
    result = studio.synthesize("سلام. حالت خوبه؟ امروز هوا خوب است.", "unused.wav", engine="omnivoice",
                               reference_text="متن کلیپ")
    assert result.warnings.count("read with the published weights") == 1
