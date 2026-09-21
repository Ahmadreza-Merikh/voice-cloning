"""Speech-recogniser decode policy and the scoring pin.

Loading a Whisper checkpoint is out of scope here: these tests lock the
arguments the Studio will pass, and the historical block the benchmark must
not drift from.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from voiceclone.config import AppConfig, ASRConfig
from voiceclone.engines.asr import (
    SCORING_ASR,
    apply_scoring_asr,
    ct2_cache_dir,
    faster_whisper_decode_options,
    is_ct2_model,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_asr_under_test", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules["benchmark_asr_under_test"] = benchmark
spec.loader.exec_module(benchmark)


def test_studio_decode_keeps_vad_on() -> None:
    options = faster_whisper_decode_options(ASRConfig(), "fa")
    assert options["vad_filter"] is True
    assert options["no_speech_threshold"] == 0.6
    assert "initial_prompt" not in options
    assert options["condition_on_previous_text"] is False
    assert options["language"] == "fa"
    assert options["vad_parameters"]["speech_pad_ms"] == 400
    assert options["vad_parameters"]["threshold"] == 0.5


def test_empty_prompt_is_omitted() -> None:
    options = faster_whisper_decode_options(ASRConfig(initial_prompt=""), "fa")
    assert "initial_prompt" not in options


def test_scoring_block_is_the_historical_measurement() -> None:
    assert SCORING_ASR == {
        "model": "large-v3-turbo",
        "vad_filter": True,
        "no_speech_threshold": 0.6,
        "initial_prompt": "",
        "vad_threshold": 0.5,
        "vad_min_silence_ms": 2000,
        "vad_speech_pad_ms": 400,
    }
    options = faster_whisper_decode_options(
        ASRConfig(
            vad_filter=True,
            no_speech_threshold=0.6,
            initial_prompt="",
            vad_threshold=0.5,
            vad_min_silence_ms=2000,
            vad_speech_pad_ms=400,
        ),
        "fa",
    )
    assert options["vad_filter"] is True
    assert options["no_speech_threshold"] == 0.6
    assert "initial_prompt" not in options
    assert options["vad_parameters"]["speech_pad_ms"] == 400
    assert options["vad_parameters"]["threshold"] == 0.5


def test_apply_scoring_asr_overrides_product_defaults() -> None:
    config = apply_scoring_asr(AppConfig.load())
    assert config.engines.asr.model == "large-v3-turbo"
    assert config.engines.asr.vad_filter is True
    assert config.engines.asr.no_speech_threshold == 0.6
    assert config.engines.asr.initial_prompt == ""
    assert config.engines.asr.vad_speech_pad_ms == 400
    assert config.engines.asr.vad_threshold == 0.5


def test_benchmark_harness_pins_the_scoring_recogniser() -> None:
    config = benchmark.harness_config(SimpleNamespace(device=None, candidates=1))
    assert config.engines.asr.model == "large-v3-turbo"
    assert config.engines.asr.vad_filter is True
    assert config.engines.asr.no_speech_threshold == 0.6
    assert config.engines.asr.initial_prompt == ""
    assert config.engines.asr.vad_speech_pad_ms == 400
    assert config.engines.omnivoice.memory_guard is False


def test_ct2_cache_dir_slugs_hub_ids(tmp_path: Path) -> None:
    path = ct2_cache_dir(tmp_path, "nezamisafa/whisper-persian-v4")
    assert path == tmp_path / "whisper" / "nezamisafa--whisper-persian-v4-ct2"
    assert not is_ct2_model(path)
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(b"")
    (path / "config.json").write_text("{}", encoding="utf-8")
    assert is_ct2_model(path)
