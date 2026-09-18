"""Every new result says what scored it.

CER on this Mac and CER on a CUDA box are not the same measurement: the
recogniser runs cpu/int8 without a GPU and cuda/float16 with one
(``engines/asr.py``). Results written before this was recorded needed a
hand-written provenance map; these tests keep new ones from needing it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_under_test", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules["benchmark_under_test"] = benchmark
spec.loader.exec_module(benchmark)

EMBEDDER = SimpleNamespace(backend="speechbrain-ecapa")


def studio(recogniser: str | None) -> SimpleNamespace:
    return SimpleNamespace(_asr=SimpleNamespace(backend=recogniser) if recogniser else None)


def test_the_recogniser_precision_is_part_of_the_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICECLONE_SCORING_ENV", raising=False)
    mac = benchmark.scoring_environment(studio("faster-whisper/large-v3-turbo (cpu/int8)"), EMBEDDER)
    gpu = benchmark.scoring_environment(studio("faster-whisper/large-v3-turbo (cuda/float16)"), EMBEDDER)
    assert mac["id"] != gpu["id"]
    assert mac["recogniser"].endswith("(cpu/int8)")


def test_the_same_setup_gets_the_same_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICECLONE_SCORING_ENV", raising=False)
    first = benchmark.scoring_environment(studio("x (cpu/int8)"), EMBEDDER)
    second = benchmark.scoring_environment(studio("x (cpu/int8)"), EMBEDDER)
    assert first["id"] == second["id"]


def test_a_kernel_can_name_its_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICECLONE_SCORING_ENV", "kaggle:some-kernel-v3")
    env = benchmark.scoring_environment(studio(None), EMBEDDER)
    assert env["id"] == "kaggle:some-kernel-v3"
    assert env["similarity_device"] == "cpu"
