"""Shared pytest fixtures.

None of the tests here need model weights or a network connection; anything
that does is marked ``slow`` and skipped by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voiceclone.audio.io import Audio  # noqa: E402
from voiceclone.config import AppConfig  # noqa: E402


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return AppConfig.load()


@pytest.fixture(autouse=True)
def _no_real_ezafe_tagger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparing text for OmniVoice would load the 620 MB ezafe tagger; no test does.

    A test that wants ezafes gives the writer a tagger of its own.
    """
    from voiceclone.errors import TextProcessingError
    from voiceclone.text import ezafe_fa

    def refuse(self: ezafe_fa.EzafeWriter) -> None:
        raise TextProcessingError("the tests do not load the ezafe tagger")

    monkeypatch.setattr(ezafe_fa.EzafeWriter, "_load", refuse)

    from voiceclone.text import g2p

    def no_model(self: g2p.PersianG2P, text: str, *, normalize: bool = True) -> None:
        raise TextProcessingError("the tests do not load Homo-GE2PE")

    # A test that wants readings gives the pipeline a G2P of its own.
    monkeypatch.setattr(g2p.PersianG2P, "phonemize", no_model)


def _synthetic_speech(
    duration: float = 6.0,
    sample_rate: int = 24000,
    *,
    noise_db: float = -40.0,
    f0: float = 130.0,
) -> np.ndarray:
    """A crude voiced signal: harmonics with syllable-rate amplitude modulation.

    Not speech, but it has the properties the analysers look at - harmonic
    structure, an energy envelope with pauses, and a noise floor.
    """
    rng = np.random.default_rng(1234)
    t = np.linspace(0.0, duration, int(duration * sample_rate), endpoint=False)

    # Harmonics up to ~8 kHz so band-limiting tests have something to remove.
    signal = np.zeros_like(t)
    harmonic = 1
    while f0 * harmonic < 8000.0:
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
        harmonic += 1
    signal /= np.max(np.abs(signal))

    # Syllable envelope at ~4 Hz. It dips but never reaches zero, because
    # voiced speech stays continuous between syllables - an envelope that
    # truly zeroes would make any VAD see hundreds of tiny pauses.
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * 4.0 * t)
    envelope[t % 3.0 > 2.6] = 0.0  # one real pause every three seconds
    signal *= envelope

    noise = rng.normal(0.0, 10.0 ** (noise_db / 20.0), size=t.shape)
    return (0.6 * signal + noise).astype(np.float32)


@pytest.fixture
def speech_audio() -> Audio:
    return Audio(_synthetic_speech(), 24000, source="<synthetic>")


@pytest.fixture
def noisy_audio() -> Audio:
    return Audio(_synthetic_speech(noise_db=-18.0), 24000, source="<synthetic-noisy>")


@pytest.fixture
def short_audio() -> Audio:
    return Audio(_synthetic_speech(duration=1.2), 24000, source="<synthetic-short>")


@pytest.fixture
def long_audio() -> Audio:
    return Audio(_synthetic_speech(duration=30.0), 24000, source="<synthetic-long>")


@pytest.fixture
def persian_text() -> str:
    return (
        "سلام، حال شما چطور است؟ امیدوارم روز خوبی داشته باشید. "
        "قیمت این کتاب ۲۵۰۰۰ تومان است."
    )


__all__ = ["Audio"]
