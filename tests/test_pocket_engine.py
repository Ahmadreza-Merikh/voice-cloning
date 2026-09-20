"""Pocket-TTS Farsi is wired in the way its model card prescribes.

The engine exists to be measured, so what these tests protect is the part a
wrong setting would silently corrupt: the pinned revisions (a re-fetch must not
change what was scored), the prompt length (an engine setting decided by
measurement, `eval/runs/persian-voice-pocket-tts-v1.json`), and the fact that
nothing here hands the model this project's IPA - it phonemises with the G2P
its own card names. No model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voiceclone.audio.io import Audio
from voiceclone.config import AppConfig
from voiceclone.engines import list_tts_engines
from voiceclone.engines.pocket_tts import find_pocket_python, trim_prompt

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "eval" / "plans" / "pocket_tts.json"


def tone(seconds: float, sample_rate: int = 24000) -> Audio:
    samples = np.sin(np.linspace(0, 400 * seconds, int(seconds * sample_rate), dtype=np.float32))
    return Audio(samples.astype(np.float32), sample_rate, source="test")


class TestTheEngine:
    def test_it_is_registered_and_reads_persian_script(self) -> None:
        engine = list_tts_engines()["pocket"]
        # The worker phonemises with the model's own G2P; handing it this
        # project's IPA would be a different recipe than the one measured.
        assert engine.capabilities.accepts_phonemes is False
        # Speaker conditioning is acoustic, so there is no transcript to align.
        assert engine.capabilities.requires_reference_text is False
        assert engine.capabilities.native_persian is True

    def test_an_interpreter_can_be_pointed_at(self, monkeypatch, tmp_path: Path) -> None:
        python = tmp_path / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("")
        monkeypatch.setenv("VOICECLONE_POCKET_PYTHON", str(python))
        assert find_pocket_python() == str(python)

    def test_a_missing_interpreter_is_reported_not_guessed(self, monkeypatch) -> None:
        monkeypatch.setenv("VOICECLONE_POCKET_PYTHON", "/nowhere/python")
        monkeypatch.setattr("voiceclone.engines.pocket_tts._VENV_CANDIDATES", ())
        assert find_pocket_python() is None


class TestThePrompt:
    def test_none_hands_over_the_whole_window(self) -> None:
        audio = tone(11.4)
        assert trim_prompt(audio, None) is audio
        assert trim_prompt(audio, 0) is audio

    def test_a_longer_reference_is_cut_to_the_setting(self) -> None:
        trimmed = trim_prompt(tone(11.4), 4)
        assert abs(trimmed.duration - 4.0) < 0.01

    def test_a_shorter_reference_is_left_alone(self) -> None:
        audio = tone(3.0)
        assert trim_prompt(audio, 4) is audio


class TestWhatWasMeasuredStaysFetchable:
    def test_the_plan_and_the_config_pin_the_same_revisions(self) -> None:
        """A re-fetch must bring back the weights the numbers were measured on."""
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        pinned = {step["name"]: step["revision"] for step in plan["snapshots"]}
        settings = AppConfig.load().engines.pocket
        assert pinned["pocket-tts-farsi-v2"] == settings.revision
        assert pinned["homo-ge2pe"] == settings.g2p_revision

    def test_the_plan_fetches_into_the_directories_the_engine_reads(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        settings = AppConfig.load().engines.pocket
        directories = {step["name"]: step["local_dir"] for step in plan["snapshots"]}
        assert directories["pocket-tts-farsi-v2"] == settings.model_dir
        assert directories["homo-ge2pe"] == settings.g2p_dir

    def test_the_chunk_budget_stays_inside_the_cards_band(self) -> None:
        """9-16 tokens is the clean band; at 21 generations stop terminating."""
        assert 9 <= AppConfig.load().engines.pocket.max_tokens <= 18
