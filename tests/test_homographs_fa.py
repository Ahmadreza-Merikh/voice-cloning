"""A homograph's first vowel goes where Homo-GE2PE's reading says, on the listed words only, for OmniVoice only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from voiceclone.config import HomographConfig, TextConfig
from voiceclone.errors import TextProcessingError
from voiceclone.text import homographs_fa as hf
from voiceclone.text.g2p import G2PResult
from voiceclone.text.homographs_data import FIRST_VOWEL_HOMOGRAPHS
from voiceclone.text.marks_fa import DAMMA, FATHA, KASRA
from voiceclone.text.pipeline import TextPipeline

ROOT = Path(__file__).resolve().parents[1]
STEP3 = json.loads((ROOT / "eval/runs/frontend-listening-v2/texts.json").read_text(encoding="utf-8"))["sentences"]


def scorer():
    spec = importlib.util.spec_from_file_location("score_frontend_components",
                                                  ROOT / "scripts" / "score_frontend_components.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_frontend_components"] = module
    spec.loader.exec_module(module)
    return module


def test_the_alignment_is_the_one_step_1_scored_with() -> None:
    sfc = scorer()
    for s in STEP3:
        ws, ts = sfc.words(s["text"]), [sfc.from_ge2pe(t) for t in s["ge2pe"]]
        assert hf.words_of(s["text"]) == ws
        assert hf.groups(ws, ts) == sfc.groups(ws, ts)


def test_step_3s_heard_marks_come_back_from_its_readings_except_on_be() -> None:
    for s in STEP3:
        text, written = hf.write_first_vowels(s["normalized"], s["texts"]["E"], s["ge2pe"])
        assert text == s["texts"]["EH"].replace("بِه ", "به ")
        assert "به" not in written


def test_the_list_holds_the_homographs_step_3_marked_and_not_be() -> None:
    assert "به" not in FIRST_VOWEL_HOMOGRAPHS
    assert {"محرم", "مهر", "شکر", "سر", "کشتی", "مسکن"} <= FIRST_VOWEL_HOMOGRAPHS
    assert len(FIRST_VOWEL_HOMOGRAPHS) == 190


class FakeG2P:
    backend = "fake"

    def __init__(self, compact: list[str]) -> None:
        self.compact = compact
        self.calls = 0
        self.released = 0

    def phonemize(self, text: str) -> G2PResult:
        self.calls += 1
        return G2PResult(text, "", [], self.backend, list(self.compact))

    def release(self) -> None:
        self.released += 1


SENTENCE = "کارمند مهر شرکت را زد و رئیس با مهر از او تشکر کرد."
READING = ["karm/nd", "mohre", "$erk/t", "ra", "z/d", "v/", "r/@is", "ba", "mehr", "@/z", "@u", "t/$/kkor", "k/rd"]


def test_the_writer_marks_reads_once_and_lets_the_model_go() -> None:
    g2p = FakeG2P(READING)
    writer = hf.HomographWriter(HomographConfig())
    first = writer.write([SENTENCE], [SENTENCE], g2p)
    second = writer.write([SENTENCE], [SENTENCE], g2p)
    assert first == second
    text, written, failed = first
    assert failed is None and written == ["مهر", "مهر", "کرد"]
    assert text[0] == "کارمند م" + DAMMA + "هر شرکت را زد و رئیس با م" + KASRA + "هر از او تشکر ک" + FATHA + "رد."
    assert g2p.calls == 1 and g2p.released == 1


def test_on_auto_it_goes_on_unmarked_without_a_g2p_and_says_why_and_when_required_it_stops() -> None:
    texts, written, failed = hf.HomographWriter(HomographConfig()).write([SENTENCE], [SENTENCE], None)
    assert (texts, written) == ([SENTENCE], []) and "G2P is turned off" in failed
    with pytest.raises(TextProcessingError):
        hf.HomographWriter(HomographConfig(enabled=True)).write([SENTENCE], [SENTENCE], None)


def test_omnivoice_gets_the_vowels_and_other_engines_plain_script() -> None:
    def pipeline() -> TextPipeline:
        p = TextPipeline(TextConfig())
        p._g2p = FakeG2P(READING)
        return p

    omni = pipeline().prepare(SENTENCE, engine_name="omnivoice")
    assert "م" + DAMMA + "هر" in omni.chunks[0].synth_text
    assert any(note.startswith("First vowel written on 3 homograph") for note in omni.notes)
    assert pipeline().prepare(SENTENCE, engine_name="f5").chunks[0].synth_text == SENTENCE
