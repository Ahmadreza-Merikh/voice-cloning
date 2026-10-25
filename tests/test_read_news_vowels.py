"""The news vowels check is read by the rules registered before anyone listened."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read_news_vowels", ROOT / "scripts" / "read_news_vowels.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

WORDS = {"n01": ["قیمت", "دلار", "پایین", "آمد"], "n02": ["تیم", "ملی", "جمعه", "باخت"]}
TEXTS = {"n01": ("قیمتِ دلار پایین آمد", "قِیمتِ دُلار پایین آمد"), "n02": ("تیمِ ملیِ جمعه باخت", "تیمِ مِلیِ جمعه باخت")}
SLOTS = {"n01": {"1": "PB", "2": "PV", "3": "AB", "4": "AV"}, "n02": {"1": "AV", "2": "AB", "3": "PV", "4": "PB"}}


def write(folder: Path, name: str, data) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def answers(marks: dict[str, dict[str, list[int]]], skip: str | None = None) -> dict:
    out = {}
    for sid, slots in SLOTS.items():
        if sid == skip:
            continue
        out[sid] = {"best": "0", "slots": {slot: ({"ok": False, "words": marks[sid][form], "note": ""}
                                                 if marks[sid][form] else {"ok": True, "words": [], "note": ""})
                                           for slot, form in slots.items()}}
    return {"listener": "test", "answers": out}


@pytest.fixture
def check(tmp_path: Path, monkeypatch):
    review, run = tmp_path / "review", tmp_path / "run"
    monkeypatch.setattr(reader, "REVIEW", review)
    monkeypatch.setattr(reader, "RUN", run)
    monkeypatch.setattr(reader, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(reader, "TEXTS", tmp_path / "texts.json")
    write(review, "key.json", {"sentences": {sid: {"slots": slots} for sid, slots in SLOTS.items()}})
    write(review, "items.json", [{"id": sid, "words": words} for sid, words in WORDS.items()])
    write(tmp_path, "texts.json", {"sentences": [{"id": sid, "texts": {"PB": b, "PV": v, "AB": b, "AV": v}}
                                                 for sid, (b, v) in TEXTS.items()],
                                   "first_vowels_against_gold": {"tally": {"right": 3, "wrong": 0, "missing": 1}}})

    def run_it(marks, cer=None, below=False):
        write(review, "answers-listener-1.json", answers(marks))
        cer = cer or dict.fromkeys(reader.FORMS, 0.03)
        write(tmp_path, "manifest.json", {"renders": [{"id": sid, "condition": f, "cer": cer[f]}
                                                      for sid in WORDS for f in reader.FORMS]})
        write(run, "similarity.json", {"contrasts": {"V": {"mean_low_high": [0.0, -0.01, 0.01],
                                                           "wholly_below_zero": below}}})
        assert reader.main() == 0
        return json.loads((review / "reading.json").read_text(encoding="utf-8"))

    return run_it


def test_fewer_words_wrong_with_the_vowels_adopts_them_and_the_weights_follow_their_own_sum(check) -> None:
    got = check({"n01": {"PB": [0, 1], "PV": [], "AB": [1], "AV": []},
                 "n02": {"PB": [2], "PV": [2], "AB": [1, 2], "AV": [2]}})
    assert got["words_marked_wrong"] == {"PB": 3, "PV": 1, "AB": 3, "AV": 1}
    assert got["V"]["adopt"] is True
    assert got["weights"]["choice"] == "adapted (A1, kept)"
    assert got["read_beside_it"]["per_word"]["P"] == {"fixed_marked": 2, "broke_marked": 0,
                                                      "fixed_unmarked": 0, "broke_unmarked": 0}
    assert got["read_beside_it"]["per_word"]["A"]["fixed_marked"] == 2


def test_a_tie_or_a_flag_does_not_adopt_the_vowels(check) -> None:
    tie = {"n01": {"PB": [0], "PV": [1], "AB": [], "AV": []}, "n02": {"PB": [], "PV": [], "AB": [], "AV": []}}
    got = check(tie)
    assert got["V"]["adopt"] is False
    assert got["read_beside_it"]["per_word"]["P"] == {"fixed_marked": 1, "broke_marked": 1,
                                                      "fixed_unmarked": 0, "broke_unmarked": 0}
    fewer = {"n01": {"PB": [0, 1], "PV": [], "AB": [], "AV": []}, "n02": {"PB": [], "PV": [], "AB": [], "AV": []}}
    got = check(fewer, cer={"PB": 0.02, "PV": 0.06, "AB": 0.02, "AV": 0.03})
    assert got["V"]["fewer_but_flagged"] is True and got["V"]["adopt"] is False
    got = check(fewer, below=True)
    assert got["V"]["adopt"] is False


def test_the_published_weights_come_back_only_with_fewer_words_wrong(check) -> None:
    got = check({"n01": {"PB": [], "PV": [], "AB": [0], "AV": []}, "n02": {"PB": [], "PV": [], "AB": [], "AV": []}})
    assert got["weights"]["choice"] == "published"
    got = check({"n01": {"PB": [0], "PV": [], "AB": [0], "AV": []}, "n02": {"PB": [], "PV": [], "AB": [], "AV": []}})
    assert got["weights"]["choice"] == "adapted (A1, kept)"
