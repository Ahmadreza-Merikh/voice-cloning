"""The XTTS check is read by the rule registered before anyone listened."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read_xtts_check", ROOT / "scripts" / "read_xtts_check.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

WORDS = {"x01": ["بانک", "مرکزی", "اعلام", "کرد"], "x02": ["قیمت", "طلا", "بالا", "رفت"]}
SLOTS = {"x01": {"1": "O", "2": "X"}, "x02": {"1": "X", "2": "O"}}


def write(folder: Path, name: str, data) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def check(tmp_path: Path, monkeypatch):
    review, run = tmp_path / "review", tmp_path / "run"
    monkeypatch.setattr(reader, "REVIEW", review)
    monkeypatch.setattr(reader, "RUN", run)
    monkeypatch.setattr(reader, "MANIFEST", tmp_path / "manifest.json")
    write(review, "key.json", {"sentences": {sid: {"slots": slots} for sid, slots in SLOTS.items()}})
    write(review, "items.json", [{"id": sid, "words": words} for sid, words in WORDS.items()])

    def run_it(marks, cer=None, below=False):
        answers = {sid: {"best": "0", "slots": {slot: ({"ok": False, "words": marks[sid][form], "note": ""}
                                                       if marks[sid][form] else {"ok": True, "words": [], "note": ""})
                                                for slot, form in slots.items()}}
                   for sid, slots in SLOTS.items()}
        write(review, "answers-listener-1.json", {"listener": "test", "answers": answers})
        cer = cer or {"O": 0.04, "X": 0.04}
        write(tmp_path, "manifest.json", {"renders": [{"id": sid, "condition": f, "cer": cer[f]}
                                                      for sid in WORDS for f in reader.FORMS]})
        write(run, "similarity.json", {"summary": {"O": {"held_out_mean": 0.6}, "X": {"held_out_mean": 0.5}},
                                       "contrasts": {"X": {"mean_low_high": [-0.1, -0.2, 0.01] if not below
                                                           else [-0.1, -0.2, -0.01], "wholly_below_zero": below}}})
        assert reader.main() == 0
        return json.loads((review / "reading.json").read_text(encoding="utf-8"))

    return run_it


def test_fewer_words_wrong_and_no_flag_adopts_xtts(check) -> None:
    got = check({"x01": {"O": [0, 1], "X": []}, "x02": {"O": [2], "X": [2]}})
    assert got["words_marked_wrong"] == {"O": 3, "X": 1}
    assert got["adopt_X"] is True


def test_a_tie_keeps_the_app_and_a_flag_goes_to_the_user(check) -> None:
    got = check({"x01": {"O": [0], "X": [1]}, "x02": {"O": [], "X": []}})
    assert got["adopt_X"] is False and got["fewer_but_flagged"] is False
    fewer = {"x01": {"O": [0, 1], "X": []}, "x02": {"O": [], "X": []}}
    got = check(fewer, below=True)
    assert got["adopt_X"] is False and got["fewer_but_flagged"] is True
    got = check(fewer, cer={"O": 0.03, "X": 0.06})
    assert got["fewer_but_flagged"] is True
