"""The ezafe-marked adaptation's listening check is read by the rule registered before any transcript was marked."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read_ezafe_check", ROOT / "scripts" / "read_ezafe_check.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

WORDS = {"e01": ["خبرنگار", "ما", "دیروز", "رسید."], "e02": ["مدیر", "مدرسه", "امروز", "آمد."]}
SLOTS = {"e01": {"1": "A1", "2": "E1", "3": "E5"}, "e02": {"1": "E5", "2": "A1", "3": "E1"}}


def write(folder: Path, name: str, data) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def answers(marks: dict[str, dict[str, list[int]]], skip: str | None = None) -> dict:
    out = {}
    for sid, slots in SLOTS.items():
        if sid == skip:
            continue
        given = {slot: ({"ok": False, "words": marks[sid][form], "note": ""} if marks[sid][form]
                        else {"ok": True, "words": [], "note": ""}) for slot, form in slots.items()}
        out[sid] = {"best": "0", "slots": given}
    return {"listener": "test", "answers": out}


@pytest.fixture
def check(tmp_path: Path, monkeypatch):
    review, run = tmp_path / "review", tmp_path / "run"
    monkeypatch.setattr(reader, "REVIEW", review)
    monkeypatch.setattr(reader, "RUN", run)
    monkeypatch.setattr(reader, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(reader, "SENTENCES", tmp_path / "sentences.json")
    write(tmp_path, "sentences.json", {"sentences": [
        {"id": "e01", "ezafe_traps": ["دیروز"]}, {"id": "e02", "ezafe_traps": ["مدرسه", "امروز"]}]})
    write(review, "key.json", {"sentences": {sid: {"slots": slots} for sid, slots in SLOTS.items()}})
    write(review, "items.json", [{"id": sid, "words": words} for sid, words in WORDS.items()])

    def run_it(marks, cer=None, below=(), held=None, extra=None):
        write(review, "answers-listener-1.json", answers(marks))
        if extra:
            write(review, "answers-listener-2.json", extra)
        cer = cer or {"A1": 0.03, "E1": 0.03, "E5": 0.03}
        write(tmp_path, "manifest.json", {"renders": [{"id": sid, "condition": f, "cer": cer[f]}
                                                      for sid in WORDS for f in reader.FORMS]})
        held = held or {"A1": 0.60, "E1": 0.60, "E5": 0.61}
        write(run, "similarity.json", {
            "summary": {f: {"held_out_mean": v} for f, v in held.items()},
            "contrasts": {f: {"mean_low_high": [0.0, -0.01, 0.01], "wholly_below_zero": f in below}
                          for f in reader.EXPORTS}})
        assert reader.main() == 0
        return json.loads((review / "reading.json").read_text(encoding="utf-8"))

    return run_it


def test_the_export_with_fewest_words_wrong_is_chosen_and_the_traps_are_counted(check) -> None:
    got = check({"e01": {"A1": [2, 3], "E1": [2], "E5": []}, "e02": {"A1": [1], "E1": [3], "E5": [3]}})
    assert got["words_marked_wrong"] == {"A1": 3, "E1": 2, "E5": 1}
    assert got["choice"] == "E5"
    assert got["read_beside_it"]["trap_words_marked"] == {
        "A1": {"ezafe_traps": 2}, "E1": {"ezafe_traps": 1}, "E5": {"ezafe_traps": 0}}
    assert got["read_beside_it"]["of_trap_words"] == {"ezafe_traps": 3}


def test_a_flagged_export_is_passed_over_and_said_so(check) -> None:
    marks = {"e01": {"A1": [2, 3], "E1": [2], "E5": []}, "e02": {"A1": [1], "E1": [3], "E5": [3]}}
    got = check(marks, cer={"A1": 0.03, "E1": 0.03, "E5": 0.051})
    assert got["exports"]["E5"]["fewer_but_flagged"] is True
    assert got["choice"] == "E1"
    got = check(marks, cer={"A1": 0.03, "E1": 0.03, "E5": 0.049})
    assert got["choice"] == "E5"
    got = check(marks, below=("E1", "E5"))
    assert got["choice"] == "A1"


def test_a_tie_between_exports_goes_to_the_more_similar_and_a_tie_with_a1_keeps_a1(check) -> None:
    got = check({"e01": {"A1": [2, 3], "E1": [2], "E5": [3]}, "e02": {"A1": [1], "E1": [], "E5": []}},
                held={"A1": 0.60, "E1": 0.62, "E5": 0.61})
    assert got["choice"] == "E1"
    got = check({"e01": {"A1": [2], "E1": [2], "E5": [3]}, "e02": {"A1": [], "E1": [], "E5": []}})
    assert got["choice"] == "A1"


def test_listeners_are_summed_and_an_incomplete_one_is_left_out(check) -> None:
    marks = {"e01": {"A1": [1], "E1": [], "E5": [2]}, "e02": {"A1": [], "E1": [], "E5": []}}
    got = check(marks, extra={**answers(marks), "listener": "second"})
    assert got["words_marked_wrong"] == {"A1": 2, "E1": 0, "E5": 2}
    assert got["of_words"] == 16
    got = check(marks, extra={**answers(marks, skip="e02"), "listener": "partial"})
    assert got["left_out_incomplete"] == ["partial"]
    assert got["words_marked_wrong"] == {"A1": 1, "E1": 0, "E5": 1}
