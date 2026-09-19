"""What the benchmark is given: the written test texts, and the stretch of reference used.

The sealed test renders the twelve texts of ``eval/manifests/test_texts_v1.json``
and nothing else, so the file is read whole and refused if its texts no longer
match the hash stored with them. And a reference longer than the preparation
window is cut while the transcript given for it is not, so every run records
the window each reference actually went through.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_inputs_under_test", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules["benchmark_inputs_under_test"] = benchmark
spec.loader.exec_module(benchmark)

TEXTS = ROOT / "eval" / "manifests" / "test_texts_v1.json"


def test_the_written_texts_are_read_whole_and_in_order() -> None:
    sentences = benchmark.load_sentences(TEXTS)
    assert [key for key, _ in sentences] == [f"t{i:02d}" for i in range(1, 13)]
    assert all(text.strip() for _, text in sentences)


def test_texts_changed_after_they_were_hashed_are_refused(tmp_path: Path) -> None:
    data = json.loads(TEXTS.read_text(encoding="utf-8"))
    data["texts"][0]["text"] += " و"
    changed = tmp_path / "texts.json"
    changed.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256"):
        benchmark.load_sentences(changed)


def test_no_built_in_sentence_is_a_test_text() -> None:
    written = {text for _, text in benchmark.load_sentences(TEXTS)}
    assert not written & {text for _, text in benchmark.SENTENCES}


class Studio:
    def __init__(self, start: float, end: float, seconds: float, fail: bool = False) -> None:
        self.calls: list[tuple[str, object]] = []
        self.result = SimpleNamespace(window_start=start, window_end=end,
                                      audio=SimpleNamespace(duration=seconds), steps=["Selected"])
        self.fail = fail

    def prepare_reference(self, source: str, overrides: object = None) -> SimpleNamespace:
        self.calls.append((source, overrides))
        if self.fail:
            raise ValueError("too quiet")
        return self.result


def test_a_clip_used_whole_is_recorded_as_whole() -> None:
    studio = Studio(0.0, 9.9, 9.7)
    window = benchmark.prepared_window(studio, Path("ref.wav"), 9.9, {})
    assert window["whole_clip"] is True
    assert window["window"] == [0.0, 9.9]
    # The same key synthesis uses: the path as a string, no overrides as None.
    assert studio.calls == [("ref.wav", None)]


def test_a_cut_window_is_recorded_as_cut() -> None:
    window = benchmark.prepared_window(Studio(0.06, 11.49, 11.42), Path("ref.wav"), 11.886, {})
    assert window["whole_clip"] is False
    assert window["prepared_seconds"] == 11.42


def test_a_reference_the_studio_refuses_is_recorded_not_raised() -> None:
    window = benchmark.prepared_window(Studio(0, 0, 0, fail=True), Path("ref.wav"), 5.0, {})
    assert window == {"error": "ValueError: too quiet"}
