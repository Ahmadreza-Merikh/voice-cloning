"""The listening page's records are split into the scorer's files, adding nothing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_listening_answers", ROOT / "scripts" / "export_listening_answers.py")
exporter = importlib.util.module_from_spec(spec)
sys.modules["export_listening_answers"] = exporter
spec.loader.exec_module(exporter)

KEYS = [{"name": "test-a", "fingerprint": "fa", "trials": {"t1": {}, "t2": {}}},
        {"name": "test-b", "fingerprint": "fb", "trials": {"t3": {}}}]
ANSWER = {"same": "1", "natural": "0", "misread": ["none"]}


def test_a_record_is_split_by_test_and_keeps_only_what_was_answered(tmp_path: Path) -> None:
    by_test, unknown = exporter.split({"listener": "L1", "answers": {"t1": ANSWER, "t3": ANSWER, "zz": ANSWER}}, KEYS)
    assert by_test["test-a"] == {"test": "test-a", "fingerprint": "fa", "listener": "L1", "answers": {"t1": ANSWER}}
    assert by_test["test-b"]["answers"] == {"t3": ANSWER}
    assert unknown == ["zz"]


def test_a_database_document_and_an_answer_code_read_the_same(tmp_path: Path) -> None:
    body = {"listener": "L2", "answers": {"t2": ANSWER}}
    saved = tmp_path / "doc.json"
    saved.write_text(json.dumps({"id": "u_x", "data": {**body, "version": "listening-v1"}, "version": 3}))
    code = tmp_path / "code.txt"
    code.write_text(json.dumps({"v": "listening-v1", **body}) + "\n")
    assert exporter.read_record(saved)["answers"] == exporter.read_record(code)["answers"] == body["answers"]


def test_the_files_written_are_what_the_scorer_reads(tmp_path: Path) -> None:
    keys = []
    for key in KEYS:
        path = tmp_path / f"{key['name']}.key.json"
        path.write_text(json.dumps(key))
        keys.append(path)
    record = tmp_path / "listener one.json"
    record.write_text(json.dumps({"listener": "L1", "answers": {"t1": ANSWER, "t3": ANSWER}}))
    exporter.main(["--keys", *map(str, keys), "--records", str(record), "--out", str(tmp_path / "out")])
    written = json.loads((tmp_path / "out" / "test-a" / "listener-one.json").read_text())
    assert written["test"] == "test-a" and written["fingerprint"] == "fa" and list(written["answers"]) == ["t1"]
    assert (tmp_path / "out" / "test-b" / "listener-one.json").is_file()
