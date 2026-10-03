"""The pronunciation review is blind, loudness-matched, and shows plain text only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_pronunciation_review", ROOT / "scripts" / "build_pronunciation_review.py")
review = importlib.util.module_from_spec(spec)
sys.modules["build_pronunciation_review"] = review
spec.loader.exec_module(review)


def manifest(tmp_path: Path) -> Path:
    rate = 24000
    t = np.arange(rate) / rate
    renders = []
    for sid in ("s01", "s02", "s03"):
        for condition, level in (("A", 0.05), ("B", 0.3), ("C", 0.1)):
            if sid == "s03" and condition == "B":
                renders.append({"id": sid, "condition": "B", "skipped": "no twin", "wav": None})
                continue
            path = tmp_path / f"{sid}_{condition}.wav"
            sf.write(path, (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)
            renders.append({"id": sid, "condition": condition, "skipped": None, "wav": str(path)})
    body = {"sentences": [{"id": s, "category": "x", "text": "او گل قرمز را، از گل باغچه کشید.",
                           "diacritized": "او گُل ..."} for s in ("s01", "s02", "s03")], "renders": renders}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return path


def test_slots_are_blind_and_the_key_holds_the_conditions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review, "ROOT", tmp_path)
    key = review.build(manifest(tmp_path), tmp_path / "out", tmp_path / "key.json", seed=5)
    items = json.loads((tmp_path / "out" / "items.json").read_text(encoding="utf-8"))
    page_text = json.dumps(items, ensure_ascii=False)
    assert "_A" not in page_text and "_B" not in page_text and "_C" not in page_text
    assert "گُل" not in page_text
    assert {len(i["slots"]) for i in items} == {2, 3}
    assert sorted(key["sentences"]["s03"]["slots"].values()) == ["A", "C"]
    assert items[0]["words"][:3] == ["او", "گل", "قرمز"] and "را" in items[0]["words"]


def test_every_render_of_a_sentence_ends_at_one_loudness(tmp_path: Path, monkeypatch) -> None:
    from voiceclone.audio.io import load_audio
    from voiceclone.audio.loudness import measure_lufs

    monkeypatch.setattr(review, "ROOT", tmp_path)
    review.build(manifest(tmp_path), tmp_path / "out", tmp_path / "key.json", seed=5)
    items = json.loads((tmp_path / "out" / "items.json").read_text(encoding="utf-8"))
    for item in items:
        levels = [measure_lufs(load_audio(tmp_path / "out" / s["audio"], mono=True)) for s in item["slots"]]
        assert max(levels) - min(levels) < 0.5
    assert {sf.info(p).subtype for p in (tmp_path / "out" / "audio").glob("*.wav")} == {"PCM_16"}


scorer_spec = importlib.util.spec_from_file_location("score_pronunciation_review", ROOT / "scripts" / "score_pronunciation_review.py")
scorer = importlib.util.module_from_spec(scorer_spec)
sys.modules["score_pronunciation_review"] = scorer
scorer_spec.loader.exec_module(scorer)


def test_answers_are_counted_per_condition_through_the_key() -> None:
    key = {"sentences": {"s01": {"category": "homograph", "slots": {"1": "C", "2": "A", "3": "B"}}},
           "wrong_in_twins": {"s01": "a twin note"}}
    items = {"s01": ["او", "گل", "قرمز", "را"]}
    answer = {"listener": "L", "answers": {"s01": {"best": "2", "slots": {
        "1": {"ok": False, "words": [1], "note": "gel"}, "2": {"ok": True, "words": []},
        "3": {"ok": False, "words": []}}}}}
    result = scorer.tabulate(key, items, [answer])
    per = result["per_condition"]
    assert per["A"]["clean"] == 1 and per["A"]["best"] == 1
    assert per["C"]["wrong"] == 1 and per["C"]["words"] == 4
    assert "B" not in per
    assert result["marked"][0]["words"] == ["گل"] and result["marked"][0]["twin_note"] == "a twin note"
