"""The ezafe voices check is read by the rule registered before any render."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read_ezafe_voices", ROOT / "scripts" / "read_ezafe_voices.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

SPEAKERS = [f"nev{i:02d}" for i in range(1, 11)]


def write(folder: Path, form: str, similarity, cer: float = 0.03) -> None:
    rows = [{"speaker": s, "sentence": f"e{j:02d}", "seed": 42, "held_out_similarity": similarity(i, j),
             "character_error_rate": cer} for i, s in enumerate(SPEAKERS) for j in range(1, 13)]
    folder.mkdir(parents=True, exist_ok=True)
    held = sum(r["held_out_similarity"] for r in rows) / len(rows)
    (folder / f"ezafe-voices-{form}.json").write_text(json.dumps(
        {"label": f"ezafe-voices-{form}", "rows": rows,
         "summary": {"held_out_mean": held, "cer_mean": cer, "renders": len(rows), "failures": 0}}))


def read(tmp_path: Path, a1, e1, e5, e5_cer: float = 0.03) -> dict:
    results = tmp_path / "results"
    write(results, "A1", a1)
    write(results, "E1", e1)
    write(results, "E5", e5, e5_cer)
    assert reader.main(results, tmp_path / "reading.json") == 0
    return json.loads((tmp_path / "reading.json").read_text(encoding="utf-8"))


def noise(i: int, j: int) -> float:
    return ((i * 7 + j * 3) % 5 - 2) * 0.004


def test_e5_is_chosen_when_its_similarity_is_not_wholly_below_e1s(tmp_path: Path) -> None:
    got = read(tmp_path, lambda i, j: 0.60 + noise(i, j), lambda i, j: 0.60 + noise(i, j),
               lambda i, j: 0.60 + noise(i, j) + (0.010 if i % 2 else -0.012))
    assert got["held_out_similarity"]["E5-E1"]["mean"] < 0
    assert got["held_out_similarity"]["E5-E1"]["speakers"] == 10
    assert not got["held_out_similarity"]["E5-E1"]["wholly_below_zero"]
    assert got["choice"] == "E5"


def test_e1_stays_when_e5_is_wholly_below_or_its_cer_rises_more_than_two_points(tmp_path: Path) -> None:
    got = read(tmp_path, lambda i, j: 0.60, lambda i, j: 0.60 + noise(i, j), lambda i, j: 0.57 + noise(i, j) / 2)
    assert got["held_out_similarity"]["E5-E1"]["wholly_below_zero"]
    assert got["choice"] == "E1"
    assert got["read_beside_it"]["below_A1"] == ["E5-A1"]
    got = read(tmp_path, lambda i, j: 0.60, lambda i, j: 0.60 + noise(i, j), lambda i, j: 0.60 + noise(j, i),
               e5_cer=0.051)
    assert got["e5_cer_rise_points"] == 2.1 and got["choice"] == "E1"
