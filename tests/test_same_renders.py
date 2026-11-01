"""Runs on the same weights agree on every row; runs on other weights do not."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("same_renders", ROOT / "scripts" / "same_renders.py")
same = importlib.util.module_from_spec(spec)
spec.loader.exec_module(same)


def write(root: Path, label: str, shift: float) -> None:
    rows = [{"speaker": f"nev{i:02d}", "sentence": f"e{j:02d}", "seed": 42,
             "held_out_similarity": round(0.6 + i / 100 + j / 1000 + shift * (j % 2), 4),
             "reference_similarity": 0.8, "transcription": "یک" if j % 2 else "دو"}
            for i in range(1, 4) for j in range(1, 5)]
    (root / f"{label}.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")


def test_the_same_run_twice_agrees_on_every_row_and_other_weights_do_not(tmp_path: Path) -> None:
    write(tmp_path, "a", 0.0)
    write(tmp_path, "b", 0.0)
    write(tmp_path, "c", 0.01)
    out = tmp_path / "same.json"
    assert same.main([str(tmp_path), "a", "b", "c", "--out", str(out)]) == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["a vs b"] == {"rows_in_both": 12, "same_held_out_similarity": 12, "same_reference_similarity": 12,
                             "same_transcription": 12}
    assert got["a vs c"]["same_held_out_similarity"] == 6 and got["b vs c"]["rows_in_both"] == 12
