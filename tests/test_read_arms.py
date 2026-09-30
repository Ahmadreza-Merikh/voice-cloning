"""The arms reading, applied to made-up results before the real ones exist.

These pin the registered outcomes: the candidate above an arm only when the
speaker-level interval is above zero, the arm above the candidate only when it
is below zero, unresolved otherwise; a speaker with a failed cell is left out.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("read_arms", ROOT / "scripts" / "read_arms.py")
arms = importlib.util.module_from_spec(spec)
sys.modules["read_arms"] = arms
spec.loader.exec_module(arms)

SPEAKERS = [f"ney{i:02d}" for i in range(1, 28)]
TEXTS = [f"t{i:02d}" for i in range(1, 13)]


def run(label: str, shift: float, seed: int, fail: str | None = None) -> dict:
    rng = random.Random(seed)
    rows = []
    for index, speaker in enumerate(SPEAKERS):
        for key in TEXTS:
            for take in (42, 43):
                if speaker == fail:
                    rows.append({"speaker": speaker, "sentence": key, "seed": take, "error": "refused"})
                    continue
                rows.append({"speaker": speaker, "sentence": key, "seed": take,
                             "held_out_similarity": round(0.55 + 0.01 * (index % 5) + shift + rng.gauss(0, 0.02), 4),
                             "character_error_rate": max(0.0, 0.05 + rng.gauss(0, 0.01))})
    return {"label": label, "rows": rows, "summary": {"held_out_mean": 0.55 + shift}}


CANDIDATE = run("candidate", 0.0, seed=1, fail="ney06")
PUBLISHED = run("published", 0.0, seed=2, fail="ney06")


def test_an_arm_well_below_the_candidate_reads_as_the_candidate_above_it() -> None:
    result = arms.reading(CANDIDATE, PUBLISHED, {"pocket": run("pocket", -0.25, seed=3)}, "primary")
    pair = result["arms"]["pocket"]["candidate_minus_arm"]
    assert pair["low"] > 0
    assert result["arms"]["pocket"]["outcome"].startswith("the candidate is above pocket")


def test_an_arm_well_above_the_candidate_is_said_to_beat_it() -> None:
    result = arms.reading(CANDIDATE, PUBLISHED, {"f5": run("f5", 0.06, seed=4)}, "primary")
    assert result["arms"]["f5"]["candidate_minus_arm"]["high"] < 0
    assert "the adaptation is not the strongest thing measured here" in result["arms"]["f5"]["outcome"]


def test_an_arm_level_with_the_candidate_is_unresolved() -> None:
    result = arms.reading(CANDIDATE, PUBLISHED, {"f5": run("f5", 0.0, seed=1)}, "primary")
    assert "unresolved" in result["arms"]["f5"]["outcome"]


def test_the_candidates_left_out_speaker_is_left_out_of_every_pair() -> None:
    result = arms.reading(CANDIDATE, PUBLISHED, {"f5": run("f5", -0.1, seed=5),
                                                  "pocket": run("pocket", -0.2, seed=6)}, "primary")
    for arm in ("f5", "pocket"):
        pair = result["arms"][arm]["candidate_minus_arm"]
        assert pair["speakers"] == 26 and "ney06" in pair["left_out"]
    assert "cer_points_f5_minus_pocket" in result
