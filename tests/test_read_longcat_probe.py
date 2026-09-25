"""The LongCat probe's reading, applied to made-up results before any real one exists.

These pin the rule fixed before the run: LongCat is "better" only if the whole
speaker-level interval of its held-out similarity minus OmniVoice's is above
the practical line; an arm with too many renders that are not the requested
English is inconclusive; runs scored in two environments are not judged.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("read_longcat_probe", ROOT / "scripts" / "read_longcat_probe.py")
probe = importlib.util.module_from_spec(spec)
sys.modules["read_longcat_probe"] = probe
spec.loader.exec_module(probe)

TEXTS = {t["key"]: t["text"] for t in json.loads(
    (ROOT / "eval" / "manifests" / "probe_texts_en_v1.json").read_text(encoding="utf-8"))["texts"]}
SPEAKERS = [f"nvl{i:02d}" for i in range(1, 26)]


def run(label: str, shift: float, prompt_gap: float = 0.15, place: str = "kaggle:probe", seed: int = 1,
        garbled: int = 0) -> dict:
    rng = random.Random(seed)
    rows = []
    for index, speaker in enumerate(SPEAKERS):
        base = 0.50 + 0.01 * (index % 7)
        effect = shift + rng.gauss(0, 0.02)
        for key, text in TEXTS.items():
            held = base + effect + rng.gauss(0, 0.01)
            heard = "zzz qqq" if len(rows) < garbled else text
            rows.append({"speaker": speaker, "sentence": key, "seed": 42, "held_out_similarity": held,
                         "reference_similarity": held + prompt_gap, "expected": text, "transcription": heard,
                         "character_error_rate": 0.02})
    mean = sum(r["held_out_similarity"] for r in rows) / len(rows)
    return {"label": label, "rows": rows, "scoring_environment": {"id": place},
            "summary": {"held_out_mean": mean, "cer_mean": 0.02, "untranscribed": 0}}


def read(longcat: dict, omnivoice: dict, secondary: dict | None = None) -> dict:
    runs = {"omni": omnivoice, "lc1b": longcat, **({"lc35": secondary} if secondary else {})}
    return probe.reading(runs, "omni", "lc1b", "lc35" if secondary else None)


def test_a_large_consistent_advantage_is_worth_costing() -> None:
    result = read(run("lc1b", 0.08, seed=2), run("omni", 0.0, seed=3))
    assert result["primary"]["clears_practical_line"] is True
    assert result["decision"]["adaptation_worth_costing"] is True
    assert result["primary"]["verdict"].startswith("better by more than the practical line")


def test_no_advantage_is_not_better_by_the_line() -> None:
    result = read(run("lc1b", 0.0, seed=2), run("omni", 0.0, seed=3))
    assert result["decision"]["adaptation_worth_costing"] is False
    assert result["primary"]["verdict"].startswith(("not better by the practical line", "not resolved"))


def test_a_small_real_advantage_does_not_pay_for_an_adaptation() -> None:
    result = read(run("lc1b", 0.02, seed=2), run("omni", 0.0, seed=3))
    facts = result["primary"]["held_out_similarity"]
    assert facts["low"] > 0
    assert result["decision"]["adaptation_worth_costing"] is False


def test_a_worse_engine_is_called_worse() -> None:
    result = read(run("lc1b", -0.08, seed=2), run("omni", 0.0, seed=3))
    assert result["primary"]["verdict"].startswith("worse than the other")


def test_renders_that_are_not_the_text_make_the_arm_inconclusive() -> None:
    result = read(run("lc1b", 0.08, seed=2, garbled=20), run("omni", 0.0, seed=3))
    assert result["primary"]["guard"]["lc1b"]["not_the_text"] == 20
    assert result["primary"]["verdict"].startswith("inconclusive")
    assert result["decision"]["adaptation_worth_costing"] is False


def test_runs_scored_in_two_places_are_not_judged() -> None:
    result = read(run("lc1b", 0.08, seed=2, place="kaggle:a"), run("omni", 0.0, seed=3, place="kaggle:b"))
    assert result["primary"]["verdict"].startswith("not judged")
    assert result["decision"]["adaptation_worth_costing"] is False


def test_the_secondary_arm_is_reported_beside_the_primary_and_not_as_the_decision() -> None:
    result = read(run("lc1b", 0.0, seed=2), run("omni", 0.0, seed=3), run("lc35", 0.08, seed=4))
    assert result["decision"]["adaptation_worth_costing"] is False
    assert result["decision"]["secondary_3_5b_clears_the_line"] is True
    assert result["secondary"]["against_the_1b"]["clears_practical_line"] is True


def test_a_wider_prompt_gap_shows_in_the_exploratory_reading() -> None:
    result = read(run("lc1b", 0.0, prompt_gap=0.25, seed=2), run("omni", 0.0, prompt_gap=0.15, seed=3))
    gap = result["exploratory_prompt_copying"]["minus_omni"]["lc1b"]
    assert abs(gap["mean"] - 0.10) < 1e-9


def test_english_wer_ignores_case_and_punctuation() -> None:
    assert probe.english_wer("Please close the window.", "please close the window") == 0.0
    assert probe.english_wer("Please close the window.", "please close window") == 0.25
    assert probe.english_wer("", "anything") is None
