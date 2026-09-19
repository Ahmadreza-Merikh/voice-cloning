"""The sealed test's reading, applied to made-up results before any real one exists.

These pin the rule the roadmap fixed: H0 of no improvement is rejected only if
the lower end of the speaker-level interval is above 0; a CER interval above
zero counts against the adaptation; runs scored in two environments are not
judged; a speaker with a missing cell is left out and listed.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("read_sealed_test", ROOT / "scripts" / "read_sealed_test.py")
sealed = importlib.util.module_from_spec(spec)
sys.modules["read_sealed_test"] = sealed
spec.loader.exec_module(sealed)

CATEGORIES = sealed.categories_of()
SPEAKERS = [f"ney{i:02d}" for i in range(1, 29)]


def run(label: str, shift: float, cer_shift: float = 0.0, place: str = "kaggle:test", seed: int = 1,
        drop: tuple | None = None) -> dict:
    rng = random.Random(seed)
    rows = []
    for index, speaker in enumerate(SPEAKERS):
        base = 0.45 + 0.01 * (index % 7)
        effect = shift + rng.gauss(0, 0.03)
        for key in CATEGORIES:
            for take in (42, 43):
                if drop == (speaker, key, take):
                    rows.append({"speaker": speaker, "sentence": key, "seed": take, "error": "worker died"})
                    continue
                rows.append({"speaker": speaker, "sentence": key, "seed": take,
                             "held_out_similarity": round(base + effect + rng.gauss(0, 0.02), 4),
                             "character_error_rate": max(0.0, 0.03 + cer_shift + rng.gauss(0, 0.01))})
    return {"label": label, "rows": rows, "scoring_environment": {"id": place}}


PUBLISHED = run("published", 0.0, seed=7)


def reading_of(candidate: dict, role: str = "primary") -> dict:
    return sealed.reading("candidate", candidate, "published", PUBLISHED, role, CATEGORIES)


def offset(data: dict, similarity: float, cer: float = 0.0, label: str = "candidate") -> dict:
    rows = []
    for row in data["rows"]:
        row = dict(row)
        if "error" not in row:
            row["held_out_similarity"] = round(row["held_out_similarity"] + similarity, 4)
            row["character_error_rate"] += cer
        rows.append(row)
    return {**data, "label": label, "rows": rows}


def test_a_clear_improvement_rejects_h0() -> None:
    result = reading_of(offset(PUBLISHED, 0.04))
    assert result["h0_rejected"] is True
    assert result["held_out_similarity"]["low"] > 0
    assert result["verdict"].startswith("higher held-out similarity")
    assert result["cer_counts_against"] is False


def test_no_difference_is_no_detectable_improvement() -> None:
    result = reading_of(run("candidate", 0.0, seed=8))
    assert result["h0_rejected"] is False
    assert "no detectable improvement" in result["verdict"] or "lower" in result["verdict"]


def test_a_cer_interval_above_zero_counts_against_whatever_the_similarity() -> None:
    result = reading_of(offset(PUBLISHED, 0.04, cer=0.02))
    assert result["h0_rejected"] is True
    assert result["cer_counts_against"] is True
    assert "counts against the adaptation" in result["verdict"]
    assert result["cer_percentage_points"]["low"] > 0


def test_runs_scored_in_two_environments_are_not_judged() -> None:
    candidate = offset(PUBLISHED, 0.04)
    candidate["scoring_environment"] = {"id": "mac"}
    result = reading_of(candidate)
    assert result["h0_rejected"] is False
    assert result["verdict"].startswith("not judged")


def test_a_speaker_with_a_failed_cell_is_left_out_and_listed() -> None:
    candidate = offset(run("candidate", 0.0, seed=7, drop=("ney05", "t03", 43)), 0.04)
    result = reading_of(candidate)
    assert result["held_out_similarity"]["speakers"] == 27
    assert list(result["held_out_similarity"]["left_out"]) == ["ney05"]


def test_the_secondary_set_is_labelled() -> None:
    assert reading_of(offset(PUBLISHED, 0.04), role="secondary")["verdict"].startswith("secondary set:")


def test_the_one_sided_p_is_the_t_tests() -> None:
    from scipy.stats import ttest_1samp

    values = [0.01, 0.03, -0.02, 0.05, 0.04, 0.0, 0.02]
    assert sealed.one_sided_p(values) == pytest.approx(ttest_1samp(values, 0.0, alternative="greater").pvalue)
    assert sealed.two_sided_p(values) == pytest.approx(ttest_1samp(values, 0.0).pvalue)


def test_holm_steps_down_and_stays_monotone() -> None:
    assert sealed.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert sealed.holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_the_categories_are_read_from_the_written_texts() -> None:
    assert sorted(set(CATEGORIES.values())) == sorted(
        {"formal", "conversational", "numbers", "names_and_loanwords", "homographs_and_ezafe", "very_short", "long"})
    exploratory = reading_of(offset(PUBLISHED, 0.04))["exploratory"]
    assert exploratory["label"].startswith("exploratory")
    assert all("p_holm" in row for row in exploratory["categories"].values())


def test_the_planning_numbers_are_the_power_designs() -> None:
    design = json.loads((ROOT / "eval" / "manifests" / "power_design_v1.json").read_text())
    assert design["alpha_one_sided"] == sealed.ALPHA_ONE_SIDED
    assert round(design["spread"]["pooled_sd"], 2) == sealed.PLANNED_SPREAD
    assert round(design["spread"]["interval_95"][1], 3) == sealed.SPREAD_UPPER
