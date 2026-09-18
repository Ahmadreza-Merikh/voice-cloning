import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("power_design", ROOT / "scripts" / "power_design.py")
power_design = importlib.util.module_from_spec(spec)
sys.modules["power_design"] = power_design
spec.loader.exec_module(power_design)

pytest.importorskip("scipy")


@pytest.mark.parametrize("n", [5, 30, 200])
def test_an_effect_equal_to_the_margin_has_power_equal_to_alpha(n):
    assert power_design.power(n, 0.03, 0.03, 0.05) == pytest.approx(power_design.ALPHA, abs=1e-9)


def test_no_size_is_offered_when_the_effect_does_not_exceed_the_margin():
    assert power_design.speakers_needed(0.03, 0.03, 0.05) is None
    assert power_design.speakers_needed(0.02, 0.03, 0.05) is None


def test_the_old_zero_margin_table_is_reproduced():
    # The roadmap's earlier table: sd 0.048, 80% power, two-sided 0.05.
    assert power_design.speakers_needed(0.04, 0.0, 0.048) == 14
    assert power_design.speakers_needed(0.03, 0.0, 0.048) == 23
    assert power_design.speakers_needed(0.02, 0.0, 0.048) == 48


def test_a_margin_costs_speakers_and_only_the_gap_to_it_matters():
    plain = power_design.speakers_needed(0.05, 0.0, 0.05)
    with_margin = power_design.speakers_needed(0.05, 0.0205, 0.05)
    assert with_margin > plain
    assert power_design.speakers_needed(0.08, 0.03, 0.05) == power_design.speakers_needed(0.05, 0.0, 0.05)


def test_the_spread_interval_brackets_the_estimate():
    low, high = power_design.spread_interval(0.048, 8)
    assert low < 0.048 < high
    assert low == pytest.approx(0.0324, abs=5e-4)
    assert high == pytest.approx(0.0920, abs=5e-4)
