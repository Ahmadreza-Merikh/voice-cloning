"""The comparison tool refuses to manufacture certainty.

Each test is a failure a reviewer reproduced, or one that sits next to it:
an interval printed from one speaker, a table lookup that quietly became 1.96,
a failed render dropped as if it had never been asked for, and a scale control
that declared every cell equal because the means were.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT)
compare_runs = importlib.util.module_from_spec(spec)
# dataclasses resolves the module's annotations through sys.modules, so the
# module has to be registered before it runs.
sys.modules["compare_runs"] = compare_runs
spec.loader.exec_module(compare_runs)


def run(label: str, scores: dict[tuple[str, str], float | None], errors: set = frozenset()) -> dict:
    rows = []
    for (speaker, sentence), value in scores.items():
        row = {"speaker": speaker, "sentence": sentence, "seed": 42,
               "held_out_similarity": value, "character_error_rate": 0.02}
        if (speaker, sentence) in errors:
            row["error"] = "SynthesisError: boom"
            row["held_out_similarity"] = None
        rows.append(row)
    return {"label": label, "rows": rows, "summary": {}}


def grid(speakers: int, sentences: int, value) -> dict[tuple[str, str], float]:
    return {(f"sp{i:02d}", f"s{j}"): value(i, j) for i in range(speakers) for j in range(sentences)}


class TestIntervals:
    def test_one_speaker_gets_no_interval_however_many_cells(self) -> None:
        # The reviewer's case: six texts from one speaker, .55 against .50.
        left = run("a", grid(1, 6, lambda i, j: 0.55))
        right = run("b", grid(1, 6, lambda i, j: 0.50))
        pair = compare_runs.compare("a", left, "b", right, compare_runs.SIMILARITY)
        assert pair.low is None and pair.high is None
        assert "1 complete speakers" in pair.note
        assert "clear of" not in compare_runs.describe(pair, compare_runs.PRACTICAL_REFERENCE)

    def test_speakers_that_do_not_vary_get_no_zero_width_interval(self) -> None:
        left = run("a", grid(8, 3, lambda i, j: 0.55))
        right = run("b", grid(8, 3, lambda i, j: 0.50))
        pair = compare_runs.compare("a", left, "b", right, compare_runs.SIMILARITY)
        assert pair.low is None
        assert "zero width" in pair.note

    def test_the_quantile_is_exact_beyond_twenty_degrees_of_freedom(self) -> None:
        # The old table fell back to 1.96 for df > 20, which is the normal
        # quantile - narrower than t at every finite df.
        for df in (21, 30, 45, 120):
            assert compare_runs.t_quantile(df) > 1.96
        assert compare_runs.t_quantile(8) == pytest.approx(2.306, abs=1e-3)

    def test_a_real_difference_across_enough_speakers_gets_an_interval(self) -> None:
        left = run("a", grid(9, 3, lambda i, j: 0.55 + 0.01 * ((i * 7) % 5)))
        right = run("b", grid(9, 3, lambda i, j: 0.50))
        pair = compare_runs.compare("a", left, "b", right, compare_runs.SIMILARITY)
        assert pair.low is not None and pair.low > 0
        assert len(pair.speakers_complete) == 9


class TestNothingIsDroppedQuietly:
    def test_a_failed_render_excludes_its_speaker_and_says_so(self) -> None:
        scores = grid(9, 3, lambda i, j: 0.55 + 0.01 * i)
        left = run("a", scores, errors={("sp03", "s1")})
        right = run("b", grid(9, 3, lambda i, j: 0.50 + 0.01 * i))
        pair = compare_runs.compare("a", left, "b", right, compare_runs.SIMILARITY)
        assert "sp03" not in pair.speakers_complete
        assert "failed" in pair.speakers_excluded["sp03"]
        assert pair.paired_cells == 24 and pair.design_cells == 27

    def test_a_non_finite_score_is_caught(self) -> None:
        scores = grid(9, 3, lambda i, j: 0.55)
        scores[("sp05", "s2")] = math.nan
        left = run("a", scores)
        right = run("b", grid(9, 3, lambda i, j: 0.50))
        pair = compare_runs.compare("a", left, "b", right, compare_runs.SIMILARITY)
        assert "non-finite" in pair.speakers_excluded["sp05"]

    def test_a_cell_one_run_never_had_is_reported(self) -> None:
        scores = grid(9, 3, lambda i, j: 0.55)
        del scores[("sp07", "s0")]
        pair = compare_runs.compare("a", run("a", scores), "b",
                                    run("b", grid(9, 3, lambda i, j: 0.50)),
                                    compare_runs.SIMILARITY)
        assert "lacks the cell" in pair.speakers_excluded["sp07"]

    def test_a_duplicated_key_refuses_the_comparison(self) -> None:
        data = run("a", grid(2, 2, lambda i, j: 0.5))
        data["rows"].append(dict(data["rows"][0]))
        with pytest.raises(compare_runs.DuplicateKeyError):
            compare_runs.cells(data, compare_runs.SIMILARITY)


class TestScaleControl:
    def test_equal_means_are_not_equal_cells(self) -> None:
        # Opposite differences cancel in the mean.
        reference = run("base", {("a", "s1"): 0.50, ("b", "s1"): 0.60})
        control = run("ctl", {("a", "s1"): 0.51, ("b", "s1"): 0.59})
        check = compare_runs.scale_control(reference, control)
        assert len(check["unequal"]) == 2
        assert check["max_abs"] == pytest.approx(0.01)

    def test_mismatched_keys_are_reported_not_ignored(self) -> None:
        reference = run("base", {("a", "s1"): 0.50, ("b", "s1"): 0.60})
        control = run("ctl", {("a", "s1"): 0.50})
        check = compare_runs.scale_control(reference, control)
        assert check["only_reference"] == [("b", "s1", 42)]

    def test_identical_cells_are_reported_as_identical(self) -> None:
        scores = {("a", "s1"): 0.5012, ("b", "s1"): 0.6034}
        check = compare_runs.scale_control(run("base", scores), run("ctl", scores))
        assert not check["unequal"] and check["shared"] == 2


class TestScoringEnvironment:
    def test_a_recorded_environment_wins(self, tmp_path: Path) -> None:
        data = {"label": "x", "scoring_environment": {"id": "kaggle:k@1"}, "rows": []}
        assert compare_runs.scoring_environment(data, root=tmp_path) == "kaggle:k@1"

    def test_the_provenance_map_covers_older_files(self, tmp_path: Path) -> None:
        (tmp_path / "_provenance.json").write_text(
            '{"runs": {"old": {"environment": "mac:project-venv", "evidence": "."}}}')
        assert compare_runs.scoring_environment({"label": "old"}, root=tmp_path) == "mac:project-venv"

    def test_device_alone_is_not_an_environment(self, tmp_path: Path) -> None:
        # Every Kaggle kernel records cuda, and each is its own install.
        data = {"label": "new", "config": {"device": "cuda"}, "rows": []}
        assert compare_runs.scoring_environment(data, root=tmp_path) is None

    def test_every_results_file_in_the_repo_has_a_scoring_environment(self) -> None:
        import json

        root = compare_runs.RESULTS
        for path in sorted(root.glob("*.json")):
            if path.name.startswith("_"):
                continue
            data = json.loads(path.read_text())
            if "rows" in data:
                assert compare_runs.scoring_environment(data) is not None, path.name
