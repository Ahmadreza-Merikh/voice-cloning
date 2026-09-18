import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from voiceclone.evaluation import recipe

ROOT = Path(__file__).resolve().parent.parent


def cell(speaker, sentence, **metadata):
    return {"speaker": speaker, "sentence": sentence, "seed": 42, "metadata": metadata}


def test_constant_metadata_is_stated_once_and_what_varies_per_cell():
    cells = [cell("a", "s1", model="k2", revision="c5fd", native="x1.wav"),
             cell("b", "s1", model="k2", revision="c5fd", native="x2.wav")]
    constant, per_cell = recipe.split_metadata(cells)
    assert constant == {"model": "k2", "revision": "c5fd"}
    assert per_cell == [{"speaker": "a", "sentence": "s1", "seed": 42, "native": "x1.wav"},
                        {"speaker": "b", "sentence": "s1", "seed": 42, "native": "x2.wav"}]


def test_nothing_varying_means_no_per_cell_list():
    constant, per_cell = recipe.split_metadata([cell("a", "s1", dtype="float16"), cell("b", "s2", dtype="float16")])
    assert constant == {"dtype": "float16"}
    assert per_cell == []


@pytest.fixture
def made(tmp_path):
    results = tmp_path / "run.json"
    results.write_text(json.dumps({"summary": {"held_out_mean": 0.5}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    built = recipe.build(
        label="run", engine="omnivoice", engine_settings={"model_id": "k2-fsa/OmniVoice", "dtype": "float16"},
        engine_options={}, preprocess_options={}, cells=[cell("a", "s1", revision="c5fd")],
        references=[{"speaker": "a", "file": "a_000.wav", "sha256": "ab", "duration": 8.0,
                     "transcript_given": "متن"}],
        corpus_manifest=manifest, sentences={"s1": "جمله"}, seeds=[42], candidates=1, reference_index=0,
        reference_transcript="corpus", replay_from=None, scoring_environment={"id": "mac"},
        commit="abc123", results_path=results,
    )
    return results, built


def test_the_recipe_names_the_results_it_belongs_to(made):
    results, built = made
    assert built["results_sha256"] == recipe.sha256_file(results)
    assert built["render"]["reported_by_every_render"] == {"revision": "c5fd"}
    assert built["references"]["clips"][0]["sha256"] == "ab"
    assert built["session"]["code_commit"] == "abc123"


def test_a_recipe_is_written_once(made, tmp_path):
    results, built = made
    path = recipe.recipe_path(results)
    assert path.name == "run.recipe.json"
    recipe.write_once(path, built)
    recipe.write_once(path, built)
    changed = dict(built, seeds=[7])
    with pytest.raises(recipe.RecipeConflict):
        recipe.write_once(path, changed)
    assert json.loads(path.read_text())["seeds"] == [42]


def test_a_replay_points_at_the_recipe_of_the_run_that_rendered(tmp_path):
    original = tmp_path / "omnivoice.json"
    original.write_text("{}")
    recipe.recipe_path(original).write_text('{"label": "omnivoice"}\n')
    results = tmp_path / "omnivoice-local.json"
    results.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    built = recipe.build(
        label="omnivoice-local", engine="f5", engine_settings=None, engine_options={}, preprocess_options={},
        cells=[], references=[], corpus_manifest=manifest, sentences={}, seeds=[42], candidates=1,
        reference_index=0, reference_transcript="corpus", replay_from="omnivoice", scoring_environment=None,
        commit=None, results_path=results,
    )
    assert built["render"]["replayed_from"]["recipe"] == "omnivoice.recipe.json"
    assert built["render"]["replayed_from"]["recipe_sha256"] == recipe.sha256_file(recipe.recipe_path(original))


def test_the_benchmark_refuses_a_label_that_already_has_a_recipe(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules["benchmark"] = benchmark
    spec.loader.exec_module(benchmark)
    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)
    (tmp_path / "taken.recipe.json").write_text("{}")
    with pytest.raises(SystemExit, match="already has a recipe"):
        benchmark.run(Namespace(label="taken"))
