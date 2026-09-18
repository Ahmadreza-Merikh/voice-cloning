"""The recipe behind a results file, kept beside it and written once.

A results file says how renders scored. It does not say what produced them:
which weights at which revision, which codec, which precision and generation
settings, which reference clips, in which session. Without that a score cannot
be reproduced or even compared honestly, and the files written before this
module existed show it - a replayed run is labelled ``engine: f5`` because the
scorer, not the renderer, wrote the label.

So every benchmark run writes ``eval/results/<label>.recipe.json`` beside
``<label>.json``:

* ``render`` - the engine, its settings section as configured, the options
  the run overrode, and what the engine itself reported for every render
  (device, dtype, model and codec revisions, sampling values - whatever it
  knows), split into what was the same for every cell and what varied
* ``references`` - each speaker's conditioning clip: file, sha256, duration,
  and the transcript the engine was given
* ``corpus``, ``sentences``, ``seeds`` - the evaluation design, with the
  corpus manifest's sha256
* ``session`` - where it ran and what scored it; the code commit when known
* ``results_sha256`` - the results file this recipe belongs to

"Written once" is the point: a label that already has a recipe is refused, so
a results file cannot be silently replaced under an old recipe. A new run
needs a new label.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

RECIPE_VERSION = 1


class RecipeConflict(RuntimeError):
    """A recipe already exists for this label and says something else."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_path(results_path: Path) -> Path:
    return results_path.with_name(results_path.stem + ".recipe.json")


def code_commit(repo_root: Path) -> str | None:
    """The commit the code ran at: git when there is a checkout, else what a kernel was told."""
    configured = os.environ.get("VOICECLONE_CODE_COMMIT")
    if configured:
        return configured
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
                                capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root,
                           text=True, capture_output=True, timeout=10).stdout.strip()
    return f"{commit}+uncommitted changes" if dirty else commit


def split_metadata(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """What every render reported identically, and per cell what differed."""
    keys = sorted({key for cell in cells for key in cell["metadata"]})
    constant, varying = {}, set()
    for key in keys:
        values = {json.dumps(cell["metadata"].get(key), sort_keys=True, default=str) for cell in cells}
        if len(values) == 1:
            constant[key] = cells[0]["metadata"].get(key)
        else:
            varying.add(key)
    per_cell = [
        {"speaker": c["speaker"], "sentence": c["sentence"], "seed": c["seed"],
         **{key: c["metadata"].get(key) for key in sorted(varying)}}
        for c in cells if varying
    ]
    return constant, per_cell


def build(
    *,
    label: str,
    engine: str,
    engine_settings: dict[str, Any] | None,
    engine_options: dict[str, Any],
    preprocess_options: dict[str, Any],
    cells: list[dict[str, Any]],
    references: list[dict[str, Any]],
    corpus_manifest: Path,
    sentences: dict[str, str],
    seeds: list[int],
    candidates: int,
    reference_index: int,
    reference_transcript: str,
    replay_from: str | None,
    scoring_environment: dict[str, Any] | None,
    commit: str | None,
    results_path: Path,
) -> dict[str, Any]:
    constant, per_cell = split_metadata(cells)
    replayed = None
    if replay_from:
        source = recipe_path(results_path.with_name(f"{replay_from}.json"))
        replayed = {
            "label": replay_from,
            "recipe": source.name if source.is_file() else None,
            "recipe_sha256": sha256_file(source) if source.is_file() else None,
            "note": "the renders were produced by that run; this one only re-scored them",
        }
    return {
        "recipe_version": RECIPE_VERSION,
        "label": label,
        "results_file": results_path.name,
        "results_sha256": sha256_file(results_path),
        "render": {
            "engine": engine,
            "replayed_from": replayed,
            "settings": engine_settings,
            "options_overridden": engine_options,
            "preprocess_overridden": preprocess_options,
            "candidates_per_cell": candidates,
            "reported_by_every_render": constant,
            "reported_per_cell": per_cell,
        },
        "references": {
            "index": reference_index,
            "transcript": reference_transcript,
            "clips": references,
        },
        "corpus": {"manifest": str(corpus_manifest.name), "sha256": sha256_file(corpus_manifest)},
        "sentences": sentences,
        "seeds": seeds,
        "session": {
            "scoring_environment": scoring_environment,
            "code_commit": commit,
        },
    }


def write_once(path: Path, recipe: dict[str, Any]) -> Path:
    """Write the recipe, or leave an identical one alone; anything else is refused."""
    text = json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") == text:
            return path
        raise RecipeConflict(f"{path.name} already records a different run; use a new label")
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["RECIPE_VERSION", "RecipeConflict", "build", "code_commit", "recipe_path",
           "sha256_file", "split_metadata", "write_once"]
