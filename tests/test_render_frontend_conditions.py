"""The front-end render script gets as far as the engine with any texts path, the engine stubbed out."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("render_frontend_conditions",
                                                  ROOT / "scripts" / "render_frontend_conditions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_frontend_conditions"] = module
    spec.loader.exec_module(module)
    return module


def test_a_relative_texts_path_reaches_the_engine_and_the_manifest_names_its_forms(tmp_path, monkeypatch):
    rfc = load()
    reached = {}

    class Studio:
        def __init__(self, config):
            reached["normaliser"] = list(config.text.marks_engines)
            reached["ezafe"] = config.text.ezafe.enabled

        def release(self):
            reached["released"] = True

    def render_group(studio, jobs, **kwargs):
        reached["jobs"] = len(jobs)

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(rfc.ab, "_check_engines_available", lambda names: None)
    monkeypatch.setattr(rfc.ab, "render_group", render_group)
    monkeypatch.setitem(sys.modules, "voiceclone.pipelines.studio", type(sys)("studio_stub"))
    sys.modules["voiceclone.pipelines.studio"].Studio = Studio
    assert rfc.main(["--texts", "eval/runs/frontend-listening-v2/texts.json", "--app-normaliser",
                     "--out", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["texts"]["file"] == "eval/runs/frontend-listening-v2/texts.json"
    assert list(manifest["conditions"]) == ["E", "EH", "ES", "EHS"]
    assert reached == {"normaliser": ["omnivoice"], "ezafe": False, "jobs": 48, "released": True}
