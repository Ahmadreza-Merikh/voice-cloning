"""The multi-reference manifest keeps its promises, checked on the file itself.

The manifest is committed so that what gets rendered is fixed before anything
is looked at. These tests are what stop a later edit - or a rebuild against a
changed corpus - from quietly reintroducing the problems it exists to rule out.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "eval/manifests/multiref_v1.json").read_text(encoding="utf-8"))
CORPUS = {e["file"]: e for e in json.loads((ROOT / "eval/corpus/manifest.json").read_text(encoding="utf-8"))["entries"]}


def prompt_files(entry: dict) -> set[str]:
    return {clip["file"] for condition in entry["conditions"].values() for clip in condition["clips"]}


def test_no_prompt_clip_is_ever_scored_against() -> None:
    for speaker, entry in MANIFEST["speakers"].items():
        assert not prompt_files(entry) & set(entry["evaluation_clips"]), speaker


def test_every_condition_shares_one_evaluation_pool() -> None:
    # The pool is stored once per speaker, not per condition, so conditions
    # cannot drift apart; this checks it is real and belongs to the speaker.
    for speaker, entry in MANIFEST["speakers"].items():
        pool = entry["evaluation_clips"]
        assert len(pool) >= 2 and len(set(pool)) == len(pool), speaker
        assert all(CORPUS[f]["speaker"] == speaker for f in pool), speaker


def test_every_clip_is_the_speakers_own_and_its_transcript_is_the_corpus_one() -> None:
    for speaker, entry in MANIFEST["speakers"].items():
        for condition in entry["conditions"].values():
            for clip in condition["clips"]:
                assert CORPUS[clip["file"]]["speaker"] == speaker
                assert clip["transcript"] == CORPUS[clip["file"]]["transcript"]


def test_the_equal_length_contrast_is_within_tolerance() -> None:
    for speaker, entry in MANIFEST["speakers"].items():
        single = entry["conditions"]["single-A"]["seconds"]
        multi = entry["conditions"]["multi-YZ"]["seconds"]
        assert abs(multi - single) / single <= MANIFEST["tolerance"], speaker


def test_multi_uses_two_different_recordings_and_starts_with_single_y() -> None:
    for speaker, entry in MANIFEST["speakers"].items():
        multi = [c["file"] for c in entry["conditions"]["multi-YZ"]["clips"]]
        assert len(set(multi)) == 2, speaker
        assert entry["conditions"]["single-Y"]["clips"][0]["file"] == multi[0], speaker
        assert entry["conditions"]["single-A"]["clips"][0]["file"] not in multi, speaker


def test_no_condition_exceeds_a_declared_engine_limit_unless_marked() -> None:
    limits = MANIFEST["engine_prompt_limits_declared"]
    for speaker, entry in MANIFEST["speakers"].items():
        for engine, limit in limits.items():
            if limit is None:
                continue
            over = sorted(n for n, c in entry["conditions"].items() if c["seconds"] > limit)
            assert over == entry["inapplicable_for_engine"].get(engine, []), (speaker, engine)


def test_the_committed_manifest_is_what_the_builder_produces() -> None:
    spec = importlib.util.spec_from_file_location("build_multiref", ROOT / "scripts/build_multiref_manifest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_multiref"] = module
    spec.loader.exec_module(module)
    assert module.build() == MANIFEST
