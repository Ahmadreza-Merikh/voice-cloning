"""The listening test is blind, fair, and scored from real, valid answers only.

Built from synthetic renders so nothing here depends on a GPU run. What is
checked is what would quietly break the test if it failed: a system name
reaching the listener, a prompt clip standing in for "the real speaker",
unbalanced order, unequal loudness - and, in scoring, an empty answer counted
as "nothing misread", an invalid value counted as a vote, one person counted
twice, and an interval that ignores shared listeners.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load("build_listening_test")
scorer = load("score_listening_test")
CORPUS = json.loads((ROOT / "eval/corpus/manifest.json").read_text(encoding="utf-8"))
MULTIREF = json.loads((ROOT / "eval/manifests/multiref_v1.json").read_text(encoding="utf-8"))
RUN = {"label": "fake", "config": {"reference_index": 0}}


def renders(folder: Path, level: float, speakers=("cv02", "cv03", "cv07", "cv08", "cv09")) -> Path:
    folder.mkdir(parents=True)
    rate = 24000
    t = np.arange(rate * 2) / rate
    for speaker in speakers:
        for sentence in ("s1", "s2"):
            tone = level * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
            sf.write(folder / f"{speaker}_{sentence}_seed42.wav", tone.astype(np.float32), rate)
    return folder


def build(tmp_path: Path, **provenance) -> tuple[dict, Path]:
    a = renders(tmp_path / "system-alpha-renders", 0.05)
    b = renders(tmp_path / "system-beta-renders", 0.4)
    out = tmp_path / "listening"
    provenance = provenance or {"a_results": RUN, "b_results": RUN}
    return builder.build("t", a, b, seed=7, out_root=out, **provenance), out


def answers_file(key: dict, listener: str, answer_for) -> tuple[str, bytes]:
    body = {"test": key["name"], "fingerprint": key["fingerprint"], "listener": listener,
            "answers": {tid: answer_for(trial) for tid, trial in key["trials"].items()}}
    return f"{listener or 'anon'}.json", json.dumps(body).encode()


class TestBuilding:
    def test_nothing_the_listener_receives_names_a_system(self, tmp_path: Path) -> None:
        key, out = build(tmp_path)
        served = out / "t"
        page = (served / "index.html").read_text(encoding="utf-8")
        for leak in ("alpha", "beta", "system-alpha-renders", "system-beta-renders", "seed42", "cv02"):
            assert leak not in page, leak
            assert not any(leak in p.name for p in (served / "audio").iterdir()), leak
        assert (out / "t.key.json").is_file() and not list(served.rglob("*.key.json"))

    def test_without_provenance_it_refuses(self, tmp_path: Path) -> None:
        a = renders(tmp_path / "a", 0.1)
        b = renders(tmp_path / "b", 0.1)
        with pytest.raises(SystemExit):
            builder.build("t", a, b, seed=1, out_root=tmp_path / "out")

    def test_the_reference_was_never_either_runs_prompt(self, tmp_path: Path) -> None:
        key, _ = build(tmp_path)
        prompts = builder.prompts_from_results(RUN, CORPUS)
        for trial in key["trials"].values():
            assert trial["reference"] not in prompts[trial["speaker"]]

    def test_with_an_experiment_manifest_the_reference_comes_from_its_pool(self, tmp_path: Path) -> None:
        # In the multi-reference manifest some clips the corpus calls held-out
        # are prompts; the reference must avoid them and come from the pool.
        speakers = tuple(MULTIREF["speakers"])
        a = renders(tmp_path / "a", 0.05, speakers)
        b = renders(tmp_path / "b", 0.3, speakers)
        key = builder.build("m", a, b, seed=3, out_root=tmp_path / "out", pool_manifest=MULTIREF)
        prompts, pools = builder.pools_from_manifest(MULTIREF)
        for trial in key["trials"].values():
            assert trial["reference"] in pools[trial["speaker"]]
            assert trial["reference"] not in prompts[trial["speaker"]]

    def test_order_is_balanced_and_loudness_is_matched(self, tmp_path: Path) -> None:
        from voiceclone.audio.io import load_audio
        from voiceclone.audio.loudness import measure_lufs

        key, out = build(tmp_path)
        firsts = [t["sample_1"] for t in key["trials"].values()]
        assert abs(firsts.count("a") - firsts.count("b")) <= 1
        levels = [measure_lufs(load_audio(p, mono=True)) for p in (out / "t" / "audio").glob("*.wav")]
        levels = [v for v in levels if v is not None]
        assert max(levels) - min(levels) < 0.5  # rendered 18 dB apart


class TestScoring:
    def test_answers_are_mapped_through_the_key(self, tmp_path: Path) -> None:
        key, _ = build(tmp_path)
        file = answers_file(key, "p1", lambda t: {"same": "1" if t["sample_1"] == "a" else "2",
                                                  "natural": "0", "misread": ["none"]})
        report = scorer.score(key, [file])
        assert report["counts"]["same"]["a"] == len(key["trials"]) and report["counts"]["same"]["b"] == 0
        assert report["misread"] == {"a": 0, "b": 0, "answered": len(key["trials"]), "unanswered": 0,
                                     "invalid": 0}

    def test_an_empty_answer_is_unanswered_not_nothing_misread(self, tmp_path: Path) -> None:
        key, _ = build(tmp_path)
        file = answers_file(key, "p1", lambda t: {"same": None, "natural": None, "misread": []})
        report = scorer.score(key, [file])
        assert report["misread"]["answered"] == 0
        assert report["misread"]["unanswered"] == len(key["trials"])
        assert report["counts"]["same"]["unanswered"] == len(key["trials"])

    def test_an_invalid_value_is_never_a_vote(self, tmp_path: Path) -> None:
        key, _ = build(tmp_path)
        trial = next(iter(key["trials"].values()))
        assert scorer.preference("invalid", trial) == "invalid"
        file = answers_file(key, "p1", lambda t: {"same": "3", "natural": "yes", "misread": ["none", "1"]})
        report = scorer.score(key, [file])
        assert report["counts"]["same"]["a"] == report["counts"]["same"]["b"] == 0
        assert report["counts"]["same"]["invalid"] == len(key["trials"])
        assert report["misread"]["invalid"] == len(key["trials"])

    def test_a_response_to_another_test_is_rejected(self, tmp_path: Path) -> None:
        key, _ = build(tmp_path)
        name, raw = answers_file(key, "p1", lambda t: {"same": "1", "natural": "1", "misread": ["none"]})
        body = json.loads(raw)
        body["fingerprint"] = "0" * 16
        report = scorer.score(key, [(name, json.dumps(body).encode())])
        assert name in report["intake"]["rejected"]
        assert report["counts"]["same"]["a"] + report["counts"]["same"]["b"] == 0

    def test_one_person_is_not_counted_twice(self, tmp_path: Path) -> None:
        key, _ = build(tmp_path)
        answer = lambda t: {"same": "1", "natural": "1", "misread": ["none"]}  # noqa: E731
        name, raw = answers_file(key, "p1", answer)
        again = answers_file(key, "p1", lambda t: {"same": "2", "natural": "2", "misread": ["none"]})
        report = scorer.score(key, [(name, raw), ("copy.json", raw), again])
        assert report["intake"]["duplicate_files"] == ["copy.json"]
        assert len(report["intake"]["duplicate_listeners"]) == 1
        assert report["intake"]["distinct_listener_ids"] == 1
        assert sum(report["counts"]["same"][k] for k in ("a", "b", "none")) == len(key["trials"])

    def test_the_crossed_interval_needs_enough_listeners(self) -> None:
        few = {(f"s{i}", f"l{j}"): [1, -1, 1] for i in range(6) for j in range(3)}
        assert scorer.crossed_bootstrap(few) is None
        rng = np.random.default_rng(1)
        many = {(f"s{i}", f"l{j}"): list(rng.choice([-1, 0, 1], 4)) for i in range(6) for j in range(6)}
        mean, low, high = scorer.crossed_bootstrap(many, draws=300)
        assert low <= mean <= high
