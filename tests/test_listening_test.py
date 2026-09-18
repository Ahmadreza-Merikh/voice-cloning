"""The listening test is blind, fair, and scored from real answers only.

Built from synthetic renders so nothing here depends on a GPU run. What is
checked is what would quietly break the test if it failed: a system name
reaching the listener, the prompt clip standing in for "the real speaker",
unbalanced order, unequal loudness, and answers mapped to the wrong system.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


def renders(folder: Path, level: float) -> Path:
    folder.mkdir(parents=True)
    rate = 24000
    t = np.arange(rate * 2) / rate
    for speaker in ("cv02", "cv03", "cv07", "cv08", "cv09"):
        for sentence in ("s1", "s2"):
            tone = level * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
            sf.write(folder / f"{speaker}_{sentence}_seed42.wav", tone.astype(np.float32), rate)
    return folder


def build(tmp_path: Path) -> tuple[dict, Path]:
    a = renders(tmp_path / "system-alpha-renders", 0.05)
    b = renders(tmp_path / "system-beta-renders", 0.4)
    out = tmp_path / "listening"
    key = builder.build("t", a, b, seed=7, out_root=out)
    return key, out


def test_nothing_the_listener_receives_names_a_system(tmp_path: Path) -> None:
    key, out = build(tmp_path)
    served = out / "t"
    page = (served / "index.html").read_text(encoding="utf-8")
    for leak in ("alpha", "beta", "system-alpha-renders", "system-beta-renders", "seed42", "cv02"):
        assert leak not in page, leak
        assert not any(leak in p.name for p in (served / "audio").iterdir()), leak
    # The key sits beside the served folder, not in it.
    assert (out / "t.key.json").is_file() and not list(served.rglob("*.key.json"))


def test_the_reference_is_a_held_out_clip_never_the_prompt(tmp_path: Path) -> None:
    key, _ = build(tmp_path)
    roles = {e["file"]: e["role"] for e in CORPUS["entries"]}
    for trial in key["trials"].values():
        assert roles[trial["reference"]] == "held_out"


def test_order_is_balanced_and_loudness_is_matched(tmp_path: Path) -> None:
    from voiceclone.audio.io import load_audio
    from voiceclone.audio.loudness import measure_lufs

    key, out = build(tmp_path)
    firsts = [t["sample_1"] for t in key["trials"].values()]
    assert abs(firsts.count("a") - firsts.count("b")) <= 1
    levels = [measure_lufs(load_audio(p, mono=True)) for p in (out / "t" / "audio").glob("*.wav")]
    levels = [v for v in levels if v is not None]
    assert max(levels) - min(levels) < 0.5
    # The two systems were rendered 18 dB apart; the listener must not hear that.


def test_answers_are_mapped_through_the_key(tmp_path: Path) -> None:
    key, _ = build(tmp_path)
    answers = {}
    for trial_id, trial in key["trials"].items():
        # This listener always picks whichever sample is system a.
        pick = "1" if trial["sample_1"] == "a" else "2"
        answers[trial_id] = {"same": pick, "natural": "0", "misread": []}
    report = scorer.score(key, [{"listener": "x", "answers": answers}])
    assert report["counts"]["same"]["a"] == len(key["trials"]) and report["counts"]["same"]["b"] == 0
    assert report["counts"]["natural"]["none"] == len(key["trials"])
    # Every speaker at +1 with no spread: no interval is invented.
    assert "do not vary" in str(report["intervals"]["same"])
