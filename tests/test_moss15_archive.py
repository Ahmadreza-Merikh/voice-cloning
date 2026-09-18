"""The MOSS v1.5 native archive never loses a render and says what made each one.

A review found the archive keyed only on text, seed and reference, so a smoke
render and the full run's render of the same cell shared one file name, and a
change of settings would have written over the earlier result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceclone.config import AppConfig
from voiceclone.device import get_device_spec
from voiceclone.engines.moss15_tts import Moss15Engine


def engine(config: AppConfig, tmp_path: Path, session: str) -> Moss15Engine:
    configured = config.copy_with(**{"engines.moss15.native_dir": str(tmp_path / "native")})
    e = Moss15Engine(configured, get_device_spec("cpu"))
    e._ready = {"model_revision": "m1", "codec_revision": "c1", "dtype": "bfloat16",
                "attention": "sdpa", "codec_weight_dtype": "bfloat16"}
    e._session = session
    return e


def files(tmp_path: Path) -> tuple[Path, Path]:
    render, reference = tmp_path / "render.wav", tmp_path / "ref.wav"
    sf.write(render, np.zeros((480, 2), dtype=np.float32) + 0.1, 48000, subtype="FLOAT")
    sf.write(reference, np.zeros(480, dtype=np.float32) + 0.2, 24000)
    return render, reference


def job(**changes) -> dict:
    base = {"language": "Persian (Farsi)", "audio_temperature": 1.7, "audio_top_p": 0.8, "audio_top_k": 25,
            "audio_repetition_penalty": 1.0, "max_new_tokens": 4096}
    return {**base, **changes}


def test_the_same_cell_twice_keeps_both(config: AppConfig, tmp_path: Path) -> None:
    render, reference = files(tmp_path)
    e = engine(config, tmp_path, "s1")
    first = e._keep_native(render, "text", 42, reference, {"sample_rate": 48000}, job())
    second = e._keep_native(render, "text", 42, reference, {"sample_rate": 48000}, job())
    assert first != second and first.exists() and second.exists()
    lines = (first.parent / "index.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(line)["sha256"] for line in lines)


def test_a_different_recipe_is_a_different_file(config: AppConfig, tmp_path: Path) -> None:
    render, reference = files(tmp_path)
    e = engine(config, tmp_path, "s1")
    a = e._keep_native(render, "text", 42, reference, {}, job())
    b = e._keep_native(render, "text", 42, reference, {}, job(audio_temperature=1.0))
    assert a.stem.split("-")[0] != b.stem.split("-")[0]
    assert json.loads((a.parent / "index.jsonl").read_text().splitlines()[1])["recipe"]["audio_temperature"] == 1.0


def test_sessions_do_not_share_a_directory(config: AppConfig, tmp_path: Path) -> None:
    render, reference = files(tmp_path)
    a = engine(config, tmp_path, "run-a")._keep_native(render, "text", 42, reference, {}, job())
    b = engine(config, tmp_path, "run-b")._keep_native(render, "text", 42, reference, {}, job())
    assert a.parent != b.parent and a.exists() and b.exists()
