"""The OmniVoice engine waits for its worker with a deadline, and recovers from losing it.

A render in the sealed test session is one of about 2,800; the benchmark writes
its results only at the end. Before this, a worker that stopped answering held
the whole run until the notebook's own timeout, losing every cell, and a worker
that died made every later cell fail on a closed pipe. A stand-in worker - this
interpreter, the same protocol, no model - shows both now cost one cell.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from voiceclone.audio.io import Audio
from voiceclone.device import DeviceSpec
from voiceclone.engines import omnivoice_tts
from voiceclone.errors import SynthesisError

WORKER = textwrap.dedent('''
    import json, sys, time, wave

    arguments = sys.argv[1:]
    dtype = arguments[arguments.index("--dtype") + 1]
    print("loading...", flush=True)
    print(json.dumps({"ready": True, "sample_rate": 24000, "device": "cpu", "dtype": dtype,
                      "model": "stand-in", "revision": None}), flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        if job.get("op") == "quit":
            break
        if job["text"] == "hang":
            time.sleep(3600)
        if job["text"] == "die":
            sys.exit(3)
        with wave.open(job["output"], "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\\x00\\x10" * 2400)
        print(json.dumps({"ok": True}), flush=True)
''')


@pytest.fixture
def engine(config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8")
    monkeypatch.setattr(omnivoice_tts, "_worker_script", lambda: script)
    monkeypatch.setenv("VOICECLONE_OMNIVOICE_PYTHON", sys.executable)
    settings = config.engines.omnivoice
    monkeypatch.setattr(settings, "python", sys.executable)
    monkeypatch.setattr(settings, "timeout_seconds", 2)
    # What the Mac has free right now is not what these tests are about.
    from voiceclone import memory

    monkeypatch.setattr(memory, "pressure_level", lambda: 1)
    monkeypatch.setattr(memory, "available_gb", lambda: 64.0)
    # Nor whether this machine holds a float16 copy of the weights.
    monkeypatch.setattr(omnivoice_tts, "cached_revision", lambda model_id: None)
    made = omnivoice_tts.OmniVoiceEngine(config, DeviceSpec(torch.device("cpu"), torch.float32, False))
    yield made
    made.unload()


REFERENCE = Audio(np.zeros(24000, dtype=np.float32) + 0.01, 24000)


def render(engine, text: str):
    return engine.synthesize(text, REFERENCE, reference_text="متن مرجع", seed=42)


def test_a_normal_reply_is_a_render(engine) -> None:
    result = render(engine, "سلام")
    assert result.audio.duration == pytest.approx(0.1)
    assert result.metadata["model"] == "stand-in"


def test_a_worker_that_stops_answering_costs_one_call(engine) -> None:
    render(engine, "سلام")
    first = engine._process.pid
    with pytest.raises(SynthesisError, match="did not reply within 2 s"):
        render(engine, "hang")
    assert not engine.loaded and engine._process is None
    assert render(engine, "سلام").audio.duration == pytest.approx(0.1)
    assert engine._process.pid != first


def test_a_worker_that_dies_is_replaced_on_the_next_call(engine) -> None:
    render(engine, "سلام")
    with pytest.raises(SynthesisError, match=r"exited \(code 3\)"):
        render(engine, "die")
    assert not engine.loaded
    assert render(engine, "سلام").audio.duration == pytest.approx(0.1)
