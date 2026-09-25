"""LongCat-AudioDiT is driven the way its own inference.py drives it, and only on CUDA.

The engine exists for one measurement (``eval/runs/persian-voice-longcat-probe.json``),
so what these tests protect is what would quietly change that measurement: the
README's voice-cloning settings reaching the worker, the prompt transcript being
joined the way upstream joins it, the duration estimate being upstream's, the
second-GPU placement moving tensors and nothing else - and the engine refusing
this Mac before a 5.7-15.3 GB fp32 model is loaded on it. No model is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from voiceclone.audio.io import Audio
from voiceclone.device import DeviceSpec
from voiceclone.engines import list_tts_engines, longcat_tts
from voiceclone.errors import EngineNotAvailableError, SynthesisError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("longcat_worker", SCRIPTS / "longcat_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

WORKER = textwrap.dedent('''
    import json, sys, time, wave

    print("loading...", flush=True)
    print(json.dumps({"ready": True, "sample_rate": 24000, "model_dir": "stand-in", "device": "cuda:0",
                      "transformer_device": "cuda:0", "dtype": {"transformer": "float32"},
                      "parameters": 1, "dit_depth": 24, "dit_dim": 1536, "max_wav_duration": 30}),
          flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        if job.get("op") == "quit":
            break
        if job["text"] == "hang":
            time.sleep(3600)
        with open(job["output"] + ".job.json", "w") as handle:
            json.dump(job, handle, ensure_ascii=False)
        with wave.open(job["output"], "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\\x00\\x10" * 2400)
        print(json.dumps({"ok": True, "frames": 138, "prompt_frames": 96,
                          "full_text": job["reference_text"] + " " + job["text"]}), flush=True)
''')

REFERENCE = Audio(np.zeros(24000, dtype=np.float32) + 0.01, 24000)
CUDA = DeviceSpec(torch.device("cuda"), torch.float32, False)


def inputs(root: Path) -> dict[str, Path]:
    places = {"model_dir": root / "model", "code_dir": root / "code", "tokenizer_dir": root / "umt5"}
    for relative in ("model/config.json", "model/model.safetensors", "code/audiodit/modeling_audiodit.py",
                     "code/utils.py", "umt5/spiece.model"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("")
    return places


@pytest.fixture
def engine(config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8")
    monkeypatch.setattr(longcat_tts, "_worker_script", lambda: script)
    monkeypatch.setenv("VOICECLONE_LONGCAT_PYTHON", sys.executable)
    settings = config.engines.longcat
    monkeypatch.setattr(settings, "python", sys.executable)
    monkeypatch.setattr(settings, "timeout_seconds", 2)
    for name, path in inputs(tmp_path).items():
        monkeypatch.setattr(settings, name, str(path))
    made = longcat_tts.LongCatEngine(config, CUDA)
    yield made
    made.unload()


def render(engine, text: str = "The bridge was closed, so we took the ferry."):
    return engine.synthesize(text, REFERENCE, reference_text="دیروز به کتابخانه رفتم.", seed=42)


class TestTheEngine:
    def test_it_is_registered_and_does_not_claim_persian(self) -> None:
        engine = list_tts_engines()["longcat"]
        assert engine.capabilities.native_persian is False
        assert engine.capabilities.requires_reference_text is True
        assert engine.capabilities.accepts_phonemes is False

    def test_the_readme_voice_cloning_settings_are_the_defaults(self, config) -> None:
        settings = config.engines.longcat
        assert (settings.nfe, settings.guidance_strength, settings.guidance_method) == (16, 4.0, "apg")
        assert settings.transformer_device == ""

    def test_this_mac_is_refused_before_a_worker_starts(self, config, monkeypatch) -> None:
        started = []
        monkeypatch.setattr(longcat_tts.subprocess, "Popen", lambda *a, **k: started.append(a))
        monkeypatch.setenv("VOICECLONE_LONGCAT_PYTHON", sys.executable)
        for device in ("cpu", "mps"):
            made = longcat_tts.LongCatEngine(config, DeviceSpec(torch.device(device), torch.float32, False))
            with pytest.raises(EngineNotAvailableError, match="runs only on CUDA"):
                made.load()
        assert started == []

    def test_missing_inputs_are_named_before_a_worker_starts(self, config, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VOICECLONE_LONGCAT_PYTHON", sys.executable)
        monkeypatch.setattr(config.engines.longcat, "model_dir", str(tmp_path / "nowhere"))
        with pytest.raises(EngineNotAvailableError, match="inputs are missing"):
            longcat_tts.LongCatEngine(config, CUDA).load()


class TestTheWorkerProtocol:
    def test_a_render_carries_the_transcript_and_the_settings(self, engine) -> None:
        result = render(engine)
        job = json.loads(Path(engine._tempdir.name, "render.wav.job.json").read_text(encoding="utf-8"))
        assert job["reference_text"] == "دیروز به کتابخانه رفتم."
        assert (job["nfe"], job["guidance_strength"], job["guidance_method"]) == (16, 4.0, "apg")
        assert job["seed"] == 42
        assert result.audio.duration == pytest.approx(0.1)
        assert result.metadata["full_text"].startswith("دیروز به کتابخانه رفتم. The bridge")
        assert result.metadata["dit_depth"] == 24 and result.metadata["frames"] == 138

    def test_a_render_without_a_transcript_is_refused(self, engine) -> None:
        with pytest.raises(SynthesisError, match="transcript"):
            engine.synthesize("Hello there.", REFERENCE, reference_text=" ", seed=42)

    def test_a_worker_that_stops_answering_costs_one_call(self, engine) -> None:
        render(engine)
        first = engine._process.pid
        with pytest.raises(SynthesisError, match="did not reply within 2 s"):
            render(engine, "hang")
        assert not engine.loaded and engine._process is None
        assert render(engine).audio.duration == pytest.approx(0.1)
        assert engine._process.pid != first


class TestUpstreamsArithmetic:
    """``inference.py`` at 12c76b51, restated as functions the worker calls."""

    @staticmethod
    def approx(text: str, max_duration: float = 30.0) -> float:
        return min(max_duration, len(text.replace(" ", "")) * 0.082)

    def test_the_transcript_comes_first_with_one_space(self) -> None:
        assert worker.joined_text("prompt words.", "new text") == "prompt words. new text"

    def test_duration_scales_by_the_prompts_own_pace(self) -> None:
        # 96 frames at 2048/24000 s is 8.192 s of prompt for 20 letters (1.64 s
        # estimated): the ratio clips at 1.5, so 10 letters' 0.82 s becomes 1.23 s.
        assert worker.frames(96, "a" * 10, "b" * 20, self.approx, 24000, 2048, 30.0) == 96 + 14

    def test_a_quick_prompt_does_not_shorten_the_text(self) -> None:
        # A prompt faster than the estimate would give a ratio under 1; it clips to 1.0.
        assert worker.frames(10, "a" * 10, "b" * 200, self.approx, 24000, 2048, 30.0) == 10 + 9

    def test_the_total_is_capped_at_the_models_window(self) -> None:
        assert worker.frames(300, "a" * 400, "b" * 20, self.approx, 24000, 2048, 30.0) == 351


class TestPlacement:
    def test_nested_tensors_move_and_the_rest_is_kept(self) -> None:
        value = {"x": torch.ones(2), "pair": (torch.zeros(1), 3), "name": "text", "layers": [torch.ones(1)]}
        moved = worker.moved(value, "meta")
        assert moved["x"].device.type == "meta" and moved["pair"][0].device.type == "meta"
        assert moved["pair"][1] == 3 and moved["name"] == "text" and moved["layers"][0].device.type == "meta"

    def test_a_placed_module_answers_where_it_was_called_from(self) -> None:
        class Echo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, x, text=None):
                return {"last_hidden_state": x * self.scale, "hidden_state": None, "text": text}

        module = Echo()
        worker.place(module, "cpu", "cpu")
        out = module(x=torch.full((2,), 2.0), text=torch.ones(1))
        assert torch.equal(out["last_hidden_state"], torch.full((2,), 2.0))
        assert out["hidden_state"] is None and out["text"].device.type == "cpu"
