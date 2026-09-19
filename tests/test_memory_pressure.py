"""Stop before the Mac thrashes, and load OmniVoice from half the bytes.

On 2026-09-19 an OmniVoice start reached warning memory pressure on this 16 GB
Mac, with no swap and 5.1 GB already compressed. Two answers, tested here
without any model: the app reads macOS's own pressure level, refuses new work
at warning and stops the OmniVoice worker the moment it gets there; and the
worker can load a float16 copy of the weights it casts to float16 anyway, which
is accepted only with a record saying every tensor was checked bit for bit.
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

from voiceclone import memory
from voiceclone.audio.io import Audio
from voiceclone.config import AppConfig
from voiceclone.device import DeviceSpec
from voiceclone.engines import omnivoice_tts
from voiceclone.errors import SynthesisError
from voiceclone.pipelines.studio import Studio

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("omnivoice_float16", ROOT / "scripts" / "omnivoice_float16.py")
float16 = importlib.util.module_from_spec(spec)
sys.modules["omnivoice_float16"] = float16
spec.loader.exec_module(float16)


class TestThePressureLevel:
    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS publishes the level; elsewhere it is None")
    def test_macos_reports_a_level(self) -> None:
        assert memory.pressure_level() in {1, 2, 4}

    def test_the_watch_fires_once_at_warning(self) -> None:
        readings = iter([1, 1, 2, 2, 4])
        fired: list[int] = []
        with memory.PressureWatch(fired.append, interval=0.01, read=lambda: next(readings, 1)) as watch:
            for _ in range(200):
                if fired:
                    break
                import time

                time.sleep(0.01)
        assert fired == [2] and watch.triggered == 2

    def test_where_no_level_exists_nothing_is_watched(self) -> None:
        with memory.PressureWatch(lambda level: pytest.fail("fired"), read=lambda: None) as watch:
            pass
        assert watch.triggered is None and watch._thread is None


def test_the_studio_starts_nothing_at_warning(config: AppConfig, monkeypatch) -> None:
    monkeypatch.setattr(memory, "pressure_level", lambda: 2)
    studio = Studio(config)
    studio.prepare_reference = lambda *a, **k: pytest.fail("prepared under pressure")
    with pytest.raises(SynthesisError, match="nothing new was started"):
        studio.synthesize("سلام", "unused.wav", engine="f5")
    off = Studio(config.copy_with(**{"runtime.stop_on_memory_pressure": False}))
    off.refuse_under_pressure()


WORKER = textwrap.dedent('''
    import json, sys, time, wave

    arguments = sys.argv[1:]
    print(json.dumps({"ready": True, "sample_rate": 24000, "device": "cpu",
                      "dtype": arguments[arguments.index("--dtype") + 1],
                      "model": arguments[arguments.index("--model") + 1], "revision": None}), flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        if job.get("op") == "quit":
            break
        if job["text"] == "slow":
            time.sleep(30)
        with wave.open(job["output"], "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\\x00\\x10" * 2400)
        print(json.dumps({"ok": True}), flush=True)
''')

REFERENCE = Audio(np.zeros(24000, dtype=np.float32) + 0.01, 24000)


@pytest.fixture
def engine(config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8")
    monkeypatch.setattr(omnivoice_tts, "_worker_script", lambda: script)
    monkeypatch.setattr(omnivoice_tts, "cached_revision", lambda model_id: None)
    monkeypatch.setattr(config.engines.omnivoice, "python", sys.executable)
    made = omnivoice_tts.OmniVoiceEngine(config, DeviceSpec(torch.device("cpu"), torch.float32, False))
    yield made
    made.unload()


def render(engine, text: str = "سلام"):
    return engine.synthesize(text, REFERENCE, reference_text="متن مرجع", seed=42)


class TestTheWorkerUnderPressure:
    def test_it_is_not_started_at_warning(self, engine, monkeypatch) -> None:
        monkeypatch.setattr(memory, "pressure_level", lambda: 2)
        with pytest.raises(SynthesisError, match="not started"):
            render(engine)
        assert engine._process is None

    def test_it_is_not_started_without_the_memory_it_needs(self, engine, config, monkeypatch) -> None:
        monkeypatch.setattr(memory, "pressure_level", lambda: 1)
        monkeypatch.setattr(memory, "available_gb", lambda: 1.5)
        monkeypatch.setattr(config.engines.omnivoice, "min_available_gb", 4.0)
        with pytest.raises(SynthesisError, match=r"needs about 4.0 GB .* 1.5 GB available"):
            render(engine)
        assert engine._process is None

    def test_it_is_stopped_when_warning_arrives_mid_render(self, engine, monkeypatch) -> None:
        level = {"now": 1}
        monkeypatch.setattr(memory, "pressure_level", lambda: level["now"])
        render(engine)
        level["now"] = 2
        with pytest.raises(SynthesisError, match="memory pressure reached warning"):
            render(engine, "slow")
        assert not engine.loaded and engine._process is None


def snapshot(tmp_path: Path) -> Path:
    from safetensors.torch import save_file

    folder = tmp_path / "snapshots" / "abc123"
    folder.mkdir(parents=True)
    torch.manual_seed(0)
    tensors = {
        "llm.w": torch.randn(16, 8) * 3,
        "llm.b": torch.randn(8),
        "heads.scale": torch.tensor(1.2345678),
        "bf": torch.randn(4, 4).to(torch.bfloat16),
        "offsets": torch.arange(8, dtype=torch.int64),
    }
    save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    (folder / "config.json").write_text('{"model_type": "omnivoice"}')
    (folder / "tokenizer.json").write_text("{}")
    (folder / "chat_template.jinja").write_text("{{ x }}")
    (folder / "README.md").write_text("readme")
    (folder / "audio_tokenizer").mkdir()
    (folder / "audio_tokenizer" / "config.json").write_text("{}")
    return folder


class TestTheFloat16Copy:
    def test_every_tensor_is_the_cast_original_bit_for_bit(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        source = snapshot(tmp_path)
        record = float16.convert(source, tmp_path / "float16" / "abc123", "k2-fsa/OmniVoice", "abc123")
        copy = tmp_path / "float16" / "abc123"
        assert record["verified"] is True and record["floating_tensors"] == 4
        with safe_open(str(source / "model.safetensors"), "pt") as a, safe_open(str(copy / "model.safetensors"), "pt") as b:
            assert b.metadata() == {"format": "pt"}
            for key in sorted(a.keys()):
                before, after = a.get_tensor(key), b.get_tensor(key)
                if before.is_floating_point():
                    assert after.dtype == torch.float16
                    assert torch.equal(after.view(torch.int16), before.to(torch.float16).view(torch.int16))
                else:
                    assert torch.equal(after, before)
        assert (copy / "audio_tokenizer").is_symlink()
        assert (copy / "config.json").read_text() == '{"model_type": "omnivoice"}'
        assert not (copy / "README.md").exists()
        assert json.loads((copy / "SOURCE.json").read_text())["revision"] == "abc123"

    def test_only_a_verified_copy_of_the_cached_revision_is_used(self, tmp_path: Path) -> None:
        float16.convert(snapshot(tmp_path), tmp_path / "float16" / "abc123", "k2-fsa/OmniVoice", "abc123")
        root = tmp_path / "float16"
        found = omnivoice_tts.float16_copy("k2-fsa/OmniVoice", root, revision="abc123")
        assert found is not None and found[0] == root / "abc123"
        assert omnivoice_tts.float16_copy("k2-fsa/OmniVoice", root, revision="newer") is None
        assert omnivoice_tts.float16_copy("someone/else", root, revision="abc123") is None
        assert omnivoice_tts.float16_copy(str(tmp_path), root, revision="abc123") is None
        record = json.loads((root / "abc123" / "SOURCE.json").read_text())
        (root / "abc123" / "SOURCE.json").write_text(json.dumps({**record, "verified": False}))
        assert omnivoice_tts.float16_copy("k2-fsa/OmniVoice", root, revision="abc123") is None

    def test_the_worker_is_given_the_copy_and_the_result_names_the_revision(
        self, engine, config, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "cache"
        float16.convert(snapshot(tmp_path), cache / "omnivoice" / "float16" / "abc123", "k2-fsa/OmniVoice", "abc123")
        monkeypatch.setattr(config.paths, "cache_dir", str(cache))
        monkeypatch.setattr(omnivoice_tts, "cached_revision", lambda model_id: "abc123")
        monkeypatch.setattr(memory, "pressure_level", lambda: 1)
        result = render(engine)
        assert result.metadata["model"] == "k2-fsa/OmniVoice"
        assert result.metadata["revision"] == "abc123"
        assert str(cache / "omnivoice" / "float16" / "abc123") in result.metadata["weights"]


class TestTheCheckItself:
    def test_tensors_split_across_many_pieces_convert_and_verify(self, tmp_path: Path, monkeypatch) -> None:
        # 16-byte pieces: four float32 values in, and the copy read back in eight-value pieces.
        monkeypatch.setattr(float16, "PIECE_BYTES", 16)
        record = float16.convert(snapshot(tmp_path), tmp_path / "out" / "abc123", "k2-fsa/OmniVoice", "abc123")
        assert record["verified"] is True
        assert record["bit_for_bit_against_the_cast_original"]["mismatched"] == []

    def test_a_single_wrong_bit_is_caught(self, tmp_path: Path) -> None:
        source = snapshot(tmp_path) / "model.safetensors"
        copy = tmp_path / "copy.safetensors"
        float16.write_float16(source, copy)
        assert float16.verify(source, copy)["mismatched"] == []
        with copy.open("rb") as handle:
            header, start = float16.read_header(handle)
        begin, _ = header["llm.w"]["data_offsets"]
        data = bytearray(copy.read_bytes())
        data[start + begin + 5] ^= 0x01
        copy.write_bytes(bytes(data))
        assert float16.verify(source, copy)["mismatched"] == ["llm.w"]
