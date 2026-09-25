"""LongCat-AudioDiT - measured against OmniVoice, not used for Persian.

On 2026-09-25 the question was whether LongCat-AudioDiT should replace
OmniVoice as the model this project adapts to Persian. What was checked live
that day (SOTA_ROADMAP section 4):

* MIT, 1B and 3.5B, a diffusion transformer in a waveform VAE's latent space
  with a UMT5 text encoder (arXiv 2603.29339)
* **Chinese and English only** - the maintainers' answer (GitHub issue #2) and
  the whole of its training data - and no training code (issue #23)
* its lead is the 3.5B's WavLM similarity on Seed-TTS-eval: scored with
  ERes2Net, or given noisy or reverberant prompts, it trails Qwen-Audio-3.0,
  dots.tts and CosyVoice3 (arXiv 2607.23938), and it had the lowest similarity
  MOS in VoxCPM2's listening test (arXiv 2606.06928)

So it cannot be asked to read Persian here. What can be measured without
training it is the part it is famous for: given a Persian speaker's clip and
English text - which it and OmniVoice both read - whose render is closer to
that speaker's *other* recordings. That is the probe this engine exists for
(``eval/runs/persian-voice-longcat-probe.json``).

Its official inference is fp32 on CUDA. The 1B's weights are 5.7 GB and the
3.5B's 15.3 GB, so on this 16 GB Mac it is refused before a worker starts.

Setup, on a CUDA machine
------------------------
The interpreter OmniVoice's pinned stack is installed in serves (transformers
5.3.0 is what LongCat's config names), plus ``einops``; the code, weights and
tokenizer are fetched by the probe's kernel at pinned revisions.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.io import Audio, load_audio, save_audio
from ..config import AppConfig
from ..device import DeviceSpec, seed_everything
from ..errors import EngineNotAvailableError, SynthesisError
from ..logging_utils import get_logger
from .base import EngineCapabilities, SynthesisResult, TTSEngine, register_tts

log = get_logger("engines.longcat")

_VENV_CANDIDATES = ("~/.cache/persian-voice-cloning/longcat/.venv",)
#: Reading 15.3 GB of fp32 weights from disk is most of a start.
_STARTUP_TIMEOUT = 1800.0

#: Generation settings forwarded to the worker, overridable per call.
GENERATION_FIELDS = ("nfe", "guidance_strength", "guidance_method")

_SETUP_HINT = (
    "LongCat-AudioDiT runs only on a CUDA GPU here - its official inference is fp32, and its "
    "weights are 5.7 GB (1B) and 15.3 GB (3.5B). It is measured in a Kaggle session "
    "(kaggle/build_longcat_probe_kernel.py), never on this Mac."
)


def find_longcat_python(configured: str | None = None) -> str | None:
    """Locate an interpreter that can run the LongCat worker."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_LONGCAT_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "longcat_worker.py"


def missing_inputs(model_dir: Path, code_dir: Path, tokenizer_dir: Path) -> list[str]:
    """What a worker would fail on minutes later, found before it is started."""
    needed = (model_dir / "config.json", model_dir / "model.safetensors",
              code_dir / "audiodit" / "modeling_audiodit.py", code_dir / "utils.py",
              tokenizer_dir / "spiece.model")
    return [str(path) for path in needed if not path.is_file()]


@register_tts
class LongCatEngine(TTSEngine):
    name = "longcat"
    display_name = "LongCat-AudioDiT (Meituan, Chinese/English only)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        # The prompt's transcript is joined before the text: the model aligns
        # the two from it and estimates the render's length from both.
        requires_reference_text=True,
        native_persian=False,
        # Its length comes from its own text estimate; there is no rate knob.
        supports_speed=False,
        supports_seed=True,
        approx_download_gb=5.7,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sample_rate = 24000
        self._facts: dict[str, Any] = {}

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_longcat_python() is None:
            return False, _SETUP_HINT
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        settings = self.config.engines.longcat
        if self.device.device.type != "cuda":
            raise EngineNotAvailableError(
                f"LongCat-AudioDiT was not started on {self.device.device}: it runs only on CUDA.",
                hint=_SETUP_HINT)
        python = find_longcat_python(settings.python)
        if python is None:
            raise EngineNotAvailableError("No interpreter for the LongCat worker.", hint=_SETUP_HINT)
        model_dir = Path(str(settings.model_dir)).expanduser()
        code_dir = Path(str(settings.code_dir)).expanduser()
        tokenizer_dir = Path(str(settings.tokenizer_dir)).expanduser()
        missing = missing_inputs(model_dir, code_dir, tokenizer_dir)
        if missing:
            raise EngineNotAvailableError(f"LongCat-AudioDiT inputs are missing: {missing}", hint=_SETUP_HINT)

        command = [
            python,
            str(_worker_script()),
            "--model-dir",
            str(model_dir),
            "--code-dir",
            str(code_dir),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--device",
            str(self.device.device),
        ]
        if settings.transformer_device:
            command += [
                "--transformer-device",
                str(settings.transformer_device),
            ]
        log.info("Starting LongCat worker (%s)", python)
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=None, text=True, bufsize=1)
        except OSError as exc:
            raise EngineNotAvailableError(f"Could not start the LongCat worker: {exc}") from exc
        self._lines = self._forward(self._process)

        ready = self._read_reply(_STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError("The LongCat worker failed to load its model.",
                                          hint=str(ready.get("error", "no reason given")))
        self._sample_rate = int(ready.get("sample_rate", 24000))
        self._facts = {key: ready.get(key) for key in (
            "model_dir", "device", "transformer_device", "dtype", "parameters", "dit_depth", "dit_dim",
            "max_wav_duration")}
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_longcat_")
        log.info("LongCat-AudioDiT ready: %s", self._facts)

    @staticmethod
    def _forward(process: subprocess.Popen[str]) -> queue.Queue[str | None]:
        """Read the worker's stdout on a thread, so a reply can be waited for with a deadline."""
        lines: queue.Queue[str | None] = queue.Queue()

        def forward() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=forward, name="longcat-worker-stdout", daemon=True).start()
        return lines

    def _read_reply(self, timeout: float) -> dict[str, Any]:
        """The worker's next JSON line within ``timeout`` seconds.

        A worker that stops answering or exits is shut down and the engine
        marked unloaded, so the next call starts a new one and one bad cell
        does not cost the rest of a run.
        """
        assert self._process is not None
        process = self._process
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self.unload()
                raise SynthesisError(f"The LongCat worker did not reply within {timeout:.0f} s and was stopped.",
                                     hint="The next request starts a new worker.") from None
            if line is None:
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    code = None
                self.unload()
                raise SynthesisError(f"The LongCat worker exited (code {code}).",
                                     hint="Its traceback was written to this console; the next request "
                                          "starts a new worker.")
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return dict(json.loads(stripped))
            except json.JSONDecodeError:
                log.debug("Ignoring non-protocol output from the worker: %r", stripped[:200])

    def _stop(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin and not self._process.stdin.closed:
                self._process.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self._process.stdin.flush()
            self._process.wait(timeout=15)
        except Exception:
            self._process.kill()
        finally:
            self._process = None

    def unload(self) -> None:
        self._stop()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        super().unload()

    # -- inference ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        reference: Audio,
        *,
        reference_text: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise SynthesisError("Nothing to synthesise (empty text).")
        if not (reference_text or "").strip():
            raise SynthesisError("LongCat needs the reference clip's transcript.")
        self.load()
        assert self._process is not None and self._process.stdin is not None

        settings = self.config.engines.longcat
        used_seed = seed_everything(seed)
        work = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        reference_path = save_audio(reference.to_mono(), work / "reference.wav", bit_depth=32)
        output_path = work / "render.wav"

        job: dict[str, Any] = {
            "text": text.strip(),
            "reference": str(reference_path),
            "reference_text": reference_text.strip(),
            "output": str(output_path),
            "seed": used_seed,
        }
        for name in GENERATION_FIELDS:
            job[name] = kwargs.get(name, getattr(settings, name))

        try:
            self._process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SynthesisError("The LongCat worker is no longer accepting jobs.") from exc

        reply = self._read_reply(float(settings.timeout_seconds))
        if not reply.get("ok"):
            raise SynthesisError(f"LongCat synthesis failed: {reply.get('error', 'unknown error')}")

        rendered = load_audio(output_path, mono=True)
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("LongCat returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="longcat"),
            engine=self.name,
            seed=used_seed,
            metadata={
                **self._facts,
                "frames": reply.get("frames"),
                "prompt_frames": reply.get("prompt_frames"),
                "full_text": reply.get("full_text"),
                "reference_seconds": round(reference.duration, 2),
                **{name: job[name] for name in GENERATION_FIELDS},
            },
        )


__all__ = ["LongCatEngine", "find_longcat_python", "missing_inputs"]
