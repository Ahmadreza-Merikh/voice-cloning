"""Higgs TTS 3 - the other unmeasured candidate, and the one that claims Persian.

Boson AI's Higgs TTS 3 is a Qwen3-4B backbone with a fused 8-codebook audio
head. What made it worth the work, checked against its own model card rather
than from memory:

* Persian sits in the model card's top tier - the 85 languages it calls
  "WER/CER under 5", not the 17 below them
* on its own multilingual voice-clone table it reports the lowest error of
  eleven systems, including three this project has already measured: Fish S2
  Pro, OmniVoice and MOSS-TTS. That ordering - Higgs about level with
  OmniVoice, both ahead of MOSS - matches what this project measured
  independently, which is a reason to take the table seriously and no reason
  to take it as neutral: it is the vendor's own

Getting it to run was the whole problem. Boson ships it for SGLang-Omni or
vLLM-Omni, which want compute capability 8.0 - the free T4 is 7.5 - and the one
public attempt at forcing the Turing path returned WAV files whose every sample
was -32768. The port this engine loads notes why that happens: the codec is
unstable in 16-bit and has to run in fp32.

Licence
-------
Boson Higgs Audio v3 Research and Non-Commercial License, with a separate grant
that lets creators monetise the *output* with attribution. Like Fish S2 Pro,
that is narrower than every other engine here, and it is why a result cannot
simply be shipped even if it wins.

Setup
-----
    uv venv --python python3.11 ~/.cache/persian-voice-cloning/higgs/.venv
    VIRTUAL_ENV=~/.cache/persian-voice-cloning/higgs/.venv uv pip install \\
        "transformers>=5.5" torch torchaudio soundfile

It needs transformers 5.5 or newer and about 10 GB of weights, so it gets its
own environment and runs out-of-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.io import Audio, load_audio, save_audio
from ..config import AppConfig
from ..device import DeviceSpec, seed_everything
from ..errors import EngineNotAvailableError, SynthesisError
from ..logging_utils import get_logger
from .base import EngineCapabilities, SynthesisResult, TTSEngine, register_tts

log = get_logger("engines.higgs")

#: Where to look for the Higgs virtualenv.
_VENV_CANDIDATES = (
    "~/.cache/persian-voice-cloning/higgs/.venv",
    "~/higgs/.venv",
    "./third_party/higgs/.venv",
)

#: The first start includes fetching about 10 GB of weights.
_STARTUP_TIMEOUT = 3600.0

#: Generation settings forwarded to the worker, overridable per call.
GENERATION_FIELDS = ("temperature", "top_p", "top_k", "max_new_tokens")


def find_higgs_python(configured: str | None = None) -> str | None:
    """Locate an interpreter with transformers 5.5 or newer installed."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_HIGGS_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "higgs_worker.py"


@register_tts
class HiggsTTSEngine(TTSEngine):
    name = "higgs"
    display_name = "Higgs TTS 3 (Boson AI, 102 languages)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        # Higgs takes the reference transcript through its own <|ref_text|>
        # slot, and supplying it keeps reference handling identical across
        # engines, so a comparison measures the engine and not two different
        # speech recognisers.
        requires_reference_text=True,
        native_persian=True,
        supports_speed=False,
        supports_seed=True,
        approx_download_gb=10.1,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sample_rate = 24000
        self._loaded_dtype = "unknown"

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_higgs_python() is None:
            return False, (
                "Higgs TTS 3 needs its own environment:\n"
                "  uv venv --python python3.11 ~/.cache/persian-voice-cloning/higgs/.venv\n"
                "  VIRTUAL_ENV=~/.cache/persian-voice-cloning/higgs/.venv"
                ' uv pip install "transformers>=5.5" torch torchaudio soundfile'
            )
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        settings = self.config.engines.higgs
        python = find_higgs_python(settings.python)
        if python is None:
            _, reason = self.is_available()
            raise EngineNotAvailableError("Higgs environment not found.", hint=reason)

        command = [
            python,
            str(_worker_script()),
            "--model",
            str(settings.model_id),
            "--revision",
            str(settings.revision),
            "--device",
            str(self.device.device),
            "--dtype",
            str(settings.dtype),
        ]

        log.info("Starting Higgs worker (%s)", python)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # diagnostics go straight to the console
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise EngineNotAvailableError(f"Could not start the Higgs worker: {exc}") from exc

        ready = self._read_reply(_STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError(
                "The Higgs worker failed to load its model.",
                hint=str(ready.get("error", "no reason given")),
            )
        self._sample_rate = int(ready.get("sample_rate", 24000))
        self._loaded_dtype = str(ready.get("dtype", "unknown"))
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_higgs_")
        if self._loaded_dtype != settings.dtype:
            log.warning(
                "Higgs loaded in %s, not the requested %s - results describe the "
                "dtype that actually ran.",
                self._loaded_dtype,
                settings.dtype,
            )
        log.info(
            "Higgs ready (%s, %s, %d Hz)",
            self.device.device,
            self._loaded_dtype,
            self._sample_rate,
        )

    def _read_reply(self, timeout: float) -> dict[str, Any]:
        """Read the worker's next JSON line, stepping over anything that is not."""
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                code = self._process.poll()
                raise SynthesisError(
                    f"The Higgs worker exited (code {code}).",
                    hint="Its traceback was written to this console.",
                )
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
        self.load()
        assert self._process is not None and self._process.stdin is not None

        if not text or not text.strip():
            raise SynthesisError("Nothing to synthesise (empty text).")

        settings = self.config.engines.higgs
        used_seed = seed_everything(seed)
        work = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        reference_path = save_audio(reference.to_mono(), work / "reference.wav", bit_depth=32)
        output_path = work / "render.wav"

        job: dict[str, Any] = {
            "text": text.strip(),
            "reference": str(reference_path),
            "reference_text": (reference_text or "").strip() or None,
            "output": str(output_path),
            "seed": used_seed,
        }
        for name in GENERATION_FIELDS:
            job[name] = kwargs.get(name, getattr(settings, name))

        try:
            self._process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SynthesisError("The Higgs worker is no longer accepting jobs.") from exc

        reply = self._read_reply(float(settings.timeout_seconds))
        if not reply.get("ok"):
            raise SynthesisError(
                f"Higgs synthesis failed: {reply.get('error', 'unknown error')}",
                hint="Try a shorter chunk or a cleaner reference.",
            )

        rendered = load_audio(output_path, mono=True)
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("Higgs returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="higgs"),
            engine=self.name,
            seed=used_seed,
            metadata={
                "device": str(self.device.device),
                "dtype": self._loaded_dtype,
                "revision": str(settings.revision),
                "reference_seconds": round(reference.duration, 2),
                **{name: job[name] for name in GENERATION_FIELDS},
            },
        )


__all__ = ["HiggsTTSEngine", "find_higgs_python"]
