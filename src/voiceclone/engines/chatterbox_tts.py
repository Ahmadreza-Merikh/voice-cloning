"""Chatterbox engine - the one whose speaker path does not depend on Persian.

F5-TTS clones in context: one model reads the reference audio, the reference
transcript and the target text together, so how well it carries a voice is
bound up with how well it was trained on the language. For Persian that
training is a community finetune of unknown provenance, and measurement puts
its identity transfer 0.17 below the human ceiling while its pronunciation is
already better than the corpus it is scored against.

Chatterbox splits the job. A voice encoder and S3Gen - a conformer plus a
flow-matching decoder and a HiFT vocoder - carry speaker identity, and they are
language-agnostic: they were trained on half a million hours and never see
Persian text. Only T3, the 0.5B Llama that turns characters into speech tokens,
is language-specific, and that is the part `Thomcles/Chatterbox-TTS-Persian-Farsi`
replaces. So the component this project is weakest on is the component Persian
data is *not* needed for.

Chatterbox pins ``torch==2.6``, ``transformers==5.2`` and ``numpy<2``, none of
which can coexist with what F5-TTS needs here, so it runs out-of-process in its
own virtualenv - the arrangement Seed-VC already uses. Unlike Seed-VC the
worker is persistent: the model costs about half a minute to load and a
measurement run asks for dozens of renders, so paying that once matters.

Setup
-----
    python3.11 -m venv ~/.cache/persian-voice-cloning/chatterbox/.venv
    ~/.cache/persian-voice-cloning/chatterbox/.venv/bin/pip install chatterbox-tts

The Persian T3 is a gated repository: accept the licence on its model page
while signed in, then ``hf auth login``. Without it this engine still runs on
the base checkpoint, which speaks Persian text with an English mouth - it is
reported clearly rather than silently.
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

log = get_logger("engines.chatterbox")

#: Where to look for the Chatterbox virtualenv.
_VENV_CANDIDATES = (
    "~/.cache/persian-voice-cloning/chatterbox/.venv",
    "~/chatterbox/.venv",
    "./third_party/chatterbox/.venv",
)

#: Seconds to wait for the model to load before giving up.
_STARTUP_TIMEOUT = 900.0


def find_chatterbox_python(configured: str | None = None) -> str | None:
    """Locate an interpreter with ``chatterbox-tts`` installed."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_CHATTERBOX_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "chatterbox_worker.py"


@register_tts
class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    display_name = "Chatterbox (Persian T3 finetune)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        # The reference transcript is not used: speaker conditioning is
        # acoustic, so there is nothing to misalign against.
        requires_reference_text=False,
        native_persian=True,
        supports_speed=False,
        supports_seed=True,
        approx_download_gb=3.5,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sample_rate = 24000
        self._using_persian_t3 = False

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_chatterbox_python() is None:
            return False, (
                "Chatterbox needs its own environment (torch 2.6, transformers 5.2):\n"
                "  python3.11 -m venv ~/.cache/persian-voice-cloning/chatterbox/.venv\n"
                "  ~/.cache/persian-voice-cloning/chatterbox/.venv/bin/pip install chatterbox-tts"
            )
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _resolve_t3(self) -> str | None:
        """Fetch the Persian T3, or explain why the base model is being used."""
        settings = self.config.engines.chatterbox
        if not settings.t3_checkpoint:
            return None

        from ..models import resolve_model

        try:
            resolved = resolve_model(
                settings.t3_checkpoint,
                cache_dir=self.config.paths.ensure("cache_dir"),
                description="Chatterbox Persian T3",
            )
            return str(resolved.path)
        except Exception as exc:
            log.warning(
                "Persian T3 unavailable (%s). Falling back to the base checkpoint, which "
                "reads Persian text with an English-trained tokenizer - expect noticeably "
                "worse pronunciation. The repository is gated: accept the licence at "
                "https://huggingface.co/%s while signed in, then run `hf auth login`.",
                exc,
                str(settings.t3_checkpoint).removeprefix("hf://").rsplit("/", 1)[0],
            )
            return None

    def _load(self) -> None:
        settings = self.config.engines.chatterbox
        python = find_chatterbox_python(settings.python)
        if python is None:
            available, reason = self.is_available()
            raise EngineNotAvailableError("Chatterbox environment not found.", hint=reason)

        checkpoint = self._resolve_t3()
        self._using_persian_t3 = checkpoint is not None

        command = [python, str(_worker_script()), "--device", str(self.device.device)]
        if checkpoint:
            command += ["--t3-checkpoint", checkpoint]

        log.info("Starting Chatterbox worker (%s)", python)
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
            raise EngineNotAvailableError(f"Could not start the Chatterbox worker: {exc}") from exc

        ready = self._read_reply(_STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError(
                "The Chatterbox worker failed to load its model.",
                hint=str(ready.get("error", "no reason given")),
            )
        self._sample_rate = int(ready.get("sample_rate", 24000))
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_chatterbox_")
        log.info(
            "Chatterbox ready (%s T3, %d Hz)",
            "Persian" if self._using_persian_t3 else "base English",
            self._sample_rate,
        )

    def _read_reply(self, timeout: float) -> dict[str, Any]:
        """Read one JSON line from the worker, or explain why there wasn't one."""
        assert self._process is not None and self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise SynthesisError(
                f"The Chatterbox worker exited (code {code}).",
                hint="Its traceback was written to this console.",
            )
        try:
            return dict(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SynthesisError(f"Unreadable reply from the Chatterbox worker: {line[:200]!r}") from exc

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

        settings = self.config.engines.chatterbox
        used_seed = seed_everything(seed)
        work = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        reference_path = save_audio(reference.to_mono(), work / "reference.wav", bit_depth=32)
        output_path = work / "render.wav"

        job = {
            "text": text.strip(),
            "reference": str(reference_path),
            "output": str(output_path),
            "seed": used_seed,
            "exaggeration": float(kwargs.get("exaggeration", settings.exaggeration)),
            "cfg_weight": float(kwargs.get("cfg_weight", settings.cfg_weight)),
            "temperature": float(kwargs.get("temperature", settings.temperature)),
            "repetition_penalty": float(kwargs.get("repetition_penalty", settings.repetition_penalty)),
            "min_p": float(kwargs.get("min_p", settings.min_p)),
            "top_p": float(kwargs.get("top_p", settings.top_p)),
        }

        try:
            self._process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SynthesisError("The Chatterbox worker is no longer accepting jobs.") from exc

        reply = self._read_reply(float(settings.timeout_seconds))
        if not reply.get("ok"):
            raise SynthesisError(
                f"Chatterbox synthesis failed: {reply.get('error', 'unknown error')}",
                hint="Try a shorter chunk or a cleaner reference.",
            )

        rendered = load_audio(output_path, mono=True)
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("Chatterbox returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="chatterbox"),
            engine=self.name,
            seed=used_seed,
            metadata={
                "persian_t3": self._using_persian_t3,
                "exaggeration": job["exaggeration"],
                "cfg_weight": job["cfg_weight"],
                "temperature": job["temperature"],
                "reference_seconds": round(reference.duration, 2),
            },
        )


__all__ = ["ChatterboxEngine", "find_chatterbox_python"]
