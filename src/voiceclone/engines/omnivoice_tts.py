"""OmniVoice engine - a newer base with Persian in its training, not bolted on.

Every result in this project so far was measured on F5-TTS continued from
``Lumos675/F5_TTS_Persian``, a community finetune whose model card is a licence
line. Its measured profile - pronunciation better than the corpus it is scored
against, identity 0.17 below the human ceiling - is what narrow-speaker
training produces, and a 12.5 h multi-speaker continuation recovered 16% of
that gap.

OmniVoice was chosen after checking the current landscape live rather than from
memory (see FINDINGS.md). What made it the first candidate, all verified against
its own repository, model card and paper (arXiv 2604.00688):

* Persian is in its training languages, and on FLEURS it reports Persian CER of
  1.90% against 1.57% for the ground-truth recordings
* on identical training data it beats F5-TTS on every speaker-similarity test
  set its authors report (0.729 vs 0.655, 0.741 vs 0.664, 0.777 vs 0.750)
* Apache-2.0, 3.3 GB, and Apple Silicon is a documented target

None of those numbers are comparable to this project's: they use a WavLM-based
extractor on other corpora and other languages. Whether it is better *here* is
what the benchmark decides.

It brings its own torch and transformers, so it runs out-of-process in its own
virtualenv, the arrangement Chatterbox and Seed-VC already use.

Setup
-----
    uv venv --python python3.11 ~/.cache/persian-voice-cloning/omnivoice/.venv
    VIRTUAL_ENV=~/.cache/persian-voice-cloning/omnivoice/.venv uv pip install omnivoice
"""

from __future__ import annotations

import contextlib
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

from .. import memory
from ..audio.io import Audio, load_audio, save_audio
from ..config import AppConfig, OmniVoiceConfig
from ..device import DeviceSpec, seed_everything
from ..errors import EngineNotAvailableError, SynthesisError
from ..logging_utils import get_logger
from .base import EngineCapabilities, SynthesisResult, TTSEngine, register_tts

log = get_logger("engines.omnivoice")

#: Where to look for the OmniVoice virtualenv.
_VENV_CANDIDATES = (
    "~/.cache/persian-voice-cloning/omnivoice/.venv",
    "~/omnivoice/.venv",
    "./third_party/omnivoice/.venv",
)

#: The first start may include fetching 3.3 GB of weights.
_STARTUP_TIMEOUT = 1800.0

#: What a start refused, or a worker stopped, for memory tells the user to do.
_PRESSURE_HINT = (
    "Close some apps - browsers, other AI apps - and try again, or pick F5-TTS. On this 16 GB Mac "
    "with no swap, OmniVoice loading beside other open apps reached warning pressure (2026-09-19)."
)

#: Generation settings forwarded to the worker, overridable per call.
GENERATION_FIELDS = (
    "num_step",
    "guidance_scale",
    "t_shift",
    "denoise",
    "postprocess_output",
    "layer_penalty_factor",
    "position_temperature",
    "class_temperature",
)


def find_omnivoice_python(configured: str | None = None) -> str | None:
    """Locate an interpreter with ``omnivoice`` installed."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_OMNIVOICE_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "omnivoice_worker.py"


def cached_revision(model_id: str) -> str | None:
    """The revision of ``model_id`` in the local Hugging Face cache, without going online."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:  # noqa: BLE001 - not cached, or no hub library: no copy applies
        return None


def float16_copy(model_id: str, root: Path, revision: str | None = None) -> tuple[Path, dict[str, Any]] | None:
    """A verified float16 copy of the cached ``model_id``, made for exactly its cached revision.

    ``scripts/omnivoice_float16.py`` writes the copy and its ``SOURCE.json``;
    anything short of a record that names this model and revision and says
    every tensor was verified is ignored, so a stale or partial copy is never
    loaded in place of the published weights.
    """
    if os.path.isdir(model_id):
        return None
    revision = revision or cached_revision(model_id)
    if not revision:
        return None
    copy = root / revision
    try:
        record = json.loads((copy / "SOURCE.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("model_id") != model_id or record.get("revision") != revision or record.get("verified") is not True:
        return None
    if not (copy / "model.safetensors").is_file():
        return None
    return copy, record


@register_tts
class OmniVoiceEngine(TTSEngine):
    name = "omnivoice"
    display_name = "OmniVoice (k2-fsa, 600+ languages)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        # OmniVoice would otherwise transcribe the reference with its own
        # Whisper inside the worker. Supplying the transcript keeps reference
        # handling identical across engines, so a comparison measures the
        # engine and not two different speech recognisers.
        requires_reference_text=True,
        native_persian=True,
        supports_speed=True,
        supports_seed=True,
        approx_download_gb=3.3,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sample_rate = 24000
        self._loaded_device = "unknown"
        self._loaded_dtype = "unknown"

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_omnivoice_python() is None:
            return False, (
                "OmniVoice needs its own environment:\n"
                "  uv venv --python python3.11 ~/.cache/persian-voice-cloning/omnivoice/.venv\n"
                "  VIRTUAL_ENV=~/.cache/persian-voice-cloning/omnivoice/.venv uv pip install omnivoice"
            )
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    # -- hooks for derived engines (omnivoice_ipa) --------------------------
    def _settings(self) -> OmniVoiceConfig:
        """Generation, dtype and guard settings for this engine.

        A subclass points its own config section here; nothing else in this
        class knows about the difference.
        """
        return self.config.engines.omnivoice

    def _model_source(self, settings: OmniVoiceConfig) -> str:
        """What the worker loads: a hub id, or a local directory of weights."""
        return str(settings.model_id)

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        settings = self._settings()
        python = find_omnivoice_python(settings.python)
        if python is None:
            _, reason = self.is_available()
            raise EngineNotAvailableError("OmniVoice environment not found.", hint=reason)
        if self.config.runtime.stop_on_memory_pressure:
            self._refuse_to_start(settings.min_available_gb)

        model = self._model_source(settings)
        copy = None
        if settings.float16_copy and str(settings.dtype) == "float16":
            copy = float16_copy(model, Path(self.config.paths.cache_dir).expanduser() / "omnivoice" / "float16")
        if copy is not None:
            log.info("Loading the verified float16 copy of %s at %s", model, copy[0])

        command = [
            python,
            str(_worker_script()),
            "--model",
            str(copy[0]) if copy else model,
            "--device",
            str(self.device.device),
            "--dtype",
            str(settings.dtype),
        ]

        log.info("Starting OmniVoice worker (%s)", python)
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
            raise EngineNotAvailableError(f"Could not start the OmniVoice worker: {exc}") from exc
        self._lines = self._forward(self._process)

        ready = self._read_reply(_STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError(
                "The OmniVoice worker failed to load its model.",
                hint=str(ready.get("error", "no reason given")),
            )
        self._sample_rate = int(ready.get("sample_rate", 24000))
        self._loaded_device = str(ready.get("device", "unknown"))
        self._loaded_dtype = str(ready.get("dtype", "unknown"))
        self._loaded_model = {"model": ready.get("model"), "revision": ready.get("revision")}
        if copy is not None:
            # The worker sees a local path; the weights are this model at this revision.
            self._loaded_model = {"model": model, "revision": copy[1]["revision"],
                                  "weights": f"verified float16 copy, {copy[0]}"}
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_omnivoice_")
        if self._loaded_device != str(self.device.device) or self._loaded_dtype != settings.dtype:
            log.warning(
                "OmniVoice loaded on %s/%s, not the requested %s/%s - results describe "
                "the configuration that actually ran.",
                self._loaded_device,
                self._loaded_dtype,
                self.device.device,
                settings.dtype,
            )
        log.info(
            "OmniVoice ready (%s, %s, %d Hz)",
            self._loaded_device,
            self._loaded_dtype,
            self._sample_rate,
        )

    @staticmethod
    def _forward(process: subprocess.Popen[str]) -> queue.Queue[str | None]:
        """Read the worker's stdout on a thread, so a reply can be waited for with a deadline."""
        lines: queue.Queue[str | None] = queue.Queue()

        def forward() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=forward, name="omnivoice-worker-stdout", daemon=True).start()
        return lines

    @staticmethod
    def _refuse_to_start(min_available_gb: float) -> None:
        """Do not start the worker into memory pressure, or with less memory than it needs."""
        level = memory.pressure_level()
        if level is None:
            return
        if level >= memory.WARNING:
            raise SynthesisError(
                f"OmniVoice was not started: the Mac's memory pressure is at {memory.describe(level)}.",
                hint=_PRESSURE_HINT,
            )
        available = memory.available_gb()
        if min_available_gb and available is not None and available < min_available_gb:
            raise SynthesisError(
                f"OmniVoice was not started: it needs about {min_available_gb:.1f} GB of memory and "
                f"the Mac has {available:.1f} GB available.",
                hint=_PRESSURE_HINT,
            )

    def _read_reply(self, timeout: float) -> dict[str, Any]:
        """Read the worker's next JSON line within ``timeout`` seconds, stepping over anything that is not.

        A worker that stops answering, or exits, is shut down and the engine
        marked unloaded, so the next call starts a new one. Before, a hung
        worker blocked its caller for good, and a dead one failed every later
        call on a closed pipe. While it loads or renders, the Mac's memory
        pressure is watched, and the worker is stopped the moment it reaches
        warning (``runtime.stop_on_memory_pressure``).
        """
        assert self._process is not None
        process = self._process

        def stop(level: int) -> None:
            log.warning("Memory pressure reached %s - stopping the OmniVoice worker", memory.describe(level))
            process.kill()

        watch = memory.PressureWatch(stop)
        guarding = self.config.runtime.stop_on_memory_pressure
        deadline = time.monotonic() + timeout
        reply: dict[str, Any] | None = None
        with watch if guarding else contextlib.nullcontext():
            while reply is None:
                try:
                    line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    self.unload()
                    raise SynthesisError(
                        f"The OmniVoice worker did not reply within {timeout:.0f} s and was stopped.",
                        hint="The next request starts a new worker.",
                    ) from None
                if line is None:
                    try:
                        code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = None
                    self.unload()
                    if watch.triggered is not None:
                        raise SynthesisError(
                            "OmniVoice was stopped: the Mac's memory pressure reached "
                            f"{memory.describe(watch.triggered)} while it was working.",
                            hint=_PRESSURE_HINT,
                        )
                    raise SynthesisError(
                        f"The OmniVoice worker exited (code {code}).",
                        hint="Its traceback was written to this console; the next request starts a new worker.",
                    )
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    reply = dict(json.loads(stripped))
                except json.JSONDecodeError:
                    log.debug("Ignoring non-protocol output from the worker: %r", stripped[:200])
        if watch.triggered is not None:
            # Stopped just as it answered: the answer stands, the worker is gone.
            self.unload()
        return reply

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

        settings = self._settings()
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
            "language": str(kwargs.get("language", settings.language)),
            "speed": float(speed),
        }
        for name in GENERATION_FIELDS:
            job[name] = kwargs.get(name, getattr(settings, name))

        try:
            self._process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SynthesisError("The OmniVoice worker is no longer accepting jobs.") from exc

        reply = self._read_reply(float(settings.timeout_seconds))
        if not reply.get("ok"):
            raise SynthesisError(
                f"OmniVoice synthesis failed: {reply.get('error', 'unknown error')}",
                hint="Try a shorter chunk or a cleaner reference.",
            )

        rendered = load_audio(output_path, mono=True)
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("OmniVoice returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="omnivoice"),
            engine=self.name,
            seed=used_seed,
            metadata={
                **getattr(self, "_loaded_model", {}),
                "device": self._loaded_device,
                "dtype": self._loaded_dtype,
                "language": job["language"],
                "reference_seconds": round(reference.duration, 2),
                **{name: job[name] for name in GENERATION_FIELDS},
            },
        )


__all__ = ["OmniVoiceEngine", "find_omnivoice_python"]
