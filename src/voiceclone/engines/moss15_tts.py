"""MOSS-TTS-Local-Transformer-v1.5, at its official recipe, out-of-process.

The v1.0 engine (`moss_tts.py`) runs in-process and measured 0.559 held-out on
the development set. v1.5 is not a new checkpoint name for it: the codec
changed (MOSS-Audio-Tokenizer-v2, 48 kHz stereo, 12 codebooks fixed), the card
recommends a language tag, and the processor is its own. It also matters for a
reason unrelated to quality: among the candidates audited in SOTA_ROADMAP.md,
it is the one whose model, codec and code were all checked and found
Apache-2.0, at pinned revisions.

What the worker does and why is in `scripts/moss15_worker.py`. What this side
adds is the audio bookkeeping the benchmark needs:

* the native render - 48 kHz stereo, exactly as the processor returned it - is
  kept under ``native_dir`` with an index line tying it to its text, seed and
  reference, so the evaluated audio can always be recovered
* the harness gets the channel mean of that file (``load_audio(mono=True)``),
  which is what every other engine's audio goes through too; the similarity
  scorer then resamples to 16 kHz. Both steps are named in the metadata

Setup
-----
    uv venv --python python3.12 ~/.cache/persian-voice-cloning/moss15/.venv
    VIRTUAL_ENV=~/.cache/persian-voice-cloning/moss15/.venv uv pip install \\
        "torch==2.9.1" "torchaudio==2.9.1" "transformers==5.0.0" "accelerate>=1.10.1" soundfile

The official stack pins torch 2.9.1+cu128; about 17.6 GB of weights (model
9.1 GB, codec 8.5 GB in fp32 on disk) are fetched at the pinned revisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
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

log = get_logger("engines.moss15")

_VENV_CANDIDATES = (
    "~/.cache/persian-voice-cloning/moss15/.venv",
    "./third_party/moss15/.venv",
)

GENERATION_FIELDS = (
    "audio_temperature",
    "audio_top_p",
    "audio_top_k",
    "audio_repetition_penalty",
    "max_new_tokens",
)


def find_moss15_python(configured: str | None = None) -> str | None:
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_MOSS15_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "moss15_worker.py"


@register_tts
class Moss15Engine(TTSEngine):
    name = "moss15"
    display_name = "MOSS-TTS Local Transformer v1.5 (48 kHz stereo, 31 languages)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        # v1.5's user message has no reference-transcript field: the prompt is
        # the reference audio's codes alone.
        requires_reference_text=False,
        native_persian=True,
        supports_speed=False,
        supports_seed=True,
        approx_download_gb=17.6,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._ready: dict[str, Any] = {}
        self._session = "unloaded"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_moss15_python() is None:
            return False, (
                "MOSS v1.5 needs its own environment:\n"
                "  uv venv --python python3.12 ~/.cache/persian-voice-cloning/moss15/.venv\n"
                "  VIRTUAL_ENV=~/.cache/persian-voice-cloning/moss15/.venv uv pip install "
                '"torch==2.9.1" "torchaudio==2.9.1" "transformers==5.0.0" "accelerate>=1.10.1" soundfile'
            )
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    def _load(self) -> None:
        settings = self.config.engines.moss15
        python = find_moss15_python(settings.python)
        if python is None:
            _, reason = self.is_available()
            raise EngineNotAvailableError("MOSS v1.5 environment not found.", hint=reason)
        command = [
            python,
            str(_worker_script()),
            "--device",
            str(self.device.device),
            "--codec-device",
            str(settings.codec_device),
        ]
        log.info("Starting MOSS v1.5 worker (%s)", python)
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=None, text=True, bufsize=1)
        except OSError as exc:
            raise EngineNotAvailableError(f"Could not start the MOSS v1.5 worker: {exc}") from exc
        ready = self._read_reply()
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError("The MOSS v1.5 worker failed to load.",
                                          hint=str(ready.get("error", "no reason given")))
        self._ready = ready
        # One directory per engine session for native renders; see _keep_native.
        self._session = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_moss15_")
        log.info("MOSS v1.5 ready: %s", json.dumps(ready))

    def _read_reply(self) -> dict[str, Any]:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise SynthesisError(f"The MOSS v1.5 worker exited (code {self._process.poll()}).",
                                     hint="Its traceback was written to this console.")
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return dict(json.loads(stripped))
            except json.JSONDecodeError:
                log.debug("Ignoring non-protocol output: %r", stripped[:200])

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

    def _keep_native(self, rendered: Path, text: str, seed: int, reference: Path, reply: dict,
                     job: dict) -> Path:
        """Keep the render exactly as written, never over an earlier one, and say what made it.

        Each engine session gets its own directory, so a later run - with other
        settings, or the same ones - cannot replace an earlier run's files. The
        file name is a hash of everything that determines the render: text,
        seed, reference, model and codec revisions, precision, language and the
        sampling values. If that name is somehow taken, a counter is added
        rather than writing over it. The index line carries the whole recipe
        and the file's own sha256.
        """
        settings = self.config.engines.moss15
        root = Path(settings.native_dir)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[3] / root
        session = root / self._session
        session.mkdir(parents=True, exist_ok=True)
        recipe = {name: job.get(name) for name in GENERATION_FIELDS}
        recipe.update({k: self._ready.get(k) for k in ("model_revision", "codec_revision", "dtype",
                                                       "attention", "codec_weight_dtype")})
        recipe["language"] = job.get("language")
        reference_sha1 = hashlib.sha1(reference.read_bytes()).hexdigest()
        identity = json.dumps({"text": text, "seed": seed, "reference": reference_sha1, **recipe},
                              sort_keys=True, ensure_ascii=False)
        stem = hashlib.sha1(identity.encode()).hexdigest()[:16]
        target, counter = session / f"{stem}.wav", 1
        while target.exists():
            target, counter = session / f"{stem}-{counter}.wav", counter + 1
        shutil.copy2(rendered, target)
        record = {"file": target.name, "text": text, "seed": seed, "reference_sha1": reference_sha1,
                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "sample_rate": reply.get("sample_rate"), "channels": reply.get("channels"),
                  "produced_dtype": reply.get("produced_dtype"), "file_subtype": reply.get("file_subtype"),
                  "seconds": reply.get("seconds"), "generate_seconds": reply.get("generate_seconds"),
                  "peak_memory_gib": reply.get("memory_gib"), "recipe": recipe}
        with (session / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return target

    def synthesize(self, text: str, reference: Audio, *, reference_text: str | None = None,
                   speed: float = 1.0, seed: int | None = None, **kwargs: Any) -> SynthesisResult:
        self.load()
        assert self._process is not None and self._process.stdin is not None
        if not text or not text.strip():
            raise SynthesisError("Nothing to synthesise (empty text).")

        settings = self.config.engines.moss15
        used_seed = seed_everything(seed)
        work = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        reference_path = save_audio(reference.to_mono(), work / "reference.wav", bit_depth=32)
        output_path = work / "render.wav"
        job: dict[str, Any] = {"text": text.strip(), "reference": str(reference_path),
                               "output": str(output_path), "seed": used_seed,
                               "language": kwargs.get("language", settings.language)}
        for name in GENERATION_FIELDS:
            job[name] = kwargs.get(name, getattr(settings, name))
        try:
            self._process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SynthesisError("The MOSS v1.5 worker is no longer accepting jobs.") from exc
        reply = self._read_reply()
        if not reply.get("ok"):
            raise SynthesisError(f"MOSS v1.5 synthesis failed: {reply.get('error', 'unknown error')}")

        native = self._keep_native(output_path, job["text"], used_seed, reference_path, reply, job)
        rendered = load_audio(native, mono=True)  # channel mean, as for every engine
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("MOSS v1.5 returned no audio.")
        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="moss15"),
            engine=self.name,
            seed=used_seed,
            metadata={
                **{k: self._ready.get(k) for k in ("model_revision", "codec_revision", "dtype",
                                                   "attention", "capability", "codec_device")},
                "language_tag": job["language"],
                "language_tag_is_documented": False,
                "native_file": f"{self._session}/{native.name}",
                "native_format": f"{reply.get('sample_rate')} Hz, {reply.get('channels')} channels",
                "scored_as": "channel mean of the native file; the similarity scorer resamples to 16 kHz",
                "generate_seconds": reply.get("generate_seconds"),
                **{name: job[name] for name in GENERATION_FIELDS},
            },
        )


__all__ = ["Moss15Engine", "find_moss15_python"]
