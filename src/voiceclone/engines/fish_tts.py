"""Fish Audio S2 Pro - the strongest-claimed open weights on the shortlist.

The candidate table in FINDINGS.md listed two models that were never measured
because their licences are research-only and neither fits this Mac. Licence and
hardware decide what may be *shipped*, not what may be *known*, and leaving the
strongest claim in the field unmeasured is how a project ends up believing its
own leaderboard. So S2 Pro is measured here on the same 27 cells as everything
else, and what its licence means is recorded next to the number.

What the claim is, checked against its own model card, repository and technical
report (arXiv 2603.08823) rather than from memory:

* a dual-autoregressive design - a 4B slow transformer over time predicting the
  semantic codebook, a 400M fast one filling in 9 residual codebooks per frame
* trained on 10M+ hours across 80+ languages, with Persian among them, though
  in the third tier: tier 1 is Japanese, English and Chinese, tier 2 is another
  seven, and Persian is in the long "other supported languages" list
* word error rates on Seed-TTS below Qwen3-TTS, MiniMax Speech-02 and Seed-TTS,
  which is what "best among open and closed source" in its README rests on

None of those numbers are comparable to this project's. Whether it is better
*here* - on Persian, on these nine speakers, against this project's held-out
similarity scale - is what the benchmark decides.

Licence
-------
Fish Audio Research License: research and non-commercial use free of charge,
commercial use by separate agreement. Every other engine in this project is
Apache-2.0 or equivalent, so a result here cannot simply be shipped even if it
wins. Measuring it is research use.

Setup
-----
    git clone https://github.com/fishaudio/fish-speech ~/.cache/persian-voice-cloning/fish
    uv venv --python python3.12 ~/.cache/persian-voice-cloning/fish/.venv
    VIRTUAL_ENV=~/.cache/persian-voice-cloning/fish/.venv uv pip install -e ~/.cache/persian-voice-cloning/fish

It brings its own pinned torch, so it gets its own environment. The 11 GB of
weights are fetched on first use.
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

log = get_logger("engines.fish")

#: Where to look for the fish-speech virtualenv.
_VENV_CANDIDATES = (
    "~/.cache/persian-voice-cloning/fish/.venv",
    "~/fish-speech/.venv",
    "./third_party/fish-speech/.venv",
)

#: The first start may include fetching about 11 GB of weights.
_STARTUP_TIMEOUT = 3600.0

#: Generation settings forwarded to the worker, overridable per call.
GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "chunk_length",
    "max_new_tokens",
)


def find_fish_python(configured: str | None = None) -> str | None:
    """Locate an interpreter with ``fish_speech`` installed."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_FISH_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "fish_worker.py"


@register_tts
class FishS2Engine(TTSEngine):
    name = "fish"
    display_name = "Fish Audio S2 Pro (dual-AR, 80+ languages)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        # S2 Pro would otherwise pick a voice at random: without a reference
        # transcript it has nothing to align the reference codes to, and
        # upstream's own CLI refuses --prompt-text without prompt audio and
        # vice versa. Supplying the transcript also keeps reference handling
        # identical across engines, so a comparison measures the engine and not
        # two different speech recognisers.
        requires_reference_text=True,
        native_persian=True,
        supports_speed=False,
        supports_seed=True,
        approx_download_gb=11.0,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sample_rate = 44100
        self._loaded_precision = "unknown"

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_fish_python() is None:
            return False, (
                "Fish S2 Pro needs its own environment:\n"
                "  git clone https://github.com/fishaudio/fish-speech"
                " ~/.cache/persian-voice-cloning/fish\n"
                "  uv venv --python python3.12 ~/.cache/persian-voice-cloning/fish/.venv\n"
                "  VIRTUAL_ENV=~/.cache/persian-voice-cloning/fish/.venv"
                " uv pip install -e ~/.cache/persian-voice-cloning/fish"
            )
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        settings = self.config.engines.fish
        python = find_fish_python(settings.python)
        if python is None:
            _, reason = self.is_available()
            raise EngineNotAvailableError("fish-speech environment not found.", hint=reason)

        command = [
            python,
            str(_worker_script()),
            "--checkpoint",
            str(settings.checkpoint),
            "--device",
            str(self.device.device),
            "--precision",
            str(settings.precision),
            "--max-seq-len",
            str(settings.max_seq_len),
        ]

        log.info("Starting Fish S2 worker (%s)", python)
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
            raise EngineNotAvailableError(f"Could not start the Fish S2 worker: {exc}") from exc

        ready = self._read_reply(_STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError(
                "The Fish S2 worker failed to load its models.",
                hint=str(ready.get("error", "no reason given")),
            )
        self._sample_rate = int(ready.get("sample_rate", 44100))
        self._loaded_precision = str(ready.get("precision", "unknown"))
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_fish_")
        if self._loaded_precision != settings.precision:
            log.warning(
                "Fish S2 loaded in %s, not the requested %s - results describe the "
                "precision that actually ran.",
                self._loaded_precision,
                settings.precision,
            )
        log.info(
            "Fish S2 ready (%s, %s, %d Hz)",
            self.device.device,
            self._loaded_precision,
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
                    f"The Fish S2 worker exited (code {code}).",
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

        settings = self.config.engines.fish
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
            raise SynthesisError("The Fish S2 worker is no longer accepting jobs.") from exc

        reply = self._read_reply(float(settings.timeout_seconds))
        if not reply.get("ok"):
            raise SynthesisError(
                f"Fish S2 synthesis failed: {reply.get('error', 'unknown error')}",
                hint="Try a shorter chunk or a cleaner reference.",
            )

        rendered = load_audio(output_path, mono=True)
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("Fish S2 returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="fish"),
            engine=self.name,
            seed=used_seed,
            metadata={
                "device": str(self.device.device),
                "precision": self._loaded_precision,
                "reference_seconds": round(reference.duration, 2),
                "frames": reply.get("frames"),
                **{name: job[name] for name in GENERATION_FIELDS},
            },
        )


__all__ = ["FishS2Engine", "find_fish_python"]
