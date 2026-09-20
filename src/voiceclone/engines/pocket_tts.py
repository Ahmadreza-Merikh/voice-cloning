"""Pocket-TTS Farsi - the Persian-specific model, here to be measured.

Every other engine in this project is multilingual with Persian somewhere
inside it, or a community finetune of one. `mehdi-hf/pocket-tts-farsi-v2`
(September 2026) is neither: 109.5 M parameters distilled from Kyutai's
pocket-tts and trained on **973 hours of Persian from 2,978 speakers** - 7.6
times the 128 hours of the adaptation this project is running - with a phoneme
frontend instead of Persian script. Its authors report, on 300 held-out Common
Voice pairs, a median per-item WER of 0.333, speaker similarity 0.859 and
UTMOS 3.11, against v1's 0.764 similarity.

Those numbers are theirs, measured their way, and are not comparable with
anything in `FINDINGS.md`: different encoder, different reference (theirs is
the prompt clip), different test set, WER against this project's CER. That is
the whole reason this engine exists here - to be scored on the same
development set, with the same ECAPA against held-out recordings of the same
speaker, beside F5, OmniVoice and the rest.

It reads **phonemes, not Persian script**: Persian text produces silence, with
no error. The worker phonemises with the G2P the model card names, which is
Homo-GE2PE - the same family as the Persian G2P research this project already
bridges to for analysis. So this is also the first engine here that can
actually synthesise from that work.

Setup
-----
    uv venv --python python3.11 ~/.cache/persian-voice-cloning/pocket/.venv
    VIRTUAL_ENV=~/.cache/persian-voice-cloning/pocket/.venv uv pip install \\
        "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main" transformers soundfile

The fork is not optional: the released `pocket-tts` package rejects the three
flags this model's config carries, and a config without them silently deletes
the first word of every generation.

Weights (0.41 GiB) and G2P (0.03 GiB) are fetched by
``scripts/fetch_inputs.py`` into that same directory; both revisions are
pinned in `configs/default.yaml`. Licence: the model is CC-BY-NC-4.0, the G2P
MIT.
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

log = get_logger("engines.pocket")

_VENV_CANDIDATES = (
    "~/.cache/persian-voice-cloning/pocket/.venv",
    "~/pocket-tts/.venv",
    "./third_party/pocket-tts/.venv",
)
#: Seconds to wait for both models to load. They are small; this is generous.
_STARTUP_TIMEOUT = 600.0

_SETUP_HINT = (
    "Pocket-TTS Farsi needs its own environment:\n"
    "  uv venv --python python3.11 ~/.cache/persian-voice-cloning/pocket/.venv\n"
    "  VIRTUAL_ENV=~/.cache/persian-voice-cloning/pocket/.venv uv pip install \\\n"
    '      "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main" transformers soundfile\n'
    "  .venv/bin/python scripts/fetch_inputs.py --plan eval/plans/pocket_tts.json --report /tmp/pocket.json"
)


def find_pocket_python(configured: str | None = None) -> str | None:
    """Locate an interpreter with ``pocket_tts`` installed."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_POCKET_PYTHON")
    if env:
        candidates.append(env)
    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def trim_prompt(audio: Audio, seconds: float | None) -> Audio:
    """The first ``seconds`` of the prompt, or all of it.

    The prompt is a prefix this model continues from, not an embedding taken
    once, and its training utterances averaged 3.8 s - so how much reference it
    is handed is a setting to measure, not a constant to assume.
    """
    if seconds is None or float(seconds) <= 0 or audio.duration <= float(seconds):
        return audio
    keep = int(float(seconds) * audio.sample_rate)
    samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)[:keep]
    return Audio(samples, audio.sample_rate, source=audio.source)


def _worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "pocket_worker.py"


@register_tts
class PocketTTSEngine(TTSEngine):
    name = "pocket"
    display_name = "Pocket-TTS Farsi v2 (973 h Persian, CPU)"
    capabilities = EngineCapabilities(
        # It takes phonemes, but not this project's IPA: the worker phonemises
        # with the model's own G2P and notation, which is what its card
        # prescribes. Handing it our IPA instead would be a different recipe.
        accepts_phonemes=False,
        # Speaker conditioning is acoustic - an embedding from the prompt clip -
        # so there is no transcript to align against.
        requires_reference_text=False,
        native_persian=True,
        supports_speed=False,
        supports_seed=True,
        approx_download_gb=0.45,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._process: subprocess.Popen[str] | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sample_rate = 24000

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_pocket_python() is None:
            return False, _SETUP_HINT
        if not _worker_script().is_file():
            return False, f"Worker script missing: {_worker_script()}"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        settings = self.config.engines.pocket
        python = find_pocket_python(settings.python)
        if python is None:
            raise EngineNotAvailableError("Pocket-TTS environment not found.", hint=_SETUP_HINT)

        model_dir = Path(str(settings.model_dir)).expanduser()
        g2p_dir = Path(str(settings.g2p_dir)).expanduser()
        for label, path in (("weights", model_dir / "model.safetensors"), ("G2P", g2p_dir / "config.json")):
            if not path.exists():
                raise EngineNotAvailableError(f"Pocket-TTS {label} not downloaded ({path}).",
                                              hint=_SETUP_HINT)

        command = [
            python,
            str(_worker_script()),
            "--model-dir",
            str(model_dir),
            "--g2p-dir",
            str(g2p_dir),
            "--max-tokens",
            str(int(settings.max_tokens)),
            "--retries",
            str(int(settings.retries)),
        ]
        log.info("Starting Pocket-TTS worker (%s)", python)
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=None, text=True, bufsize=1)
        except OSError as exc:
            raise EngineNotAvailableError(f"Could not start the Pocket-TTS worker: {exc}") from exc

        ready = self._read_reply(_STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self._stop()
            raise EngineNotAvailableError("The Pocket-TTS worker failed to load its models.",
                                          hint=str(ready.get("error", "no reason given")))
        self._sample_rate = int(ready.get("sample_rate", 24000))
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_pocket_")
        log.info("Pocket-TTS Farsi ready (cpu, %d Hz)", self._sample_rate)

    def _read_reply(self, timeout: float) -> dict[str, Any]:
        """Read the worker's next JSON line, stepping over anything that is not."""
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                code = self._process.poll()
                raise SynthesisError(f"The Pocket-TTS worker exited (code {code}).",
                                     hint="Its traceback was written to this console.")
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

        settings = self.config.engines.pocket
        used_seed = seed_everything(seed)
        work = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        # The prompt is a prefix this model continues from, not an embedding
        # taken once: its training utterances averaged 3.8 s, and its own
        # announcement asks for "a 5-second sample". So how much of the
        # reference it is given is an engine setting, measured rather than
        # assumed - `prompt_seconds: null` hands over the whole window.
        prompt = trim_prompt(reference.to_mono(), kwargs.get("prompt_seconds", settings.prompt_seconds))
        # 16-bit PCM, not float: pocket-tts reads prompts with Python's `wave`
        # module, which rejects IEEE-float WAV outright ("unknown format: 3").
        reference_path = save_audio(prompt, work / "reference.wav", bit_depth=16)
        output_path = work / "render.wav"

        job = {"text": text.strip(), "reference": str(reference_path),
               "output": str(output_path), "seed": used_seed}
        try:
            self._process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SynthesisError("The Pocket-TTS worker is no longer accepting jobs.") from exc

        reply = self._read_reply(float(settings.timeout_seconds))
        if not reply.get("ok"):
            raise SynthesisError(f"Pocket-TTS synthesis failed: {reply.get('error', 'unknown error')}",
                                 hint="Try a shorter sentence or a cleaner reference.")

        rendered = load_audio(output_path, mono=True)
        samples = np.asarray(rendered.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("Pocket-TTS returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, rendered.sample_rate, source="pocket"),
            engine=self.name,
            seed=used_seed,
            metadata={
                "phonemes": reply.get("phonemes"),
                "retried_runaways": reply.get("retried", 0),
                "max_tokens": int(settings.max_tokens),
                "prompt_seconds": round(prompt.duration, 2),
                "reference_seconds": round(reference.duration, 2),
            },
        )


__all__ = ["PocketTTSEngine", "find_pocket_python", "trim_prompt"]
