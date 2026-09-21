"""Speech recognition.

Two jobs:

1. Transcribe the reference clip - F5-TTS conditions on ``(audio, transcript)``
   and a wrong transcript degrades the clone badly.
2. Drive the speech-to-speech cascade, where source audio becomes text,
   passes through the Persian pipeline, and is re-spoken in the target voice.

Backends: ``faster-whisper`` (CTranslate2, several times quicker on CPU) with
a Transformers fallback so the feature works with what f5-tts already pulls in.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audio.io import Audio, save_audio
from ..config import AppConfig, ASRConfig
from ..device import DeviceSpec
from ..errors import DependencyMissingError, VoiceCloneError
from ..logging_utils import get_logger

log = get_logger("engines.asr")

#: Whisper family models expect 16 kHz mono.
ASR_SAMPLE_RATE = 16000

#: Short names -> Hub ids for the Transformers backend.
_TRANSFORMERS_MODELS = {
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small",
    "base": "openai/whisper-base",
}

#: Files a CTranslate2 Whisper conversion needs beside ``model.bin``.
_CT2_COPY_FILES = ["tokenizer.json", "preprocessor_config.json"]

#: The recogniser FINDINGS.md CER was measured with. Benchmarks that write a
#: CER must use this block, not the Studio's product defaults, or the number
#: is not the same measurement.
SCORING_ASR: dict[str, object] = {
    "model": "large-v3-turbo",
    "vad_filter": True,
    "no_speech_threshold": 0.6,
    "initial_prompt": "",
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 2000,
    "vad_speech_pad_ms": 400,
}


#: Whisper's mean token log-probability below which a transcript is suspect.
#: -1.0 was far too lax: a clip transcribed with a wrong consonant
#: (``عوارز`` for ``عوارض``) passed it comfortably. Good Persian output from
#: large-v3-turbo sits around -0.2 to -0.4.
MIN_AVG_LOGPROB = -0.55

#: Persian words that continue a sentence rather than open one. A reference
#: transcript starting with any of them is a fragment of something longer.
_CONTINUATION_WORDS = frozenset(
    {
        "یا", "و", "که", "اما", "ولی", "چون", "زیرا", "تا", "پس", "اگرچه",
        "بلکه", "یعنی", "هرچند", "وگرنه", "بنابراین", "درنتیجه", "همچنین",
    }
)


@dataclass
class Transcript:
    """Recognised text plus per-segment timing and confidence."""

    text: str
    language: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    #: Mean log-probability across segments; near 0 is confident.
    avg_logprob: float | None = None

    @property
    def is_confident(self) -> bool:
        return self.avg_logprob is None or self.avg_logprob > MIN_AVG_LOGPROB

    @property
    def starts_mid_sentence(self) -> bool:
        """True when the text opens on a word that cannot begin an utterance.

        A reference clip cut out of a longer recording often starts partway
        through a sentence. Whisper transcribes the fragment perfectly and
        reports high confidence, so the confidence signal never fires - but
        F5-TTS conditions on this text, and a transcript that does not match
        the audio it describes degrades the clone.

        Observed on a real clip: ``یا بدون نسخ مصرف می‌کنید ...`` - opening on
        *or*, with no warning raised.
        """
        first = self.text.strip().split()
        return bool(first) and first[0].strip("،؛:") in _CONTINUATION_WORDS

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.text


def faster_whisper_decode_options(
    settings: ASRConfig, language: str | None, *, with_timestamps: bool = False
) -> dict[str, Any]:
    """Keyword arguments for ``WhisperModel.transcribe``.

    Split out so the decode policy can be tested without loading a model.
    ``condition_on_previous_text`` stays off: Whisper otherwise invents a
    continuation over silence.
    """
    options: dict[str, Any] = {
        "language": language,
        "beam_size": int(settings.beam_size),
        "vad_filter": bool(settings.vad_filter),
        "word_timestamps": with_timestamps,
        "condition_on_previous_text": False,
        "no_speech_threshold": float(settings.no_speech_threshold),
    }
    if options["vad_filter"]:
        options["vad_parameters"] = {
            "threshold": float(settings.vad_threshold),
            "min_silence_duration_ms": int(settings.vad_min_silence_ms),
            "speech_pad_ms": int(settings.vad_speech_pad_ms),
        }
    prompt = (settings.initial_prompt or "").strip()
    if prompt:
        options["initial_prompt"] = prompt
    return options


def ct2_cache_dir(cache_root: Path, model: str) -> Path:
    """Where a one-time CTranslate2 conversion of ``model`` is stored."""
    slug = model.replace("/", "--")
    return Path(cache_root) / "whisper" / f"{slug}-ct2"


def is_ct2_model(path: Path) -> bool:
    """True when ``path`` looks like a converted faster-whisper checkpoint."""
    return path.is_dir() and (path / "model.bin").is_file() and (path / "config.json").is_file()


def apply_scoring_asr(config: AppConfig) -> AppConfig:
    """Pin the historical recogniser so a new CER is comparable with FINDINGS."""
    return config.copy_with(**{f"engines.asr.{key}": value for key, value in SCORING_ASR.items()})


class ASREngine:
    """Lazily-loaded speech recogniser."""

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        self.config = config
        self.device = device
        self._model: Any = None
        self._pipeline: Any = None
        self.backend = ""

    # -- availability ------------------------------------------------------
    @staticmethod
    def available_backends() -> dict[str, bool]:
        return {
            "faster-whisper": importlib.util.find_spec("faster_whisper") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
        }

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if any(cls.available_backends().values()):
            return True, ""
        return False, "pip install faster-whisper"

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None or self._pipeline is not None:
            return

        settings = self.config.engines.asr
        preferred = str(settings.backend).lower()
        backends = self.available_backends()

        if (
            preferred in {"faster-whisper", "auto"}
            and backends["faster-whisper"]
            and self._load_faster_whisper()
        ):
            return
        if backends["transformers"] and self._load_transformers():
            return
        if preferred == "faster-whisper" and not backends["faster-whisper"]:
            raise DependencyMissingError("faster-whisper", "speech recognition")

        raise DependencyMissingError("faster-whisper", "speech recognition")

    def _load_faster_whisper(self) -> bool:
        from faster_whisper import WhisperModel  # type: ignore

        settings = self.config.engines.asr
        # CTranslate2 has no Metal backend, so Apple Silicon runs this on CPU
        # with int8 - still faster than float32 Whisper through PyTorch.
        if self.device.is_cuda:
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
        if settings.compute_type and settings.compute_type != "auto":
            compute_type = settings.compute_type

        try:
            resolved = self._resolve_faster_whisper_model(settings.model)
            self._model = WhisperModel(
                resolved,
                device=device,
                compute_type=compute_type,
                download_root=str(self.config.paths.ensure("cache_dir") / "whisper"),
            )
            self.backend = f"faster-whisper/{settings.model} ({device}/{compute_type})"
            log.info("ASR ready: %s", self.backend)
            return True
        except Exception:
            log.warning("faster-whisper could not load '%s'; trying Transformers.", settings.model, exc_info=True)
            self._model = None
            return False

    def _resolve_faster_whisper_model(self, model: str) -> str:
        """A path or id faster-whisper can load.

        Short names (``large-v3-turbo``) are fetched by the library. A Hub id
        that is already CTranslate2 loads as-is. A Transformers Whisper
        checkpoint is converted once into ``cache_dir/whisper/<id>-ct2``.
        """
        if "/" not in model:
            return model

        cached = ct2_cache_dir(self.config.paths.ensure("cache_dir"), model)
        if is_ct2_model(cached):
            log.info("ASR using cached CTranslate2 conversion at %s", cached)
            return str(cached)

        converted = self._convert_to_ct2(model, cached)
        if converted is not None:
            return converted
        # Do not hand a Transformers snapshot to WhisperModel: it looks for
        # model.bin, fails, and the traceback is a worse fallback than going
        # straight to the Transformers pipeline.
        raise VoiceCloneError(
            f"{model} is not a CTranslate2 Whisper checkpoint and conversion failed"
        )

    def _convert_to_ct2(self, model_id: str, output_dir: Path) -> str | None:
        """Convert a Transformers Whisper checkpoint. Returns the path or None."""
        try:
            from ctranslate2.converters import TransformersConverter
        except ImportError:
            log.info("ctranslate2 converters unavailable; not converting %s", model_id)
            return None

        log.info("Converting %s to CTranslate2 (once) at %s", model_id, output_dir)
        try:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            converter = TransformersConverter(model_id, copy_files=_CT2_COPY_FILES)
            converter.convert(str(output_dir), quantization="int8", force=False)
        except Exception:
            log.warning("CTranslate2 conversion of %s failed.", model_id, exc_info=True)
            if output_dir.exists() and not is_ct2_model(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            return None
        if is_ct2_model(output_dir):
            return str(output_dir)
        log.warning("CTranslate2 conversion of %s produced no model.bin.", model_id)
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        return None

    def _load_transformers(self) -> bool:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError:
            return False

        settings = self.config.engines.asr
        model_id = _TRANSFORMERS_MODELS.get(settings.model, settings.model)
        try:
            import torch

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=self.device.device,
                torch_dtype=torch.float32,
                chunk_length_s=30,
            )
            self.backend = f"transformers/{model_id}"
            log.info("ASR ready: %s", self.backend)
            return True
        except Exception:
            log.warning("Transformers ASR could not load '%s'.", model_id, exc_info=True)
            self._pipeline = None
            return False

    def unload(self) -> None:
        self._model = None
        self._pipeline = None
        self.backend = ""

    # -- inference ---------------------------------------------------------
    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None = None,
        with_timestamps: bool = False,
    ) -> Transcript:
        """Transcribe ``audio``. ``language=None`` lets the model auto-detect."""
        self.load()

        settings = self.config.engines.asr
        target_language = language if language is not None else settings.language
        if target_language in {"auto", ""}:
            target_language = None

        work = audio.to_mono().resample(ASR_SAMPLE_RATE)
        if work.duration < 0.2:
            return Transcript("", target_language or "", [], self.backend)

        if self._model is not None:
            return self._transcribe_faster_whisper(work, target_language, with_timestamps)
        if self._pipeline is not None:
            return self._transcribe_transformers(work, target_language, with_timestamps)
        raise VoiceCloneError("No speech-recognition backend is loaded.")

    def _transcribe_faster_whisper(
        self, audio: Audio, language: str | None, with_timestamps: bool
    ) -> Transcript:
        settings = self.config.engines.asr
        # ``audio`` is resampled to 16 kHz above; faster-whisper assumes that
        # rate for a numpy buffer and has no sampling_rate argument.
        segments, info = self._model.transcribe(
            audio.samples,
            **faster_whisper_decode_options(settings, language, with_timestamps=with_timestamps),
        )

        parts: list[str] = []
        collected: list[dict[str, Any]] = []
        logprobs: list[float] = []

        for segment in segments:
            parts.append(segment.text)
            logprobs.append(float(getattr(segment, "avg_logprob", 0.0)))
            collected.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                }
            )

        return Transcript(
            text=" ".join(part.strip() for part in parts).strip(),
            language=str(getattr(info, "language", language or "")),
            segments=collected,
            backend=self.backend,
            avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
        )

    def _transcribe_transformers(
        self, audio: Audio, language: str | None, with_timestamps: bool
    ) -> Transcript:
        generate_kwargs: dict[str, Any] = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"

        result = self._pipeline(
            {"raw": audio.samples, "sampling_rate": audio.sample_rate},
            return_timestamps=with_timestamps,
            generate_kwargs=generate_kwargs or None,
        )

        text = (result.get("text") if isinstance(result, dict) else str(result)) or ""
        segments: list[dict[str, Any]] = []
        for chunk in (result.get("chunks") or []) if isinstance(result, dict) else []:
            start, end = (chunk.get("timestamp") or (None, None))[:2]
            segments.append({"start": start, "end": end, "text": (chunk.get("text") or "").strip()})

        return Transcript(text.strip(), language or "", segments, self.backend)

    def transcribe_file(self, path: str | Path, **kwargs: Any) -> Transcript:
        from ..audio.io import load_audio

        return self.transcribe(load_audio(path), **kwargs)


def write_temp_wav(audio: Audio, directory: Path | None = None) -> Path:
    """Helper for backends that only accept file paths."""
    target = Path(directory or tempfile.gettempdir()) / f"asr_{abs(hash(audio.num_frames))}.wav"
    return save_audio(audio.to_mono().resample(ASR_SAMPLE_RATE), target)


__all__ = [
    "ASR_SAMPLE_RATE",
    "ASREngine",
    "SCORING_ASR",
    "Transcript",
    "apply_scoring_asr",
    "ct2_cache_dir",
    "faster_whisper_decode_options",
    "is_ct2_model",
    "write_temp_wav",
]
