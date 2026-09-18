"""F5-TTS engine - the fast path.

F5-TTS is a flow-matching diffusion transformer (~330 M parameters) that
clones a voice from a few seconds of reference audio. It is small enough to
run comfortably on Apple Silicon and CPU, which makes it the default here.

Persian support comes from a community finetune whose tokenizer extends the
base character vocabulary with the Persian alphabet. It is a *grapheme*
tokenizer, so this engine reads Persian script directly and does not accept
IPA - normalisation still applies, but pronunciation is up to the model.
Use the MOSS engine when phoneme-level control matters.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.io import Audio, save_audio
from ..config import AppConfig
from ..device import DeviceSpec, seed_everything
from ..errors import SynthesisError
from ..logging_utils import get_logger
from ..models import resolve_model, slim_f5_checkpoint
from .base import EngineCapabilities, SynthesisResult, TTSEngine, register_tts

log = get_logger("engines.f5")

#: F5-TTS renders at 24 kHz.
F5_SAMPLE_RATE = 24000


@register_tts
class F5TTSEngine(TTSEngine):
    name = "f5"
    display_name = "F5-TTS (Persian finetune)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        requires_reference_text=True,
        # F5-TTS's preprocess_ref_audio_text clips a reference over 12 s - at a
        # silence if it finds one, else hard at 12 s - but keeps the transcript
        # it was given whole. That transcript then describes speech the audio
        # has lost, and the output length F5 derives from their ratio comes out
        # short: a 12.4 s reference turned a four-word greeting into 1.1 s of
        # garble. 11.5 leaves a margin under its 12 s check.
        max_reference_seconds=11.5,
        native_persian=True,
        supports_speed=True,
        supports_seed=True,
        approx_download_gb=5.4,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._api: Any = None
        self._reference_cache: dict[int, str] = {}
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._loaded_from: dict[str, Any] = {}

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("f5_tts") is None:
            return False, "pip install f5-tts"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        from f5_tts.api import F5TTS  # type: ignore

        settings = self.config.engines.f5
        cache_dir = self.config.paths.ensure("cache_dir")

        checkpoint = self._resolve_checkpoint(settings, cache_dir)
        vocab_path = ""
        if settings.vocab:
            try:
                vocab_path = str(
                    resolve_model(settings.vocab, cache_dir=cache_dir, description="F5 vocabulary").path
                )
            except Exception:
                log.warning("Could not fetch the Persian vocabulary; using the model default.", exc_info=True)

        log.info("F5-TTS checkpoint: %s", checkpoint)
        if vocab_path:
            log.info("F5-TTS vocabulary:  %s", vocab_path)

        self._api = F5TTS(
            model=settings.model_arch,
            ckpt_file=str(checkpoint),
            vocab_file=vocab_path,
            device=str(self.device.device),
        )
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_f5_")

    def _resolve_checkpoint(self, settings: Any, cache_dir: Path) -> Path:
        """Fetch the Persian checkpoint, falling back to the base model."""
        try:
            resolved = resolve_model(settings.checkpoint, cache_dir=cache_dir, description="F5-TTS checkpoint")
            self._loaded_from = {"checkpoint": settings.checkpoint, "resolved": str(resolved.path),
                                 "fallback": False}
            # Community finetunes ship full training checkpoints; strip the
            # optimizer state once so every later load is cheap.
            return slim_f5_checkpoint(resolved.path)
        except Exception as exc:
            log.warning("Persian F5-TTS checkpoint unavailable (%s); falling back to the base model.", exc)

        if not settings.fallback_checkpoint:
            raise
        fallback = resolve_model(
            settings.fallback_checkpoint, cache_dir=cache_dir, description="F5-TTS base checkpoint"
        )
        log.warning(
            "Using the multilingual base checkpoint. Persian pronunciation will be noticeably worse - "
            "consider the MOSS engine instead."
        )
        self._loaded_from = {"checkpoint": settings.fallback_checkpoint, "resolved": str(fallback.path),
                             "fallback": True}
        return fallback.path

    def unload(self) -> None:
        self._api = None
        self._reference_cache.clear()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        super().unload()

    # -- inference ---------------------------------------------------------
    def _reference_path(self, reference: Audio) -> str:
        """Write the reference to a temp WAV (F5-TTS takes a file path).

        Keyed by content hash so repeated chunks of one job reuse the same
        file - which also lets F5-TTS reuse its own reference cache.
        """
        mono = reference.to_mono().resample(F5_SAMPLE_RATE)
        digest = hashlib.blake2b(mono.samples.tobytes(), digest_size=8).hexdigest()

        cached = self._reference_cache.get(digest)
        if cached and Path(cached).is_file():
            return cached

        directory = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        path = directory / f"ref_{digest}.wav"
        save_audio(mono, path, bit_depth=32)
        self._reference_cache[digest] = str(path)
        return str(path)

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

        if not text or not text.strip():
            raise SynthesisError("Nothing to synthesise (empty text).")
        if not reference_text or not reference_text.strip():
            raise SynthesisError(
                "F5-TTS needs a transcript of the reference clip.",
                hint="Enable automatic transcription, or type the reference text yourself.",
            )

        settings = self.config.engines.f5
        used_seed = seed_everything(seed)
        reference_path = self._reference_path(reference)

        try:
            wav, sample_rate, _ = self._api.infer(
                ref_file=reference_path,
                ref_text=reference_text.strip(),
                gen_text=text.strip(),
                nfe_step=int(kwargs.get("nfe_step", settings.nfe_step)),
                cfg_strength=float(kwargs.get("cfg_strength", settings.cfg_strength)),
                sway_sampling_coef=float(kwargs.get("sway_sampling_coef", settings.sway_sampling_coef)),
                speed=float(speed if speed else settings.speed),
                cross_fade_duration=float(settings.cross_fade_duration),
                remove_silence=False,
                seed=used_seed,
                # Silence F5's own console chatter; `progress` is left at its
                # default because infer_process calls `progress.tqdm(...)`.
                show_info=lambda *_args, **_kw: None,
            )
        except Exception as exc:
            raise SynthesisError(
                f"F5-TTS synthesis failed: {exc}",
                hint="Try a shorter chunk, a cleaner reference, or the CPU device.",
            ) from exc

        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SynthesisError("F5-TTS returned no audio.")

        return SynthesisResult(
            audio=Audio(samples, int(sample_rate or F5_SAMPLE_RATE), source="f5-tts"),
            engine=self.name,
            seed=used_seed,
            metadata={
                **self._loaded_from,
                "device": str(self.device.device),
                "nfe_step": settings.nfe_step,
                "cfg_strength": settings.cfg_strength,
                "speed": speed,
                "reference_seconds": round(reference.duration, 2),
            },
        )


__all__ = ["F5TTSEngine"]
