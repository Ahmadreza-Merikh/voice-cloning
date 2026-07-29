"""MOSS-TTS engine - the quality-first path.

MOSS-TTS is an autoregressive discrete-token TTS foundation model with:

* **official Persian support** (one of 20 trained languages, not an accident
  of multilingual pretraining)
* **zero-shot voice cloning** from a short reference clip
* **IPA phoneme input**, which is what makes it the right partner for the
  Persian G2P front end: short vowels, ezafe and homographs are resolved
  *before* the model ever sees the text, so it has nothing left to guess
* Apache-2.0 licensing

Two sizes are published. The default here is the 1.7B "Local" variant, which
fits in roughly 3.5 GB at bfloat16 and therefore runs on a 16 GB laptop. On a
24 GB+ GPU, point ``engines.moss.model_id`` at ``OpenMOSS-Team/MOSS-TTS`` for
the 8B flagship.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.io import Audio, save_audio
from ..config import AppConfig
from ..device import DeviceSpec, seed_everything
from ..errors import EngineNotAvailableError, SynthesisError
from ..logging_utils import get_logger
from .base import EngineCapabilities, SynthesisResult, TTSEngine, register_tts

log = get_logger("engines.moss")

#: Minimum transformers version that ships the MOSS model classes.
_MIN_TRANSFORMERS = (5, 0)


#: ``MossTTSDelayConfig`` copies ``hidden_size`` and ``vocab_size`` up from its
#: Qwen3 ``language_config`` but stops there. Transformers allocates the KV
#: cache for the backbone from the *top-level* config
#: (``cache_utils.get_layer_types_and_kwargs``), so generation dies on the
#: first attribute it does not find. These are the fields that path consults.
_LANGUAGE_CONFIG_MIRRORED = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "sliding_window",
    "max_position_embeddings",
)


def _mirror_language_config(config: Any) -> None:
    """Copy backbone geometry onto the wrapper config, where absent.

    Only fills gaps: anything MOSS sets deliberately is left alone.
    """
    language_config = getattr(config, "language_config", None)
    if language_config is None:
        return
    for name in _LANGUAGE_CONFIG_MIRRORED:
        if getattr(config, name, None) is None and hasattr(language_config, name):
            setattr(config, name, getattr(language_config, name))


@contextlib.contextmanager
def _hub_cache(cache_dir: str) -> Iterator[None]:
    """Point huggingface_hub at ``cache_dir`` for the duration of the block.

    ``ProcessorMixin.from_pretrained`` forwards whatever kwargs it does not
    consume straight into the processor constructor, which accepts only its
    declared sub-processors and raises ``TypeError: Unexpected keyword
    argument cache_dir``. The download layer reads ``HF_HUB_CACHE`` from the
    environment, so routing the cache that way keeps the weights in our
    directory without the kwarg ever reaching ``__init__``.
    """
    previous = os.environ.get("HF_HUB_CACHE")
    os.environ["HF_HUB_CACHE"] = cache_dir
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = previous


def _transformers_version() -> tuple[int, ...]:
    try:
        import transformers  # type: ignore

        return tuple(int(part) for part in transformers.__version__.split(".")[:2])
    except Exception:
        return (0,)


@register_tts
class MossTTSEngine(TTSEngine):
    name = "moss"
    display_name = "MOSS-TTS (Persian + IPA control)"
    capabilities = EngineCapabilities(
        accepts_phonemes=True,
        requires_reference_text=False,
        native_persian=True,
        supports_speed=False,  # pacing is controlled with token budgets, not a rate
        supports_seed=True,
        approx_download_gb=6.2,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self._model: Any = None
        self._processor: Any = None
        self._sample_rate = 24000
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("transformers") is None:
            return False, "pip install 'transformers>=5.0.0' accelerate"
        version = _transformers_version()
        if version < _MIN_TRANSFORMERS:
            found = ".".join(str(part) for part in version)
            return False, f"needs transformers>=5.0 (found {found}): pip install -U transformers"
        return True, ""

    # -- loading -----------------------------------------------------------
    def _resolve_dtype(self) -> Any:
        import torch

        # bfloat16 halves the footprint and is well supported on both CUDA and
        # recent MPS builds. CPU stays float32: bf16 matmuls there are slow.
        if self.device.is_cuda:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.device.is_mps:
            return torch.bfloat16
        return torch.float32

    def _attn_implementation(self) -> str:
        if self.device.is_cuda:
            if importlib.util.find_spec("flash_attn") is not None:
                try:
                    import torch

                    if torch.cuda.get_device_capability()[0] >= 8:
                        return "flash_attention_2"
                except Exception:
                    pass
            return "sdpa"
        # MPS and CPU: eager is the reliable choice for custom modelling code.
        return "eager"

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor  # type: ignore

        settings = self.config.engines.moss
        cache_dir = str(self.config.paths.ensure("cache_dir"))
        dtype = self._resolve_dtype()

        log.info("Loading MOSS-TTS '%s' (%s, %s) ...", settings.model_id, self.device.device, dtype)

        try:
            with _hub_cache(cache_dir):
                self._processor = AutoProcessor.from_pretrained(
                    settings.model_id, trust_remote_code=True
                )
            # The audio tokenizer is a separate module and must move explicitly.
            if hasattr(self._processor, "audio_tokenizer"):
                self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(self.device.device)

            self._model = AutoModel.from_pretrained(
                settings.model_id,
                trust_remote_code=True,
                attn_implementation=self._attn_implementation(),
                torch_dtype=dtype,
                cache_dir=cache_dir,
            ).to(self.device.device)
            self._model.eval()
            _mirror_language_config(self._model.config)
        except Exception as exc:
            raise EngineNotAvailableError(
                f"Could not load MOSS-TTS ({settings.model_id}): {exc}",
                hint=(
                    "The 1.7B model needs ~6.2 GB of download and ~3.5 GB of memory. "
                    "Run  python scripts/download_models.py --engine moss  first, "
                    "or switch to the F5-TTS engine."
                ),
            ) from exc

        model_config = getattr(self._processor, "model_config", None)
        self._sample_rate = int(getattr(model_config, "sampling_rate", 24000) or 24000)
        self._tempdir = tempfile.TemporaryDirectory(prefix="voiceclone_moss_")

        # Some builds warn about SDPA kernels that are broken on CUDA.
        if self.device.is_cuda and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        super().unload()

    # -- inference ---------------------------------------------------------
    @staticmethod
    def format_ipa(phonemes: str) -> str:
        """Wrap an IPA string in the slashes MOSS-TTS uses to mark phonemes."""
        cleaned = (phonemes or "").strip().strip("/")
        return f"/{cleaned}/" if cleaned else ""

    def _write_reference(self, reference: Audio) -> str:
        directory = Path(self._tempdir.name) if self._tempdir else Path(tempfile.gettempdir())
        path = directory / "reference.wav"
        save_audio(reference.to_mono().resample(self._sample_rate), path, bit_depth=32)
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

        import torch

        settings = self.config.engines.moss
        used_seed = seed_everything(seed)

        # Phonemes arrive already converted to IPA by the text pipeline; the
        # slash delimiters tell MOSS to read them as phonemes, not letters.
        payload = self.format_ipa(text) if kwargs.get("is_phonemes") else text.strip()
        reference_path = self._write_reference(reference)

        try:
            message = self._processor.build_user_message(text=payload, reference=[reference_path])
            batch = self._processor([[message]], mode="generation")

            input_ids = batch["input_ids"].to(self.device.device)
            attention_mask = batch["attention_mask"].to(self.device.device)

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": int(kwargs.get("max_new_tokens", settings.max_new_tokens)),
            }
            # These are MOSS-specific sampling controls; older builds may not
            # accept them, so degrade rather than fail.
            tuning = {
                "audio_temperature": float(settings.temperature),
                "audio_top_p": float(settings.top_p),
                "audio_top_k": int(settings.top_k),
                "audio_repetition_penalty": float(settings.repetition_penalty),
            }

            with torch.no_grad():
                try:
                    outputs = self._model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs, **tuning
                    )
                except TypeError:
                    log.debug("MOSS sampling kwargs rejected; using defaults.", exc_info=True)
                    outputs = self._model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
                    )

            decoded = list(self._processor.decode(outputs))
        except Exception as exc:
            raise SynthesisError(
                f"MOSS-TTS synthesis failed: {exc}",
                hint="Try a shorter chunk, or reduce engines.moss.max_new_tokens.",
            ) from exc

        if not decoded:
            raise SynthesisError("MOSS-TTS returned no audio.")

        codes = getattr(decoded[0], "audio_codes_list", None)
        if not codes:
            raise SynthesisError("MOSS-TTS produced a message without audio.")

        waveform = codes[0]
        samples = waveform.detach().to(torch.float32).cpu().numpy().reshape(-1)
        if samples.size == 0:
            raise SynthesisError("MOSS-TTS returned an empty waveform.")

        return SynthesisResult(
            audio=Audio(np.asarray(samples, dtype=np.float32), self._sample_rate, source="moss-tts"),
            engine=self.name,
            seed=used_seed,
            metadata={
                "model_id": settings.model_id,
                "phonemes": bool(kwargs.get("is_phonemes")),
                "reference_seconds": round(reference.duration, 2),
            },
        )


__all__ = ["MossTTSEngine"]
