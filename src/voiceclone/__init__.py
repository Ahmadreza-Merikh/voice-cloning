"""Persian Voice Cloning Studio.

High-fidelity zero-shot voice cloning with first-class Persian pronunciation.

    from voiceclone import Studio

    studio = Studio()
    result = studio.synthesize("سلام، حال شما چطور است؟", "reference.wav")
    print(result.output_path, result.scores.reference_similarity)

The heavy submodules (engines, pipelines) are imported lazily so that
``import voiceclone`` stays fast and works even when optional engine
dependencies are missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import AppConfig, load_config
from .errors import (
    AudioLoadError,
    ConfigError,
    DependencyMissingError,
    EngineNotAvailableError,
    ReferenceQualityError,
    SynthesisError,
    TextProcessingError,
    VoiceCloneError,
)
from .logging_utils import get_logger, setup_logging

__version__ = "1.0.0"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .audio import Audio
    from .pipelines import CloneResult, Studio, VoiceLibrary

_LAZY: dict[str, tuple[str, str]] = {
    "Studio": (".pipelines", "Studio"),
    "CloneResult": (".pipelines", "CloneResult"),
    "VoiceLibrary": (".pipelines", "VoiceLibrary"),
    "VoiceProfile": (".pipelines", "VoiceProfile"),
    "Audio": (".audio", "Audio"),
    "load_audio": (".audio", "load_audio"),
    "save_audio": (".audio", "save_audio"),
    "TextPipeline": (".text", "TextPipeline"),
    "PersianG2P": (".text", "PersianG2P"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for the expensive submodules."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'voiceclone' has no attribute {name!r}")

    import importlib

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "AppConfig",
    "Audio",
    "AudioLoadError",
    "CloneResult",
    "ConfigError",
    "DependencyMissingError",
    "EngineNotAvailableError",
    "PersianG2P",
    "ReferenceQualityError",
    "Studio",
    "SynthesisError",
    "TextPipeline",
    "TextProcessingError",
    "VoiceCloneError",
    "VoiceLibrary",
    "VoiceProfile",
    "__version__",
    "get_logger",
    "load_audio",
    "load_config",
    "save_audio",
    "setup_logging",
]
