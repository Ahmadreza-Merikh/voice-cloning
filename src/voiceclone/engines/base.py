"""Engine interfaces and the engine registry.

Two abstractions:

* :class:`TTSEngine` - text (or phonemes) + reference voice -> speech
* :class:`VCEngine`  - source speech + reference voice -> converted speech

Both load lazily and report their own availability, so a machine that only
has one engine installed still runs the full application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..audio.io import Audio
from ..config import AppConfig
from ..device import DeviceSpec, empty_cache
from ..errors import EngineNotAvailableError
from ..logging_utils import get_logger

log = get_logger("engines")


@dataclass
class EngineCapabilities:
    """What an engine can and cannot do - drives UI state and validation."""

    #: Accepts IPA phoneme strings instead of graphemes.
    accepts_phonemes: bool = False
    #: Needs a transcript of the reference clip (F5-TTS does; MOSS does not).
    requires_reference_text: bool = False
    #: Officially supports Persian rather than approximating it.
    native_persian: bool = False
    #: Can control speaking rate.
    supports_speed: bool = True
    #: Reproducible given a seed.
    supports_seed: bool = True
    #: Approximate weight footprint, for the download planner.
    approx_download_gb: float = 0.0


@dataclass
class SynthesisResult:
    """Rendered audio plus provenance."""

    audio: Audio
    engine: str
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Engine(ABC):
    """Shared lifecycle for every engine."""

    #: Registry key, e.g. ``"f5"``.
    name: ClassVar[str] = "base"
    #: Human-readable label for the UI.
    display_name: ClassVar[str] = "Base engine"
    capabilities: ClassVar[EngineCapabilities] = EngineCapabilities()

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        self.config = config
        self.device = device
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """``(available, reason)``. Reason explains *why not* when False."""
        return True, ""

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def _load(self) -> None:
        """Bring the model into memory. Called once, lazily."""

    def load(self) -> None:
        if self._loaded:
            return
        available, reason = self.is_available()
        if not available:
            raise EngineNotAvailableError(f"Engine '{self.display_name}' is unavailable.", hint=reason)
        log.info("Loading %s on %s ...", self.display_name, self.device)
        self._load()
        self._loaded = True
        log.info("%s ready.", self.display_name)

    def unload(self) -> None:
        """Drop model references and release accelerator memory."""
        self._loaded = False
        empty_cache(self.device)

    def __enter__(self) -> Engine:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()


class TTSEngine(Engine):
    """Text-to-speech with zero-shot voice cloning."""

    @abstractmethod
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
        """Render ``text`` in the voice of ``reference``.

        Args:
            text: graphemes, or IPA when :attr:`capabilities.accepts_phonemes`.
            reference: preprocessed voice prompt.
            reference_text: transcript of ``reference``, when required.
            speed: 1.0 is the engine's natural pace.
            seed: for reproducibility.
        """


class VCEngine(Engine):
    """Speech-to-speech voice conversion."""

    @abstractmethod
    def convert(
        self,
        source: Audio,
        reference: Audio,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Re-voice ``source`` with the timbre of ``reference``."""


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
_TTS_ENGINES: dict[str, type[TTSEngine]] = {}
_VC_ENGINES: dict[str, type[VCEngine]] = {}


def register_tts(cls: type[TTSEngine]) -> type[TTSEngine]:
    """Class decorator: add a TTS engine to the registry."""
    _TTS_ENGINES[cls.name] = cls
    return cls


def register_vc(cls: type[VCEngine]) -> type[VCEngine]:
    _VC_ENGINES[cls.name] = cls
    return cls


def get_tts_engine(name: str, config: AppConfig, device: DeviceSpec) -> TTSEngine:
    cls = _TTS_ENGINES.get(name.lower())
    if cls is None:
        known = ", ".join(sorted(_TTS_ENGINES)) or "none registered"
        raise EngineNotAvailableError(f"Unknown TTS engine '{name}'.", hint=f"Available: {known}")
    return cls(config, device)


def get_vc_engine(name: str, config: AppConfig, device: DeviceSpec) -> VCEngine:
    cls = _VC_ENGINES.get(name.lower())
    if cls is None:
        known = ", ".join(sorted(_VC_ENGINES)) or "none registered"
        raise EngineNotAvailableError(f"Unknown voice-conversion engine '{name}'.", hint=f"Available: {known}")
    return cls(config, device)


def list_tts_engines() -> dict[str, type[TTSEngine]]:
    return dict(_TTS_ENGINES)


def list_vc_engines() -> dict[str, type[VCEngine]]:
    return dict(_VC_ENGINES)


def engine_status() -> list[dict[str, Any]]:
    """Availability of every registered engine, for the Diagnostics tab."""
    rows: list[dict[str, Any]] = []
    for kind, registry in (("tts", _TTS_ENGINES), ("vc", _VC_ENGINES)):
        for name, cls in sorted(registry.items()):
            available, reason = cls.is_available()
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "display_name": cls.display_name,
                    "available": available,
                    "reason": reason,
                    "native_persian": cls.capabilities.native_persian,
                    "accepts_phonemes": cls.capabilities.accepts_phonemes,
                    "download_gb": cls.capabilities.approx_download_gb,
                }
            )
    return rows


__all__ = [
    "Engine",
    "EngineCapabilities",
    "SynthesisResult",
    "TTSEngine",
    "VCEngine",
    "engine_status",
    "get_tts_engine",
    "get_vc_engine",
    "list_tts_engines",
    "list_vc_engines",
    "register_tts",
    "register_vc",
]
