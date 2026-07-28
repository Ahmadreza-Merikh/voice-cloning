"""Typed errors for the voice cloning stack.

Every failure the user can plausibly cause or fix gets its own class, so the
UI can turn it into an actionable message instead of a stack trace.
"""

from __future__ import annotations


class VoiceCloneError(Exception):
    """Base class for all errors raised by this project."""

    #: Short, user-facing hint rendered under the error in the UI.
    hint: str = ""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if hint is not None:
            self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\n  -> {self.hint}"
        return self.message


class ConfigError(VoiceCloneError):
    """Malformed or contradictory configuration."""


class DependencyMissingError(VoiceCloneError):
    """An optional dependency is required for the requested feature."""

    def __init__(self, package: str, feature: str, install: str | None = None) -> None:
        install = install or f"pip install {package}"
        super().__init__(
            f"'{feature}' requires the optional package '{package}', which is not installed.",
            hint=f"Install it with:  {install}",
        )
        self.package = package
        self.feature = feature


class AudioLoadError(VoiceCloneError):
    """The audio file could not be decoded."""


class ReferenceQualityError(VoiceCloneError):
    """The reference recording is unusable for cloning."""


class TextProcessingError(VoiceCloneError):
    """Text normalisation / G2P failed."""


class EngineNotAvailableError(VoiceCloneError):
    """A synthesis engine cannot be constructed on this machine."""


class SynthesisError(VoiceCloneError):
    """The engine ran but did not produce usable audio."""
