"""The voice library.

A saved voice is a preprocessed reference clip plus its transcript and quality
report. Saving one means the expensive work - decoding, denoising, window
selection, transcription - happens once instead of on every synthesis, and it
guarantees that repeat renders of the same character actually match.

Everything lives under ``paths.voices_dir`` as plain WAV + JSON, so a voice is
easy to inspect, copy between machines, or delete.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..audio.io import Audio, load_audio, save_audio
from ..errors import VoiceCloneError
from ..logging_utils import get_logger

log = get_logger("voices")

_SAFE_NAME_RE = re.compile(r"[^\w؀-ۿ\- ]+", re.UNICODE)
_MANIFEST = "voice.json"
_AUDIO = "reference.wav"


def slugify(name: str) -> str:
    """Filesystem-safe identifier that still allows Persian names."""
    cleaned = _SAFE_NAME_RE.sub("", (name or "").strip())
    cleaned = re.sub(r"\s+", "-", cleaned).strip("-")
    return cleaned or f"voice-{int(time.time())}"


@dataclass
class VoiceProfile:
    """A saved reference voice."""

    name: str
    slug: str
    transcript: str = ""
    language: str = "fa"
    notes: str = ""
    created_at: str = ""
    duration: float = 0.0
    sample_rate: int = 24000
    quality_score: float = 0.0
    quality_verdict: str = ""
    preprocessing: list[str] = field(default_factory=list)
    #: Absolute path to the stored audio, filled in on load.
    audio_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceProfile:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    @property
    def label(self) -> str:
        """Dropdown label: name, length and quality at a glance."""
        parts = [self.name]
        if self.duration:
            parts.append(f"{self.duration:.0f}s")
        if self.quality_verdict:
            parts.append(self.quality_verdict)
        return " - ".join(parts)


class VoiceLibrary:
    """Store and retrieve reference voices on disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def _directory(self, slug: str) -> Path:
        return self.root / slug

    def exists(self, slug: str) -> bool:
        return (self._directory(slug) / _MANIFEST).is_file()

    # -- write -------------------------------------------------------------
    def save(
        self,
        name: str,
        audio: Audio,
        *,
        transcript: str = "",
        language: str = "fa",
        notes: str = "",
        quality_score: float = 0.0,
        quality_verdict: str = "",
        preprocessing: list[str] | None = None,
        overwrite: bool = True,
    ) -> VoiceProfile:
        """Persist a preprocessed reference clip under ``name``."""
        if audio.num_frames == 0:
            raise VoiceCloneError("Cannot save an empty voice.")

        slug = slugify(name)
        directory = self._directory(slug)
        if directory.exists() and not overwrite:
            raise VoiceCloneError(
                f"A voice named '{name}' already exists.",
                hint="Pick another name, or enable overwrite.",
            )
        directory.mkdir(parents=True, exist_ok=True)

        audio_path = save_audio(audio.to_mono(), directory / _AUDIO, bit_depth=24)

        profile = VoiceProfile(
            name=name.strip() or slug,
            slug=slug,
            transcript=transcript.strip(),
            language=language,
            notes=notes.strip(),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=round(audio.duration, 2),
            sample_rate=audio.sample_rate,
            quality_score=round(quality_score, 1),
            quality_verdict=quality_verdict,
            preprocessing=list(preprocessing or []),
            audio_path=str(audio_path),
        )

        (directory / _MANIFEST).write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Saved voice '%s' (%.1fs) -> %s", profile.name, profile.duration, directory)
        return profile

    # -- read --------------------------------------------------------------
    def load(self, slug: str) -> tuple[VoiceProfile, Audio]:
        """Load a voice by slug. Raises if it is missing or corrupt."""
        directory = self._directory(slug)
        manifest = directory / _MANIFEST
        if not manifest.is_file():
            raise VoiceCloneError(
                f"Voice '{slug}' not found.",
                hint=f"Available: {', '.join(profile.slug for profile in self.list()) or 'none'}",
            )

        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VoiceCloneError(f"Voice manifest for '{slug}' is corrupt: {exc}") from exc

        profile = VoiceProfile.from_dict(data)
        audio_path = directory / _AUDIO
        if not audio_path.is_file():
            raise VoiceCloneError(f"Voice '{slug}' has no audio file at {audio_path}.")

        profile.audio_path = str(audio_path)
        return profile, load_audio(audio_path, mono=True)

    def list(self) -> list[VoiceProfile]:
        """Every saved voice, newest first."""
        profiles: list[VoiceProfile] = []
        for manifest in sorted(self.root.glob(f"*/{_MANIFEST}")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                profile = VoiceProfile.from_dict(data)
                profile.audio_path = str(manifest.parent / _AUDIO)
                profiles.append(profile)
            except Exception:
                log.warning("Skipping unreadable voice manifest: %s", manifest, exc_info=True)

        profiles.sort(key=lambda item: item.created_at, reverse=True)
        return profiles

    def choices(self) -> list[tuple[str, str]]:
        """``(label, slug)`` pairs for a Gradio dropdown."""
        return [(profile.label, profile.slug) for profile in self.list()]

    # -- delete ------------------------------------------------------------
    def delete(self, slug: str) -> bool:
        """Remove a saved voice. Returns False when it did not exist."""
        directory = self._directory(slug)
        if not directory.is_dir():
            return False

        # Only ever remove the files this class writes.
        for name in (_MANIFEST, _AUDIO):
            (directory / name).unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            log.warning("Voice directory %s is not empty; left in place.", directory)
        log.info("Deleted voice '%s'", slug)
        return True


__all__ = ["VoiceLibrary", "VoiceProfile", "slugify"]
