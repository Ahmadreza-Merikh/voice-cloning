"""The end-to-end text preparation pipeline.

    raw text
      -> language detection
      -> Persian normalisation (characters, ZWNJ, punctuation)
      -> number / date / currency verbalisation
      -> chunking on sentence boundaries
      -> per-chunk G2P to IPA (for engines that accept phonemes)
      -> PreparedText

Chunking happens *before* G2P on purpose: the G2P model disambiguates
homographs from sentence context, so it should see whole sentences, not the
entire document at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import TextConfig
from ..logging_utils import get_logger
from .chunker import TextChunk, chunk_text, estimate_duration
from .g2p import PersianG2P, get_g2p
from .lang import detect_language, is_mixed_script
from .normalize_fa import NormalizationOptions, normalize

log = get_logger("text.pipeline")


@dataclass
class PreparedText:
    """Everything the synthesis pipeline needs to know about the input text."""

    original: str
    normalized: str
    language: str
    chunks: list[TextChunk] = field(default_factory=list)
    used_phonemes: bool = False
    g2p_backend: str = "disabled"
    notes: list[str] = field(default_factory=list)

    @property
    def estimated_seconds(self) -> float:
        return sum(chunk.estimated_seconds for chunk in self.chunks)

    @property
    def phoneme_preview(self) -> str:
        return " | ".join(chunk.phonemes for chunk in self.chunks if chunk.phonemes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "normalized": self.normalized,
            "chunks": len(self.chunks),
            "estimated_seconds": round(self.estimated_seconds, 1),
            "used_phonemes": self.used_phonemes,
            "g2p_backend": self.g2p_backend,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Language: {self.language} | {len(self.chunks)} chunk(s) | "
            f"~{self.estimated_seconds:.0f}s estimated",
            f"G2P: {self.g2p_backend}"
            + (" (phonemes fed to the engine)" if self.used_phonemes else " (graphemes fed to the engine)"),
        ]
        lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


class TextPipeline:
    """Turns arbitrary user text into engine-ready chunks."""

    def __init__(self, config: TextConfig) -> None:
        self.config = config
        self._g2p: PersianG2P | None = None

    # -- G2P ---------------------------------------------------------------
    @property
    def g2p(self) -> PersianG2P | None:
        """The G2P bridge, or ``None`` when disabled in config."""
        setting = self.config.g2p.enabled
        if setting is False or str(setting).lower() in {"false", "off", "no"}:
            return None
        if self._g2p is None:
            repo = self.config.g2p.repo_path
            self._g2p = get_g2p(
                None if not repo or repo == "auto" else str(repo),
                inject_ezafe=bool(self.config.g2p.inject_ezafe),
            )
        return self._g2p

    def g2p_status(self) -> str:
        engine = self.g2p
        return engine.status if engine is not None else "disabled in config"

    def should_use_phonemes(self, engine_name: str) -> bool:
        """Whether ``engine_name`` should be fed phonemes rather than script.

        Grapheme-only tokenizers (F5-TTS's Persian character vocabulary, for
        instance) would treat IPA as out-of-vocabulary noise, so phonemes are
        opt-in per engine.
        """
        if str(self.config.g2p.output).lower() == "off":
            return False
        allowed = {name.lower() for name in (self.config.g2p.apply_to_engines or [])}
        return engine_name.lower() in allowed

    # -- main entry point --------------------------------------------------
    def prepare(
        self,
        text: str,
        *,
        engine_name: str = "",
        language: str | None = None,
        use_phonemes: bool | None = None,
    ) -> PreparedText:
        """Normalise, chunk and (optionally) phonemise ``text``."""
        original = text or ""
        notes: list[str] = []

        if not original.strip():
            return PreparedText(original, "", language or "fa", [], False, "disabled", ["Empty input"])

        # -- language ------------------------------------------------------
        configured = (language or self.config.language or "auto").lower()
        detected = detect_language(original) if configured == "auto" else configured
        if is_mixed_script(original):
            notes.append("Mixed Persian/Latin script detected.")

        # -- normalisation -------------------------------------------------
        if self.config.normalize:
            options = NormalizationOptions(
                expand_numbers=bool(self.config.expand_numbers),
                strip_bracketed_tags=bool(self.config.strip_bracketed_tags),
                # The G2P wants diacritics; grapheme engines do not care.
                remove_diacritics=False,
            )
            normalized = normalize(original, options)
            if normalized != original.strip():
                notes.append("Text normalised (characters, spacing, numbers).")
        else:
            normalized = original.strip()

        if not normalized:
            return PreparedText(original, "", detected, [], False, "disabled", ["Nothing left after normalisation"])

        # -- chunking ------------------------------------------------------
        chunks = chunk_text(
            normalized,
            max_chars=self.config.chunk.max_chars,
            max_seconds=self.config.chunk.max_seconds,
            min_chars=self.config.chunk.min_chars,
        )
        if len(chunks) > 1:
            notes.append(f"Split into {len(chunks)} chunks for stable long-form quality.")

        # -- phonemisation -------------------------------------------------
        wants_phonemes = (
            use_phonemes if use_phonemes is not None else self.should_use_phonemes(engine_name)
        )
        used_phonemes = False
        backend = "disabled"

        if wants_phonemes and detected == "fa":
            engine = self.g2p
            if engine is None:
                backend = "disabled in config"
                notes.append("G2P disabled; the engine will read raw Persian script.")
            else:
                used_phonemes = self._apply_g2p(chunks, notes)
                backend = engine.backend if used_phonemes else engine.status
        elif wants_phonemes and detected != "fa":
            backend = "skipped (non-Persian text)"
        elif engine_name:
            backend = "not used by this engine"

        prepared = PreparedText(
            original=original,
            normalized=normalized,
            language=detected,
            chunks=chunks,
            used_phonemes=used_phonemes,
            g2p_backend=backend,
            notes=notes,
        )
        log.info(
            "Text prepared: %s, %d chunk(s), ~%.0fs, phonemes=%s",
            detected,
            len(chunks),
            prepared.estimated_seconds,
            used_phonemes,
        )
        return prepared

    def _apply_g2p(self, chunks: list[TextChunk], notes: list[str]) -> bool:
        """Phonemise each chunk in place. Returns True if any chunk got IPA."""
        engine = self.g2p
        if engine is None:
            return False

        from .g2p import words_to_ipa

        succeeded = 0
        for chunk in chunks:
            try:
                result = engine.phonemize(chunk.text)
            except Exception:
                log.warning("G2P failed on chunk %d; using graphemes.", chunk.index, exc_info=True)
                continue

            if not result.phonemes:
                continue

            # Word-aligned output keeps phrasing; flat output is still usable.
            chunk.phonemes = words_to_ipa(result.words) if result.words else result.phonemes.replace(" ", "")
            succeeded += 1

        if not succeeded:
            notes.append(f"G2P unavailable ({engine.status}); reading raw Persian script instead.")
            return False

        if succeeded < len(chunks):
            notes.append(f"G2P applied to {succeeded}/{len(chunks)} chunks.")
        else:
            notes.append("Persian phonemes (IPA) generated - short vowels, ezafe and homographs resolved.")
        return True

    # -- helpers -----------------------------------------------------------
    def preview(self, text: str, *, engine_name: str = "") -> dict[str, Any]:
        """Dry-run the pipeline, for the UI's text-inspection panel."""
        prepared = self.prepare(text, engine_name=engine_name)
        return {
            **prepared.to_dict(),
            "chunk_texts": [chunk.text for chunk in prepared.chunks],
            "chunk_phonemes": [chunk.phonemes for chunk in prepared.chunks],
        }


def estimate_synthesis_seconds(text: str) -> float:
    """Public helper used by the UI to warn about very long inputs."""
    return estimate_duration(text)


__all__ = ["PreparedText", "TextPipeline", "estimate_synthesis_seconds"]
