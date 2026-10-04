"""The end-to-end text preparation pipeline.

    raw text
      -> language detection
      -> Persian normalisation (characters, ZWNJ, punctuation)
      -> number / date / currency verbalisation
      -> chunking on sentence boundaries
      -> per-chunk G2P to IPA (for engines that accept phonemes)
      -> the ezafe written in (for engines that read marks from the script)
      -> the first vowel of a homograph, from Homo-GE2PE (the same engines)
      -> PreparedText

Chunking happens *before* G2P on purpose: the G2P model disambiguates
homographs from sentence context, so it should see whole sentences, not the
entire document at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..config import TextConfig
from ..logging_utils import get_logger
from .chunker import TextChunk, chunk_text, estimate_duration
from .ezafe_fa import EzafeWriter
from .g2p import PersianG2P, get_g2p
from .homographs_fa import HomographWriter
from .lang import detect_language, is_mixed_script
from .normalize_fa import NormalizationOptions, normalize
from .prosody_fa import break_vocative_clauses

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
        self._ezafe: EzafeWriter | None = None
        self._homographs: HomographWriter | None = None

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

    def reads_marks(self, engine_name: str) -> bool:
        """Whether ``engine_name`` reads Persian marks (harakat, hamza) from the script."""
        return engine_name.lower() in {name.lower() for name in (self.config.marks_engines or [])}

    @property
    def ezafe(self) -> EzafeWriter:
        if self._ezafe is None:
            self._ezafe = EzafeWriter(self.config.ezafe)
        return self._ezafe

    @property
    def homographs(self) -> HomographWriter:
        if self._homographs is None:
            self._homographs = HomographWriter(self.config.homographs)
        return self._homographs

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
        reads_marks = self.reads_marks(engine_name)
        if self.config.normalize:
            options = NormalizationOptions(
                expand_numbers=bool(self.config.expand_numbers),
                strip_bracketed_tags=bool(self.config.strip_bracketed_tags),
                # The G2P wants diacritics; grapheme engines do not care.
                remove_diacritics=False,
                # An engine that reads marks gets the ones the writer put there.
                keep_hamza_alef=reads_marks,
                ezafe_heh_as_hamza=reads_marks,
            )
            if detected == "en":
                # Every other step writes Persian: English text came out with
                # "،" for "," and "۲۳۰۰۰" read as "بیست و سه هزار".
                options = replace(options, unify_characters=False, fix_zwnj=False,
                                  normalize_punctuation=False, expand_abbreviations=False,
                                  expand_numbers=False)
            normalized = normalize(original, options)
            if normalized != original.strip():
                notes.append("Text normalised (characters, spacing, numbers).")
        else:
            normalized = original.strip()

        if not normalized:
            return PreparedText(original, "", detected, [], False, "disabled", ["Nothing left after normalisation"])

        # -- vocative / clause boundary ------------------------------------
        # Unmarked "سلام من …" is read as ezafe (سلامِ من) by grapheme engines.
        # A comma is a clause boundary; ezafe cannot cross it. Noun-phrase
        # ezafe (کتاب من) is not a vocative and is left alone.
        if detected == "fa":
            with_breaks = break_vocative_clauses(normalized)
            if with_breaks != normalized:
                notes.append(
                    "Inserted a pause after a greeting so it is not read as ezafe "
                    "(سلام من → سلام، من)."
                )
                normalized = with_breaks

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
                used_phonemes = self._apply_g2p(chunks, notes, engine_name)
                backend = engine.backend if used_phonemes else engine.status
        elif wants_phonemes and detected != "fa":
            backend = "skipped (non-Persian text)"
        elif engine_name:
            backend = "not used by this engine"

        # -- ezafe -----------------------------------------------------------
        marks_apply = detected == "fa" and reads_marks and not used_phonemes
        plains = [chunk.text for chunk in chunks]
        if marks_apply and self.ezafe.enabled:
            texts, written, failed = self.ezafe.write([chunk.text for chunk in chunks])
            for chunk, text in zip(chunks, texts, strict=True):
                chunk.text = text
            if failed:
                notes.append(f"Ezafe not written ({failed}); the engine reads plain script.")
            elif written:
                notes.append(f"Ezafe written in on {len(written)} word(s) by hazm's ParsBERT tagger: "
                             + "، ".join(written[:12]) + ("…" if len(written) > 12 else "") + ".")

        # -- homograph vowels ------------------------------------------------
        if marks_apply and self.homographs.enabled:
            texts, written, failed = self.homographs.write(plains, [chunk.text for chunk in chunks], self.g2p)
            for chunk, text in zip(chunks, texts, strict=True):
                chunk.text = text
            if failed:
                notes.append(f"Homograph vowels not written ({failed}).")
            elif written:
                notes.append(f"First vowel written on {len(written)} homograph(s) from Homo-GE2PE: "
                             + "، ".join(written[:12]) + ("…" if len(written) > 12 else "") + ".")

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

    def _apply_g2p(self, chunks: list[TextChunk], notes: list[str], engine_name: str = "") -> bool:
        """Phonemise each chunk in place. Returns True if any chunk got IPA.

        ``engine_name`` selects the canonical normaliser: for the IPA
        fine-tune (``omnivoice_ipa``) the GE2PE output is converted into that
        model's convention; for every other engine it passes through exactly
        as it did before.
        """
        engine = self.g2p
        if engine is None:
            return False

        from .g2p import words_to_ipa
        from .ipa_normalize import normalizer_for

        normalize = normalizer_for(engine_name)
        succeeded = 0
        changed = 0
        unknown: set[str] = set()

        for chunk in chunks:
            try:
                result = engine.phonemize(chunk.text)
            except Exception:
                log.warning("G2P failed on chunk %d; using graphemes.", chunk.index, exc_info=True)
                continue

            if not result.phonemes:
                continue

            rendered = None
            if normalize is not None and result.words:
                normalized = normalize(result.words, compact_words=result.compact_words)
                if normalized.text:
                    rendered = normalized.text
                    changed += sum(item.count for item in normalized.changes)
                    unknown.update(normalized.unknown)

            # Word-aligned output keeps phrasing; flat output is still usable.
            if rendered is None:
                rendered = (
                    words_to_ipa(result.words) if result.words else result.phonemes.replace(" ", "")
                )
            chunk.phonemes = rendered
            succeeded += 1

        if not succeeded:
            notes.append(f"G2P unavailable ({engine.status}); reading raw Persian script instead.")
            return False

        if succeeded < len(chunks):
            notes.append(f"G2P applied to {succeeded}/{len(chunks)} chunks.")
        else:
            notes.append("Persian phonemes (IPA) generated - short vowels, ezafe and homographs resolved.")

        if normalize is not None and succeeded:
            if changed:
                notes.append(
                    f"IPA canonicalised for {engine_name}: {changed} symbol change(s) "
                    f"({engine.backend} output -> the convention the fine-tune was trained on)."
                )
            if unknown:
                notes.append(
                    f"Unknown IPA symbol(s) kept and reported: {', '.join(sorted(unknown))}."
                )
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
