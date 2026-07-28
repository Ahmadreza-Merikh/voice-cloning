"""Persian grapheme-to-phoneme conversion.

Why this matters more than anything else in the Persian pipeline
----------------------------------------------------------------
Persian orthography does not write short vowels, marks the ezafe construction
with nothing at all, and is full of homographs:

    گل    -> /gol/  (flower)   or  /gel/  (mud)
    مرد   -> /mærd/ (man)      or  /mord/ (died)
    کتاب من -> /ketɒbe mæn/    ("my book" - the /e/ is invisible in writing)

A TTS model fed raw Persian script has to guess all of this from context. A
model fed *phonemes* does not have to guess at all. This module bridges to the
Homo-GE2PE based Persian G2P system (76.9% homograph accuracy, 3.98% PER) and
converts its output into standard IPA that phoneme-aware engines accept.

The bridge is entirely optional: if the G2P project is not installed, the rest
of the pipeline runs on graphemes and says so in Diagnostics.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..logging_utils import get_logger
from .normalize_fa import normalize_for_g2p

log = get_logger("text.g2p")

#: Marker the flat G2P API emits for the Kasra-e-Ezafe (a short /e/).
EZAFE_TOKEN = "ezafe"

# --------------------------------------------------------------------------
# GE2PE compact romanisation -> IPA
# --------------------------------------------------------------------------
#  GE2PE emits a compact ASCII romanisation with several non-obvious symbol
#  choices. This table was verified empirically against the model rather than
#  assumed, because three of the mappings are counter-intuitive:
#
#      شکفت -> $ekoft      $ is ش  /ʃ/      (not a literal dollar sign)
#      ژاله -> ;ale        ; is ژ  /ʒ/
#      جان  -> jan         j is ج  /dʒ/     (NOT the palatal glide)
#      یار  -> yar         y is ی  /j/      (the glide is `y`)
#      چای  -> cay         c is چ  /tʃ/     (not /k/)
#      آب   -> @ab         @ marks an alef onset; not a phoneme
#      اسب  -> @/sb        / is the short vowel æ
#      قند  -> q/nd        ق and غ both collapse to q
#
#  Getting `j` or `c` wrong turns جان into "yân" and چای into "kâi", so these
#  are load-bearing.
COMPACT_DIGRAPHS: tuple[tuple[str, str], ...] = (
    ("tʃ", "tʃ"),
    ("dʒ", "dʒ"),
    # Latin digraphs, for hand-written romanisation fed in directly. GE2PE
    # itself uses the single-character forms above.
    ("ch", "tʃ"),
    ("zh", "ʒ"),
    ("sh", "ʃ"),
    ("kh", "x"),
    ("gh", "ɣ"),
    ("qh", "q"),
)

COMPACT_TO_IPA: dict[str, str] = {
    # -- consonants --------------------------------------------------------
    "b": "b", "p": "p", "t": "t", "d": "d", "k": "k", "g": "g",
    "q": "q", "f": "f", "v": "v", "w": "v", "s": "s", "z": "z",
    "x": "x", "h": "h", "m": "m", "n": "n", "l": "l", "r": "r",
    "$": "ʃ",    # ش
    ";": "ʒ",    # ژ
    "j": "dʒ",   # ج
    "c": "tʃ",   # چ
    "y": "j",    # ی (the glide)
    "'": "ʔ", "ʔ": "ʔ",
    "ʃ": "ʃ", "ʒ": "ʒ", "ɣ": "ɣ",
    # -- vowels ------------------------------------------------------------
    # GE2PE writes the long back vowel (آب) as `a` and the short one (اَسب)
    # as `/`. Rendering them as ɒ / æ is what makes the IPA correct Persian
    # rather than English-flavoured.
    "a": "ɒ",
    "/": "æ",
    "æ": "æ",
    "i": "i",
    "e": "e",
    "o": "o",
    "u": "u",
}

#: Bookkeeping symbols, not phonemes. `@` marks an alef onset and `1` is a
#: stress marker; neither is pronounced.
_COMPACT_NOISE = frozenset("@1. ")

#: Where to look for the Persian G2P research project.
_REPO_CANDIDATES = (
    "~/Desktop/G2P Research",
    "~/G2P Research",
    "~/Documents/G2P Research",
    "~/projects/G2P Research",
    "~/Desktop/g2p-research",
    "~/g2p-research",
)


@dataclass
class G2PResult:
    """Phonemised text alongside the input it came from."""

    text: str
    phonemes: str
    words: list[tuple[str, str]]
    backend: str

    @property
    def has_ezafe(self) -> bool:
        return any("ezafe" in phones for _, phones in self.words)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.phonemes


# --------------------------------------------------------------------------
# Repo discovery
# --------------------------------------------------------------------------
def find_g2p_repo(configured: str | None = None) -> Path | None:
    """Locate the Persian G2P project directory.

    Order: explicit config -> ``$VOICECLONE_G2P_REPO`` -> known locations.
    """
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_G2P_REPO")
    if env:
        candidates.append(env)
    candidates.extend(_REPO_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_dir():
            continue
        # Must actually look like the project.
        if (path / "persian_g2p.py").is_file() or (path / "scripts" / "homo_ge2pe_backend.py").is_file():
            return path

    return None


def compact_word_to_ipa(word: str) -> str:
    """Convert one GE2PE compact word into space-separated IPA phones."""
    cleaned = (word or "").strip().lower()
    if not cleaned:
        return ""

    phones: list[str] = []
    index = 0
    while index < len(cleaned):
        for source, target in COMPACT_DIGRAPHS:
            if cleaned.startswith(source, index):
                phones.append(target)
                index += len(source)
                break
        else:
            char = cleaned[index]
            if char in COMPACT_TO_IPA:
                phones.append(COMPACT_TO_IPA[char])
            elif char not in _COMPACT_NOISE:
                log.debug("Dropping unmapped G2P symbol %r", char)
            index += 1

    return " ".join(phones)


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------
class PersianG2P:
    """Wrapper around the Homo-GE2PE Persian G2P engine.

    Loads lazily: constructing this object is cheap, the first
    :meth:`phonemize` call pays for the model.
    """

    def __init__(
        self,
        repo_path: str | Path | None = None,
        *,
        inject_ezafe: bool = True,
        use_gpu: bool = False,
    ) -> None:
        self.repo_path = find_g2p_repo(str(repo_path) if repo_path else None)
        self.inject_ezafe = inject_ezafe
        self.use_gpu = use_gpu
        self._engine: object | None = None
        self._flat_fn: object | None = None
        self.backend = "unavailable"
        self._load_error: str | None = None
        self._loaded = False

    # -- availability ------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when the engine loaded (or can still be attempted)."""
        if self._loaded:
            return self.backend != "unavailable"
        return self.repo_path is not None

    @property
    def status(self) -> str:
        """Human-readable state, safe to call before the model is loaded.

        Loading pulls in a T5, so diagnostics must be able to report on the
        G2P without triggering it - and must not describe a perfectly good
        installation as broken merely because nothing has needed it yet.
        """
        if self.backend != "unavailable":
            return f"ready ({self.backend})"
        if self.repo_path is None:
            return "not found - set text.g2p.repo_path or $VOICECLONE_G2P_REPO"
        if not self._loaded:
            return f"found at {self.repo_path} (loads on first use)"
        return f"failed to load: {self._load_error or 'unknown error'}"

    # -- loading -----------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self.backend != "unavailable"
        self._loaded = True

        if self.repo_path is None:
            self._load_error = "G2P project directory not found"
            log.info("Persian G2P unavailable: %s", self._load_error)
            return False

        repo = str(self.repo_path)
        scripts = str(self.repo_path / "scripts")
        added = [p for p in (repo, scripts) if p not in sys.path]
        sys.path[:0] = added

        # Preferred: the low-level backend, which exposes the raw engine and
        # therefore preserves word boundaries.
        try:
            import homo_ge2pe_backend as backend  # type: ignore

            # `use_rules=True` makes GE2PE apply its ezafe rules itself, which
            # is what the word-aligned path relies on.
            phonemize = backend.build_backend(
                gpu=self.use_gpu,
                use_rules=True,
                inject_ezafe=self.inject_ezafe,
            )
            self._engine = getattr(phonemize, "engine", None)
            self._flat_fn = phonemize
            self.backend = "homo-ge2pe" if self._engine is not None else "homo-ge2pe (flat)"
            log.info("Persian G2P ready: %s (%s)", self.backend, self.repo_path)
            return True
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.debug("homo_ge2pe_backend import failed", exc_info=True)

        # Fallback: the project's public API. Flat output, no word boundaries.
        try:
            from persian_g2p import PersianG2PSystem  # type: ignore

            system = PersianG2PSystem(gpu=self.use_gpu, inject_ezafe=self.inject_ezafe)
            self._flat_fn = system.phonemize
            self.backend = "persian-g2p (flat)"
            log.info("Persian G2P ready: %s", self.backend)
            return True
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.warning("Persian G2P could not be loaded: %s", self._load_error)

        self.backend = "unavailable"
        return False

    # -- inference ---------------------------------------------------------
    def phonemize(self, text: str, *, normalize: bool = True) -> G2PResult:
        """Convert Persian text to IPA.

        Word boundaries are preserved when the raw engine is reachable, which
        keeps phrasing natural in the synthesised speech.
        """
        source = text or ""
        if not source.strip():
            return G2PResult(source, "", [], self.backend)

        prepared = normalize_for_g2p(source) if normalize else source

        if not self._ensure_loaded():
            return G2PResult(source, "", [], "unavailable")

        if self._engine is not None:
            try:
                return self._phonemize_word_aligned(source, prepared)
            except Exception:
                log.warning("Word-aligned G2P failed; falling back to flat output.", exc_info=True)

        return self._phonemize_flat(source, prepared)

    def _phonemize_word_aligned(self, source: str, prepared: str) -> G2PResult:
        """Run the raw engine so each orthographic word keeps its phones."""
        cwd = os.getcwd()
        repo_root = self.repo_path / "third_party" / "Homo-GE2PE-Persian" if self.repo_path else None
        try:
            if repo_root and repo_root.is_dir():
                os.chdir(repo_root)  # GE2PE resolves its resources relatively
            raw_list = self._engine.generate([prepared], batch_size=1, use_rules=True)  # type: ignore[union-attr]
        finally:
            os.chdir(cwd)

        raw = (raw_list[0] if raw_list else "") or ""
        compact_words = raw.split()
        source_words = prepared.split()

        # GE2PE's own rule engine already writes the ezafe vowel into its
        # output (کتاب من -> "ketabe m/n"), so no second injection pass here:
        # adding one turns گلِ into "golee".
        words: list[tuple[str, str]] = []
        for index, compact in enumerate(compact_words):
            ipa = compact_word_to_ipa(compact)
            if not ipa:
                continue
            orth = source_words[index] if index < len(source_words) else ""
            words.append((orth, ipa))

        phonemes = " ".join(ipa for _, ipa in words)
        return G2PResult(source, phonemes, words, self.backend)

    def _phonemize_flat(self, source: str, prepared: str) -> G2PResult:
        """Flat fallback: correct phones, word boundaries approximated."""
        raw = str(self._flat_fn(prepared))  # type: ignore[misc]
        # The flat API already returns project phones; only `a` and the ezafe
        # token need translating into standard IPA.
        tokens = []
        for token in raw.split():
            if token == EZAFE_TOKEN:
                tokens.append("e")
            elif token == "a":
                tokens.append("ɒ")
            else:
                tokens.append(token)
        phonemes = " ".join(tokens)
        return G2PResult(source, phonemes, [(prepared, phonemes)], self.backend)

    def batch(self, texts: list[str]) -> list[G2PResult]:
        return [self.phonemize(text) for text in texts]


# --------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_g2p(repo_path: str | None = None, *, inject_ezafe: bool = True) -> PersianG2P:
    """Cached G2P instance - loading the T5 model twice is pure waste."""
    return PersianG2P(repo_path, inject_ezafe=inject_ezafe)


def to_ipa_string(phonemes: str, *, delimiter: str = " ") -> str:
    """Collapse space-separated phones into contiguous IPA words.

    ``"k e t ɒ b e   m æ n"`` -> ``"ketɒbe mæn"``. Engines that take IPA
    expect words, not individual phones.
    """
    if not phonemes:
        return ""
    return delimiter.join(part.replace(" ", "") for part in phonemes.split("  ") if part)


def words_to_ipa(words: list[tuple[str, str]]) -> str:
    """Render word-aligned phones as spaced IPA words."""
    return " ".join(phones.replace(" ", "") for _, phones in words if phones)


__all__ = [
    "COMPACT_TO_IPA",
    "EZAFE_TOKEN",
    "G2PResult",
    "PersianG2P",
    "compact_word_to_ipa",
    "find_g2p_repo",
    "get_g2p",
    "to_ipa_string",
    "words_to_ipa",
]
