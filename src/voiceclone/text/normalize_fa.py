"""Persian text normalisation.

Persian text in the wild is messy: Arabic letterforms mixed with Persian ones,
four different digit systems, missing or doubled ZWNJ, stray bidi control
marks. Any of these can push a word out of the TTS vocabulary and produce a
mispronunciation, so normalisation runs before anything else.

Deliberately conservative: it fixes what is unambiguously wrong and leaves
genuine orthographic choices alone.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .numbers_fa import expand_numbers, to_ascii_digits

# -- Unicode landmarks -----------------------------------------------------
ZWNJ = "‌"  # zero-width non-joiner (Persian "half space")
ZWJ = "‍"
TATWEEL = "ـ"  # kashida, purely decorative elongation

#: Bidi controls, BOM and other invisibles that break tokenisation.
_INVISIBLES = re.compile(
    "["
    "​"      # zero-width space
    "‎‏"  # LRM / RLM
    "‪-‮"  # bidi embedding/override
    "⁦-⁩"  # bidi isolates
    "﻿"      # BOM
    "­"      # soft hyphen
    "]"
)

#: Arabic letterforms -> their Persian equivalents.
CHARACTER_UNIFICATION = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH
    "ے": "ی",  # YEH BARREE        -> FARSI YEH
    "ك": "ک",  # ARABIC KAF        -> KEHEH
    "ڪ": "ک",  # SWASH KAF         -> KEHEH
    "ة": "ه",  # TEH MARBUTA       -> HEH
    "ۀ": "ه",  # HEH WITH YEH ABOVE-> HEH
    "ہ": "ه",  # HEH GOAL          -> HEH
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    "ٱ": "ا",  # ALEF WASLA        -> ALEF
    "٠": "۰", "١": "۱", "٢": "۲",
    "٣": "۳", "٤": "۴", "٥": "۵",
    "٦": "۶", "٧": "۷", "٨": "۸",
    "٩": "۹",
}

#: Arabic harakat (short-vowel marks). Optional to strip - the G2P can use them.
DIACRITICS = re.compile("[ً-ْٰٕٖٓ-ٟۖ-ۭ]")

#: ASCII punctuation -> the Persian forms the models were trained on.
PUNCTUATION_MAP = {
    ",": "،",
    ";": "؛",
    "?": "؟",
}

#: Normalise exotic quote/dash forms to a single style.
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-",
    "…": "...",
}

#: Verb prefixes that must attach with a ZWNJ, not a space.
_VERB_PREFIX_RE = re.compile(r"(?<![؀-ۿ])(ن?می)\s+(?=[؀-ۿ]{2,})")

#: Nominal suffixes. Joining these is *usually* right but not always
#: ("تر" is also a standalone adjective), so it is opt-in.
_SUFFIX_RE = re.compile(
    r"(?<=[؀-ۿ]{2})\s+(ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان|تر|ترین)"
    r"(?![؀-ۿ])"
)

#: Safe, unambiguous abbreviation expansions.
ABBREVIATIONS = {
    "ه.ق": "هجری قمری",
    "ه.ش": "هجری شمسی",
    "ه.خ": "هجری خورشیدی",
    "ق.م": "قبل از میلاد",
    "ب.م": "بعد از میلاد",
    "و غیره": "و غیره",
    "الخ": "و غیره",
    "ج.ا.ا": "جمهوری اسلامی ایران",
    "Dr.": "دکتر",
    "Mr.": "آقای",
    "Mrs.": "خانم",
}

_BRACKETED_TAG_RE = re.compile(r"[\[\(]\s*(?:[a-zA-Z_\- ]{2,30})\s*[\]\)]")
_MULTI_SPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_ZWNJ_RE = re.compile(f"{ZWNJ}{{2,}}")
_ZWNJ_AROUND_SPACE_RE = re.compile(f"\\s*{ZWNJ}\\s*(?=\\s)|(?<=\\s)\\s*{ZWNJ}\\s*")
_REPEATED_PUNCT_RE = re.compile(r"([،؛!؟.])\1{1,}")
#: A thousands separator with no digits around it was meant as a comma.
_STRAY_SEPARATOR_RE = re.compile(r"(?<!\d)[٬,](?!\d)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؛؟!.:])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"([،؛؟!:])(?=[^\s\d])")


@dataclass
class NormalizationOptions:
    """Switches for :func:`normalize`."""

    unify_characters: bool = True
    remove_diacritics: bool = False
    remove_tatweel: bool = True
    fix_zwnj: bool = True
    join_verb_prefixes: bool = True
    join_suffixes: bool = False
    normalize_punctuation: bool = True
    expand_abbreviations: bool = True
    expand_numbers: bool = True
    strip_bracketed_tags: bool = True
    ascii_digits: bool = False


def remove_invisibles(text: str) -> str:
    """Strip bidi marks, BOM and other zero-width characters (keeping ZWNJ)."""
    return _INVISIBLES.sub("", text)


def unify_characters(text: str) -> str:
    """Map Arabic letterforms and digits onto their Persian equivalents."""
    return text.translate(str.maketrans(CHARACTER_UNIFICATION))


def remove_diacritics(text: str) -> str:
    """Drop harakat. Only do this when the downstream G2P ignores them."""
    return DIACRITICS.sub("", text)


def fix_zwnj(text: str, *, join_verb_prefixes: bool = True, join_suffixes: bool = False) -> str:
    """Clean up half-space usage.

    - collapse repeated ZWNJ
    - drop ZWNJ that sits next to a real space (it does nothing there)
    - optionally reattach ``می``/``نمی`` verb prefixes typed with a space
    """
    text = _MULTI_ZWNJ_RE.sub(ZWNJ, text)
    text = _ZWNJ_AROUND_SPACE_RE.sub(" ", text)
    text = text.replace(ZWJ, "")

    if join_verb_prefixes:
        text = _VERB_PREFIX_RE.sub(lambda m: m.group(1) + ZWNJ, text)
    if join_suffixes:
        text = _SUFFIX_RE.sub(lambda m: ZWNJ + m.group(1), text)

    # A ZWNJ at a word boundary is meaningless.
    return text.strip(ZWNJ)


def normalize_punctuation(text: str) -> str:
    """Use Persian punctuation and tidy the spacing around it."""
    text = text.translate(str.maketrans(_QUOTE_MAP))
    text = _STRAY_SEPARATOR_RE.sub("،", text)
    text = text.translate(str.maketrans(PUNCTUATION_MAP))
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    return text


def expand_abbreviations(text: str) -> str:
    for short, long in ABBREVIATIONS.items():
        text = text.replace(short, long)
    return text


def strip_bracketed_tags(text: str) -> str:
    """Remove director's notes such as ``[laughter]`` or ``(pause)``.

    Persian content in brackets is left alone - only Latin-only tags match.
    """
    return _BRACKETED_TAG_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize(text: str, options: NormalizationOptions | None = None) -> str:
    """Run the full normalisation chain.

    >>> normalize("سلام ٬ ايران ١٤٠٣")
    'سلام، ایران هزار و چهارصد و سه'
    """
    if not text:
        return ""

    options = options or NormalizationOptions()

    # NFC first: composes stray combining marks into canonical letters.
    text = unicodedata.normalize("NFC", text)
    text = remove_invisibles(text)

    if options.remove_tatweel:
        text = text.replace(TATWEEL, "")
    if options.strip_bracketed_tags:
        text = strip_bracketed_tags(text)
    if options.unify_characters:
        text = unify_characters(text)
    if options.remove_diacritics:
        text = remove_diacritics(text)
    if options.expand_abbreviations:
        text = expand_abbreviations(text)
    if options.expand_numbers:
        text = expand_numbers(text)
    elif options.ascii_digits:
        text = to_ascii_digits(text)
    if options.normalize_punctuation:
        text = normalize_punctuation(text)
    if options.fix_zwnj:
        text = fix_zwnj(
            text,
            join_verb_prefixes=options.join_verb_prefixes,
            join_suffixes=options.join_suffixes,
        )

    return collapse_whitespace(text)


def normalize_for_g2p(text: str) -> str:
    """Normalisation tuned for the Homo-GE2PE front end.

    Diacritics are preserved: the model uses an explicit kasra as evidence for
    ezafe, and they help disambiguate homographs.
    """
    return normalize(
        text,
        NormalizationOptions(
            remove_diacritics=False,
            expand_numbers=True,
            normalize_punctuation=True,
            join_suffixes=False,
        ),
    )


__all__ = [
    "ABBREVIATIONS",
    "CHARACTER_UNIFICATION",
    "NormalizationOptions",
    "ZWNJ",
    "collapse_whitespace",
    "expand_abbreviations",
    "fix_zwnj",
    "normalize",
    "normalize_for_g2p",
    "normalize_punctuation",
    "remove_diacritics",
    "remove_invisibles",
    "strip_bracketed_tags",
    "unify_characters",
]
