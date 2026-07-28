"""Script and language detection for mixed Persian/Latin input."""

from __future__ import annotations

import unicodedata
from collections import Counter

#: Perso-Arabic block ranges (Arabic, Arabic Supplement, Extended-A, Presentation Forms).
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

#: Letters that exist in Persian but not standard Arabic - a strong signal.
PERSIAN_ONLY = frozenset("پچژگک‌ی")


def is_arabic_script(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _ARABIC_RANGES)


def script_ratios(text: str) -> dict[str, float]:
    """Fraction of *letters* belonging to each script."""
    counts: Counter[str] = Counter()
    total = 0
    for char in text:
        if not char.isalpha():
            continue
        total += 1
        if is_arabic_script(char):
            counts["arabic"] += 1
        elif char.isascii():
            counts["latin"] += 1
        else:
            counts[unicodedata.name(char, "other").split()[0].lower()] += 1

    if not total:
        return {}
    return {key: value / total for key, value in counts.items()}


def detect_language(text: str, default: str = "fa") -> str:
    """Return ``'fa'``, ``'en'`` or ``default`` for empty/ambiguous input.

    Persian is assumed whenever Perso-Arabic script is present at all: a
    Persian sentence containing an English brand name is still Persian.
    """
    if not text or not text.strip():
        return default

    ratios = script_ratios(text)
    if not ratios:
        return default

    arabic = ratios.get("arabic", 0.0)
    latin = ratios.get("latin", 0.0)

    if arabic >= 0.15:
        return "fa"
    if latin >= 0.85:
        return "en"
    return default


def has_persian(text: str) -> bool:
    return any(is_arabic_script(char) for char in text)


def has_latin(text: str) -> bool:
    return any(char.isascii() and char.isalpha() for char in text)


def is_mixed_script(text: str) -> bool:
    return has_persian(text) and has_latin(text)


__all__ = [
    "PERSIAN_ONLY",
    "detect_language",
    "has_latin",
    "has_persian",
    "is_arabic_script",
    "is_mixed_script",
    "script_ratios",
]
