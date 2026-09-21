"""Clause-boundary cues that stop a grapheme TTS inventing ezafe.

Persian does not write the ezafe vowel. A greeting followed by a personal
pronoun is therefore ambiguous in script:

    سلام من اذیت می‌کنم

is either a vocative plus a new clause ("Hello. I annoy…") or a noun phrase
("my greeting"). OmniVoice and F5 pick the noun-phrase parse and speak
``سلامِ من``. A comma is a clause boundary; ezafe cannot cross it.

This is a high-precision rewrite, not a parser. Only utterance-initial
vocatives / discourse markers followed by a personal pronoun are touched.
Real ezafe (``کتاب من``, ``گل باغ``) is left alone.
"""

from __future__ import annotations

import re

#: Longest first so ``سلام علیکم`` wins over ``سلام``.
_VOCATIVES: tuple[str, ...] = (
    "سلام علیکم",
    "صبح بخیر",
    "ظهر بخیر",
    "عصر بخیر",
    "شب بخیر",
    "خداحافظ",
    "درود",
    "سلام",
    "مرسی",
    "ممنون",
    "سپاس",
    "باشه",
    "چشم",
    "بله",
    "آره",
    "آها",
    "خب",
    "نه",
)

_PRONOUNS: tuple[str, ...] = (
    "ایشان",
    "شما",
    "من",
    "تو",
    "او",
    "ما",
)

#: A vocative already closed by punctuation is left alone.
_TERMINATORS = "،.؟!؛:"

_VOCATIVE_CLAUSE_RE = re.compile(
    rf"(?:^|(?<=[\s{_TERMINATORS}\n]))"
    rf"({'|'.join(re.escape(word) for word in _VOCATIVES)})"
    rf"(?![{re.escape(_TERMINATORS)}])"
    rf"\s+"
    rf"({'|'.join(re.escape(word) for word in _PRONOUNS)})"
    rf"(?![\w؀-ۿ])"
)


def break_vocative_clauses(text: str) -> str:
    """Insert a Persian comma after a greeting that opens a new clause.

    >>> break_vocative_clauses("سلام من اذیت می‌کنم")
    'سلام، من اذیت می‌کنم'
    >>> break_vocative_clauses("کتاب من روی میز است")
    'کتاب من روی میز است'
    """
    if not text:
        return text
    return _VOCATIVE_CLAUSE_RE.sub(r"\1، \2", text)


__all__ = ["break_vocative_clauses"]
