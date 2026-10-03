"""Write what plain Persian script leaves out back into it, as marks a TTS model reads.

The pronunciation review (eval/runs/persian-voice-pronunciation-review-v1.json)
found OmniVoice's errors on plain script are mostly the ezafe and short vowels,
and that it reads the diacritics it is given - right or wrong. This module only
places marks; what goes where is decided elsewhere (a tagger for the ezafe, a
lexicon for vowels), so a wrong decision is never this module's guess.

Three kinds of mark, each optional:

* ezafe - the linking -e/-ye after a word: a kasra on a final consonant, ٔ on a
  final silent h, ی after a final ا/آ/و
* no ezafe - a sukun on a final consonant, where a word must not run into the
  next with an -e
* first vowel - fatha, kasra or damma on a word's first letter, when its reading
  starts with a short vowel the script does not write

Nothing is pure: a word that already carries a mark at the place one would go
is left as it is.
"""

from __future__ import annotations

from dataclasses import dataclass

FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SUKUN = "ْ"
HAMZA_ABOVE = "ٔ"
ZWNJ = "‌"
HARAKAT = "ًٌٍَُِّْٰٕٔ"
PUNCT = "،؛؟!?.,:;«»\"'()[]{}…–—-*/"
VOWEL_LETTERS = "اآوی"
SHORT = {"a": FATHA, "e": KASRA, "o": DAMMA}
#: Words whose final ه is the consonant h though no ا or و comes before it, so
#: their ezafe is a kasra (توجهِ, tavajjoh-e), not هٔ. Written from knowledge of
#: the words and checked against KaamelDict on 2026-10-03, which agrees for
#: every one (گره, مه, ده and مشابه it also lists without the h); its own data
#: is not used here.
PRONOUNCED_H = frozenset({
    "توجه", "تنبه", "تفقه", "تشبه", "تنزه", "تشابه", "وجه", "فقه", "شبه", "گره", "زره", "مه", "ده",
    "شبیه", "فقیه", "سفیه", "وجیه", "کریه", "توجیه", "تنبیه", "تشبیه", "تنزیه", "ترفیه",
    "متوجه", "متنبه", "منزه", "موجه", "مشتبه", "مشابه", "متشابه", "مواجه", "الله", "اله",
})


@dataclass(frozen=True)
class Word:
    """One word of a sentence and what is known about it."""

    text: str
    ezafe: bool = False
    pos: str = ""
    reading: str | None = None
    next_pos: str | None = None


def bare(word: str) -> str:
    return "".join(c for c in word if c not in HARAKAT and c != ZWNJ)


def last_letter(word: str) -> str:
    b = bare(word)
    return b[-1] if b else ""


def silent_h(word: str, reading: str | None = None) -> bool:
    """A final ه that is the vowel e, not the consonant h.

    The reading settles it when there is one (xAne vs mAh); without one a ه after
    ا or و is taken as pronounced (راه، کوه), as is every word of
    ``PRONOUNCED_H`` (توجه), and any other as silent (خانه).
    """
    if last_letter(word) != "ه":
        return False
    if reading:
        r = reading.rstrip("-")
        return not r.endswith("h")
    b = bare(word)
    return not ((len(b) >= 2 and b[-2] in "او") or b in PRONOUNCED_H)


def with_ezafe(word: str, reading: str | None = None) -> str:
    if not bare(word):
        return word
    # A mark already on the last letter is the writer's, whatever it is.
    if word[-1] in HARAKAT or word.endswith("ۀ"):
        return word
    last = last_letter(word)
    if last == "ه":
        return word + (HAMZA_ABOVE if silent_h(word, reading) else KASRA)
    if last in "اآو":
        return word + "ی"
    return word + KASRA


def with_sukun(word: str, reading: str | None = None) -> str:
    if not bare(word) or word[-1] in HARAKAT:
        return word
    last = last_letter(word)
    if last in VOWEL_LETTERS or (last == "ه" and silent_h(word, reading)):
        return word
    return word + SUKUN


def first_short_vowel(word: str, reading: str | None) -> str | None:
    """The short vowel after a word's first sound, when the script does not write it."""
    if not reading or len(word) < 2 or word[1] in HARAKAT:
        return None
    first, second = word[0], word[1]
    r = reading.replace("-", "")
    if first == "آ" or len(r) < 2:
        return None
    if r[0] == "?":
        r = r[1:]
        if first not in "اعءأإ" or not r:
            return None
        vowel, after = r[0], r[1:2]
    else:
        if r[0] in "Aaeoiu" or first in "اعءأإ":
            return None
        vowel, after = r[1], r[2:3]
    if vowel not in SHORT or second == ZWNJ or second in "اآ":
        return None
    if second in "وی" and after != ("v" if second == "و" else "y"):
        return None
    return vowel


def with_first_vowel(word: str, reading: str | None) -> str:
    """Put the reading's first short vowel on the first letter, if the script leaves it out."""
    vowel = first_short_vowel(word, reading)
    return word[0] + SHORT[vowel] + word[1:] if vowel else word


def mark_words(words: list[Word], *, ezafe: bool = True, sukun: bool = False, vowels: bool = False,
               sukun_heads: tuple[str, ...] = ("NOUN", "ADJ", "NUM"),
               sukun_next: tuple[str, ...] = ("NOUN", "ADJ", "NUM", "VERB", "ADV", "PRON", "DET"),
               vowel_pos: tuple[str, ...] = ("NOUN", "ADJ")) -> list[str]:
    out = []
    for w in words:
        text = w.text
        if vowels and w.pos in vowel_pos:
            text = with_first_vowel(text, w.reading)
        if ezafe and w.ezafe:
            text = with_ezafe(text, w.reading)
        elif sukun and not w.ezafe and w.pos in sukun_heads and w.next_pos in sukun_next:
            text = with_sukun(text, w.reading)
        out.append(text)
    return out


def rebuild(sentence: str, marked: list[str]) -> str:
    """Put marked words back into the sentence's own spacing and punctuation."""
    pieces, k = [], 0
    for raw in sentence.split(" "):
        core = raw.strip(PUNCT)
        if not core:
            pieces.append(raw)
            continue
        start = raw.index(core)
        pieces.append(raw[:start] + marked[k] + raw[start + len(core):])
        k += 1
    if k != len(marked):
        raise ValueError(f"{len(marked)} marked words for {k} words in {sentence!r}")
    return " ".join(pieces)


__all__ = ["Word", "first_short_vowel", "mark_words", "rebuild", "silent_h", "with_ezafe", "with_first_vowel",
           "with_sukun"]
