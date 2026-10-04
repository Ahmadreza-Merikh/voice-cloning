"""Write the first short vowel of a homograph, from Homo-GE2PE's reading of the sentence.

Step 3 (FINDINGS section 17): with the first vowel of each of HomoRich's
homographs that it tells apart written in, the listener heard 8 of 23
homographs wrong against 14, and 25 words of 242 against 27, with no rise in
CER and no drop in speaker similarity. This places that mark:

* which words: :data:`homographs_data.FIRST_VOWEL_HOMOGRAPHS` (HomoRich, CC0),
  less به - every mark on it was heard wrong
* which vowel: Homo-GE2PE's reading of the whole sentence, its words aligned to
  the sentence's by consonant skeleton - the alignment step 1 scored with
  (``scripts/score_frontend_components.py``), carried over unchanged
* how: :func:`marks_fa.with_first_vowel`, only where the script leaves that
  vowel out, on top of whatever the ezafe step wrote

Homo-GE2PE runs in this process, loaded for one text and released after it -
0.66 GB at its peak, 5.5 s to load, 0.6 s a sentence (2026-10-04) - and only
with enough memory free. Sentences it has read are remembered.
"""

from __future__ import annotations

import gc
from collections import OrderedDict

from ..config import HomographConfig
from ..errors import TextProcessingError
from ..logging_utils import get_logger
from .ezafe_fa import plan
from .homographs_data import FIRST_VOWEL_HOMOGRAPHS
from .marks_fa import HARAKAT, PUNCT, ZWNJ, bare, first_short_vowel, with_first_vowel

log = get_logger("text.homographs")

CACHE_SIZE = 4096

# -- the alignment step 1 scored with (scripts/score_frontend_components.py) --
SOUNDS = set("bptdkgqfszSZxCjmnlr")
LETTER = {"ب": "b", "پ": "p", "ت": "t", "ط": "t", "ث": "s", "س": "s", "ص": "s", "ج": "j", "چ": "C",
          "خ": "x", "د": "d", "ذ": "z", "ز": "z", "ض": "z", "ظ": "z", "ر": "r", "ژ": "Z", "ش": "S",
          "غ": "q", "ق": "q", "ف": "f", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n"}
GE2PE = {"/": "a", "a": "A", "$": "S", ";": "Z", "c": "C", "@": "?", "'": "?"}
MERGE, SKIP, LEADING_AND = 0.2, 0.6, 0.1


def clean(text: str) -> str:
    for a, b in (("ي", "ی"), ("ى", "ی"), ("ك", "ک"), ("ۀ", "هٔ"), ("ة", "ه"), ("ـ", ""),
                 (" ", " "), ("‏", ""), ("‎", "")):
        text = text.replace(a, b)
    return text


def words_of(sentence: str) -> list[str]:
    return [w for w in (t.strip(PUNCT) for t in clean(sentence).split()) if w]


def letters(word: str) -> str:
    return "".join(c for c in word if c not in HARAKAT and c != ZWNJ)


def from_ge2pe(token: str) -> str:
    return "".join(GE2PE.get(c, c) for c in token)


def ezafe_split(word: str, token: str) -> tuple[str, bool]:
    """A reading without the ezafe Homo-GE2PE writes into it."""
    t = token.strip(".,،؟!")
    marked = t.endswith("1")
    t = t.rstrip("1")
    if t.endswith("-ye"):
        return t[:-3], True
    if t.endswith("-e"):
        return t[:-2], True
    last = letters(word)[-1:] if letters(word) else ""
    if last == "ه":
        if len(t) > 3 and t.endswith("eye"):
            return t[:-2], True
        if len(t) > 2 and t.endswith("he"):
            return t[:-1], True
        return t, marked
    if last in ("ا", "آ", "و", "ی") and len(t) > 2 and t.endswith("ye"):
        return t[:-2], True
    if len(t) > 1 and t.endswith("e"):
        return t[:-1], True
    return t, marked


def distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def align(ws: list[str], ts: list[str]) -> list[tuple[list[int], list[int]]]:
    """Words to reading tokens by consonant skeleton, allowing merges and skips."""
    gw = ["".join(LETTER.get(c, "") for c in letters(w)) for w in ws]
    gt = ["".join(c for c in t if c in SOUNDS) for t in ts]
    n, m = len(ws), len(ts)
    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            here = cost[i][j]
            if here == float("inf"):
                continue
            moves = [(a, b, distance("".join(gw[i:i + a]), "".join(gt[j:j + b])) + extra
                      + (LEADING_AND if a > 1 and letters(ws[i]) == "و" else 0.0))
                     for a, b, extra in ((1, 1, 0.0), (2, 1, MERGE), (3, 1, 2 * MERGE), (1, 2, MERGE))
                     if i + a <= n and j + b <= m]
            if i < n:
                moves.append((1, 0, SKIP))
            if j < m:
                moves.append((0, 1, SKIP))
            for a, b, d in moves:
                if here + d < cost[i + a][j + b]:
                    cost[i + a][j + b] = here + d
                    back[i + a][j + b] = (a, b)
    out = []
    i, j = n, m
    while i or j:
        a, b = back[i][j]
        out.append((list(range(i - a, i)), list(range(j - b, j))))
        i, j = i - a, j - b
    return out[::-1]


def groups(ws: list[str], ts: list[str]) -> list[tuple[list[int], str]]:
    out = []
    for wi, ti in align(ws, ts):
        if not wi or not ti:
            continue
        token = "".join(ts[k] for k in ti)
        if len(wi) == 2 and letters(ws[wi[1]]) == "و":
            token = token[:-2] if token.endswith("-o") else token[:-1] if token.endswith("o") else token
            out.append(([wi[0]], token))
        elif len(wi) == 3 and letters(ws[wi[1]]) == "و" and "-o-" in token:
            left, right = token.split("-o-", 1)
            out += [([wi[0]], left), ([wi[2]], right)]
        else:
            out.append((wi, token))
    return out


def readings(text: str, compact_words: list[str]) -> list[str | None]:
    """Homo-GE2PE's reading of each word, without its ezafe, where a word maps to a token of its own."""
    ws = words_of(text)
    out: list[str | None] = [None] * len(ws)
    for wi, token in groups(ws, [from_ge2pe(t) for t in compact_words]):
        if len(wi) == 1:
            out[wi[0]] = ezafe_split(ws[wi[0]], token)[0]
    return out


def write_first_vowels(plain: str, marked: str, compact_words: list[str],
                       homographs: frozenset[str] = FIRST_VOWEL_HOMOGRAPHS) -> tuple[str, list[str]]:
    """``marked`` with each homograph's first short vowel written, read from ``plain``; and those words.

    ``marked`` is ``plain`` with marks added and nothing else changed, so their
    words sit in the same places.
    """
    parts, sentences = plan(plain)
    out, marked_sentences = plan(marked)
    order = [i for g in sentences for i in g if parts[i].strip(PUNCT)]
    if order != [i for g in marked_sentences for i in g if out[i].strip(PUNCT)]:
        raise TextProcessingError("the marked text no longer lines up with the plain one")
    found = readings(plain, compact_words)
    if len(found) != len(order):
        return marked, []
    written = []
    for k, i in enumerate(order):
        core = parts[i].strip(PUNCT)
        if bare(core) not in homographs or not first_short_vowel(core, found[k]):
            continue
        raw = out[i]
        word = raw.strip(PUNCT)
        start = raw.index(word)
        new = with_first_vowel(word, found[k])
        if new != word:
            out[i] = raw[:start] + new + raw[start + len(word):]
            written.append(core)
    return "".join(out), written


class HomographWriter:
    """Runs Homo-GE2PE for a batch of texts, within the memory it is allowed."""

    def __init__(self, config: HomographConfig) -> None:
        self.config = config
        self._cache: OrderedDict[str, list[str]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        setting = self.config.enabled
        return not (setting is False or str(setting).lower() in {"false", "off", "no"})

    @property
    def required(self) -> bool:
        setting = self.config.enabled
        return setting is True or str(setting).lower() in {"true", "on", "yes", "required"}

    def _room(self) -> str | None:
        from .. import memory

        level = memory.pressure_level()
        if level is not None and level >= memory.WARNING:
            return f"memory pressure is at {memory.describe(level)}"
        available = memory.available_gb()
        needed = float(self.config.min_available_gb)
        if available is not None and available < needed:
            return f"{available:.1f} GB of memory available, Homo-GE2PE needs {needed:.1f}"
        return None

    def write(self, plains: list[str], marked: list[str], g2p) -> tuple[list[str], list[str], str | None]:
        """Each marked text with its homographs' first vowels, the words marked, and why not if it could not."""
        loaded = False

        def read(text: str) -> list[str]:
            nonlocal loaded
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
            if g2p is None:
                raise TextProcessingError("the Persian G2P is turned off (text.g2p.enabled)")
            if not loaded:
                reason = self._room()
                if reason:
                    raise TextProcessingError(f"Homo-GE2PE was not started: {reason}.")
                loaded = True
            result = g2p.phonemize(text)
            if not result.compact_words:
                raise TextProcessingError(f"Homo-GE2PE gave no reading ({result.backend})")
            self._cache[text] = list(result.compact_words)
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
            return self._cache[text]

        try:
            done = [write_first_vowels(p, m, read(p)) for p, m in zip(plains, marked, strict=True)]
        except Exception as exc:
            if self.required:
                raise
            log.warning("Homograph vowels not written: %s", exc)
            return list(marked), [], str(getattr(exc, "message", exc))
        finally:
            if loaded and hasattr(g2p, "release"):
                g2p.release()
                gc.collect()
        return [text for text, _ in done], [w for _, words in done for w in words], None


__all__ = ["HomographWriter", "align", "groups", "readings", "write_first_vowels"]
