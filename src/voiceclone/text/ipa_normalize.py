"""Canonical Persian IPA: Homo-GE2PE's output in, the OmniVoice IPA
fine-tune's convention out.

Why this layer exists
---------------------
``christopherthompson81/omnivoice-ipa-onnx`` is ``k2-fsa/OmniVoice`` continued
on IPA produced by ``vernacula-phonemizer`` (its Persian golden set is
``csharp/goldens/fa.tsv`` in that project's repository). This repo's G2P is
Homo-GE2PE, and the two disagree in ways that change pronunciation. Handing
GE2PE's output to that model unchanged would be a distribution shift the
model cannot report: it would still speak, just not the pronunciation that
was resolved.

Both conventions were read from their sources, not assumed.

Canonical (``fa.tsv``, 200 pairs, + ``src/languages/persian/persian.ts``)::

    vowels      a  aː  e  o  iː  uː   - `æ` count 0, `ɒ` count 0
    consonants  ɾ (U+027E), ɡ (U+0261) - ASCII `r` and `g` count 0
    affricates  t͡ʃ  d͡ʒ with tie bar U+0361 - bare tʃ/dʒ count 0
    alef onset  ʔ - 855 word-initial occurrences (اضافه -> ʔazaafa)
    stress      ˈ immediately before the LAST vowel letter of the word, on the
                string AFTER mapping (persian.ts: "Persian stress is (mostly)
                word-FINAL: mark the last vowel nucleus")
    length      every Persian i and u in the goldens is long; the single
                bare-i sits inside a foreign word
    punctuation ASCII, spaced off the word: `.` `,` `!` reach the IPA;
                `:` and «» and () appear in the orthography and not in it

Homo-GE2PE (this repo's ``g2p.py``, verified against live model output)::

    vowels      æ  ɒ  e  o  i  u      - no length marking at all
    consonants  ASCII r, ASCII g; affricates as the tokens tʃ / dʒ
    alef onset  `@` - a bookkeeping marker that ``compact_word_to_ipa`` drops;
                it is GE2PE's ONLY signal of the word-initial glottal stop
                (از -> @/z, which must become ʔaz, not az). That is why the
                normaliser prefers the compact stream over the rendered IPA.
    stress      `1` - bookkeeping. GE2PE has already made its choice; what
                the model learned is the canonical word-final rule.
    ezafe       already written into GE2PE's own output (کتاب -> ketabe)

Contract
--------
``normalize_ge2pe_ipa_for_omnivoice`` never drops a phoneme silently:

* every symbol swap is counted in ``changes`` as (source, target, count);
* every symbol it cannot place in the canonical inventory is KEPT in the
  output, listed in ``unknown``, and logged at warning level;
* deliberate removals (GE2PE's stress digit, punctuation the goldens drop)
  appear in ``changes`` with an empty target;
* the mapping lives in tables (``SYMBOL_MAP``, plus ``COMPACT_TO_IPA`` and
  ``COMPACT_DIGRAPHS`` from :mod:`.g2p`) so fixtures can pin every row.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..logging_utils import get_logger
from .g2p import COMPACT_DIGRAPHS, COMPACT_TO_IPA

log = get_logger("text.ipa_normalize")

#: GE2PE symbol -> canonical symbol. Every row is justified in the module
#: docstring; ``tests/test_omnivoice_ipa.py`` pins all of them.
SYMBOL_MAP: dict[str, str] = {
    "æ": "a",       # fa.tsv writes short a as `a`; `æ` appears 0 times
    "ɒ": "aː",      # ... and long a as `aː`; `ɒ` appears 0 times
    "i": "iː",      # every Persian i in the goldens is long
    "u": "uː",      # ... and so is every u
    "r": "ɾ",       # goldens use the tap U+027E; ASCII r appears 0 times
    "g": "ɡ",       # goldens use script g U+0261; ASCII g appears 0 times
    "tʃ": "t͡ʃ",    # affricates always carry the tie bar (U+0361)
    "dʒ": "d͡ʒ",
    "@": "ʔ",       # alef onset: GE2PE's bookkeeping marker IS the glottal stop
    "ezafe": "e",   # the flat G2P API's marker; rendered as the ezafe vowel
}

#: Bookkeeping GE2PE may leave in its compact form; not a phoneme.
_STRESS_DIGIT = "1"

#: Characters the canonical goldens keep in the IPA (the `fa.tsv` IPA column).
#: A token whose characters all sit here passes through untouched; anything
#: else is unknown - kept in the output, reported, never silently dropped.
#: `?` and `;` are the Persian `؟` and `؛` mapped by analogy with the verified
#: `،` -> `,` (the goldens contain no questions or semicolons to verify on).
CANONICAL_INVENTORY = frozenset(
    "aeiou"              # vowel letters; length is its own symbol
    "ːˈʔ"                # length, stress, glottal stop
    "bptdkɡfqvszʃʒɣxhmlnɾj"  # the consonants the goldens actually use
    "͡"                  # tie bar, half of t͡ʃ / d͡ʒ
    ".,!?;"              # punctuation kept, spaced off the word
)

#: Punctuation that may appear on the edge of a word (both scripts).
ALL_PUNCT = frozenset(".,!?;:؟،؛«»()\"…٪-–—")

#: Edge punctuation worth KEEPING in the IPA. `.` `,` `!` are verified in the
#: goldens; `؟` -> `?` is by analogy with `!` (no question appears in them).
#: Everything else on a word edge is dropped - as the goldens drop it - but
#: counted with an empty target, so nothing disappears quietly.
KEEP_PUNCT = ".,!؟،"
PUNCT_ASCII = {"،": ",", "؟": "?"}

#: The source rule the stress mark comes from, applied to the MAPPED word:
#: insert ˈ before the last vowel letter (persian.ts phonemizeWordCore).
_VOWEL = re.compile(r"[aeiouɒ]")


@dataclass
class SymbolChange:
    """One symbol-level edit, counted across the text.

    ``target`` is empty for deliberate removals (the stress digit, dropped
    punctuation): they are edits too, and hiding them would be the silent
    loss this module refuses to do.
    """

    source: str
    target: str
    count: int


@dataclass
class NormalizedIPA:
    """Canonical IPA plus the full report of what was done to get there."""

    #: The canonical text: words joined by spaces, punctuation its own token.
    text: str
    #: The entries ``text`` was joined from (a word may carry its punctuation).
    words: list[str]
    #: Every symbol edit, most frequent first.
    changes: list[SymbolChange]
    #: Symbols kept in the output but not recognised. Empty in practice: the
    #: GE2PE inventory is fully covered by SYMBOL_MAP. This list is the alarm.
    unknown: list[str]
    #: Human-readable events: what could not be applied, and why.
    notes: list[str]
    #: Counters: stress_inserted, punct_restored, stress_digit_removed, ...
    stats: dict[str, int]
    #: Which input produced this: "compact" | "ipa" | "flat" | "empty".
    source: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; the A/B manifest stores this per render."""
        return {
            "text": self.text,
            "words": list(self.words),
            "changes": [
                {"source": c.source, "target": c.target, "count": c.count} for c in self.changes
            ],
            "unknown": list(self.unknown),
            "notes": list(self.notes),
            "stats": dict(self.stats),
            "source": self.source,
        }


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------
def _tokens_from_compact(compact: str) -> tuple[list[str], list[str], int]:
    """GE2PE compact word -> (phone tokens, punctuation chars, stress digits).

    Mirrors :func:`g2p.compact_word_to_ipa` except where the canonical
    convention differs: `@` survives as a phone (mapped to ʔ later), `1` is
    counted rather than dropped by ``_COMPACT_NOISE``, and punctuation is
    carried out for edge recovery - ``compact_word_to_ipa`` drops it, which
    is why the raw path loses the full stop the goldens keep.
    """
    cleaned = (compact or "").strip().lower()
    tokens: list[str] = []
    punct: list[str] = []
    stress_digits = 0
    index = 0
    while index < len(cleaned):
        for source, target in COMPACT_DIGRAPHS:
            if cleaned.startswith(source, index):
                tokens.append(target)
                index += len(source)
                break
        else:
            char = cleaned[index]
            index += 1
            if char == "@":
                tokens.append("@")
            elif char == _STRESS_DIGIT:
                stress_digits += 1
            elif char in COMPACT_TO_IPA:
                tokens.append(COMPACT_TO_IPA[char])
            elif char in ALL_PUNCT:
                punct.append(char)
            elif char.isspace():
                continue
            else:
                tokens.append(char)  # unknown: the mapping stage reports it
    return tokens, punct, stress_digits


def _orth_edges(orth: str) -> tuple[str, str]:
    """(leading, trailing) punctuation run of an orthographic word."""
    text = orth or ""
    start = 0
    end = len(text)
    while start < end and text[start] in ALL_PUNCT:
        start += 1
    while end > start and text[end - 1] in ALL_PUNCT:
        end -= 1
    return text[:start], text[end:]


def _punct_tokens(chars: str, changes: Counter[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Render edge punctuation: keep what the goldens keep, log the rest.

    Returns (tokens kept as their own word, chars dropped).
    """
    kept: list[str] = []
    dropped: list[str] = []
    for char in chars:
        if char in KEEP_PUNCT:
            target = PUNCT_ASCII.get(char, char)
            if target != char:
                changes[(char, target)] += 1
            kept.append(target)
        else:
            changes[(char, "")] += 1  # removed on purpose, and reported here
            dropped.append(char)
    return kept, dropped


def _insert_stress(word: str) -> tuple[str, bool]:
    """Place ``ˈ`` before the last vowel letter - the canonical rule.

    Returns ``(word, placed)``; ``placed`` is False when the word has no
    vowel or already carries a mark (so re-running is idempotent).
    """
    if not word or "ˈ" in word:
        return word, False
    last = None
    for match in _VOWEL.finditer(word):
        last = match
    if last is None:
        return word, False
    return word[: last.start()] + "ˈ" + word[last.start() :], True


def _map_tokens(
    tokens: Sequence[str],
    changes: Counter[tuple[str, str]],
    unknown: Counter[str],
) -> str:
    """Apply SYMBOL_MAP token-wise and join into one continuous word.

    Token-wise (not char-wise) so `tʃ` maps as a unit and an already-canonical
    `iː` passes through instead of becoming `iːː`.
    """
    mapped: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        target = SYMBOL_MAP.get(token)
        if target is not None:
            if target != token:
                changes[(token, target)] += 1
            mapped.append(target)
        elif all(char in CANONICAL_INVENTORY for char in token):
            mapped.append(token)  # already canonical
        else:
            unknown[token] += 1
            mapped.append(token)  # kept, reported - never dropped
    return "".join(mapped)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def normalize_ge2pe_ipa_for_omnivoice(
    words: Sequence[tuple[str, str]],
    compact_words: Sequence[str] | None = None,
) -> NormalizedIPA:
    """Convert Homo-GE2PE's word-aligned output into canonical Persian IPA.

    Args:
        words: ``(orthography, space-separated phones)`` pairs, as
            :class:`g2p.G2PResult` carries them.
        compact_words: the compact romanisation each word came from, parallel
            to ``words``. Preferred: only the compact stream carries the
            alef-onset marker that becomes the word-initial ``ʔ``. Empty for
            flat G2P output, which then falls back to the rendered IPA with
            the loss reported in ``notes``.
    """
    changes: Counter[tuple[str, str]] = Counter()
    unknown: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    out_words: list[str] = []
    notes: list[str] = []
    dropped_punct: list[str] = []
    kept_punct: list[str] = []

    compacts = list(compact_words) if compact_words else []
    if compacts and len(compacts) == len(words) and words:
        mode = "compact"
    elif len(words) == 1 and len((words[0][0] or "").split()) > 1:
        # One entry whose orthography is a whole sentence: the flat fallback,
        # where word boundaries (and therefore per-word stress) are gone.
        mode = "flat"
    elif words:
        mode = "ipa"
    else:
        mode = "empty"

    for index, (orth, ipa) in enumerate(words):
        punct_seen: list[str] = []
        stress_digits = 0
        if mode == "compact":
            tokens, punct_seen, stress_digits = _tokens_from_compact(compacts[index])
        else:
            tokens = (ipa or "").split()

        if stress_digits:
            stats["stress_digit_removed"] += stress_digits
            changes[(_STRESS_DIGIT, "")] += stress_digits

        core = _map_tokens(tokens, changes, unknown)

        # Punctuation: recover it from the orthography (GE2PE's compact drops
        # it), deduplicated against whatever the compact itself carried.
        lead, trail = _orth_edges(orth)
        recovered = set(lead) | set(trail)
        for char in punct_seen:
            if char not in recovered:
                changes[(char, "")] += 1  # punct GE2PE kept that orth has not
        lead_tokens, lead_dropped = _punct_tokens(lead, changes)
        trail_tokens, trail_dropped = _punct_tokens(trail, changes)
        dropped_punct.extend(lead_dropped)
        dropped_punct.extend(trail_dropped)
        kept_punct.extend(lead_tokens)
        kept_punct.extend(trail_tokens)
        stats["punct_restored"] += len(lead_tokens) + len(trail_tokens)

        if mode == "flat":
            stats["stress_skipped"] += 1
        else:
            core, placed = _insert_stress(core)
            if placed:
                stats["stress_inserted"] += 1
            elif core and "ˈ" in core:
                stats["stress_already_present"] += 1

        entry = " ".join(piece for piece in (*lead_tokens, core, *trail_tokens) if piece)
        if entry:
            out_words.append(entry)

    # -- notes: everything that could not be applied, said out loud --------
    if mode == "empty":
        notes.append("No G2P words to normalise.")
    if stats.get("stress_digit_removed"):
        notes.append(
            "GE2PE's stress digit (1) removed; the canonical word-final rule applied instead."
        )
    if stats.get("stress_skipped"):
        notes.append(
            "Flat G2P output: no word boundaries, so the canonical word-final stress "
            "was not inserted."
        )
    if stats.get("stress_already_present"):
        notes.append("A stress mark was already present; not re-inserted.")
    if mode == "ipa":
        notes.append(
            "No compact stream: word-initial ʔ (alef onset) cannot be recovered "
            "from GE2PE's rendered IPA alone."
        )
    if kept_punct:
        notes.append(
            "Punctuation restored from orthography: "
            + " ".join(sorted(set(kept_punct)))
            + "."
        )
    if dropped_punct:
        notes.append(
            "Punctuation dropped, the way the goldens drop it: "
            + " ".join(sorted(set(dropped_punct)))
            + "."
        )
    if unknown:
        listing = ", ".join(f"{symbol}×{count}" for symbol, count in sorted(unknown.items()))
        notes.append(f"Unknown symbol(s) kept in the output and reported: {listing}.")
        log.warning("ipa_normalize: unknown symbols kept in the IPA: %s", dict(unknown))

    text = " ".join(out_words)
    log.debug(
        "canonical IPA: %d word(s), source=%s, %d change(s), %d unknown",
        len(out_words),
        mode,
        sum(changes.values()),
        sum(unknown.values()),
    )

    ordered = [
        SymbolChange(source, target, count)
        for (source, target), count in sorted(changes.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return NormalizedIPA(
        text=text,
        words=out_words,
        changes=ordered,
        unknown=sorted(unknown),
        notes=notes,
        stats=dict(stats),
        source=mode,
    )


# --------------------------------------------------------------------------
# Engine registry
# --------------------------------------------------------------------------
#: Engines whose weights were trained on canonical Persian IPA. Phonemes are
#: normalised into that convention for exactly these engines and left as
#: GE2PE produced them everywhere else - ``omnivoice`` and ``f5`` included.
CANONICAL_ENGINES: frozenset[str] = frozenset({"omnivoice_ipa"})


def normalizer_for(engine_name: str) -> Callable[..., NormalizedIPA] | None:
    """The normaliser ``engine_name`` needs, or None to leave IPA alone."""
    if engine_name and engine_name.strip().lower() in CANONICAL_ENGINES:
        return normalize_ge2pe_ipa_for_omnivoice
    return None


__all__ = [
    "CANONICAL_ENGINES",
    "ALL_PUNCT",
    "CANONICAL_INVENTORY",
    "NormalizedIPA",
    "SymbolChange",
    "SYMBOL_MAP",
    "normalize_ge2pe_ipa_for_omnivoice",
    "normalizer_for",
]
