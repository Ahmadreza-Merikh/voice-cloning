"""Write the ezafe that hazm's ParsBERT tagger predicts into Persian text.

The listening test (FINDINGS section 15) heard OmniVoice read 9 words of 147
wrong with the tagger's ezafes written in, against 12 on plain script, and no
step raised CER. This decides where the ezafe goes - the tagger, one sentence
at a time, on the tokens it was measured on (each punctuation mark a token of
its own) - and leaves the writing to :func:`marks_fa.with_ezafe`.

Three things overrule the tagger:

* ``NO_EZAFE_ON``: words the ezafe is never written on. روی: in the روی check
  (FINDINGS section 18) it was heard wrong once in 12 with the kasra and never
  without it, and nothing else got worse
* ``NO_EZAFE_BEFORE``: pairs written apart and read as one word. The listener
  heard مدیر عامل right only without the ezafe the tagger gave it
* a word that does not end in an Arabic-script letter (Latin, digits) is left
  alone

The tagger (:mod:`ezafe_tagger`) runs in this process, loaded for one text and
released after it - 0.86 GB at its peak - and only with enough memory free.
Sentences it has tagged are remembered, so text heard again costs nothing.
"""

from __future__ import annotations

import gc
import re
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from ..config import EzafeConfig
from ..errors import TextProcessingError
from ..logging_utils import get_logger
from .marks_fa import PUNCT, bare, last_letter, with_ezafe

log = get_logger("text.ezafe")

SENTENCE_END = ".!؟?"
#: Words the ezafe is never written on, each heard at least as well without it.
NO_EZAFE_ON = frozenset({"روی"})
#: Pairs the ezafe must not join, each heard right only without it.
NO_EZAFE_BEFORE = frozenset({("مدیر", "عامل")})
#: The one-time export reads a 655 MB file into memory twice over (2.3 GB at its peak).
EXPORT_EXTRA_GB = 1.5
CACHE_SIZE = 4096
_SPACE = re.compile(r"(\s+)")

Tagger = Callable[[list[str]], list[str]]


def plan(text: str) -> tuple[list[str], list[list[int]]]:
    """Cut ``text`` into parts and group the word parts into sentences.

    The parts are the words with their punctuation and the whitespace between
    them, so joining them gives the text back. A sentence ends at a part whose
    trailing punctuation holds . ! ؟ or ?, or at a line break.
    """
    parts = _SPACE.split(text)
    groups: list[list[int]] = []
    current: list[int] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        if part.isspace():
            if "\n" in part and current:
                groups.append(current)
                current = []
            continue
        current.append(i)
        core = part.strip(PUNCT)
        tail = part[part.index(core) + len(core):] if core else part
        if any(c in SENTENCE_END for c in tail):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return parts, groups


def tokens_of(parts: list[str], group: list[int]) -> tuple[list[str], list[int | None]]:
    """One sentence as the tagger was measured on it: every punctuation mark its own token.

    ``where`` gives, for each token, the part it is the word of (None for punctuation).
    """
    tokens: list[str] = []
    where: list[int | None] = []
    for i in group:
        raw = parts[i]
        core = raw.strip(PUNCT)
        if not core:
            tokens += list(raw)
            where += [None] * len(raw)
            continue
        start = raw.index(core)
        lead, trail = raw[:start], raw[start + len(core):]
        tokens += [*lead, core, *trail]
        where += [None] * len(lead) + [i] + [None] * len(trail)
    return tokens, where


def _persian(word: str) -> bool:
    last = last_letter(word)
    return bool(last) and "؀" <= last <= "ۿ"


def write_ezafe(text: str, tag: Tagger) -> tuple[str, list[str]]:
    """``text`` with the ezafe written on every word ``tag`` gives one, and those words."""
    parts, groups = plan(text)
    written: list[str] = []
    for group in groups:
        tokens, where = tokens_of(parts, group)
        tags = tag(tokens)
        words = [(i, t) for i, t in zip(where, tags, strict=True) if i is not None]
        for n, (i, t) in enumerate(words):
            if ",EZ" not in t:
                continue
            raw = parts[i]
            core = raw.strip(PUNCT)
            following = parts[words[n + 1][0]].strip(PUNCT) if n + 1 < len(words) else ""
            if bare(core) in NO_EZAFE_ON or (bare(core), bare(following)) in NO_EZAFE_BEFORE or not _persian(core):
                continue
            marked = with_ezafe(core)
            if marked != core:
                start = raw.index(core)
                parts[i] = raw[:start] + marked + raw[start + len(core):]
                written.append(core)
    return "".join(parts), written


class EzafeWriter:
    """Runs the tagger for a batch of texts, within the memory it is allowed."""

    def __init__(self, config: EzafeConfig) -> None:
        self.config = config
        self._cache: OrderedDict[tuple[str, ...], list[str]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        setting = self.config.enabled
        return not (setting is False or str(setting).lower() in {"false", "off", "no"})

    @property
    def required(self) -> bool:
        setting = self.config.enabled
        return setting is True or str(setting).lower() in {"true", "on", "yes", "required"}

    @property
    def export_dir(self) -> Path:
        from .ezafe_tagger import REVISION

        return Path(self.config.export_dir).expanduser() / REVISION[:12]

    def _room(self, needed_gb: float) -> str | None:
        """Why the tagger may not load now, or None if it may."""
        from .. import memory

        level = memory.pressure_level()
        if level is not None and level >= memory.WARNING:
            return f"memory pressure is at {memory.describe(level)}"
        available = memory.available_gb()
        if available is not None and available < needed_gb:
            return f"{available:.1f} GB of memory available, the tagger needs {needed_gb:.1f}"
        return None

    def _load(self):
        from . import ezafe_tagger

        out = self.export_dir
        needed = float(self.config.min_available_gb)
        if not ezafe_tagger.exported(out):
            reason = self._room(needed + EXPORT_EXTRA_GB)
            if reason:
                raise TextProcessingError(f"The ezafe tagger was not set up: {reason}.")
            log.info("Exporting the ezafe tagger to %s (once)", out)
            ezafe_tagger.export(ezafe_tagger.snapshot(download=bool(self.config.download)), out)
            gc.collect()
        reason = self._room(needed)
        if reason:
            raise TextProcessingError(f"The ezafe tagger was not started: {reason}.")
        return ezafe_tagger.ParsBertTagger(out)

    def write(self, texts: list[str]) -> tuple[list[str], list[str], str | None]:
        """Each text with its ezafes written, the words marked, and why not if it could not run."""
        loaded = None

        def tag(tokens: list[str]) -> list[str]:
            nonlocal loaded
            key = tuple(tokens)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if loaded is None:
                loaded = self._load()
            tags = loaded.tag(tokens)
            self._cache[key] = tags
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
            return tags

        try:
            done = [write_ezafe(text, tag) for text in texts]
        except Exception as exc:
            if self.required:
                raise
            log.warning("Ezafe not written: %s", exc)
            return list(texts), [], str(getattr(exc, "message", exc))
        finally:
            loaded = None
            gc.collect()
        return [text for text, _ in done], [w for _, words in done for w in words], None


__all__ = ["NO_EZAFE_BEFORE", "NO_EZAFE_ON", "EzafeWriter", "plan", "tokens_of", "write_ezafe"]
