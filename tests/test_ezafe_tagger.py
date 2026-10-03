from pathlib import Path

import pytest

from voiceclone.text import ezafe_tagger as et

EXPORT = Path("~/.cache/persian-voice-cloning/ezafe-tagger").expanduser() / et.REVISION[:12]


def test_strided_spans_follow_spacy_transformers():
    assert et.strided_spans(0) == []
    assert et.strided_spans(5) == [(0, 5)]
    assert et.strided_spans(95) == [(0, 95)]
    assert et.strided_spans(96) == [(0, 96)]
    assert et.strided_spans(128) == [(0, 128)]
    assert et.strided_spans(129) == [(0, 128), (96, 129)]
    assert et.strided_spans(200) == [(0, 128), (96, 200)]
    assert et.strided_spans(300) == [(0, 128), (96, 224), (192, 300)]


def test_a_wordpiece_belongs_to_every_token_whose_characters_it_covers():
    # Special tokens carry no characters; the space after a token is the token's own.
    assert et.align(["کتاب", "من", "."], [(0, 0), (0, 4), (5, 7), (8, 9), (0, 0)]) == [[1], [2], [3]]
    assert et.align(["ab", "cd"], [(0, 0), (1, 4), (0, 0)]) == [[1], [1]]
    assert et.align(["ab", "cd"], [(0, 1), (1, 2), (3, 5)]) == [[0, 1], [2]]


@pytest.mark.skipif(not et.exported(EXPORT), reason="the tagger has not been exported on this machine")
def test_the_exported_tagger_gives_the_spacy_pipelines_tags():
    tokens = ["دانشجو", "کتاب", "را", "روی", "میز", "گذاشت", "و", "دو", "ساعت", "منتظر", "استاد", "ماند", "."]
    # What scripts/ezafe_worker.py returned through spaCy on 2026-10-03.
    assert et.ParsBertTagger(EXPORT).tag(tokens) == [
        "NOUN", "NOUN", "ADP", "ADP,EZ", "NOUN", "VERB", "CCONJ", "NUM", "NOUN", "ADJ,EZ", "NOUN", "VERB", "PUNCT"]
