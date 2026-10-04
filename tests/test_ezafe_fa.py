"""The ezafe goes where the tagger says, in the form the word asks for, and only for engines that read it."""

from __future__ import annotations

import pytest

from voiceclone.config import EzafeConfig, TextConfig
from voiceclone.errors import TextProcessingError
from voiceclone.text.ezafe_fa import EzafeWriter, plan, tokens_of, write_ezafe
from voiceclone.text.marks_fa import HAMZA_ABOVE, KASRA
from voiceclone.text.pipeline import TextPipeline


class FakeTagger:
    """Gives the ezafe to the words it is told to, and records what it was asked."""

    def __init__(self, *with_ezafe: str) -> None:
        self.with_ezafe = set(with_ezafe)
        self.asked: list[list[str]] = []

    def tag(self, tokens: list[str]) -> list[str]:
        self.asked.append(list(tokens))
        return ["NOUN,EZ" if t in self.with_ezafe else "NOUN" for t in tokens]


def test_the_parts_give_the_text_back_and_a_sentence_ends_at_its_stop_or_a_line_break() -> None:
    text = "کتاب من اینجاست. «خانه» کجاست؟\nباغ  زیبا"
    parts, groups = plan(text)
    assert "".join(parts) == text
    assert [[parts[i] for i in g] for g in groups] == [["کتاب", "من", "اینجاست."], ["«خانه»", "کجاست؟"],
                                                       ["باغ", "زیبا"]]


def test_every_punctuation_mark_is_a_token_of_its_own() -> None:
    parts, groups = plan("«خانه»، کو؟")
    tokens, where = tokens_of(parts, groups[0])
    assert tokens == ["«", "خانه", "»", "،", "کو", "؟"]
    assert [parts[i] if i is not None else None for i in where] == [None, "«خانه»،", None, None, "کو؟", None]


def test_the_ezafe_is_written_in_the_form_each_word_asks_for() -> None:
    tagger = FakeTagger("کتاب", "خانه", "دانشجو", "پای", "توجه", "شبیه")
    text, written = write_ezafe("کتاب و «خانه» و دانشجو پای میز. توجه و شبیه.", tagger.tag)
    assert text == ("کتاب" + KASRA + " و «خانه" + HAMZA_ABOVE + "» و دانشجوی پای" + KASRA + " میز. توجه" + KASRA
                    + " و شبیه" + KASRA + ".")
    assert written == ["کتاب", "خانه", "دانشجو", "پای", "توجه", "شبیه"]


def test_rui_never_gets_the_ezafe_and_a_kasra_the_writer_put_on_it_stays() -> None:
    tagger = FakeTagger("لیوان", "روی", "رویِ")
    text, written = write_ezafe("لیوان چای را روی میز گذاشت و کتاب را رویِ زانو باز کرد.", tagger.tag)
    assert text == "لیوان" + KASRA + " چای را روی میز گذاشت و کتاب را رویِ زانو باز کرد."
    assert written == ["لیوان"]


def test_each_sentence_is_tagged_on_its_own() -> None:
    tagger = FakeTagger()
    write_ezafe("کتاب من. خانه تو!", tagger.tag)
    assert tagger.asked == [["کتاب", "من", "."], ["خانه", "تو", "!"]]


def test_a_pair_read_as_one_word_a_latin_word_and_a_marked_one_are_left_alone() -> None:
    tagger = FakeTagger("مدیر", "عامل", "iPhone", "کتابْ")
    text, written = write_ezafe("مدیر عامل شرکت با iPhone و کتابْ آمد.", tagger.tag)
    assert text == "مدیر عامل" + KASRA + " شرکت با iPhone و کتابْ آمد."
    assert written == ["عامل"]


def writer(config: EzafeConfig, tagger: FakeTagger | None = None, loads: list[int] | None = None) -> EzafeWriter:
    w = EzafeWriter(config)

    def load() -> FakeTagger:
        if loads is not None:
            loads.append(1)
        if tagger is None:
            raise TextProcessingError("The ezafe tagger was not started: 0.4 GB of memory available, the tagger needs 1.5.")
        return tagger

    w._load = load  # type: ignore[method-assign]
    return w


def test_on_auto_text_goes_on_unmarked_when_the_tagger_cannot_run_and_says_why() -> None:
    texts, written, failed = writer(EzafeConfig(enabled="auto")).write(["کتاب من"])
    assert (texts, written) == (["کتاب من"], [])
    assert "0.4 GB" in failed


def test_when_required_it_stops_instead() -> None:
    with pytest.raises(TextProcessingError):
        writer(EzafeConfig(enabled=True)).write(["کتاب من"])


def test_off_means_off() -> None:
    assert not EzafeWriter(EzafeConfig(enabled=False)).enabled
    assert not EzafeWriter(EzafeConfig(enabled="false")).enabled
    assert EzafeWriter(EzafeConfig()).enabled


def test_a_sentence_tagged_once_is_remembered_and_the_tagger_not_loaded_again() -> None:
    loads: list[int] = []
    tagger = FakeTagger("کتاب")
    w = writer(EzafeConfig(), tagger, loads)
    first = w.write(["کتاب من."])
    second = w.write(["کتاب من."])
    assert first == second == (["کتاب" + KASRA + " من."], ["کتاب"], None)
    assert len(loads) == 1 and len(tagger.asked) == 1


def pipeline(tagger: FakeTagger | None) -> TextPipeline:
    p = TextPipeline(TextConfig())
    p._ezafe = writer(EzafeConfig(), tagger)
    return p


def test_omnivoice_gets_the_ezafe_and_other_engines_plain_script() -> None:
    tagger = FakeTagger("کتاب")
    omni = pipeline(tagger).prepare("کتاب من روی میز است.", engine_name="omnivoice")
    assert omni.chunks[0].synth_text == "کتاب" + KASRA + " من روی میز است."
    assert any("Ezafe written in on 1 word" in note for note in omni.notes)
    f5 = pipeline(tagger).prepare("کتاب من روی میز است.", engine_name="f5")
    assert f5.chunks[0].synth_text == "کتاب من روی میز است."


def test_the_normaliser_keeps_hamza_and_a_written_ezafe_for_engines_that_read_marks() -> None:
    omni = pipeline(FakeTagger()).prepare("تأخیر خانۀ ما", engine_name="omnivoice")
    f5 = pipeline(FakeTagger()).prepare("تأخیر خانۀ ما", engine_name="f5")
    assert omni.normalized == "تأخیر خانه" + HAMZA_ABOVE + " ما"
    assert f5.normalized == "تاخیر خانه ما"


def test_a_tagger_that_cannot_run_leaves_the_text_plain_and_says_so() -> None:
    prepared = pipeline(None).prepare("کتاب من روی میز است.", engine_name="omnivoice")
    assert prepared.chunks[0].synth_text == "کتاب من روی میز است."
    assert any(note.startswith("Ezafe not written") for note in prepared.notes)
