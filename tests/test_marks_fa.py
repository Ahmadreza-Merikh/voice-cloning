"""Marks go where the decision says, in the form Persian writes them, and nowhere else."""

from __future__ import annotations

from voiceclone.text.marks_fa import (
    DAMMA,
    FATHA,
    HAMZA_ABOVE,
    KASRA,
    SUKUN,
    Word,
    mark_words,
    rebuild,
    with_ezafe,
    with_first_vowel,
    with_sukun,
)


def test_the_ezafe_takes_the_written_form_its_last_letter_asks_for() -> None:
    assert with_ezafe("کتاب") == "کتاب" + KASRA
    assert with_ezafe("خانه") == "خانه" + HAMZA_ABOVE
    assert with_ezafe("دانشگاه") == "دانشگاه" + KASRA
    assert with_ezafe("توجه", "tavajjoh") == "توجه" + KASRA
    assert with_ezafe("صدا") == "صدای"
    assert with_ezafe("دانشجو") == "دانشجوی"
    assert with_ezafe("عمومی") == "عمومی" + KASRA
    assert with_ezafe("هفتهٔ") == "هفتهٔ"
    assert with_ezafe("کتاب" + KASRA) == "کتاب" + KASRA


def test_a_sukun_only_closes_a_final_consonant() -> None:
    assert with_sukun("باغ") == "باغ" + SUKUN
    assert with_sukun("دانشگاه") == "دانشگاه" + SUKUN
    for word in ("خانه", "دانشجو", "عمومی", "صدا", "باغ" + KASRA):
        assert with_sukun(word) == word


def test_the_first_vowel_goes_on_the_first_letter_only_when_the_script_leaves_it_out() -> None:
    assert with_first_vowel("تبر", "tabar") == "ت" + FATHA + "بر"
    assert with_first_vowel("کشاورز", "keSAvarz") == "ک" + KASRA + "شاورز"
    assert with_first_vowel("قفل", "qofl") == "ق" + DAMMA + "فل"
    assert with_first_vowel("امانت", "?amAnat") == "ا" + FATHA + "مانت"
    assert with_first_vowel("عمل", "?amal") == "ع" + FATHA + "مل"
    assert with_first_vowel("سوار", "savAr") == "س" + FATHA + "وار"
    for word, reading in (("آب", "?Ab"), ("دانشجو", "dAneSju"), ("خود", "xod"), ("کوچک", "kuCak"),
                          ("بیا", "biyA"), ("میز", "miz"), ("تبر", None)):
        assert with_first_vowel(word, reading) == word, word


def test_a_sentence_gets_each_kind_of_mark_only_when_asked() -> None:
    sentence = "دانشجو کتاب را روی میز گذاشت و دو ساعت منتظر استاد ماند."
    words = [Word("دانشجو", pos="NOUN", next_pos="NOUN", reading="dAneSju"),
             Word("کتاب", pos="NOUN", next_pos="ADP", reading="ketAb"),
             Word("را", pos="ADP", next_pos="ADP"),
             Word("روی", ezafe=True, pos="ADP", next_pos="NOUN"),
             Word("میز", pos="NOUN", next_pos="VERB", reading="miz"),
             Word("گذاشت", pos="VERB", next_pos="CCONJ"),
             Word("و", pos="CCONJ", next_pos="NUM"),
             Word("دو", pos="NUM", next_pos="NOUN"),
             Word("ساعت", pos="NOUN", next_pos="ADJ", reading="sA?at"),
             Word("منتظر", ezafe=True, pos="ADJ", next_pos="NOUN", reading="montazer"),
             Word("استاد", pos="NOUN", next_pos="VERB", reading="?ostAd"),
             Word("ماند", pos="VERB", next_pos=None)]
    assert rebuild(sentence, mark_words(words, ezafe=False)) == sentence
    e = rebuild(sentence, mark_words(words))
    assert e == "دانشجو کتاب را رویِ میز گذاشت و دو ساعت منتظرِ استاد ماند."
    en = rebuild(sentence, mark_words(words, sukun=True))
    assert en == "دانشجو کتاب را رویِ میزْ گذاشت و دو ساعتْ منتظرِ استادْ ماند."
    env = rebuild(sentence, mark_words(words, sukun=True, vowels=True))
    assert env == "دانشجو کِتاب را رویِ میزْ گذاشت و دو ساعتْ مُنتظرِ اُستادْ ماند."
