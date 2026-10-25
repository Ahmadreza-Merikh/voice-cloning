"""Persian text normalisation, number verbalisation and language detection."""

from __future__ import annotations

import pytest

from voiceclone.text.lang import detect_language, has_persian, is_mixed_script
from voiceclone.text.normalize_fa import (
    ZWNJ,
    NormalizationOptions,
    fix_zwnj,
    normalize,
    normalize_for_g2p,
    remove_ezafe_marks,
    strip_bracketed_tags,
    unify_characters,
)
from voiceclone.text.numbers_fa import (
    decimal_to_words,
    expand_numbers,
    number_to_words,
    ordinal_to_words,
    to_ascii_digits,
)
from voiceclone.text.pipeline import TextPipeline
from voiceclone.text.prosody_fa import break_vocative_clauses


class TestCharacterUnification:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("ايران", "ایران"),        # Arabic yeh -> Persian yeh
            ("كتاب", "کتاب"),          # Arabic kaf -> keheh
            ("مدرسة", "مدرسه"),        # teh marbuta -> heh
            ("أحمد", "احمد"),          # hamza above alef -> bare alef
            ("علي", "علی"),
        ],
    )
    def test_arabic_forms_become_persian(self, source: str, expected: str) -> None:
        assert unify_characters(source) == expected

    def test_arabic_indic_digits_become_persian(self) -> None:
        assert unify_characters("٢٥٠٠") == "۲۵۰۰"

    def test_tatweel_is_removed(self) -> None:
        assert "ـ" not in normalize("مـــیرود")

    def test_persian_letters_are_untouched(self) -> None:
        for word in ("پنجره", "چای", "ژاله", "گل", "کتاب"):
            assert unify_characters(word) == word


class TestZWNJ:
    def test_repeated_zwnj_collapses(self) -> None:
        assert fix_zwnj(f"می{ZWNJ}{ZWNJ}{ZWNJ}رود") == f"می{ZWNJ}رود"

    def test_verb_prefix_is_reattached(self) -> None:
        assert fix_zwnj("من می خواهم") == f"من می{ZWNJ}خواهم"
        assert fix_zwnj("او نمی رود") == f"او نمی{ZWNJ}رود"

    def test_suffix_joining_is_opt_in(self) -> None:
        assert fix_zwnj("کتاب ها") == "کتاب ها"
        assert fix_zwnj("کتاب ها", join_suffixes=True) == f"کتاب{ZWNJ}ها"


class TestNumbers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "صفر"),
            (1, "یک"),
            (11, "یازده"),
            (20, "بیست"),
            (21, "بیست و یک"),
            (100, "صد"),
            (234, "دویست و سی و چهار"),
            (1000, "هزار"),
            (1234, "هزار و دویست و سی و چهار"),
            (2000, "دو هزار"),
            (1403, "هزار و چهارصد و سه"),
            (1_000_000, "یک میلیون"),
            (2_500_000, "دو میلیون و پانصد هزار"),
            (1_000_000_000, "یک میلیارد"),
        ],
    )
    def test_cardinals(self, value: int, expected: str) -> None:
        assert number_to_words(value) == expected

    def test_negative_numbers(self) -> None:
        assert number_to_words(-5).startswith("منفی")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, "اول"), (2, "دوم"), (3, "سوم"), (10, "دهم"), (23, "بیست و سوم"), (30, "سی‌ام")],
    )
    def test_ordinals(self, value: int, expected: str) -> None:
        assert ordinal_to_words(value) == expected

    def test_decimals(self) -> None:
        assert decimal_to_words("3", "14") == "سه ممیز چهارده صدم"

    def test_digit_conversion(self) -> None:
        assert to_ascii_digits("۱۴۰۳") == "1403"
        assert to_ascii_digits("٢٥") == "25"


class TestNumberExpansionInContext:
    def test_currency(self) -> None:
        assert "تومان" in expand_numbers("قیمت ۲۵۰۰ تومان")
        assert "۲۵۰۰" not in expand_numbers("قیمت ۲۵۰۰ تومان")

    def test_percent(self) -> None:
        assert "درصد" in expand_numbers("رشد 75%")

    def test_time_is_not_duplicated(self) -> None:
        result = expand_numbers("ساعت 14:30 قرار داریم")
        assert result.count("ساعت") == 1

    def test_bare_time_gains_the_word(self) -> None:
        assert expand_numbers("جلسه 09:05 است").count("ساعت") == 1

    def test_date(self) -> None:
        assert "مرداد" in expand_numbers("امروز ۱۴۰۳/۰۵/۰۶ است")

    def test_long_digit_runs_read_individually(self) -> None:
        result = expand_numbers("شماره 09121234567")
        assert "صفر" in result and "نه" in result

    def test_text_without_digits_is_untouched(self) -> None:
        original = "سلام دنیا"
        assert expand_numbers(original) == original

    def test_a_model_name_is_spelled_and_its_number_read_apart(self) -> None:
        # The user's news text: «تراشه M6» reached OmniVoice as «Mشش» and was said "me shesh".
        assert expand_numbers("تراشه M6 و A17") == "تراشه اِم شش و اِی هفده"
        assert expand_numbers("پردازنده i7") == "پردازنده آی هفت"

    def test_a_unit_after_a_number_is_named(self) -> None:
        assert expand_numbers("حافظه 8GB و نمایشگر 120Hz و شبکه 5G") == (
            "حافظه هشت گیگابایت و نمایشگر صد و بیست هرتز و شبکه پنج جی")

    def test_other_latin_words_are_only_parted_from_their_number(self) -> None:
        assert expand_numbers("گوشی iPhone15 و 15Pro") == "گوشی iPhone پانزده و پانزده Pro"


class TestNormalizePipeline:
    def test_whitespace_collapses(self) -> None:
        assert normalize("این  یک   تست  است  .") == "این یک تست است."

    def test_latin_tags_are_stripped(self) -> None:
        assert "laughter" not in normalize("متن [laughter] ادامه")

    def test_persian_brackets_survive(self) -> None:
        assert "توضیح" in strip_bracketed_tags("متن (توضیح) ادامه")

    def test_empty_input(self) -> None:
        assert normalize("") == ""

    def test_number_expansion_can_be_disabled(self) -> None:
        options = NormalizationOptions(expand_numbers=False)
        assert "۱۴۰۳" in normalize("سال ۱۴۰۳", options)

    def test_a_spelled_letter_keeps_its_mark_only_where_marks_are_kept(self) -> None:
        assert normalize("تراشه M6", NormalizationOptions(remove_diacritics=False)) == "تراشه اِم شش"
        assert normalize("تراشه M6", NormalizationOptions(remove_diacritics=True)) == "تراشه ام شش"

    def test_stray_thousands_separator_becomes_a_comma(self) -> None:
        assert "،" in normalize("سلام ٬ ایران")

    def test_digits_inside_numbers_keep_their_separator(self) -> None:
        # 1,500 must read as one number, not "one" and "five hundred".
        assert normalize("مبلغ 1,500 تومان").count("هزار") == 1


class TestEzafeMarks:
    """A written ezafe must not survive into the G2P.

    GE2PE emits a vowel for the kasra *and* applies its own ezafe rule, so
    the marked spelling doubles: کتابِ من -> "ketaebe" rather than "ketabe".
    Quoting the real model output here because a regression is silent - it
    mispronounces rather than raising.
    """

    @pytest.mark.parametrize(
        ("marked", "expected"),
        [
            ("کتابِ من", "کتاب من"),
            ("گلِ باغ", "گل باغ"),
            ("شهرِ ما", "شهر ما"),
            ("جشنِ بزرگِ بهار", "جشن بزرگ بهار"),
            ("باغِ.", "باغ."),
            ("خانهِ", "خانه"),
        ],
    )
    def test_word_final_kasra_is_dropped(self, marked: str, expected: str) -> None:
        assert remove_ezafe_marks(marked) == expected

    @pytest.mark.parametrize("word", ["کِرم", "کَرَم", "مُرده"])
    def test_word_internal_harakat_survive(self, word: str) -> None:
        # These are the only thing disambiguating a homograph; stripping them
        # would trade one mispronunciation for another.
        assert remove_ezafe_marks(word) == word

    def test_g2p_normalisation_applies_it(self) -> None:
        assert normalize_for_g2p("گلِ زیبای باغ") == "گل زیبای باغ"

    def test_plain_normalisation_leaves_it_alone(self) -> None:
        # Only the G2P front end has this quirk; the engines see the original.
        assert "ِ" in normalize("گلِ زیبای باغ")


class TestVocativeClauseBreak:
    """A greeting + pronoun is a new clause, not ezafe.

    Graphemic engines (OmniVoice, F5) otherwise speak ``سلامِ من`` for
    ``سلام من اذیت می‌کنم``. Real noun-phrase ezafe is not a vocative.
    """

    def test_reported_sentence_gains_a_comma(self) -> None:
        assert break_vocative_clauses("سلام من اذیت می‌کنم") == "سلام، من اذیت می‌کنم"

    def test_noun_phrase_ezafe_is_untouched(self) -> None:
        assert break_vocative_clauses("کتاب من روی میز است") == "کتاب من روی میز است"
        assert break_vocative_clauses("گل باغ امروز شکفت") == "گل باغ امروز شکفت"

    def test_already_punctuated_is_untouched(self) -> None:
        assert break_vocative_clauses("سلام. من می‌روم") == "سلام. من می‌روم"
        assert break_vocative_clauses("سلام، من می‌روم") == "سلام، من می‌روم"

    def test_multiword_greeting(self) -> None:
        assert break_vocative_clauses("صبح بخیر من بیدار شدم") == "صبح بخیر، من بیدار شدم"
        assert break_vocative_clauses("سلام علیکم شما خوبید") == "سلام علیکم، شما خوبید"

    def test_other_pronouns(self) -> None:
        assert break_vocative_clauses("درود تو آمدی") == "درود، تو آمدی"
        assert break_vocative_clauses("بله ما رفتیم") == "بله، ما رفتیم"

    def test_pipeline_applies_it(self) -> None:
        from voiceclone.config import TextConfig

        prepared = TextPipeline(TextConfig()).prepare("سلام من اذیت می‌کنم", engine_name="omnivoice")
        assert prepared.normalized.startswith("سلام، من")
        assert any("ezafe" in note for note in prepared.notes)


class TestLanguageDetection:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("سلام دنیا", "fa"), ("Hello world", "en"), ("سلام Hello", "fa")],
    )
    def test_detection(self, text: str, expected: str) -> None:
        assert detect_language(text) == expected

    def test_mixed_script(self) -> None:
        assert is_mixed_script("سلام Hello")
        assert not is_mixed_script("سلام دنیا")

    def test_has_persian(self) -> None:
        assert has_persian("کتاب")
        assert not has_persian("book")

    def test_empty_falls_back_to_default(self) -> None:
        assert detect_language("", default="en") == "en"
