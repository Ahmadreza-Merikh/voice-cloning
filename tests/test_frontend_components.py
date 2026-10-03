"""The front-end scorer reads ezafe and vowels off pronunciations the same way for gold and predictions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sfc = load("score_frontend_components")
mfc = load("measure_frontend_components")


def test_ezafe_is_read_off_the_ending_by_the_last_letter() -> None:
    cases = [
        ("کتاب", "ketAb", ("ketAb", False)), ("کتاب", "ketAbe", ("ketAb", True)), ("کتاب", "ketAb-e", ("ketAb", True)),
        ("خانه", "xAne", ("xAne", False)), ("خانه", "xAneye", ("xAne", True)), ("خانه", "xAne-ye", ("xAne", True)),
        ("همسایه", "hamsAye", ("hamsAye", False)), ("همسایه", "hamsAyeye", ("hamsAye", True)),
        ("راه", "rAhe", ("rAh", True)), ("که", "ke", ("ke", False)),
        ("شخصی", "Saxsi", ("Saxsi", False)), ("شخصی", "Saxsiye", ("Saxsi", True)), ("برای", "barAye", ("barA", True)),
        ("بالا", "bAlAye", ("bAlA", True)), ("شروع", "Soru?", ("Soru?", False)),
        ("مدرسه", "madrese1", ("madrese", True)), ("هفتهٔ", "hafteye", ("hafte", True)),
    ]
    for word, token, expected in cases:
        assert sfc.ezafe_split(word, token) == expected, (word, token)


def test_ge2pe_letters_become_sentencebench_letters() -> None:
    assert sfc.from_ge2pe("m/dreseye") == "madreseye"
    assert sfc.from_ge2pe("@iran") == "?irAn"
    assert sfc.from_ge2pe("ke$id") == "keSid"
    assert sfc.from_ge2pe(";ale") == "ZAle"
    assert sfc.from_ge2pe("cay") == "CAy"


def test_a_conjunction_merged_into_the_word_before_it_is_split_off() -> None:
    ws = sfc.words("به دست و صورتم کرم زدم.")
    gold = sfc.gold_readings(ws, ["be", "dast-o", "suratam", "kerem", "zadam"])
    assert gold[1]["vowels"] == ["a"] and gold[1]["ezafe"] is False
    assert gold[2] is None
    assert gold[3]["vowels"] == ["uaa"]


def test_two_words_merged_into_one_token_are_unscorable_in_gold() -> None:
    ws = sfc.words("اینجا ملک ما است")
    gold = sfc.gold_readings(ws, ["?injA", "melk", "mAst"])
    assert gold[2] is None and gold[3] is None
    assert gold[1]["vowels"] == ["e"]


def test_a_prediction_that_merges_words_is_shared_out_by_the_gold() -> None:
    ws = sfc.words("به طور خلاصه")
    gold = sfc.gold_readings(ws, ["be", "tor-e", "xolAse"])
    got = sfc.predicted_readings(ws, [sfc.from_ge2pe(t) for t in ["betore", "xolase"]], gold)
    assert got[0] == {"ezafe": False, "vowels": "e"}
    assert got[1] == {"ezafe": True, "vowels": "o"}
    assert got[2]["vowels"] == "oAe" and got[2]["ezafe"] is False


def test_a_word_the_prediction_drops_counts_as_wrong_not_unanswered() -> None:
    ws = sfc.words("او گل قرمز را")
    gold = sfc.gold_readings(ws, ["?u", "gol-e", "qermez", "rA"])
    got = sfc.predicted_readings(ws, ["?u", "gole", "rA"], gold)
    assert got[2] == {"ezafe": False, "vowels": ""}


def sentence(sid: str, text: str, phonemes: str, homograph: dict | None = None) -> dict:
    ws = sfc.words(text)
    return {"id": sid, "subset": "x", "text": text, "words": ws, "gold": sfc.gold_readings(ws, phonemes.split()),
            "homograph": homograph}


def test_tallies_count_added_and_missed_ezafe_and_the_never_baseline() -> None:
    sets = {"sentencebench": [sentence("a", "کتاب خوب برادرم", "ketAb-e xub-e barAdaram"),
                              sentence("b", "باران بند آمد", "bArAn band ?Amad",
                                       {"index": 1, "vowels": "a"})]}
    preds = {
        "tagger": {"sentencebench": {"a": {"id": "a", "ezafe": [True, False, False]},
                                     "b": {"id": "b", "ezafe": [True, False, False]}}},
        "homo-ge2pe": {"sentencebench": {"a": {"id": "a", "tokens": ["ketabe", "xube", "b/rad/r/m"], "scheme": "ge2pe"},
                                         "b": {"id": "b", "tokens": ["baran", "bond", "@am/d"], "scheme": "ge2pe"}}},
        "kaamel": {"sentencebench": {"a": {"id": "a", "lexicon": [["ketAb"], ["xub"], []]},
                                     "b": {"id": "b", "lexicon": [["bArAn"], ["band", "bond"], ["?Amad"]]}}},
    }
    result = sfc.score(sets, preds)["sentencebench"]
    tagger = result["ezafe"]["tagger"]
    assert (tagger["tp"], tagger["fp"], tagger["fn"], tagger["tn"]) == (1, 1, 1, 3)
    assert tagger["clean_sentences"] == 0
    never = result["ezafe"]["never"]
    assert (never["tp"], never["fn"], never["tn"]) == (0, 2, 4) and never["error_rate"] == round(2 / 6, 4)
    assert result["ezafe"]["homo-ge2pe"]["errors"] == 0
    assert result["ezafe_best_minus_runner_up"]["best"] == "homo-ge2pe"
    ge = result["vowels"]["homo-ge2pe"]
    assert (ge["words"], ge["correct"], ge["homographs_correct"]) == (6, 5, 0)
    lex = result["vowels"]["kaamel"]
    assert (lex["answered"], lex["correct"]) == (4, 4)
    joined = result["vowels"]["kaamel-then-homo-ge2pe"]
    assert (joined["answered"], joined["correct"]) == (6, 5)


def test_review_gold_words_are_the_texts_words_and_the_heard_errors_are_nineteen() -> None:
    body = json.loads(sfc.REVIEW_GOLD.read_text(encoding="utf-8"))
    sentences = sfc.review14()
    assert len(sentences) == 14 and sum(len(s["words"]) for s in sentences) == 199
    assert sum(len(s["heard"]) for s in body["sentences"]) == 19
    for s in body["sentences"]:
        for h in s["heard"]:
            w = s["words"][h["index"]]
            assert w["from"] == "review"
            assert ("ezafe" if h["kind"] == "ezafe" else "pron") in w["review_confirms"]


def test_taggers_get_punctuation_as_its_own_tokens_and_ezafe_lands_on_the_gold_word() -> None:
    tokens, where = mfc.tagger_tokens("«ژاله» زیر چراغ، چترش را بست.", 6)
    assert tokens == ["«", "ژاله", "»", "زیر", "چراغ", "،", "چترش", "را", "بست", "."]
    tags = ["PUNCT", "NOUN", "PUNCT", "ADP,EZ", "NOUN", "PUNCT", "NOUN", "ADP", "VERB", "PUNCT"]
    assert mfc.ezafe_from_tags(tags, where, 6) == [False, True, False, False, False, False]


def test_overlap_finds_equal_and_contained_sentences_only_by_letters() -> None:
    sentences = [{"id": "a", "subset": "x", "text": "کتاب خوب برادرم روی میز اتاق است."},
                 {"id": "b", "subset": "x", "text": "باران بند آمد"},
                 {"id": "c", "subset": "y", "text": "ژاله زیر چراغ، چترش را در باران شدید بست."}]
    others = ["کتابِ خوبِ برادرم، روی میز اتاق است", "باران بند آمد و آفتاب شد", "چیزی دیگر"]
    by = sfc.overlap(sentences, others)
    assert by["x"] == {"sentences": 2, "equal": 1, "contained": 0}
    assert by["y"] == {"sentences": 1, "equal": 0, "contained": 0}
