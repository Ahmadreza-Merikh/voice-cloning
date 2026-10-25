"""Persian number verbalisation.

TTS models read digits badly and inconsistently, so every numeric form is
expanded to words before synthesis. Implemented natively rather than via
``num2words`` so that the connector "و" placement, ordinals, currency and
Persian decimal/thousand separators all behave correctly - and so the package
works without that dependency.
"""

from __future__ import annotations

import re

from ..logging_utils import get_logger

log = get_logger("text.numbers")

ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
TEENS = [
    "ده", "یازده", "دوازده", "سیزده", "چهارده",
    "پانزده", "شانزده", "هفده", "هجده", "نوزده",
]
TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
HUNDREDS = [
    "", "صد", "دویست", "سیصد", "چهارصد",
    "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد",
]
#: Short scales, ascending. Persian uses میلیارد (10^9), not "بیلیون".
SCALES = [
    (10**24, "سپتیلیون"),
    (10**21, "سکستیلیون"),
    (10**18, "کوینتیلیون"),
    (10**15, "کوادریلیون"),
    (10**12, "تریلیون"),
    (10**9, "میلیارد"),
    (10**6, "میلیون"),
    (10**3, "هزار"),
]

#: Ordinals that are irregular; everything else is <cardinal> + "م".
IRREGULAR_ORDINALS = {
    1: "اول",
    2: "دوم",
    3: "سوم",
}

DECIMAL_PLACES = ["", "دهم", "صدم", "هزارم", "ده‌هزارم", "صدهزارم", "میلیونیم"]

CURRENCIES = {
    "تومان": "تومان",
    "ريال": "ریال",
    "ریال": "ریال",
    "$": "دلار",
    "دلار": "دلار",
    "€": "یورو",
    "یورو": "یورو",
    "£": "پوند",
    "پوند": "پوند",
}

MONTHS_JALALI = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]

# Digit forms: Persian (U+06F0), Arabic-Indic (U+0660), ASCII.
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS, "0123456789" * 2
)


def to_ascii_digits(text: str) -> str:
    """Convert Persian/Arabic-Indic digits to ASCII."""
    return text.translate(_DIGIT_TRANSLATION)


def to_persian_digits(text: str) -> str:
    """Convert ASCII digits to Persian digits."""
    return text.translate(str.maketrans("0123456789", PERSIAN_DIGITS))


# --------------------------------------------------------------------------
# Cardinals
# --------------------------------------------------------------------------
def _three_digits_to_words(value: int) -> list[str]:
    """Words for 1..999 (no scale suffix)."""
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        parts.append(HUNDREDS[hundreds])

    if remainder >= 20:
        tens, ones = divmod(remainder, 10)
        parts.append(TENS[tens])
        if ones:
            parts.append(ONES[ones])
    elif remainder >= 10:
        parts.append(TEENS[remainder - 10])
    elif remainder:
        parts.append(ONES[remainder])

    return parts


def number_to_words(value: int) -> str:
    """Persian words for an integer, e.g. ``1234`` -> 'هزار و دویست و سی و چهار'."""
    value = int(value)
    if value == 0:
        return "صفر"

    prefix = ""
    if value < 0:
        prefix = "منفی "
        value = -value

    parts: list[str] = []
    remainder = value

    for scale_value, scale_name in SCALES:
        if remainder >= scale_value:
            count, remainder = divmod(remainder, scale_value)
            # "هزار" not "یک هزار"; every other scale keeps its multiplier.
            if count == 1 and scale_value == 1000:
                parts.append(scale_name)
            else:
                # `count` is < 1000 for every scale except the largest one.
                count_words = (
                    number_to_words(count)
                    if count >= 1000
                    else " و ".join(_three_digits_to_words(count))
                )
                parts.append(f"{count_words} {scale_name}")

    if remainder:
        parts.extend(_three_digits_to_words(remainder))

    return prefix + " و ".join(part for part in parts if part)


def ordinal_to_words(value: int) -> str:
    """Persian ordinal, e.g. ``3`` -> 'سوم', ``21`` -> 'بیست و یکم'."""
    value = int(value)
    if value in IRREGULAR_ORDINALS:
        return IRREGULAR_ORDINALS[value]

    cardinal = number_to_words(value)
    # 'سه' -> 'سوم' (23 -> بیست و سوم), and words ending in a vowel letter
    # take '‌ام' with a ZWNJ (30 -> سی‌ام) rather than a bare 'م'.
    if cardinal.endswith("سه"):
        return cardinal[: -len("سه")] + "سوم"
    if cardinal.endswith(("ی", "و", "ا")):
        return cardinal + "‌ام"
    return cardinal + "م"


def decimal_to_words(integer_part: str, fraction_part: str) -> str:
    """Verbalise a decimal number.

    ``3.14`` -> 'سه ممیز چهارده صدم'. Long fractions fall back to
    digit-by-digit reading, which is how Persian speakers handle them.
    """
    integer_value = int(integer_part or 0)
    fraction = fraction_part.rstrip("0") or "0"

    if fraction == "0":
        return number_to_words(integer_value)

    head = number_to_words(integer_value)
    if len(fraction) < len(DECIMAL_PLACES):
        return f"{head} ممیز {number_to_words(int(fraction))} {DECIMAL_PLACES[len(fraction)]}"

    digits = " ".join(number_to_words(int(d)) for d in fraction)
    return f"{head} ممیز {digits}"


# --------------------------------------------------------------------------
# Pattern-level expansion
# --------------------------------------------------------------------------
_TIME_RE = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?![\d:])")
_DATE_RE = re.compile(r"(?<![\d/])(1[0-5]\d{2}|[12]\d{3})[/-](\d{1,2})[/-](\d{1,2})(?![\d/])")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[٪%]")
_ORDINAL_RE = re.compile(r"(?<!\d)(\d+)\s*(?:ام|ـم|م)(?=\s|$|[،.؛؟!])")
_RANGE_RE = re.compile(r"(?<![\d.])(\d+)\s*(?:-|–|تا)\s*(\d+)(?![\d.])")
_DECIMAL_RE = re.compile(r"(?<![\d.])(\d+)[.،٫](\d+)(?![\d.])")
_INTEGER_RE = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
_THOUSANDS_RE = re.compile(r"(?<=\d)[,٬](?=\d{3}\b)")
_CURRENCY_RE = re.compile(
    r"(\d[\d,٬]*(?:\.\d+)?)\s*(تومان|ریال|ريال|دلار|یورو|پوند)|([$€£])\s*(\d[\d,٬]*(?:\.\d+)?)"
)
#: Long digit runs (phone numbers, IDs, card numbers) read one digit at a time.
_DIGIT_RUN_RE = re.compile(r"(?<!\d)(\d{7,})(?!\d)")
#: Latin letters as a Persian reader names them, for model names like M6 or A17.
#: Marked where the plain spelling reads another way (ام is also om).
LATIN_LETTER_NAMES = {
    "a": "اِی", "b": "بی", "c": "سی", "d": "دی", "e": "ای", "f": "اِف", "g": "جی", "h": "اِچ", "i": "آی",
    "j": "جِی", "k": "کِی", "l": "اِل", "m": "اِم", "n": "اِن", "o": "اُ", "p": "پی", "q": "کیو", "r": "آر",
    "s": "اِس", "t": "تی", "u": "یو", "v": "وی", "w": "دابلیو", "x": "اِکس", "y": "وای", "z": "زِد",
}
#: Units written straight after a number, as in 8GB or 120Hz.
LATIN_UNITS = {
    "kb": "کیلوبایت", "mb": "مگابایت", "gb": "گیگابایت", "tb": "ترابایت", "hz": "هرتز", "khz": "کیلوهرتز",
    "mhz": "مگاهرتز", "ghz": "گیگاهرتز", "mp": "مگاپیکسل", "mah": "میلی‌آمپر ساعت", "w": "وات", "kw": "کیلووات",
    "g": "جی", "k": "کِی",
}
_LETTERS_THEN_DIGITS_RE = re.compile(r"(?<![A-Za-z])([A-Za-z]{1,3})(?=\d)")
_DIGITS_THEN_UNIT_RE = re.compile(r"(?<=\d)([A-Za-z]{1,3})(?![A-Za-z])")
_LATIN_DIGIT_SEAM_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def spell_latin(letters: str) -> str:
    return " ".join(LATIN_LETTER_NAMES[c] for c in letters.lower())


def _part_latin_from_digits(text: str) -> str:
    """M6 as اِم 6 and 8GB as 8 گیگابایت, so the number is read apart from the letters it touches.

    A run of one to three letters before a number is a model name and is
    spelled out; a known unit after one is named; any other letters touching a
    number are only parted from it.
    """
    text = _LETTERS_THEN_DIGITS_RE.sub(lambda m: spell_latin(m.group(1)) + " ", text)
    text = _DIGITS_THEN_UNIT_RE.sub(
        lambda m: " " + LATIN_UNITS[m.group(1).lower()] if m.group(1).lower() in LATIN_UNITS else m.group(1), text)
    return _LATIN_DIGIT_SEAM_RE.sub(" ", text)


def _expand_time(match: re.Match[str]) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    second = match.group(3)
    # Do not say "ساعت" twice when the writer already did.
    preceding = match.string[: match.start()].rstrip()
    prefix = "" if preceding.endswith("ساعت") else "ساعت "
    text = f"{prefix}{number_to_words(hour)}"
    if minute:
        text += f" و {number_to_words(minute)} دقیقه"
    if second and int(second):
        text += f" و {number_to_words(int(second))} ثانیه"
    return text


def _expand_date(match: re.Match[str]) -> str:
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    month_name = MONTHS_JALALI[month - 1] if 1 <= month <= 12 else number_to_words(month)
    return f"{number_to_words(day)} {month_name} {number_to_words(year)}"


def _expand_currency(match: re.Match[str]) -> str:
    if match.group(1):
        amount, unit = match.group(1), match.group(2)
    else:
        amount, unit = match.group(4), match.group(3)
    clean = amount.replace(",", "").replace("٬", "")
    unit_word = CURRENCIES.get(unit, unit)
    if "." in clean:
        integer_part, fraction_part = clean.split(".", 1)
        return f"{decimal_to_words(integer_part, fraction_part)} {unit_word}"
    return f"{number_to_words(int(clean))} {unit_word}"


def _expand_digit_run(match: re.Match[str]) -> str:
    return " ".join(number_to_words(int(digit)) for digit in match.group(1))


def expand_numbers(text: str) -> str:
    """Replace every numeric form in ``text`` with Persian words.

    Applied most-specific first so that e.g. ``۱۴۰۳/۰۵/۰۶`` becomes a date
    rather than three separate integers.
    """
    if not text:
        return text

    result = to_ascii_digits(text)
    if not any(char.isdigit() for char in result):
        return text

    try:
        result = _THOUSANDS_RE.sub("", result)
        result = _part_latin_from_digits(result)
        result = _CURRENCY_RE.sub(_expand_currency, result)
        result = _DATE_RE.sub(_expand_date, result)
        result = _TIME_RE.sub(_expand_time, result)
        result = _PERCENT_RE.sub(
            lambda m: f"{number_to_words(int(float(m.group(1))))} درصد"
            if float(m.group(1)).is_integer()
            else f"{decimal_to_words(*m.group(1).split('.'))} درصد",
            result,
        )
        result = _ORDINAL_RE.sub(lambda m: ordinal_to_words(int(m.group(1))), result)
        result = _RANGE_RE.sub(
            lambda m: f"{number_to_words(int(m.group(1)))} تا {number_to_words(int(m.group(2)))}", result
        )
        result = _DIGIT_RUN_RE.sub(_expand_digit_run, result)
        result = _DECIMAL_RE.sub(lambda m: decimal_to_words(m.group(1), m.group(2)), result)
        result = _INTEGER_RE.sub(lambda m: number_to_words(int(m.group(1))), result)
    except Exception:
        log.warning("Number expansion failed; leaving digits as written.", exc_info=True)
        return text

    return result


__all__ = [
    "decimal_to_words",
    "expand_numbers",
    "number_to_words",
    "ordinal_to_words",
    "to_ascii_digits",
    "to_persian_digits",
]
