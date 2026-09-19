#!/usr/bin/env python3
"""Write the sealed test sets' sentences, and prove none of them was trained on.

SOTA_ROADMAP.md section 3.3 asks for twelve texts per test speaker, two in each
of six categories - formal written, conversational, numbers/dates/prices,
proper nouns and borrowed words, homographs and ezafe, very short or long -
none of which may appear in any prompt, any development text or any training
text. These are written for this project; they come from no corpus.

The check: CER normalisation (letters only, Arabic letterforms folded, no
spaces or ZWNJ), then equality, or containment either way where the contained
text is at least 20 letters and half the length of the other - against every
clip's text in the adaptation split (training, valid and test alike: a test
reference's transcript must not be a sentence the model is asked to say),
every development corpus transcript and every benchmark sentence. Any hit stops
the build. The length floor is what separates a leaked sentence from a shared
word: Neyshekar has one-word prompts ("گزارش", "اصفهان", "دی") that any
sentence containing them would otherwise "contain". (The split builder's own
benchmark check has no such floor, so it dropped six one-word Neyshekar clips
from training - conservative, and left as drawn.)

Every test speaker renders all twelve, so the design stays paired.

    .venv/bin/python scripts/build_test_texts.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUT = REPO_ROOT / "eval" / "manifests" / "test_texts_v1.json"
SPLITS = REPO_ROOT / "eval" / "manifests" / "adaptation_splits_v1.jsonl.gz"

TEXTS: list[tuple[str, str, str]] = [
    ("t01", "formal", "کمیتهٔ علمی همایش پس از بررسی مقاله‌ها، فهرست نهایی سخنرانان را در پایگاه اینترنتی خود منتشر کرد."),
    ("t02", "formal", "بر اساس گزارش سازمان هواشناسی، بارش‌های پاییزی امسال در بیشتر استان‌ها کمتر از میانگین بلندمدت بوده است."),
    ("t03", "conversational", "ببین، اگه فردا یه کم زودتر بیای، با هم می‌ریم نون تازه می‌خریم."),
    ("t04", "conversational", "راستش حوصلهٔ سینما رو ندارم، بیا امشب تو خونه یه فیلم ببینیم."),
    ("t05", "numbers", "جلسهٔ بعدی روز سه‌شنبه ۱۴ آبان ۱۴۰۵ ساعت ۹:۳۰ صبح برگزار می‌شود."),
    ("t06", "numbers", "قیمت هر کیلو پنیر از ۱۸۵ هزار تومان به ۲۱۲ هزار و ۵۰۰ تومان رسید."),
    ("t07", "names_and_loanwords", "سارا برای دیدن نمایشگاه آثار فرشچیان از شیراز به اصفهان رفت."),
    ("t08", "names_and_loanwords", "او اپلیکیشن تازه‌ای روی موبایلش نصب کرد تا بلیت کنسرت را آنلاین بخرد."),
    ("t09", "homographs_and_ezafe", "پس از پایان مسابقهٔ کشتی، ورزشکاران با کشتی به جزیره برگشتند."),
    ("t10", "homographs_and_ezafe", "دختر همسایهٔ روبه‌رویی ما کتاب تازهٔ نویسندهٔ محبوبش را امانت گرفت."),
    ("t11", "very_short", "باران بند آمد."),
    ("t12", "long", "وقتی به ایستگاه قطار رسیدیم، دیدیم که قطار شبانه به دلیل برف سنگین و بسته شدن جاده‌های "
                    "کوهستانی دست‌کم سه ساعت تأخیر دارد و مسافران خسته در سالن انتظار نشسته‌اند و چای می‌نوشند."),
]


def main() -> int:
    from build_adaptation_splits import benchmark_sentences, development

    from voiceclone.evaluation.similarity import normalize_for_cer

    texts = {}
    with gzip.open(SPLITS, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.setdefault(normalize_for_cer(record["text"]), f"{record['split']} clip {record['clip']}")
    _, dev_texts = development(REPO_ROOT / "eval" / "corpus" / "manifest.json")
    for text in dev_texts:
        texts.setdefault(normalize_for_cer(text), "development corpus")
    for key, sentence in benchmark_sentences().items():
        texts.setdefault(normalize_for_cer(sentence), f"benchmark sentence {key}")

    def overlaps(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        short, long = sorted((a, b), key=len)
        return short in long and len(short) >= max(20, len(long) // 2)

    hits = []
    for key, _, sentence in TEXTS:
        own = normalize_for_cer(sentence)
        for other, where in texts.items():
            if overlaps(own, other):
                hits.append(f"{key} overlaps {where}")
    if hits:
        raise SystemExit("\n".join(hits))

    words = {key: len(sentence.split()) for key, _, sentence in TEXTS}
    assert words["t11"] < 5 and words["t12"] > 25, words
    manifest = {
        "version": 1,
        "purpose": "the sentences every sealed test speaker renders (SOTA_ROADMAP section 3.3)",
        "written_for_this_project": True,
        "texts": [{"key": k, "category": c, "text": s, "words": words[k]} for k, c, s in TEXTS],
        "checked_against": {
            "adaptation_split_texts": SPLITS.name,
            "distinct_texts_checked": len(texts),
            "rule": "CER normalisation, then equality, or containment either way of at least 20 letters "
                    "and half the longer text; none found",
        },
    }
    manifest["sha256_of_texts"] = hashlib.sha256(
        json.dumps(manifest["texts"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(TEXTS)} texts, none in {len(texts)} checked texts; written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
