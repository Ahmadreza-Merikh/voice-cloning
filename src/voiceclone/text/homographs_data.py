"""The homographs whose first short vowel the app writes, from HomoRich.

HomoRich-G2P-Persian (huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian
@ 21b189a7, data/train-01.parquet, CC0-1.0) lists 286 homograph words with
their readings. These are the 190 whose readings differ in the first short
vowel (eval/manifests/homorich_homographs_v1.json, first_vowel_tells_apart),
less three:

* the preposition به: in step 3 every mark on it was heard wrong and it was
  never wrong unmarked (FINDINGS section 17)
* اپل: in the user's news text (2026-10-25) Homo-GE2PE read the company as the
  epaulette, epol, and the app wrote اِپل, which OmniVoice said as epel
* کند: the verb (konad) and the adjective (kond) share their first vowel, so
  the one mark the app can write, کُند, is how the adjective is spelled; the
  user heard عرضه کند said as kond
"""

FIRST_VOWEL_HOMOGRAPHS = frozenset({
    "ابی", "اجر", "ادبا", "ارایه", "اره", "اسکی", "اشراف", "اشغال", "اشکال", "اعمال", "المان", "امل", "امین",
    "اه", "اوا", "ببر", "بخار", "بخور", "بده", "بدو", "بر", "بردار", "برده", "برس", "برم", "برند",
    "برنده", "بره", "بری", "برید", "بعد", "بکن", "بین", "ترک", "ترکان", "تست", "تن", "تند", "تنگ", "تکه",
    "جدا", "جرم", "جست", "جلد", "جنب", "جنگ", "جو", "جوید", "حسن", "حقه", "خرد", "خرم", "خفت", "خلف", "خلق",
    "خیر", "درهم", "دم", "ده", "دود", "دوران", "دوم", "دین", "رب", "رس", "رسم", "رسی", "رم", "رو", "رود",
    "روم", "روی", "رویه", "زهره", "سحر", "سر", "سرم", "سرو", "سرور", "سمبل", "سمت", "سنت", "سند", "سنی",
    "سیر", "شبه", "شدید", "شرف", "شست", "شش", "شل", "شما", "شنود", "شهرت", "شهره", "شو", "شوم", "شوید",
    "شکر", "صرف", "صفر", "طبق", "عرضه", "عرق", "عقبی", "علم", "علی", "فراری", "فرانک", "فرق", "فک", "قسم",
    "قطر", "قطعه", "قل", "قمری", "قوت", "لب", "لغز", "لنگ", "لی‌لی", "مبلغ", "مثل", "مجاز", "محرم", "مد",
    "مراجع", "مرد", "مردم", "مسکن", "مصر", "معبر", "مفصل", "مقدم", "مقطع", "ملا", "ملاک", "مهر", "میدانی",
    "نسبی", "نشست", "نشسته", "نصاب", "نقل", "نمونه", "هری", "هل", "هلیم", "ور", "وسطی", "ولو", "پر", "پرس",
    "پرش", "پست", "پلاس", "پهن", "پی", "چشم", "چک", "کت", "کر", "کرات", "کرد", "کرم", "کره", "کش", "کشت",
    "کشتی", "کشند", "کشیم", "کل", "کن", "کنده", "کندی", "کنف", "کنه", "کیف", "گرد", "گردان", "گرده",
    "گرم", "گزید", "گل", "گله", "گنج", "گنگ", "یمن",
})
