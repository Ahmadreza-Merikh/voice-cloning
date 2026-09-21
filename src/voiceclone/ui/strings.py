"""Bilingual UI strings (English / Persian).

The interface chrome can be shown in either language; the *content* being
synthesised is independent of this choice.
"""

from __future__ import annotations

from typing import Any

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "title": "Persian Voice Cloning Studio",
        "subtitle": "High-fidelity zero-shot cloning with phoneme-accurate Persian pronunciation",
        # tabs
        "tab_tts": "Text to Speech",
        "tab_s2s": "Speech to Speech",
        "tab_voices": "Voice Library",
        "tab_diagnostics": "Diagnostics",
        # reference
        "reference": "Reference voice",
        "reference_help": "8-12 seconds of clean, continuous speech from one speaker clones best.",
        "reference_text": "Reference transcript",
        "reference_text_help": (
            "What the reference says. Left empty it is transcribed for you, before the engine "
            "loads. Correcting it word for word improves quality - the engine reads this text "
            "and the clip together."
        ),
        "transcript_is_window": (
            "This is the transcript of the {kept}s window kept from your {whole}s recording, starting at "
            "{start}s - that window is what the engine is given, so the text has to match it, not the whole file."
        ),
        "analyse": "Analyse reference",
        "quality_report": "Reference quality",
        # text
        "text": "Text to speak",
        "text_placeholder": "متن فارسی خود را اینجا بنویسید...",
        "source_audio": "Source audio",
        "source_audio_help": "The speech whose content (and, in conversion mode, delivery) you want.",
        # controls
        "engine": "Engine",
        "mode": "Conversion mode",
        "speed": "Speed",
        "seed": "Seed",
        "best_of": "Takes (best-of-N)",
        "best_of_help": (
            "Render this many takes and keep the best. Seeds vary more than any "
            "sampling setting does, so this is the cheapest quality win - at N times "
            "the wait."
        ),
        "best_of_cascade_help": (
            "Cascade mode only - it re-synthesises, so best-of-N applies. Seed-VC "
            "transforms the source directly and ignores this."
        ),
        "seed_help": "0 draws a new random seed. Reuse a seed to reproduce a take exactly.",
        "quality_preset": "Quality preset",
        "advanced": "Advanced settings",
        "preprocessing": "Audio preprocessing",
        "denoise": "Denoise",
        "separate_vocals": "Separate voice from music",
        "trim_silence": "Trim silence",
        "auto_window": "Auto-select best window",
        "use_phonemes": "Use Persian G2P (phonemes)",
        "use_phonemes_help": (
            "Sends IPA instead of Persian script. Measured worse on every engine here - "
            "MOSS's phoneme channel is English/pinyin-trained and collapses on Persian. "
            "Leave off unless you are testing a new engine."
        ),
        # actions
        "generate": "Generate",
        "generating": "Generating...",
        "stop": "Stop",
        "output": "Generated audio",
        "details": "Details",
        "save_voice": "Save to library",
        "voice_name": "Voice name",
        "delete": "Delete",
        "refresh": "Refresh",
        "saved_voice": "Saved voice",
        "use_saved": "Use a saved voice",
        # messages
        "no_reference": "Please provide a reference voice.",
        "no_text": "Please enter some text.",
        "no_source": "Please provide source audio.",
        "done": "Done",
        "error": "Error",
    },
    "fa": {
        "title": "استودیو شبیه‌سازی صدای فارسی",
        "subtitle": "شبیه‌سازی صدا با کیفیت بالا و تلفظ دقیق فارسی بر پایه واج",
        "tab_tts": "متن به گفتار",
        "tab_s2s": "گفتار به گفتار",
        "tab_voices": "کتابخانه صدا",
        "tab_diagnostics": "وضعیت سامانه",
        "reference": "صدای مرجع",
        "reference_help": "هشت تا دوازده ثانیه گفتار تمیز و پیوسته از یک گوینده بهترین نتیجه را می‌دهد.",
        "reference_text": "متن صدای مرجع",
        "reference_text_help": (
            "متن صدای مرجع. اگر خالی بماند، پیش از بارگذاری موتور خودکار نوشته می‌شود. "
            "اگر کلمه‌به‌کلمه درستش کنید کیفیت بهتر می‌شود؛ موتور این متن و صدا را با هم می‌خواند."
        ),
        "transcript_is_window": (
            "این متنِ همان {kept} ثانیه‌ای است که از فایل {whole} ثانیه‌ای شما نگه داشته شد (از ثانیهٔ {start}). "
            "موتور هم فقط همین تکه را می‌گیرد، پس متن باید با همین بخش بخواند، نه با کل فایل."
        ),
        "analyse": "تحلیل صدای مرجع",
        "quality_report": "کیفیت صدای مرجع",
        "text": "متن برای خواندن",
        "text_placeholder": "متن فارسی خود را اینجا بنویسید...",
        "source_audio": "فایل صوتی مبدأ",
        "source_audio_help": "گفتاری که محتوا (و در حالت تبدیل، لحن) آن را می‌خواهید.",
        "engine": "موتور",
        "mode": "حالت تبدیل",
        "speed": "سرعت",
        "seed": "دانه تصادفی",
        "best_of": "تعداد برداشت (بهترین از N)",
        "best_of_help": (
            "این تعداد برداشت تولید می‌شود و بهترین آن نگه داشته می‌شود. تفاوت بین "
            "دانه‌های تصادفی از هر تنظیم دیگری بیشتر است، پس ارزان‌ترین راه بهبود "
            "کیفیت است - با N برابر زمان."
        ),
        "best_of_cascade_help": (
            "فقط در حالت آبشاری - چون دوباره تولید می‌کند. Seed-VC مستقیم روی صدای "
            "مبدأ کار می‌کند و این مقدار را نادیده می‌گیرد."
        ),
        "seed_help": "صفر یعنی دانه تصادفی جدید. برای بازتولید دقیق، همان دانه را دوباره وارد کنید.",
        "quality_preset": "پیش‌تنظیم کیفیت",
        "advanced": "تنظیمات پیشرفته",
        "preprocessing": "پیش‌پردازش صدا",
        "denoise": "حذف نویز",
        "separate_vocals": "جداسازی صدا از موسیقی",
        "trim_silence": "حذف سکوت",
        "auto_window": "انتخاب خودکار بهترین بازه",
        "use_phonemes": "استفاده از واج‌نگار فارسی",
        "use_phonemes_help": (
            "به جای خط فارسی، IPA فرستاده می‌شود. در اندازه‌گیری‌ها روی همه موتورهای "
            "موجود بدتر بود؛ کانال واجی MOSS با انگلیسی و پینیین آموزش دیده و روی "
            "فارسی فرو می‌ریزد. جز برای آزمودن موتور تازه، خاموش بماند."
        ),
        "generate": "تولید صدا",
        "generating": "در حال تولید...",
        "stop": "توقف",
        "output": "صدای تولید شده",
        "details": "جزئیات",
        "save_voice": "ذخیره در کتابخانه",
        "voice_name": "نام صدا",
        "delete": "حذف",
        "refresh": "بازخوانی",
        "saved_voice": "صدای ذخیره‌شده",
        "use_saved": "استفاده از صدای ذخیره‌شده",
        "no_reference": "لطفاً یک صدای مرجع انتخاب کنید.",
        "no_text": "لطفاً متن را وارد کنید.",
        "no_source": "لطفاً فایل صوتی مبدأ را انتخاب کنید.",
        "done": "انجام شد",
        "error": "خطا",
    },
}

#: Ready-made quality presets exposed in the UI.
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "Maximum quality (slowest)": {
        "description": "64 ODE steps. For final renders where nothing matters but fidelity.",
        "nfe_step": 64,
        "cfg_strength": 2.0,
        "denoise": "auto",
        "diffusion_steps": 75,
    },
    "High quality (recommended)": {
        "description": "48 ODE steps. The project default - clearly better than stock, still practical.",
        "nfe_step": 48,
        "cfg_strength": 2.0,
        "denoise": "auto",
        "diffusion_steps": 50,
    },
    "Balanced": {
        "description": "32 ODE steps, the upstream default. Roughly twice as fast.",
        "nfe_step": 32,
        "cfg_strength": 2.0,
        "denoise": "auto",
        "diffusion_steps": 30,
    },
    "Fast preview": {
        "description": "16 ODE steps. For checking phrasing and pacing, not for delivery.",
        "nfe_step": 16,
        "cfg_strength": 1.8,
        "denoise": "none",
        "diffusion_steps": 15,
    },
}

DEFAULT_PRESET = "Maximum quality (slowest)"

SAMPLE_TEXTS: list[str] = [
    "سلام، حال شما چطور است؟ امیدوارم روز خوبی داشته باشید.",
    "گلِ زیبای باغ ما امروز شکفت و عطرش تمام خانه را پر کرد.",
    "او مردی بزرگ بود که در سال ۱۳۴۵ به دنیا آمد و ۷۸ سال زندگی کرد.",
    "هوش مصنوعی امروزه در پزشکی، آموزش و صنعت کاربردهای گسترده‌ای دارد.",
    "قیمت این کتاب ۲۵۰۰۰ تومان است و تخفیف ۱۵٪ هم دارد.",
]


def get_strings(language: str = "fa") -> dict[str, str]:
    """Strings for ``language``, falling back to English for missing keys."""
    base = dict(STRINGS["en"])
    base.update(STRINGS.get(language, {}))
    return base


__all__ = ["DEFAULT_PRESET", "QUALITY_PRESETS", "SAMPLE_TEXTS", "STRINGS", "get_strings"]
