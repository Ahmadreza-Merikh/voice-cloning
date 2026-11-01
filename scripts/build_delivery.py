#!/usr/bin/env python3
"""Build the zip handed to the customer: the app alone, comments stripped, the adapted weights inside.

What goes in is the app's runtime and nothing of the research around it:
``app.py``, ``src/voiceclone`` without the experimental engines and the
benchmark recipe, the OmniVoice worker, a configuration without their
sections, a Persian README, the requirements, and the E5 weights under
``models/omnivoice-fa``. Every Python file loses its comments and docstrings,
and the build stops unless each file's syntax tree, docstrings aside, is the
same before and after.

    .venv/bin/python scripts/build_delivery.py --out ~/Desktop/persian-voice-cloning.zip
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NAME = "persian-voice-cloning"
WEIGHTS = Path("~/.cache/persian-voice-cloning/omnivoice/adapted/omnivoice-fa-ezafe-5000-float16").expanduser()
WEIGHTS_SHA256 = "1ecbaa58e2cf3efb753714818c974f1027b61c1564e54b489fc5f23af3595a4e"
TOKENIZER_SHA256 = "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d"
MODEL_DIR = "models/omnivoice-fa"
G2P_SOURCE = Path("~/Desktop/G2P Research").expanduser()
HOMO = "third_party/Homo-GE2PE-Persian"
G2P_FILES = [f"{HOMO}/{name}" for name in (
    "GE2PE.py", "NOTICE.md", "LICENSES/LICENSE.txt", "LICENSES/apache-2.0.md",
    "homo-ge2pe/config.json", "homo-ge2pe/generation_config.json", "homo-ge2pe/model.safetensors",
    "homo-ge2pe/added_tokens.json", "homo-ge2pe/special_tokens_map.json", "homo-ge2pe/tokenizer_config.json",
    "Parsivar/normalizer.py", "Parsivar/tokenizer.py", "Parsivar/data_helper.py", "Parsivar/token_merger.py",
    "Parsivar/resource/normalizer/Dic1_new.txt", "Parsivar/resource/normalizer/Dic2_new.txt",
    "Parsivar/resource/normalizer/Dic3_new.txt", "Parsivar/resource/normalizer/N_cctt.txt",
    "Parsivar/resource/normalizer/Normalizer_WrongCompound.txt", "Parsivar/resource/tokenizer/enDict",
    "Parsivar/resource/tokenizer/faDict", "g2p_resources/list_homograph.pkl",
    "g2p_resources/look_up_t_farsi_extra_of_ariana_b_punch_EN_extra_of_nevisa.pkl")]
G2P_RULES = ("EZAFE_TOKEN", "_EZAFE_BLOCK_NEXT", "_EZAFE_HEAD_CANDIDATES", "_NO_EZAFE_BIGRAMS", "_NO_EZAFE_HOST",
             "_ends_with_heh_ezafe_host", "_strip_diacritics", "_word_has_kasra_ezafe")
PARSIVAR_LICENSE = """\
The MIT License (MIT)

Copyright (c) 2013 BeRO

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
SHIPPED = ["app.py", "src", "scripts/omnivoice_worker.py", "configs/default.yaml"]
DROPPED = [f"src/voiceclone/engines/{name}.py" for name in (
    "chatterbox_tts", "fish_tts", "higgs_tts", "longcat_tts", "moss15_tts", "moss_tts", "omnivoice_ipa",
    "pocket_tts", "vc_seedvc")] + ["src/voiceclone/evaluation/recipe.py"]
DROPPED_CONFIGS = ("MossConfig", "ChatterboxConfig", "OmniVoiceIPAConfig", "FishS2Config", "HiggsConfig",
                   "Moss15Config", "SeedVCConfig", "PocketConfig", "LongCatConfig")
DROPPED_SECTIONS = ("moss", "chatterbox", "omnivoice_ipa", "fish", "higgs", "moss15", "pocket", "longcat", "seedvc")
AI_SIGNS = re.compile(r"claude|anthropic|chatgpt|copilot|codex|\bllm\b|findings|sota_|eval/runs|kaggle|marsmanam|"
                      r"ahmadrezamerikh|@gmail|§|\bmana\b|listener", re.I)
LEFTOVERS = re.compile(r"seed-?vc|\bmoss|chatterbox|higgs|longcat|pocket_tts|fish_tts|omnivoice_ipa", re.I)

ENGINES_INIT = '''\
from .asr import ASREngine, Transcript
from .base import (
    Engine,
    EngineCapabilities,
    SynthesisResult,
    TTSEngine,
    VCEngine,
    engine_status,
    get_tts_engine,
    get_vc_engine,
    list_tts_engines,
    list_vc_engines,
    register_tts,
    register_vc,
)
from .f5_tts import F5TTSEngine
from .omnivoice_tts import OmniVoiceEngine

__all__ = [
    "ASREngine",
    "Engine",
    "EngineCapabilities",
    "F5TTSEngine",
    "OmniVoiceEngine",
    "SynthesisResult",
    "TTSEngine",
    "Transcript",
    "VCEngine",
    "engine_status",
    "get_tts_engine",
    "get_vc_engine",
    "list_tts_engines",
    "list_vc_engines",
    "register_tts",
    "register_vc",
]
'''

EDITS: dict[str, list[tuple[str, str]]] = {
    "scripts/omnivoice_worker.py": [
        ('argparse.ArgumentParser(description=__doc__.split("\\n")[0])', 'argparse.ArgumentParser()'),
    ],
    "app.py": [
        ('help="Default TTS engine (f5 / moss)"', 'help="Default TTS engine (omnivoice / f5)"'),
    ],
    "src/voiceclone/config.py": [
        ('apply_to_engines: list[str] = field(default_factory=lambda: ["omnivoice_ipa"])',
         "apply_to_engines: list[str] = field(default_factory=list)"),
        ('adapted_dir: str = "~/.cache/persian-voice-cloning/omnivoice/adapted/omnivoice-fa-ezafe-5000-float16"',
         f'adapted_dir: str = "{MODEL_DIR}"'),
    ],
    "src/voiceclone/engines/omnivoice_tts.py": [
        ('    "Close some apps - browsers, other AI apps - and try again, or pick F5-TTS. On this 16 GB Mac "\n'
         '    "with no swap, OmniVoice loading beside other open apps reached warning pressure (2026-09-19)."\n',
         '    "Close some other apps and try again, or pick F5-TTS."\n'),
        ('FETCH_HINT = "Fetch them with: .venv/bin/python scripts/fetch_adapted_weights.py (about 2 GB)."',
         f'FETCH_HINT = "The adapted weights belong in {MODEL_DIR} inside the project folder."'),
        ("    folder = Path(os.path.expandvars(str(settings.adapted_dir))).expanduser()\n",
         "    folder = Path(os.path.expandvars(str(settings.adapted_dir))).expanduser()\n"
         "    if not folder.is_absolute():\n"
         "        folder = Path(__file__).resolve().parents[3] / folder\n"),
        ('    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)\n',
         '    candidates.extend(f"{path}/bin/python" for path in _VENV_CANDIDATES)\n'
         '    candidates.extend(f"{path}/Scripts/python.exe" for path in _VENV_CANDIDATES)\n'),
        ("f\"the ezafe-marked adaptation's step-5,000 export, sha256 {settings.adapted_sha256}\"",
         'f"{settings.adapted_dir}, sha256 {settings.adapted_sha256}"'),
    ],
    "src/voiceclone/engines/f5_tts.py": [
        ('            "Using the multilingual base checkpoint. Persian pronunciation will be noticeably worse - "\n'
         '            "consider the MOSS engine instead."\n',
         '            "Using the multilingual base checkpoint. Persian pronunciation will be noticeably worse."\n'),
    ],
    "src/voiceclone/text/g2p.py": [
        ('_REPO_CANDIDATES = (\n'
         '    "~/Desktop/G2P Research",\n'
         '    "~/G2P Research",\n'
         '    "~/Documents/G2P Research",\n'
         '    "~/projects/G2P Research",\n'
         '    "~/Desktop/g2p-research",\n'
         '    "~/g2p-research",\n'
         ')\n',
         '_REPO_CANDIDATES = (str(Path(__file__).resolve().parents[3] / "g2p"),)\n'),
    ],
    "src/voiceclone/text/ipa_normalize.py": [
        ('CANONICAL_ENGINES: frozenset[str] = frozenset({"omnivoice_ipa"})', "CANONICAL_ENGINES: frozenset[str] = frozenset()"),
    ],
    "src/voiceclone/pipelines/studio.py": [
        ('hint="Close some apps - browsers, other AI apps - and try again once it is back to normal.",',
         'hint="Close some other apps and try again once it is back to normal.",'),
    ],
    "src/voiceclone/pipelines/results.py": [
        ('"as close as another recording of the same speaker on this corpus"',
         '"as close as another recording of the same speaker"'),
        ('"above every different-speaker pair measured, below same-speaker"',
         '"closer than different speakers usually are"'),
        ('"as close to the prompt as a working engine gets here"', '"high"'),
        ('"weak - the render may have come apart"', '"low - try another seed or a cleaner reference"'),
        ('"inside the range different speakers occupy - not distinguishable"', '"about as close as a different speaker"'),
        ("            if self.held_out_similarity is None:\n"
         '                label += " (shares its channel; not identity transfer)"\n', ""),
    ],
    "src/voiceclone/models.py": [
        ('hint="Run  python scripts/download_models.py  while online.",',
         'hint="Connect to the internet once so it can be downloaded.",'),
    ],
    "src/voiceclone/cli.py": [
        ('choices=["cascade", "seedvc"]', 'choices=["cascade"]'),
        ('text_cmd.add_argument("-e", "--engine", default="moss", help="Engine the text is destined for")',
         'text_cmd.add_argument("-e", "--engine", default="omnivoice", help="Engine the text is destined for")'),
        ('            help=(\n'
         '                "Other recordings of the same speaker, not used as the prompt. "\n'
         '                "Scoring against these is the only similarity figure that measures "\n'
         '                "identity rather than channel."\n'
         '            ),\n',
         '            help="Other recordings of the same speaker, used only for scoring.",\n'),
    ],
    "src/voiceclone/ui/app.py": [
        ('    ("Voice conversion (Seed-VC) - keeps original delivery", "seedvc"),\n', ""),
        ('        if engine == "seedvc":\n'
         '            return {"diffusion_steps": preset["diffusion_steps"]} if "diffusion_steps" in preset else {}\n',
         ""),
        ("                    tts_phonemes = gr.Checkbox(\n"
         "                        value=False,\n",
         "                    tts_phonemes = gr.Checkbox(\n"
         "                        value=False,\n"
         "                        visible=False,\n"),
        ("                            s2s_mode = gr.Radio(\n"
         "                                choices=CONVERSION_MODES,\n"
         '                                value="cascade",\n',
         "                            s2s_mode = gr.Radio(\n"
         "                                choices=CONVERSION_MODES,\n"
         '                                value="cascade",\n'
         "                                visible=False,\n"),
        ('gr.Checkbox(value=False, label=strings["use_phonemes"])',
         'gr.Checkbox(value=False, visible=False, label=strings["use_phonemes"])'),
    ],
    "src/voiceclone/ui/strings.py": [
        ('"subtitle": "High-fidelity zero-shot cloning with phoneme-accurate Persian pronunciation",',
         '"subtitle": "Voice cloning with accurate Persian pronunciation",'),
        ('"subtitle": "شبیه‌سازی صدا با کیفیت بالا و تلفظ دقیق فارسی بر پایه واج",',
         '"subtitle": "شبیه‌سازی صدا با تلفظ درست فارسی",'),
        ('        "best_of_cascade_help": (\n'
         '            "Cascade mode only - it re-synthesises, so best-of-N applies. Seed-VC "\n'
         '            "transforms the source directly and ignores this."\n'
         '        ),\n',
         '        "best_of_cascade_help": "Render this many takes of the re-spoken audio and keep the best.",\n'),
        ('        "best_of_cascade_help": (\n'
         '            "فقط در حالت آبشاری - چون دوباره تولید می‌کند. Seed-VC مستقیم روی صدای "\n'
         '            "مبدأ کار می‌کند و این مقدار را نادیده می‌گیرد."\n'
         '        ),\n',
         '        "best_of_cascade_help": "این تعداد برداشت از گفتار بازسازی‌شده تولید می‌شود و بهترینش نگه داشته '
         'می‌شود.",\n'),
        ('        "use_phonemes_help": (\n'
         '            "Sends IPA instead of Persian script. Measured worse on every engine here - "\n'
         '            "MOSS\'s phoneme channel is English/pinyin-trained and collapses on Persian. "\n'
         '            "Leave off unless you are testing a new engine."\n'
         '        ),\n',
         '        "use_phonemes_help": "Sends IPA instead of Persian script.",\n'),
        ('        "use_phonemes_help": (\n'
         '            "به جای خط فارسی، IPA فرستاده می‌شود. در اندازه‌گیری‌ها روی همه موتورهای "\n'
         '            "موجود بدتر بود؛ کانال واجی MOSS با انگلیسی و پینیین آموزش دیده و روی "\n'
         '            "فارسی فرو می‌ریزد. جز برای آزمودن موتور تازه، خاموش بماند."\n'
         '        ),\n',
         '        "use_phonemes_help": "به جای خط فارسی، IPA فرستاده می‌شود.",\n'),
        ('"description": "48 ODE steps. The project default - clearly better than stock, still practical.",',
         '"description": "48 ODE steps. A good balance of quality and speed.",'),
        ('        "diffusion_steps": 75,\n', ""),
        ('        "diffusion_steps": 50,\n', ""),
        ('        "diffusion_steps": 30,\n', ""),
        ('        "diffusion_steps": 15,\n', ""),
    ],
}

REQUIREMENTS = """\
torch>=2.2.0
torchaudio>=2.2.0
numpy>=1.26.0,<3.0
scipy>=1.11.0
soundfile>=0.12.1
librosa>=0.10.1
pyloudnorm>=0.1.1
noisereduce>=3.0.0
pydub>=0.25.1
av>=11.0.0
num2words>=0.5.13
regex>=2023.10.3
python-bidi>=0.4.2
f5-tts>=1.1.0
faster-whisper>=1.0.3
huggingface-hub>=0.24.0
safetensors>=0.4.3
transformers>=4.40.0
msgpack>=1.0.0
nltk>=3.9
Unidecode>=1.3
silero-vad>=5.1
onnxruntime>=1.17.0
speechbrain>=1.0.0
gradio>=5.0.0
PyYAML>=6.0.1
tqdm>=4.66.0
rich>=13.7.0
"""

PYPROJECT = """\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "persian-voice-cloning"
version = "1.0.0"
description = "Persian voice cloning"
readme = "README.md"
requires-python = ">=3.10,<3.13"
authors = [{ name = "Ahmadreza" }]
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[project.scripts]
voiceclone = "voiceclone.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
"""

GITIGNORE = """\
.venv/
__pycache__/
*.pyc
outputs/
voices/
.DS_Store
"""

README = """\
# شبیه‌سازی صدای فارسی

این برنامه با چند ثانیه صدای یک نفر، هر متن فارسی را با همان صدا می‌خواند. متن قبل از رسیدن به مدل مرتب می‌شود: \
عددها و تاریخ‌ها به حروف درمی‌آیند، حرف‌های عربی یکدست می‌شوند، کسره‌ی اضافه نوشته می‌شود و روی کلمه‌های \
هم‌نگاشت مثل «گل» و «مهر» حرکت گذاشته می‌شود تا مدل درست بخواند.

موتور اصلی OmniVoice است که روی حدود ۱۲۸ ساعت گفتار فارسی (Common Voice و نی‌شکر) دوباره آموزش داده شده. \
وزن‌هایش در پوشه‌ی `models/omnivoice-fa` است. اگر OmniVoice نصب نباشد، برنامه با F5-TTS کار می‌کند.

## پیش‌نیازها

- Python 3.11 (نسخه‌های 3.10 تا 3.12 هم کار می‌کنند)
- ffmpeg
- حدود ۱۵ گیگابایت فضای خالی برای کتابخانه‌ها و مدل‌هایی که بار اول دانلود می‌شوند
- حداقل ۱۶ گیگابایت رم. کارت گرافیک NVIDIA لازم نیست ولی سرعت را خیلی بالا می‌برد. روی مک‌های Apple Silicon هم اجرا می‌شود.
- اینترنت برای اجرای اول

## نصب

macOS و لینوکس:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

ویندوز:

```bat
py -3.11 -m venv .venv
.venv\\Scripts\\activate
pip install -e .
```

### محیط OmniVoice

OmniVoice نسخه‌ی torch و transformers خودش را می‌خواهد، برای همین در یک محیط جدا نصب می‌شود:

```bash
pip install uv
uv venv --python 3.11 ~/.cache/persian-voice-cloning/omnivoice/.venv
VIRTUAL_ENV=~/.cache/persian-voice-cloning/omnivoice/.venv uv pip install omnivoice==0.2.1
```

در ویندوز (PowerShell):

```powershell
pip install uv
uv venv --python 3.11 $HOME\\.cache\\persian-voice-cloning\\omnivoice\\.venv
$env:VIRTUAL_ENV = "$HOME\\.cache\\persian-voice-cloning\\omnivoice\\.venv"
uv pip install omnivoice==0.2.1
```

اگر این محیط را جای دیگری ساختید، مسیر پایتونش را در متغیر `VOICECLONE_OMNIVOICE_PYTHON` بگذارید.

## اجرا

```bash
python app.py
```

بعد در مرورگر به `http://127.0.0.1:7860` بروید. با `--port` می‌شود پورت را عوض کرد و با `--language en` رابط انگلیسی می‌شود.

## استفاده

۱. در تب «متن به گفتار» یک فایل صدای مرجع بدهید: هشت تا دوازده ثانیه، تمیز و فقط از یک گوینده.

۲. متن دقیق همان فایل مرجع را بنویسید. اگر خالی بماند خودکار نوشته می‌شود، ولی وقتی دستی درستش کنید نتیجه بهتر است. \
روی سیستم‌های ۱۶ گیگابایتی بهتر است همیشه دستی بنویسید.

۳. متن فارسی را وارد کنید و «تولید صدا» را بزنید.

بار اول مدل بارگذاری می‌شود و کمی بیشتر طول می‌کشد. صداهایی را که زیاد استفاده می‌کنید می‌توانید در «کتابخانه صدا» ذخیره کنید.

از خط فرمان هم می‌شود:

```bash
voiceclone tts "سلام، حال شما چطور است؟" -r ref.wav --reference-text "متن فایل مرجع" -o out.wav
```

خروجی‌ها در پوشه‌ی `outputs` ذخیره می‌شوند.

## اگر کار نکرد

- تب «وضعیت سامانه» نشان می‌دهد چه چیزهایی نصب و آماده است.
- اگر در خروجی هشدار آمد که با وزن‌های اصلی OmniVoice خوانده شده، پوشه‌ی `models/omnivoice-fa` ناقص است.
- اگر به جای OmniVoice با F5 خوانده شد، محیط OmniVoice پیدا نشده است.

## مجوزها

این برنامه از مدل‌های دیگران استفاده می‌کند و هر کدام مجوز خودش را دارد:

- OmniVoice (`k2-fsa/OmniVoice`): کد با Apache 2.0 و وزن‌های از پیش آموزش‌دیده با CC-BY-NC، یعنی فقط برای استفاده‌ی \
غیرتجاری. وزن‌های `models/omnivoice-fa` از روی همین مدل ساخته شده‌اند و همین محدودیت را دارند.
- کدک صوتی داخل `models/omnivoice-fa/audio_tokenizer` از Boson AI است و مجوز Boson Higgs Audio 2 Community License \
را دارد. متن کامل مجوز در همان پوشه است.
- F5-TTS: وزن‌های پایه با CC-BY-NC-4.0. نسخه‌ی فارسی از `Lumos675/F5_TTS_Persian` گرفته می‌شود.
- Whisper large-v3-turbo برای نوشتن متن صدای مرجع: MIT.
- ECAPA از SpeechBrain برای سنجش شباهت صدا: Apache 2.0.
- برچسب‌گذار ParsBERT از hazm (`roshan-research/hazm-parsbert-postagger`) برای تشخیص کسره‌ی اضافه.
- Homo-GE2PE (`MahtaFetrat/Homo-GE2PE-Persian`) برای خواندن کلمه‌های هم‌نگاشت: MIT، ساخته‌شده روی \
`google/byt5-small` با Apache 2.0. فایل‌هایش در پوشه‌ی `g2p` است.
- Parsivar برای یکدست کردن متن ورودی Homo-GE2PE: MIT.
- داده‌های آموزش فارسی: Common Voice و نی‌شکر، هر دو با CC0.
"""


def docstring_targets(tree: ast.AST) -> list[tuple[ast.AST, ast.Expr]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.append((node, first))
    return found


def shape_without_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node, _ in docstring_targets(tree):
        node.body = node.body[1:] or ([] if isinstance(node, ast.Module) else [ast.Pass()])
    return ast.dump(tree)


def drop_docstrings(source: str) -> str:
    lines = source.splitlines(keepends=True)
    spans = [(first.lineno, first.col_offset, first.end_lineno, first.end_col_offset,
              len(node.body) == 1 and not isinstance(node, ast.Module))
             for node, first in docstring_targets(ast.parse(source))]
    for lineno, col, end_lineno, end_col, needs_pass in sorted(spans, reverse=True):
        prefix = lines[lineno - 1].encode()[:col].decode()
        rest = lines[end_lineno - 1].encode()[end_col:].decode()
        if rest.strip() and not rest.strip().startswith("#"):
            raise SystemExit(f"a docstring shares its line with code: line {lineno}")
        if not prefix.strip() and not needs_pass:
            del lines[lineno - 1:end_lineno]
        else:
            lines[lineno - 1:end_lineno] = [prefix + ("pass" if needs_pass else "") + "\n"]
    return "".join(lines)


def drop_comments(source: str) -> str:
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = source.splitlines(keepends=True)
    inside_strings = set()
    for token in tokens:
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            inside_strings.update(range(token.start[0], token.end[0] + 1))
    gone = set()
    for token in tokens:
        if token.type != tokenize.COMMENT or (token.start[0] == 1 and token.string.startswith("#!")):
            continue
        row, col = token.start
        before = lines[row - 1][:col].rstrip()
        if before:
            lines[row - 1] = before + "\n"
        else:
            gone.add(row)
    kept, blank_run, opener = [], 0, False
    for number, line in enumerate(lines, start=1):
        if number in gone:
            continue
        if number in inside_strings:
            kept.append(line)
            blank_run, opener = 0, False
            continue
        if not line.strip():
            blank_run += 1
            if opener or blank_run > 2:
                continue
            kept.append("\n")
            continue
        blank_run = 0
        opener = line.rstrip().endswith(":") and not line.lstrip().startswith(("#", "@"))
        kept.append(line.rstrip() + "\n")
    while kept and not kept[-1].strip():
        kept.pop()
    return "".join(kept)


def drop_classes(source: str, names: tuple[str, ...]) -> str:
    lines = source.splitlines(keepends=True)
    spans = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            spans.append((min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno))
    if len(spans) != len(names):
        raise SystemExit(f"config.py: found {len(spans)} of the {len(names)} engine settings to drop")
    for start, end in sorted(spans, reverse=True):
        del lines[start - 1:end]
    text = "".join(lines)
    fields = re.compile(r"^    (" + "|".join(DROPPED_SECTIONS) + r"): \w+ = field\(default_factory=\w+\)\n", re.M)
    text, count = fields.subn("", text)
    if count != len(DROPPED_SECTIONS):
        raise SystemExit(f"config.py: dropped {count} of the {len(DROPPED_SECTIONS)} engine fields")
    return text


def edit(stage: Path) -> None:
    for name in DROPPED:
        (stage / name).unlink()
    (stage / "src/voiceclone/engines/__init__.py").write_text(ENGINES_INIT, encoding="utf-8")
    config = stage / "src/voiceclone/config.py"
    config.write_text(drop_classes(config.read_text(encoding="utf-8"), DROPPED_CONFIGS), encoding="utf-8")
    for name, pairs in EDITS.items():
        path = stage / name
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if text.count(old) != 1:
                raise SystemExit(f"{name}: expected one match of {old[:70]!r}, found {text.count(old)}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def settings(stage: Path) -> None:
    path = stage / "configs/default.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in DROPPED_SECTIONS:
        data["engines"].pop(section, None)
    data["engines"]["omnivoice"]["adapted_dir"] = MODEL_DIR
    data["engines"]["omnivoice"]["adapted_sha256"] = WEIGHTS_SHA256
    data["text"]["g2p"]["apply_to_engines"] = []
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8")


def strip_python(stage: Path) -> int:
    files = sorted(p for p in stage.rglob("*.py") if "third_party" not in p.relative_to(stage).parts)
    shapes = {path: shape_without_docstrings(path.read_text(encoding="utf-8")) for path in files}
    for path in files:
        path.write_text(drop_comments(drop_docstrings(path.read_text(encoding="utf-8"))), encoding="utf-8")
    ruff = [str(ROOT / ".venv/bin/ruff")]
    subprocess.run(ruff + ["format", "--no-cache", "--quiet", "--line-length", "100", *map(str, files)], check=True)
    for path in files:
        source = path.read_text(encoding="utf-8")
        if ast.dump(ast.parse(source)) != shapes[path]:
            raise SystemExit(f"{path.relative_to(stage)}: the code changed, not only its comments")
        if "__doc__" in source:
            raise SystemExit(f"{path.relative_to(stage)} reads __doc__, which is gone")
    check = subprocess.run(ruff + ["check", "--no-cache", "--select", "F821,F822,F823,E9", "--line-length", "100",
                                   *map(str, files)],
                           capture_output=True, text=True)
    if check.returncode:
        raise SystemExit(check.stdout + check.stderr)
    return len(files)


def rules_slice(source: str) -> str:
    tree = ast.parse(source)
    nodes = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nodes[node.targets[0].id] = node
    wanted, todo = set(), list(G2P_RULES)
    while todo:
        name = todo.pop()
        if name in wanted:
            continue
        if name not in nodes:
            raise SystemExit(f"dataset.py no longer defines {name}")
        wanted.add(name)
        todo += [n.id for n in ast.walk(nodes[name]) if isinstance(n, ast.Name) and n.id in nodes]
    lines = source.splitlines()
    picked = sorted((nodes[name].lineno, nodes[name].end_lineno) for name in wanted)
    return "import re\n\n" + "\n".join("\n".join(lines[a - 1:b]) for a, b in picked) + "\n"


def vendor_g2p(stage: Path) -> None:
    target = stage / "g2p"
    for name in G2P_FILES:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(G2P_SOURCE / name, target / name)
    (target / HOMO / "Parsivar/__init__.py").write_text("", encoding="utf-8")
    (target / HOMO / "Parsivar/LICENSE.txt").write_text(PARSIVAR_LICENSE, encoding="utf-8")
    backend = (G2P_SOURCE / "scripts/homo_ge2pe_backend.py").read_text(encoding="utf-8")
    if backend.count("from dataset import (") != 1:
        raise SystemExit("homo_ge2pe_backend.py no longer imports its rules from dataset.py")
    (target / "scripts").mkdir(parents=True, exist_ok=True)
    (target / "scripts/homo_ge2pe_backend.py").write_text(
        backend.replace("from dataset import (", "from ezafe_rules import ("), encoding="utf-8")
    (target / "ezafe_rules.py").write_text(rules_slice((G2P_SOURCE / "dataset.py").read_text(encoding="utf-8")),
                                           encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights(stage: Path) -> None:
    target = stage / MODEL_DIR
    skip = {"SOURCE.json", "audio_tokenizer/.gitattributes", "audio_tokenizer/README.md"}
    for path in sorted(WEIGHTS.rglob("*")):
        name = path.relative_to(WEIGHTS).as_posix()
        if path.is_file() and name not in skip:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / name)
    model = target / "model.safetensors"
    if sha256(model) != WEIGHTS_SHA256 or sha256(target / "audio_tokenizer/model.safetensors") != TOKENIZER_SHA256:
        raise SystemExit("the copied weights do not match their sha256")
    record = {"model_sha256": WEIGHTS_SHA256, "model_bytes": model.stat().st_size, "verified": True}
    (target / "SOURCE.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


def scan(stage: Path) -> list[str]:
    hits = []
    for path in sorted(stage.rglob("*")):
        parts = path.relative_to(stage).parts
        if not path.is_file() or parts[0] == "models" or "third_party" in parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if AI_SIGNS.search(line) or (path.name != "README.md" and LEFTOVERS.search(line)):
                hits.append(f"{path.relative_to(stage)}:{number}: {line.strip()[:120]}")
    return hits


def pack(stage: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in stage.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(out, "w") as archive:
        for path in files:
            method = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            archive.write(path, f"{NAME}/{path.relative_to(stage).as_posix()}", compress_type=method)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the customer delivery zip.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", type=Path, help="also leave the unpacked tree here")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="delivery-") as temp:
        stage = Path(temp) / NAME
        stage.mkdir()
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD", *SHIPPED], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(stage)], input=archive, check=True)
        edit(stage)
        settings(stage)
        vendor_g2p(stage)
        count = strip_python(stage)
        (stage / "README.md").write_text(README, encoding="utf-8")
        (stage / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
        (stage / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
        (stage / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        hits = scan(stage)
        if hits:
            raise SystemExit("left in the delivery:\n" + "\n".join(hits))
        weights(stage)
        files = pack(stage, args.out.expanduser())
        if args.keep:
            shutil.copytree(stage, args.keep.expanduser() / NAME)
    print(f"{count} Python files stripped and checked; {files} files in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
