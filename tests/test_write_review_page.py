"""Review pages are written from one template whose better-reading question takes more than one pick."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("write_review_page", ROOT / "scripts" / "write_review_page.py")
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)

ITEMS = [{"id": f"s{i}", "text": "باران آمد.", "words": ["باران", "آمد"],
          "slots": [{"slot": n, "audio": f"audio/{i}{n}.wav"} for n in (1, 2, 3)]} for i in range(4)]


def test_the_page_is_filled_in_and_takes_several_picks(tmp_path: Path) -> None:
    folder, evidence = tmp_path / "listening" / "x-review-v1", tmp_path / "evidence"
    folder.mkdir(parents=True)
    (folder / "items.json").write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    page = writer.write(folder, evidence, "x-review-v1", "دور نهم بررسی تلفظ", 5).read_text(encoding="utf-8")
    assert not re.search(r"__[A-Z]+__", page)
    assert page.count("<title>دور نهم بررسی تلفظ</title>") == 1
    assert "سه خوانش" in page and "<b>۴</b> جمله" in page and "<b>۱۲</b> خوانش" in page and "<b>۵</b> دقیقه" in page
    assert 'const VERSION = "x-review-v1";' in page
    assert 'input.type = "checkbox"' in page and 'input.type = "radio"' not in page
    embedded = re.search(r'<script type="application/json" id="items">(.*?)</script>', page, re.S).group(1)
    assert json.loads(embedded) == ITEMS
    assert (evidence / "page.html").read_text(encoding="utf-8") == page


def test_sentences_with_different_numbers_of_readings_are_refused(tmp_path: Path) -> None:
    uneven = [ITEMS[0], {**ITEMS[1], "slots": ITEMS[1]["slots"][:2]}]
    (tmp_path / "items.json").write_text(json.dumps(uneven, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit):
        writer.write(tmp_path, tmp_path / "evidence", "x", "t", 5)
