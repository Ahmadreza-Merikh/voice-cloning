#!/usr/bin/env python3
"""Write a blind review page for the items build_pronunciation_review.py prepared.

The page is ``scripts/review_page.html`` with the round's title, counts and
items filled in. Until the ezafe check every round copied the last round's page
by hand; from 2026-11-01 the better-reading question takes more than one pick,
because the user found two readings often equally right and could choose only
one. The page saves ``best`` as a list of slots (``["0"]`` for no difference),
and the reading scripts take that list as well as the single slot earlier pages
saved. A copy of the page and its items goes beside the key in the evidence.

    .venv/bin/python scripts/write_review_page.py --review <name>-review-v1 --title "دور نهم بررسی تلفظ" --minutes 15
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "review_page.html"
COUNTS = {2: "دو", 3: "سه", 4: "چهار", 5: "پنج"}


def persian(number: int) -> str:
    return str(number).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def write(folder: Path, evidence: Path, version: str, title: str, minutes: int) -> Path:
    items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
    per = {len(item["slots"]) for item in items}
    if len(per) != 1 or not per <= set(COUNTS):
        raise SystemExit(f"every sentence needs the same number of readings, two to five; found {sorted(per)}")
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in (("__TITLE__", title), ("__VERSION__", version), ("__PER__", COUNTS[per.pop()]),
                       ("__SENTENCES__", persian(len(items))),
                       ("__READINGS__", persian(sum(len(item["slots"]) for item in items))),
                       ("__MINUTES__", persian(minutes)), ("__ITEMS__", json.dumps(items, ensure_ascii=False))):
        page = page.replace(key, value)
    (folder / "index.html").write_text(page, encoding="utf-8")
    evidence.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / "index.html", evidence / "page.html")
    shutil.copyfile(folder / "items.json", evidence / "items.json")
    return folder / "index.html"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--review", required=True, help="the folder under listening/, also the page's version")
    parser.add_argument("--title", required=True)
    parser.add_argument("--minutes", type=int, required=True)
    args = parser.parse_args(argv)
    page = write(ROOT / "listening" / args.review, ROOT / "eval" / "runs" / "evidence" / args.review,
                 args.review, args.title, args.minutes)
    print(f"{page.relative_to(ROOT)}; a copy in eval/runs/evidence/{args.review}/page.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
