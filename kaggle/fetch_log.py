#!/usr/bin/env python3
"""Print a Kaggle kernel's log as plain lines.

``kaggle kernels output`` returns nothing while a kernel runs and, depending on
the CLI version, does not save the log at all. The API's ``kernels_logs`` does,
as a JSON list of stream records; this prints them once each, with the kernel's
own clock, so a failure can be read without opening a browser.

    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/python kaggle/fetch_log.py marsmanam/<slug>
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle-oauth"))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out = api.kernels_logs(sys.argv[1])
    raw = out if isinstance(out, str) else json.dumps(out)
    rows = json.loads(raw if raw.strip().startswith("[") else "[" + raw.strip().lstrip(",") + "]")
    previous = None
    for row in rows:
        data = row["data"].rstrip()
        if data != previous:
            print(f"{row['time']:8.1f} {data}")
        previous = data
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
