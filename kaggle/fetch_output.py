#!/usr/bin/env python3
"""Download chosen files from a finished Kaggle kernel's output, with retries.

``kaggle kernels output`` fetches everything, and a training session's output
holds a checkpoint of several GiB. This lists the output, writes the listing
and the kernel's log into the target folder, and downloads only the files whose
path contains one of the given fragments - each request retried, because the
connection from here drops.

    .venv/bin/python kaggle/fetch_output.py marsmanam/<slug> remote/<slug> session/ logs/
    .venv/bin/python kaggle/fetch_output.py marsmanam/<slug> remote/<slug>        # listing and log only

Where ``~/.kaggle-oauth`` exists it is used as ``KAGGLE_CONFIG_DIR`` (this Mac's
OAuth login lives there); elsewhere the Kaggle CLI's own default applies.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def retry(action, what, tries=12, pause=10):
    for attempt in range(1, tries + 1):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - the connection drops; try again
            print(f"  {what}: try {attempt} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(pause)
    raise SystemExit(f"{what}: gave up after {tries} tries")


def main() -> int:
    if len(sys.argv) < 3 or "/" not in sys.argv[1]:
        raise SystemExit(__doc__)
    owner, slug = sys.argv[1].split("/", 1)
    target, fragments = Path(sys.argv[2]), sys.argv[3:]
    oauth = Path.home() / ".kaggle-oauth"
    if oauth.is_dir():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(oauth))
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    files, token, log = [], None, None
    while True:
        def page(token=token):
            with api.build_kaggle_client() as kaggle:
                request = ApiListKernelSessionOutputRequest()
                request.user_name, request.kernel_slug, request.page_size = owner, slug, 100
                if token:
                    request.page_token = token
                return kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
        response = retry(page, "listing")
        files.extend((item.file_name, item.url) for item in response.files or [])
        log = log or response.log
        token = response.next_page_token
        if not token:
            break

    target.mkdir(parents=True, exist_ok=True)
    (target / "_listing.txt").write_text("\n".join(name for name, _ in files) + "\n", encoding="utf-8")
    if log:
        (target / f"{slug}.log").write_text(log, encoding="utf-8")
    fetched = kept = 0
    for name, url in files:
        if not fragments or not any(fragment in name for fragment in fragments):
            continue
        out = target / name
        if out.is_file() and out.stat().st_size > 0:
            kept += 1
            continue

        def get(url=url):
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response.content
        data = retry(get, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        fetched += 1
    print(f"{len(files)} files in the output; fetched {fetched}, already here {kept}; "
          f"log {'saved' if log else 'not returned'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
