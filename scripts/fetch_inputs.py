#!/usr/bin/env python3
"""Fetch what a GPU session reads, every step bounded in time, reporting as it goes.

Session a of the OmniVoice adaptation (2026-09-19) printed its last line 138 s
in, as its downloads began, and nothing after that until it was cancelled:
its download cell ran ``snapshot_download`` and a ``urlopen`` with no timeout
inside the notebook's own process, so a stall held a T4 pair for the rest of
the session. Here every request has a timeout, each download is tried a
bounded number of times and says when it finishes, how much it fetched and
how long it took, and the notebook runs this script under ``run_bounded``,
which kills it at the stage's budget - so a stall ends the stage, not the week's
GPU quota.

The plan is a JSON file::

    {"snapshots": [{"name": "base", "repo": "k2-fsa/OmniVoice", "revision": "...",
                    "repo_type": "model", "allow_patterns": null, "local_dir": "/tmp/omnivoice-base",
                    "sha256": {"audio_tokenizer/model.safetensors": "..."},
                    "forbid": ["test-*.parquet"]}],
     "urls": [{"name": "sdpa config", "url": "https://...", "path": "/tmp/config.json",
               "git_blob_prefix": "..."}]}

and the report, written after every step so a killed run still leaves one,
says what arrived, its size and time, the attempts, and any check that failed.

    python scripts/fetch_inputs.py --plan plan.json --report report.json
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Seconds a single request may go without an answer before it is abandoned.
REQUEST_TIMEOUT = 60
ATTEMPTS = 3
PAUSE = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def gib(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**30


def attempt(name: str, action: Callable[[], Any], attempts: int = ATTEMPTS, pause: float = PAUSE,
            log: Callable[[str], None] = print) -> tuple[Any, list[dict[str, Any]]]:
    """Run ``action`` up to ``attempts`` times, reporting each try; re-raise the last failure."""
    tries: list[dict[str, Any]] = []
    for number in range(1, attempts + 1):
        started = time.time()
        try:
            result = action()
        except Exception as exc:  # noqa: BLE001 - recorded, then retried or re-raised
            tries.append({"attempt": number, "seconds": round(time.time() - started, 1),
                          "error": f"{type(exc).__name__}: {exc}"[:500]})
            log(f"{name}: attempt {number} failed after {tries[-1]['seconds']} s - {tries[-1]['error']}")
            if number == attempts:
                raise
            time.sleep(pause)
            continue
        tries.append({"attempt": number, "seconds": round(time.time() - started, 1)})
        return result, tries
    raise AssertionError("unreachable")


def fetch_url(url: str, timeout: float = REQUEST_TIMEOUT) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def run(plan: dict[str, Any], report_path: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    report: dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "steps": [], "ok": False}

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    for item in plan.get("urls", []):
        step = {"name": item["name"], "kind": "url"}
        report["steps"].append(step)
        data, step["tries"] = attempt(item["name"], lambda item=item: fetch_url(item["url"]), log=log)
        blob = git_blob(data)
        if item.get("git_blob_prefix") and not blob.startswith(item["git_blob_prefix"]):
            step["failed_check"] = f"git blob {blob} does not start with {item['git_blob_prefix']}"
            save()
            raise SystemExit(f"{item['name']}: {step['failed_check']}")
        Path(item["path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(item["path"]).write_bytes(data)
        step.update({"bytes": len(data), "git_blob": blob})
        log(f"{item['name']}: {len(data)} bytes, git blob {blob[:12]}")
        save()
    for item in plan.get("snapshots", []):
        step = {"name": item["name"], "kind": "snapshot", "repo": item["repo"], "revision": item["revision"]}
        report["steps"].append(step)
        started = time.time()
        folder, step["tries"] = attempt(item["name"], lambda item=item: snapshot_download(
            item["repo"], repo_type=item.get("repo_type", "model"), revision=item["revision"],
            allow_patterns=item.get("allow_patterns"), local_dir=item["local_dir"]), log=log)
        folder = Path(folder)
        size = gib(folder)
        step.update({"bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
                     "gib": round(size, 3), "seconds": round(time.time() - started, 1),
                     "mib_per_second": round(size * 1024 / max(time.time() - started, 1e-6), 1)})
        forbidden = [str(p.relative_to(folder)) for pattern in item.get("forbid", [])
                     for p in folder.rglob("*") if fnmatch.fnmatch(p.name, pattern)]
        if forbidden:
            step["failed_check"] = f"forbidden files arrived: {forbidden[:5]}"
            save()
            raise SystemExit(f"{item['name']}: {step['failed_check']}")
        for relative, expected in (item.get("sha256") or {}).items():
            actual = sha256(folder / relative)
            if actual != expected:
                step["failed_check"] = f"{relative} has sha256 {actual}, expected {expected}"
                save()
                raise SystemExit(f"{item['name']}: {step['failed_check']}")
        log(f"{item['name']}: {size:.2f} GiB in {step['seconds']:.0f} s ({step['mib_per_second']} MiB/s)")
        save()
    report["ok"] = True
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-xet", action="store_true",
                        help="Download over plain HTTP rather than Hugging Face's Xet client")
    args = parser.parse_args(argv)
    # Every Hugging Face request gives up after REQUEST_TIMEOUT seconds instead of waiting for good.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(REQUEST_TIMEOUT))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(REQUEST_TIMEOUT))
    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    def log(line: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)

    report = run(json.loads(args.plan.read_text(encoding="utf-8")), args.report, log=log)
    log("all fetched" if report["ok"] else "incomplete")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
