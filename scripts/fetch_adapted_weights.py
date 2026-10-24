#!/usr/bin/env python3
"""Fetch the adapted OmniVoice weights the app reads with, and verify them.

The adapted reading check (eval/runs/persian-voice-adapted-reading-v1.json)
chose the Persian adaptation's float16 export at step 1,000. It lives in the
output of the Kaggle session that trained it,
``marsmanam/persian-voice-omnivoice-train-a``, under ``run/export-1000-float16/``.
This downloads that folder into ``engines.omnivoice.adapted_dir`` - about 2 GB,
each file streamed to disk and retried, because the connection from here drops
- checks ``model.safetensors`` against the sha256 the sealed test and the check
recorded and the audio tokenizer against the base's, and only then writes
``SOURCE.json``, the record the OmniVoice engine looks for before it loads them.

    .venv/bin/python scripts/fetch_adapted_weights.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.engines.omnivoice_tts import adapted_weights  # noqa: E402

spec = importlib.util.spec_from_file_location("probe", ROOT / "kaggle" / "build_omnivoice_adapt_kernel.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

KERNEL = "persian-voice-omnivoice-train-a"
PREFIX = "run/export-1000-float16/"
NEEDED_GB = 3.0
TRIES = 6


def listing() -> list[tuple[str, str]]:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    files, token = [], None
    while True:
        with api.build_kaggle_client() as kaggle:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug, request.page_size = probe.USER, KERNEL, 100
            if token:
                request.page_token = token
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
        files += [(item.file_name, item.url) for item in response.files or [] if item.file_name.startswith(PREFIX)]
        token = response.next_page_token
        if not token:
            return files


def stream(url: str, out: Path) -> str:
    import requests

    digest = hashlib.sha256()
    part = out.with_name(out.name + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with open(part, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
                digest.update(block)
    part.replace(out)
    return digest.hexdigest()


def main() -> int:
    settings = AppConfig.load(None).engines.omnivoice
    if adapted_weights(settings) is not None:
        print(f"already here and verified: {Path(settings.adapted_dir).expanduser()}")
        return 0
    target = Path(os.path.expandvars(settings.adapted_dir)).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free / 1e9
    if free < NEEDED_GB:
        raise SystemExit(f"{free:.1f} GB free at {target}; the weights need about {NEEDED_GB:.0f} GB")
    oauth = Path.home() / ".kaggle-oauth"
    if oauth.is_dir():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(oauth))

    files = listing()
    if not any(name == PREFIX + "model.safetensors" for name, _ in files):
        raise SystemExit(f"no {PREFIX}model.safetensors in {probe.USER}/{KERNEL}'s output")
    hashes = {}
    for name, url in files:
        out = target / name[len(PREFIX):]
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, TRIES + 1):
            try:
                hashes[name[len(PREFIX):]] = stream(url, out)
                break
            except Exception as exc:  # noqa: BLE001 - the connection drops; try again, then list anew
                print(f"  {name}: try {attempt} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                if attempt == TRIES:
                    raise SystemExit(f"{name}: gave up after {TRIES} tries") from exc
                time.sleep(10)
                url = dict(listing()).get(name, url)
        print(f"  {name[len(PREFIX):]}  {out.stat().st_size / 1e6:8.1f} MB", flush=True)

    if hashes.get("model.safetensors") != settings.adapted_sha256:
        raise SystemExit("model.safetensors is not the export the check heard; nothing was marked verified")
    if hashes.get("audio_tokenizer/model.safetensors") != probe.TOKENIZER_SHA256:
        raise SystemExit("the audio tokenizer is not the base's; nothing was marked verified")
    record = {"from": f"kaggle:{probe.USER}/{KERNEL}, {PREFIX}", "step": 1000,
              "chosen_by": "eval/runs/persian-voice-adapted-reading-v1.json",
              "model_sha256": hashes["model.safetensors"],
              "model_bytes": (target / "model.safetensors").stat().st_size,
              "audio_tokenizer_sha256": hashes["audio_tokenizer/model.safetensors"],
              "files": dict(sorted(hashes.items())), "verified": True,
              "fetched": datetime.now(UTC).isoformat(timespec="seconds")}
    (target / "SOURCE.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"verified -> {target}")
    return 0 if adapted_weights(settings) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
