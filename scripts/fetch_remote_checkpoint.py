#!/usr/bin/env python3
"""Fetch the finetuned checkpoint from Kaggle and prove it is worth measuring.

Two failure modes this guards against, both of which produce a file that looks
entirely normal:

*The run trained nothing.* `finetune_cli` has more than one way to complete
without taking a gradient step - see `docs/REMOTE-TRAINING.md`. The symptom is
weights bit-identical to the ones sent up, and the only way to notice is to
compare them.

*The benchmark measures the old weights anyway.* The engine converts a training
checkpoint to EMA-only safetensors and caches the result as
`<stem>_ema.safetensors` beside the input, so a second `model_last.pt` landing
in a directory that already holds one is served from the first one's cache. The
fetched file is given a name of its own.

    .venv/bin/python scripts/fetch_remote_checkpoint.py
    .venv/bin/python scripts/fetch_remote_checkpoint.py --label finetuned-80ep
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KERNEL = "marsmanam/persian-f5-multispeaker-train"
BASELINE_REPO = "Lumos675/F5_TTS_Persian"
CACHE = Path("~/.cache/persian-voice-cloning").expanduser()

#: Tensors that must move if anything was learned. The text embedding is the
#: last thing a continuation touches, so it is the strictest of the three.
PROBES = (
    "transformer.transformer_blocks.0.attn.to_q.weight",
    "transformer.norm_out.linear.weight",
    "transformer.text_embed.text_embed.weight",
)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def output_urls() -> dict[str, str]:
    """Signed download URLs for the kernel's output files, freshly minted.

    They expire, so this is called again on every retry rather than cached.
    """
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path("~/.kaggle-oauth").expanduser()))
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    owner, slug = KERNEL.split("/")
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    return {f.file_name: f.url for f in response.files or []}


def download_resumable(name: str, destination: Path, attempts: int = 40) -> Path:
    """Fetch one output file, continuing where a broken transfer left off.

    `kaggle kernels output` reads the whole response into memory with a single
    `.content` and has no resume, so a 2 GB file over a flaky link never lands -
    successive attempts got 946 MB, then 4 MB, then 8 MB, each starting again
    from zero. Range requests turn that into progress that accumulates.
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    total: int | None = None

    for attempt in range(1, attempts + 1):
        have = destination.stat().st_size if destination.is_file() else 0
        if total is not None and have >= total:
            break

        url = output_urls().get(name)
        if url is None:
            raise SystemExit(f"{name} is not among the kernel's output files")

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                if have and response.status_code == 200:
                    # Server ignored the range; start over rather than corrupt.
                    have, mode = 0, "wb"
                elif response.status_code in (200, 206):
                    mode = "ab" if have else "wb"
                else:
                    print(f"  attempt {attempt}: HTTP {response.status_code}", flush=True)
                    time.sleep(5)
                    continue

                if total is None:
                    span = response.headers.get("Content-Range")
                    if span and "/" in span:
                        total = int(span.rsplit("/", 1)[1])
                    elif response.headers.get("Content-Length"):
                        total = have + int(response.headers["Content-Length"])

                with destination.open(mode) as out:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        out.write(chunk)
        except Exception as error:  # noqa: BLE001 - every transport failure is retryable
            got = destination.stat().st_size if destination.is_file() else 0
            print(
                f"  attempt {attempt}: {type(error).__name__} at "
                f"{got / 1e6:.0f} MB{f' of {total / 1e6:.0f}' if total else ''}",
                flush=True,
            )
            time.sleep(min(5 * attempt, 30))
            continue

        got = destination.stat().st_size
        print(f"  attempt {attempt}: {got / 1e9:.2f} GB", flush=True)
        if total is not None and got >= total:
            break
        time.sleep(2)

    got = destination.stat().st_size if destination.is_file() else 0
    if total is not None and got < total:
        raise SystemExit(f"{name}: stopped at {got} of {total} bytes")
    return destination


def kaggle(*args: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    # The legacy kaggle.json authenticates reads but not writes, and shadows the
    # OAuth credentials when both are present in the same directory.
    environment.setdefault("KAGGLE_CONFIG_DIR", str(Path("~/.kaggle-oauth").expanduser()))
    return subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "kaggle"), *args],
        capture_output=True,
        text=True,
        env=environment,
    )


def ready(kernel: str) -> bool:
    result = kaggle("kernels", "status", kernel)
    status = (result.stdout or result.stderr).strip()
    print(f"  {status}")
    return "COMPLETE" in status


def newest_checkpoint(directory: Path) -> Path | None:
    """Prefer a numbered save over model_last.pt: same weights, honest name."""
    numbered = sorted(
        (p for p in directory.rglob("model_*.pt") if p.name != "model_last.pt"),
        key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0),
    )
    if numbered:
        return numbered[-1]
    last = list(directory.rglob("model_last.pt"))
    return last[0] if last else None


def weights_of(path: Path) -> dict:
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("ema_model_state_dict", "model_state_dict"):
        if key in state:
            weights = state[key]
            break
    else:
        weights = state
    return {
        (k[len("ema_model.") :] if k.startswith("ema_model.") else k): v
        for k, v in weights.items()
        if k not in {"initted", "step"}
    }


def compare(fetched: Path, baseline: Path) -> bool:
    """True when the weights actually moved."""
    import torch

    after = weights_of(fetched)
    before = weights_of(baseline)

    moved = True
    for name in PROBES:
        if name not in after or name not in before:
            print(f"  {name}: missing from one side - layouts differ")
            continue
        a, b = after[name].float(), before[name].float()
        if a.shape != b.shape:
            print(f"  {name}: shape changed {tuple(b.shape)} -> {tuple(a.shape)}")
            continue
        delta = (a - b).abs()
        relative = delta.mean().item() / (b.abs().mean().item() + 1e-12)
        identical = torch.equal(a, b)
        print(
            f"  {name}\n"
            f"      mean |delta| {delta.mean().item():.3e}   "
            f"relative {relative:.4%}   {'IDENTICAL' if identical else 'changed'}"
        )
        if identical:
            moved = False
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kernel", default=KERNEL)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "remote")
    parser.add_argument("--label", default="finetuned")
    parser.add_argument("--file", default="ckpts/model_last.pt")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    _rule("Kernel")
    if not ready(args.kernel):
        # A kernel that ran out of disk while saving still has every checkpoint
        # it wrote before that, and those are worth having. Only a run with no
        # output at all is nothing to fetch - which the file listing decides,
        # not the status.
        print("  not COMPLETE - taking whatever it managed to save")

    args.out.mkdir(parents=True, exist_ok=True)
    named = args.out / f"{args.label}.pt"

    if not args.skip_download:
        _rule("Output files")
        available = output_urls()
        for name in sorted(available):
            print(f"  {name}")
        # model_last.pt is written once more when training ends, so it is the
        # furthest along; the numbered saves are earlier snapshots.
        wanted = args.file if args.file in available else "ckpts/model_last.pt"
        if wanted not in available:
            print(f"\n{wanted} is not there - the run saved nothing.", file=sys.stderr)
            return 2

        _rule(f"Fetching {wanted}")
        download_resumable(wanted, named)

    if not named.is_file():
        print(f"\nNo checkpoint at {named}.", file=sys.stderr)
        return 2
    print(f"\n  {named}  ({named.stat().st_size / 1e9:.2f} GB)")

    _rule("Did it learn anything")
    from huggingface_hub import hf_hub_download

    baseline = Path(hf_hub_download(BASELINE_REPO, "model_last.pt", cache_dir=str(CACHE)))
    moved = compare(named, baseline)

    if not moved:
        print("\nWeights are identical to the ones sent up. The run took no")
        print("gradient step - measuring this would waste an hour to reproduce")
        print("the baseline. Read the kernel log before rerunning.", file=sys.stderr)
        return 3

    _rule("Measure it")
    print(f"  VOICECLONE__ENGINES__F5__CHECKPOINT={named} \\")
    print(f"    .venv/bin/python scripts/benchmark.py --label {args.label} --device mps")
    print(f"\n  .venv/bin/python scripts/benchmark.py --compare baseline {args.label}")
    print("\n  Held-out similarity must move more than 0.0205 against 0.520,")
    print("  with CER staying at or below the 7.8% floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
