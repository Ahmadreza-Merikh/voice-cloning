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
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    _rule("Kernel")
    if not ready(args.kernel) and not args.skip_download:
        print("\nNot complete yet. Nothing to fetch.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        _rule("Fetching output")
        result = kaggle("kernels", "output", args.kernel, "-p", str(args.out))
        print((result.stdout or result.stderr).strip()[:2000])
        if result.returncode != 0:
            return 1

    checkpoint = newest_checkpoint(args.out)
    if checkpoint is None:
        print(f"\nNo model_*.pt under {args.out}.", file=sys.stderr)
        print("The run finished without reaching a save point.", file=sys.stderr)
        return 2

    named = args.out / f"{args.label}.pt"
    if checkpoint != named:
        checkpoint.replace(named)
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
