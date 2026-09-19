#!/usr/bin/env python3
"""Make a verified float16 copy of the cached OmniVoice weights, so the worker reads half the bytes.

The worker already runs OmniVoice in float16: it reads the published float32
weights and casts every one of them as it loads - neither OmniVoice nor its
Qwen3 backbone declares ``_keep_in_fp32_modules``, so nothing stays float32.
Casting once, here, gives the same numbers from a file half the size, and
spares each start the float32 reads; on 2026-09-19 a start reached warning
memory pressure on this Mac while those weights were loading.

The copy is checked before it can be used: every tensor, read back from the
new file, must equal the float32 original cast again, bit for bit; tensors
that are not floating point must be unchanged; and where Metal is present, a
sample is also cast on it to show the conversion rounds the same way there,
since the worker may cast on either side. ``SOURCE.json`` records the model,
the revision, both files' sha256 and the checks; the engine loads a copy only
if that record names the revision cached now and says ``verified``.

The configuration, the text tokenizer's files and the audio tokenizer come
along unchanged; the audio tokenizer, which loads in float32, is linked, not
converted.

Run it with the OmniVoice worker's Python:

    ~/.cache/persian-voice-cloning/omnivoice/.venv/bin/python scripts/omnivoice_float16.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

#: safetensors dtype names: bytes per element, and which are cast.
ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FLOATING = {"F64", "F32", "BF16"}

MODEL_ID = "k2-fsa/OmniVoice"
DESTINATION = Path("~/.cache/persian-voice-cloning/omnivoice/float16").expanduser()
LINKED = ("audio_tokenizer",)
SKIPPED = ("model.safetensors", "README.md", ".gitattributes")
#: Metal is checked on a sample; this bounds each sampled tensor's size.
METAL_SAMPLE, METAL_MAX_BYTES = 4, 64 * 2**20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same_bits(a: Any, b: Any) -> bool:
    import torch

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.float16:
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    return bool(torch.equal(a, b))


def write_float16(source: Path, target: Path) -> int:
    """Write ``source`` with every floating tensor cast to float16, one tensor in memory at a time.

    safetensors' own writer needs every tensor at once - 1.2 GB here - so the
    file is written directly in its format: an 8-byte little-endian header
    length, a JSON header of dtypes, shapes and byte offsets padded to 8 bytes,
    then the raw bytes in offset order. The verification reads it back with
    safetensors itself.
    """
    import torch
    from safetensors import safe_open

    with safe_open(str(source), framework="pt") as handle:
        keys = sorted(handle.keys())
        header: dict[str, Any] = {}
        metadata = handle.metadata()
        if metadata:
            header["__metadata__"] = metadata
        offset, floating = 0, 0
        for key in keys:
            piece = handle.get_slice(key)
            dtype = piece.get_dtype()
            if dtype in FLOATING:
                dtype, floating = "F16", floating + 1
            size = ITEMSIZE[dtype] * _count(piece.get_shape())
            header[key] = {"dtype": dtype, "shape": piece.get_shape(), "data_offsets": [offset, offset + size]}
            offset += size
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)
        with target.open("wb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for key in keys:
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(torch.float16)
                out.write(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return floating


def convert(snapshot: Path, destination: Path, model_id: str, revision: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    source = snapshot / "model.safetensors"
    work = destination.with_name(destination.name + ".partial")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    started = time.time()
    floating = write_float16(source, work / "model.safetensors")

    mismatched, metal = [], {"available": bool(torch.backends.mps.is_available()), "checked": [], "unequal": []}
    with safe_open(str(source), framework="pt") as original, safe_open(str(work / "model.safetensors"),
                                                                       framework="pt") as copy:
        keys = list(original.keys())
        if sorted(keys) != sorted(copy.keys()):
            mismatched.append("the tensor names differ")
        for key in keys:
            before, after = original.get_tensor(key), copy.get_tensor(key)
            expected = before.to(torch.float16) if before.is_floating_point() else before
            if not same_bits(expected, after):
                mismatched.append(key)
        if metal["available"]:
            sizes = sorted(((original.get_slice(k).get_shape(), k) for k in keys), key=lambda s: -_count(s[0]))
            sample = [k for shape, k in sizes if 4 * _count(shape) <= METAL_MAX_BYTES and len(shape) == 2]
            for key in sample[:METAL_SAMPLE]:
                before = original.get_tensor(key)
                on_metal = before.to("mps").to(torch.float16).cpu()
                metal["checked"].append(key)
                if not same_bits(before.to(torch.float16), on_metal):
                    metal["unequal"].append(key)

    for item in snapshot.iterdir():
        if item.name in SKIPPED:
            continue
        target = work / item.name
        if item.name in LINKED:
            target.symlink_to(item.resolve() if not item.is_dir() else item, target_is_directory=item.is_dir())
        elif item.is_file():
            shutil.copyfile(item, target)

    verified = not mismatched and not metal["unequal"]
    record = {
        "model_id": model_id,
        "revision": revision,
        "verified": verified,
        "what": "every floating tensor of model.safetensors cast float32 -> float16; everything else as published",
        "source_sha256": sha256(source),
        "float16_sha256": sha256(work / "model.safetensors"),
        "tensors": len(keys),
        "floating_tensors": floating,
        "bytes": {"source": source.stat().st_size, "float16": (work / "model.safetensors").stat().st_size},
        "bit_for_bit_against_the_cast_original": {"mismatched": mismatched},
        "metal_cast_sample": metal,
        "linked": list(LINKED),
        "seconds": round(time.time() - started, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "made_by": "scripts/omnivoice_float16.py",
    }
    (work / "SOURCE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    work.rename(destination)
    return record


def _count(shape: list[int]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None, help="Default: the revision cached now")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(args.model, revision=args.revision, local_files_only=True))
    record = convert(snapshot, args.destination / snapshot.name, args.model, snapshot.name)
    print(json.dumps(record, indent=2))
    return 0 if record["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
