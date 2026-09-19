#!/usr/bin/env python3
"""Make a verified float16 copy of the cached OmniVoice weights, so the worker reads half the bytes.

The worker already runs OmniVoice in float16: it reads the published float32
weights and casts every one of them as it loads - neither OmniVoice nor its
Qwen3 backbone declares ``_keep_in_fp32_modules``, so nothing stays float32.
Casting once, here, gives the same numbers from a file half the size, and
spares each start the float32 reads; on 2026-09-19 a start reached warning
memory pressure on this Mac while those weights were loading.

The copy is checked before it can be used: every tensor must equal the
float32 original cast again, bit for bit; tensors that are not floating point
must be unchanged; safetensors itself must accept the new file's header; and
where Metal is present, a sample is also cast on it, to show the conversion
rounds the same way there, since the worker may cast on either side.
``SOURCE.json`` records the model, the revision, both files' sha256 and the
checks; the engine loads a copy only if that record names the revision cached
now and says ``verified``.

Memory: the first version read through safetensors' memory map, and reading
both files whole took this Mac to warning pressure (4.1 GB resident) before
it finished. Everything here goes through plain reads in 64 MB pieces with the
page cache bypassed (``F_NOCACHE``), so about 200 MB is ever resident.

The configuration, the text tokenizer's files and the audio tokenizer come
along unchanged; the audio tokenizer, which loads in float32, is linked, not
converted.

Run it with the OmniVoice worker's Python:

    ~/.cache/persian-voice-cloning/omnivoice/.venv/bin/python scripts/omnivoice_float16.py
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import struct
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

#: safetensors dtype names: bytes per element, and which are cast.
ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FLOATING = {"F64", "F32", "BF16"}
PIECE_BYTES = 64 * 2**20

MODEL_ID = "k2-fsa/OmniVoice"
DESTINATION = Path("~/.cache/persian-voice-cloning/omnivoice/float16").expanduser()
LINKED = ("audio_tokenizer",)
SKIPPED = ("model.safetensors", "README.md", ".gitattributes")
#: Metal is checked on a sample; this bounds each sampled tensor's size.
METAL_SAMPLE, METAL_MAX_BYTES = 4, 64 * 2**20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with uncached(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def uncached(path: Path, mode: str) -> BinaryIO:
    """Open a file whose reads and writes bypass the page cache where the system allows it."""
    handle = open(path, mode)  # noqa: SIM115 - returned to the caller's with-block
    if hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
    return handle


def read_header(handle: BinaryIO) -> tuple[dict[str, Any], int]:
    """A safetensors header and the offset where its data begins."""
    handle.seek(0)
    (length,) = struct.unpack("<Q", handle.read(8))
    return json.loads(handle.read(length)), 8 + length


def torch_dtype(name: str) -> Any:
    import torch

    return {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
            "BOOL": torch.bool}[name]


def pieces(handle: BinaryIO, start: int, info: dict[str, Any]) -> Iterator[bytes]:
    """A tensor's bytes in whole-element pieces of at most PIECE_BYTES."""
    begin, end = info["data_offsets"]
    step = PIECE_BYTES - PIECE_BYTES % ITEMSIZE[info["dtype"]]
    handle.seek(start + begin)
    remaining = end - begin
    while remaining:
        data = handle.read(min(step, remaining))
        if not data:
            raise EOFError("the file ends inside a tensor")
        remaining -= len(data)
        yield data


def as_float16(data: bytes, dtype: str) -> bytes:
    import torch

    values = torch.frombuffer(bytearray(data), dtype=torch_dtype(dtype))
    return values.to(torch.float16).numpy().tobytes()


def write_float16(source: Path, target: Path) -> int:
    """Write ``source`` with every floating tensor cast to float16, a piece at a time.

    safetensors' own writer needs every tensor at once - 1.2 GB here - so the
    file is written in its format directly: an 8-byte little-endian header
    length, a JSON header of dtypes, shapes and byte offsets padded to 8 bytes,
    then the raw bytes in offset order. Casting is elementwise, so casting a
    piece gives the bytes the whole tensor's cast would.
    """
    with uncached(source, "rb") as reader:
        header, start = read_header(reader)
        keys = sorted(k for k in header if k != "__metadata__")
        out_header: dict[str, Any] = {"__metadata__": header["__metadata__"]} if header.get("__metadata__") else {}
        offset, floating = 0, 0
        for key in keys:
            dtype, shape = header[key]["dtype"], header[key]["shape"]
            if dtype in FLOATING:
                dtype, floating = "F16", floating + 1
            size = ITEMSIZE[dtype] * _count(shape)
            out_header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
        raw = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)
        with uncached(target, "wb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for key in keys:
                dtype = header[key]["dtype"]
                for data in pieces(reader, start, header[key]):
                    out.write(as_float16(data, dtype) if dtype in FLOATING else data)
    return floating


def verify(source: Path, copy: Path) -> dict[str, Any]:
    """Every tensor of ``copy`` against ``source`` cast again, bit for bit, a piece at a time."""
    from safetensors import safe_open

    mismatched: list[str] = []
    with safe_open(str(copy), framework="pt") as parsed:
        accepted = sorted(parsed.keys())
    with uncached(source, "rb") as original, uncached(copy, "rb") as converted:
        before, before_start = read_header(original)
        after, after_start = read_header(converted)
        keys = sorted(k for k in before if k != "__metadata__")
        if keys != sorted(k for k in after if k != "__metadata__") or keys != accepted:
            mismatched.append("the tensor names differ")
        if before.get("__metadata__") != after.get("__metadata__"):
            mismatched.append("the metadata differs")
        for key in keys:
            if key not in after or after[key]["shape"] != before[key]["shape"]:
                mismatched.append(key)
                continue
            dtype = before[key]["dtype"]
            wanted = "F16" if dtype in FLOATING else dtype
            ok = after[key]["dtype"] == wanted
            ours = pieces(converted, after_start, after[key])
            buffered = b""
            for data in pieces(original, before_start, before[key]):
                expected = as_float16(data, dtype) if dtype in FLOATING else data
                while ok and len(buffered) < len(expected):
                    buffered += next(ours, b"")
                    if not buffered:
                        break
                ok = ok and buffered[: len(expected)] == expected
                buffered = buffered[len(expected):]
            if not ok or buffered or next(ours, None) is not None:
                mismatched.append(key)
    return {"mismatched": mismatched, "safetensors_accepts_the_header": True}


def metal_sample(source: Path) -> dict[str, Any]:
    """Cast a sample of tensors on Metal and on the CPU; the bits must agree."""
    import torch

    result: dict[str, Any] = {"available": bool(torch.backends.mps.is_available()), "checked": [], "unequal": []}
    if not result["available"]:
        return result
    with uncached(source, "rb") as reader:
        header, start = read_header(reader)
        candidates = sorted(((k, v) for k, v in header.items() if k != "__metadata__" and v["dtype"] in FLOATING
                             and len(v["shape"]) == 2 and ITEMSIZE[v["dtype"]] * _count(v["shape"]) <= METAL_MAX_BYTES),
                            key=lambda item: -_count(item[1]["shape"]))
        for key, info in candidates[:METAL_SAMPLE]:
            data = b"".join(pieces(reader, start, info))
            values = torch.frombuffer(bytearray(data), dtype=torch_dtype(info["dtype"]))
            on_cpu = values.to(torch.float16)
            on_metal = values.to("mps").to(torch.float16).cpu()
            result["checked"].append(key)
            if not torch.equal(on_cpu.view(torch.int16), on_metal.view(torch.int16)):
                result["unequal"].append(key)
    return result


def convert(snapshot: Path, destination: Path, model_id: str, revision: str) -> dict[str, Any]:
    source = snapshot / "model.safetensors"
    work = destination.with_name(destination.name + ".partial")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    started = time.time()
    floating = write_float16(source, work / "model.safetensors")
    checked = verify(source, work / "model.safetensors")
    metal = metal_sample(source)

    for item in snapshot.iterdir():
        if item.name in SKIPPED:
            continue
        target = work / item.name
        if item.name in LINKED:
            target.symlink_to(item, target_is_directory=item.is_dir())
        elif item.is_file():
            shutil.copyfile(item, target)

    with uncached(source, "rb") as reader:
        header, _ = read_header(reader)
    verified = not checked["mismatched"] and not metal["unequal"]
    record = {
        "model_id": model_id,
        "revision": revision,
        "verified": verified,
        "what": "every floating tensor of model.safetensors cast to float16; everything else as published",
        "source_sha256": sha256(source),
        "float16_sha256": sha256(work / "model.safetensors"),
        "tensors": len([k for k in header if k != "__metadata__"]),
        "floating_tensors": floating,
        "bytes": {"source": source.stat().st_size, "float16": (work / "model.safetensors").stat().st_size},
        "bit_for_bit_against_the_cast_original": checked,
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
