#!/usr/bin/env python3
"""Fold the published OmniVoice IPA diff into a loadable set of weights, once.

What arrives, and what this does
--------------------------------
``christopherthompson81/omnivoice-ipa-onnx`` ships *no* full model: it ships
``ipa_diff.onnx``, a diff of ``k2-fsa/OmniVoice`` continued on canonical
Persian IPA (``vernacula-phonemizer``). The diff holds, per its metadata and
initializers:

* 196 LoRA modules - ``<module>.lora_A`` and ``<module>.lora_B``, scale from
  ``lora_scale=2.0``;
* ``embed_rows`` (5790, 1024) float16 + ``embed_idx`` (5790,) int32 - the
  changed vocabulary rows of ``llm.embed_tokens.weight``, selected at
  ``embed_threshold=0.001``.

The fold, into a *local copy* of the base weights (never the HF cache)::

    W += lora_scale * (B @ A)                      # (out,r) @ (r,in) = (out,in)
    llm.embed_tokens.weight[embed_idx] = embed_rows

No transpose is assumed: the orientation is *derived from the shapes*
(``_resolve_orientation``), so a differently-stored diff fails loudly instead
of silently corrupting weights.

Everything is validated *before* a single byte is patched, the patch runs
through a whole-file ``mmap`` (a 2.45 GB file; only touched pages load), a
sample of the result is re-read against base + recomputed delta, and
``IPA_ADAPTATION.json`` - sha256 of the diff, scale, counts, licences - is
written last: its presence IS the "built" marker the engine checks.

Licences: base Apache-2.0, fine-tune CC-BY-NC-4.0 (non-commercial); the
merged weights carry the fine-tune's licence.

Dependencies: numpy (the fold) and onnx (reading the diff), both optional and
imported where used with actionable errors. Nothing from the project.

Usage
-----
    .venv/bin/python scripts/build_omnivoice_ipa.py
    .venv/bin/python scripts/build_omnivoice_ipa.py --diff ipa_diff.onnx
    .venv/bin/python scripts/build_omnivoice_ipa.py --out DIR --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BASE_REPO = "k2-fsa/OmniVoice"  # Apache-2.0
DIFF_REPO = "christopherthompson81/omnivoice-ipa-onnx"  # CC-BY-NC-4.0
DIFF_FILE = "ipa_diff.onnx"
#: Keep in sync with src/voiceclone/engines/omnivoice_ipa.py (pinned by tests).
PROVENANCE_NAME = "IPA_ADAPTATION.json"
DEFAULT_OUT = "~/.cache/persian-voice-cloning/omnivoice-ipa"
EMBED_TARGET = "llm.embed_tokens.weight"
WEIGHT_DTYPE = np.float32  # the base stores every llm tensor as F32


# --------------------------------------------------------------------------
# Pure pieces (unit-tested without onnx, network or a 2.5 GB file)
# --------------------------------------------------------------------------
def parse_safetensors_header(path: Path | str) -> tuple[int, dict[str, Any]]:
    """``(data_section_offset, header)`` of a safetensors file.

    The header is an 8-byte little-endian length followed by that many bytes
    of JSON; tensor ``data_offsets`` are relative to the byte after it.
    """
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return 8 + length, header


def _resolve_orientation(a: np.ndarray, b: np.ndarray, weight_shape: tuple[int, ...]) -> str:
    """Which product produces the weight: ``"direct"`` (B @ A) or ``"transposed"``.

    Direct storage is ``A: (rank, in)``, ``B: (out, rank)`` - the documented
    shape of the published diff. The transposed storage (``A: (in, rank)``,
    ``B: (rank, out)``) is accepted too, because *deriving* the orientation is
    the whole safeguard: guessing would corrupt weights without an error.
    Raises ``ValueError`` when no orientation fits the actual weight shape.
    """
    tried: list[tuple[str, tuple[int, ...]]] = []
    if a.ndim == 2 and b.ndim == 2 and a.shape[0] == b.shape[1]:
        tried.append(("direct", (b.shape[0], a.shape[1])))
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
        # delta = (A @ B).T, so its shape is the transpose of A @ B's.
        tried.append(("transposed", (b.shape[1], a.shape[0])))
    for name, shape in tried:
        if shape == tuple(weight_shape):
            return name
    detail = (
        "; ".join(f"{name} -> {shape}" for name, shape in tried)
        if tried
        else "the rank dimensions do not align at all"
    )
    raise ValueError(
        f"no orientation of A{tuple(a.shape)} and B{tuple(b.shape)} produces "
        f"weight {tuple(weight_shape)} (tried: {detail})"
    )


def fold_lora_delta(
    weight: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float
) -> np.ndarray:
    """``weight += scale * (B @ A)``, in place, and return the weight.

    The delta is computed in float32 (the base's own dtype) from the shapes'
    resolved orientation, then added in place. ``weight`` must be float32 -
    the format this project's base ships in.
    """
    if weight.dtype != WEIGHT_DTYPE:
        raise ValueError(f"weight dtype {weight.dtype} is not {WEIGHT_DTYPE}")
    orientation = _resolve_orientation(a, b, weight.shape)
    a32 = a.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)
    if orientation == "direct":
        delta = b32 @ a32
    else:
        delta = (a32 @ b32).T
    weight += (delta * float(scale)).astype(weight.dtype, copy=False)
    return weight


def apply_embed_rows(table: np.ndarray, idx: np.ndarray, rows: np.ndarray) -> None:
    """``table[idx] = rows`` in place: row *selection*, not accumulation."""
    if table.ndim != 2 or rows.ndim != 2:
        raise ValueError("table and rows must be 2-D")
    if rows.shape != (idx.size, table.shape[1]):
        raise ValueError(
            f"rows {rows.shape} do not fit table {table.shape} for {idx.size} indices"
        )
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= table.shape[0]):
        raise ValueError(f"embed index out of range for {table.shape[0]} rows")
    table[idx] = rows.astype(table.dtype, copy=False)


# --------------------------------------------------------------------------
# Locating the inputs (no dependency required; network only as a fallback)
# --------------------------------------------------------------------------
def _snapshot_from_hub_cache() -> Path | None:
    """Newest cached snapshot of ``BASE_REPO`` holding model.safetensors."""
    hub = Path("~/.cache/huggingface/hub").expanduser()
    root = hub / f"models--{BASE_REPO.replace('/', '--')}" / "snapshots"
    if not root.is_dir():
        return None
    candidates = [
        path for path in sorted(root.iterdir()) if (path / "model.safetensors").exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def locate_base(explicit: str | None = None) -> Path:
    """A local snapshot of the base model with its weights."""
    if explicit:
        path = Path(explicit).expanduser()
        if not (path / "model.safetensors").is_file():
            raise SystemExit(f"--base {path} has no model.safetensors")
        return path
    try:  # the project's environment already has hub; its cache is checked first
        from huggingface_hub import snapshot_download

        try:
            return Path(snapshot_download(BASE_REPO, local_files_only=True))
        except Exception:
            pass
        try:  # only reached when the base was never fetched - it must be, to run
            return Path(snapshot_download(BASE_REPO))
        except Exception:
            pass
    except ImportError:
        pass
    cached = _snapshot_from_hub_cache()
    if cached is not None:
        return cached
    raise SystemExit(
        f"Base model {BASE_REPO} not found locally.\n"
        "Fetch it once (it is what the engine runs on anyway):\n"
        f"  huggingface-cli download {BASE_REPO}\n"
        "or pass an existing snapshot with --base DIR."
    )


def locate_diff(explicit: str | None = None) -> Path:
    """The published diff, as a local file."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"--diff {path} is not a file")
        return path
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(DIFF_REPO, DIFF_FILE))
    except ImportError:
        pass
    except Exception:
        pass
    raise SystemExit(
        f"{DIFF_FILE} not available from {DIFF_REPO}.\n"
        "Pass a downloaded copy with --diff PATH "
        f"(https://huggingface.co/{DIFF_REPO}/resolve/main/{DIFF_FILE})."
    )


# --------------------------------------------------------------------------
# Reading the diff
# --------------------------------------------------------------------------
@dataclass
class ModulePlan:
    """One LoRA module, ready to validate and apply."""

    module: str
    weight_key: str = ""
    a_name: str = ""
    b_name: str = ""


def load_diff(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The diff's initializers as arrays, plus its metadata.

    ``onnx`` is imported here so the fold's math stays testable without it.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Reading the diff needs onnx (not a project dependency, only this script):\n"
            "  uv pip install onnx"
        ) from exc
    model = onnx.load(str(path))
    meta = {prop.key: prop.value for prop in model.metadata_props}
    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    return arrays, meta


def plan_modules(arrays: dict[str, np.ndarray]) -> list[ModulePlan]:
    """Pair every ``.lora_A``/``.lora_B`` initializer into module plans."""
    plans: dict[str, ModulePlan] = {}
    for name in sorted(arrays):
        if name.endswith(".lora_A"):
            module = name[: -len(".lora_A")]
            plans.setdefault(module, ModulePlan(module)).a_name = name
        elif name.endswith(".lora_B"):
            module = name[: -len(".lora_B")]
            plans.setdefault(module, ModulePlan(module)).b_name = name
    incomplete = [p.module for p in plans.values() if not (p.a_name and p.b_name)]
    if incomplete:
        raise SystemExit(f"diff has an unpaired LoRA factor for: {', '.join(incomplete[:5])}")
    for plan in plans.values():
        plan.weight_key = f"{plan.module}.weight"
    return sorted(plans.values(), key=lambda plan: plan.module)


def validate_targets(
    header: dict[str, Any],
    plans: Sequence[ModulePlan],
    arrays: dict[str, np.ndarray],
) -> list[str]:
    """Every patch target must check out BEFORE any byte is written."""
    problems: list[str] = []
    for plan in plans:
        entry = header.get(plan.weight_key)
        if entry is None:
            problems.append(f"{plan.weight_key}: not in the base weights")
            continue
        if entry.get("dtype") != "F32":
            problems.append(f"{plan.weight_key}: dtype {entry.get('dtype')}, expected F32")
            continue
        try:
            _resolve_orientation(arrays[plan.a_name], arrays[plan.b_name], entry["shape"])
        except ValueError as exc:
            problems.append(f"{plan.weight_key}: {exc}")

    embed = header.get(EMBED_TARGET)
    rows = arrays.get("embed_rows")
    idx = arrays.get("embed_idx")
    if embed is None:
        problems.append(f"{EMBED_TARGET}: not in the base weights")
    elif embed.get("dtype") != "F32":
        problems.append(f"{EMBED_TARGET}: dtype {embed.get('dtype')}, expected F32")
    elif rows is None or idx is None:
        problems.append("diff has no embed_rows/embed_idx pair")
    else:
        shape = tuple(embed["shape"])
        if rows.ndim != 2 or rows.shape[1] != shape[1]:
            problems.append(f"embed_rows {tuple(rows.shape)} does not match {shape}")
        elif idx.size == 0:
            problems.append("embed_idx is empty")
        elif int(idx.max()) >= shape[0] or int(idx.min()) < 0:
            problems.append(f"embed_idx range [{int(idx.min())}, {int(idx.max())}] outside {shape[0]} rows")

    known = {p.a_name for p in plans} | {p.b_name for p in plans} | {"embed_rows", "embed_idx"}
    extra = set(arrays) - known
    if extra:
        problems.append(f"unexpected initializers: {sorted(extra)[:5]}")

    if problems:
        raise SystemExit("diff does not match the base weights:\n  " + "\n  ".join(problems))
    return [plan.weight_key for plan in plans]


# --------------------------------------------------------------------------
# The patch
# --------------------------------------------------------------------------
def _tensor_view(mm: mmap.mmap, header_offset: int, entry: dict[str, Any]) -> np.ndarray:
    """A writable float32 view of one tensor inside the mapped file."""
    shape = tuple(entry["shape"])
    count = int(np.prod(shape)) if shape else 1
    start = header_offset + int(entry["data_offsets"][0])
    end = header_offset + int(entry["data_offsets"][1])
    if entry.get("dtype") != "F32" or end - start != count * 4:
        raise ValueError(f"tensor is not a plain F32 block: {entry}")
    return np.frombuffer(mm, WEIGHT_DTYPE, count=count, offset=start).reshape(shape)


def _fold_one(
    mm: mmap.mmap,
    header_offset: int,
    header: dict[str, Any],
    plan: ModulePlan,
    arrays: dict[str, np.ndarray],
    scale: float,
) -> None:
    """Fold one module. The view lives and dies here: an np.frombuffer array
    exports the mmap's buffer, and mmap.close() refuses while any export is
    alive (BufferError), so no view may survive into the enclosing ``with``."""
    weight = _tensor_view(mm, header_offset, header[plan.weight_key])
    fold_lora_delta(weight, arrays[plan.a_name], arrays[plan.b_name], scale)


def _embed_one(
    mm: mmap.mmap, header_offset: int, header: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    """Write the vocabulary rows, releasing the exported view on return."""
    table = _tensor_view(mm, header_offset, header[EMBED_TARGET])
    apply_embed_rows(table, arrays["embed_idx"], arrays["embed_rows"])


def patch_weights(
    safetensors: Path,
    header_offset: int,
    header: dict[str, Any],
    plans: Sequence[ModulePlan],
    arrays: dict[str, np.ndarray],
    scale: float,
) -> None:
    """Apply every LoRA fold and the embedding rows, in place, via mmap."""
    with open(safetensors, "r+b") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_WRITE
    ) as mm:
        for plan in plans:
            _fold_one(mm, header_offset, header, plan, arrays, scale)
        _embed_one(mm, header_offset, header, arrays)
        mm.flush()


def read_tensor_at(path: Path, header_offset: int, entry: dict[str, Any]) -> np.ndarray:
    """Read one tensor in full - for verification only, never for patching."""
    shape = tuple(entry["shape"])
    count = int(np.prod(shape)) if shape else 1
    with open(path, "rb") as handle:
        handle.seek(header_offset + int(entry["data_offsets"][0]))
        data = np.fromfile(handle, WEIGHT_DTYPE, count)
    if data.size != count:
        raise ValueError(f"short read from {path}")
    return data.reshape(shape)


def verify(
    patched: Path,
    base_safetensors: Path,
    patched_offset: int,
    base_offset: int,
    patched_header: dict[str, Any],
    base_header: dict[str, Any],
    plans: Sequence[ModulePlan],
    arrays: dict[str, np.ndarray],
    scale: float,
) -> list[str]:
    """Re-read a sample of the patched file against base + recomputed delta."""
    if not plans:
        return ["nothing to verify"]
    # ModulePlan is a mutable dataclass (unhashable): dedupe by module name.
    picked = {p.module: p for p in (plans[0], plans[len(plans) // 2], plans[-1])}
    sample = sorted(picked.values(), key=lambda p: p.module)
    for plan in sample:
        before = read_tensor_at(base_safetensors, base_offset, base_header[plan.weight_key])
        expected = fold_lora_delta(
            before.copy(), arrays[plan.a_name], arrays[plan.b_name], scale
        )
        after = read_tensor_at(patched, patched_offset, patched_header[plan.weight_key])
        if not np.allclose(after, expected, rtol=1e-6, atol=1e-6):
            raise SystemExit(f"verification failed for {plan.weight_key}: folded values differ")

    before = read_tensor_at(base_safetensors, base_offset, base_header[EMBED_TARGET])
    after = read_tensor_at(patched, patched_offset, patched_header[EMBED_TARGET])
    idx = arrays["embed_idx"]
    rows = arrays["embed_rows"].astype(np.float32)
    # Later writes win for a repeated id, exactly as numpy's assignment does.
    written = {int(vocab_id): rows[position] for position, vocab_id in enumerate(idx)}
    probes = list(dict.fromkeys([int(i) for i in idx[:5]] + [int(i) for i in idx[-5:]]))
    for vocab_id in probes:
        if not np.array_equal(after[vocab_id], written[vocab_id]):
            raise SystemExit(f"verification failed for embedding row {vocab_id}")
    if np.array_equal(after, before):
        raise SystemExit("verification failed: the embedding table did not change")
    return [
        f"verified {len(sample)} folded modules against base + delta "
        f"(out of {len(plans)}), plus {len(probes)} embedding rows"
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=None, help=f"snapshot of {BASE_REPO} (default: cache)")
    parser.add_argument("--diff", default=None, help=f"path to {DIFF_FILE} (default: hub cache)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where the merged weights are written")
    parser.add_argument("--force", action="store_true", help="rebuild even if already built")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser()
    marker = out / PROVENANCE_NAME
    if marker.is_file() and not args.force:
        print(f"already built: {marker}")
        print("re-run with --force to rebuild it from scratch")
        return 0

    base = locate_base(args.base)
    diff = locate_diff(args.diff)
    print(f"base:  {base}")
    print(f"diff:  {diff}")

    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"--out {out} exists and is not a directory")
        has_weights = (out / "model.safetensors").exists()
        if not has_weights and any(out.iterdir()):
            raise SystemExit(
                f"--out {out} exists but contains no model.safetensors; "
                "refusing to delete something that is not a build output"
            )
        print(f"removing previous output at {out}")
        shutil.rmtree(out)
    print(f"copying the base snapshot -> {out} (materialising symlinks)")
    shutil.copytree(base, out, symlinks=False)

    print("reading the diff ...")
    arrays, meta = load_diff(diff)
    scale = float(meta.get("lora_scale", 1.0))
    plans = plan_modules(arrays)
    safetensors = out / "model.safetensors"
    header_offset, header = parse_safetensors_header(safetensors)
    keys = validate_targets(header, plans, arrays)
    print(
        f"validated {len(plans)} LoRA modules (lora_scale={scale}) + "
        f"{int(arrays['embed_rows'].shape[0])} embedding rows before touching a byte"
    )

    patch_weights(safetensors, header_offset, header, plans, arrays, scale)

    base_offset, base_header = parse_safetensors_header(base / "model.safetensors")
    notes = verify(
        safetensors,
        base / "model.safetensors",
        header_offset,
        base_offset,
        header,
        base_header,
        plans,
        arrays,
        scale,
    )
    for note in notes:
        print(note)

    provenance = {
        "builder": "scripts/build_omnivoice_ipa.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_repo": BASE_REPO,
        "base_source": str(base),
        "base_revision": base.name if base.parent.name == "snapshots" else "",
        "diff_repo": DIFF_REPO,
        "diff_file": DIFF_FILE,
        "diff_source": str(diff),
        "diff_sha256": sha256_of(diff),
        "lora_scale": scale,
        "embed_threshold": float(meta.get("embed_threshold", 0.0)),
        "modules_folded": len(plans),
        "embed_rows_written": int(arrays["embed_rows"].shape[0]),
        "weight_targets": len(keys),
        "method": "W += lora_scale * (B @ A); llm.embed_tokens.weight[embed_idx] = embed_rows",
        "verification": "; ".join(notes),
        "licences": {"base": "Apache-2.0", "fine_tune": "CC-BY-NC-4.0"},
    }
    marker.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {marker}")
    print("select it with:  --engine omnivoice_ipa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
