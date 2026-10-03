"""hazm's ParsBERT part-of-speech tagger, run without spaCy.

The text-only comparison (FINDINGS section 14) chose this tagger to place the
ezafe, and the listening test (section 15) found OmniVoice reads better with
its ezafes written in. As published it is a spaCy pipeline: loading that
imports spaCy and spacy-transformers and peaked at 4.3 GB for 13 s on the M1
Pro (2026-10-03), too much to run beside a loaded OmniVoice on a 16 GB Mac.

This runs the same weights - ParsBERT and the tagger's output layer, read out
of the pipeline's own files once (:func:`export`) - with transformers alone,
doing what spacy-transformers does for one sentence:

* the sentence's tokens joined by single spaces, cut into strided spans of 128
  tokens with a stride of 96 (one span below 96 tokens)
* each span through ParsBERT's fast tokenizer; a wordpiece belongs to every
  token whose characters - the space after it included - it covers
* each token the mean of its wordpieces' last hidden states (zero if it has
  none), then the output layer; the tag is the highest-scoring label

Checked tag for tag against the spaCy pipeline on the sentences it was
measured on (``eval/runs/persian-voice-ezafe-tagger-v1.json``).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

REPO = "roshan-research/hazm-parsbert-postagger"
REVISION = "eba1d2b30cc48206c0144b36d2b663652d44faf9"
#: spacy-transformers.strided_spans.v1 in the pipeline's config.cfg.
WINDOW = 128
STRIDE = 96
EXPORT_VERSION = 1
FILES = ("config.json", "model.safetensors", "tokenizer.json", "head.safetensors")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(*, download: bool = True) -> Path:
    """The pinned pipeline in the Hugging Face cache, fetched once if allowed."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(REPO, revision=REVISION, local_files_only=True))
    except Exception:
        if not download:
            raise
        return Path(snapshot_download(REPO, revision=REVISION))


def _array(entry: dict):
    import numpy as np

    return np.frombuffer(entry[b"data"], dtype=entry[b"type"]).reshape(entry[b"shape"]).copy()


def export(source: Path, out: Path) -> dict:
    """Read ParsBERT and the output layer out of the spaCy pipeline's files.

    ``transformer/model`` is a thinc model whose PyTorch shim holds the
    Hugging Face config, the tokenizer files and the state dict; ``tagger/model``
    holds the output layer's weights. Nothing is converted: the same tensors
    are written as safetensors.
    """
    import msgpack
    import torch
    from safetensors.torch import save_file
    from transformers import BertConfig

    raw = (source / "transformer" / "model").read_bytes()
    shim = msgpack.unpackb(raw, raw=False, strict_map_key=False)["shims"][1][0]
    del raw
    inner = msgpack.unpackb(shim, raw=False, strict_map_key=False)
    del shim
    state = torch.load(io.BytesIO(inner.pop("state")), map_location="cpu", weights_only=True)
    out.mkdir(parents=True, exist_ok=True)
    BertConfig.from_dict(inner["config"]).save_pretrained(out)
    (out / "tokenizer.json").write_bytes(inner["tokenizer"]["tokenizer.json"])
    save_file({k: v.contiguous() for k, v in state.items()}, str(out / "model.safetensors"))
    del state, inner

    tagger = msgpack.unpackb((source / "tagger" / "model").read_bytes(), raw=False, strict_map_key=False)
    names = [node["name"] for node in tagger["nodes"]]
    layer = names.index("softmax")
    settings = {k: msgpack.unpackb(v) for k, v in tagger["attrs"][layer].items()}
    if settings != {"softmax_normalize": False, "softmax_temperature": 1.0}:
        raise ValueError(f"unexpected output layer settings: {settings}")
    params = tagger["params"][layer]
    save_file({"W": torch.from_numpy(_array(params["W"])), "b": torch.from_numpy(_array(params["b"]))},
              str(out / "head.safetensors"))
    labels = json.loads((source / "tagger" / "cfg").read_text(encoding="utf-8"))["labels"]
    record = {"version": EXPORT_VERSION, "source": {"repo": REPO, "revision": REVISION},
              "labels": labels, "files": {name: _sha256(out / name) for name in FILES}}
    (out / "export.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def exported(out: Path) -> bool:
    """Whether ``out`` holds a complete export of the pinned revision."""
    try:
        record = json.loads((out / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (record.get("version") == EXPORT_VERSION and record.get("source", {}).get("revision") == REVISION
            and all((out / name).is_file() for name in FILES))


def strided_spans(n: int, window: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    """spacy-transformers' strided spans over ``n`` tokens, as (start, end) pairs."""
    spans: list[tuple[int, int]] = []
    start = 0
    for _ in range(n // stride):
        spans.append((start, min(start + window, n)))
        if start + window >= n:
            break
        start += stride
    else:
        if start < n:
            spans.append((start, n))
    return spans


def align(tokens: list[str], offsets: list[tuple[int, int]]) -> list[list[int]]:
    """The wordpieces of each token in one span, as spacy-transformers aligns them.

    Each token owns its characters and the one after it (the space), and a
    wordpiece belongs to every token owning one of the characters it covers.
    """
    text_length = sum(len(t) for t in tokens) + max(len(tokens) - 1, 0)
    owner = [-1] * (text_length + 1)
    position = 0
    for i, token in enumerate(tokens):
        for c in range(position, position + len(token) + 1):
            owner[c] = i
        position += len(token) + 1
    found: list[set[int]] = [set() for _ in tokens]
    for j, (start, end) in enumerate(offsets):
        for c in range(start, end):
            if 0 <= c < len(owner) and owner[c] >= 0:
                found[owner[c]].add(j)
    return [sorted(s) for s in found]


class ParsBertTagger:
    """The exported tagger: tokens in, hazm's tags out (``NOUN,EZ`` and so on)."""

    def __init__(self, export_dir: Path, *, device: str = "cpu") -> None:
        import torch
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import BertModel

        record = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
        self.labels: list[str] = record["labels"]
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(str(export_dir / "tokenizer.json"))
        # transformers calls it with neither: a long span is never cut.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        from transformers.utils import logging as hf_logging

        bars = hf_logging.is_progress_bar_enabled()
        hf_logging.disable_progress_bar()
        try:
            self.model = BertModel.from_pretrained(str(export_dir)).to(self.device).eval()
        finally:
            if bars:
                hf_logging.enable_progress_bar()
        head = load_file(str(export_dir / "head.safetensors"))
        self.W = head["W"].to(self.device)
        self.b = head["b"].to(self.device)

    def tag(self, tokens: list[str]) -> list[str]:
        import torch

        if not tokens:
            return []
        width = self.W.shape[1]
        pooled: list[list] = [[] for _ in tokens]
        with torch.inference_mode():
            for start, end in strided_spans(len(tokens)):
                piece = tokens[start:end]
                encoding = self.tokenizer.encode(" ".join(piece))
                ids = torch.tensor([encoding.ids], device=self.device)
                hidden = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    token_type_ids=torch.tensor([encoding.type_ids], device=self.device)
                                    ).last_hidden_state[0]
                for i, js in enumerate(align(piece, encoding.offsets)):
                    pooled[start + i].extend(hidden[j] for j in js)
            vectors = torch.stack([torch.stack(v).mean(0) if v else torch.zeros(width, device=self.device)
                                   for v in pooled])
            scores = vectors @ self.W.T + self.b
        return [self.labels[i] for i in scores.argmax(1).tolist()]


__all__ = ["REPO", "REVISION", "ParsBertTagger", "align", "export", "exported", "snapshot", "strided_spans"]
