#!/usr/bin/env python3
"""Check that an adaptation export loads the way the published model does, on a tiny model.

The training sessions write a float16 copy of the weights every 1,000 steps
with ``export_float16`` (``scripts/omnivoice_adapt.py``), and the development
session loads each one through ``OmniVoice.from_pretrained``. The probe never
reached a save, so this checks the path before the first real export exists:
a 10.8M-parameter OmniVoice (the published config with a two-layer, 64-wide
backbone), a stand-in for the official checkpoint holding what the trainer's
``tokenizer.save_pretrained`` writes, the export, and a load of the export with
the published snapshot's audio tokenizer. It passes if every floating tensor
comes back equal to the float16 of the original, and both tokenizers load.

No training and no synthesis: one small model built, written, read back.
Run it with the OmniVoice worker's Python, which has the package installed:

    ~/.cache/persian-voice-cloning/omnivoice/.venv/bin/python scripts/check_omnivoice_export.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    import torch
    from huggingface_hub import snapshot_download
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("adapt", REPO / "scripts" / "omnivoice_adapt.py")
    adapt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapt)

    snapshot = Path(snapshot_download("k2-fsa/OmniVoice", revision="c5fdb5ccb189668d56333f77ba2629f4cd7535f4"))
    config = OmniVoiceConfig.from_pretrained(snapshot)
    small = config.llm_config
    small.hidden_size, small.intermediate_size, small.num_hidden_layers = 64, 128, 2
    small.num_attention_heads, small.num_key_value_heads, small.head_dim = 2, 1, 32
    if getattr(small, "layer_types", None):
        small.layer_types = small.layer_types[:2]
    torch.manual_seed(0)
    model = OmniVoice(config)

    work = Path(tempfile.mkdtemp())
    try:
        checkpoint = work / "checkpoint-1000"
        checkpoint.mkdir()
        AutoTokenizer.from_pretrained(snapshot).save_pretrained(checkpoint)
        destination = work / "export-1000-float16"
        adapt.export_float16(model, checkpoint, destination, snapshot / "audio_tokenizer")

        loaded = OmniVoice.from_pretrained(str(destination), dtype=torch.float16, device_map="cpu")
        original = {k: v.to(torch.float16) for k, v in model.state_dict().items() if v.is_floating_point()}
        state = loaded.state_dict()
        unequal = [k for k, v in original.items() if k not in state or not torch.equal(state[k].to(torch.float16), v)]
        result = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "export_files": sorted(p.name for p in destination.iterdir()),
            "published_files": sorted(p.name for p in snapshot.iterdir()
                                      if p.name != "README.md" and not p.name.startswith(".")),
            "text_tokenizer": type(loaded.text_tokenizer).__name__,
            "audio_tokenizer": type(loaded.audio_tokenizer).__name__,
            "tensors_compared": len(original),
            "tensors_unequal": unequal,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(result, indent=2))
    ok = (not unequal and result["export_files"] == result["published_files"]
          and loaded.text_tokenizer is not None and loaded.audio_tokenizer is not None)
    print("export loads like the published model" if ok else "EXPORT DOES NOT LOAD LIKE THE PUBLISHED MODEL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
