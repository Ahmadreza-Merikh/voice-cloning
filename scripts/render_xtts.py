#!/usr/bin/env python3
"""Render the XTTS check's X form with ParsVoice-XTTS on this Mac. It runs in the XTTS environment.

Registered before it ran (eval/runs/persian-voice-xtts-check-v1.json). The
weights are ParsVoice-XTTS at 05784cc4, fetched with the user's token and not
re-hosted. The published file is a training checkpoint (5.7 GB, with its
optimizer), so it is read memory-mapped, tensors only, and its model saved
alone once. Everything else is the model's own notebook: the XTTS fork it was
trained with (8e59ec37), the reference prepared as 24 kHz mono at -20 dBFS
RMS, Persian punctuation mapped, the sentence split on . ! ?, and its
inference settings. On CPU, seed 42 before every piece. It will not start with
less than 6 GB available - the 2.1 GB model has to fit above the floor - and
it stops if available memory falls under 3.4 GB or pressure reaches warning.

    ~/.cache/persian-voice-cloning/xtts/.venv/bin/python scripts/render_xtts.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home() / ".cache" / "persian-voice-cloning" / "xtts"
FORK = HOME / "fork"
FORK_COMMIT = "8e59ec37e8c9cf9343503ca7430f5b23729f6389"
MODEL_DIR = HOME / "ParsVoice-XTTS"
REVISION = "05784cc45b83c2d1cd70bd51e779454e2787af6e"
WEIGHTS = MODEL_DIR / "model-weights-only.pth"
TEXTS = ROOT / "eval" / "runs" / "xtts-check-v1" / "texts.json"
OUT = ROOT / "eval" / "runs" / "xtts-check-v1" / "X"
REFERENCE = ROOT / "eval" / "corpus" / "mana_000.wav"
SEED = 42
START_GB, STOP_GB = 6.0, 3.4
SETTINGS = {"temperature": 0.1, "length_penalty": 1.0, "repetition_penalty": 10.0, "top_k": 10, "top_p": 0.3}
PUNCTUATION = {"؟": "?", "،": ",", "؛": ";", "«": '"', "»": '"', "٬": ",", "ـ": "-", "…": "..."}

spec = importlib.util.spec_from_file_location("memory", ROOT / "src" / "voiceclone" / "memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


def watch() -> None:
    while True:
        level, free = memory.pressure_level(), memory.available_gb()
        if (level is not None and level >= memory.WARNING) or (free is not None and free < STOP_GB):
            print(f"stopped: {free:.2f} GB available, pressure {memory.describe(level)}", flush=True)
            os._exit(3)
        time.sleep(0.3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_only(torch) -> dict:
    """The training checkpoint's model, saved alone; read memory-mapped so the optimizer never loads."""
    record_path = MODEL_DIR / "weights-only.json"
    if WEIGHTS.is_file() and record_path.is_file():
        return json.loads(record_path.read_text())
    checkpoint = torch.load(MODEL_DIR / "model.pth", map_location="cpu", mmap=True, weights_only=True)
    keys = sorted(checkpoint)
    torch.save({"model": checkpoint["model"]}, WEIGHTS)
    record = {"from": f"MohammadJRanjbar/ParsVoice-XTTS@{REVISION} model.pth", "checkpoint_keys": keys,
              "model_pth_sha256": sha256(MODEL_DIR / "model.pth"), "weights_sha256": sha256(WEIGHTS),
              "weights_bytes": WEIGHTS.stat().st_size}
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def reference(torch, torchaudio, path: Path) -> Path:
    import torchaudio.functional as F

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = F.resample(wav, orig_freq=sr, new_freq=24000)
    wav = wav * (10 ** (-20 / 20) / (wav.pow(2).mean().sqrt() + 1e-9))
    if wav.abs().max() > 1.0:
        wav = wav / wav.abs().max()
    out = OUT / "reference_normalized.wav"
    torchaudio.save(str(out), wav, 24000)
    return out


def main() -> int:
    free = memory.available_gb()
    if free is not None and free < START_GB:
        raise SystemExit(f"{free:.1f} GB available; XTTS needs {START_GB} to start here. Close other apps first.")
    threading.Thread(target=watch, daemon=True).start()
    sys.path.insert(0, str(FORK))
    import soundfile as sf
    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wavs").mkdir(exist_ok=True)
    record = weights_only(torch)
    config = XttsConfig()
    config.load_json(str(MODEL_DIR / "config.json"))
    started = time.time()
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(WEIGHTS), vocab_path=str(MODEL_DIR / "vocab.json"),
                          use_deepspeed=False)
    model.eval()
    load_seconds = round(time.time() - started, 1)
    print(f"ParsVoice-XTTS loaded on cpu in {load_seconds}s; {memory.available_gb():.2f} GB available", flush=True)
    ref = reference(torch, torchaudio, REFERENCE)
    latent, speaker = model.get_conditioning_latents(audio_path=str(ref), gpt_cond_len=model.config.gpt_cond_len,
                                                     max_ref_length=model.config.max_ref_len,
                                                     sound_norm_refs=model.config.sound_norm_refs)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "eval" / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in corpus["entries"] if e["file"] == REFERENCE.name)
    renders = []
    for s in texts["sentences"]:
        text = s["texts"]["X"]
        for fa_char, replacement in PUNCTUATION.items():
            text = text.replace(fa_char, replacement)
        pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
        started = time.time()
        chunks = []
        with torch.inference_mode():
            for piece in pieces:
                torch.manual_seed(SEED)
                result = model.inference(text=piece, language="fa", gpt_cond_latent=latent,
                                         speaker_embedding=speaker, **SETTINGS)
                chunks.append(torch.as_tensor(result["wav"]))
        audio = torch.cat(chunks).numpy()
        wav = OUT / "wavs" / f"{s['id']}_X.wav"
        sf.write(str(wav), audio, 24000, subtype="PCM_24")
        seconds = round(time.time() - started, 1)
        renders.append({"id": s["id"], "condition": "X", "engine": "parsvoice-xtts", "input_text": s["texts"]["X"],
                        "fed_texts": pieces, "seed": SEED, "reference": REFERENCE.name,
                        "wav": str(wav.relative_to(ROOT)), "duration_s": round(len(audio) / 24000, 3),
                        "timings": {"synthesis": seconds}})
        print(f"  {s['id']} {len(audio) / 24000:5.1f}s audio in {seconds}s; {memory.available_gb():.2f} GB available",
              flush=True)
    manifest = {"version": 1, "purpose": "the XTTS check's X form: ParsVoice-XTTS on this Mac",
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "texts": {"file": str(TEXTS.relative_to(ROOT)), "sentences_sha256": texts["sentences_sha256"]},
                "settings": {"model": "MohammadJRanjbar/ParsVoice-XTTS", "revision": REVISION, "weights": record,
                             "fork": f"nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages@{FORK_COMMIT}",
                             "device": "cpu", "seed": SEED, "language": "fa", **SETTINGS,
                             "torch": torch.__version__, "load_seconds": load_seconds},
                "reference": {"speaker": "mana", "file": REFERENCE.name, "transcript": entry["transcript"],
                              "prepared": "24 kHz mono, -20 dBFS RMS, as the model's notebook"},
                "conditions": {"X": texts["conditions"]["X"]},
                "renders": renders}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(renders)} renders -> {(OUT / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
