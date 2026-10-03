#!/usr/bin/env python3
"""Prepare the A/B/C renders for a blind pronunciation review. It collects nothing.

The A/B/C run (``eval/runs/omnivoice-ipa-ab-v3/manifest.json``) rendered hard
Persian sentences three ways: OmniVoice on plain script (A), on a diacritised
twin (B, where one was written) and the IPA fine-tune on canonical IPA (C).
Nobody has listened. This writes what a listener needs to mark, word by word,
what each render got wrong - and nothing that says which render is which:

* per sentence, the renders take slots 1-3 in an order drawn from a fixed seed,
  and the audio files get anonymous names; which condition sits in which slot
  is written to a key outside the material a listener receives
* every render of a sentence is matched to one loudness by gain alone, because
  louder is heard as clearer
* the page shows the plain sentence only, never a diacritised twin as "the
  correct reading": some twins are wrong (``WRONG_IN_TWINS``)

    .venv/bin/python scripts/build_pronunciation_review.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "eval" / "runs" / "omnivoice-ipa-ab-v3" / "manifest.json"
OUT = ROOT / "listening" / "pronunciation-review-v1"
KEY = ROOT / "eval" / "runs" / "evidence" / "pronunciation-review-v1" / "key.json"
TARGET_LUFS = -26.0
SEED = 20261004

#: Diacritics the twins got wrong, found before anyone listened (2026-10-03).
#: Conditions B and C read the twins, so an error on these words there may be
#: the input's, not the model's.
WRONG_IN_TWINS = {
    "t11": "بارانِ بَند آمد: no ezafe on باران (the rain stopped)",
    "s08": "حِرفا should be حَرفا; خُریدار should be خَریدار",
    "t10": "اِمانت should be اَمانت",
}


def words_of(text: str) -> list[str]:
    """The sentence's words as a listener sees them, punctuation dropped."""
    return [w for w in (re.sub(r"[.,،؛:!؟?«»()\"]", "", token) for token in text.split()) if w]


def build(manifest_path: Path = MANIFEST, out: Path = OUT, key_path: Path = KEY, seed: int = SEED) -> dict:
    from voiceclone.audio.io import load_audio, save_audio
    from voiceclone.audio.loudness import apply_gain, measure_lufs, peak_dbfs

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    renders = {}
    for render in manifest["renders"]:
        if render.get("skipped") or not render.get("wav"):
            continue
        path = Path(render["wav"])
        renders.setdefault(render["id"], {})[render["condition"]] = path if path.is_absolute() else ROOT / path
    rng = random.Random(seed)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def anonymous(tag: str) -> str:
        return hashlib.sha1(f"{seed}:{tag}".encode()).hexdigest()[:12]

    items, key = [], {"seed": seed, "manifest": str(manifest_path.relative_to(ROOT)), "target_lufs": TARGET_LUFS,
                      "wrong_in_twins": WRONG_IN_TWINS, "sentences": {}}
    # Balanced, not just shuffled: sentences with the same conditions cycle
    # through every order of them, so no condition keeps one slot.
    cycles: dict[tuple, list[list[str]]] = {}
    orders = {}
    for sentence in manifest["sentences"]:
        conditions = tuple(sorted(renders.get(sentence["id"], {})))
        if len(conditions) < 2:
            continue
        if conditions not in cycles:
            from itertools import permutations
            cycle = [list(p) for p in permutations(conditions)]
            rng.shuffle(cycle)
            cycles[conditions] = cycle
        used = sum(1 for o in orders.values() if tuple(sorted(o)) == conditions)
        orders[sentence["id"]] = cycles[conditions][used % len(cycles[conditions])]
    for sentence in manifest["sentences"]:
        if sentence["id"] not in orders:
            continue
        order = orders[sentence["id"]]
        loaded = {c: load_audio(renders[sentence["id"]][c], mono=True) for c in order}
        loudness = {c: measure_lufs(a) for c, a in loaded.items()}
        target = TARGET_LUFS
        for c, a in loaded.items():
            if loudness[c] is not None:
                headroom = peak_dbfs(a) + (target - loudness[c])
                if headroom > -0.1:
                    target -= headroom + 0.1
        slots, gains = [], {}
        for slot, condition in enumerate(order, start=1):
            name = f"{anonymous(sentence['id'] + ':' + condition)}.wav"
            gain = 0.0 if loudness[condition] is None else target - loudness[condition]
            gains[str(slot)] = round(gain, 2)
            save_audio(apply_gain(loaded[condition], gain), audio_dir / name, bit_depth=16)
            slots.append({"slot": slot, "audio": f"audio/{name}"})
        items.append({"id": sentence["id"], "text": sentence["text"], "words": words_of(sentence["text"]),
                      "slots": slots})
        key["sentences"][sentence["id"]] = {"category": sentence["category"],
                                            "slots": {str(i): c for i, c in enumerate(order, start=1)},
                                            "gains_db": gains, "target_lufs": round(target, 2)}
    rng.shuffle(items)
    key["order"] = [item["id"] for item in items]
    key["fingerprint"] = hashlib.sha1(json.dumps(key["sentences"], sort_keys=True).encode()).hexdigest()[:16]
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    key = build(seed=args.seed)
    slots = sum(len(s["slots"]) for s in key["sentences"].values())
    print(f"{len(key['sentences'])} sentences, {slots} renders in {OUT.relative_to(ROOT)}; key at {KEY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
