#!/usr/bin/env python3
"""Prepare a blind A/B listening test between two systems. It collects nothing.

Similarity and CER are the only instruments this project has, and neither is a
listener. This builds the material for the one kind of evidence the roadmap
says is missing - and stops there: no response is invented, and nothing is
scored until real responses exist (``scripts/score_listening_test.py``).

What makes it blind and fair:

* **no system names reach the listener.** Every clip is copied to a random id;
  which system made which clip lives only in a key file written *outside* the
  folder a listener is given
* **order is randomised per trial** from a fixed seed, and balanced: each
  system is "sample 1" in half the trials, give or take one
* **loudness is matched by pure gain** - no limiter, no compression - to one
  target for every clip, including the reference, because louder is heard as
  better. If any clip would clip at that target, the target is lowered for all
* **the reference is a real recording of the speaker that neither system was
  conditioned on** - a held-out clip, never the prompt - so "same person?" asks
  about identity, not about copying the prompt's recording

Three questions, kept separate because they are separate: which sample sounds
more like the person in the reference; which sounds more natural; and whether
either one misread or garbled words (the intended text is shown for that one).

    .venv/bin/python scripts/build_listening_test.py --name omnivoice-vs-moss15 \\
        --a-dir path/to/omnivoice/renders --b-dir path/to/moss15/renders
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "eval" / "corpus"
OUT = ROOT / "listening"
TARGET_LUFS = -26.0


def sentences() -> dict[str, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark_texts", ROOT / "scripts" / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_texts"] = module
    spec.loader.exec_module(module)
    return dict(module.SENTENCES)


def held_out_reference(speaker: str, manifest: dict) -> dict:
    """The first held-out clip: a real recording the systems never conditioned on."""
    clips = sorted((e for e in manifest["entries"] if e["speaker"] == speaker and e["role"] == "held_out"),
                   key=lambda e: e["file"])
    if not clips:
        raise SystemExit(f"{speaker} has no held-out clip")
    return clips[0]


def build(name: str, a_dir: Path, b_dir: Path, seed: int, out_root: Path = OUT) -> dict:
    from voiceclone.audio.io import load_audio, save_audio
    from voiceclone.audio.loudness import apply_gain, measure_lufs, peak_dbfs

    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    texts = sentences()
    shared = sorted({p.name for p in a_dir.glob("*.wav")} & {p.name for p in b_dir.glob("*.wav")})
    if not shared:
        raise SystemExit("the two render folders have no file names in common")

    rng = random.Random(seed)
    items = []
    for filename in shared:
        speaker, sentence, _ = filename[:-4].split("_", 2)
        items.append({"speaker": speaker, "sentence": sentence, "file": filename,
                      "reference": held_out_reference(speaker, manifest)})
    rng.shuffle(items)
    a_first = [i < len(items) // 2 + (len(items) % 2 and rng.random() < 0.5) for i in range(len(items))]
    rng.shuffle(a_first)

    served = out_root / name
    audio_dir = served / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def anonymous(tag: str) -> str:
        return hashlib.sha1(f"{seed}:{name}:{tag}".encode()).hexdigest()[:12]

    sources = []
    for item, first in zip(items, a_first, strict=True):
        pair = [("a", a_dir / item["file"]), ("b", b_dir / item["file"])]
        if not first:
            pair.reverse()
        trial_id = anonymous(f"trial:{item['file']}")
        item.update({"id": trial_id, "order": [system for system, _ in pair]})
        sources.append((f"{trial_id}-ref", CORPUS / item["reference"]["file"]))
        for position, (_, path) in enumerate(pair, start=1):
            sources.append((f"{trial_id}-{position}", path))

    loaded = {tag: load_audio(path, mono=True) for tag, path in sources}
    loudness = {tag: measure_lufs(audio) for tag, audio in loaded.items()}
    target = TARGET_LUFS
    for tag, audio in loaded.items():
        if loudness[tag] is not None:
            headroom = peak_dbfs(audio) + (target - loudness[tag])
            if headroom > -0.1:
                target -= headroom + 0.1
    gains = {}
    for tag, audio in loaded.items():
        gain = 0.0 if loudness[tag] is None else target - loudness[tag]
        gains[tag] = round(gain, 2)
        save_audio(apply_gain(audio, gain), audio_dir / f"{anonymous(tag)}.wav", bit_depth=32)

    trials = [{"id": item["id"], "text": texts[item["sentence"]],
               "reference": f"audio/{anonymous(item['id'] + '-ref')}.wav",
               "sample_1": f"audio/{anonymous(item['id'] + '-1')}.wav",
               "sample_2": f"audio/{anonymous(item['id'] + '-2')}.wav"} for item in items]
    (served / "index.html").write_text(page(name, trials), encoding="utf-8")

    key = {
        "name": name, "seed": seed, "target_lufs": round(target, 2),
        "systems": {"a": str(a_dir), "b": str(b_dir)},
        "trials": {item["id"]: {"speaker": item["speaker"], "sentence": item["sentence"],
                                "file": item["file"], "reference": item["reference"]["file"],
                                "sample_1": item["order"][0], "sample_2": item["order"][1],
                                "gains_db": {k: gains[f"{item['id']}-{k}"] for k in ("ref", "1", "2")}}
                   for item in items},
    }
    (out_root / f"{name}.key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    return key


def page(name: str, trials: list[dict]) -> str:
    rows = []
    for number, trial in enumerate(trials, start=1):
        tid = html.escape(trial["id"])
        rows.append(f"""
<section class="trial" data-id="{tid}">
  <h2>{number} / {len(trials)}</h2>
  <p>مرجع (صدای واقعی گوینده): <audio controls preload="none" src="{trial['reference']}"></audio></p>
  <p>نمونهٔ ۱: <audio controls preload="none" src="{trial['sample_1']}"></audio></p>
  <p>نمونهٔ ۲: <audio controls preload="none" src="{trial['sample_2']}"></audio></p>
  <p class="text">متن درست: {html.escape(trial['text'])}</p>
  <fieldset><legend>کدام نمونه بیشتر شبیه همان آدمِ مرجع است؟</legend>
    <label><input type="radio" name="{tid}-same" value="1">۱</label>
    <label><input type="radio" name="{tid}-same" value="2">۲</label>
    <label><input type="radio" name="{tid}-same" value="0">فرقی ندارد</label></fieldset>
  <fieldset><legend>کدام طبیعی‌تر است؟</legend>
    <label><input type="radio" name="{tid}-natural" value="1">۱</label>
    <label><input type="radio" name="{tid}-natural" value="2">۲</label>
    <label><input type="radio" name="{tid}-natural" value="0">فرقی ندارد</label></fieldset>
  <fieldset><legend>کدام کلمه‌ای را اشتباه یا نامفهوم گفت؟</legend>
    <label><input type="checkbox" name="{tid}-misread" value="1">۱</label>
    <label><input type="checkbox" name="{tid}-misread" value="2">۲</label></fieldset>
</section>""")
    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>آزمون شنیداری</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px}}
.trial{{border-top:1px solid #ccc;padding:12px 0}} .text{{color:#555}} fieldset{{margin:6px 0}}</style>
</head><body>
<h1>آزمون شنیداری</h1>
<p>هر بار یک صدای واقعی و دو نمونهٔ ساختگی می‌شنوید. با هدفون گوش کنید. جواب‌ها روی همین مرورگر ذخیره
می‌شوند؛ در پایان دکمهٔ «ذخیرهٔ پاسخ‌ها» را بزنید و فایل را بفرستید.</p>
<p>نام یا کد شما: <input id="listener" autocomplete="off"></p>
{"".join(rows)}
<p><button id="save">ذخیرهٔ پاسخ‌ها</button></p>
<script>
const KEY = "listening:{html.escape(name)}";
function collect() {{
  const out = {{listener: document.getElementById("listener").value, answers: {{}}}};
  document.querySelectorAll(".trial").forEach(t => {{
    const id = t.dataset.id;
    const pick = q => (t.querySelector(`input[name="${{id}}-${{q}}"]:checked`) || {{}}).value || null;
    const misread = [...t.querySelectorAll(`input[name="${{id}}-misread"]:checked`)].map(i => i.value);
    out.answers[id] = {{same: pick("same"), natural: pick("natural"), misread}};
  }});
  return out;
}}
document.addEventListener("change", () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(collect())); }} catch (e) {{}} }});
document.getElementById("save").onclick = () => {{
  const blob = new Blob([JSON.stringify(collect(), null, 1)], {{type: "application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "responses.json"; link.click();
}};
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    key = build(args.name, args.a_dir, args.b_dir, args.seed)
    print(f"{len(key['trials'])} trials in listening/{args.name}/ ; key kept at listening/{args.name}.key.json")
    print("Give listeners only the folder. The key must not travel with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
