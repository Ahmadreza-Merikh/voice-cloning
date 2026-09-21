import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/ahmadrezamerikh/Desktop/Voice Cloning/src")

from voiceclone.audio.io import load_audio
from voiceclone.config import AppConfig
from voiceclone.device import get_device_spec
from voiceclone.engines.asr import ASREngine
from voiceclone.evaluation.similarity import character_error_rate, word_error_rate

ROOT = Path("/Users/ahmadrezamerikh/Desktop/Voice Cloning")
entries = json.loads((ROOT / "eval/corpus/manifest.json").read_text(encoding="utf-8"))["entries"]
clips = [e for e in entries if e.get("transcript")][:12]

results = {}
for model in ("large-v3-turbo", "large-v3"):
    config = AppConfig.load().copy_with(**{"engines.asr.model": model})
    engine = ASREngine(config, get_device_spec())
    rows = []
    started = time.time()
    for entry in clips:
        audio = load_audio(ROOT / "eval/corpus" / entry["file"], mono=True)
        heard = engine.transcribe(audio).text
        rows.append({
            "file": entry["file"],
            "seconds": round(audio.duration, 1),
            "cer": round(character_error_rate(entry["transcript"], heard) or -1, 4),
            "wer": round(word_error_rate(entry["transcript"], heard) or -1, 4),
            "reference_words": len(entry["transcript"].split()),
            "heard_words": len(heard.split()),
            "heard": heard,
        })
    engine.unload()
    good = [r for r in rows if r["cer"] >= 0]
    results[model] = {
        "seconds": round(time.time() - started, 1),
        "cer_mean": round(sum(r["cer"] for r in good) / len(good), 4),
        "wer_mean": round(sum(r["wer"] for r in good) / len(good), 4),
        "words_dropped": sum(max(0, r["reference_words"] - r["heard_words"]) for r in good),
        "worst": sorted(good, key=lambda r: -r["cer"])[:3],
        "rows": rows,
    }
    print(f"{model}: CER {results[model]['cer_mean']:.4f} WER {results[model]['wer_mean']:.4f} "
          f"words short {results[model]['words_dropped']} in {results[model]['seconds']}s", flush=True)

out = Path("/private/tmp/claude-501/-Users-ahmadrezamerikh/7399d521-4d73-4ca8-a72e-90625f919bff/scratchpad/asr_probe_models.json")
out.write_text(json.dumps({"clips": len(clips), "model": AppConfig.load().engines.asr.model, "results": results},
                          indent=1, ensure_ascii=False), encoding="utf-8")
for name, data in results.items():
    print("\n==", name)
    for row in data["worst"]:
        print(f"  {row['file']} CER {row['cer']:.3f} words {row['reference_words']}->{row['heard_words']}")
        print("    heard:", row["heard"][:100])
