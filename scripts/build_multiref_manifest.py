#!/usr/bin/env python3
"""Fix the multi-reference experiment in a file before anything is rendered.

The question is whether conditioning on clips from two different recordings
changes held-out similarity. Three things make a naive version of it
worthless, and this manifest exists to rule them out in advance:

* **leakage** - a clip used as a prompt must never be scored against. Held-out
  similarity against the prompt is the channel-inflated number (0.11 to 0.16
  higher) this whole benchmark was built to avoid
* **a moving target** - every condition for a speaker is scored against the
  *same* evaluation clips, or differences between conditions are partly
  differences between evaluation sets
* **budget** - a two-clip prompt that is simply longer is a different
  experiment from a two-clip prompt of the same length. Both are allowed, under
  different names, and never compared as if they were one

What the current corpus can and cannot support
----------------------------------------------
Each speaker has five clips, and only whole clips have exact transcripts. The
equal-*speech* design - trim single and multi prompts to the same seconds of
speech - needs cuts inside utterances, and an energy VAD over the prompt clips
finds few internal pauses (none at all in two of them), so the cuts would land
mid-word and the trimmed audio would have no transcript. That design is
therefore not built here. It is a requirement for the next corpus: speakers
with several short, separately transcribed recordings.

What is built instead uses whole clips only:

* ``single-A`` - the speaker's reference-role clip, whole
* ``multi-YZ`` - two *other* clips, chosen so their combined length is the
  closest available to A's (within TOLERANCE), joined by GAP seconds of silence
* ``single-Y`` - the first of those two clips alone

and the evaluation pool is the two remaining clips, shared by all three.
Each contrast carries one known confound, and the manifest names it:

* multi-YZ vs single-A - about the same length; *different source clips*
* multi-YZ vs single-Y - the same first clip; *twice the length*

An improvement would be consistent with the channel hypothesis in
SOTA_ROADMAP.md, not proof of it: a more varied prompt can help for reasons
that have nothing to do with channel.

    .venv/bin/python scripts/build_multiref_manifest.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval" / "corpus"
OUT = ROOT / "eval" / "manifests" / "multiref_v1.json"

#: Multi-YZ may differ in total length from single-A by at most this fraction.
TOLERANCE = 0.15

#: Silence between the two clips of a multi prompt.
GAP = 0.3

#: Development texts - the benchmark's s1-s3, so nothing new is exposed.
TEXTS = ("s1", "s2", "s3")
SEEDS = (42, 43)


def engine_limits() -> dict[str, float | None]:
    """Each engine's *declared* prompt limit. None means none is declared,
    which is not the same as there being none."""
    sys.path.insert(0, str(ROOT / "src"))
    from voiceclone.engines import list_tts_engines

    return {name: cls.capabilities.max_reference_seconds for name, cls in sorted(list_tts_engines().items())}


def build() -> dict:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    by_speaker: dict[str, list[dict]] = {}
    for entry in manifest["entries"]:
        by_speaker.setdefault(entry["speaker"], []).append(entry)

    limits = engine_limits()
    speakers: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for speaker, entries in sorted(by_speaker.items()):
        reference = [e for e in entries if e["role"] == "reference"]
        others = sorted((e for e in entries if e["role"] != "reference"), key=lambda e: e["file"])
        if len(reference) != 1 or len(others) < 4:
            excluded[speaker] = f"needs 1 reference and 4 other clips, has {len(reference)} and {len(others)}"
            continue
        a = reference[0]
        best = None
        for y, z in combinations(others, 2):
            total = y["duration"] + GAP + z["duration"]
            mismatch = abs(total - a["duration"]) / a["duration"]
            key = (mismatch, y["file"], z["file"])
            if best is None or key < best[0]:
                best = (key, y, z, total)
        (mismatch, _, _), y, z, total = best
        if mismatch > TOLERANCE:
            excluded[speaker] = (f"closest pair of clips is {total:.2f}s against a {a['duration']:.2f}s "
                                 f"reference, {mismatch:.0%} apart (tolerance {TOLERANCE:.0%})")
            continue
        evaluation = [e["file"] for e in others if e["file"] not in (y["file"], z["file"])]

        def clip(e: dict) -> dict:
            return {"file": e["file"], "seconds": e["duration"], "transcript": e["transcript"]}

        conditions = {
            "single-A": {"clips": [clip(a)], "seconds": a["duration"],
                         "transcript": a["transcript"]},
            "multi-YZ": {"clips": [clip(y), clip(z)], "gap_seconds": GAP, "seconds": round(total, 3),
                         "transcript": f"{y['transcript']} {z['transcript']}"},
            "single-Y": {"clips": [clip(y)], "seconds": y["duration"], "transcript": y["transcript"]},
        }
        inapplicable = {
            engine: sorted(name for name, c in conditions.items() if c["seconds"] > limit)
            for engine, limit in limits.items()
            if limit is not None and any(c["seconds"] > limit for c in conditions.values())
        }
        speakers[speaker] = {
            "evaluation_clips": evaluation,
            "conditions": conditions,
            "length_mismatch_multi_vs_single_A": round(mismatch, 4),
            "inapplicable_for_engine": inapplicable,
        }

    return {
        "name": "multiref_v1",
        "built_by": "scripts/build_multiref_manifest.py",
        "set": "development - the nine benchmark speakers; never a test set",
        "texts": list(TEXTS),
        "seeds": list(SEEDS),
        "generation_budget": "each engine's configured defaults in configs/default.yaml at the "
                             "commit that renders; identical across conditions",
        "reference_transcripts": "the corpus transcripts of whole clips; a multi prompt uses "
                                 "the two joined with a space, in clip order",
        "contrasts": {
            "multi-YZ vs single-A": "about the same length; different source clips",
            "multi-YZ vs single-Y": "same first clip; about twice the length",
        },
        "rules": [
            "no clip in any condition's prompt is ever in that speaker's evaluation_clips",
            "every condition of a speaker is scored against the same evaluation_clips",
            "an engine is not run on a condition listed in inapplicable_for_engine; the "
            "prompt is never cut to fit, because that would be a different condition",
        ],
        "known_limits": [
            "equal speech budgets need cuts inside utterances, which the corpus cannot "
            "support with exact transcripts; lengths are matched on whole clips within "
            f"{TOLERANCE:.0%} instead",
            "two evaluation clips per speaker instead of the benchmark's four, so "
            "held-out scores here are noisier than the benchmark's and not comparable to them",
        ],
        "tolerance": TOLERANCE,
        "engine_prompt_limits_declared": limits,
        "speakers": speakers,
        "excluded": excluded,
    }


def main() -> int:
    manifest = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(manifest['speakers'])} speakers, "
          f"{len(manifest['excluded'])} excluded")
    for speaker, reason in manifest["excluded"].items():
        print(f"  excluded {speaker}: {reason}")
    for speaker, entry in manifest["speakers"].items():
        c = entry["conditions"]
        print(f"  {speaker}: A {c['single-A']['seconds']:.1f}s  YZ {c['multi-YZ']['seconds']:.1f}s "
              f"({entry['length_mismatch_multi_vs_single_A']:.0%})  Y {c['single-Y']['seconds']:.1f}s  "
              f"eval {entry['evaluation_clips']}"
              + (f"  not for {entry['inapplicable_for_engine']}" if entry["inapplicable_for_engine"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
