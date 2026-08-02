#!/usr/bin/env python3
"""Assemble a persistent, labelled evaluation corpus.

Why this exists
---------------
Every calibration number this project quotes - the 0.288 different-speaker
floor, the 0.647 same-speaker ceiling - came from audio that no longer exists
on disk. They cannot be re-derived, so they cannot be checked, and every
threshold built on them is an article of faith.

Worse, the only speaker in ``voices/`` is one person from one corpus. A
voice-cloning score measured on a single speaker is an anecdote: it cannot
distinguish "the engine clones well" from "the engine happens to suit this
voice".

This script fixes both. It writes a versioned corpus with real speaker labels,
so the floor, the ceiling and the clone score are all reproducible from a
single command and are averaged over many speakers.

Sources
-------
**Common Voice 26 Persian (Clean)** - ``pymmdrza/Common-Voice-Speech-26.0-
Persian-Clean``, ungated, 34 134 clips from 3 791 labelled speakers, already
resampled to 24 kHz mono. Crowdsourced, so the channel varies between speakers
the way it does in real use. Each clip is a separate recording, which is what
makes a held-out clip genuinely held out.

**Mana-TTS** - one studio speaker, read from the parquet already in the HF
cache. Clip ids look like ``102-61``: source recording 102, segment 61. Held-out
clips are taken from *different source recordings*, so they are not neighbouring
slices of one waveform.

Layout
------
``eval/corpus/<speaker>_<clip>.wav`` plus ``manifest.json``. The flat
``<speaker>_<clip>`` naming is what ``scripts/measure_ceiling.py`` already
groups on, so both tools read the same corpus.

The manifest carries the *ground-truth* transcript for every clip. That matters:
F5-TTS conditions on a transcript of the reference, and measuring with ASR
transcripts confounds the engine with the recogniser.

Usage
-----
    .venv/bin/python scripts/build_eval_corpus.py                # default 8 + Mana
    .venv/bin/python scripts/build_eval_corpus.py --speakers 12
    .venv/bin/python scripts/build_eval_corpus.py --skip-mana
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"
MANA_REPO = "MahtaFetrat/Mana-TTS"

#: A reference prompt shorter than this starves the cloner of timbre.
MIN_REFERENCE_SECONDS = 6.5
#: Held-out clips only need to be long enough for a stable speaker embedding.
MIN_HELD_OUT_SECONDS = 3.5
#: Clips per speaker: one reference, the rest held out.
DEFAULT_HELD_OUT = 4


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ----------------------------------------------------------------------------
# Common Voice
# ----------------------------------------------------------------------------
def _download_csv(name: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=CV_REPO, filename=name, repo_type="dataset", cache_dir=str(cache)
        )
    )


def _pick_cv_speakers(
    kept: list[dict[str, str]],
    genders: dict[str, str],
    wanted: int,
    held_out: int,
    present: set[str],
) -> list[tuple[str, str, list[dict[str, str]]]]:
    """Choose speakers that can supply one long reference plus held-out clips.

    Balanced by gender where the labels allow it, because a corpus that is all
    male would leave the floor measured on only half the voice space.
    """
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in kept:
        by_speaker[row["client_id"]].append(row)

    def usable(row: dict[str, str]) -> bool:
        # kept.csv lists every clip the cleaning pass accepted, but the uploader
        # published only about a third of them, so check the file exists first.
        if row["output_audio"] not in present:
            return False
        # Clipped or very quiet clips make a poor prompt and a noisy embedding.
        return float(row["clipping_ratio"]) <= 0.001 and float(row["rms_dbfs"]) >= -34.0

    candidates: list[tuple[float, str, list[dict[str, str]]]] = []
    for client_id, rows in by_speaker.items():
        clean = [r for r in rows if usable(r)]
        longest = sorted(clean, key=lambda r: -float(r["output_duration"]))
        if not longest or float(longest[0]["output_duration"]) < MIN_REFERENCE_SECONDS:
            continue
        rest = [r for r in longest[1:] if float(r["output_duration"]) >= MIN_HELD_OUT_SECONDS]
        if len(rest) < held_out:
            continue
        chosen = [longest[0], *rest[:held_out]]
        # Prefer speakers whose reference is long: more timbre for the cloner.
        candidates.append((float(longest[0]["output_duration"]), client_id, chosen))

    candidates.sort(key=lambda item: -item[0])

    picked: list[tuple[str, str, list[dict[str, str]]]] = []
    quota = {"female_feminine": wanted // 2, "male_masculine": wanted - wanted // 2}
    spare: list[tuple[str, str, list[dict[str, str]]]] = []
    for _, client_id, rows in candidates:
        gender = genders.get(client_id, "")
        entry = (client_id, gender or "unknown", rows)
        if quota.get(gender, 0) > 0:
            quota[gender] -= 1
            picked.append(entry)
        else:
            spare.append(entry)
        if len(picked) == wanted:
            break

    # Persian Common Voice is heavily male; top up rather than return short.
    for entry in spare:
        if len(picked) >= wanted:
            break
        picked.append(entry)
    return picked


def build_common_voice(
    out_dir: Path, cache: Path, wanted: int, held_out: int
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    _rule(f"Common Voice 26 Persian (Clean) - {wanted} speakers x {held_out + 1} clips")

    kept_path = _download_csv("reports/kept.csv", cache)
    speakers_path = _download_csv("reports/speakers.csv", cache)
    with kept_path.open(encoding="utf-8") as handle:
        kept = list(csv.DictReader(handle))
    with speakers_path.open(encoding="utf-8") as handle:
        genders = {r["client_id"]: r.get("gender", "") for r in csv.DictReader(handle)}

    present = {f for f in list_repo_files(CV_REPO, repo_type="dataset") if f.startswith("wavs/")}
    print(
        f"{len(kept)} clips in the manifest, {len(present)} actually published, "
        f"{len({r['client_id'] for r in kept})} labelled speakers"
    )

    picked = _pick_cv_speakers(kept, genders, wanted, held_out, present)
    if len(picked) < wanted:
        print(f"  only {len(picked)} speakers met the criteria", file=sys.stderr)

    entries: list[dict[str, Any]] = []
    for index, (client_id, gender, rows) in enumerate(picked, start=1):
        speaker = f"cv{index:02d}"
        print(f"  {speaker}  {gender:15s} {len(rows)} clips  ref {float(rows[0]['output_duration']):.1f}s")
        for position, row in enumerate(rows):
            source = hf_hub_download(
                repo_id=CV_REPO,
                filename=row["output_audio"],
                repo_type="dataset",
                cache_dir=str(cache),
            )
            role = "reference" if position == 0 else "held_out"
            name = f"{speaker}_{position:03d}.wav"
            shutil.copyfile(source, out_dir / name)
            entries.append(
                {
                    "speaker": speaker,
                    "file": name,
                    "role": role,
                    "transcript": row["text"].strip(),
                    "duration": round(float(row["output_duration"]), 3),
                    "gender": gender,
                    "corpus": "common_voice_26_fa_clean",
                    # Each Common Voice clip is its own recording, so the clip id
                    # is the session id.
                    "session": Path(row["output_audio"]).stem,
                    "speaker_hash": client_id[:16],
                }
            )
    return entries


# ----------------------------------------------------------------------------
# Mana-TTS
# ----------------------------------------------------------------------------
def build_mana(out_dir: Path, held_out: int) -> list[dict[str, Any]]:
    """One studio speaker, clips drawn from distinct source recordings."""
    import glob
    import os

    import numpy as np
    import pyarrow.parquet as pq

    from voiceclone.audio.io import Audio, save_audio

    _rule("Mana-TTS - one studio speaker, clips from distinct source recordings")

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--MahtaFetrat--Mana-TTS/snapshots/*/dataset/*.parquet"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print("  Mana-TTS parquet not in the HF cache; skipping.", file=sys.stderr)
        return []

    # Two passes. The `audio` column is float64 PCM: reading all 1102 rows at
    # once costs a couple of GB, which is not something to do on a 16 GB laptop
    # that is already swapping. Pass one reads only the small columns to decide
    # what is wanted; pass two streams the audio back in small batches and keeps
    # just those rows.
    handle = pq.ParquetFile(files[0])
    meta = handle.read(columns=["file name", "transcript", "duration", "match quality"]).to_pylist()

    # "102-61" -> source recording 102. Held-out clips must come from different
    # source recordings, or they are neighbouring slices of one waveform and the
    # shared channel inflates the score.
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in meta:
        # "HIGH" means the ASR hypothesis matched the published transcript, so
        # the transcript can be trusted as ground truth.
        if str(row["match quality"]).upper() != "HIGH":
            continue
        seconds = float(row["duration"])
        if not MIN_HELD_OUT_SECONDS <= seconds <= 20.0:
            continue
        by_source[str(row["file name"]).split("-")[0]].append(row)

    # Longest clip from each of the most distinct sources available.
    best_per_source = [
        max(items, key=lambda r: float(r["duration"])) for items in by_source.values()
    ]
    best_per_source.sort(key=lambda r: -float(r["duration"]))
    chosen = best_per_source[: held_out + 1]
    if len(chosen) < 2:
        print("  not enough distinct source recordings; skipping.", file=sys.stderr)
        return []

    print(f"  {len(by_source)} distinct source recordings available; taking {len(chosen)}")
    wanted = {str(row["file name"]): position for position, row in enumerate(chosen)}
    audio_by_name: dict[str, tuple[np.ndarray, int]] = {}
    for batch in handle.iter_batches(batch_size=16, columns=["file name", "audio", "samplerate"]):
        names = batch.column("file name").to_pylist()
        if not any(name in wanted for name in names):
            continue
        for row_index, name in enumerate(names):
            if name not in wanted:
                continue
            samples = np.asarray(batch.column("audio")[row_index].as_py(), dtype=np.float32)
            audio_by_name[name] = (samples, int(batch.column("samplerate")[row_index].as_py()))
        if len(audio_by_name) == len(wanted):
            break

    entries: list[dict[str, Any]] = []
    for position, row in enumerate(chosen):
        name = str(row["file name"])
        if name not in audio_by_name:
            print(f"  audio for {name} not found; skipping.", file=sys.stderr)
            continue
        samples, sample_rate = audio_by_name[name]
        audio = Audio(samples, sample_rate, source="mana-tts")
        filename = f"mana_{position:03d}.wav"
        save_audio(audio, out_dir / filename, bit_depth=24)
        role = "reference" if position == 0 else "held_out"
        print(f"  mana_{position:03d}  {float(row['duration']):5.1f}s  source {name.split('-')[0]}")
        entries.append(
            {
                "speaker": "mana",
                "file": filename,
                "role": role,
                "transcript": str(row["transcript"]).strip(),
                "duration": round(float(row["duration"]), 3),
                "gender": "female_feminine",
                "corpus": "mana_tts",
                "session": name.split("-")[0],
                "speaker_hash": "mana",
            }
        )
    return entries


# ----------------------------------------------------------------------------
# verification - a speaker that cannot be told apart is not ground truth
# ----------------------------------------------------------------------------
def verify(entries: list[dict[str, Any]], out_dir: Path, keep: int) -> list[dict[str, Any]]:
    """Drop speakers that cannot serve as ground truth, and say why.

    Two failures show up in crowdsourced audio and both silently corrupt every
    number downstream:

    *Incoherent speakers.* Common Voice ``client_id`` is an account, not a
    person: households share them, and one contributor can record on a laptop
    one week and a phone in a car the next. The first build of this corpus
    produced a speaker whose own two recordings scored 0.099 against each
    other - lower than most *different-speaker* pairs. Held-out scoring against
    that speaker measures nothing.

    *Confusable speakers.* Two accounts scored 0.645 against each other, as
    high as a typical same-speaker pair. Whether that is two similar voices or
    one person with two accounts, keeping both puts a false ceiling on the
    floor.

    The rule is separability, which needs no invented threshold: keep a speaker
    when their *worst* pair against themselves beats their *best* pair against
    anyone else. When two speakers are mutually confusable the less coherent
    one goes.
    """
    from itertools import combinations

    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation import CachedEmbedder

    _rule("Verification - is each speaker coherent and distinguishable?")

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_speaker[entry["speaker"]].append(entry)

    embedder = CachedEmbedder()
    audio = {e["file"]: load_audio(str(out_dir / e["file"])) for e in entries}

    def score(a: dict[str, Any], b: dict[str, Any]) -> float:
        return embedder.similarity(audio[a["file"]], a["file"], audio[b["file"]], b["file"])

    within: dict[str, list[float]] = {
        speaker: [score(a, b) for a, b in combinations(clips, 2)]
        for speaker, clips in sorted(by_speaker.items())
    }
    between: dict[tuple[str, str], float] = {}
    for left, right in combinations(sorted(by_speaker), 2):
        between[(left, right)] = max(
            score(a, b) for a in by_speaker[left] for b in by_speaker[right]
        )

    def worst_within(speaker: str) -> float:
        return min(within[speaker]) if within[speaker] else 0.0

    def best_between(speaker: str, pool: set[str]) -> float:
        values = [
            value
            for (left, right), value in between.items()
            if (left == speaker and right in pool) or (right == speaker and left in pool)
        ]
        return max(values) if values else -1.0

    surviving = set(by_speaker)
    audit: list[dict[str, Any]] = []

    # Confusable pairs first: dropping one may rescue the other.
    for (left, right), value in sorted(between.items(), key=lambda kv: -kv[1]):
        if left not in surviving or right not in surviving:
            continue
        if value < min(worst_within(left), worst_within(right)):
            continue
        loser = left if statistics.mean(within[left]) < statistics.mean(within[right]) else right
        surviving.discard(loser)
        audit.append(
            {
                "speaker": loser,
                "reason": f"confusable with {right if loser == left else left} at {value:.3f}",
            }
        )
        print(f"  drop {loser:6s} confusable with {right if loser == left else left} at {value:.3f}")

    for speaker in sorted(surviving.copy()):
        inside, outside = worst_within(speaker), best_between(speaker, surviving - {speaker})
        if inside <= outside:
            surviving.discard(speaker)
            audit.append(
                {
                    "speaker": speaker,
                    "reason": f"own clips disagree ({inside:.3f}) more than they differ from others ({outside:.3f})",
                }
            )
            print(f"  drop {speaker:6s} own worst pair {inside:.3f} <= best other-speaker pair {outside:.3f}")

    ranked = sorted(surviving, key=lambda s: -statistics.mean(within[s]))
    if keep and len(ranked) > keep:
        for speaker in ranked[keep:]:
            audit.append({"speaker": speaker, "reason": "surplus to --speakers"})
        ranked = ranked[:keep]
    kept = set(ranked)

    print("\n  Final state. 'best against others' is measured against the speakers that")
    print("  survived, since those are the ones the corpus has to keep apart.\n")
    for speaker in sorted(by_speaker):
        mark = "keep" if speaker in kept else "drop"
        against = best_between(speaker, kept - {speaker})
        print(
            f"  {mark} {speaker:6s} within {statistics.mean(within[speaker]):.3f} "
            f"(worst {worst_within(speaker):.3f})  best against others "
            f"{'    n/a' if against < 0 else f'{against:.3f}'}"
        )

    for entry in entries:
        if entry["speaker"] not in kept:
            (out_dir / entry["file"]).unlink(missing_ok=True)

    print(f"\n  {len(kept)} of {len(by_speaker)} speakers usable as ground truth")
    if audit:
        print("  rejected: " + ", ".join(f"{a['speaker']} ({a['reason']})" for a in audit))
    return [e for e in entries if e["speaker"] in kept]


# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "eval" / "corpus")
    parser.add_argument("--speakers", type=int, default=8, help="Common Voice speakers")
    parser.add_argument("--held-out", type=int, default=DEFAULT_HELD_OUT)
    parser.add_argument("--skip-mana", action="store_true")
    parser.add_argument("--skip-common-voice", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="Keep unverified speakers")
    parser.add_argument(
        "--oversample",
        type=float,
        default=2.5,
        help="Candidates to download per wanted speaker, since verification rejects some",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/persian-voice-cloning/eval-data").expanduser(),
        help="Where the raw downloads land (not the corpus itself)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    if not args.skip_common_voice:
        # Over-select: verification rejects speakers, and downloading a few
        # spares is cheaper than discovering the shortfall afterwards.
        entries += build_common_voice(
            out_dir, args.cache, round(args.speakers * args.oversample), args.held_out
        )
    if not args.skip_mana:
        entries += build_mana(out_dir, args.held_out)

    if not entries:
        print("Nothing built.", file=sys.stderr)
        return 1

    if not args.skip_verify:
        entries = verify(entries, out_dir, keep=args.speakers + (0 if args.skip_mana else 1))

    speakers = sorted({e["speaker"] for e in entries})
    manifest = {
        "version": 1,
        "speakers": len(speakers),
        "clips": len(entries),
        "sources": sorted({e["corpus"] for e in entries}),
        "notes": (
            "Transcripts are ground truth from the source corpus, not ASR. "
            "Held-out clips are separate recordings of the same speaker: for "
            "Common Voice each clip is its own recording session, for Mana each "
            "comes from a different source recording."
        ),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _rule("Built")
    total = sum(e["duration"] for e in entries)
    print(f"{len(entries)} clips, {len(speakers)} speakers, {total / 60:.1f} minutes")
    print(f"  {out_dir}")
    print(f"  {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
