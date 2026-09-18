#!/usr/bin/env python3
"""Split two public Persian speech corpora into train, valid and test for adapting OmniVoice.

What this is for
----------------
OmniVoice's official adaptation path trains on a JSONL of ``{id, audio_path,
text, language_id}`` plus a dev JSONL for its evaluation loss. This script
decides which clip goes where, once, before any audio is read, and writes the
decision down with enough source identifiers to check every line of it.

Sources, both CC0 and ungated, both pinned to a revision
--------------------------------------------------------
* **Common Voice 26 Persian, cleaned** (``pymmdrza/Common-Voice-Speech-26.0-
  Persian-Clean``): validated clips with at least two up-votes and none down,
  2-10 s, text de-duplicated, at most 400 clips per speaker - 34,134 clips,
  3,791 speakers, 43.1 h, 24 kHz. The project already uses it for its
  development corpus. ``reports/kept.csv`` names every clip's ``client_id``,
  original Common Voice file and ``sentence_id``.
* **Neyshekar v6** (``shekar-ai/neyshekar``): read speech from 190 volunteer
  contributors on a separate platform, 62,279 clips, 99 h, 16 kHz, published
  with speaker-disjoint train / validation / test splits. It is not on
  OmniVoice's published list of training data, so its test speakers are the
  only ones here that OmniVoice has - on that list's evidence - not heard.

Rules, fixed before the split was drawn
---------------------------------------
* **Development speakers stay development.** Every clip by the eight Common
  Voice speakers of ``eval/corpus/manifest.json`` is excluded from every split
  here. ``mana`` is from a third corpus and never appears.
* **Speakers come from source labels, never from an encoder.** No audio is read
  and no speaker-similarity model is consulted, so the judging ECAPA cannot
  shape any population here. Identity is each source's own speaker id - an
  account, not a verified person; one person with two accounts, or with
  recordings on both platforms, could straddle two splits undetected.
* **Neyshekar keeps its own splits**: its test speakers are test, its
  validation speakers are valid, its train speakers are train.
* **Common Voice test speakers** have 5-40 clips, a clip of at least 6.5 s for
  the reference and four more of at least 3.5 s to hold out - the development
  corpus's own rule. 30 are drawn in a seeded order, 15 per self-reported
  gender, with a reserve of 10 that replaces a test speaker only for a
  mechanical failure (unreadable or missing audio). The reserve never trains.
  40 more eligible speakers, in a second seeded order, are valid.
* **Test roles**: in both sources a test speaker's longest clip is the
  reference and the next four of at least 3.5 s are held out, when the speaker
  has them; a speaker without them stays in test, unused, and never trains.
* **No text crosses into train.** A training clip is dropped if its text,
  normalised the way CER normalises it, equals the text of any development,
  test, reserve or valid clip in either source, or contains or is contained in
  a benchmark sentence.
* **No session crosses.** Each clip is its own recording in both sources, so
  with speakers disjoint no recording is shared.

The test splits are sealed: they are listed so they cannot drift, and nothing
is rendered or scored on them until the final candidates are fixed. The
Neyshekar test speakers are the primary test (independent of OmniVoice's
published training data); the Common Voice ones are secondary, because
OmniVoice's training data includes Common Voice and all but 4 of that source's
speakers already had clips in Common Voice 22.0 (FINDINGS §9).

Usage
-----
    .venv/bin/python scripts/build_adaptation_splits.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"
CV_REVISION = "142e9bb40b56d9491faa0ec0987ee8d408bd9c05"
CV_KEPT = "reports/kept.csv"
CV_KEPT_SHA256 = "f3633e017ec5342201cd7ca68b6dbacdf0a781bb50a756b95fd7c91d1e303b09"

NEY_REPO = "shekar-ai/neyshekar"
NEY_REVISION = "30696ada746a66bb3047c984744b5ecfd60fe4e9"
NEY_COLUMNS = ("id", "text", "duration", "speaker_id")
NEY_SPLITS = {"train": "train", "validation": "valid", "test": "test"}

SEED = 20260919
TEST_PER_GENDER = 15
RESERVE_PER_GENDER = 5
VALID_SPEAKERS = 40
MIN_CLIPS, MAX_CLIPS = 5, 40
MIN_REFERENCE_SECONDS = 6.5
MIN_HELD_OUT_SECONDS = 3.5
HELD_OUT = 4
GENDERS = ("female_feminine", "male_masculine")
NOT_TRAIN = {"excluded-development", "test", "test-reserve", "valid"}

OUT_DIR = REPO_ROOT / "eval" / "manifests"
NAME = "adaptation_splits_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(text: str) -> str:
    from voiceclone.evaluation.similarity import normalize_for_cer

    return normalize_for_cer(text)


def benchmark_sentences() -> dict[str, str]:
    """The benchmark's sentences, read from its source without importing it."""
    tree = ast.parse((REPO_ROOT / "scripts" / "benchmark.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "SENTENCES":
            return dict(ast.literal_eval(node.value))
    raise SystemExit("SENTENCES not found in scripts/benchmark.py")


def development(manifest: Path) -> tuple[set[str], list[str]]:
    """Speaker-id prefixes and transcripts of the Common Voice development speakers."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    hashes, texts = set(), []
    for entry in data["entries"]:
        texts.append(entry["transcript"])
        if entry["corpus"].startswith("common_voice"):
            hashes.add(entry["speaker_hash"])
    return hashes, texts


def cv_rows(kept: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "source": "cv26",
            "clip": Path(r["output_audio"]).name,
            "speaker": r["client_id"],
            "gender_label": r["gender"],
            "text": r["text"],
            "duration": float(r["output_duration"]),
            "source_ref": r["source_audio"],
            "sentence_id": r["sentence_id"],
            "official_split": None,
        }
        for r in kept
    ]


def neyshekar_rows(meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": "neyshekar",
            "clip": f"neyshekar-{r['id']}",
            "speaker": f"ney-{r['speaker_id']}",
            "gender_label": "",
            "text": r["text"],
            "duration": float(r["duration"]),
            "source_ref": f"{r['file']}#id={r['id']}",
            "sentence_id": None,
            "official_split": r["split"],
        }
        for r in meta
    ]


def by_duration(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(clips, key=lambda r: (-r["duration"], r["clip"]))


def has_test_roles(clips: list[dict[str, Any]]) -> bool:
    ordered = by_duration(clips)
    if ordered[0]["duration"] < MIN_REFERENCE_SECONDS:
        return False
    return sum(1 for r in ordered[1:] if r["duration"] >= MIN_HELD_OUT_SECONDS) >= HELD_OUT


def eligible(clips: list[dict[str, Any]]) -> bool:
    return MIN_CLIPS <= len(clips) <= MAX_CLIPS and has_test_roles(clips)


def speaker_gender(clips: list[dict[str, Any]]) -> str:
    """The self-reported label, when the speaker's clips agree on one.

    Common Voice stores the label per clip, from the profile at recording
    time, so one speaker can have labelled and unlabelled clips, or two labels.
    """
    labels = {r["gender_label"] for r in clips if r["gender_label"]}
    if not labels:
        return "unknown"
    return labels.pop() if len(labels) == 1 else "mixed"


def test_roles(clips: list[dict[str, Any]]) -> dict[str, str]:
    """Longest clip is the reference, the next four long enough are held out."""
    if not has_test_roles(clips):
        return {r["clip"]: "unused" for r in clips}
    ordered = by_duration(clips)
    roles = {ordered[0]["clip"]: "reference"}
    held = [r for r in ordered[1:] if r["duration"] >= MIN_HELD_OUT_SECONDS][:HELD_OUT]
    for row in held:
        roles[row["clip"]] = "held_out"
    for row in clips:
        roles.setdefault(row["clip"], "other")
    return roles


def draw_common_voice(
    by_speaker: dict[str, list[dict[str, Any]]], excluded: set[str], seed: int
) -> dict[str, str]:
    pool = sorted(c for c, clips in by_speaker.items() if c not in excluded and eligible(clips))
    random.Random(seed).shuffle(pool)
    gender = {c: speaker_gender(by_speaker[c]) for c in pool}
    test, reserve = [], []
    for label in GENDERS:
        ranked = [c for c in pool if gender[c] == label]
        test += ranked[:TEST_PER_GENDER]
        reserve += ranked[TEST_PER_GENDER : TEST_PER_GENDER + RESERVE_PER_GENDER]
    shortfall = 2 * TEST_PER_GENDER - len(test)
    if shortfall:
        extra = [c for c in pool if c not in test and c not in reserve and gender[c] == "unknown"]
        test += extra[:shortfall]
    # A separate order: test and reserve took the first labelled speakers of
    # each gender, so the rest of this order starts with unlabelled ones.
    rest = sorted(c for c in pool if c not in set(test) | set(reserve))
    random.Random(seed + 1).shuffle(rest)
    assigned = dict.fromkeys(test, "test")
    assigned.update(dict.fromkeys(reserve, "test-reserve"))
    assigned.update(dict.fromkeys(rest[:VALID_SPEAKERS], "valid"))
    return assigned


def build_splits(
    rows: list[dict[str, Any]],
    dev_hashes: set[str],
    dev_texts: list[str],
    sentences: dict[str, str],
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    sources = {speaker: clips[0]["source"] for speaker, clips in by_speaker.items()}
    if any(len({r["source"] for r in clips}) > 1 for clips in by_speaker.values()):
        raise SystemExit("a speaker id appears in two sources")

    cv_speakers = {s: c for s, c in by_speaker.items() if sources[s] == "cv26"}
    dev_ids = {c for c in cv_speakers if any(c.startswith(h) for h in dev_hashes)}
    found = {h for h in dev_hashes if any(c.startswith(h) for c in dev_ids)}
    if found != dev_hashes:
        raise SystemExit(f"development speakers missing from the source: {sorted(dev_hashes - found)}")

    split_of = dict.fromkeys(dev_ids, "excluded-development")
    split_of.update(draw_common_voice(cv_speakers, dev_ids, seed))
    for speaker, clips in by_speaker.items():
        if sources[speaker] == "neyshekar":
            official = {r["official_split"] for r in clips}
            if len(official) != 1:
                raise SystemExit(f"{speaker} straddles the source's own splits: {sorted(official)}")
            split_of[speaker] = NEY_SPLITS[official.pop()]

    protected = {normalise(t) for t in dev_texts}
    for speaker, split in split_of.items():
        if split in NOT_TRAIN:
            protected.update(normalise(r["text"]) for r in by_speaker[speaker])
    protected.discard("")
    bench = {key: normalise(text) for key, text in sentences.items()}

    records: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker):
        clips = by_speaker[speaker]
        split = split_of.get(speaker, "train")
        roles = test_roles(clips) if split in {"test", "test-reserve"} else {}
        gender = speaker_gender(clips)
        for row in sorted(clips, key=lambda r: r["clip"]):
            record_split, reason = split, None
            if split == "train":
                text = normalise(row["text"])
                if text in protected:
                    record_split, reason = "excluded-text", "text of a development, test, reserve or valid clip"
                else:
                    for key, sentence in bench.items():
                        if sentence and (sentence in text or text in sentence):
                            record_split, reason = "excluded-text", f"overlaps benchmark sentence {key}"
                            break
            records.append(
                {
                    "source": row["source"],
                    "clip": row["clip"],
                    "split": record_split,
                    "role": roles.get(row["clip"]),
                    "speaker": speaker[:16] if row["source"] == "cv26" else speaker,
                    "gender": gender,
                    "text": row["text"],
                    "duration": round(row["duration"], 3),
                    "source_ref": row["source_ref"],
                    **({"sentence_id": row["sentence_id"]} if row["sentence_id"] else {}),
                    **({"excluded_because": reason} if reason else {}),
                }
            )

    return records, audit(records, sentences, len(dev_ids))


def audit(records: list[dict[str, Any]], sentences: dict[str, str], dev_speakers: int) -> dict[str, Any]:
    table: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    speakers: dict[str, set[str]] = defaultdict(set)
    genders: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for record in records:
        entry = table[record["source"]].setdefault(record["split"], {"clips": 0, "seconds": 0.0})
        entry["clips"] += 1
        entry["seconds"] += record["duration"]
        speakers[record["split"]].add(record["speaker"])
        genders[(record["source"], record["split"])][record["speaker"]] = record["gender"]
    splits = {}
    for source, entries in sorted(table.items()):
        splits[source] = {}
        for name, entry in sorted(entries.items()):
            members = genders[(source, name)]
            splits[source][name] = {
                "clips": entry["clips"],
                "speakers": len(members),
                "hours": round(entry["seconds"] / 3600, 3),
                "genders": dict(Counter(members.values())),
            }

    overlaps = {}
    names = sorted(speakers)
    # A speaker whose text-overlapping clips were dropped from train still
    # trains on the rest, so "excluded-text" shares speakers with train by
    # design and is not a leak.
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if {left, right} == {"train", "excluded-text"}:
                continue
            overlaps[f"{left} & {right}"] = len(speakers[left] & speakers[right])

    train_texts = {normalise(r["text"]) for r in records if r["split"] == "train"}
    other_texts = {normalise(r["text"]) for r in records if r["split"] in NOT_TRAIN}
    bench_hits = {
        key: sum(1 for t in train_texts if normalise(s) in t or t in normalise(s))
        for key, s in sentences.items()
    }
    roles = Counter((r["source"], r["split"], r["role"]) for r in records if r["role"])
    return {
        "splits": splits,
        "development_speakers_excluded": dev_speakers,
        "speaker_overlap_between_splits": overlaps,
        "train_texts_shared_with_other_splits": len(train_texts & other_texts),
        "train_texts_overlapping_benchmark_sentences": bench_hits,
        "test_roles": {f"{src}/{s}/{r}": n for (src, s, r), n in sorted(roles.items())},
    }


def read_neyshekar(cache: Path | None) -> list[dict[str, Any]]:
    """Every clip's metadata at the pinned revision, without downloading audio."""
    import pyarrow.parquet as pq

    if cache is not None and cache.is_file():
        return pq.read_table(cache).to_pylist()
    import pyarrow as pa
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{NEY_REPO}@{NEY_REVISION}/data"
    tables = []
    for path in sorted(p for p in fs.ls(base, detail=False) if p.endswith(".parquet")):
        name = path.split("/")[-1]
        with fs.open(path, "rb", block_size=1 << 20) as handle:
            table = pq.ParquetFile(handle).read(columns=list(NEY_COLUMNS))
        table = table.append_column("split", pa.array([name.split("-")[0]] * table.num_rows))
        tables.append(table.append_column("file", pa.array([name] * table.num_rows)))
        print(f"  read {name}: {table.num_rows} rows", flush=True)
    table = pa.concat_tables(tables)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, cache)
    return table.to_pylist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kept", type=Path, help="A local copy of Common Voice reports/kept.csv at the pinned revision")
    parser.add_argument("--neyshekar-meta", type=Path,
                        default=Path("~/.cache/persian-voice-cloning/neyshekar-v6-meta.parquet").expanduser(),
                        help="Where Neyshekar's metadata is cached (read from the Hub if absent)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    kept = args.kept
    if kept is None:
        from huggingface_hub import hf_hub_download

        kept = Path(hf_hub_download(CV_REPO, CV_KEPT, repo_type="dataset", revision=CV_REVISION))
    digest = sha256_file(kept)
    if digest != CV_KEPT_SHA256:
        raise SystemExit(f"{kept} has sha256 {digest}, expected {CV_KEPT_SHA256}")
    with kept.open(encoding="utf-8") as handle:
        rows = cv_rows(list(csv.DictReader(handle)))
    rows += neyshekar_rows(read_neyshekar(args.neyshekar_meta))

    dev_hashes, dev_texts = development(REPO_ROOT / "eval" / "corpus" / "manifest.json")
    sentences = benchmark_sentences()
    records, summary = build_splits(rows, dev_hashes, dev_texts, sentences)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    splits_path = args.out_dir / f"{NAME}.jsonl.gz"
    with gzip.GzipFile(splits_path, "wb", mtime=0) as handle:
        handle.write(lines.encode("utf-8"))

    summary = {
        "sources": {
            "cv26": {"repo": CV_REPO, "revision": CV_REVISION, "file": CV_KEPT, "sha256": digest,
                     "licence": "CC0-1.0 (the dataset card; Common Voice upstream)", "gated": False,
                     "version": "cv-corpus-26.0-2026-06-12"},
            "neyshekar": {"repo": NEY_REPO, "revision": NEY_REVISION, "columns": list(NEY_COLUMNS),
                          "licence": "CC0-1.0 (the dataset card)", "gated": False, "version": "V6.0 (2026-09-01)",
                          "terms": "the card prohibits any attempt to identify speakers"},
        },
        "rules": {
            "seed": SEED,
            "cv26_test": f"{TEST_PER_GENDER} per self-reported gender, reserve {RESERVE_PER_GENDER} per gender",
            "cv26_valid_speakers": VALID_SPEAKERS,
            "cv26_eligible": (
                f"{MIN_CLIPS}-{MAX_CLIPS} clips, one of at least {MIN_REFERENCE_SECONDS} s and "
                f"{HELD_OUT} more of at least {MIN_HELD_OUT_SECONDS} s"
            ),
            "neyshekar": "the source's own speaker-disjoint splits: test is test, validation is valid",
            "selection": "seeded random order over eligible speakers; no audio read, no encoder",
            "text": "a train clip is dropped if its CER-normalised text equals any non-train clip's "
                    "in either source, or overlaps a benchmark sentence",
        },
        "records": str(splits_path.relative_to(REPO_ROOT)),
        "records_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        **summary,
        "primary_test": "neyshekar test speakers with a reference and four held-out clips",
        "secondary_test": "cv26 test speakers",
        "sealed": "test and test-reserve are not rendered or scored until the final candidates are fixed",
        "caveat": (
            "OmniVoice's training data includes Common Voice; all but 4 of the cv26 speakers had clips "
            "in Common Voice 22.0, so it has very probably heard the cv26 test speakers. Neyshekar is "
            "not on OmniVoice's published list of training data."
        ),
    }
    summary_path = args.out_dir / f"{NAME}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for source, entries in summary["splits"].items():
        for name, entry in entries.items():
            print(f"  {source:10s} {name:22s} {entry['clips']:6d} clips {entry['speakers']:5d} speakers "
                  f"{entry['hours']:8.3f} h  {entry['genders']}")
    leaks = {k: v for k, v in summary["speaker_overlap_between_splits"].items() if v}
    print(f"  speaker overlaps between splits: {leaks or 'none'}")
    print(f"  train texts shared with other splits: {summary['train_texts_shared_with_other_splits']}")
    print(f"  test roles: {summary['test_roles']}")
    print(f"  written: {splits_path}\n           {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
