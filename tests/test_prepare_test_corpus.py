"""The sealed test corpus: whole references, the split's speakers, the split's reserve rule.

Nothing here reads a test clip's audio. The checks on the real split use its
metadata only - durations, roles, speaker ids - which is what the role rule was
decided on; the writing is tested on a parquet file made up here.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("prepare_test_corpus", ROOT / "scripts" / "prepare_test_corpus.py")
corpus = importlib.util.module_from_spec(spec)
sys.modules["prepare_test_corpus"] = corpus
spec.loader.exec_module(corpus)

SPLITS = ROOT / "eval" / "manifests" / "adaptation_splits_v1.jsonl.gz"


def clip(name: str, seconds: float, **extra) -> dict:
    return {"clip": name, "duration": seconds, **extra}


def test_the_reference_is_the_longest_clip_the_pipeline_keeps_whole() -> None:
    clips = [clip("a", 24.5), clip("b", 11.0), clip("c", 11.45), clip("d", 6.0), clip("e", 5.0), clip("f", 3.0)]
    reference, held = corpus.whole_reference_roles(clips)
    assert reference["clip"] == "b"
    assert [c["clip"] for c in held] == ["a", "c", "d", "e"]


def test_a_speaker_without_a_whole_reference_or_four_to_hold_out_has_no_roles() -> None:
    assert corpus.whole_reference_roles([clip("a", 12.0), clip("b", 6.0), *(clip(f"x{i}", 5.0) for i in range(4))]) is None
    assert corpus.whole_reference_roles([clip("a", 9.0), clip("b", 5.0), clip("c", 5.0), clip("d", 3.4)]) is None


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return corpus.load_records(SPLITS)


def roles_differ(records: list[dict], source: str) -> tuple[int, list[str], int]:
    test, _, clips = corpus.sealed(records, source)
    usable, unusable, changed = 0, [], 0
    for speaker in test:
        roles = corpus.whole_reference_roles(clips[speaker])
        if roles is None:
            unusable.append(speaker)
            continue
        usable += 1
        old = ([c["clip"] for c in clips[speaker] if c["role"] == "reference"],
               sorted(c["clip"] for c in clips[speaker] if c["role"] == "held_out"))
        changed += old != ([roles[0]["clip"]], sorted(c["clip"] for c in roles[1]))
    return usable, unusable, changed


def test_the_amended_rule_keeps_28_neyshekar_speakers_and_moves_13_references(records: list[dict]) -> None:
    usable, unusable, changed = roles_differ(records, "neyshekar")
    assert usable == 28
    assert unusable == ["ney-61189", "ney-97668"]
    assert changed == 13


def test_the_amended_rule_changes_no_common_voice_speaker(records: list[dict]) -> None:
    assert roles_differ(records, "cv26") == (30, [], 0)


def test_no_neyshekar_reference_is_longer_than_the_window_limit(records: list[dict]) -> None:
    test, _, clips = corpus.sealed(records, "neyshekar")
    for speaker in test:
        roles = corpus.whole_reference_roles(clips[speaker])
        if roles:
            assert roles[0]["duration"] <= corpus.WHOLE_REFERENCE_SECONDS


def test_the_limit_sits_under_the_preparation_window(config) -> None:
    threshold = config.audio.reference.target_seconds * 1.15
    # The file may differ from the split's duration by 50 ms; the margin is twice that.
    assert threshold + 1e-9 >= corpus.WHOLE_REFERENCE_SECONDS + 0.1


def test_the_reserve_order_is_the_split_builders(records: list[dict]) -> None:
    order = corpus.reserve_order(records)
    assert {gender: len(speakers) for gender, speakers in order.items()} == dict.fromkeys(corpus.GENDERS, 5)
    reserve = {r["speaker"] for r in records if r["source"] == "cv26" and r["split"] == "test-reserve"}
    assert {s for speakers in order.values() for s in speakers} == reserve


def test_only_sealed_records_are_looked_up(records: list[dict]) -> None:
    for source in ("neyshekar", "cv26"):
        _, _, clips = corpus.sealed(records, source)
        assert {c["split"] for rows in clips.values() for c in rows} <= {"test", "test-reserve"}


def wav(seconds: float, rate: int) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    samples = 0.1 * np.sin(np.linspace(0, 2 * np.pi * 150 * seconds, int(round(seconds * rate))))
    soundfile.write(buffer, samples.astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def speaker_records(speaker: str, split: str, gender: str, first: int) -> list[dict]:
    lengths = [8.0, 5.0, 4.5, 4.0, 3.8]
    return [{"clip": f"c{first + i}", "duration": seconds, "gender": gender, "speaker": speaker,
             "split": split, "source": "cv26", "text": f"متن {first + i}", "source_ref": None,
             "role": "reference" if i == 0 else "held_out"} for i, seconds in enumerate(lengths)]


def test_a_mechanical_failure_takes_the_next_reserve_of_the_same_gender() -> None:
    clips = {"s1": speaker_records("s1", "test", "female_feminine", 0),
             "s2": speaker_records("s2", "test", "male_masculine", 10),
             "r1": speaker_records("r1", "test-reserve", "male_masculine", 20),
             "r2": speaker_records("r2", "test-reserve", "female_feminine", 30),
             "r3": speaker_records("r3", "test-reserve", "female_feminine", 40)}
    audio = {c["clip"]: wav(c["duration"], 24000) for rows in clips.values() for c in rows}
    audio["c3"] = b"not audio"
    del audio["c32"]
    result = corpus.roster("cv26", ["s1", "s2"], {"female_feminine": ["r2", "r3"], "male_masculine": ["r1"]},
                           clips, audio)
    assert [row["speaker"] for row in result["chosen"]] == ["r3", "s2"]
    assert result["replaced"] == [{"speaker": "s1", "reason": "c3: unreadable: LibsndfileError", "by": "r3"}]
    assert result["left_out"] == [{"speaker": "r2", "reason": "reserve, unusable: c32: not in the parquet files"}]
    assert [row["label"] for row in result["chosen"]] == ["cvt01", "cvt02"]


def test_a_neyshekar_speaker_with_a_failure_is_left_out_and_listed(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for speaker, first in (("ney-1", 100), ("ney-2", 200)):
        for i, seconds in enumerate([9.5, 6.0, 5.0, 4.0, 3.6, 2.0]):
            records.append({"clip": f"neyshekar-{first + i}", "duration": seconds, "gender": "unknown",
                            "speaker": speaker, "split": "test", "source": "neyshekar", "text": f"جمله {first + i}",
                            "source_ref": f"test.parquet#id={first + i}", "role": "other"})
    records.append({**records[0], "clip": "neyshekar-999", "split": "train", "speaker": "ney-9"})
    splits = tmp_path / "splits.jsonl.gz"
    with gzip.open(splits, "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    rows = [{"id": int(r["clip"].split("-")[1]), "audio": {"bytes": wav(r["duration"], 16000), "path": None}}
            for r in records if r["clip"] not in {"neyshekar-203", "neyshekar-999"}]
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), parquet / "test-00000.parquet")

    out = tmp_path / "corpus"
    assert corpus.main(["--splits", str(splits), "--source", "neyshekar", "--parquet-dir", str(parquet),
                        "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    roster = json.loads((out / "roster.json").read_text(encoding="utf-8"))
    assert [e["file"] for e in manifest["entries"]] == [f"ney01_{i:03d}.wav" for i in range(5)]
    assert [e["role"] for e in manifest["entries"]] == ["reference"] + ["held_out"] * 4
    assert manifest["entries"][0]["session"] == "neyshekar-100"
    assert roster["left_out"] == [{"speaker": "ney-2", "reason": "neyshekar-203: not in the parquet files"}]
    assert roster["replaced"] == []
    for name, digest in roster["files"].items():
        import hashlib

        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
