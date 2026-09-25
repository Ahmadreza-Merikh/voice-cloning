"""The LongCat probe's corpus: Neyshekar's valid speakers, the sealed test's roles, nothing else.

The speakers are checked on the real split's metadata; the writing on a parquet
file made up here, which also carries a test clip and a training clip that must
never be written.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("prepare_probe_corpus", ROOT / "scripts" / "prepare_probe_corpus.py")
probe = importlib.util.module_from_spec(spec)
sys.modules["prepare_probe_corpus"] = probe
spec.loader.exec_module(probe)

SPLITS = ROOT / "eval" / "manifests" / "adaptation_splits_v1.jsonl.gz"


def test_the_split_gives_25_usable_speakers_none_of_them_sealed_or_trained_on() -> None:
    records = probe.load_records(SPLITS)
    clips = probe.valid_speakers(records)
    assert len(clips) == 26
    assert {c["split"] for rows in clips.values() for c in rows} == {"valid"}
    assert sum(probe.whole_reference_roles(rows) is not None for rows in clips.values()) == 25
    elsewhere = {r["speaker"] for r in records if r["split"] != "valid"}
    assert not set(clips) & elsewhere


def wav(seconds: float, rate: int) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    samples = 0.1 * np.sin(np.linspace(0, 2 * np.pi * 150 * seconds, int(round(seconds * rate))))
    soundfile.write(buffer, samples.astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_only_valid_clips_are_written_and_a_broken_speaker_is_listed(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for speaker, first, split in (("ney-1", 100, "valid"), ("ney-2", 200, "valid"), ("ney-3", 300, "test"),
                                  ("ney-4", 400, "train")):
        for i, seconds in enumerate([9.5, 6.0, 5.0, 4.0, 3.6, 2.0]):
            records.append({"clip": f"neyshekar-{first + i}", "duration": seconds, "gender": "unknown",
                            "speaker": speaker, "split": split, "source": "neyshekar",
                            "text": f"جمله {first + i}", "source_ref": f"x.parquet#id={first + i}", "role": None})
    splits = tmp_path / "splits.jsonl.gz"
    with gzip.open(splits, "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    rows = [{"id": int(r["clip"].split("-")[1]), "audio": {"bytes": wav(r["duration"], 16000), "path": None}}
            for r in records if r["clip"] != "neyshekar-202"]
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), parquet / "validation-00000.parquet")

    out = tmp_path / "corpus"
    assert probe.main(["--splits", str(splits), "--parquet-dir", str(parquet), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    roster = json.loads((out / "roster.json").read_text(encoding="utf-8"))
    assert [e["file"] for e in manifest["entries"]] == [f"nvl01_{i:03d}.wav" for i in range(5)]
    assert [e["role"] for e in manifest["entries"]] == ["reference"] + ["held_out"] * 4
    assert {e["session"] for e in manifest["entries"]} <= {f"neyshekar-{100 + i}" for i in range(6)}
    assert roster["left_out"] == [{"speaker": "ney-2", "reason": "neyshekar-202: not in the parquet files"}]
    assert sorted(p.name for p in out.glob("*.wav")) == sorted(roster["files"])
    for name, digest in roster["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
