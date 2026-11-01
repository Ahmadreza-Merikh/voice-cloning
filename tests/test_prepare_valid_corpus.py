"""Neyshekar's validation speakers as a corpus: the test sets' roles, validation records only.

The real split is read for its metadata alone; the writing is tested on a
parquet file made up here.
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
spec = importlib.util.spec_from_file_location("prepare_valid_corpus", ROOT / "scripts" / "prepare_valid_corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)

SPLITS = ROOT / "eval" / "manifests" / "adaptation_splits_v1.jsonl.gz"


def wav(seconds: float, rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * rate), dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_the_split_gives_25_of_its_26_validation_speakers_roles() -> None:
    speakers, clips = corpus.validation(corpus.load_records(SPLITS))
    assert len(speakers) == 26
    with_roles = [s for s in speakers if corpus.whole_reference_roles(clips[s]) is not None]
    assert len(with_roles) == 25 and "ney-70652" not in with_roles


def test_only_validation_clips_are_written_and_a_failure_is_left_out(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for speaker, first, split in (("ney-1", 100, "valid"), ("ney-2", 200, "valid"), ("ney-3", 300, "test")):
        for i, seconds in enumerate([9.5, 6.0, 5.0, 4.0, 3.6, 2.0]):
            records.append({"clip": f"neyshekar-{first + i}", "duration": seconds, "gender": "unknown",
                            "speaker": speaker, "split": split, "source": "neyshekar", "text": f"جمله {first + i}",
                            "source_ref": f"validation.parquet#id={first + i}", "role": "other"})
    splits = tmp_path / "splits.jsonl.gz"
    with gzip.open(splits, "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    rows = [{"id": int(r["clip"].split("-")[1]), "audio": {"bytes": wav(r["duration"], 16000), "path": None}}
            for r in records if r["clip"] != "neyshekar-203"]
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), parquet / "validation-00000.parquet")

    out = tmp_path / "corpus"
    assert corpus.main(["--splits", str(splits), "--parquet-dir", str(parquet), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    roster = json.loads((out / "roster.json").read_text(encoding="utf-8"))
    assert [e["file"] for e in manifest["entries"]] == [f"nev01_{i:03d}.wav" for i in range(5)]
    assert [e["role"] for e in manifest["entries"]] == ["reference"] + ["held_out"] * 4
    assert {e["session"] for e in manifest["entries"]} == {f"neyshekar-{i}" for i in range(100, 105)}
    assert roster["left_out"] == [{"speaker": "ney-2", "reason": "neyshekar-203: not in the parquet files"}]
    assert sorted(p.name for p in out.glob("*.wav")) == sorted(roster["files"])
    for name, digest in roster["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
