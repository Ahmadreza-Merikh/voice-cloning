import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_adaptation_splits", ROOT / "scripts" / "build_adaptation_splits.py")
splits = importlib.util.module_from_spec(spec)
sys.modules["build_adaptation_splits"] = splits
spec.loader.exec_module(splits)


def cv_speaker(client, gender, durations, texts=None):
    texts = texts or [f"جمله {client} شماره {i}" for i in range(len(durations))]
    return splits.cv_rows([
        {
            "client_id": client,
            "output_audio": f"wavs/{client}_{i:03d}.wav",
            "source_audio": f"clips/common_voice_fa_{i}.mp3",
            "gender": gender,
            "sentence_id": f"{client}-{i}",
            "text": texts[i],
            "output_duration": str(d),
        }
        for i, d in enumerate(durations)
    ])


def ney_speaker(speaker_id, split, durations, texts=None, start=0):
    texts = texts or [f"نی {speaker_id} شماره {i}" for i in range(len(durations))]
    return splits.neyshekar_rows([
        {"id": start + i, "speaker_id": speaker_id, "split": split, "text": texts[i],
         "duration": d, "file": f"{split}-00000-of-00001.parquet"}
        for i, d in enumerate(durations)
    ])


ELIGIBLE = [8.0, 5.0, 5.0, 4.0, 4.0, 3.0]


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(splits, "TEST_PER_GENDER", 2)
    monkeypatch.setattr(splits, "RESERVE_PER_GENDER", 1)
    monkeypatch.setattr(splits, "VALID_SPEAKERS", 2)
    rows = []
    for i in range(4):
        rows += cv_speaker(f"f{i:015d}aa", "female_feminine", ELIGIBLE)
        rows += cv_speaker(f"m{i:015d}bb", "male_masculine", ELIGIBLE)
    for i in range(4):
        rows += cv_speaker(f"u{i:015d}cc", "", ELIGIBLE)
    rows += cv_speaker("dev0000000000000xx", "male_masculine", ELIGIBLE)
    rows += cv_speaker("t" * 16 + "dd", "", [3.0] * 3)
    rows += ney_speaker("101", "train", [5.0] * 4, start=0)
    rows += ney_speaker("202", "validation", [5.0] * 3, start=100)
    rows += ney_speaker("303", "test", ELIGIBLE, start=200)
    rows += ney_speaker("404", "test", [3.0, 3.0], start=300)
    return rows


def run(rows, sentences=None):
    return splits.build_splits(rows, {"dev0000000000000"}, ["متن توسعه"], sentences or {"s1": "جمله محک"})


def by_split(records):
    out = {}
    for r in records:
        out.setdefault(r["split"], set()).add(r["speaker"])
    return out


def test_development_speakers_never_reach_any_other_split(small):
    records, summary = run(small)
    groups = by_split(records)
    assert groups["excluded-development"] == {"dev0000000000000"}
    assert all("dev0000000000000" not in members for name, members in groups.items() if name != "excluded-development")
    assert summary["development_speakers_excluded"] == 1


def test_no_speaker_sits_in_two_splits(small):
    _, summary = run(small)
    assert all(count == 0 for count in summary["speaker_overlap_between_splits"].values())


def test_common_voice_test_is_balanced_and_the_reserve_never_trains(small):
    records, _ = run(small)
    groups = by_split(records)
    test_genders = sorted(r["gender"] for r in records
                          if r["source"] == "cv26" and r["split"] == "test" and r["role"] == "reference")
    assert test_genders == ["female_feminine"] * 2 + ["male_masculine"] * 2
    assert len(groups["test-reserve"]) == 2
    assert not groups["test-reserve"] & groups["train"]


def test_each_usable_test_speaker_has_one_reference_and_four_held_out(small):
    records, _ = run(small)
    for member in by_split(records)["test"]:
        roles = [r["role"] for r in records if r["speaker"] == member]
        if "unused" in roles:
            continue
        assert roles.count("reference") == 1
        assert roles.count("held_out") == 4
        reference = next(r for r in records if r["speaker"] == member and r["role"] == "reference")
        assert reference["duration"] >= splits.MIN_REFERENCE_SECONDS


def test_neyshekar_keeps_its_own_splits(small):
    records, _ = run(small)
    split_of = {r["speaker"]: r["split"] for r in records if r["source"] == "neyshekar"}
    assert split_of == {"ney-101": "train", "ney-202": "valid", "ney-303": "test", "ney-404": "test"}
    assert {r["role"] for r in records if r["speaker"] == "ney-404"} == {"unused"}


def test_a_neyshekar_speaker_in_two_of_its_splits_stops_the_build(small):
    rows = small + ney_speaker("101", "test", [5.0], start=900)
    with pytest.raises(SystemExit, match="straddles"):
        run(rows)


def test_ineligible_common_voice_speakers_only_train(small):
    records, _ = run(small)
    assert {r["split"] for r in records if r["speaker"] == "t" * 16} == {"train"}


def test_a_train_clip_repeating_a_test_text_from_either_source_is_dropped(small):
    records, _ = run(small)
    ney_test_text = next(r["text"] for r in records if r["speaker"] == "ney-303")
    variant = ney_test_text.replace(" ", "‌", 1)
    rows = small + cv_speaker("z" * 16 + "ee", "", [3.0, 3.0], [variant, "متنی دیگر"])
    records, summary = run(rows)
    dropped = [r for r in records if r["speaker"] == "z" * 16 and r["split"] == "excluded-text"]
    assert [r["text"] for r in dropped] == [variant]
    assert summary["train_texts_shared_with_other_splits"] == 0


def test_a_train_clip_containing_a_benchmark_sentence_is_dropped(small):
    rows = small + cv_speaker("y" * 16 + "ff", "", [3.0], ["دیروز جمله محک را خواندم"])
    records, summary = run(rows, {"s1": "جمله محک"})
    assert [r["split"] for r in records if r["speaker"] == "y" * 16] == ["excluded-text"]
    assert summary["train_texts_overlapping_benchmark_sentences"] == {"s1": 0}


def test_the_draw_is_fixed_by_the_seed(small):
    first, _ = run(small)
    second, _ = run(list(reversed(small)))
    assert first == second


def test_a_missing_development_speaker_stops_the_build(small):
    rows = [r for r in small if not r["speaker"].startswith("dev")]
    with pytest.raises(SystemExit, match="development speakers missing"):
        run(rows)


def test_speaker_gender_needs_agreement():
    assert splits.speaker_gender([{"gender_label": ""}, {"gender_label": "male_masculine"}]) == "male_masculine"
    assert splits.speaker_gender([{"gender_label": ""}]) == "unknown"
    assert splits.speaker_gender([{"gender_label": "male_masculine"},
                                  {"gender_label": "female_feminine"}]) == "mixed"
