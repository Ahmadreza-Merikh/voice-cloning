#!/usr/bin/env python3
"""A/B/C listening-test harness: does the OmniVoice IPA fine-tune read
Persian well while keeping pronunciation and the cloned voice?

Conditions - one reference, one seed, identical generation settings,
differing only where the condition says they differ:

  A  ``omnivoice``      + raw Persian script      (today's behaviour)
  B  ``omnivoice``      + diacritised script      (where a variant exists;
                                                   otherwise recorded as skip)
  C  ``omnivoice_ipa``  + canonical Persian IPA   (Homo-GE2PE -> the
                                                   normaliser -> the fine-tune)

Sentences: nine purpose-written ambiguity probes (s01-s09: homographs,
ezafe chains, خ/ق/غ vs ژ/چ/ش, clusters, colloquial, long), the four
*sealed* held-out texts t09-t12 read from
``eval/manifests/test_texts_v1.json`` (never copied), and one long real
corpus transcript (r01). Probe sentences are checked against the
adaptation split with the sealed texts' own overlap rule and any hit is
recorded in the manifest.

Why the ordering is fixed
-------------------------
Two OmniVoice workers must not run at once on this Mac, so renders are
grouped by engine: all of A/B first, ``studio.release()``, then C.
The G2P/normaliser capture runs *before* any engine can load, so the
manifest describes the phonemes even if a render later fails.

Evaluation is off during rendering (``render_config``): nothing may pull
the speaker encoder or the recogniser in beside a 2 GB worker, and the
listening test - not a metric - is what decides. ``--cer`` transcribes
each saved wav *after* every engine is unloaded; CER then compares
against the condition's own input (diacritics are stripped by
``normalize_for_cer``, so A and B score on identical words).

Manifest (``<out>/manifest.json``), one record per render: input text,
engine, requested/received phoneme channel, seed, reference voice, wav
path, duration, timings, warnings - and per sentence: raw GE2PE compact,
the canonical IPA fed to C, every symbol change, unknown symbols, notes.

Usage
-----
    .venv/bin/python scripts/ab_omnivoice_ipa.py                  # all 39 renders
    .venv/bin/python scripts/ab_omnivoice_ipa.py --only s01,t10   # a subset
    .venv/bin/python scripts/ab_omnivoice_ipa.py --cer            # + ASR after unload
    .venv/bin/python scripts/ab_omnivoice_ipa.py --speaker cv02 --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("ab_omnivoice_ipa")

DEFAULT_SEED = 42
ENGINE_A = "omnivoice"
ENGINE_C = "omnivoice_ipa"
CONDITIONS = ("A", "B", "C")
ENGINE_FOR = {"A": ENGINE_A, "B": ENGINE_A, "C": ENGINE_C}
PHONEMES_FOR = {"A": False, "B": False, "C": True}
#: The sealed texts used here, read from the manifest - never re-typed.
SEALED_KEYS = ("t09", "t10", "t11", "t12")
#: Sentences deliberately left without a diacritised variant, so the skip
#: path is exercised by the default run itself (long or corpus texts).
NO_VARIANT = frozenset({"s09", "t12", "r01"})

#: Purpose-written ambiguity probes. Plain script only - the diacritised
#: twins live in DIACRITIZED and differ by harakat and nothing else.
PROBES: tuple[tuple[str, str, str], ...] = (
    ("s01", "homograph_ghol_gil",
     "او گل قرمز را از گل باغچه بیرون کشید."),
    ("s02", "homograph_mard_murd",
     "پیرمرد گفت این مرد هفتهٔ پیش مرده است."),
    ("s03", "ezafe_chain",
     "کتاب خوب برادرم روی میز اتاق است."),
    ("s04", "homograph_rooye",
     "لیوان آب را روی میز بگذار و روی صورتت آب بکش."),
    ("s05", "consonants_kha_qha_gha",
     "قلعهٔ قدیمی بالای تپهٔ خاکی، هنگام غروب دیدنی بود."),
    ("s06", "consonants_zhe_ch_sh",
     "ژاله زیر چراغ، چترش را در باران شدید بست."),
    ("s07", "clusters",
     "مسؤول استقبال استاد از سرسرای دانشگاه ایستاد."),
    ("s08", "colloquial",
     "برو بابا، این حرفا پیش من خریدار نداره، بشین ببین چی میگم."),
    ("s09", "long",
     "وقتی مترو از ایستگاه ترمینال جنوب رد شد، خانمی که کنار پنجره نشسته بود "
     "کتابش را بست، کیفش را برداشت و به سمت در رفت، بیآنکه متوجه شود ایستگاه "
     "بعدی همان مقصدی است که نیم ساعت پیش از آن آمده بود."),
)

#: Condition B: the same words with short-vowel marks, so a tokenizer that
#: reads harakat can show it. Keys absent here -> B is skipped for that
#: sentence (recorded, never silently dropped).
DIACRITIZED: dict[str, str] = {
    "s01": "او گُل قرمز را از گِل باغچه بیرون کشید.",
    "s02": "پیرمَرد گفت این مَرد هفتهٔ پیش مُرده است.",
    "s03": "کتابِ خوبِ برادرم رویِ میزِ اتاق است.",
    "s04": "لیوانِ آب را رویِ میز بگذار و رویِ صورتت آب بکش.",
    "s05": "قَلعهٔ قدیمی بالای تَپهٔ خاکی، هِنگام غُروب دیدنی بود.",
    "s06": "ژاله زیر چَراغ، چَترش را در بارانِ شَدید بست.",
    "s07": "مَسؤول اِستقبالِ استاد از سَرسَرای دانشگاه ایستاد.",
    "s08": "بُرو بابا، این حِرفا پیشِ من خُریدار نَداره، بِشین بِبین چی میگم.",
    "t09": "پس از پایان مُسابقهٔ کُشتی، وَرزشکاران با کِشتی به جَزیره برگشتند.",
    "t10": "دخترِ همسایهٔ روبه‌رویی ما کتابِ تازهٔ نویسندهٔ مَحبوبش را اِمانت گرفت.",
    "t11": "بارانِ بَند آمد.",
}

#: The generation fields recorded for both engines so a reader can see the
#: A/B/C difference is weights, input text, and the language tag the fine-tune
#: card requires (language-agnostic for C, named for A/B) - and nothing else.
_SETTINGS_FIELDS = (
    "python", "language", "dtype", "num_step", "guidance_scale", "t_shift",
    "denoise", "postprocess_output", "layer_penalty_factor",
    "position_temperature", "class_temperature", "timeout_seconds",
    "memory_guard", "min_available_gb",
)


# --------------------------------------------------------------------------
# Sentences and jobs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Sentence:
    """One test item: the text all three conditions speak."""

    id: str
    category: str
    text: str
    diacritized: str | None
    source: str  # "probe" | "sealed" | "corpus"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "source": self.source,
            "text": self.text,
            "diacritized": self.diacritized,
        }


@dataclass(frozen=True)
class Job:
    """One render (or a recorded skip): sentence x condition."""

    id: str
    condition: str
    engine: str
    use_phonemes: bool
    input_text: str
    seed: int
    skipped: str | None = None

    @property
    def wav_name(self) -> str:
        return f"{self.id}_{self.condition}.wav"


def build_sentences(
    sealed: dict[str, Any], corpus: dict[str, Any], *, speaker: str = "mana"
) -> list[Sentence]:
    """Probes + the sealed t09-t12 (read from the manifest) + a corpus text.

    ``r01`` is the first held-out transcript of ``speaker``: a real, long,
    natural sentence from the evaluation corpus.
    """
    sentences = [
        Sentence(key, category, text, DIACRITIZED.get(key), "probe")
        for key, category, text in PROBES
    ]
    by_key = {entry["key"]: entry for entry in sealed["texts"]}
    missing = [key for key in SEALED_KEYS if key not in by_key]
    if missing:
        raise SystemExit(f"sealed manifest is missing {', '.join(missing)}")
    for key in SEALED_KEYS:
        entry = by_key[key]
        sentences.append(
            Sentence(key, entry["category"], entry["text"], DIACRITIZED.get(key), "sealed")
        )
    held_out = [
        entry
        for entry in corpus["entries"]
        if entry["speaker"] == speaker and entry["role"] == "held_out"
    ]
    if held_out:
        sentences.append(
            Sentence(
                "r01",
                "corpus_long",
                held_out[0]["transcript"],
                DIACRITIZED.get("r01"),
                "corpus",
            )
        )

    # NO_VARIANT is a promise, not a comment: if a probe is ever added
    # without a diacritised twin - or one silently loses its twin - the
    # default run would skip B more often than documented without saying so.
    ids = {sentence.id for sentence in sentences}
    unmarked = {sentence.id for sentence in sentences if not sentence.diacritized}
    expected = NO_VARIANT & ids
    if unmarked != expected:
        raise SystemExit(
            "sentences without a diacritised variant disagree with NO_VARIANT: "
            f"unmarked={sorted(unmarked)} expected={sorted(expected)}"
        )
    return sentences


def select_sentences(sentences: Sequence[Sentence], only: str | None) -> list[Sentence]:
    """Filter to ``--only s01,t10``; an unknown id is an error, not a shrug."""
    if not only:
        return list(sentences)
    wanted = [item.strip() for item in only.split(",") if item.strip()]
    known = {sentence.id for sentence in sentences}
    unknown = [item for item in wanted if item not in known]
    if unknown:
        raise SystemExit(
            f"unknown sentence id(s): {', '.join(unknown)} (have: {', '.join(sorted(known))})"
        )
    keep = set(wanted)
    return [sentence for sentence in sentences if sentence.id in keep]


def build_jobs(sentences: Sequence[Sentence], *, seed: int = DEFAULT_SEED) -> list[Job]:
    """A, B, C per sentence - B as an explicit skip when no variant exists."""
    jobs: list[Job] = []
    for sentence in sentences:
        for condition in CONDITIONS:
            engine = ENGINE_FOR[condition]
            phonemes = PHONEMES_FOR[condition]
            if condition == "B":
                if sentence.diacritized:
                    text, skip = sentence.diacritized, None
                else:
                    text, skip = "", "no diacritised variant written for this sentence"
            else:
                text, skip = sentence.text, None
            jobs.append(Job(sentence.id, condition, engine, phonemes, text, seed, skip))
    return jobs


def group_jobs_by_engine(jobs: Sequence[Job]) -> dict[str, list[Job]]:
    """First-seen order: the A/B engine first, the fine-tune last.

    The caller renders one group, calls ``studio.release()``, then the next -
    two OmniVoice workers must not overlap on this machine.
    """
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(job.engine, []).append(job)
    return groups


def expected_text(sentence: Sentence, condition: str) -> str:
    """What condition ``condition`` actually said (what CER compares to).

    For B that is the diacritised input; ``normalize_for_cer`` strips the
    marks before scoring, so A and B are scored on identical words.
    """
    if condition == "B" and sentence.diacritized:
        return sentence.diacritized
    return sentence.text


def reference_entry(corpus: dict[str, Any], speaker: str) -> dict[str, Any]:
    for entry in corpus["entries"]:
        if entry["speaker"] == speaker and entry["role"] == "reference":
            return entry
    raise SystemExit(f"corpus manifest has no reference clip for speaker {speaker!r}")


def held_out_entries(corpus: dict[str, Any], speaker: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in corpus["entries"]
        if entry["speaker"] == speaker and entry["role"] == "held_out"
    ]


# --------------------------------------------------------------------------
# G2P / normaliser capture - before any engine can load
# --------------------------------------------------------------------------
def capture_ipa(text: str, pipeline: Any) -> dict[str, Any]:
    """Raw GE2PE + the canonical IPA the pipeline would feed ``omnivoice_ipa``.

    Runs the exact prepare() the fine-tune's render will run, then replays
    the G2P call to recover the *raw* stream (prepare keeps only the
    normalised result). ``matches_pipeline`` compares the replay against
    what prepare stored: any divergence would mean the manifest describes
    phonemes the engine never saw, and it is flagged, never hidden.
    """
    from voiceclone.text.ipa_normalize import normalize_ge2pe_ipa_for_omnivoice

    prepared = pipeline.prepare(text, engine_name=ENGINE_C)
    g2p = pipeline.g2p
    chunks: list[dict[str, Any]] = []
    for chunk in prepared.chunks:
        entry: dict[str, Any] = {
            "chunk_text": chunk.text,
            "canonical_ipa": chunk.phonemes,
        }
        if g2p is None:
            entry["error"] = "g2p unavailable"
            chunks.append(entry)
            continue
        raw = g2p.phonemize(chunk.text)
        normalized = normalize_ge2pe_ipa_for_omnivoice(
            raw.words, compact_words=raw.compact_words
        )
        entry["raw_ge2pe_compact"] = (
            " ".join(raw.compact_words) if raw.compact_words else raw.phonemes
        )
        entry["raw_ge2pe_words"] = [f"{orth}\t{phones}" for orth, phones in raw.words]
        entry["normalization"] = normalized.to_dict()
        entry["matches_pipeline"] = (
            not chunk.phonemes or normalized.text == chunk.phonemes
        )
        if not entry["matches_pipeline"]:
            log.warning(
                "capture drift on %r: prepare fed %r, replay normalised to %r",
                chunk.text,
                chunk.phonemes,
                normalized.text,
            )
        chunks.append(entry)
    return {
        "used_phonemes": prepared.used_phonemes,
        "g2p_backend": prepared.g2p_backend,
        "notes": list(prepared.notes),
        "chunks": chunks,
    }


def render_config(config: AppConfig) -> AppConfig:
    """Evaluation off for the renders themselves.

    ``_finalise`` would otherwise load the speaker encoder (and, with
    transcribe_check, the recogniser) beside a loaded 2 GB worker - on this
    16 GB Mac that is exactly the coexistence the memory guard exists to
    prevent, and the listening test does not need it. ``--cer`` transcribes
    after every engine is unloaded.
    """
    return config.copy_with(**{"evaluation.enabled": False})


# --------------------------------------------------------------------------
# Probe leakage check - the sealed texts' own rule
# --------------------------------------------------------------------------
def overlaps(a: str, b: str) -> bool:
    """build_test_texts.py's rule: equality, or containment of a long half."""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return short in long and len(short) >= max(20, len(long) // 2)


def check_probe_overlap(
    sentences: Sequence[Sentence],
    *,
    splits_path: Path | None = None,
    include_extra_sources: bool = True,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Check probe sentences against training/dev text, like the sealed set.

    A probe overlapping the adaptation split would be memorised rather than
    synthesised; hits are recorded in the manifest and printed loudly (the
    sealed t09-t12 anchor the test regardless).
    """
    from voiceclone.evaluation.similarity import normalize_for_cer

    splits = splits_path or (repo_root / "eval" / "manifests" / "adaptation_splits_v1.jsonl.gz")
    if not splits.is_file():
        return {
            "available": False,
            "reason": f"{splits} not found",
            "checked": 0,
            "overlaps": [],
        }

    texts: dict[str, str] = {}
    with gzip.open(splits, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.setdefault(
                normalize_for_cer(record["text"]),
                f"{record['split']} clip {record['clip']}",
            )
    extra_note = ""
    if include_extra_sources:
        try:
            scripts_dir = str(repo_root / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from build_adaptation_splits import benchmark_sentences, development

            _, dev_texts = development(repo_root / "eval" / "corpus" / "manifest.json")
            for text in dev_texts:
                texts.setdefault(normalize_for_cer(text), "development corpus")
            for key, sentence in benchmark_sentences().items():
                texts.setdefault(normalize_for_cer(sentence), f"benchmark sentence {key}")
        except Exception as exc:  # the split check is the one that matters
            extra_note = f"extra sources skipped: {type(exc).__name__}: {exc}"

    probes = [sentence for sentence in sentences if sentence.source == "probe"]
    hits: list[str] = []
    for sentence in probes:
        own = normalize_for_cer(sentence.text)
        for other, where in texts.items():
            if overlaps(own, other):
                hits.append(f"{sentence.id} overlaps {where}")
    return {
        "available": True,
        "checked": len(texts),
        "probes_checked": len(probes),
        "rule": "CER normalisation, then equality, or containment either way of at "
        "least 20 letters and half the longer text",
        "overlaps": hits,
        **({"note": extra_note} if extra_note else {}),
    }


# --------------------------------------------------------------------------
# Manifest plumbing
# --------------------------------------------------------------------------
def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomic write: a crash mid-run must not destroy what already rendered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def render_entry(job: Job, *, reference_file: str) -> dict[str, Any]:
    return {
        "id": job.id,
        "condition": job.condition,
        "engine": job.engine,
        "input_text": job.input_text or None,
        "use_phonemes_requested": job.use_phonemes,
        "seed": job.seed,
        "reference": reference_file,
        "wav": None,
        "skipped": job.skipped,
    }


def render_group(
    studio: Any,
    jobs: Sequence[Job],
    *,
    out_dir: Path,
    reference_path: Path,
    reference_text: str,
    reference_file: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Render one engine's jobs, writing the manifest after every item."""
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        entry = render_entry(job, reference_file=reference_file)
        if job.skipped:
            print(f"  [{job.condition}] {job.id}: skipped ({job.skipped})")
            manifest["renders"].append(entry)
            write_manifest(manifest_path, manifest)
            continue

        print(f"  [{job.condition}] {job.id} on {job.engine} ...", flush=True)
        result = studio.synthesize(
            job.input_text,
            str(reference_path),
            engine=job.engine,
            reference_text=reference_text,
            seed=job.seed,
            output_path=wavs / job.wav_name,
            use_phonemes=job.use_phonemes,
            candidates=1,
        )
        entry.update(
            {
                "wav": str(result.output_path),
                "duration_s": round(result.duration, 3),
                "used_phonemes": bool(result.text.used_phonemes) if result.text else None,
                # The exact strings the engine received - proof, not inference.
                "fed_texts": [chunk.synth_text for chunk in result.text.chunks]
                if result.text
                else [],
                "notes": list(result.text.notes) if result.text else [],
                "warnings": list(result.warnings),
                "timings": {key: round(value, 3) for key, value in result.timings.items()},
            }
        )
        manifest["renders"].append(entry)
        write_manifest(manifest_path, manifest)
        print(f"      {entry['duration_s']:.1f}s -> {entry['wav']}")


def score_cer(
    studio: Any,
    manifest: dict[str, Any],
    sentences: Sequence[Sentence],
    *,
    manifest_path: Path,
) -> None:
    """Transcribe each wav with every engine already unloaded.

    Engines were released before this is called; the recogniser therefore
    loads alone. CER compares against the condition's own input text.
    """
    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.similarity import character_error_rate

    by_id = {sentence.id: sentence for sentence in sentences}
    manifest["events"].append({"event": "cer_started", "engines_unloaded": True})
    for entry in manifest["renders"]:
        if not entry.get("wav"):
            continue
        expected = expected_text(by_id[entry["id"]], entry["condition"])
        heard = studio.asr.transcribe(load_audio(entry["wav"])).text.strip()
        entry["asr_text"] = heard
        entry["cer"] = character_error_rate(expected, heard)
        marker = "n/a" if entry["cer"] is None else f"{entry['cer'] * 100:.1f}%"
        print(f"  [{entry['condition']}] {entry['id']} CER {marker}")
        write_manifest(manifest_path, manifest)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Render order: all A/B, release, then C. CER only with --cer, after release.",
    )
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--out",
        default=str(ROOT / "eval" / "runs" / "omnivoice-ipa-ab-v1"),
        help="directory for wavs/ and manifest.json",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", default=None, help="comma-separated sentence ids (s01,t10)")
    parser.add_argument("--speaker", default="mana", help="reference speaker from the corpus")
    parser.add_argument(
        "--cer",
        action="store_true",
        help="transcribe every render AFTER all engines are unloaded",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def _check_engines_available(names: Sequence[str]) -> None:
    from voiceclone.engines import list_tts_engines

    engines = list_tts_engines()
    for name in names:
        engine = engines.get(name)
        if engine is None:
            raise SystemExit(f"engine {name!r} is not registered")
        available, reason = engine.is_available()
        if not available:
            raise SystemExit(f"engine {name!r} is unavailable: {reason}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, force=True)
    out_dir = Path(args.out).expanduser()
    manifest_path = out_dir / "manifest.json"

    config = render_config(AppConfig.load(args.config))
    sealed = json.loads(
        (ROOT / "eval" / "manifests" / "test_texts_v1.json").read_text(encoding="utf-8")
    )
    corpus = json.loads((ROOT / "eval" / "corpus" / "manifest.json").read_text(encoding="utf-8"))

    sentences = select_sentences(
        build_sentences(sealed, corpus, speaker=args.speaker), args.only
    )
    jobs = build_jobs(sentences, seed=args.seed)
    reference = reference_entry(corpus, args.speaker)
    reference_path = ROOT / "eval" / "corpus" / reference["file"]

    # -- capture phonemes FIRST: before any engine exists in this process --
    from voiceclone.text.pipeline import TextPipeline

    pipeline = TextPipeline(config.text)
    captures = {sentence.id: capture_ipa(sentence.text, pipeline) for sentence in sentences}
    broken = [key for key, cap in captures.items() if not cap["used_phonemes"]]
    if broken:
        raise SystemExit(
            f"G2P produced no phonemes for {', '.join(broken)} "
            f"({pipeline.g2p_status()}). Condition C would silently fall back "
            "to script - refusing to render a meaningless comparison."
        )

    leakage = check_probe_overlap(sentences)
    if leakage.get("overlaps"):
        print(
            "WARNING: probe sentence(s) overlap training text: "
            + "; ".join(leakage["overlaps"]),
            file=sys.stderr,
        )

    _check_engines_available(sorted({job.engine for job in jobs}))

    from voiceclone.engines.omnivoice_ipa import PROVENANCE_NAME, ipa_model_dir
    from voiceclone.pipelines.studio import Studio

    ipa_settings = config.engines.omnivoice_ipa
    provenance_path = ipa_model_dir(config) / PROVENANCE_NAME
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else None
    )

    manifest: dict[str, Any] = {
        "version": 1,
        "purpose": "A/B/C listening test of the OmniVoice IPA fine-tune",
        "hypothesis": (
            "the IPA fine-tune reads Persian with acceptable pronunciation and "
            "voice cloning, compared against OmniVoice on raw and diacritised script"
        ),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            "seed": args.seed,
            "speed": 1.0,
            "candidates": 1,
            "evaluation_enabled": False,
            "cer_after_unload": bool(args.cer),
            "generation": {
                "omnivoice": {
                    field: getattr(config.engines.omnivoice, field)
                    for field in _SETTINGS_FIELDS
                },
                "omnivoice_ipa": {
                    field: getattr(ipa_settings, field) for field in _SETTINGS_FIELDS
                },
                "omnivoice_ipa_model_dir": ipa_settings.model_dir,
                "omnivoice_model_id": config.engines.omnivoice.model_id,
                "g2p_apply_to_engines": list(config.text.g2p.apply_to_engines),
            },
        },
        "reference": {
            "speaker": args.speaker,
            "file": reference["file"],
            "transcript": reference["transcript"],
            "duration_s": reference.get("duration"),
        },
        "held_out": [
            {"file": entry["file"], "transcript": entry["transcript"]}
            for entry in held_out_entries(corpus, args.speaker)
        ],
        "fine_tune": provenance,
        "sealed_texts": {
            "path": "eval/manifests/test_texts_v1.json",
            "sha256_of_texts": sealed.get("sha256_of_texts"),
        },
        "probe_leakage_check": leakage,
        "conditions": {
            "A": f"{ENGINE_A} + raw Persian script",
            "B": f"{ENGINE_A} + diacritised script (skipped where no variant)",
            "C": f"{ENGINE_C} + canonical Persian IPA (GE2PE -> normaliser)",
        },
        "sentences": [
            {**sentence.to_dict(), "ipa": captures[sentence.id]}
            for sentence in sentences
        ],
        "events": [
            {
                "event": "phonemes_captured_before_any_engine",
                "sentences": len(sentences),
            }
        ],
        "renders": [],
    }
    write_manifest(manifest_path, manifest)

    studio = Studio(config)
    groups = group_jobs_by_engine(jobs)
    print(
        f"{len(sentences)} sentence(s), {sum(1 for j in jobs if not j.skipped)} render(s) "
        f"+ {sum(1 for j in jobs if j.skipped)} skip(s) -> {manifest_path}"
    )
    for engine_name, group in groups.items():
        print(f"\n== group {engine_name} ({len(group)} jobs) ==")
        render_group(
            studio,
            group,
            out_dir=out_dir,
            reference_path=reference_path,
            reference_text=reference["transcript"],
            reference_file=reference["file"],
            manifest=manifest,
            manifest_path=manifest_path,
        )
        studio.release()
        manifest["events"].append({"event": "released", "engine": engine_name})
        write_manifest(manifest_path, manifest)

    if args.cer:
        print("\n== CER (engines unloaded) ==")
        score_cer(studio, manifest, sentences, manifest_path=manifest_path)
        studio.release()
    write_manifest(manifest_path, manifest)

    rendered = sum(1 for entry in manifest["renders"] if entry.get("wav"))
    skipped = sum(1 for entry in manifest["renders"] if entry.get("skipped"))
    print(f"\n{rendered} rendered, {skipped} skipped. Manifest: {manifest_path}")
    if not args.cer:
        print("CER not computed (rerun with --cer, or listen first - ears decide).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
