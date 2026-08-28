#!/usr/bin/env python3
"""Select over reference clips the way best-of-N selects over seeds.

The reference axis carries +0.027 of headroom against best-of-4's +0.014 on the
seed axis, and no property of a clip predicts which one to take. So take the
same trade: render from several, and choose by measurement.

**The trap.** The obvious selector - score each render against the speaker's
other recordings and keep the best - is the evaluation metric itself. Selecting
on it and then reporting it would inflate the number by exactly the amount the
selection chose, and report the inflation as a gain.

So the evaluation clip is withheld from the decision. For each held-out clip h:

    candidates      every reference i, i != h
    selection score render_i against the speaker's clips, excluding i and h
    reported score  the chosen render against h alone

h never touches the choice. Rotating h over every clip and averaging gives what
this would be worth on a recording the selector has not seen - which is the only
version of the question that means anything.

No rendering: the five reference sweeps are already on disk.

    .venv/bin/python scripts/best_of_references.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

LABELS = {
    0: "finetuned",
    1: "finetuned-ref1",
    2: "finetuned-ref2",
    3: "finetuned-ref3",
    4: "finetuned-ref4",
}
SENTENCES = ("s1", "s2", "s3")


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "eval" / "corpus")
    parser.add_argument("--renders", type=Path, default=REPO_ROOT / "outputs" / "benchmark")
    args = parser.parse_args()

    from voiceclone.audio.io import load_audio
    from voiceclone.evaluation.embedding import CachedEmbedder

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    clips: dict[str, list[dict]] = {}
    for entry in manifest["entries"]:
        clips.setdefault(entry["speaker"], []).append(entry)
    for group in clips.values():
        group.sort(key=lambda e: e["file"])
    speakers = sorted(clips)

    embedder = CachedEmbedder()

    _rule("Loading")
    real: dict[tuple[str, int], np.ndarray] = {}
    for speaker in speakers:
        for index, entry in enumerate(clips[speaker]):
            path = args.corpus / entry["file"]
            real[(speaker, index)] = embedder.embed(load_audio(path), key=str(path))

    # renders[(speaker, sentence, reference index)]
    renders: dict[tuple[str, str, int], np.ndarray] = {}
    for index, label in LABELS.items():
        for speaker in speakers:
            for sentence in SENTENCES:
                path = args.renders / label / f"{speaker}_{sentence}_seed42.wav"
                if path.is_file():
                    renders[(speaker, sentence, index)] = embedder.embed(
                        load_audio(path), key=str(path)
                    )
    print(f"  {len(real)} recordings, {len(renders)} renders, backend {embedder.backend}")

    _rule("Selecting over references, evaluation clip withheld")
    print("  For each speaker, sentence and withheld clip: choose a reference by")
    print("  scoring against the other recordings, then report against the")
    print("  withheld one only.\n")
    print(f"  {'speaker':9s} {'default':>9s} {'selected':>9s} {'gain':>8s} {'oracle':>8s} {'n':>4s}")

    totals: dict[str, list[float]] = {"default": [], "selected": [], "oracle": []}
    per_speaker: list[tuple[str, float, float, float, int]] = []

    for speaker in speakers:
        available = sorted({i for (s, _, i) in renders if s == speaker})
        rows = {"default": [], "selected": [], "oracle": []}

        for sentence in SENTENCES:
            usable = [i for i in available if (speaker, sentence, i) in renders]
            if len(usable) < 2:
                continue
            for held in range(len(clips[speaker])):
                candidates = [i for i in usable if i != held]
                if len(candidates) < 2:
                    continue
                target = real[(speaker, held)]

                def score_against_rest(
                    index: int, speaker: str = speaker, sentence: str = sentence, held: int = held
                ) -> float:
                    pool = [
                        real[(speaker, j)]
                        for j in range(len(clips[speaker]))
                        if j not in (index, held)
                    ]
                    render = renders[(speaker, sentence, index)]
                    return float(np.mean([cosine(render, p) for p in pool]))

                chosen = max(candidates, key=score_against_rest)
                fallback = candidates[0]

                rows["default"].append(cosine(renders[(speaker, sentence, fallback)], target))
                rows["selected"].append(cosine(renders[(speaker, sentence, chosen)], target))
                rows["oracle"].append(
                    max(cosine(renders[(speaker, sentence, i)], target) for i in candidates)
                )

        if not rows["default"]:
            continue
        d, s_, o = (float(np.mean(rows[k])) for k in ("default", "selected", "oracle"))
        per_speaker.append((speaker, d, s_, o, len(rows["default"])))
        for k in totals:
            totals[k].extend(rows[k])
        print(f"  {speaker:9s} {d:9.3f} {s_:9.3f} {s_ - d:+8.3f} {o:8.3f} {len(rows['default']):4d}")

    if not totals["default"]:
        print("Nothing to compare.", file=sys.stderr)
        return 1

    default = np.array(totals["default"])
    selected = np.array(totals["selected"])
    oracle = np.array(totals["oracle"])
    difference = selected - default

    _rule("Overall")
    print(f"  decisions            {len(difference)}")
    print(f"  default              {default.mean():.4f}")
    print(f"  selected             {selected.mean():.4f}")
    print(f"  oracle               {oracle.mean():.4f}")
    print(f"\n  gain from selecting  {difference.mean():+.4f}")
    error = difference.std(ddof=1) / np.sqrt(len(difference))
    print(f"  standard error       {error:.4f}   ({difference.mean() / error:.1f} SE from zero)")
    headroom = float((oracle - default).mean())
    share = difference.mean() / headroom * 100 if headroom else 0.0
    print(f"  oracle headroom      {headroom:+.4f}   captured {share:.0f}%")
    # A tie means the selector chose the clip the default would have used, so
    # most of the "unchanged" column is the rule agreeing rather than failing.
    ties = int(np.isclose(difference, 0).sum())
    better = int((difference > 1e-9).sum())
    worse = len(difference) - ties - better
    print(f"  changed the pick     {better + worse} of {len(difference)}")
    print(f"    better             {better}")
    print(f"    worse              {worse}")
    print(f"    same clip chosen   {ties}")

    _rule("Reading it")
    if difference.mean() > 0.0205:
        print("  Above the encoder's noise, chosen without seeing the target.")
        print("  Selecting over references is worth doing.")
    elif difference.mean() > 0:
        print("  Positive but under the encoder's noise for a single render.")
        print("  Worth what the paired figure says it is, not more.")
    else:
        print("  No gain. Scoring a render against the speaker's other recordings")
        print("  does not identify which reference produced it best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
