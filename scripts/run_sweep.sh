#!/usr/bin/env bash
# Run the benchmark once per hypothesis, changing exactly one thing each time.
#
# Every run shares the corpus, the sentences, the seed and the device, so the
# only difference between two results files is the flag named in the label.
# Sequential on purpose: two synthesis processes on one 16 GB laptop would
# swap, and a measurement taken under memory pressure is a measurement of the
# memory pressure.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOGS=outputs/logs
mkdir -p "$LOGS"

DEVICE="${DEVICE:-mps}"
SENTENCES="${SENTENCES:-3}"

run() {
  local label="$1"; shift
  if [ -f "eval/results/${label}.json" ]; then
    echo "== ${label}: already measured, skipping"
    return 0
  fi
  echo "== ${label}: $* "
  $PY -u scripts/benchmark.py --label "$label" --device "$DEVICE" \
      --max-sentences "$SENTENCES" "$@" > "$LOGS/${label}.log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "== ${label}: FAILED (exit $status), see $LOGS/${label}.log"
  else
    grep -E "held-out similarity|round-trip CER|position on scale" "$LOGS/${label}.log"
  fi
  return 0
}

# Trimming removes about a quarter of every reference (9.9s -> 7.4s on cv02).
# Two reasons that might cost quality: a zero-shot cloner has less timbre to
# work from, and F5 is handed a transcript describing the *untrimmed* clip, so
# text and audio no longer describe the same utterance.
run trim-off --preprocess-option trim_silence=false

# Measured separately: denoise=auto already resolves to a no-op on clips this
# clean (identical embeddings for 7 of 9 speakers), so this is expected to be
# inert. Run anyway - "expected to be inert" is a prediction, not a result.
run denoise-off --preprocess-option denoise=none

# Classifier-free guidance decides how hard the model is pushed toward its
# conditioning, which is where speaker identity enters. The likeliest knob.
run cfg25 --engine-option cfg_strength=2.5
run cfg15 --engine-option cfg_strength=1.5

# More ODE steps: fidelity, at proportional cost.
run nfe64 --engine-option nfe_step=64

echo "== sweep done"
