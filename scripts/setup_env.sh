#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Persian Voice Cloning Studio - environment bootstrap
#
#      ./scripts/setup_env.sh              # core + F5-TTS engine
#      ./scripts/setup_env.sh --full       # + MOSS-TTS, denoisers, metrics
#      ./scripts/setup_env.sh --seedvc     # + an isolated env for Seed-VC
#
#  Python 3.11 is the target. 3.10 and 3.12 also work.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
FULL=0
SEEDVC=0

for arg in "$@"; do
    case "$arg" in
        --full)   FULL=1 ;;
        --seedvc) SEEDVC=1 ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# -- locate a suitable interpreter ------------------------------------------
PYTHON=""
for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        case "$version" in
            3.10|3.11|3.12) PYTHON="$candidate"; break ;;
        esac
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.10-3.12 is required but was not found." >&2
    echo "  macOS:  brew install python@3.11" >&2
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv" >&2
    exit 1
fi

echo "==> Using $PYTHON ($("$PYTHON" --version 2>&1))"

# -- ffmpeg is needed for mp3/m4a/video input -------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "==> Warning: ffmpeg not found. Install it for broad audio format support:"
    echo "      macOS:  brew install ffmpeg"
    echo "      Ubuntu: sudo apt install ffmpeg"
fi

# -- virtual environment ----------------------------------------------------
if [ ! -d "$VENV" ]; then
    echo "==> Creating virtual environment at .venv"
    "$PYTHON" -m venv "$VENV"
fi

PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

echo "==> Upgrading pip"
"$PIP" install --quiet --upgrade pip wheel setuptools

echo "==> Installing core requirements (this takes a few minutes)"
"$PIP" install -r "${REPO_ROOT}/requirements.txt"

# soxr gives the highest-quality resampling; it is small and pure win.
"$PIP" install --quiet soxr || true

if [ "$FULL" -eq 1 ]; then
    echo "==> Installing optional extras"
    "$PIP" install -r "${REPO_ROOT}/requirements-optional.txt" || {
        echo "==> Some optional packages failed to install. That is fine -"
        echo "    every one of them degrades gracefully. See the Diagnostics tab."
    }
fi

# -- Seed-VC lives in its own environment -----------------------------------
if [ "$SEEDVC" -eq 1 ]; then
    SEEDVC_DIR="${SEEDVC_DIR:-$HOME/seed-vc}"
    if [ ! -d "$SEEDVC_DIR" ]; then
        echo "==> Cloning Seed-VC into $SEEDVC_DIR"
        git clone --depth 1 https://github.com/Plachtaa/seed-vc "$SEEDVC_DIR"
    fi
    if [ ! -d "${SEEDVC_DIR}/.venv" ]; then
        echo "==> Creating an isolated environment for Seed-VC"
        echo "    (it pins transformers 4.46 / numpy 1.26, which conflict with F5-TTS)"
        "$PYTHON" -m venv "${SEEDVC_DIR}/.venv"
        "${SEEDVC_DIR}/.venv/bin/pip" install --quiet --upgrade pip wheel
        "${SEEDVC_DIR}/.venv/bin/pip" install -r "${SEEDVC_DIR}/requirements.txt" || {
            echo "==> Seed-VC dependencies failed to install. Voice conversion will be"
            echo "    unavailable; the ASR->TTS cascade still works."
        }
    fi
    echo "==> Seed-VC ready at $SEEDVC_DIR"
fi

# -- report -----------------------------------------------------------------
echo
echo "==> Verifying the installation"
"$PY" - <<'PYCODE'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
try:
    from voiceclone.engines import engine_status
    for row in engine_status():
        mark = "OK " if row["available"] else "-- "
        note = "" if row["available"] else f"  ({row['reason']})"
        print(f"  {mark}[{row['kind']}] {row['name']}: {row['display_name']}{note}")
except Exception as exc:
    print(f"  !! Verification failed: {exc}")
PYCODE

cat <<'EOF'

Setup complete.

  1. Download model weights:
       .venv/bin/python scripts/download_models.py

  2. Launch the studio:
       .venv/bin/python app.py

  3. Or use the CLI:
       .venv/bin/python -m voiceclone.cli doctor

EOF
