#!/usr/bin/env bash
# Run from any directory. See README.md for the five input directories.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${QMPROT_PYTHON:-$HERE/../framework/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON. Set QMPROT_PYTHON to your WSL Python executable." >&2
  exit 1
fi
export MPLBACKEND=Agg
exec "$PYTHON" "$HERE/generate_rerun.py" "$@"
