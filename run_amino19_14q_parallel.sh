#!/usr/bin/env bash
# Run from WSL. Pass --help for worker, output, and resume options.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$repo_dir/framework/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  printf 'WSL Python not found at %s. Set up framework/.venv first.\n' "$python" >&2
  exit 1
fi

export PYTHONPATH="$repo_dir/framework${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m experiments.amino19_parallel "$@"
