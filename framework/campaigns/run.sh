#!/usr/bin/env bash
# Run every experiment JSON in one campaign folder.
# Usage, from anywhere: framework/campaigns/run.sh <campaign-id> [--dry-run]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec python "$HERE/run_campaign.py" "$@"
