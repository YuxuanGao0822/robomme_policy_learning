#!/usr/bin/env bash
set -euo pipefail

# Run via nohup on the Linux training server; no scheduler or sudo is needed.
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This pipeline runs on the Linux GPU server." >&2
    exit 1
fi
if [[ -f "$HOME/.config/t2mem/env.sh" ]]; then
    source "$HOME/.config/t2mem/env.sh"
fi
export PATH="$HOME/.local/bin:$PATH"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --no-sync python scripts/t2mem/run_pipeline.py "$@"
