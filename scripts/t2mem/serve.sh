#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
# One A100 is shared with the simulator. This is a starting budget, not a
# measured simulator footprint. Disable JAX preallocation to leave free memory.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/t2mem/serve.sh CHECKPOINT [--ablation full|memory-free|no-online-write] [other serve.py options]" >&2
    exit 2
fi
checkpoint="$1"
shift
exec uv run scripts/t2mem/serve.py --checkpoint "$checkpoint" "$@"
