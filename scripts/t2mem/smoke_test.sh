#!/usr/bin/env bash
# Linux-only GPU tests and three optimizer updates. This is not an experiment.
set -euo pipefail
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "GPU smoke tests require Linux A100; Mac permits static checks only." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
uv run pytest -q tests/test_t2mem*.py
uv run python scripts/t2mem/train.py --config configs/t2mem/t2mem_smoke.yaml --stage stage1 \
    --checkpoint-dir runs/ckpts/t2mem_smoke_stage1
uv run python scripts/t2mem/train.py --config configs/t2mem/t2mem_smoke.yaml --stage alternating \
    --init-checkpoint runs/ckpts/t2mem_smoke_stage1/1 --checkpoint-dir runs/ckpts/t2mem_smoke_stage2 --cycles 1
