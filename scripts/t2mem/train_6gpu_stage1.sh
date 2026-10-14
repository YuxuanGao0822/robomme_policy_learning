#!/usr/bin/env bash
set -euo pipefail

# Defaults to a new run on GPUs 1–6 after the earlier GPU 6 process is stopped.
# Pass --config runs/environment/t2mem_6gpu_tuned.yaml after server-side tuning.
# Explicit --stage resume --resume PATH preserves an existing checkpoint's
# progress and optimizer state; choose a separate --checkpoint-dir to migrate.
# Override this wrapper explicitly with T2MEM_DP_DEVICES; an inherited single-
# GPU CUDA_VISIBLE_DEVICES is replaced. Direct train.py respects CUDA selection.
export CUDA_VISIBLE_DEVICES="${T2MEM_DP_DEVICES:-1,2,3,4,5,6}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --no-sync python scripts/t2mem/train.py \
    --config configs/t2mem/t2mem_6gpu.yaml \
    --stage stage1 \
    --num-devices 6 \
    --gpu-id 1 \
    --save-interval 5000 \
    "$@"
