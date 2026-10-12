#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
exec uv run python scripts/t2mem/prepare_data.py "$@"
