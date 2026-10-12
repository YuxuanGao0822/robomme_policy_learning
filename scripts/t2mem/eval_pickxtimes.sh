#!/usr/bin/env bash
set -euo pipefail

# Activate the separate RoboMME environment; start serve.sh in another terminal.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec "${ROBOMME_PYTHON:-$repo_root/.venv-robomme/bin/python}" examples/robomme/eval.py --t2mem \
    --tasks PickXtimes \
    --seeds 7,17,27 \
    --episodes-per-task 50 \
    "$@"
