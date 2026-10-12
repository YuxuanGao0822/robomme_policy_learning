#!/usr/bin/env bash
# Run manually on Linux; never execute this dependency installation on the Mac.
set -euo pipefail
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "T²Mem dependency installation is restricted to the Linux server." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 2
fi
git submodule init
# The committed public submodule URL uses SSH. A fresh server needs no GitHub
# SSH key when this checkout's local URL is overridden to HTTPS.
git config submodule.third_party/robomme_benchmark.url https://github.com/RoboMME/robomme_benchmark.git
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --group dev
uv pip install --python .venv/bin/python -e .
# Keep simulator's PyTorch/ManiSkill dependencies separate from pinned JAX.
uv venv --python 3.11 .venv-robomme
uv pip install --python .venv-robomme/bin/python \
    "git+https://github.com/YinpeiDai/ManiSkill.git@dev" \
    "numpy==1.26.4" "opencv-python==4.10.0.84" "imageio==2.36.1" \
    "moviepy==2.2.1" \
    -e third_party/robomme_benchmark -e packages/openpi-client
mkdir -p runs/environment
uv pip freeze --python .venv/bin/python > runs/environment/policy_freeze.txt
uv pip freeze --python .venv-robomme/bin/python > runs/environment/simulator_freeze.txt
git rev-parse HEAD > runs/environment/policy_git_revision.txt
git -C third_party/robomme_benchmark rev-parse HEAD > runs/environment/benchmark_git_revision.txt
echo "Policy: .venv/bin/python; simulator: .venv-robomme/bin/python"
echo "GPU tests and simulator smoke tests must now be run manually."
