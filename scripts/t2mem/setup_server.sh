#!/usr/bin/env bash
# User-space installation on Linux; never execute it on the development Mac.
set -euo pipefail
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "T²Mem dependency installation is restricted to the Linux server." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v git >/dev/null 2>&1; then
    echo "git must be available through the server, a cluster module, or your user environment." >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    # The official standalone installer needs no system Python or root access.
    t2mem_uv_installer="$(mktemp)"
    trap 'rm -f -- "$t2mem_uv_installer"' EXIT
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh -o "$t2mem_uv_installer"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$t2mem_uv_installer" https://astral.sh/uv/install.sh
    else
        echo "Install uv in your user directory; curl or wget is needed for automatic bootstrap." >&2
        exit 2
    fi
    UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$t2mem_uv_installer"
    rm -f -- "$t2mem_uv_installer"
    trap - EXIT
fi
# uv downloads a missing compatible interpreter to its user-space storage.
# A path to an existing Python 3.11 can be supplied for an offline/module setup.
t2mem_python="${T2MEM_PYTHON:-3.11}"
git submodule init
# The committed public submodule URL uses SSH. A fresh server needs no GitHub
# SSH key when this checkout's local URL is overridden to HTTPS.
git config submodule.third_party/robomme_benchmark.url https://github.com/RoboMME/robomme_benchmark.git
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --python "$t2mem_python" --group dev
uv pip install --python .venv/bin/python --no-deps -e .
# Keep simulator's PyTorch/ManiSkill dependencies separate from pinned JAX.
uv venv --python "$t2mem_python" .venv-robomme
uv pip install --python .venv-robomme/bin/python \
    "mani-skill @ git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd" \
    "numpy==1.26.4" "opencv-python==4.11.0.86" "imageio==2.36.1" \
    "imageio-ffmpeg>=0.6.0" \
    "moviepy==2.2.1" \
    -e third_party/robomme_benchmark -e packages/openpi-client
mkdir -p runs/environment
uv pip freeze --python .venv/bin/python > runs/environment/policy_freeze.txt
uv pip freeze --python .venv-robomme/bin/python > runs/environment/simulator_freeze.txt
git rev-parse HEAD > runs/environment/policy_git_revision.txt
git -C third_party/robomme_benchmark rev-parse HEAD > runs/environment/benchmark_git_revision.txt
echo "Policy: .venv/bin/python; simulator: .venv-robomme/bin/python"
echo "No sudo or system package changes were used. NVIDIA/Vulkan drivers must already be available."
echo "GPU tests and simulator smoke tests must now be run manually."
