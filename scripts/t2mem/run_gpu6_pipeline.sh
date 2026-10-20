#!/usr/bin/env bash
set -euo pipefail

# Continue the completed Stage-1 model on physical GPU6 only.
# The shared pipeline directory lets --replace-waiter stop the old idle queue.
if (( $# != 0 )); then
    echo "This preset takes no arguments; use run_pipeline.sh for custom options." >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_pipeline.sh" \
    --stage1-ready \
    --replace-waiter \
    --devices 6 --policy-gpu 6 --sim-gpu 6 \
    --gpu-policy exclusive \
    --check-interval 2 --poll-interval 600 \
    --cycles 4
