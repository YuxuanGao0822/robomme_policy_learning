#!/usr/bin/env python3
"""Index official PickXtimes training episodes and compute train-only statistics."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mme_vla_suite.models.t2mem.dataset import build_manifest, compute_norm_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-train-root", required=True, help="Extracted official training HDF5 release")
    parser.add_argument("--output-dir", default="data/t2mem")
    parser.add_argument("--max-episodes", type=int, help="Small training subset for smoke tests only")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Manifest already exists: {output}; choose another output directory")
    manifest = build_manifest(args.official_train_root, output / "manifest.json", max_episodes=args.max_episodes)
    compute_norm_stats(manifest, output / "assets" / "robomme")
    print(f"Indexed train-only episodes: {manifest['task_counts']}")
    print(f"Manifest: {output / 'manifest.json'}; normalization: {output / 'assets/robomme/norm_stats.json'}")


if __name__ == "__main__":
    main()
