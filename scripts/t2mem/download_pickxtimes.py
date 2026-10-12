#!/usr/bin/env python3
"""Download only the official PickXtimes training archive on the Linux server."""

import argparse
from pathlib import Path
import platform


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Download training data only on the Linux server")
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/robomme_pickxtimes_h5")
    args = parser.parse_args()
    snapshot_download(repo_id="Yinpei/robomme_data_h5", repo_type="dataset",
                      allow_patterns=["*PickXtimes*"], local_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
