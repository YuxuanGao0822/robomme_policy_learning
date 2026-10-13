"""Download π0.5-base and verify cached or transferred filenames and sizes."""

import argparse
import logging

from openpi.shared import download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Replace the existing π0.5-base cache with a fresh download"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    path = download.maybe_download(
        "gs://openpi-assets/checkpoints/pi05_base",
        force_download=args.force,
        verify_cache=True,
        # gcsfs's size checker needs no extra dependency and works for composite
        # objects that have no MD5. Metadata comparison also checks remote sizes.
        gs={"token": "anon", "consistency": "size"},
    )
    print(f"Verified π0.5-base cache: {path}")


if __name__ == "__main__":
    main()
