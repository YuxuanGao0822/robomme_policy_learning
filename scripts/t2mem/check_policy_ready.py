#!/usr/bin/env python3
"""Bounded T²Mem websocket metadata probe for unattended evaluation.

This helper does not initialize JAX or create an episode. Run it with the policy
environment's Python. A successful TCP or HTTP connection alone is insufficient:
the websocket greeting must identify the requested checkpoint and experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def validate_metadata(metadata: dict, checkpoint: Path, ablation: str) -> None:
    """Validate an exact completed snapshot and its served experiment identity."""
    if not isinstance(metadata, dict):
        raise ValueError("The websocket greeting must contain a metadata mapping")
    expected = json.loads((checkpoint / "metadata.json").read_text())
    progress = expected.get("progress", {})
    if expected.get("complete") is not True or progress.get("complete") is not True:
        raise ValueError("The requested snapshot must contain a completed training run")
    served_path = metadata.get("checkpoint")
    if not isinstance(served_path, str) or Path(served_path).resolve() != checkpoint:
        raise ValueError("The server loaded a different checkpoint")
    if metadata.get("checkpoint_progress") != progress:
        raise ValueError("The server training progress differs from the requested snapshot")
    required = {
        "protocol": "t2mem.v1",
        "tasks": ["PickXtimes"],
        "ablation": ablation,
        "action_horizon": 64,
        "execution_horizon": 32,
        "memory_write_stride": 16,
        "demo_write_offset": 0,
        "flow_denoising_steps": 10,
        "schedule_override": False,
        "approximations": [],
    }
    for field, value in required.items():
        if metadata.get(field) != value:
            raise ValueError(f"Server {field} is {metadata.get(field)!r}; expected {value!r}")
    memory_enabled = ablation != "memory-free"
    if metadata.get("memory_enabled") is not memory_enabled:
        raise ValueError("Server memory mode differs from the selected experiment")
    if ablation == "memory-free":
        if progress.get("mode") != "stage1":
            raise ValueError("Experiment A requires a Stage-1 snapshot")
    else:
        if progress.get("mode") != "alternating":
            raise ValueError("Experiments B/C require a completed alternating snapshot")
        if metadata.get("online_write") is not (ablation == "full"):
            raise ValueError("Server online-write mode differs from the selected experiment")
    stats = checkpoint / "assets" / "robomme" / "norm_stats.json"
    expected_digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if metadata.get("preprocessing", {}).get("normalization_sha256") != expected_digest:
        raise ValueError("The server uses different normalization statistics")


def probe(host: str, port: int, checkpoint: Path, ablation: str, timeout: float) -> dict:
    # Import transport dependencies only when executed in the server environment.
    # In particular, do not use WebsocketClientPolicy: its constructor retries
    # refused connections indefinitely and cannot supply a readiness deadline.
    import websockets.sync.client
    from openpi_client import msgpack_numpy

    deadline = time.monotonic() + timeout
    uri_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    with websockets.sync.client.connect(
        f"ws://{uri_host}:{port}",
        compression=None,
        max_size=1024 * 1024,
        open_timeout=timeout,
        close_timeout=min(timeout, 1.0),
        proxy=None,
    ) as connection:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Websocket greeting deadline expired")
        greeting = connection.recv(timeout=remaining)
        if not isinstance(greeting, bytes):
            raise ValueError("The server greeting must contain binary msgpack metadata")
        metadata = msgpack_numpy.unpackb(greeting)
        validate_metadata(metadata, checkpoint, ablation)
    return {
        "ready": True,
        "checkpoint": str(checkpoint),
        "ablation": ablation,
        "protocol": metadata["protocol"],
        "checkpoint_progress": metadata["checkpoint_progress"],
        "normalization_sha256": metadata["preprocessing"]["normalization_sha256"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ablation", choices=("memory-free", "full", "no-online-write"), required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 0 < args.timeout <= 60:
        parser.error("Use a valid TCP port and a timeout in (0, 60] seconds")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = probe(args.host, args.port, args.checkpoint, args.ablation, args.timeout)
    except Exception as error:
        print(json.dumps({"ready": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
