#!/usr/bin/env python3
"""Measure six-GPU Stage1 batches and write a throughput-selected fresh-run config."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from mme_vla_suite.models.t2mem.configuration import load_run_config


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def select_fastest(reports: list[dict]) -> dict:
    usable = [report for report in reports if math.isfinite(report["samples_per_second"])
              and report["samples_per_second"] > 0]
    if not usable:
        raise ValueError("No successful batch measurements; no training config was written")
    return max(usable, key=lambda report: report["samples_per_second"])


def is_memory_failure(log: str) -> bool:
    lowered = log.lower()
    return any(marker in lowered for marker in (
        "resource_exhausted", "out of memory", "cuda_error_out_of_memory",
        "failed to allocate", "unable to allocate device memory",
    ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/t2mem/t2mem_6gpu.yaml")
    parser.add_argument("--devices", default=os.environ.get("T2MEM_DP_DEVICES", "1,2,3,4,5,6"))
    parser.add_argument("--candidates", type=positive_int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--steps", type=positive_int, default=3)
    parser.add_argument("--warmup-steps", type=positive_int, default=2)
    parser.add_argument("--benchmark-root", help="New directory for trial logs and measurements")
    parser.add_argument("--output-config", default="runs/environment/t2mem_6gpu_tuned.yaml")
    return parser.parse_args(argv)


def require_idle_gpus(physical: list[str]) -> None:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True,
    )
    selected = {row[1].strip(): row[0].strip() for row in csv.reader(inventory.splitlines())
                if row[0].strip() in physical}
    if len(selected) != len(physical):
        raise ValueError(f"Some selected GPUs are unavailable: {physical}")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True,
    )
    busy = [(selected[row[0].strip()], row[1].strip()) for row in csv.reader(processes.splitlines())
            if len(row) >= 2 and row[0].strip() in selected]
    if busy:
        raise RuntimeError(f"Selected GPUs already have compute processes (GPU, PID): {busy}. "
                           "Stop the current training before benchmarking these GPUs.")


def main() -> None:
    args = parse_args()
    os.chdir(REPOSITORY)
    config = load_run_config(args.config)
    physical = args.devices.split(",")
    if (config.num_devices != 6 or len(physical) != 6 or len(set(physical)) != 6
            or any(not value.isdigit() for value in physical)):
        raise ValueError("Select exactly six distinct physical GPUs")
    config.gpu_id = int(physical[0])
    output = Path(args.output_config).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output config already exists: {output}; choose a new --output-config")
    require_idle_gpus(physical)
    root = (Path(args.benchmark_root).expanduser() if args.benchmark_root else
            Path("runs/benchmarks") / f"stage1_batch_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.devices
    # A fresh child per candidate releases JAX/XLA buffers on exit, including
    # initialization caches on the first device. Reserve some driver headroom.
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    reports, failures = [], []
    for micro in sorted(set(args.candidates)):
        trial = root / f"per_device_{micro}"
        logfile = root / f"per_device_{micro}.log"
        command = [
            sys.executable, str(REPOSITORY / "scripts/t2mem/train.py"),
            "--config", str(Path(args.config).resolve()), "--stage", "stage1",
            "--num-devices", "6", "--gpu-id", str(config.gpu_id),
            "--stage1-per-device-batch", str(micro), "--stage1-accumulation-steps", "1",
            "--benchmark-steps", str(args.steps), "--benchmark-warmup-steps", str(args.warmup_steps),
            "--checkpoint-dir", str(trial),
        ]
        print(f"Testing per-GPU batch={micro}, global={6 * micro}; log={logfile}", flush=True)
        with logfile.open("w") as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            memory_failure = is_memory_failure(logfile.read_text(errors="replace"))
            failures.append({"per_device_batch": micro, "exit_code": result.returncode,
                             "memory_failure": memory_failure, "log": str(logfile)})
            (root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
            if memory_failure:
                print(f"Batch {micro} exceeded memory capacity; stopping the larger-batch sweep.", flush=True)
                break
            raise RuntimeError(f"Trial failed for a reason other than memory capacity; inspect {logfile}")
        report = json.loads((trial / "benchmark.json").read_text())
        reports.append(report)
        (root / "measurements.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"Batch {micro}: {report['samples_per_second']:.3f} samples/s", flush=True)
    best = select_fastest(reports)
    config.training.stage1_per_device_batch_size = best["per_device_batch"]
    config.training.stage1_accumulation_steps = 1
    config.initialization_source = None
    config.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))
    selection = {"config": str(output), "selected": best, "measurements": str(root / "measurements.json"),
                 "failures": failures, "training_started": False}
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected per-GPU batch={best['per_device_batch']}, global={best['global_batch']}; "
          f"{best['samples_per_second']:.3f} samples/s\nConfig: {output}\n"
          f"Start fresh Stage1: T2MEM_DP_DEVICES={shlex.quote(args.devices)} "
          f"bash scripts/t2mem/train_6gpu_stage1.sh --config {shlex.quote(str(output))} "
          f"--gpu-id {config.gpu_id}", flush=True)


if __name__ == "__main__":
    main()
