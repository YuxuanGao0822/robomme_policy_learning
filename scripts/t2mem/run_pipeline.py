#!/usr/bin/env python3
"""Wait for an existing Stage-1 job, then queue Stage 2 and matched A/B/C evaluations.

The supervisor uses only Python's standard library. Model code runs exclusively
in child processes on the Linux server; importing this file starts no jobs.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("t2mem.pipeline")


class PipelineCancelled(RuntimeError):
    pass


class WorkerDependencyExited(RuntimeError):
    pass


class WorkerFailure(RuntimeError):
    pass


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _device_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated physical GPU indices") from error
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("GPU indices must be nonnegative and unique")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=_positive_int, default=1776511)
    parser.add_argument("--poll-interval", type=_positive_int, default=600)
    parser.add_argument("--stage1-root", type=Path, default=Path("runs/ckpts/pickxtimes_stage1_6gpu"))
    parser.add_argument("--stage1-steps", type=_positive_int, default=20000)
    parser.add_argument("--stage2-root", type=Path, default=Path("runs/ckpts/pickxtimes_t2mem_6gpu"))
    parser.add_argument("--pipeline-dir", type=Path, default=Path("runs/pipeline/pickxtimes_6gpu"))
    parser.add_argument("--cycles", type=_positive_int, default=4)
    parser.add_argument("--devices", type=_device_list, default=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--policy-gpu", type=int, default=6)
    parser.add_argument("--sim-gpu", type=int, default=6)
    parser.add_argument("--port", type=_positive_int, default=8090)
    parser.add_argument("--eval-retries", type=_positive_int, default=3)
    parser.add_argument("--server-ready-timeout", type=_positive_int, default=900)
    parser.add_argument("--eval-idle-timeout", type=_positive_int, default=3600,
                        help="Restart an evaluator with no completed-episode progress for this many seconds")
    parser.add_argument("--seeds", type=_device_list, default=(7, 17, 27))
    parser.add_argument("--episodes-per-task", type=_positive_int, default=50)
    parser.add_argument("--max-steps", type=_positive_int, default=1300)
    parser.add_argument("--sim-python", type=Path, default=Path(".venv-robomme/bin/python"))
    args = parser.parse_args(argv)
    if args.policy_gpu < 0 or args.sim_gpu < 0 or args.port > 65535:
        parser.error("GPU indices must be nonnegative and the port must be 1–65535")
    if args.policy_gpu not in args.devices or args.sim_gpu not in args.devices:
        parser.error("Evaluation GPUs must be members of --devices")
    if args.episodes_per_task != 50 or len(args.seeds) != 3:
        parser.error("This formal pipeline requires 50 episodes and three unique seeds per model")
    return args


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def progress_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def read_process_identity(pid: int, proc_root: Path = Path("/proc")) -> dict | None:
    """Return Linux PID/start-time/owner identity, including zombie state."""
    proc = Path(proc_root) / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        uid_line = next(line for line in (proc / "status").read_text().splitlines() if line.startswith("Uid:"))
        return {"pid": pid, "start_time": fields[19], "uid": int(uid_line.split()[1]), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (IndexError, StopIteration, ValueError) as error:
        raise RuntimeError(f"Cannot identify Linux process {pid}") from error


def same_process(expected: dict, current: dict | None) -> bool:
    return bool(current is not None and current.get("state") not in ("Z", "X")
                and all(str(expected.get(field)) == str(current.get(field))
                        for field in ("pid", "start_time", "uid")))


def _read_snapshot(path: Path) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Checkpoint must be a real directory: {path}")
    marker = path / "metadata.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(f"Checkpoint has no real commit marker: {path}")
    metadata = json.loads(marker.read_text())
    if (not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int
            or metadata.get("format_version") != 1 or metadata.get("complete") is not True):
        raise ValueError(f"Checkpoint save was not committed: {path}")
    progress = metadata.get("progress")
    if not isinstance(progress, dict) or type(progress.get("global_step")) is not int:
        raise ValueError(f"Checkpoint has no valid optimizer progress: {path}")
    if not path.name.isdigit() or int(path.name) != progress["global_step"]:
        raise ValueError(f"Checkpoint directory and optimizer step disagree: {path}")
    if not isinstance(metadata.get("config"), dict):
        raise ValueError(f"Checkpoint has no saved configuration: {path}")
    for part in ("params", "train_state"):
        if (path / part).is_symlink() or not (path / part).is_dir():
            raise ValueError(f"Checkpoint is missing {part}: {path}")
    norm = path / "assets/robomme/norm_stats.json"
    if norm.is_symlink() or not norm.is_file() or norm.stat().st_size == 0:
        raise ValueError(f"Checkpoint has no immutable normalization statistics: {path}")
    # A truncated/non-JSON statistics file must never be handed to training.
    if not isinstance(json.loads(norm.read_text()), dict):
        raise ValueError(f"Invalid normalization statistics: {norm}")
    return metadata


def latest_complete_checkpoint(run_root: Path) -> Path | None:
    """Find the last committed snapshot, even when its training budget is unfinished."""
    if not run_root.exists():
        return None
    for directory in sorted((p for p in run_root.iterdir() if p.name.isdigit()),
                            key=lambda p: int(p.name), reverse=True):
        try:
            _read_snapshot(directory)
        except (OSError, ValueError, TypeError):
            continue
        return directory
    return None


def _source_path(value) -> Path | None:
    if isinstance(value, dict):
        value = value.get("checkpoint", value.get("path"))
    return Path(value).expanduser().resolve() if isinstance(value, str) and value else None


def validate_checkpoint(path: Path, mode: str, required_steps: int, *, cycles: int | None = None,
                        source: Path | None = None, require_complete: bool = True) -> dict:
    metadata = _read_snapshot(path)
    progress, config = metadata["progress"], metadata["config"]
    if progress.get("mode") != mode:
        raise ValueError(f"Expected {mode} checkpoint: {path}")
    if require_complete and progress.get("complete") is not True:
        raise ValueError(f"Configured training budget is unfinished: {path}")
    if progress["global_step"] < required_steps:
        raise ValueError(f"Checkpoint has {progress['global_step']} steps; requires {required_steps}: {path}")
    if mode == "stage1" and require_complete:
        if (progress.get("phase") != "stage1" or progress.get("stage_updates", {}).get("stage1") != progress["global_step"]
                or progress.get("phase_step") != progress["global_step"]):
            raise ValueError(f"Stage-1 counters disagree with saved optimizer progress: {path}")
    if metadata.get("approximations"):
        raise ValueError(f"Formal pipeline requires full observation history: {path}")
    if source is not None:
        recorded_source = _source_path(metadata.get("initialization_source", config.get("initialization_source")))
        if recorded_source != source.resolve():
            raise ValueError(f"Checkpoint was not initialized from the selected Stage-1 model: {path}")
        if hashlib.sha256((source / "assets/robomme/norm_stats.json").read_bytes()).digest() != hashlib.sha256(
                (path / "assets/robomme/norm_stats.json").read_bytes()).digest():
            raise ValueError(f"Stage-1 and Stage-2 normalization statistics differ: {path}")
    if cycles is not None:
        training = config.get("training", {})
        if training.get("alternating_cycles") != cycles:
            raise ValueError(f"Stage-2 cycle budget differs from this pipeline: {path}")
        budgets = {"stage2a": training.get("memory_updates_per_cycle"),
                   "stage2b": training.get("policy_updates_per_cycle")}
        if any(type(value) is not int or value <= 0 for value in budgets.values()):
            raise ValueError(f"Invalid Stage-2 phase budgets: {path}")
        phase, cycle, phase_step = (progress.get(field) for field in ("phase", "cycle", "phase_step"))
        if (phase not in budgets or type(cycle) is not int or not 0 <= cycle < cycles
                or type(phase_step) is not int or not 0 <= phase_step <= budgets[phase]):
            raise ValueError(f"Invalid Stage-2 phase/cycle progress: {path}")
        counts = {"stage2a": cycle * budgets["stage2a"], "stage2b": cycle * budgets["stage2b"]}
        counts[phase] += phase_step
        if phase == "stage2b":
            counts["stage2a"] += budgets["stage2a"]
        if (any(progress.get("stage_updates", {}).get(stage) != count for stage, count in counts.items())
                or progress["global_step"] != sum(counts.values())
                or progress.get("stage_updates", {}).get("stage1") != 0):
            raise ValueError(f"Stage-2 optimizer counters do not match its phase/cycle: {path}")
        if require_complete:
            expected = {stage: cycles * count for stage, count in budgets.items()}
            if any(progress.get("stage_updates", {}).get(stage) != count for stage, count in expected.items()):
                raise ValueError(f"Stage-2 phase counters do not complete all cycles: {path}")
            if progress["global_step"] != sum(expected.values()) or progress.get("cycle") != cycles - 1:
                raise ValueError(f"Stage-2 total/cycle counters differ from its budget: {path}")
    return metadata


def validate_evaluation(directory: Path, checkpoint: Path, ablation: str, *, seeds=(7, 17, 27),
                        episodes: int = 50, max_steps: int = 1300) -> bool:
    """Return completion; reject provenance/protocol mismatches before retrying."""
    marker = directory / "manifest.json"
    if not marker.exists():
        if (directory / "episodes.json").exists() or (directory / "metrics.json").exists():
            raise ValueError(f"Evaluation records exist without their provenance manifest: {directory}")
        return False
    manifest = json.loads(marker.read_text())
    server = manifest.get("server", {})
    expected = {"tasks": ["PickXtimes"], "seeds": list(seeds), "episodes_per_task": episodes,
                "max_steps": max_steps, "simulation_mode": "gpu"}
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError(f"Existing evaluation has a different protocol: {directory}")
    if _source_path(server.get("checkpoint")) != checkpoint.resolve() or server.get("ablation") != ablation:
        raise ValueError(f"Existing evaluation belongs to a different checkpoint/ablation: {directory}")
    if any(server.get(field) != value for field, value in
           (("action_horizon", 64), ("execution_horizon", 32), ("memory_write_stride", 16))):
        raise ValueError(f"Existing evaluation does not implement H64/E32/stride16: {directory}")
    saved = _read_snapshot(checkpoint)
    if server.get("checkpoint_progress") != saved["progress"]:
        raise ValueError(f"Evaluation checkpoint progress differs from saved weights: {directory}")
    try:
        metrics = json.loads((directory / "metrics.json").read_text())
        records = json.loads((directory / "episodes.json").read_text())
    except FileNotFoundError:
        return False
    total = len(seeds) * episodes
    if (metrics.get("complete") is not True or metrics.get("expected_rollouts") != total
            or metrics.get("valid_rollouts") != total or metrics.get("infrastructure_errors") != 0
            or metrics.get("paper_difficulty_counts_match") is not True or not isinstance(records, list)):
        return False
    target = {("PickXtimes", seed, episode) for seed in seeds for episode in range(episodes)}
    try:
        keys = [(record["task"], record["seed"], record["episode_id"]) for record in records]
    except (KeyError, TypeError):
        return False
    if len(keys) != total or len(set(keys)) != total or set(keys) != target:
        return False
    if any(record.get("status") == "error" or type(record.get("success")) is not bool for record in records):
        return False
    if any(not isinstance(record.get("status"), str) or record["status"] == "unknown"
           or record["success"] != (record["status"] == "success") for record in records):
        return False
    counts = {f"PickXtimes/seed{seed}": dict(collections.Counter(
        record.get("difficulty") for record in records if record["seed"] == seed)) for seed in seeds}
    if any(value != {"easy": 26, "medium": 12, "hard": 12} for value in counts.values()):
        return False
    return metrics.get("difficulty_counts") == counts


def validate_stage2_configuration(saved: dict, source: dict) -> None:
    """Allow placement/save-budget overrides while preserving the selected experiment."""
    if (saved.get("model") != source.get("model") or saved.get("seed") != source.get("seed")
            or any(saved["training"].get(field) != source["training"].get(field)
                   for field in source["training"] if field not in ("alternating_cycles", "save_interval"))
            or any(saved["data"].get(field) != source["data"].get(field)
                   for field in source["data"] if field != "norm_stats_dir")):
        raise ValueError("Existing Stage-2 configuration differs from the selected Stage-1 experiment")


def gpu_inventory(runner=subprocess.run) -> dict[int, str]:
    result = runner(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True, timeout=30)
    inventory = {}
    for line in result.stdout.splitlines():
        if line.strip():
            index, uuid = (value.strip() for value in line.split(",", 1))
            inventory[int(index)] = uuid
    return inventory


def gpu_is_idle(devices: tuple[int, ...], runner=subprocess.run) -> bool:
    inventory = gpu_inventory(runner)
    if any(device not in inventory for device in devices):
        return False
    selected = {inventory[device] for device in devices}
    result = runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True, timeout=30)
    for line in result.stdout.splitlines():
        if not line.strip() or line.strip() == "No running processes found":
            continue
        uuid, _pid = (value.strip() for value in line.split(",", 1))
        if uuid in selected:
            return False
    return True


class FileLock:
    """A cooperative same-user lock; it does not reserve hardware from other users."""
    def __init__(self, path: Path):
        self.path, self.file = path, None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            self.file = None
            return False
        return True

    def release(self) -> None:
        if self.file is not None:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None


class Pipeline:
    def __init__(self, options: argparse.Namespace):
        self.options = options
        self.root = ROOT
        for field in ("stage1_root", "stage2_root", "pipeline_dir", "sim_python"):
            value = getattr(options, field).expanduser()
            absolute = self.root / value if not value.is_absolute() else value
            # Preserve the venv interpreter symlink: resolving it would run the
            # base uv Python outside .venv-robomme and lose simulator packages.
            setattr(options, field, Path(os.path.abspath(absolute)) if field == "sim_python" else absolute.resolve())
        self.directory = options.pipeline_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "logs").mkdir(exist_ok=True)
        self.cancelled = threading.Event()
        self.children: dict[int, subprocess.Popen] = {}
        self.state = {"driver_pid": os.getpid(), "status": "RUNNING", "stage": "STARTING"}
        previous = self.directory / "state.json"
        if previous.exists():
            data = json.loads(previous.read_text())
            if data.get("watched_process", {}).get("pid") == options.watch_pid:
                self.state["watched_process"] = data["watched_process"]
        self.gpu_locks: list[FileLock] = []

    def update(self, stage: str, **values) -> None:
        self.state.update(stage=stage, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values)
        atomic_json(self.directory / "state.json", self.state)
        LOGGER.info("%s %s", stage, json.dumps(values, ensure_ascii=False))

    def _pause(self, seconds: float) -> None:
        if self.cancelled.wait(seconds):
            raise PipelineCancelled("Pipeline stopped by signal; the existing Stage-1 job was left running")

    def preflight(self) -> None:
        if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
            raise RuntimeError("This supervisor must run on the Linux GPU server")
        for relative in ("scripts/t2mem/train.py", "scripts/t2mem/serve.py", "scripts/t2mem/check_policy_ready.py",
                         "scripts/t2mem/compare_results.py", "examples/robomme/eval.py"):
            if not (self.root / relative).is_file():
                raise FileNotFoundError(f"Required pipeline script is missing: {relative}; pull all new files")
        if not self.options.sim_python.is_file():
            raise FileNotFoundError(f"Simulator Python environment is missing: {self.options.sim_python}")
        inventory = gpu_inventory()
        if any(device not in inventory for device in self.options.devices):
            raise ValueError(f"Requested GPUs not present: {self.options.devices}; found {inventory}")

    def wait_for_training(self) -> Path:
        final = self.options.stage1_root / str(self.options.stage1_steps)
        observed = read_process_identity(self.options.watch_pid)
        expected = self.state.get("watched_process")
        if expected is None and observed is not None and observed.get("state") not in ("Z", "X"):
            if observed["uid"] != os.getuid():
                raise ValueError(f"Watched PID {self.options.watch_pid} belongs to another account")
            try:
                command = Path(f"/proc/{self.options.watch_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                command = ""
            current = read_process_identity(self.options.watch_pid)
            if not same_process(observed, current):
                # The process may finish (including becoming a zombie) between
                # reading stat and cmdline. Its final checkpoint is decisive.
                observed = current
            else:
                if "scripts/t2mem/train.py" not in command:
                    raise ValueError(f"Watched PID is not the T²Mem training process: {command}")
                expected = observed
                observed = current
                self.state["watched_process"] = expected
        while expected is not None and same_process(expected, observed):
            self.update("WAIT_STAGE1", watched_process=expected, last_process_state=observed["state"],
                        expected_checkpoint=str(final))
            self._pause(self.options.poll_interval)
            observed = read_process_identity(self.options.watch_pid)
        metadata = validate_checkpoint(final, "stage1", self.options.stage1_steps)
        # Training completion is proved by committed final weights, not disappearance of a PID.
        if metadata["config"]["training"].get("stage1_updates") != self.options.stage1_steps:
            raise ValueError("Stage-1 saved budget differs from the selected final step")
        self.update("STAGE1_COMPLETE", stage1_checkpoint=str(final))
        return final

    def _gpu_env(self, devices: tuple[int, ...], *, simulator: bool = False,
                 memory_fraction: str = "0.90") -> dict[str, str]:
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, devices)), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION=memory_fraction, PYTHONPATH=str(self.root / "src"))
        # ManiSkill's pinned backend selects the CUDA/render device from this
        # visible-device mapping. SAPIEN_RENDER_DEVICE is not its selector.
        return env

    def wait_for_gpus(self, devices: tuple[int, ...], stage: str) -> None:
        self.release_gpus()
        inventory = gpu_inventory()
        while True:
            if self.cancelled.is_set():
                raise PipelineCancelled("Pipeline cancelled while waiting for GPUs")
            locks = [FileLock(Path.home() / ".cache/t2mem/gpu_locks" / f"{inventory[device]}.lock")
                     for device in sorted(set(devices))]
            acquired = []
            for lock in locks:
                if not lock.acquire():
                    break
                acquired.append(lock)
            try:
                if len(acquired) == len(locks) and gpu_is_idle(devices):
                    self.gpu_locks = acquired
                    self.update(stage, devices=list(devices))
                    return
            finally:
                if self.gpu_locks != acquired:
                    for lock in acquired:
                        lock.release()
            self.update("WAIT_GPUS", next_stage=stage, devices=list(devices))
            self._pause(self.options.poll_interval)

    def release_gpus(self) -> None:
        for lock in self.gpu_locks:
            lock.release()
        self.gpu_locks = []

    def start_child(self, command: list[str], *, env: dict | None, log_name: str) -> subprocess.Popen:
        if self.cancelled.is_set():
            raise PipelineCancelled("Pipeline cancelled before starting a worker")
        logfile = self.directory / "logs" / log_name
        LOGGER.info("Starting worker: %s; output=%s", command, logfile)
        with logfile.open("a") as output:
            child = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[child.pid] = child
        return child

    def stop_child(self, child: subprocess.Popen) -> None:
        """Stop only a worker spawned in this supervisor's own new session."""
        if self.children.get(child.pid) is not child:
            return
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        self.children.pop(child.pid, None)

    def run_checked(self, command: list[str], *, env: dict | None = None,
                    log_name: str = "worker.log", allow_failure: bool = False,
                    timeout: float | None = None,
                    supervised_server: subprocess.Popen | None = None,
                    progress_path: Path | None = None,
                    idle_timeout: float | None = None) -> int:
        child = self.start_child(command, env=env, log_name=log_name)
        started = time.monotonic()
        last_progress = started
        try:
            signature = progress_signature(progress_path) if progress_path is not None else None
            while child.poll() is None:
                if supervised_server is not None and supervised_server.poll() is not None:
                    raise WorkerDependencyExited("Policy server exited while evaluation was running")
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Worker timed out after {timeout}s: {command}")
                if progress_path is not None and idle_timeout is not None:
                    current = progress_signature(progress_path)
                    if current != signature:
                        signature, last_progress = current, time.monotonic()
                    if time.monotonic() - last_progress >= idle_timeout:
                        raise TimeoutError(f"No completed evaluation episode for {idle_timeout}s; stopping this attempt")
                self._pause(1)
            code = child.returncode
            if code != 0 and not allow_failure:
                raise WorkerFailure(f"Worker exited with code {code}; inspect {self.directory / 'logs' / log_name}")
            return code
        finally:
            self.stop_child(child)

    def train_stage2(self, stage1: Path) -> Path:
        a = validate_checkpoint(stage1, "stage1", self.options.stage1_steps)
        config = copy.deepcopy(a["config"])
        training = config["training"]
        phases = (training.get("memory_updates_per_cycle"), training.get("policy_updates_per_cycle"))
        if any(type(value) is not int or value <= 0 for value in phases):
            raise ValueError("Stage-1 metadata has no valid Stage-2 phase budgets")
        total = self.options.cycles * sum(phases)
        final = self.options.stage2_root / str(total)
        latest = latest_complete_checkpoint(self.options.stage2_root)
        if latest is not None:
            metadata = validate_checkpoint(latest, "alternating", 0, cycles=self.options.cycles,
                                           source=stage1, require_complete=False)
            validate_stage2_configuration(metadata["config"], config)
            for directory in self.options.stage2_root.iterdir():
                if directory.name.isdigit() and (directory / "metadata.json").exists():
                    try:
                        _read_snapshot(directory)
                    except (OSError, ValueError, TypeError) as error:
                        raise ValueError(f"Stage-2 contains a corrupt committed marker: {directory}; inspect it before resuming") from error
            if metadata["progress"].get("complete"):
                validate_checkpoint(latest, "alternating", total, cycles=self.options.cycles, source=stage1)
                if latest != final:
                    raise ValueError("Completed Stage-2 directory does not match the expected final budget")
                self.update("STAGE2_COMPLETE", stage2_checkpoint=str(final), resumed_existing=True)
                return final
            command = [sys.executable, "scripts/t2mem/train.py", "--stage", "resume", "--resume", str(latest)]
        else:
            if self.options.stage2_root.exists() and any(self.options.stage2_root.iterdir()):
                raise FileExistsError(f"Stage-2 directory has files but no restorable checkpoint: {self.options.stage2_root}. "
                                      "Inspect it and select a new --stage2-root; the pipeline does not overwrite runs.")
            # Preserve the exact completed A configuration and immutable normalization assets.
            config["data"]["norm_stats_dir"] = str(stage1 / "assets/robomme")
            config_path = self.directory / "stage2_config.json"
            atomic_json(config_path, {"config": config})
            command = [sys.executable, "scripts/t2mem/train.py", "--config", str(config_path),
                       "--stage", "alternating", "--init-checkpoint", str(stage1)]
        overrides = ["--checkpoint-dir", str(self.options.stage2_root), "--num-devices", str(len(self.options.devices)),
                     "--gpu-id", str(self.options.devices[0]), "--cycles", str(self.options.cycles),
                     "--save-interval", "5000"]
        command += overrides
        self.wait_for_gpus(self.options.devices, "TRAIN_STAGE2")
        try:
            # A worker from a terminated supervisor can finish while this new
            # supervisor is queued. Recheck after GPU ownership is available.
            refreshed = latest_complete_checkpoint(self.options.stage2_root)
            if refreshed != latest and refreshed is not None:
                metadata = validate_checkpoint(refreshed, "alternating", 0, cycles=self.options.cycles,
                                               source=stage1, require_complete=False)
                validate_stage2_configuration(metadata["config"], config)
                if metadata["progress"].get("complete"):
                    validate_checkpoint(refreshed, "alternating", total, cycles=self.options.cycles, source=stage1)
                    if refreshed != final:
                        raise ValueError("Completed Stage-2 directory does not match the expected final budget")
                    self.update("STAGE2_COMPLETE", stage2_checkpoint=str(final), completed_while_queued=True)
                    return final
                command = [sys.executable, "scripts/t2mem/train.py", "--stage", "resume", "--resume", str(refreshed)] + overrides
            elif refreshed is None and self.options.stage2_root.exists() and any(self.options.stage2_root.iterdir()):
                raise FileExistsError(f"Stage-2 directory became nonempty while queued, without a restorable snapshot: "
                                      f"{self.options.stage2_root}; inspect it before starting another run")
            self.run_checked(command, env=self._gpu_env(self.options.devices), log_name="train_stage2.log")
        finally:
            self.release_gpus()
        validate_checkpoint(final, "alternating", total, cycles=self.options.cycles, source=stage1)
        self.update("STAGE2_COMPLETE", stage2_checkpoint=str(final))
        return final

    def _check_port_available(self) -> None:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", self.options.port))

    def wait_for_server(self, server: subprocess.Popen, checkpoint: Path, ablation: str, name: str) -> None:
        deadline = time.monotonic() + self.options.server_ready_timeout
        while True:
            if server.poll() is not None:
                raise WorkerFailure(f"Policy server exited before readiness; inspect logs/server_{name}.log")
            try:
                with socket.create_connection(("127.0.0.1", self.options.port), timeout=2):
                    break
            except (ConnectionError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Policy server was not ready within {self.options.server_ready_timeout}s")
                self._pause(2)
        # A bounded helper verifies the live websocket metadata, including exact weights and ablation.
        command = [sys.executable, "scripts/t2mem/check_policy_ready.py", "--host", "127.0.0.1",
                   "--port", str(self.options.port), "--checkpoint", str(checkpoint), "--ablation", ablation,
                   "--timeout", "5"]
        code = self.run_checked(command, env=self._gpu_env((self.options.policy_gpu,)), log_name=f"ready_{name}.log",
                                allow_failure=True, timeout=10)
        ready_log = self.directory / "logs" / f"ready_{name}.log"
        try:
            reply = json.loads(ready_log.read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError) as error:
            raise WorkerFailure(f"Readiness helper did not return a valid response: {ready_log}") from error
        if code != 0 or reply.get("ready") is not True:
            error = reply.get("error", "Readiness probe failed")
            if error.startswith("ValueError:"):
                raise ValueError(f"Policy-server metadata mismatch: {error}")
            raise WorkerFailure(error)
        if server.poll() is not None:
            raise WorkerFailure("Policy server exited during metadata verification")

    def evaluate(self, name: str, checkpoint: Path, ablation: str) -> Path:
        directory = self.directory / "evaluation" / name
        protocol = {"seeds": self.options.seeds, "episodes": self.options.episodes_per_task,
                    "max_steps": self.options.max_steps}
        if validate_evaluation(directory, checkpoint, ablation, **protocol):
            self.update("EVALUATION_COMPLETE", model=name, results=str(directory), resumed_existing=True)
            return directory
        devices = tuple(sorted({self.options.policy_gpu, self.options.sim_gpu}))
        for attempt in range(1, self.options.eval_retries + 1):
            self.wait_for_gpus(devices, f"EVALUATE_{name}")
            self.update(f"EVALUATE_{name}", model=name, attempt=attempt, results=str(directory))
            server = None
            try:
                self._check_port_available()
                server = self.start_child([sys.executable, "scripts/t2mem/serve.py", "--checkpoint", str(checkpoint),
                                           "--host", "127.0.0.1", "--port", str(self.options.port),
                                           "--ablation", ablation], env=self._gpu_env((self.options.policy_gpu,),
                                                                                   memory_fraction="0.70"),
                                          log_name=f"server_{name}.log")
                self.wait_for_server(server, checkpoint, ablation, name)
                command = [str(self.options.sim_python), "examples/robomme/eval.py", "--t2mem", "--host", "127.0.0.1",
                           "--port", str(self.options.port), "--tasks", "PickXtimes", "--seeds",
                           ",".join(map(str, self.options.seeds)), "--episodes-per-task", str(self.options.episodes_per_task),
                           "--max-steps", str(self.options.max_steps), "--save-dir", str(directory),
                           "--memory-write-stride", "16", "--sim-mode", "gpu"]
                code = self.run_checked(command, env=self._gpu_env((self.options.sim_gpu,), simulator=True),
                                        log_name=f"eval_{name}.log", allow_failure=True,
                                        supervised_server=server, progress_path=directory / "episodes.json",
                                        idle_timeout=self.options.eval_idle_timeout)
                completed = validate_evaluation(directory, checkpoint, ablation, **protocol)
                if code == 0 and completed:
                    self.update("EVALUATION_COMPLETE", model=name, results=str(directory))
                    return directory
                LOGGER.warning("Evaluation %s attempt %s/%s incomplete (exit=%s); valid rollouts will be reused",
                               name, attempt, self.options.eval_retries, code)
            except (WorkerDependencyExited, WorkerFailure, TimeoutError) as error:
                LOGGER.warning("Evaluation %s attempt %s/%s failed: %s; stopping owned workers before retrying",
                               name, attempt, self.options.eval_retries, error)
            finally:
                if server is not None:
                    self.stop_child(server)
                self.release_gpus()
        raise RuntimeError(f"Evaluation {name} did not complete after {self.options.eval_retries} attempts; inspect logs/eval_{name}.log")

    def compare(self, results: dict[str, Path]) -> None:
        self.update("COMPARE")
        destination = self.directory / "comparison"
        self.run_checked([sys.executable, "scripts/t2mem/compare_results.py", "--baseline", str(results["A"]),
                          "--full", str(results["B"]), "--no-write", str(results["C"]), "--output-dir", str(destination)],
                         log_name="comparison.log")
        report = json.loads((destination / "comparison_metadata.json").read_text())
        if report.get("protocol_validation") != "matched" or report.get("paper_evaluation_protocol_match") is not True:
            raise ValueError("Comparison did not validate the matched formal evaluation protocol")
        self.update("COMPLETE", status="COMPLETE", comparison=str(destination / "comparison.md"))

    def run(self) -> None:
        locks_dir = self.root / "runs/locks"
        identity = hashlib.sha256(str(self.options.stage2_root).encode()).hexdigest()[:24]
        locks = [FileLock(locks_dir / f"t2mem_stage2_{identity}.lock"), FileLock(self.directory / "pipeline.lock")]
        acquired = []
        try:
            for lock in locks:
                if not lock.acquire():
                    raise RuntimeError("Another pipeline already owns this Stage-2 run or pipeline directory")
                acquired.append(lock)
            (self.directory / "driver.pid").write_text(f"{os.getpid()}\n")
            self.preflight()
            stage1 = self.wait_for_training()
            stage2 = self.train_stage2(stage1)
            results = {"A": self.evaluate("A", stage1, "memory-free"),
                       "B": self.evaluate("B", stage2, "full"),
                       "C": self.evaluate("C", stage2, "no-online-write")}
            self.compare(results)
        except BaseException as error:
            # A duplicate driver must not overwrite the active driver's state file.
            if len(acquired) == len(locks):
                self.update("CANCELLED" if isinstance(error, PipelineCancelled) else "FAILED",
                            status="CANCELLED" if isinstance(error, PipelineCancelled) else "FAILED", error=str(error))
            raise
        finally:
            for child in list(self.children.values()):
                self.stop_child(child)
            self.release_gpus()
            for lock in reversed(acquired):
                lock.release()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    options = parse_args(argv)
    pipeline = Pipeline(options)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: pipeline.cancelled.set())
    try:
        pipeline.run()
    except PipelineCancelled:
        LOGGER.info("Pipeline cancelled; no existing training process was stopped")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("Pipeline failed; no later training/evaluation stages were started")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
