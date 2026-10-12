"""RoboMME H64/E32 evaluation using incremental, episode-local T²Mem RPCs.

Run this in the separate RoboMME simulator environment. JAX is not imported by
this process. Main evaluation pools 50 episodes per task over three policy seeds.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("PickXtimes",)


def _raw_observation(image, wrist_image, state, prompt: str) -> dict:
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize(records: list[dict], expected_rollouts: int) -> dict:
    """Report valid rollout denominators explicitly; infrastructure errors are separate."""
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    task_groups: dict[str, list[dict]] = collections.defaultdict(list)
    errors = [record for record in records if record["status"] == "error"]
    for record in records:
        if record["status"] != "error":
            groups[(record["task"], record["difficulty"])].append(record)
            task_groups[record["task"]].append(record)

    def rates(group):
        successes = sum(bool(record["success"]) for record in group)
        return {"successes": successes, "rollouts": len(group), "success_rate": successes / len(group) if group else None}

    tasks = {
        task: {
            **rates(group),
            "by_difficulty": {
                difficulty: rates(groups[(task, difficulty)])
                for difficulty in sorted({record["difficulty"] for record in group})
            },
            "by_seed": {
                str(seed): rates([record for record in group if record["seed"] == seed])
                for seed in sorted({record["seed"] for record in group})
            },
        }
        for task, group in sorted(task_groups.items())
    }
    valid = [record for record in records if record["status"] != "error"]
    return {
        "experiment": "single-task reproduction",
        "complete": len(valid) == expected_rollouts and not errors,
        "expected_rollouts": expected_rollouts,
        "valid_rollouts": len(valid),
        "infrastructure_errors": len(errors),
        "tasks": tasks,
        "overall_pooled": rates(valid),
        "overall_macro_success_rate": float(np.mean([group["success_rate"] for group in tasks.values()])) if tasks else None,
    }


def _configure_simulator(mode: str) -> None:
    if mode == "cpu-render":
        # These flags come from this repository's README troubleshooting advice.
        # The simulator submodule is absent locally, so this is not a verified CPU
        # physics/render path. Do not hide CUDA: an unsupported renderer must fail.
        os.environ["SAPIEN_RENDER_DEVICE"] = "cpu"
        os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")


def rollout(client, runner, args, seed: int) -> dict:
    pre = runner.get_init_obs()
    prompt = pre["task_goal"]
    lengths = {len(pre[field]) for field in ("images", "wrist_images", "states")}
    if lengths != {1}:
        raise ValueError("PickXtimes reset must return exactly one execution observation and no demonstration prefix.")
    fingerprint = hashlib.sha256(prompt.encode("utf-8"))
    for field in ("images", "wrist_images", "states"):
        fingerprint.update(np.ascontiguousarray(pre[field][-1]).tobytes())
    context = {"env_id": "robomme", "task": runner.env_id, "episode_id": f"seed{seed}/episode{runner.episode_id}"}
    client.start_episode(**context, prompt=prompt, seed=seed)
    recorder = None
    if args.save_videos:
        from utils import RolloutRecorder

        recorder = RolloutRecorder(args.save_dir / "videos" / f"seed{seed}", prompt)
    steps = 0
    status = "unknown"
    start_time = time.monotonic()
    try:
        image, wrist, state = pre["images"][-1], pre["wrist_images"][-1], pre["states"][-1]
        if recorder:
            recorder.record(image.copy(), wrist.copy(), state.copy())
        stop = False
        while steps < args.max_steps and not stop:
            output = client.infer_at(**context, index=steps, observation=_raw_observation(image, wrist, state, prompt))
            actions = np.asarray(output["actions"])
            if actions.shape != (64, 8) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite H64 joint-angle actions of shape (64, 8), got {actions.shape}.")
            for action in actions[:32]:
                (image, wrist, state), stop, status = runner.step(action)
                steps += 1
                if status == "error" or image is None:
                    raise RuntimeError("Simulator step failed.")
                if recorder:
                    recorder.record(image.copy(), wrist.copy(), state.copy(), action=action.copy())
                if stop or steps >= args.max_steps:
                    break
                if steps % args.memory_write_stride == 0 and steps % 32:
                    # Midpoint memory update only: no replan and no RNG advancement.
                    client.observe(**context, phase="execution", frames=[{
                        "index": steps,
                        "observation": _raw_observation(image, wrist, state, prompt),
                    }])
        if not stop:
            status = "timeout"
        if status == "unknown":
            raise RuntimeError("Terminal predicate was missing (status=unknown).")
        if recorder:
            recorder.save_video(f"{runner.env_id}_ep{runner.episode_id}_{status}.mp4")
        return {
            "status": status,
            "success": status == "success",
            "steps": steps,
            "duration_seconds": time.monotonic() - start_time,
            "instruction": prompt,
            "initial_observation_sha256": fingerprint.hexdigest(),
            "demo_frames": 0,
        }
    finally:
        # Preserve the original inference/simulator traceback if the server has
        # already closed a failed connection. Disconnect also releases its state.
        with contextlib.suppress(Exception):
            client.end_episode(**context)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tasks", "--only-tasks", default="PickXtimes")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/evaluation/t2mem"))
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--memory-write-stride", type=int, default=16)
    parser.add_argument("--sim-mode", choices=("gpu", "cpu-render"), default="gpu")
    args = parser.parse_args(argv)
    args.tasks = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
    args.seeds = tuple(int(part) for part in args.seeds.split(","))
    if args.tasks != TASKS:
        parser.error("This single-task reproduction supports only PickXtimes.")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error("Evaluation seeds must be nonempty and unique.")
    if not 1 <= args.episodes_per_task <= 50:
        parser.error("Use 1–50 PickXtimes episodes per seed.")
    if args.memory_write_stride <= 0 or 32 % args.memory_write_stride:
        parser.error("Write stride must be positive and divide execution horizon 32.")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    _configure_simulator(args.sim_mode)
    # Reuse the benchmark's actual wrapper after setting device/render variables.
    sys.path.insert(0, str(ROOT / "examples" / "robomme"))
    from env_runner import EnvRunner
    from openpi_client.websocket_client_policy import T2MemWebsocketClientPolicy

    args.save_dir.mkdir(parents=True, exist_ok=True)
    client = T2MemWebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata.get("action_horizon") != 64 or metadata.get("execution_horizon") != 32:
        raise ValueError("Evaluation requires server architecture H64/E32.")
    if metadata.get("memory_write_stride") != args.memory_write_stride:
        raise ValueError("Client/server memory write stride mismatch.")
    manifest = {
        "experiment": "single-task reproduction",
        "tasks": list(args.tasks),
        "seeds": list(args.seeds),
        "episodes_per_task": args.episodes_per_task,
        "max_steps": args.max_steps,
        "simulation_mode": args.sim_mode,
        "cpu_render_support_verified": False,
        "server": metadata,
    }
    manifest_path = args.save_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume manifest differs; select a new --save-dir for this experiment.")
    _atomic_json(manifest_path, manifest)
    progress_path = args.save_dir / "episodes.json"
    records = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed = {(record["task"], record["seed"], record["episode_id"]) for record in records if record["status"] != "error"}
    expected = len(args.tasks) * len(args.seeds) * args.episodes_per_task
    try:
        for seed in args.seeds:
            for task in args.tasks:
                runner = EnvRunner(task, args.save_dir / "videos", max_steps=args.max_steps)
                if runner.num_episodes < args.episodes_per_task:
                    raise ValueError(f"Official test split has only {runner.num_episodes} {task} episodes.")
                for episode_id in range(args.episodes_per_task):
                    key = (task, seed, episode_id)
                    if key in completed:
                        continue
                    random.seed(seed)
                    np.random.seed(seed)
                    runner.make_env(episode_id)
                    difficulty = str(runner.difficulty).lower().rsplit(".", 1)[-1]
                    if difficulty not in ("easy", "medium", "hard"):
                        raise ValueError(f"Unsupported official difficulty label: {runner.difficulty!r}")
                    record = {"task": task, "seed": seed, "episode_id": episode_id, "difficulty": difficulty}
                    try:
                        record.update(rollout(client, runner, args, seed))
                        completed.add(key)
                    except Exception:
                        record.update(status="error", success=None, error=traceback.format_exc())
                        print(record["error"], file=sys.stderr)
                        # Server closes failed RPC connections and releases their states.
                        client.close()
                        client = T2MemWebsocketClientPolicy(args.host, args.port)
                    finally:
                        runner.close_env()
                    records = [old for old in records if (old["task"], old["seed"], old["episode_id"]) != key]
                    records.append(record)
                    _atomic_json(progress_path, records)
                    _atomic_json(args.save_dir / "metrics.json", summarize(records, expected))
                    print(f"{task} seed={seed} episode={episode_id} difficulty={record['difficulty']} status={record['status']}", flush=True)
        metrics = summarize(records, expected)
        if args.episodes_per_task == 50:
            expected_difficulties = {"easy": 26, "medium": 12, "hard": 12}
            actual = {
                f"{task}/seed{seed}": dict(collections.Counter(record["difficulty"] for record in records if record["task"] == task and record["seed"] == seed))
                for task in args.tasks for seed in args.seeds
            }
            metrics["paper_difficulty_counts_match"] = all(counts == expected_difficulties for counts in actual.values())
            metrics["difficulty_counts"] = actual
        _atomic_json(args.save_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        if not metrics["complete"]:
            raise SystemExit("Evaluation incomplete; infrastructure errors can be retried with the same command.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
