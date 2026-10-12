"""Episode-indexed RoboMME sequences for T²Mem, backed by the official raw HDF5.

The legacy pickle builder stores execution frames only and 20-action padded
chunks; neither its cached visual features nor its padding recover the missing
demonstration observations or a masked H64 target. This loader reads the raw
observations lazily, keeps one episode local to each sequence, and never stores
copies of an image for all of its possible history prefixes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

TASKS = ("PickXtimes",)
MANIFEST_VERSION = 1


def _scalar(value: Any) -> Any:
    value = np.asarray(value)
    result = value.reshape(-1)[0].item()
    return result.decode() if isinstance(result, bytes) else result


def validate_train_source(path: Path) -> None:
    """Reject known held-out paths; provenance still requires an official source."""
    for component in path.parts:
        if re.search(r"(^|[_\-.])(test|validation|valid|val)([_\-.]|$)", component, re.IGNORECASE):
            raise ValueError(f"Held-out data must not enter training: {path}")


def build_manifest(raw_root: str | Path, output: str | Path, *, max_episodes: int | None = None) -> dict:
    """Index only exact official training filenames, preserving original indices.

    `raw_root` must be the extracted official Yinpei/robomme_data_h5 training
    release. Filenames alone cannot establish provenance: that source assertion
    is recorded in the manifest rather than inferred from image content.
    """
    root = Path(raw_root).expanduser().resolve()
    validate_train_source(root)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    episodes = []
    for task in TASKS:
        path = root / f"data_{task}.h5"
        validate_train_source(path)
        if not path.is_file():
            raise FileNotFoundError(f"Expected official task training file: {path}")
        with h5py.File(path, "r") as file:
            split = _scalar(file.attrs["split"]) if "split" in file.attrs else "train"
            if str(split).lower() not in ("train", "training"):
                raise ValueError(f"Non-training split in {path}: {split}")
            keys = sorted((k for k in file if re.fullmatch(r"episode_\d+", k)), key=lambda k: int(k[8:]))
            if max_episodes is not None:
                keys = keys[:max_episodes]
            for key in keys:
                episode = file[key]
                if "split" in episode.attrs and str(_scalar(episode.attrs["split"])).lower() not in ("train", "training"):
                    raise ValueError(f"Non-training episode: {path}:{key}")
                steps = sorted(int(k[9:]) for k in episode if re.fullmatch(r"timestep_\d+", k))
                if not steps or steps != list(range(steps[-1] + 1)):
                    raise ValueError(f"Expected contiguous physical observation timeline: {path}:{key}")
                demo = [bool(_scalar(episode[f"timestep_{t}/info/is_video_demo"][()])) for t in steps]
                execution_start = next((t for t, is_demo in enumerate(demo) if not is_demo), len(steps))
                if any(demo[execution_start:]) or execution_start == len(steps):
                    raise ValueError(f"Expected demonstration prefix followed by execution: {path}:{key}")
                action_steps = [t for t in steps[execution_start:] if f"timestep_{t}/action/joint_action" in episode]
                if not action_steps:
                    raise ValueError(f"No execution action labels: {path}:{key}")
                action_widths = {int(np.asarray(episode[f"timestep_{t}/action/joint_action"][()]).size) for t in action_steps}
                if action_widths != {8}:
                    raise ValueError(f"RoboMME expects 8 action dimensions, found {action_widths}: {path}:{key}")
                episodes.append({"task": task, "path": str(path), "episode": key, "steps": len(steps),
                                 "execution_start": execution_start, "action_steps": action_steps,
                                 "prompt": str(_scalar(episode["setup/task_goal"][()]))})
    result = {"version": MANIFEST_VERSION, "split": "train", "tasks": list(TASKS),
              "source": "official Yinpei/robomme_data_h5 training release (operator supplied)",
              "raw_root": str(root), "episode_subset": max_episodes,
              "episodes": episodes, "task_counts": dict(Counter(e["task"] for e in episodes))}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("split") != "train":
        raise ValueError("Unsupported manifest or non-training split")
    if tuple(manifest.get("tasks", ())) != TASKS:
        raise ValueError(f"Training manifest must contain exactly {TASKS}")
    for entry in manifest["episodes"]:
        if entry["task"] not in TASKS:
            raise ValueError(f"Unexpected task {entry['task']}")
        validate_train_source(Path(entry["path"]))
        if Path(entry["path"]).name != f"data_{entry['task']}.h5":
            raise ValueError("Episode task identity must match the official HDF5 filename")
    if set(e["task"] for e in manifest["episodes"]) != set(TASKS):
        raise ValueError("PickXtimes needs at least one official training episode")
    return manifest


def make_transforms(model_config: Any, norm_stats: dict) -> Any:
    """Reuse OpenPI's robot transforms, quantile normalization and tokenizer."""
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from mme_vla_suite.policies.robomme_policy import RoboMMEInputs
    from .observation import TokenizeT2MemPrompt

    return transforms.compose([
        RoboMMEInputs(model_type=model_config.model_type),
        transforms.DeltaActions(transforms.make_bool_mask(7, -1)),
        transforms.Normalize(norm_stats, use_quantiles=True),
        transforms.ResizeImages(224, 224),
        # The identical A/B/C input contains both native pi0.5's state prompt
        # and a VL-only prompt. The model chooses the native form for memory-free
        # Stage 1; T²Mem puts continuous proprioception in the suffix so memory
        # queries can never read state tokens through the VL prefix.
        TokenizeT2MemPrompt(PaligemmaTokenizer(model_config.max_token_len)),
        transforms.PadStatesAndActions(model_config.action_dim),
    ])


def read_frame(episode: h5py.Group, step: int, prompt: str, horizon: int, action_dim: int) -> tuple[dict, np.ndarray]:
    """Read one image pair and future chunk; padded elements get no supervision."""
    frame = episode[f"timestep_{step}"]
    obs = frame["obs"]
    state = np.concatenate((np.asarray(obs["joint_state"][()]).reshape(-1),
                            np.asarray(obs["gripper_state"][()]).reshape(-1)[:1])).astype(np.float32)
    actions = np.zeros((horizon, 8), np.float32)
    mask = np.zeros((horizon, action_dim), bool)
    is_demo = bool(_scalar(frame["info/is_video_demo"][()]))
    if not is_demo:
        for future in range(horizon):
            key = f"timestep_{step + future}"
            if key not in episode:
                break
            source = episode[key]
            if bool(_scalar(source["info/is_video_demo"][()])) or "action/joint_action" not in source:
                continue
            action = np.asarray(source["action/joint_action"][()], dtype=np.float32).reshape(-1)
            if action.size != 8 or not np.all(np.isfinite(action)):
                raise ValueError(f"Invalid expert action at {episode.name}/{key}")
            actions[future] = action
            mask[future, :8] = True
    raw = {"observation/image": np.asarray(obs["front_rgb"][()]),
           "observation/wrist_image": np.asarray(obs["wrist_rgb"][()]),
           "observation/state": state, "actions": actions, "prompt": prompt}
    return raw, mask


@dataclass
class SequenceSample:
    observation: dict
    actions: np.ndarray
    action_mask: np.ndarray
    write_mask: np.ndarray
    observation_mask: np.ndarray
    physical_steps: np.ndarray
    task: str
    episode: str
    approximations: tuple[str, ...] = ()


def _stack_observations(frames: list[dict]) -> dict:
    return {"image": {key: np.stack([f["image"][key] for f in frames]) for key in frames[0]["image"]},
            "image_mask": {key: np.stack([f["image_mask"][key] for f in frames]) for key in frames[0]["image_mask"]},
            "state": np.stack([f["state"] for f in frames]),
            "tokenized_prompt": np.stack([f["tokenized_prompt"] for f in frames]),
            "tokenized_prompt_mask": np.stack([f["tokenized_prompt_mask"] for f in frames]),
            "tokenized_pi05_prompt": np.stack([f["tokenized_pi05_prompt"] for f in frames]),
            "tokenized_pi05_prompt_mask": np.stack([f["tokenized_pi05_prompt_mask"] for f in frames])}


class EpisodeSampler:
    """Task-homogeneous effective batches with resumable host RNG and counts."""

    def __init__(self, manifest: dict, transform: Any, *, horizon: int = 64, action_dim: int = 32,
                 write_stride: int = 16, demonstration_write_offset: int = 0,
                 action_supervision_stride: int = 1, task_sampling: str = "uniform",
                 task_weights: dict[str, float] | None = None, seed: int = 0,
                 max_sequence_length: int | None = None, truncate_prefix: bool = False):
        self.by_task = {task: [e for e in manifest["episodes"] if e["task"] == task] for task in TASKS}
        self.transform, self.horizon, self.action_dim = transform, horizon, action_dim
        self.write_stride, self.demo_offset = write_stride, demonstration_write_offset
        self.supervision_stride = action_supervision_stride
        self.max_length, self.truncate_prefix = max_sequence_length, truncate_prefix
        if write_stride < 1 or action_supervision_stride < 1 or not 0 <= demonstration_write_offset < write_stride:
            raise ValueError("Invalid observation timeline stride or demonstration offset")
        if max_sequence_length is not None and (max_sequence_length < 1 or not truncate_prefix):
            raise ValueError("Sequence truncation requires explicit truncate_prefix approximation")
        if task_sampling == "uniform":
            weights = np.ones(len(TASKS))
        elif task_sampling == "episodes":
            weights = np.asarray([len(self.by_task[t]) for t in TASKS], dtype=float)
        elif task_sampling == "frames":
            weights = np.asarray([sum(len(e["action_steps"]) for e in self.by_task[t]) for t in TASKS], dtype=float)
        elif task_sampling == "weighted":
            if task_weights is None or set(task_weights) != set(TASKS):
                raise ValueError("Weighted sampling requires a weight for exactly each target task")
            weights = np.asarray([task_weights[t] for t in TASKS], dtype=float)
        else:
            raise ValueError(f"Unknown task sampling strategy {task_sampling}")
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Task sampling weights must be finite, nonnegative and have a positive sum")
        self.probabilities = weights / weights.sum()
        self.rng = np.random.default_rng(seed)
        self.task_updates = Counter()
        self.task_samples = Counter()

    def next_task(self) -> str:
        task = str(self.rng.choice(TASKS, p=self.probabilities))
        self.task_updates[task] += 1
        return task

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "task_updates": dict(self.task_updates),
                "task_samples": dict(self.task_samples)}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.task_updates, self.task_samples = Counter(state["task_updates"]), Counter(state["task_samples"])

    def sample(self, task: str, *, stage1: bool) -> SequenceSample:
        entries = self.by_task[task]
        probabilities = None
        if stage1:
            sizes = np.asarray([len(e["action_steps"]) for e in entries], dtype=float)
            probabilities = sizes / sizes.sum()
        entry = entries[int(self.rng.choice(len(entries), p=probabilities))]
        start = entry["execution_start"]
        if stage1:
            steps = [int(self.rng.choice(entry["action_steps"]))]
        else:
            # Demo frames without scheduled writes cannot affect later state.
            # Execution keeps every supervised frame plus midpoint/write-only
            # observations. Omitted frames cause neither writes nor losses.
            steps = [t for t in range(entry["steps"]) if
                     (t < start and t % self.write_stride == self.demo_offset) or
                     (t >= start and ((t - start) % self.write_stride == 0 or
                                      (t - start) % self.supervision_stride == 0))]
        approximations = []
        if self.max_length is not None and len(steps) > self.max_length and not stage1:
            steps = steps[-self.max_length:]
            approximations.append("truncated_observation_prefix")
        if not stage1 and self.supervision_stride != 1:
            approximations.append("sparse_execution_supervision")
        frames, masks, write_mask = [], [], []
        with h5py.File(entry["path"], "r") as source:
            episode = source[entry["episode"]]
            for t in steps:
                raw, mask = read_frame(episode, t, entry["prompt"], self.horizon, self.action_dim)
                if t < start or (not stage1 and (t - start) % self.supervision_stride != 0):
                    mask[:] = False
                frames.append(self.transform(raw))
                masks.append(mask)
                write_mask.append(not stage1 and ((t < start and t % self.write_stride == self.demo_offset) or
                                  (t >= start and (t - start) % self.write_stride == 0)))
        self.task_samples[task] += 1
        sample = SequenceSample(_stack_observations(frames), np.stack([f["actions"] for f in frames]),
                                np.stack(masks), np.asarray(write_mask, bool), np.ones(len(steps), bool),
                                np.asarray(steps, np.int32), task, entry["episode"], tuple(approximations))
        if not sample.action_mask.any():
            raise ValueError(f"Sequence has no action supervision: {entry['path']}:{entry['episode']}")
        return sample


def collate_sequences(samples: list[SequenceSample]) -> dict:
    """Time-major padded batch; each column resets independently to learned W0."""
    if len({sample.task for sample in samples}) != 1:
        raise ValueError("T²Mem batches must be task homogeneous")
    maximum = max(len(sample.write_mask) for sample in samples)
    length = 1 if maximum == 1 else 1 << (maximum - 1).bit_length()

    def pad(array: np.ndarray, fill: Any = 0) -> np.ndarray:
        return np.pad(array, ((0, length - len(array)), *[(0, 0)] * (array.ndim - 1)), constant_values=fill)

    def collate(key: str) -> np.ndarray:
        return np.stack([pad(getattr(s, key)) for s in samples], axis=1)

    observations = {}
    for key in samples[0].observation:
        value = samples[0].observation[key]
        if isinstance(value, dict):
            observations[key] = {name: np.stack([pad(s.observation[key][name]) for s in samples], axis=1) for name in value}
        else:
            observations[key] = np.stack([pad(s.observation[key]) for s in samples], axis=1)
    return {"observation": observations, "actions": collate("actions"), "action_mask": collate("action_mask"),
            "write_mask": collate("write_mask"), "observation_mask": collate("observation_mask")}


def compute_norm_stats(manifest: dict, output: str | Path) -> None:
    """PickXtimes train-only statistics, with OpenPI's joint-delta convention."""
    from openpi.shared import normalize

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for entry in manifest["episodes"]:
        with h5py.File(entry["path"], "r") as source:
            episode = source[entry["episode"]]
            for step in entry["action_steps"]:
                raw, mask = read_frame(episode, step, entry["prompt"], 1, 8)
                if not mask.any():
                    continue
                state = raw["observation/state"]
                action = raw["actions"].copy()
                action[:, :7] -= state[:7]
                stats["state"].update(state[None])
                stats["actions"].update(action)
    normalize.save(output, {key: running.get_statistics() for key, running in stats.items()})
