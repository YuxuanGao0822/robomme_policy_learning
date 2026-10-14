"""Resolved experiment configuration shared by training and policy serving."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataSettings:
    manifest: str = "data/t2mem/manifest.json"
    norm_stats_dir: str = "data/t2mem/assets/robomme"
    write_stride: int = 16
    demonstration_write_offset: int = 0
    action_supervision_stride: int = 1
    task_sampling: str = "uniform"
    task_weights: dict[str, float] | None = None
    max_sequence_length: int | None = None
    truncate_prefix: bool = False
    # Cache only small expert-action timelines on the CPU, not images.
    cache_actions: bool = False


@dataclass
class TrainingSettings:
    micro_batch_size: int = 1
    # Opt-in throughput mode. Accumulation then counts full waves per GPU;
    # None preserves the global-batch semantics in existing checkpoints.
    stage1_per_device_batch_size: int | None = None
    stage1_accumulation_steps: int = 64
    stage2_accumulation_steps: int = 32
    stage1_updates: int = 20_000
    memory_updates_per_cycle: int = 500
    policy_updates_per_cycle: int = 500
    # Table 3 does not publish the total cycle count. Supply it explicitly.
    alternating_cycles: int | None = None
    stage1_lr: float = 2e-5
    memory_lr: float = 7.5e-5
    gate_lr: float = 5e-4
    policy_lr: float = 3e-5
    warmup_steps: int = 0
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 1e-10
    outer_gradient_clip: float = 1.0
    save_interval: int = 100
    log_interval: int = 1
    # None preserves the original keep-all behavior, including old metadata.
    keep_last_checkpoints: int | None = None


@dataclass
class RunConfig:
    name: str = "t2mem_pickxtimes"
    seed: int = 7
    gpu_id: int = 6
    num_devices: int = 1
    checkpoint_dir: str = "runs/ckpts/t2mem_pickxtimes"
    pretrained_params: str = "${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params"
    model: dict[str, Any] = field(default_factory=dict)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    experiment_label: str = "single-task reproduction: PickXtimes"
    initialization_source: str | None = None

    def create_model_config(self) -> Any:
        from .memory import MemoryConfig
        from .model import T2MemConfig

        values = copy.deepcopy(self.model)
        memory = values.pop("memory", {})
        return T2MemConfig(**values, memory=MemoryConfig(**memory))

    def to_dict(self) -> dict:
        return asdict(self)

    def per_device_batch(self, stage: str) -> int:
        if stage not in ("stage1", "stage2a", "stage2b"):
            raise ValueError(f"Unknown training stage: {stage}")
        if stage == "stage1" and self.training.stage1_per_device_batch_size is not None:
            return self.training.stage1_per_device_batch_size
        return self.training.micro_batch_size

    def effective_batch(self, stage: str) -> int:
        micro = self.per_device_batch(stage)
        accumulation = (self.training.stage1_accumulation_steps if stage == "stage1"
                        else self.training.stage2_accumulation_steps)
        devices = self.num_devices if stage == "stage1" and self.training.stage1_per_device_batch_size is not None else 1
        return devices * micro * accumulation

    def approximation_labels(self) -> list[str]:
        labels = []
        if self.data.max_sequence_length is not None:
            labels.append("truncated_observation_prefix")
        if self.data.action_supervision_stride != 1:
            labels.append("sparse_execution_supervision")
        memory = self.model.get("memory", {})
        if memory.get("detach_fast_state", False):
            labels.append("detached_fast_state")
        for key, paper in (("num_heads", 16), ("head_dim", 64), ("hidden_dim", 256),
                           ("num_tokens", 16), ("width", 1024)):
            if key in memory and memory[key] != paper:
                labels.append(f"reduced_memory_{key}")
        return labels

    def validate(self) -> None:
        if type(self.data.cache_actions) is not bool:
            raise ValueError("data.cache_actions must be a boolean")
        if type(self.num_devices) is not int or self.num_devices <= 0:
            raise ValueError("num_devices must be a positive integer")
        if type(self.gpu_id) is not int or self.gpu_id < 0:
            raise ValueError("gpu_id must be a nonnegative physical GPU index")
        values = self.training
        stage1_micro = values.stage1_per_device_batch_size
        if stage1_micro is not None and (type(stage1_micro) is not int or stage1_micro <= 0):
            raise ValueError("training.stage1_per_device_batch_size must be None or a positive integer")
        if self.num_devices > 1 and (type(values.micro_batch_size) is not int or values.micro_batch_size != 1):
            raise ValueError("Multi-device T²Mem training requires training.micro_batch_size=1 per episode")
        for name in ("micro_batch_size", "stage1_accumulation_steps", "stage2_accumulation_steps",
                     "stage1_updates", "memory_updates_per_cycle", "policy_updates_per_cycle",
                     "save_interval", "log_interval"):
            if type(getattr(values, name)) is not int or getattr(values, name) <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        if values.alternating_cycles is not None and values.alternating_cycles <= 0:
            raise ValueError("alternating_cycles must be positive")
        keep = values.keep_last_checkpoints
        if keep is not None and (type(keep) is not int or keep <= 0):
            raise ValueError("training.keep_last_checkpoints must be None or a positive integer")
        if self.data.max_sequence_length is not None and not self.data.truncate_prefix:
            raise ValueError("max_sequence_length requires explicit truncate_prefix=true")
        block_size = self.model.get("temporal_remat_block_size", 16)
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("model.temporal_remat_block_size must be a positive integer")
        robot_action_dim = self.model.get("robot_action_dim", 8)
        if type(robot_action_dim) is not int or robot_action_dim != 8 or robot_action_dim > self.model.get("action_dim", 32):
            raise ValueError("PickXtimes requires model.robot_action_dim=8 within model.action_dim")


def _merge(parent: dict, child: dict) -> dict:
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path, seen: frozenset[Path] = frozenset()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic configuration inheritance at {path}")
    values = yaml.safe_load(path.read_text()) or {}
    parent = values.pop("extends", None)
    return _merge(_read_yaml(path.parent / parent, seen | {path}), values) if parent else values


def config_from_dict(values: dict) -> RunConfig:
    values = copy.deepcopy(values)
    data, training = values.pop("data", {}), values.pop("training", {})
    config = RunConfig(**values, data=DataSettings(**data), training=TrainingSettings(**training))
    config.validate()
    return config


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path).expanduser()
    values = json_load(path) if path.suffix == ".json" else _read_yaml(path)
    # metadata.json wraps a fully resolved experiment config.
    if "config" in values:
        values = values["config"]
    return config_from_dict(values)


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def expand_path(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "${" in expanded or "$OPENPI_DATA_HOME" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return Path(expanded).expanduser().resolve()
