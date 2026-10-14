"""Differentiable sequence training with independent phase optimizers.

Freezing selects optimizer inputs; it never detaches AE activations or fast
states. The integrated model owns the chronological scan and read-before-write
loss, so both alternating phases retain the complete memory recurrence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
import shutil
import time
from typing import Any

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.models import model as observation_model
from openpi.shared import normalize

from .configuration import RunConfig
from .dataset import EpisodeSampler, collate_sequences
from .observation import T2Observation
from .parallel import (
    DATA_AXIS, collate_parallel_wave, make_gradient_reducer, make_mesh,
    make_parallel_accumulator, make_parallel_grad, wave_sizes,
)

STAGES = ("stage1", "stage2a", "stage2b")


def parameter_group(path: tuple[Any, ...]) -> str:
    """Name-based ownership for the OpenPI-compatible scanned Gemma tree."""
    name = "/" + "/".join(map(str, path)) + "/"
    if any("memory" in str(part) for part in path):
        return "gate" if re.search(r"/(alpha|fusion_gate|gate)/", name) else "memory"
    if "/img/" in name:
        return "frozen"
    if "/llm/" in name:
        # Existing Gemma names its AE stream with suffix _1. Full AE
        # parameters, including adaptive normalization, are trainable.
        if "lora" in name.lower() or any(str(part).endswith("_1") for part in path):
            return "policy"
        return "frozen"
    if any(str(part) in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
                         "action_time_mlp_in", "action_time_mlp_out") for part in path):
        return "policy"
    if any(str(part) in ("state_proj", "proprio_proj") for part in path):
        return "proprioception"
    return "frozen"


def stage_filter(stage: str) -> Any:
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage {stage}")
    allowed = {"memory", "gate"} if stage == "stage2a" else {"policy"}
    if stage == "stage2b":
        allowed.add("proprioception")
    return nnx.All(nnx.Param, lambda path, _: parameter_group(path) in allowed)


def optimizer_for(stage: str, params: dict, config: RunConfig) -> optax.GradientTransformation:
    settings = config.training
    rates = {"policy": settings.stage1_lr if stage == "stage1" else settings.policy_lr,
             "memory": settings.memory_lr, "gate": settings.gate_lr}

    def schedule(rate: float) -> Any:
        if settings.warmup_steps == 0:
            return rate
        return optax.join_schedules([
            optax.linear_schedule(rate / (settings.warmup_steps + 1), rate, settings.warmup_steps),
            optax.constant_schedule(rate)], [settings.warmup_steps])

    labels = traverse_util.unflatten_dict({path: ("policy" if parameter_group(path) == "proprioception"
                                                else parameter_group(path))
                                         for path in traverse_util.flatten_dict(params)})
    transforms = {name: optax.adamw(schedule(rate), b1=settings.adam_b1, b2=settings.adam_b2,
                                   eps=settings.adam_eps, weight_decay=settings.weight_decay)
                  for name, rate in rates.items()}
    return optax.chain(optax.clip_by_global_norm(settings.outer_gradient_clip),
                       optax.multi_transform(transforms, labels))


@dataclass
class PhaseProgress:
    mode: str
    phase: str
    global_step: int = 0
    phase_step: int = 0
    cycle: int = 0
    stage_updates: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STAGES, 0))
    complete: bool = False

    @classmethod
    def start(cls, mode: str) -> "PhaseProgress":
        if mode not in (*STAGES, "alternating"):
            raise ValueError(f"Unknown mode {mode}")
        return cls(mode=mode, phase="stage2a" if mode == "alternating" else mode)

    def advance(self, config: RunConfig) -> None:
        self.global_step += 1
        self.phase_step += 1
        self.stage_updates[self.phase] += 1
        settings = config.training
        budget = {"stage1": settings.stage1_updates, "stage2a": settings.memory_updates_per_cycle,
                  "stage2b": settings.policy_updates_per_cycle}[self.phase]
        if self.phase_step < budget:
            return
        if self.mode != "alternating":
            self.complete = True
        elif self.phase == "stage2a":
            self.phase, self.phase_step = "stage2b", 0
        elif self.cycle + 1 >= settings.alternating_cycles:
            self.complete = True
        else:
            self.cycle += 1
            self.phase, self.phase_step = "stage2a", 0

    def set_cycle_budget(self, cycles: int) -> None:
        """Explicitly extend a completed run without resetting learned state."""
        if self.mode != "alternating":
            raise ValueError("--cycles applies only to an alternating run")
        if cycles < self.cycle + 1:
            raise ValueError("Cycle budget cannot discard a completed or active cycle")
        if self.complete and cycles > self.cycle + 1:
            self.cycle += 1
            self.phase, self.phase_step, self.complete = "stage2a", 0, False


def latest_checkpoint(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if (path / "metadata.json").exists():
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit() and (p / "metadata.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No complete T²Mem checkpoints found under {path}")
    return max(candidates, key=lambda p: int(p.name))


def _prune_checkpoints(run_root: Path, keep_last: int | None, *, current_checkpoint: Path,
                       initialization_source: str | None = None) -> None:
    """Remove old committed snapshots from this run only, after a new commit.

    Failed saves, unrelated directories and symlinks are never candidates.
    An initialization checkpoint inside this run remains protected even when
    it is older than the requested retention window.
    """
    if keep_last is None:
        return
    root = run_root.expanduser().resolve()
    current = current_checkpoint.expanduser().resolve()
    source = Path(initialization_source).expanduser().resolve() if initialization_source else None

    def committed_step(path: Path) -> int | None:
        if path.parent != root or path.is_symlink() or not path.is_dir() or not re.fullmatch(r"[0-9]+", path.name):
            return None
        marker = path / "metadata.json"
        if marker.is_symlink() or not marker.is_file():
            return None
        try:
            metadata = json.loads(marker.read_text())
        except OSError:
            logging.warning("Cannot inspect checkpoint for retention: %s", path, exc_info=True)
            return None
        except (ValueError, UnicodeError):
            return None
        if not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int:
            return None
        progress = metadata.get("progress")
        if metadata["format_version"] != 1 or metadata.get("complete") is not True or not isinstance(progress, dict):
            return None
        step = progress.get("global_step")
        if type(step) is not int or step != int(path.name):
            return None
        if any((path / part).is_symlink() or not (path / part).is_dir() for part in ("params", "train_state")):
            return None
        return step

    completed = []
    for path in root.iterdir():
        step = committed_step(path)
        if step is not None:
            completed.append((step, path))
    completed.sort(key=lambda item: item[0])
    for step, path in completed[:-keep_last]:
        if path == current or (source is not None and (source == path or source.is_relative_to(path))):
            continue
        # Recheck before removing anything; do not follow a replaced symlink or
        # prune a directory whose committed marker changed during inspection.
        if committed_step(path) != step:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            logging.warning("Checkpoint saved, but retention could not remove %s", path, exc_info=True)
        else:
            logging.info("Removed old checkpoint after successful save: %s", path)


def load_initial_parameters(model: Any, path: str | Path, *, fresh_memory: bool,
                            allow_new_policy_parameters: bool = False) -> None:
    """Load pretrained/Stage-1 policy while leaving genuinely new memory fresh."""
    path = Path(path).expanduser().resolve()
    if (path / "params").is_dir():
        path = path / "params"
    loaded = observation_model.restore_params(path, restore_type=np.ndarray)
    state = nnx.state(model, nnx.Param)
    reference = traverse_util.flatten_dict(state.to_pure_dict())
    source = traverse_util.flatten_dict(loaded)
    merged = {}
    missing = []
    for key, expected in reference.items():
        group = parameter_group(key)
        if fresh_memory and group in ("memory", "gate"):
            merged[key] = expected
            continue
        if key not in source:
            if allow_new_policy_parameters and (group == "proprioception" or "lora" in "/".join(map(str, key)).lower()):
                merged[key] = expected
                continue
            missing.append("/".join(map(str, key)))
            continue
        actual = source[key]
        if actual.shape != expected.shape:
            raise ValueError(f"Checkpoint shape mismatch at {key}: {actual.shape} != {expected.shape}")
        merged[key] = jnp.asarray(actual, dtype=expected.dtype)
    if missing:
        raise ValueError(f"Checkpoint is missing required pretrained parameters: {missing[:12]}")
    state.replace_by_pure_dict(traverse_util.unflatten_dict(merged))
    nnx.update(model, state)


class Trainer:
    def __init__(self, model: Any, config: RunConfig, progress: PhaseProgress, sampler: EpisodeSampler):
        self.config, self.progress, self.sampler = config, progress, sampler
        # Lazy for step-only trainers. Once loaded these exact bytes survive
        # retention deleting the checkpoint from which this run was resumed.
        self._norm_stats_bytes: bytes | None = None
        self.graphdef, self.state = nnx.split(model)
        self.rng = jax.random.PRNGKey(config.seed + 1)
        self.mesh = make_mesh(config.num_devices) if config.num_devices > 1 else None
        self.replica_sharding = None
        self.slot_sharding = None
        if self.mesh is not None:
            if config.per_device_batch("stage2a") != 1:
                raise ValueError("Data-parallel Stage 2 requires micro_batch_size=1 per device")
            self.replica_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            self.slot_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(DATA_AXIS))
            self.state = self._put_replicated(self.state)
            self.rng = self._put_replicated(self.rng)
            self.parallel_accumulate = make_parallel_accumulator(self.mesh)
            self.reduce_gradients = make_gradient_reducer(self.mesh)
            self.split_rng = jax.jit(jax.random.split, in_shardings=self.replica_sharding,
                                    out_shardings=self.replica_sharding)
        self.optimizers: dict[str, Any] = {}
        self.optimizer_states: dict[str, Any] = {}
        self.grad_fns: dict[str, Any] = {}
        self.update_fns: dict[str, Any] = {}
        self.accumulate = jax.jit(lambda old, new: jax.tree.map(jnp.add, old, new))
        self.audit_groups()

    def _put_replicated(self, tree: Any) -> Any:
        # Named replication keeps every logical leaf shape unchanged, so saved
        # checkpoints remain readable by both single-device trainers and policy.
        return jax.device_put(tree, self.replica_sharding) if self.replica_sharding is not None else jax.device_put(tree)

    def audit_groups(self) -> None:
        pure = self.state.filter(nnx.Param).to_pure_dict()
        counts: dict[str, int] = {}
        for path, value in traverse_util.flatten_dict(pure).items():
            group = parameter_group(path)
            counts[group] = counts.get(group, 0) + value.size
        logging.info("Parameter ownership (scalar counts): %s", counts)
        for group in ("policy", "memory", "gate"):
            if counts.get(group, 0) == 0:
                raise ValueError(f"Model has no parameters in required {group} group; audit model tree naming")

    def ensure_optimizer(self, stage: str) -> None:
        if stage in self.optimizers:
            return
        active_filter = stage_filter(stage)
        active = self.state.filter(active_filter).to_pure_dict()
        tx = optimizer_for(stage, active, self.config)
        self.optimizers[stage] = tx
        self.optimizer_states[stage] = self._put_replicated(tx.init(active))

        def loss_fn(train_params: dict, all_state: Any, rng: Any, batch: dict) -> Any:
            model = nnx.merge(self.graphdef, all_state)
            train_state = nnx.state(model, active_filter)
            train_state.replace_by_pure_dict(train_params)
            nnx.update(model, train_state)
            model.train()
            observation = T2Observation.from_dict(batch["observation"])
            loss, stats = model.compute_sequence_loss(
                rng, observation, batch["actions"], batch["action_mask"],
                batch["write_mask"], batch["observation_mask"],
                train=True, memory_enabled=stage != "stage1",
            )
            if stage == "stage1" and self.config.per_device_batch(stage) > 1:
                # Stage 1 samples exactly one frame per example. The objective
                # is already a mean of normalized chunk losses, but these two
                # auxiliary fields are sums. Convert them to per-real-frame
                # means before the parallel wrapper weights every local mean.
                if batch["actions"].shape[0] != 1:
                    raise ValueError("Batched Stage 1 requires one-frame samples")
                real_frames = jnp.sum(batch["observation_mask"] & batch["action_mask"].any(axis=(-2, -1)))
                stats = {name: value / jnp.maximum(real_frames, 1)
                         if name in ("supervised_chunks", "scheduled_writes") else value
                         for name, value in stats.items()}
            return loss, stats

        self.grad_fns[stage] = (make_parallel_grad(loss_fn, self.mesh, self.config.per_device_batch(stage))
                               if self.mesh is not None
                               else jax.jit(jax.value_and_grad(loss_fn, has_aux=True)))

        def apply(params: Any, gradients: Any, opt_state: Any) -> Any:
            updates, opt_state = tx.update(gradients, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        if self.mesh is not None:
            self.update_fns[stage] = jax.jit(
                apply, in_shardings=(self.replica_sharding,) * 3,
                out_shardings=(self.replica_sharding,) * 2,
            )
        else:
            self.update_fns[stage] = jax.jit(apply)

    def _single_gradients(self, stage: str, active: dict, task: str, accumulation: int) -> tuple:
        gradients = None
        loss_sum = 0.0
        stats_sums: dict[str, float] = {}
        sequence_lengths, supervised_frames, memory_writes = [], [], []
        data_seconds = 0.0
        micro_batch_size = self.config.per_device_batch(stage)
        for _ in range(accumulation):
            data_started = time.perf_counter()
            samples = [self.sampler.sample(task, stage1=stage == "stage1")
                       for _ in range(micro_batch_size)]
            for sample in samples:
                sequence_lengths.append(int(sample.observation_mask.sum()))
                supervised_frames.append(int(sample.action_mask.any(axis=(-2, -1)).sum()))
                memory_writes.append(int(sample.write_mask.sum()))
            batch = collate_sequences(samples)
            data_seconds += time.perf_counter() - data_started
            batch = jax.device_put(batch)
            self.rng, loss_rng = jax.random.split(self.rng)
            (loss, stats), grads = self.grad_fns[stage](active, self.state, loss_rng, batch)
            gradients = grads if gradients is None else self.accumulate(gradients, grads)
            loss_sum += float(jax.device_get(loss))
            for name, value in stats.items():
                stats_sums[name] = stats_sums.get(name, 0.0) + float(np.asarray(jax.device_get(value)).mean())
            # Only the running gradient remains live across microbatches. Fast
            # state and recurrent activations belong to this one value_and_grad.
            del batch, grads, loss, stats
        gradients = jax.tree.map(lambda value: value / accumulation, gradients)
        return (gradients, loss_sum / accumulation,
                {name: value / accumulation for name, value in stats_sums.items()},
                sequence_lengths, supervised_frames, memory_writes, accumulation, data_seconds)

    def _parallel_gradients(self, stage: str, active: dict, task: str, global_batch: int) -> tuple:
        local_sums = None
        sequence_lengths, supervised_frames, memory_writes = [], [], []
        data_seconds = 0.0
        micro_batch_size = self.config.per_device_batch(stage)
        sizes = wave_sizes(global_batch, self.config.num_devices, micro_batch_size)
        for size in sizes:
            data_started = time.perf_counter()
            samples = []
            for _ in range(size):
                sample = self.sampler.sample(task, stage1=stage == "stage1")
                samples.append(sample)
                sequence_lengths.append(int(sample.observation_mask.sum()))
                supervised_frames.append(int(sample.action_mask.any(axis=(-2, -1)).sum()))
                memory_writes.append(int(sample.write_mask.sum()))
            batch, valid = collate_parallel_wave(samples, self.config.num_devices, micro_batch_size)
            data_seconds += time.perf_counter() - data_started
            keys = []
            for real_count in valid:
                # One independent key per real device microbatch. With M=1 this
                # is the original per-episode stream; empty devices consume none.
                if real_count > 0:
                    self.rng, loss_rng = self.split_rng(self.rng)
                    keys.append(loss_rng)
                else:
                    keys.append(keys[0])
            batch = jax.device_put(batch, self.slot_sharding)
            valid = jax.device_put(valid, self.slot_sharding)
            keys = jax.device_put(jnp.stack(keys), self.slot_sharding)
            (loss, stats), grads = self.grad_fns[stage](active, self.state, keys, batch, valid)
            wave = {"gradients": grads, "loss": loss, "stats": stats}
            local_sums = wave if local_sums is None else self.parallel_accumulate(local_sums, wave)
            # Dispatch is asynchronous. Finish this wave before loading another
            # long prefix, so queued inputs/activations cannot grow with the
            # number of accumulation waves. This adds no cross-device reduction.
            jax.block_until_ready(local_sums)
            # Retain only one gradient accumulator per device, with no recurrent
            # graph or fast-state carry between independently sampled episodes.
            del sample, samples, batch, valid, keys, grads, loss, stats, wave
        # One collective over all waves, before clipping and a single AdamW
        # update. Every real example gets weight 1/global_batch, including tails.
        reduced = self.reduce_gradients(local_sums, self._put_replicated(jnp.asarray(global_batch, jnp.float32)))
        loss, stats = jax.device_get((reduced["loss"], reduced["stats"]))
        return (reduced["gradients"], float(loss),
                {name: float(np.asarray(value).mean()) for name, value in stats.items()},
                sequence_lengths, supervised_frames, memory_writes, len(sizes), data_seconds)

    def step(self) -> dict:
        started = time.perf_counter()
        stage = self.progress.phase
        cycle, phase_step = self.progress.cycle, self.progress.phase_step
        self.ensure_optimizer(stage)
        active = self.state.filter(stage_filter(stage)).to_pure_dict()
        accumulation = (self.config.training.stage1_accumulation_steps if stage == "stage1"
                        else self.config.training.stage2_accumulation_steps)
        task = self.sampler.next_task()
        effective_batch = self.config.effective_batch(stage)
        if self.mesh is None:
            result = self._single_gradients(stage, active, task, accumulation)
        else:
            result = self._parallel_gradients(stage, active, task, effective_batch)
        gradients, loss, stats, sequence_lengths, supervised_frames, memory_writes, waves, data_seconds = result
        gradient_norm = float(jax.device_get(optax.global_norm(gradients)))
        if not np.isfinite(loss) or not np.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite {stage} objective/gradient at step {self.progress.global_step}")
        new_params, new_opt_state = self.update_fns[stage](active, gradients, self.optimizer_states[stage])
        # Include device work in timing, rather than only asynchronous dispatch.
        jax.block_until_ready((new_params, new_opt_state))
        active_state = self.state.filter(stage_filter(stage))
        active_state.replace_by_pure_dict(new_params)
        model = nnx.merge(self.graphdef, self.state)
        nnx.update(model, active_state)
        self.state = nnx.state(model)
        self.optimizer_states[stage] = new_opt_state
        self.progress.advance(self.config)
        elapsed = time.perf_counter() - started
        return {"step": self.progress.global_step, "stage": stage, "cycle": cycle, "phase_step": phase_step + 1,
                "loss": loss, "gradient_norm": gradient_norm, "task": task,
                "effective_batch": effective_batch, "num_devices": self.config.num_devices,
                "parallel_waves": waves, "per_device_batch_size": self.config.per_device_batch(stage),
                "step_seconds": elapsed, "samples_per_second": effective_batch / elapsed,
                # Remaining time includes transfer, dispatch, compilation,
                # collectives, optimization and host control, not pure GPU time.
                "data_seconds": data_seconds, "train_seconds": elapsed - data_seconds,
                "sequence_length_mean": float(np.mean(sequence_lengths)), "sequence_length_max": max(sequence_lengths),
                "supervised_frames": sum(supervised_frames), "scheduled_memory_writes": sum(memory_writes),
                "task_updates": dict(self.sampler.task_updates), "task_samples": dict(self.sampler.task_samples),
                "approximations": self.config.approximation_labels(),
                **stats}

    def save(self) -> Path:
        destination = Path(self.config.checkpoint_dir).expanduser().resolve() / str(self.progress.global_step)
        if destination.exists():
            if (destination / "metadata.json").is_file():
                return destination
            shutil.rmtree(destination)
        if self._norm_stats_bytes is None:
            self._norm_stats_bytes = (Path(self.config.data.norm_stats_dir) / "norm_stats.json").read_bytes()
        destination.mkdir(parents=True)
        # Synchronous host snapshots avoid retaining an asynchronous duplicate
        # of GPU buffers while the next large recurrent graph executes.
        params = jax.device_get(self.state.filter(nnx.Param).to_pure_dict())
        train_state = jax.device_get({"optimizer_states": self.optimizer_states, "rng": self.rng,
                                     "model_variables": self.state.filter(nnx.Not(nnx.Param)).to_pure_dict()})
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(destination / "params", args=ocp.args.PyTreeSave({"params": params}))
            checkpointer.save(destination / "train_state", args=ocp.args.PyTreeSave(train_state))
        assets = destination / "assets" / "robomme"
        assets.mkdir(parents=True)
        (assets / "norm_stats.json").write_bytes(self._norm_stats_bytes)
        metadata = {"format_version": 1, "config": self.config.to_dict(), "progress": asdict(self.progress),
                    "sampler": self.sampler.state_dict(), "optimizer_stages": list(self.optimizer_states),
                    "initialization_source": self.config.initialization_source,
                    "approximations": self.config.approximation_labels(), "complete": True}
        # Presence of metadata marks a fully committed checkpoint. An interrupted
        # save is ignored by latest_checkpoint and rebuilt when retried.
        temporary = destination / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.rename(destination / "metadata.json")
        try:
            _prune_checkpoints(destination.parent, self.config.training.keep_last_checkpoints,
                               current_checkpoint=destination, initialization_source=self.config.initialization_source)
        except Exception:
            # A committed checkpoint remains usable even if listing/removing
            # older snapshots fails. Make storage failures visible in the log.
            logging.warning("Checkpoint saved, but retention failed under %s", destination.parent, exc_info=True)
        return destination

    def restore(self, path: str | Path) -> None:
        path = latest_checkpoint(path)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["config"]["model"] != self.config.model:
            raise ValueError("Resume model architecture must exactly match the saved configuration")
        # Use the selected snapshot's immutable normalization, independently of
        # preparation paths or an earlier source snapshot that retention prunes.
        self._norm_stats_bytes = (path / "assets" / "robomme" / "norm_stats.json").read_bytes()
        model = nnx.merge(self.graphdef, self.state)
        load_initial_parameters(model, path, fresh_memory=False)
        self.state = self._put_replicated(nnx.state(model))
        # Restore may be called on an already-used Trainer. Only the optimizer
        # groups present in this snapshot belong to its saved PyTree structure.
        self.optimizers.clear()
        self.optimizer_states.clear()
        self.grad_fns.clear()
        self.update_fns.clear()
        for stage in metadata["optimizer_stages"]:
            self.ensure_optimizer(stage)
        target = {"optimizer_states": self.optimizer_states, "rng": self.rng,
                  "model_variables": self.state.filter(nnx.Not(nnx.Param)).to_pure_dict()}
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(path / "train_state", args=ocp.args.PyTreeRestore(item=target))
        self.optimizer_states = self._put_replicated(restored["optimizer_states"])
        self.rng = self._put_replicated(restored["rng"])
        variables = self.state.filter(nnx.Not(nnx.Param))
        variables.replace_by_pure_dict(self._put_replicated(restored["model_variables"]))
        model = nnx.merge(self.graphdef, self.state)
        nnx.update(model, variables)
        self.state = self._put_replicated(nnx.state(model))
        self.progress = PhaseProgress(**metadata["progress"])
        self.sampler.load_state_dict(metadata["sampler"])


def load_norm_stats(config: RunConfig) -> dict:
    return normalize.load(config.data.norm_stats_dir)
