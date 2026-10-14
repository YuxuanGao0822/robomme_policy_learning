#!/usr/bin/env python3
"""Train Stage 1, Stage 2A/2B, alternate, or resume a complete T²Mem run."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mme_vla_suite.models.t2mem.configuration import expand_path, load_run_config


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def should_save_checkpoint(global_step: int, save_interval: int, *, complete: bool) -> bool:
    """Save periodic optimizer updates and the final state, regardless of phase."""
    if type(save_interval) is not int or save_interval <= 0:
        raise ValueError("save_interval must be a positive integer")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer")
    return complete or (global_step > 0 and global_step % save_interval == 0)


def validate_stage1_initialization(source_progress: dict, required_updates: int) -> None:
    """Require a completed Stage-1 source with this destination's A budget."""
    if not source_progress["complete"]:
        raise ValueError("Finish the configured Stage-1 budget before initializing Experiment B")
    completed_updates = source_progress["global_step"]
    if type(completed_updates) is not int or completed_updates < required_updates:
        raise ValueError(
            f"Stage 2 requires a completed Stage-1 checkpoint with at least {required_updates} optimizer updates; "
            f"the selected checkpoint has {completed_updates}. A completed smoke run cannot initialize formal B."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/t2mem/t2mem_a100_80gb.yaml")
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2a", "stage2b", "alternating", "resume"))
    parser.add_argument("--init-checkpoint", help="Stage-1 or preceding phase checkpoint directory")
    parser.add_argument("--resume", help="Complete checkpoint directory or run root; preserves progress, optimizers and RNG")
    parser.add_argument("--checkpoint-dir", help="Output run directory; resume can migrate into a separate directory")
    parser.add_argument("--num-devices", type=_positive_int,
                        help="Override visible local device count for a fresh run or explicit resume")
    parser.add_argument("--gpu-id", type=int,
                        help="First physical GPU for default contiguous device selection; also applies on resume")
    parser.add_argument("--cycles", type=int, help="Explicit total alternating cycles; paper does not specify it")
    parser.add_argument("--updates", type=int, help="Override budget for a standalone stage")
    parser.add_argument("--save-interval", type=_positive_int,
                        help="Checkpoint interval in optimizer updates; also applies when resuming")
    parser.add_argument("--stage1-per-device-batch", type=_positive_int,
                        help="Enable batched Stage1: global batch = devices * this value * accumulation waves")
    parser.add_argument("--stage1-accumulation-steps", type=_positive_int,
                        help="Override Stage1 accumulation; throughput mode counts waves per GPU")
    parser.add_argument("--benchmark-steps", type=_positive_int,
                        help="Fresh Stage1 only: measure this many updates after warmup, without saving weights")
    parser.add_argument("--benchmark-warmup-steps", type=_positive_int, default=2,
                        help="Discard these initial benchmark updates, including compilation (default: 2)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    resume_path = None
    if args.stage == "resume":
        if not args.resume or args.init_checkpoint:
            raise ValueError("resume requires --resume and cannot use --init-checkpoint")
        # Resolve saved config before initializing JAX, including GPU selection.
        requested = expand_path(args.resume)
        if (requested / "metadata.json").is_file():
            resume_path = requested
        else:
            candidates = [p for p in requested.iterdir() if p.is_dir() and p.name.isdigit()
                          and (p / "metadata.json").is_file()]
            if not candidates:
                raise FileNotFoundError(f"No complete T²Mem checkpoints under {requested}")
            resume_path = max(candidates, key=lambda p: int(p.name))
        config = load_run_config(resume_path / "metadata.json")
        # Resume with the immutable statistics saved with these model/optimizer
        # weights, even if the original preparation directory was moved/edited.
        config.data.norm_stats_dir = str(resume_path / "assets" / "robomme")
    else:
        if args.resume:
            raise ValueError("Use --stage resume to restore optimizer, phase and random state")
        config = load_run_config(args.config)
    # Device placement can change on explicit resume without changing logical
    # model/optimizer shapes or resetting completed updates and sampling state.
    if args.num_devices is not None:
        config.num_devices = args.num_devices
    if args.gpu_id is not None:
        config.gpu_id = args.gpu_id
    if args.checkpoint_dir:
        config.checkpoint_dir = str(expand_path(args.checkpoint_dir))
    if args.cycles is not None:
        config.training.alternating_cycles = args.cycles
    if args.updates is not None:
        if args.stage in ("alternating", "resume"):
            raise ValueError("--updates applies to a standalone stage; use --cycles for alternation")
        name = {"stage1": "stage1_updates", "stage2a": "memory_updates_per_cycle",
                "stage2b": "policy_updates_per_cycle"}[args.stage]
        setattr(config.training, name, args.updates)
    if args.save_interval is not None:
        config.training.save_interval = args.save_interval
    if args.stage1_per_device_batch is not None:
        config.training.stage1_per_device_batch_size = args.stage1_per_device_batch
    if args.stage1_accumulation_steps is not None:
        config.training.stage1_accumulation_steps = args.stage1_accumulation_steps
    if args.benchmark_steps is not None:
        if args.stage != "stage1" or args.resume or args.init_checkpoint or args.updates is not None:
            raise ValueError("Benchmark requires fresh Stage1 from official base, without resume/init/updates")
        config.training.stage1_updates = args.benchmark_warmup_steps + args.benchmark_steps
    config.validate()
    # Environment is set before importing any JAX/Flax modules. A single GPU
    # preserves the original physical index; DP defaults to a contiguous range.
    # Existing user/scheduler selections take precedence over these defaults.
    visible_default = ",".join(str(config.gpu_id + offset) for offset in range(config.num_devices))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible_default)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from flax import nnx

    from mme_vla_suite.models.t2mem.dataset import EpisodeSampler, load_manifest, make_transforms
    from mme_vla_suite.models.t2mem.training import (
        PhaseProgress, Trainer, latest_checkpoint, load_initial_parameters, load_norm_stats,
    )

    devices = jax.local_devices()
    if (jax.process_count() != 1 or len(devices) != config.num_devices
            or jax.device_count() != config.num_devices or any(device.platform != "gpu" for device in devices)):
        raise RuntimeError(
            f"This runner requires one JAX process and {config.num_devices} visible local GPUs; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, devices={jax.devices()}. "
            "Use a distinct run directory for each independent training run."
        )
    device_kinds = {device.device_kind for device in devices}
    if len(device_kinds) != 1:
        raise RuntimeError(f"T²Mem synchronous DP requires GPUs of the same device kind; got {device_kinds}")
    logging.info("JAX devices: CUDA_VISIBLE_DEVICES=%s; process_index=%s; local_devices=%s; kind=%s",
                 os.environ["CUDA_VISIBLE_DEVICES"], jax.process_index(), devices, next(iter(device_kinds)))
    progress = PhaseProgress.start(args.stage) if resume_path is None else PhaseProgress(**json.loads(
        (resume_path / "metadata.json").read_text())["progress"])
    if progress.mode == "alternating" and config.training.alternating_cycles is None:
        raise ValueError("Supply --cycles; total alternating cycles are not specified by Table 3")
    if args.cycles is not None:
        progress.set_cycle_budget(args.cycles)
    output = expand_path(config.checkpoint_dir)
    if output.exists() and any(output.iterdir()) and (resume_path is None or output != resume_path.parent):
        raise FileExistsError(f"Run directory already contains files: {output}; use a new output directory "
                              "or resume within the source checkpoint's own run directory")
    output.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir = str(output)
    model_config = config.create_model_config()
    logging.info("Experiment: %s; approximations: %s", config.experiment_label, config.approximation_labels())
    logging.info("Resolved config: %s", json.dumps(config.to_dict()))
    logging.info("Batch settings: Stage1 per-device=%s, global=%s; Stage2 per-device=%s, global=%s",
                 config.per_device_batch("stage1"), config.effective_batch("stage1"),
                 config.per_device_batch("stage2a"), config.effective_batch("stage2a"))
    model = model_config.create(jax.random.PRNGKey(config.seed))
    if args.init_checkpoint:
        source = latest_checkpoint(args.init_checkpoint)
        source_metadata = json.loads((source / "metadata.json").read_text())
        if source_metadata["config"]["model"] != config.model:
            raise ValueError("Initial phase checkpoint architecture must match this run")
        source_mode = source_metadata["progress"]["mode"]
        if args.stage == "alternating" and source_mode != "stage1":
            raise ValueError("Experiment B must initialize from Experiment A's Stage-1 checkpoint; resume an existing B run instead")
        if args.stage == "stage2b" and source_mode == "stage1":
            raise ValueError("Stage 2B requires a preceding memory-learning phase, not a Stage-1 checkpoint")
        if source_mode == "stage1":
            validate_stage1_initialization(source_metadata["progress"], config.training.stage1_updates)
        source_stats = (source / "assets/robomme/norm_stats.json").read_bytes()
        destination_stats = (Path(config.data.norm_stats_dir) / "norm_stats.json").read_bytes()
        if hashlib.sha256(source_stats).digest() != hashlib.sha256(destination_stats).digest():
            raise ValueError("Stage 1 and Stage 2 must use identical observation/action normalization statistics")
        config.initialization_source = str(source)
        load_initial_parameters(model, source, fresh_memory=source_mode == "stage1")
    elif resume_path is None and args.stage == "stage1":
        config.initialization_source = str(expand_path(config.pretrained_params))
        load_initial_parameters(model, expand_path(config.pretrained_params), fresh_memory=True,
                                allow_new_policy_parameters=True)
    elif resume_path is None:
        raise ValueError("Stage 2 requires --init-checkpoint pointing to Stage 1 or a preceding memory phase")
    # Base vision/VLM weights remain frozen and occupy BF16 storage. Slow
    # memory and all potentially trainable policy tensors retain FP32 master
    # weights; fast state is FP32 inside the model.
    from mme_vla_suite.models.t2mem.training import parameter_group

    state = nnx.state(model)
    state = state.map(lambda path, variable: variable.replace(variable.value.astype(jax.numpy.bfloat16))
                      if isinstance(variable, nnx.VariableState) and issubclass(variable.type, nnx.Param)
                      and parameter_group(path) == "frozen" else variable)
    nnx.update(model, state)
    manifest = load_manifest(config.data.manifest)
    transform = make_transforms(model_config, load_norm_stats(config))
    sampler = EpisodeSampler(manifest, transform, horizon=model_config.action_horizon,
                             action_dim=model_config.action_dim, write_stride=config.data.write_stride,
                             demonstration_write_offset=config.data.demonstration_write_offset,
                             action_supervision_stride=config.data.action_supervision_stride,
                             task_sampling=config.data.task_sampling, task_weights=config.data.task_weights,
                             seed=config.seed, max_sequence_length=config.data.max_sequence_length,
                             truncate_prefix=config.data.truncate_prefix, cache_actions=config.data.cache_actions)
    logging.info("Task sampling probabilities: %s", dict(zip(manifest["tasks"], sampler.probabilities.tolist())))
    trainer = Trainer(model, config, progress, sampler)
    # The trainer owns the canonical model state. Release the initial single-
    # device arrays so they cannot pin obsolete parameter buffers on GPU 0.
    del model, state
    if resume_path is not None:
        trainer.restore(resume_path)
        if args.cycles is not None:
            trainer.progress.set_cycle_budget(args.cycles)
    (output / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    benchmark_metrics = []
    with (output / "metrics.jsonl").open("a") as log:
        while not trainer.progress.complete:
            metrics = trainer.step()
            if args.benchmark_steps is not None:
                benchmark_metrics.append(metrics)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if trainer.progress.global_step % config.training.log_interval == 0:
                logging.info("%s", json.dumps(metrics))
            # Phase changes preserve optimizer/progress in memory, and are
            # included in the next snapshot. Save cadence counts optimizer
            # updates, independently of memory/policy phase boundaries.
            if args.benchmark_steps is None and should_save_checkpoint(
                    trainer.progress.global_step, config.training.save_interval, complete=trainer.progress.complete):
                logging.info("Saved %s", trainer.save())
    if args.benchmark_steps is not None:
        measured = benchmark_metrics[args.benchmark_warmup_steps:]
        seconds = sum(item["step_seconds"] for item in measured)
        samples = sum(item["effective_batch"] for item in measured)
        memory = []
        for device in devices:
            stats = device.memory_stats() or {}
            memory.append({"device": str(device), "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                           "bytes_limit": stats.get("bytes_limit")})
        report = {"per_device_batch": config.per_device_batch("stage1"),
                  "global_batch": config.effective_batch("stage1"), "num_devices": config.num_devices,
                  "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                  "warmup_steps": args.benchmark_warmup_steps, "measured_steps": len(measured),
                  "step_seconds": [item["step_seconds"] for item in measured],
                  "data_seconds": [item.get("data_seconds") for item in measured],
                  "train_seconds": [item.get("train_seconds") for item in measured],
                  "samples_per_second": samples / seconds, "memory": memory,
                  "weights_saved": False, "initialization_source": config.initialization_source}
        (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        logging.info("Benchmark (weights discarded): %s", json.dumps(report))
    logging.info("Training budget completed: %s", trainer.progress)


if __name__ == "__main__":
    main()
