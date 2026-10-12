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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/t2mem/t2mem_a100_80gb.yaml")
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2a", "stage2b", "alternating", "resume"))
    parser.add_argument("--init-checkpoint", help="Stage-1 or preceding phase checkpoint directory")
    parser.add_argument("--resume", help="Complete checkpoint directory or its run root")
    parser.add_argument("--checkpoint-dir", help="Output run directory")
    parser.add_argument("--cycles", type=int, help="Explicit total alternating cycles; paper does not specify it")
    parser.add_argument("--updates", type=int, help="Override budget for a standalone stage")
    return parser.parse_args()


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
    # Environment is set before importing any JAX/Flax modules.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu_id))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from flax import nnx

    from mme_vla_suite.models.t2mem.dataset import EpisodeSampler, load_manifest, make_transforms
    from mme_vla_suite.models.t2mem.training import (
        PhaseProgress, Trainer, latest_checkpoint, load_initial_parameters, load_norm_stats,
    )

    if jax.process_count() != 1 or jax.device_count() != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError(f"This runner requires one visible GPU and one JAX process, got {jax.devices()}")
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
    config.validate()
    progress = PhaseProgress.start(args.stage) if resume_path is None else PhaseProgress(**json.loads(
        (resume_path / "metadata.json").read_text())["progress"])
    if progress.mode == "alternating" and config.training.alternating_cycles is None:
        raise ValueError("Supply --cycles; total alternating cycles are not specified by Table 3")
    if args.cycles is not None:
        progress.set_cycle_budget(args.cycles)
    output = expand_path(config.checkpoint_dir)
    if resume_path is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {output}; choose another --checkpoint-dir or resume")
    output.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir = str(output)
    model_config = config.create_model_config()
    logging.info("Experiment: %s; approximations: %s", config.experiment_label, config.approximation_labels())
    logging.info("Resolved config: %s", json.dumps(config.to_dict()))
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
        if source_mode == "stage1" and not source_metadata["progress"]["complete"]:
            raise ValueError("Finish the configured Stage-1 budget before initializing Experiment B")
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
                             truncate_prefix=config.data.truncate_prefix)
    logging.info("Task sampling probabilities: %s", dict(zip(manifest["tasks"], sampler.probabilities.tolist())))
    trainer = Trainer(model, config, progress, sampler)
    if resume_path is not None:
        trainer.restore(resume_path)
        if args.cycles is not None:
            trainer.progress.set_cycle_budget(args.cycles)
    (output / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    with (output / "metrics.jsonl").open("a") as log:
        while not trainer.progress.complete:
            old_phase = trainer.progress.phase
            metrics = trainer.step()
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if trainer.progress.global_step % config.training.log_interval == 0:
                logging.info("%s", json.dumps(metrics))
            if trainer.progress.global_step % config.training.save_interval == 0 or trainer.progress.complete or old_phase != trainer.progress.phase:
                logging.info("Saved %s", trainer.save())
    logging.info("Training budget completed: %s", trainer.progress)


if __name__ == "__main__":
    main()
