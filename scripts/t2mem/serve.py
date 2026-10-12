"""Serve a trained T²Mem checkpoint with incremental episode-local memory."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path

# Set before importing JAX. GPU 6 is physical; this process sees logical GPU 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")


def load_policy(
    checkpoint: Path | str,
    *,
    seed: int = 7,
    ablation: str = "full",
    num_steps: int = 10,
    memory_write_stride: int | None = None,
    demo_write_offset: int | None = None,
):
    import jax

    from openpi import transforms
    from openpi.models.model import restore_params
    from openpi.shared import normalize
    from mme_vla_suite.models.t2mem.configuration import load_run_config
    from mme_vla_suite.models.t2mem.dataset import make_transforms
    from mme_vla_suite.models.t2mem.policy import T2MemPolicy
    from mme_vla_suite.models.t2mem.policy import WriteSchedule
    from mme_vla_suite.models.t2mem.training import latest_checkpoint
    from mme_vla_suite.policies.robomme_policy import RoboMMEOutputs

    if len(jax.devices()) != 1 or jax.default_backend() != "gpu":
        raise RuntimeError("T²Mem serving requires one visible GPU; set CUDA_VISIBLE_DEVICES=6 on the server.")
    checkpoint = latest_checkpoint(Path(checkpoint).expanduser().resolve())
    run = load_run_config(checkpoint / "metadata.json")
    saved_metadata = json.loads((checkpoint / "metadata.json").read_text())
    model_config = run.create_model_config()
    checkpoint_mode = saved_metadata["progress"]["mode"]
    if ablation == "memory-free":
        if checkpoint_mode != "stage1":
            raise ValueError("Experiment A must load its own Stage-1 checkpoint.")
        model_config = dataclasses.replace(model_config, memory_enabled=False)
    elif ablation == "no-online-write":
        if checkpoint_mode == "stage1" or not model_config.memory_enabled:
            raise ValueError("No-Write must load Experiment B's memory-trained checkpoint.")
        model_config = dataclasses.replace(model_config, memory=dataclasses.replace(model_config.memory, online_write=False))
    elif ablation != "full":
        raise ValueError(f"Unknown ablation: {ablation}.")
    elif checkpoint_mode == "stage1" or not model_config.memory_enabled or not model_config.memory.online_write:
        raise ValueError("Full T²Mem requires a memory-trained checkpoint with memory and online writing enabled.")
    if model_config.action_horizon != 64:
        raise ValueError("The RoboMME T²Mem server expects action horizon 64.")
    if num_steps <= 0:
        raise ValueError("Flow denoising steps must be positive.")
    # Preserve checkpoint dtypes, especially learned W0/projections/fast-state
    # precision. No blanket BF16 conversion of all saved parameters is permitted.
    model = model_config.load(restore_params(checkpoint / "params"))
    norm_stats = normalize.load(checkpoint / "assets" / "robomme")
    normalization_digest = hashlib.sha256((checkpoint / "assets" / "robomme" / "norm_stats.json").read_bytes()).hexdigest()
    schedule = WriteSchedule(
        stride=memory_write_stride if memory_write_stride is not None else run.data.write_stride,
        demo_offset=demo_write_offset if demo_write_offset is not None else run.data.demonstration_write_offset,
    )
    approximation_labels = run.approximation_labels()
    return T2MemPolicy(
        model,
        seed=seed,
        num_steps=num_steps,
        schedule=schedule,
        transforms_in=(make_transforms(model_config, norm_stats),),
        transforms_out=(
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            transforms.AbsoluteActions(transforms.make_bool_mask(7, -1)),
            RoboMMEOutputs(),
        ),
        metadata={
            "checkpoint": str(checkpoint),
            "checkpoint_progress": saved_metadata["progress"],
            "initialization_source": saved_metadata.get("initialization_source"),
            "training_config": run.to_dict(),
            "preprocessing": {
                "pipeline": "RoboMMEInputs;delta-joints;train-quantile-normalization;resize224;dual-pi05-instruction-prompts;pad:v1",
                "action_dim": model_config.action_dim,
                "robot_action_dim": model_config.robot_action_dim,
                "max_token_len": model_config.max_token_len,
                "normalization_sha256": normalization_digest,
                "tokenizer": "openpi.PaligemmaTokenizer",
                "cameras": ["base_0_rgb", "left_wrist_0_rgb"],
            },
            "success_predicate": "RoboMME terminal info.status == success",
            "ablation": ablation,
            "memory_enabled": model_config.memory_enabled,
            "online_write": model_config.memory.online_write,
            "adaptive_write": model_config.memory.adaptive_write,
            "curvature_calibration": True,
            "model_config": dataclasses.asdict(model_config),
            "approximations": approximation_labels,
            "schedule_override": memory_write_stride is not None or demo_write_offset is not None,
            "flow_denoising_steps": num_steps,
        },
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved optimizer-step directory or its run root (latest completed checkpoint).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ablation", choices=("full", "memory-free", "no-online-write"), default="full")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--memory-write-stride", type=int)
    parser.add_argument("--demo-write-offset", type=int)
    args = parser.parse_args(argv)
    from mme_vla_suite.models.t2mem.policy import T2MemWebsocketPolicyServer

    policy = load_policy(
        args.checkpoint, seed=args.seed, ablation=args.ablation, num_steps=args.num_denoise_steps,
        memory_write_stride=args.memory_write_stride, demo_write_offset=args.demo_write_offset,
    )
    logging.info("Serving %s", json.dumps(policy.metadata, indent=2))
    T2MemWebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
