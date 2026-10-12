"""Server tests: NOT RUN locally; requires Linux JAX/Flax runtime.

Small synthetic fixtures exercise filtering, padding, ownership and complete
resume without constructing the pi0.5 backbone or downloading anything.
"""

import json
from pathlib import Path
import tempfile

from flax import nnx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.t2mem.configuration import RunConfig
from mme_vla_suite.models.t2mem.dataset import (
    EpisodeSampler, TASKS, build_manifest, collate_sequences, load_manifest, read_frame,
)
from mme_vla_suite.models.t2mem.training import (
    PhaseProgress, Trainer, latest_checkpoint, load_initial_parameters, parameter_group, stage_filter,
)


@pytest.fixture
def training_data():
    # Keep the fixture outside pytest's test_* directory: production deliberately
    # rejects explicitly held-out path components.
    with tempfile.TemporaryDirectory(prefix="t2mem_official_train_") as directory:
        root = Path(directory)
        path = root / "data_PickXtimes.h5"
        with h5py.File(path, "w") as source:
            source.attrs["split"] = "train"
            episode = source.create_group("episode_4")
            episode.create_dataset("setup/task_goal", data=np.asarray([b"Pick up three times"]))
            for index in range(3):
                frame = episode.create_group(f"timestep_{index}")
                frame.create_dataset("info/is_video_demo", data=False)
                frame.create_dataset("obs/front_rgb", data=np.full((4, 4, 3), index, np.uint8))
                frame.create_dataset("obs/wrist_rgb", data=np.full((4, 4, 3), index, np.uint8))
                frame.create_dataset("obs/joint_state", data=np.arange(7, dtype=np.float32))
                frame.create_dataset("obs/gripper_state", data=np.ones(1, np.float32))
                frame.create_dataset("action/joint_action", data=np.full(8, index, np.float32))
        # A contaminating unrelated task must never even be opened.
        (root / "data_VideoUnmask.h5").write_bytes(b"not an HDF5 training file")
        manifest = build_manifest(root, root / "manifest.json")
        yield root, manifest


def transform(raw):
    return {"image": {"base_0_rgb": raw["observation/image"], "left_wrist_0_rgb": raw["observation/wrist_image"]},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_},
            "state": np.pad(raw["observation/state"], (0, 24)),
            "actions": np.pad(raw["actions"], ((0, 0), (0, 24))),
            "tokenized_prompt": np.asarray([1, 2], np.int32),
            "tokenized_prompt_mask": np.ones(2, bool),
            "tokenized_pi05_prompt": np.asarray([3, 4], np.int32),
            "tokenized_pi05_prompt_mask": np.ones(2, bool)}


def test_only_pickxtimes_is_indexed(training_data):
    root, manifest = training_data
    assert TASKS == ("PickXtimes",)
    assert manifest["tasks"] == ["PickXtimes"]
    assert [entry["task"] for entry in manifest["episodes"]] == ["PickXtimes"]
    assert load_manifest(root / "manifest.json")["split"] == "train"
    invalid = dict(manifest, tasks=["PickXtimes", "VideoUnmask"])
    (root / "invalid.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="exactly"):
        load_manifest(root / "invalid.json")


def test_official_release_filename_preserves_episode_sequence(training_data):
    root, legacy_manifest = training_data
    official = root / "record_dataset_PickXtimes.h5"
    (root / "data_PickXtimes.h5").rename(official)
    # Match the official RecordWrapper's variable-length task_goal encoding;
    # indexing this HDF5 object array returns Python bytes, not np.bytes_.
    with h5py.File(official, "r+") as source:
        del source["episode_4/setup/task_goal"]
        source.create_dataset("episode_4/setup/task_goal", data=np.asarray(["Pick up three times"], dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
    manifest = build_manifest(root, root / "official_manifest.json")
    assert manifest["task_counts"] == {"PickXtimes": 1}
    assert manifest["episodes"][0]["path"] == str(official)
    assert manifest["episodes"][0]["prompt"] == "Pick up three times"
    # File naming must not alter original episode IDs, chronology, targets,
    # or the expected physical-time write schedule.
    assert manifest["episodes"][0]["episode"] == legacy_manifest["episodes"][0]["episode"] == "episode_4"
    assert load_manifest(root / "official_manifest.json")["episodes"] == manifest["episodes"]
    sample = EpisodeSampler(manifest, transform).sample("PickXtimes", stage1=False)
    np.testing.assert_array_equal(sample.physical_steps, [0, 1, 2])
    np.testing.assert_array_equal(sample.write_mask, [True, False, False])
    np.testing.assert_array_equal(sample.actions[0, :3, 0], [0, 1, 2])


def test_ambiguous_official_and_legacy_files_are_rejected(training_data):
    root, _ = training_data
    # Even identical content under two names is ambiguous: choosing one could
    # silently use a stale copy after the official release is updated.
    (root / "record_dataset_PickXtimes.h5").symlink_to(root / "data_PickXtimes.h5")
    with pytest.raises(ValueError, match="Ambiguous PickXtimes"):
        build_manifest(root, root / "ambiguous_manifest.json")


def test_manifest_cannot_relabel_another_task_as_pickxtimes(training_data):
    root, manifest = training_data
    for filename in ("data_VideoUnmask.h5", "record_dataset_VideoUnmask.h5",
                     "record_dataset_PatternLock.h5"):
        invalid = {**manifest, "episodes": [{**manifest["episodes"][0], "path": str(root / filename)}]}
        (root / "relabelled_manifest.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="task identity"):
            load_manifest(root / "relabelled_manifest.json")


def test_held_out_split_rejected(training_data):
    root, _ = training_data
    with h5py.File(root / "data_PickXtimes.h5", "r+") as source:
        source.attrs["split"] = "test"
    with pytest.raises(ValueError, match="Non-training split"):
        build_manifest(root, root / "held_out.json")


def test_sequence_order_future_chunk_padding_and_write_cadence(training_data):
    _, manifest = training_data
    sampler = EpisodeSampler(manifest, transform, seed=7)
    sample = sampler.sample("PickXtimes", stage1=False)
    np.testing.assert_array_equal(sample.physical_steps, [0, 1, 2])
    np.testing.assert_array_equal(sample.write_mask, [True, False, False])
    assert sample.action_mask[0].sum() == 3 * 8
    assert sample.action_mask[2].sum() == 8
    assert not sample.action_mask[..., 8:].any()
    batch = collate_sequences([sample])
    assert batch["actions"].shape == (4, 1, 64, 32)
    assert not batch["observation_mask"][3].any()
    assert not batch["action_mask"][3].any()
    assert not batch["write_mask"][3].any()
    state = sampler.state_dict()
    next_sample = sampler.sample(sampler.next_task(), stage1=True)
    sampler.load_state_dict(state)
    resumed_sample = sampler.sample(sampler.next_task(), stage1=True)
    np.testing.assert_array_equal(next_sample.physical_steps, resumed_sample.physical_steps)
    assert not resumed_sample.write_mask.any()


def test_demo_actions_are_never_supervised(training_data):
    root, _ = training_data
    with h5py.File(root / "data_PickXtimes.h5", "r+") as source:
        source["episode_4/timestep_0/info/is_video_demo"][()] = True
    manifest = build_manifest(root, root / "demo.json")
    sampler = EpisodeSampler(manifest, transform)
    sample = sampler.sample("PickXtimes", stage1=False)
    assert sample.write_mask[0]
    assert not sample.action_mask[0].any()
    assert sample.action_mask[1:].any()
    with h5py.File(root / "data_PickXtimes.h5", "r") as source:
        _, mask = read_frame(source["episode_4"], 0, "instruction", 64, 32)
        assert not mask.any()


class TinyModel(nnx.Module):
    def __init__(self):
        self.PaliGemma = nnx.Dict(llm=nnx.Dict(layers=nnx.Dict(
            memory=nnx.Dict(w0=nnx.Param(jnp.asarray(0.2)), alpha=nnx.Param(jnp.asarray(0.3))),
            expert_1=nnx.Param(jnp.asarray(0.4)), lora=nnx.Param(jnp.asarray(0.5)),
            base=nnx.Param(jnp.asarray(0.6))), memory_interface_tokens=nnx.Param(jnp.asarray(0.1))),
            img=nnx.Dict(kernel=nnx.Param(jnp.asarray(0.7))))
        self.state_proj = nnx.Linear(32, 1, rngs=nnx.Rngs(0))
        self.action_out_proj = nnx.Linear(1, 32, rngs=nnx.Rngs(1))
        self.deterministic = True

    def compute_sequence_loss(self, rng, observation, actions, action_mask, write_mask, observation_mask,
                              *, train, memory_enabled):
        del rng, observation, actions, action_mask, write_mask, observation_mask, train, memory_enabled
        loss = sum(jnp.sum(value**2) for value in jax.tree.leaves(nnx.state(self, nnx.Param).to_pure_dict()))
        return loss, {"synthetic": loss}


def test_stage_parameter_filters():
    state = nnx.state(TinyModel(), nnx.Param)
    for stage, allowed in (("stage1", {"policy"}), ("stage2a", {"memory", "gate"}),
                           ("stage2b", {"policy", "proprioception"})):
        from flax import traverse_util
        paths = set(traverse_util.flatten_dict(state.filter(stage_filter(stage)).to_pure_dict()))
        assert paths
        assert all(parameter_group(path) in allowed for path in paths)
        assert not any("base" in path or "img" in path for path in paths)
        if stage == "stage1":
            assert not any("state_proj" in path for path in paths)
    assert parameter_group(("PaliGemma", "llm", "memory_interface_tokens")) == "memory"
    assert parameter_group(("PaliGemma", "llm", "layers", "memory", "alpha")) == "gate"


def test_alternating_phase_progress():
    config = RunConfig()
    config.training.memory_updates_per_cycle = 1
    config.training.policy_updates_per_cycle = 1
    config.training.alternating_cycles = 2
    progress = PhaseProgress.start("alternating")
    assert progress.phase == "stage2a"
    progress.advance(config)
    assert progress.phase == "stage2b" and progress.phase_step == 0
    progress.advance(config)
    assert progress.phase == "stage2a" and progress.cycle == 1
    progress.advance(config)
    progress.advance(config)
    assert progress.complete and progress.global_step == 4
    assert progress.stage_updates == {"stage1": 0, "stage2a": 2, "stage2b": 2}
    progress.set_cycle_budget(3)
    assert not progress.complete and progress.cycle == 2
    assert progress.phase == "stage2a" and progress.phase_step == 0
    assert progress.global_step == 4
    assert progress.stage_updates == {"stage1": 0, "stage2a": 2, "stage2b": 2}
    with pytest.raises(ValueError, match="discard"):
        progress.set_cycle_budget(2)
    with pytest.raises(ValueError, match="alternating"):
        PhaseProgress.start("stage1").set_cycle_budget(3)


def test_optimizer_rng_phase_and_sampler_resume(training_data):
    root, manifest = training_data
    config = RunConfig(checkpoint_dir=str(root / "checkpoints"))
    config.data.norm_stats_dir = str(root / "assets")
    (root / "assets").mkdir()
    (root / "assets/norm_stats.json").write_text("{}")
    config.training.memory_updates_per_cycle = 1
    config.training.policy_updates_per_cycle = 1
    config.training.alternating_cycles = 2
    config.training.stage2_accumulation_steps = 1
    trainer = Trainer(TinyModel(), config, PhaseProgress.start("alternating"), EpisodeSampler(manifest, transform))
    trainer.step()
    checkpoint = trainer.save()
    assert latest_checkpoint(root / "checkpoints") == checkpoint
    expected = trainer.step()
    resumed = Trainer(TinyModel(), config, PhaseProgress.start("alternating"), EpisodeSampler(manifest, transform))
    resumed.restore(checkpoint)
    actual = resumed.step()
    assert actual["step"] == expected["step"]
    assert actual["stage"] == expected["stage"]
    assert actual["cycle"] == expected["cycle"] == 0
    assert actual["phase_step"] == expected["phase_step"] == 1
    np.testing.assert_allclose(actual["loss"], expected["loss"], rtol=0, atol=0)
    for original, restored in zip(jax.tree.leaves(trainer.state), jax.tree.leaves(resumed.state), strict=True):
        np.testing.assert_array_equal(original, restored)
    # Rewinding this used Trainer drops its new Stage-2B optimizer before restoring
    # the older checkpoint, which only contains Stage-2A's optimizer subtree.
    resumed.restore(checkpoint)
    assert set(resumed.optimizer_states) == {"stage2a"}
    rewound = resumed.step()
    np.testing.assert_allclose(rewound["loss"], expected["loss"], rtol=0, atol=0)


def test_temporal_rematerialization_configuration():
    config = RunConfig(model={"temporal_remat_block_size": 8})
    config.validate()
    assert config.create_model_config().temporal_remat_block_size == 8
    assert config.approximation_labels() == []
    for invalid in (0, -1, 1.5, True):
        config.model["temporal_remat_block_size"] = invalid
        with pytest.raises(ValueError, match="positive integer"):
            config.validate()
    for invalid in (7, 9, True):
        config = RunConfig(model={"robot_action_dim": invalid})
        with pytest.raises(ValueError, match="robot_action_dim=8"):
            config.validate()
