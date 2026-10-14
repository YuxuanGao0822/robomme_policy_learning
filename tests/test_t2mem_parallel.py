"""Server-side data-parallel regressions; never run on the local Mac.

Multi-device cases skip when the process exposes too few devices. Run them on
the server with JAX_PLATFORMS=cpu and
XLA_FLAGS=--xla_force_host_platform_device_count=6. Synthetic observations,
small NNX parameters and resumable host sampling require no model downloads.
"""

from collections import Counter
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import runpy
import shutil
import sys

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from mme_vla_suite.models.t2mem.configuration import RunConfig, config_from_dict, load_run_config
from mme_vla_suite.models.t2mem.dataset import SequenceSample, collate_sequences
from mme_vla_suite.models.t2mem.observation import T2Observation
from mme_vla_suite.models.t2mem.parallel import (
    collate_parallel_wave,
    make_gradient_reducer,
    make_parallel_accumulator,
    make_parallel_grad,
    wave_sizes,
)
from mme_vla_suite.models.t2mem.sequence import differentiable_scan
from mme_vla_suite.models.t2mem.training import PhaseProgress, Trainer


@pytest.mark.parametrize(
    ("global_batch", "num_devices", "expected"),
    [(64, 6, (6,) * 10 + (4,)), (32, 6, (6,) * 5 + (2,)), (3, 6, (3,)), (5, 2, (2, 2, 1)), (3, 1, (1, 1, 1))],
)
def test_waves_preserve_true_global_batch_and_only_pad_the_final_wave(global_batch, num_devices, expected):
    actual = wave_sizes(global_batch, num_devices)
    assert actual == expected
    assert sum(actual) == global_batch
    assert all(0 < count <= num_devices for count in actual)


@pytest.mark.parametrize(("global_batch", "num_devices"), [(0, 6), (32, 0), (True, 6), (32, 1.5)])
def test_invalid_wave_counts_fail_before_sampling(global_batch, num_devices):
    with pytest.raises(ValueError, match="positive integer"):
        wave_sizes(global_batch, num_devices)


@pytest.mark.parametrize(
    ("global_batch", "num_devices", "micro_batch_size", "expected"),
    [(6, 2, 2, (4, 2)), (7, 2, 2, (4, 3)), (14, 6, 2, (12, 2)), (3, 6, 2, (3,))],
)
def test_per_device_microbatches_bound_waves_without_rounding_up_global_batch(
    global_batch, num_devices, micro_batch_size, expected,
):
    actual = wave_sizes(global_batch, num_devices, micro_batch_size=micro_batch_size)
    assert actual == expected
    assert sum(actual) == global_batch
    assert max(actual) <= num_devices * micro_batch_size


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_parallel_microbatch_fails_before_sampling(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        wave_sizes(12, 6, micro_batch_size=invalid)


def test_six_gpu_throughput_preset_and_smoke_tail_use_distinct_stage1_batch_modes():
    repository = Path(__file__).resolve().parents[1]
    formal = load_run_config(repository / "configs/t2mem/t2mem_6gpu.yaml")
    smoke = load_run_config(repository / "configs/t2mem/t2mem_6gpu_smoke.yaml")
    assert formal.gpu_id == smoke.gpu_id == 1
    assert formal.num_devices == smoke.num_devices == 6
    assert formal.training.micro_batch_size == smoke.training.micro_batch_size == 1
    assert formal.training.stage1_per_device_batch_size == 2
    assert formal.training.stage1_accumulation_steps == 1
    assert formal.per_device_batch("stage1") == 2
    assert formal.effective_batch("stage1") == 12
    assert formal.per_device_batch("stage2a") == formal.per_device_batch("stage2b") == 1
    assert formal.training.stage2_accumulation_steps == 36
    assert formal.effective_batch("stage2a") == formal.effective_batch("stage2b") == 36
    assert formal.training.stage1_updates == 20000
    assert formal.training.save_interval == smoke.training.save_interval == 5000
    assert formal.training.keep_last_checkpoints == smoke.training.keep_last_checkpoints == 2
    assert "truncated_observation_prefix" not in formal.approximation_labels()
    assert "detached_fast_state" not in formal.approximation_labels()
    assert smoke.training.stage1_per_device_batch_size is None
    assert smoke.training.stage1_accumulation_steps == smoke.training.stage2_accumulation_steps == 8
    assert smoke.effective_batch("stage1") == smoke.effective_batch("stage2a") == 8
    assert smoke.training.stage1_updates == smoke.training.alternating_cycles == 1
    assert smoke.data.max_sequence_length == 32 and smoke.data.truncate_prefix
    assert "truncated_observation_prefix" in smoke.approximation_labels()


@pytest.mark.parametrize("num_devices", [1, 6])
def test_legacy_metadata_without_stage1_per_device_batch_keeps_64_and_32(num_devices):
    saved = RunConfig(num_devices=num_devices).to_dict()
    saved["training"].pop("stage1_per_device_batch_size", None)
    restored = config_from_dict(saved)
    assert restored.training.stage1_per_device_batch_size is None
    assert restored.per_device_batch("stage1") == 1
    assert restored.effective_batch("stage1") == 64
    assert restored.effective_batch("stage2a") == restored.effective_batch("stage2b") == 32


def test_new_stage1_accumulation_counts_per_device_waves_without_changing_stage2():
    config = RunConfig(num_devices=6)
    config.training.stage1_per_device_batch_size = 2
    config.training.stage1_accumulation_steps = 3
    config.validate()
    assert config.per_device_batch("stage1") == 2
    assert config.effective_batch("stage1") == 36
    assert config.per_device_batch("stage2a") == 1
    assert config.effective_batch("stage2a") == 32


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_stage1_per_device_batch_rejects_nonpositive_or_noninteger_values(invalid):
    config = RunConfig(num_devices=6)
    config.training.stage1_per_device_batch_size = invalid
    with pytest.raises(ValueError, match="stage1_per_device_batch_size"):
        config.validate()


@pytest.fixture(scope="module")
def cli_helpers():
    script = Path(__file__).resolve().parents[1] / "scripts/t2mem/train.py"
    return runpy.run_path(str(script))


def test_resume_cli_parses_device_and_save_overrides_from_process_arguments(cli_helpers, monkeypatch):
    source = "runs/ckpts/saved_single_device/5000"
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--stage", "resume", "--resume", source,
                      "--num-devices", "6", "--gpu-id", "1", "--save-interval", "5000"],
    )
    args = cli_helpers["parse_args"]()
    assert args.stage == "resume" and args.resume == source
    assert args.num_devices == 6 and args.gpu_id == 1 and args.save_interval == 5000
    assert args.updates is None and args.cycles is None  # Migration does not reset the saved budget.


def test_cli_defaults_preserve_loaded_configuration_and_later_explicit_flags_win(cli_helpers):
    parse = cli_helpers["parse_args"]
    defaults = parse(["--stage", "stage1"])
    assert defaults.num_devices is None and defaults.gpu_id is None and defaults.save_interval is None
    args = parse([
        "--num-devices", "6", "--gpu-id", "1", "--save-interval", "5000",
        "--stage", "resume", "--resume", "runs/ckpts/source/5000",
        "--num-devices", "2", "--gpu-id", "3", "--save-interval", "1000",
    ])
    assert args.num_devices == 2 and args.gpu_id == 3 and args.save_interval == 1000


def test_stage1_throughput_and_benchmark_cli_flags_have_explicit_positive_values(cli_helpers):
    parse = cli_helpers["parse_args"]
    defaults = parse(["--stage", "stage1"])
    assert defaults.stage1_per_device_batch is None
    assert defaults.stage1_accumulation_steps is None
    assert defaults.benchmark_steps is None
    assert defaults.benchmark_warmup_steps == 2
    args = parse([
        "--stage", "stage1", "--stage1-per-device-batch", "4", "--stage1-accumulation-steps", "3",
        "--benchmark-steps", "5", "--benchmark-warmup-steps", "2",
    ])
    assert args.stage1_per_device_batch == 4 and args.stage1_accumulation_steps == 3
    assert args.benchmark_steps == 5 and args.benchmark_warmup_steps == 2


def test_resume_explicitly_selects_stage1_throughput_without_resetting_saved_update_budget(cli_helpers):
    args = cli_helpers["parse_args"]([
        "--stage", "resume", "--resume", "runs/ckpts/saved_single_device/5000",
        "--num-devices", "6", "--stage1-per-device-batch", "8", "--stage1-accumulation-steps", "1",
    ])
    assert args.stage1_per_device_batch == 8 and args.stage1_accumulation_steps == 1
    assert args.num_devices == 6 and args.updates is None and args.cycles is None
    assert args.benchmark_steps is None


@pytest.mark.parametrize(
    "flag", ["--stage1-per-device-batch", "--stage1-accumulation-steps", "--benchmark-steps", "--benchmark-warmup-steps"],
)
@pytest.mark.parametrize("invalid", ["0", "-1", "1.5"])
def test_stage1_throughput_cli_rejects_invalid_counts_before_loading_a_model(cli_helpers, flag, invalid):
    with pytest.raises(SystemExit) as error:
        cli_helpers["parse_args"](["--stage", "stage1", flag, invalid])
    assert error.value.code == 2


@pytest.fixture(scope="module")
def tuner_helpers():
    script = Path(__file__).resolve().parents[1] / "scripts/t2mem/tune_stage1_batch.py"
    return runpy.run_path(str(script))


def test_stage1_tuner_selects_measured_throughput_instead_of_largest_batch(tuner_helpers):
    reports = [
        {"per_device_batch": 1, "samples_per_second": float("nan")},
        {"per_device_batch": 2, "samples_per_second": 0.0},
        {"per_device_batch": 4, "samples_per_second": 72.0},
        {"per_device_batch": 8, "samples_per_second": 65.0},
        {"per_device_batch": 16, "samples_per_second": float("inf")},
    ]
    selected = tuner_helpers["select_fastest"](reports)
    assert selected is reports[2]
    assert selected["per_device_batch"] == 4


@pytest.mark.parametrize("rates", [[], [0.0, -1.0, float("nan"), float("inf")]])
def test_stage1_tuner_refuses_a_config_without_successful_finite_measurements(tuner_helpers, rates):
    reports = [{"per_device_batch": index + 1, "samples_per_second": rate} for index, rate in enumerate(rates)]
    with pytest.raises(ValueError, match="No successful batch measurements"):
        tuner_helpers["select_fastest"](reports)


@pytest.mark.parametrize(
    "log", ["RESOURCE_EXHAUSTED: HBM allocation", "CUDA_ERROR_OUT_OF_MEMORY", "failed to allocate 123 bytes",
            "unable to allocate device memory", "GPU out of memory"],
)
def test_stage1_tuner_recognizes_actual_memory_capacity_failures(tuner_helpers, log):
    assert tuner_helpers["is_memory_failure"](log)


@pytest.mark.parametrize("log", ["FileNotFoundError: no training manifest", "undefined symbol in libcuda.so", "KeyboardInterrupt"])
def test_stage1_tuner_does_not_mistake_setup_or_interrupt_failures_for_batch_capacity(tuner_helpers, log):
    assert not tuner_helpers["is_memory_failure"](log)


@pytest.mark.parametrize("invalid", ["0", "-1"])
def test_resume_cli_rejects_invalid_device_count_before_loading_a_model(cli_helpers, invalid):
    with pytest.raises(SystemExit) as error:
        cli_helpers["parse_args"](["--stage", "resume", "--resume", "unused", "--num-devices", invalid])
    assert error.value.code == 2


def test_negative_physical_gpu_override_is_rejected_by_configuration(cli_helpers):
    args = cli_helpers["parse_args"](["--stage", "resume", "--resume", "unused", "--gpu-id", "-1"])
    config = RunConfig(gpu_id=args.gpu_id)
    with pytest.raises(ValueError, match="gpu_id.*nonnegative"):
        config.validate()


def test_save_interval_5000_ignores_phase_boundaries_and_still_saves_the_final_state(cli_helpers):
    should_save = cli_helpers["should_save_checkpoint"]
    for step in (0, 1, 500, 1000, 4999, 5500):
        assert not should_save(step, 5000, complete=False)
    for step in (5000, 10000, 20000):
        assert should_save(step, 5000, complete=False)
    assert should_save(1500, 5000, complete=True)
    assert should_save(4999, 5000, complete=True)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="save_interval"):
            should_save(1, invalid, complete=False)


def test_formal_stage2_requires_completed_formal_stage1_budget(cli_helpers):
    validate = cli_helpers["validate_stage1_initialization"]
    with pytest.raises(ValueError, match="Finish"):
        validate({"complete": False, "global_step": 20000}, 20000)
    with pytest.raises(ValueError, match="at least 20000"):
        validate({"complete": True, "global_step": 1}, 20000)
    with pytest.raises(ValueError, match="at least 20000"):
        validate({"complete": True, "global_step": 19999}, 20000)
    validate({"complete": True, "global_step": 20000}, 20000)
    validate({"complete": True, "global_step": 1}, 1)


def _sample(length, marker, *, episode="synthetic", demonstration=False):
    """Preserve chronology, demo observations, H2 targets and eight robot axes."""
    time = np.arange(length, dtype=np.float32)
    state = np.zeros((length, 32), np.float32)
    state[:, 0] = marker + 0.1 * time
    actions = np.zeros((length, 2, 32), np.float32)
    actions[..., :8] = (0.5 * marker + 0.03 * time)[:, None, None]
    action_mask = np.zeros(actions.shape, bool)
    action_mask[..., :8] = True
    if demonstration:
        action_mask[0] = False
    # The last chunk has only one future action. Chunk padding stays distinct
    # from sequence padding and never adds the 24 representation-only axes.
    action_mask[-1, 1] = False
    actions[-1, 1] = 0.0
    observation = {
        "image": {"base_0_rgb": np.full((length, 2, 2, 3), 127, np.uint8)},
        "image_mask": {"base_0_rgb": np.ones(length, bool)},
        "state": state,
        "tokenized_prompt": np.tile(np.asarray([1, 2], np.int32), (length, 1)),
        "tokenized_prompt_mask": np.ones((length, 2), bool),
        "tokenized_pi05_prompt": np.tile(np.asarray([3, 4], np.int32), (length, 1)),
        "tokenized_pi05_prompt_mask": np.ones((length, 2), bool),
    }
    return SequenceSample(
        observation=observation,
        actions=actions,
        action_mask=action_mask,
        write_mask=np.arange(length) % 2 == 0,
        observation_mask=np.ones(length, bool),
        physical_steps=np.arange(length, dtype=np.int32),
        task="PickXtimes",
        episode=episode,
    )


def test_collation_preserves_independent_histories_and_masks_dummy_slots():
    samples = [_sample(3, -0.7, demonstration=True), _sample(5, 1.2, demonstration=True)]
    batch, valid = collate_parallel_wave(samples, 6)
    np.testing.assert_array_equal(valid, [True, True, False, False, False, False])
    expected = collate_sequences(samples)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(batch), jax.tree.leaves(expected), strict=True):
        assert actual_leaf.shape[:3] == (6, 8, 1)
        for device in range(2):
            np.testing.assert_array_equal(actual_leaf[device, :, 0], expected_leaf[:, device])
    for name in ("action_mask", "write_mask", "observation_mask"):
        assert not batch[name][2:].any()
        assert not batch[name][:, 5:].any()
    assert not batch["action_mask"][..., 8:].any()
    assert not batch["action_mask"][:2, 0].any()  # Demo frames still enter the history.
    assert batch["observation_mask"][:2, 0].all()
    assert batch["write_mask"][:2, 0].all()
    # Dummy slots keep usable template images/tokens, preventing a malformed
    # all-masked attention input while all loss and write masks stay false.
    for device in range(2, 6):
        np.testing.assert_array_equal(batch["observation"]["state"][device], batch["observation"]["state"][0])
        np.testing.assert_array_equal(
            batch["observation"]["tokenized_prompt_mask"][device],
            batch["observation"]["tokenized_prompt_mask"][0],
        )
    np.testing.assert_array_equal(samples[0].physical_steps, [0, 1, 2])
    assert samples[0].observation_mask.all()  # Collation must not mutate the sampler result.


def test_stage1_microbatch_collation_masks_partial_device_without_mutating_real_frames():
    samples = [_sample(1, marker) for marker in (0.7, 1.1, 2.3)]
    for sample in samples:
        sample.write_mask[:] = False
    batch, real_counts = collate_parallel_wave(samples, 2, micro_batch_size=2)
    np.testing.assert_array_equal(real_counts, [2, 1])
    assert np.issubdtype(real_counts.dtype, np.integer)
    expected = collate_sequences(samples)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(batch), jax.tree.leaves(expected), strict=True):
        assert actual_leaf.shape[:3] == (2, 1, 2)
        for frame in range(3):
            np.testing.assert_array_equal(actual_leaf[frame // 2, :, frame % 2], expected_leaf[:, frame])
    for name in ("action_mask", "write_mask", "observation_mask"):
        assert not batch[name][1, :, 1].any()
    np.testing.assert_array_equal(
        batch["observation"]["tokenized_prompt_mask"][1, :, 1],
        batch["observation"]["tokenized_prompt_mask"][0, :, 0],
    )
    assert batch["observation_mask"].sum() == 3
    assert not batch["write_mask"].any()
    assert not batch["action_mask"][..., 8:].any()
    assert all(sample.observation_mask.all() and sample.action_mask.any() for sample in samples)


def test_pinned_partitionable_threefry_keeps_per_frame_keys_stable_under_padding():
    if not jax.config.jax_threefry_partitionable:
        pytest.skip("Prefix stability applies to the pinned JAX default, not the optional legacy RNG override")
    key = jax.random.PRNGKey(7)
    short, long = jax.random.split(key, 4), jax.random.split(key, 8)
    np.testing.assert_array_equal(jax.device_get(short), jax.device_get(long[:4]))


def _mesh(num_devices):
    devices = jax.local_devices()
    if len(devices) < num_devices:
        pytest.skip(f"Needs {num_devices} visible devices; found {len(devices)}")
    return Mesh(np.asarray(devices[:num_devices]), ("data",))


def _tree_allclose(left, right, *, rtol=1e-5, atol=1e-6):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for original, actual in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert original.shape == actual.shape
        np.testing.assert_allclose(jax.device_get(actual), jax.device_get(original), rtol=rtol, atol=atol)


def _episode_loss(params, all_state, rng, batch):
    del all_state, rng
    valid = batch["observation_mask"] & batch["action_mask"].any(axis=(-2, -1))
    target = batch["actions"][..., 0, 0]
    residual = jnp.where(valid, params["weight"] - target, 0.0)
    count = jnp.sum(valid)
    loss = jnp.sum(residual**2) / jnp.maximum(count, 1)
    return loss, {"supervised_chunks": count}


def _stage1_frame_loss(params, all_state, rng, batch):
    """Normalize each frame by its own valid H/action axes before the local mean."""
    del all_state, rng
    mask = batch["action_mask"] & batch["observation_mask"][..., None, None]
    counts = jnp.sum(mask, axis=(-2, -1))
    labeled = counts > 0
    predicted = jnp.where(mask, params["weight"], 0.0)
    target = jnp.where(mask, batch["actions"], 0.0)
    per_frame = jnp.sum((predicted - target)**2, axis=(-2, -1)) / jnp.maximum(counts, 1)
    real = jnp.sum(labeled)
    loss = jnp.sum(per_frame) / jnp.maximum(real, 1)
    return loss, {"supervised_chunks": (real > 0).astype(jnp.float32)}


def test_stage1_microbatch_tail_uses_real_frame_weights_and_per_frame_action_normalization():
    mesh = _mesh(2)
    targets = np.asarray([0.0, 10.0, -3.0, 4.0, 21.0, -7.0, 1.0], np.float32)
    params = {"weight": jnp.asarray(0.25, jnp.float32)}
    gradient = make_parallel_grad(_stage1_frame_loss, mesh, micro_batch_size=2)
    accumulate = make_parallel_accumulator(mesh)
    reduce = make_gradient_reducer(mesh)
    running, offset = None, 0
    sizes = wave_sizes(len(targets), 2, micro_batch_size=2)
    assert sizes == (4, 3)
    for count in sizes:
        samples = [_sample(1, float(target)) for target in targets[offset:offset + count]]
        for index, sample in enumerate(samples):
            sample.actions[:] = targets[offset + index]
            sample.write_mask[:] = False
            # Different real frames have 16, 8 or 4 supervised action elements.
            sample.action_mask[:] = False
            sample.action_mask[0, :2 if (offset + index) % 3 == 0 else 1,
                               :4 if (offset + index) % 3 == 2 else 8] = True
        batch, real_counts = collate_parallel_wave(samples, 2, micro_batch_size=2)
        if count == 3:
            np.testing.assert_array_equal(real_counts, [2, 1])
            # A copied dummy target must never affect the local mean or gradient.
            batch["actions"][1, :, 1] = 1e6
        rngs = jnp.stack([jax.random.PRNGKey(offset + device) for device in range(2)])
        (loss, stats), grads = gradient(params, {}, rngs, batch, real_counts)
        wave = {"loss": loss, "stats": stats, "gradients": grads}
        running = wave if running is None else accumulate(running, wave)
        offset += count
    actual = reduce(running, len(targets))
    expected_gradient = 2 * (0.25 - targets.mean())
    expected_loss = np.mean((0.25 - targets)**2)
    np.testing.assert_allclose(jax.device_get(actual["gradients"]["weight"]), expected_gradient,
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jax.device_get(actual["loss"]), expected_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jax.device_get(actual["stats"]["supervised_chunks"]), 1.0, rtol=0, atol=0)
    assert offset == len(targets)
    # Means of device means would give the single-frame last device two votes.
    incorrect_mean = (2 * (0.25 - targets[:2].mean()) + 2 * (0.25 - targets[2:4].mean())
                      + 2 * (0.25 - targets[4:6].mean()) + 2 * (0.25 - targets[6])) / 4
    assert not np.isclose(expected_gradient, incorrect_mean)


@pytest.mark.parametrize("num_devices", [2, 6])
def test_stage2_reduction_weights_short_and_long_episodes_equally(num_devices):
    mesh = _mesh(num_devices)
    short, long = _sample(1, 0.0), _sample(5, 20.0)
    short.actions[..., 0, 0] = 0.0
    long.actions[..., 0, 0] = 10.0
    batch, valid = collate_parallel_wave([short, long], num_devices)
    params = {"weight": jnp.asarray(0.2, jnp.float32)}
    rngs = jnp.stack([jax.random.PRNGKey(i) for i in range(num_devices)])
    (_, stats), per_device = make_parallel_grad(_episode_loss, mesh)(params, {}, rngs, batch, valid)
    reduced = make_gradient_reducer(mesh)(per_device, 2)
    # Mean episode gradient: (2*(.2-0) + 2*(.2-10))/2. Averaging all
    # frames instead would incorrectly give the longer episode five votes.
    np.testing.assert_allclose(jax.device_get(reduced["weight"]), -9.6, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(jax.device_get(stats["supervised_chunks"])[:2], [1, 5])
    assert np.asarray(jax.device_get(per_device["weight"]))[2:].sum() == 0
    assert not np.isclose(-9.6, (2 * 0.2 + 5 * 2 * (0.2 - 10)) / 6)


@pytest.mark.parametrize("num_devices", [2, 6])
def test_all_waves_are_summed_before_dividing_by_real_sample_count(num_devices):
    mesh = _mesh(num_devices)
    global_batch = 7
    params = {"weight": jnp.asarray(0.2, jnp.float32)}
    gradient = make_parallel_grad(_episode_loss, mesh)
    accumulate = make_parallel_accumulator(mesh)
    reduce = make_gradient_reducer(mesh)
    running = None
    offset = 0
    targets = np.arange(global_batch, dtype=np.float32)
    for count in wave_sizes(global_batch, num_devices):
        samples = [_sample(1, float(2 * target)) for target in targets[offset:offset + count]]
        batch, valid = collate_parallel_wave(samples, num_devices)
        rngs = jnp.stack([jax.random.PRNGKey(offset + slot) for slot in range(num_devices)])
        (_, _), grads = gradient(params, {}, rngs, batch, valid)
        running = grads if running is None else accumulate(running, grads)
        offset += count
    actual = reduce(running, global_batch)
    expected = 2 * (0.2 - targets.mean())
    np.testing.assert_allclose(jax.device_get(actual["weight"]), expected, rtol=1e-6, atol=1e-6)
    assert offset == global_batch
    assert actual["weight"].shape == ()  # Reduction removes the device axis.


@pytest.mark.parametrize("num_devices", [2, 6])
def test_inactive_slot_nan_values_and_gradients_cannot_contaminate_reduction(num_devices):
    mesh = _mesh(num_devices)
    batch, valid = collate_parallel_wave([_sample(1, 0.7)], num_devices)
    batch["observation"]["state"][1:] = np.nan

    def loss_fn(params, all_state, rng, local_batch):
        del all_state, rng
        value = local_batch["observation"]["state"][0, 0, 0]
        return params["weight"] * value, {"value": value}

    params = {"weight": jnp.asarray(0.2, jnp.float32)}
    rngs = jnp.stack([jax.random.PRNGKey(i) for i in range(num_devices)])
    (loss, stats), grads = make_parallel_grad(loss_fn, mesh)(params, {}, rngs, batch, valid)
    for values in (loss, stats["value"], grads["weight"]):
        actual = np.asarray(jax.device_get(values))
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual[1:], 0.0)
    reduced = make_gradient_reducer(mesh)(grads, 1)
    np.testing.assert_allclose(jax.device_get(reduced["weight"]), 0.7, rtol=1e-6, atol=1e-6)


def _meta_sequence_loss(params, all_state, rng, batch, *, detach_inner=False, detach_fast=False):
    """Late reads depend on a genuine inner gradient and writes in prior blocks."""
    del all_state, rng
    feature = batch["observation"]["state"][..., 0]
    target = batch["actions"][..., 0, 0]
    supervised = batch["action_mask"].any(axis=(-2, -1))

    def step(fast, inputs):
        x, y, write, observed, labeled = inputs
        # Reads precede this frame's write. Differentiating the later loss
        # through this gradient needs the inner loss's Hessian with respect
        # to fast weights; stopping only the gradient changes the meta-gradient.
        read = fast * x
        inner_grad = jax.grad(lambda weight: 0.5 * jnp.sum((weight * x - (x + 0.2))**2))(fast)
        if detach_inner:
            inner_grad = jax.lax.stop_gradient(inner_grad)
        updated = fast - 0.2 * inner_grad
        if detach_fast:
            updated = jax.lax.stop_gradient(updated)
        next_fast = jnp.where(write & observed, updated, fast)
        valid = labeled & observed
        residual = jnp.where(valid, read - y, 0.0)
        return next_fast, (jnp.sum(residual**2), jnp.sum(valid))

    _, (losses, counts) = differentiable_scan(
        step, params["theta"][None],
        (feature, target, batch["write_mask"], batch["observation_mask"], supervised),
        remat=True, block_size=2,
    )
    count = jnp.sum(counts)
    return jnp.sum(losses) / jnp.maximum(count, 1), {"supervised_chunks": count}


@pytest.mark.parametrize("num_devices", [2, 6])
def test_shard_map_preserves_second_order_meta_gradients_through_earlier_writes(num_devices):
    mesh = _mesh(num_devices)
    samples = [_sample(3, 0.7), _sample(5, 1.2)]
    for sample in samples:
        sample.action_mask[:] = False
        sample.action_mask[-1, 0, 0] = True  # Only the late read is supervised.
    batch, valid = collate_parallel_wave(samples, num_devices)
    assert batch["actions"].shape[1] == 8  # Unequal real lengths share a temporal bucket.
    params = {"theta": jnp.asarray(0.3, jnp.float32)}
    rngs = jnp.stack([jax.random.PRNGKey(i) for i in range(num_devices)])
    (losses, _), grads = make_parallel_grad(_meta_sequence_loss, mesh)(params, {}, rngs, batch, valid)
    reduced = make_gradient_reducer(mesh)({"loss": losses, "gradient": grads}, 2)

    def serial_loss(theta, **options):
        losses = [
            _meta_sequence_loss({"theta": theta}, {}, rngs[index], collate_sequences([sample]), **options)[0]
            for index, sample in enumerate(samples)
        ]
        return sum(losses) / len(losses)

    expected_loss, expected_grad = jax.value_and_grad(serial_loss)(params["theta"])
    np.testing.assert_allclose(jax.device_get(reduced["loss"]), jax.device_get(expected_loss), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        jax.device_get(reduced["gradient"]["theta"]), jax.device_get(expected_grad), rtol=1e-5, atol=1e-6,
    )
    assert abs(float(jax.device_get(expected_grad))) > 1e-3
    epsilon = 1e-3
    finite_difference = (
        serial_loss(params["theta"] + epsilon) - serial_loss(params["theta"] - epsilon)
    ) / (2 * epsilon)
    np.testing.assert_allclose(jax.device_get(expected_grad), jax.device_get(finite_difference), rtol=2e-3, atol=2e-4)
    detached_inner = jax.grad(lambda theta: serial_loss(theta, detach_inner=True))(params["theta"])
    assert abs(float(jax.device_get(detached_inner - expected_grad))) > 1e-3
    detached_state = jax.grad(lambda theta: serial_loss(theta, detach_fast=True))(params["theta"])
    np.testing.assert_allclose(jax.device_get(detached_state), 0.0, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(jax.device_get(grads["theta"]))[2:], 0.0)


class TinyParallelModel(nnx.Module):
    """RNG-sensitive objective with real policy, memory, gate and frozen groups."""

    def __init__(self):
        self.PaliGemma = nnx.Dict(
            llm=nnx.Dict(
                layers=nnx.Dict(
                    memory=nnx.Dict(
                        w0=nnx.Param(jnp.asarray([0.2, -0.1], jnp.float32)),
                        alpha=nnx.Param(jnp.asarray(0.3, jnp.float32)),
                    ),
                    expert_1=nnx.Param(jnp.asarray([0.4, -0.2, 0.1], jnp.float32)),
                    lora=nnx.Param(jnp.asarray(0.5, jnp.float32)),
                    base=nnx.Param(jnp.asarray(0.6, jnp.float32)),
                ),
                memory_interface_tokens=nnx.Param(jnp.asarray(0.1, jnp.float32)),
            ),
            img=nnx.Dict(kernel=nnx.Param(jnp.asarray(0.7, jnp.float32))),
        )
        self.state_proj = nnx.Dict(kernel=nnx.Param(jnp.asarray(0.2, jnp.float32)))
        self.action_out_proj = nnx.Dict(kernel=nnx.Param(jnp.asarray(-0.1, jnp.float32)))
        self.auxiliary = nnx.Variable(jnp.asarray([0.8, 0.9], jnp.float32))
        self.deterministic = True

    def compute_sequence_loss(self, rng, observation, actions, action_mask, write_mask, observation_mask,
                              *, train, memory_enabled):
        del train
        llm = self.PaliGemma.llm
        layer = llm.layers
        feature = observation.state[..., 0]
        expert = layer.expert_1.value
        memory = layer.memory.w0.value
        prediction = (
            expert[0] + expert[1] * feature + expert[2] * feature**2
            + layer.lora.value * jnp.sin(feature)
            + layer.base.value + self.PaliGemma.img.kernel.value + self.auxiliary.value[0]
            + self.action_out_proj.kernel.value * feature**3
            + (self.state_proj.kernel.value * feature if memory_enabled else 0.0)
            + (memory[0] * jnp.cos(feature) + memory[1] * feature + layer.memory.alpha.value * feature**2
               + llm.memory_interface_tokens.value if memory_enabled else 0.0)
            + 0.2 * jax.random.uniform(rng, (), dtype=jnp.float32)
        )
        valid = observation_mask & action_mask.any(axis=(-2, -1))
        residual = jnp.where(valid, prediction - actions[..., 0, 0], 0.0)
        count = jnp.sum(valid)
        loss = jnp.sum(residual**2) / jnp.maximum(count, 1)
        return loss, {"action_loss": loss, "supervised_chunks": count,
                      "scheduled_writes": jnp.sum(write_mask & observation_mask) * memory_enabled}


class TinyStage1BatchedModel(TinyParallelModel):
    """One-frame objective with stochastic targets that survive regrouping.

    Frame-keyed noise keeps the same frame's objective identical in a serial
    batch and smaller device batches. Trainer PRNG consumption is checked
    separately, because throughput mode deliberately changes microbatch keys.
    """

    def compute_sequence_loss(self, rng, observation, actions, action_mask, write_mask, observation_mask,
                              *, train, memory_enabled):
        del rng, train
        assert not memory_enabled
        assert actions.shape[0] == 1
        feature = observation.state[..., 0]
        frame_ids = (jnp.abs(feature[0]) * 100_000).astype(jnp.uint32)
        frame_keys = jax.vmap(jax.random.PRNGKey)(frame_ids)
        noise = jax.vmap(lambda key: jax.random.uniform(key, (), dtype=jnp.float32))(frame_keys)[None]
        layer = self.PaliGemma.llm.layers
        expert = layer.expert_1.value
        prediction = (expert[0] + expert[1] * feature + expert[2] * feature**2
                      + layer.lora.value * noise + self.action_out_proj.kernel.value * feature**3
                      + layer.base.value + self.PaliGemma.img.kernel.value + self.auxiliary.value[0])
        mask = action_mask & observation_mask[..., None, None]
        counts = jnp.sum(mask, axis=(-2, -1))
        residual = (jnp.where(mask, prediction[..., None, None], 0.0)
                    - jnp.where(mask, actions, 0.0))
        per_frame = jnp.sum(residual**2, axis=(-2, -1)) / jnp.maximum(counts, 1)
        supervised = jnp.sum(counts > 0)
        loss = jnp.sum(per_frame) / jnp.maximum(supervised, 1)
        return loss, {"action_loss": loss, "supervised_chunks": supervised,
                      "scheduled_writes": jnp.sum(write_mask & observation_mask)}


class SyntheticSampler:
    """Host RNG and counters expose dummy-slot consumption or ordering errors."""

    def __init__(self, seed=17):
        self.rng = np.random.default_rng(seed)
        self.task_updates = Counter()
        self.task_samples = Counter()

    def next_task(self):
        self.rng.integers(0, 1000)  # A resumable host draw even with one task.
        self.task_updates["PickXtimes"] += 1
        return "PickXtimes"

    def sample(self, task, *, stage1):
        assert task == "PickXtimes"
        marker = float(self.rng.normal())
        length = 1 if stage1 else int(self.rng.choice([1, 3, 5]))
        self.task_samples[task] += 1
        sample = _sample(length, marker, episode=str(self.task_samples[task]), demonstration=length > 1)
        if stage1:
            sample.write_mask[:] = False
        return sample

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "task_updates": dict(self.task_updates),
                "task_samples": dict(self.task_samples)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.task_updates = Counter(state["task_updates"])
        self.task_samples = Counter(state["task_samples"])


class Stage1FrameSampler(SyntheticSampler):
    """Random real frames with unequal valid action horizons and no writes."""

    def sample(self, task, *, stage1):
        assert stage1
        sample = super().sample(task, stage1=True)
        index = int(sample.episode)
        sample.action_mask[:] = False
        sample.action_mask[0, :2 if index % 3 == 0 else 1, :4 if index % 3 == 2 else 8] = True
        return sample


def _stage1_batched_trainer(tmp_path, num_devices, per_device_batch, accumulation=2):
    config = RunConfig(num_devices=num_devices,
                       checkpoint_dir=str(tmp_path / f"stage1_batch_{num_devices}_{per_device_batch}"))
    config.training.stage1_per_device_batch_size = per_device_batch
    config.training.stage1_accumulation_steps = accumulation
    config.training.stage1_updates = 2
    config.training.save_interval = 5000
    config.training.outer_gradient_clip = 1.0
    assets = tmp_path / "stage1_batch_assets"
    assets.mkdir(exist_ok=True)
    (assets / "norm_stats.json").write_text("{}")
    config.data.norm_stats_dir = str(assets)
    config.validate()
    return Trainer(TinyStage1BatchedModel(), config, PhaseProgress.start("stage1"), Stage1FrameSampler())


@pytest.mark.parametrize("num_devices", [2, 6])
def test_stage1_batched_device_updates_match_same_global_serial_batch_and_restore(tmp_path, num_devices):
    _mesh(num_devices)
    accumulation, per_device_batch = 2, 2
    serial = _stage1_batched_trainer(tmp_path, 1, num_devices * per_device_batch, accumulation)
    parallel = _stage1_batched_trainer(tmp_path, num_devices, per_device_batch, accumulation)
    global_batch = num_devices * per_device_batch * accumulation
    assert serial.config.effective_batch("stage1") == parallel.config.effective_batch("stage1") == global_batch
    shapes = [leaf.shape for leaf in jax.tree.leaves(serial.state)]
    expected_serial_rng, expected_parallel_rng = serial.rng, parallel.rng
    restored = None
    for update in (1, 2):
        expected = serial.step()
        actual = parallel.step()
        for _ in range(accumulation):
            expected_serial_rng, _ = jax.random.split(expected_serial_rng)
        for _ in range(num_devices * accumulation):
            expected_parallel_rng, _ = jax.random.split(expected_parallel_rng)
        for name in ("step", "stage", "effective_batch", "parallel_waves", "sequence_length_max",
                     "supervised_frames", "scheduled_memory_writes", "task_updates", "task_samples"):
            assert actual[name] == expected[name]
        for name in ("loss", "gradient_norm", "action_loss", "supervised_chunks", "scheduled_writes"):
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6)
        assert actual["per_device_batch_size"] == 2
        assert expected["per_device_batch_size"] == num_devices * 2
        assert actual["parallel_waves"] == accumulation
        assert actual["supervised_frames"] == global_batch
        assert actual["supervised_chunks"] == 1.0
        assert actual["scheduled_memory_writes"] == actual["scheduled_writes"] == 0
        assert actual["gradient_norm"] > 1.0  # Exercises clipping after the global mean.
        assert actual["task_samples"] == {"PickXtimes": global_batch * update}
        assert np.isfinite(actual["step_seconds"]) and actual["step_seconds"] > 0
        assert np.isfinite(actual["samples_per_second"]) and actual["samples_per_second"] > 0
        np.testing.assert_array_equal(jax.device_get(serial.rng), jax.device_get(expected_serial_rng))
        np.testing.assert_array_equal(jax.device_get(parallel.rng), jax.device_get(expected_parallel_rng))
        assert parallel.sampler.state_dict() == serial.sampler.state_dict()
        _tree_allclose(serial.state, parallel.state)
        _tree_allclose(serial.optimizer_states, parallel.optimizer_states)
        assert [leaf.shape for leaf in jax.tree.leaves(parallel.state)] == shapes
        if update == 1:
            checkpoint = parallel.save()
            restored = _stage1_batched_trainer(tmp_path, num_devices, per_device_batch, accumulation)
            restored.restore(checkpoint)
            _tree_allclose(parallel.state, restored.state, rtol=0, atol=0)
            _tree_allclose(parallel.optimizer_states, restored.optimizer_states, rtol=0, atol=0)
            np.testing.assert_array_equal(jax.device_get(parallel.rng), jax.device_get(restored.rng))
            assert restored.sampler.state_dict() == parallel.sampler.state_dict()
        else:
            resumed = restored.step()
            np.testing.assert_allclose(resumed["loss"], actual["loss"], rtol=1e-5, atol=1e-6)
            _tree_allclose(parallel.state, restored.state)
            _tree_allclose(parallel.optimizer_states, restored.optimizer_states)
            np.testing.assert_array_equal(jax.device_get(parallel.rng), jax.device_get(restored.rng))
            assert restored.sampler.state_dict() == parallel.sampler.state_dict()
            assert asdict(restored.progress) == asdict(parallel.progress)
    assert parallel.progress.complete
    assert set(parallel.optimizer_states) == {"stage1"}


def _trainer(tmp_path, num_devices, mode, global_batch):
    config = RunConfig(num_devices=num_devices, checkpoint_dir=str(tmp_path / f"{mode}_{num_devices}"))
    config.training.micro_batch_size = 1
    config.training.stage1_accumulation_steps = global_batch
    config.training.stage2_accumulation_steps = global_batch
    config.training.stage1_updates = 1
    config.training.memory_updates_per_cycle = 1
    config.training.policy_updates_per_cycle = 1
    config.training.alternating_cycles = 1
    config.training.save_interval = 5000
    config.training.outer_gradient_clip = 1.0
    config.data.norm_stats_dir = str(tmp_path / "assets")
    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "norm_stats.json").write_text("{}")
    config.validate()
    return Trainer(TinyParallelModel(), config, PhaseProgress.start(mode), SyntheticSampler())


@pytest.mark.parametrize("num_devices", [2, 6])
def test_same_root_resume_keeps_exact_norm_assets_after_source_checkpoint_is_pruned(tmp_path, num_devices):
    _mesh(num_devices)
    source = _trainer(tmp_path, 1, "stage1", 7)
    source.config.checkpoint_dir = str(tmp_path / "same_root_resume")
    source.config.training.stage1_updates = 4
    source.config.training.keep_last_checkpoints = 1
    preparation = Path(source.config.data.norm_stats_dir)
    norm_bytes = (
        b'{\n  "norm_stats": {\n'
        b'    "state": {"mean": [0.125], "std": [1.5]},\n'
        b'    "actions": {"mean": [0.25], "std": [0.75]}\n'
        b'  }\n}\n'
    )
    (preparation / "norm_stats.json").write_bytes(norm_bytes)
    source.step()
    original = source.save()
    original_assets = original / "assets/robomme"
    assert (original_assets / "norm_stats.json").read_bytes() == norm_bytes

    # Match main's resume routing: the saved snapshot supplies normalization,
    # while the new run writes into that same checkpoint root with retention.
    config = copy.deepcopy(source.config)
    config.num_devices = num_devices
    # A used Trainer must refresh its immutable bytes when selecting another
    # snapshot; warm an intentionally different cache through the public save.
    stale_preparation = tmp_path / "stale_norm_assets"
    stale_preparation.mkdir()
    stale_bytes = b'{"norm_stats": {}}\n'
    (stale_preparation / "norm_stats.json").write_bytes(stale_bytes)
    config.checkpoint_dir = str(tmp_path / "cache_warmup_only")
    config.data.norm_stats_dir = str(stale_preparation)
    resumed = Trainer(TinyParallelModel(), config, PhaseProgress.start("stage1"), SyntheticSampler())
    warmup = resumed.save()
    assert (warmup / "assets/robomme/norm_stats.json").read_bytes() == stale_bytes
    assert stale_bytes != norm_bytes
    config.checkpoint_dir = source.config.checkpoint_dir
    config.data.norm_stats_dir = str(original_assets)
    resumed.restore(original)
    shutil.rmtree(preparation)
    shutil.rmtree(stale_preparation)
    assert not preparation.exists()
    expected_hash = hashlib.sha256(norm_bytes).hexdigest()
    previous = original
    for step in (2, 3):
        assert resumed.step()["step"] == step
        saved = resumed.save()
        assert saved.parent == original.parent
        assert not previous.exists()  # keep1 really removes the source, not just other runs.
        assert not original_assets.exists()
        actual = (saved / "assets/robomme/norm_stats.json").read_bytes()
        assert actual == norm_bytes
        assert hashlib.sha256(actual).hexdigest() == expected_hash
        previous = saved
    assert resumed.sampler.task_samples == {"PickXtimes": 21}
    assert resumed.progress.global_step == 3


@pytest.mark.parametrize("num_devices", [2, 6])
@pytest.mark.parametrize(("mode", "global_batch", "num_steps"), [("stage1", 64, 1), ("alternating", 32, 2)])
def test_parallel_updates_match_serial_and_checkpoint_restores_for_single_device(
    tmp_path, num_devices, mode, global_batch, num_steps,
):
    _mesh(num_devices)
    serial = _trainer(tmp_path, 1, mode, global_batch)
    parallel = _trainer(tmp_path, num_devices, mode, global_batch)
    shapes = [leaf.shape for leaf in jax.tree.leaves(serial.state)]
    expected_rng = serial.rng
    for update in range(num_steps):
        expected_metrics = serial.step()
        actual_metrics = parallel.step()
        for _ in range(global_batch):
            expected_rng, _ = jax.random.split(expected_rng)
        for name in ("step", "stage", "cycle", "phase_step", "effective_batch", "sequence_length_max",
                     "supervised_frames", "scheduled_memory_writes", "task_updates", "task_samples"):
            assert actual_metrics[name] == expected_metrics[name]
        for name in ("loss", "gradient_norm", "action_loss", "supervised_chunks", "scheduled_writes"):
            np.testing.assert_allclose(actual_metrics[name], expected_metrics[name], rtol=1e-5, atol=1e-6)
        assert actual_metrics["effective_batch"] == global_batch
        if mode == "stage1":
            assert actual_metrics["scheduled_memory_writes"] == 0
        assert actual_metrics["parallel_waves"] == len(wave_sizes(global_batch, num_devices))
        assert parallel.rng.shape == (2,)
        assert actual_metrics["task_samples"] == {"PickXtimes": global_batch * (update + 1)}
        np.testing.assert_array_equal(jax.device_get(parallel.rng), jax.device_get(expected_rng))
        assert parallel.sampler.state_dict() == serial.sampler.state_dict()
        _tree_allclose(serial.state, parallel.state)
        _tree_allclose(serial.optimizer_states, parallel.optimizer_states)
        assert [leaf.shape for leaf in jax.tree.leaves(parallel.state)] == shapes
    expected_stages = {"stage1"} if mode == "stage1" else {"stage2a", "stage2b"}
    assert set(parallel.optimizer_states) == expected_stages
    assert asdict(parallel.progress) == asdict(serial.progress)
    # Make the saved non-parameter value differ from initialization, so silently
    # omitting model_variables restoration cannot pass a mere shape comparison.
    for trainer in (serial, parallel):
        saved_model = nnx.merge(trainer.graphdef, trainer.state)
        saved_model.auxiliary.value = jax.device_put(jnp.asarray([1.3, 1.4], jnp.float32), trainer.replica_sharding)
        trainer.state = nnx.state(saved_model)
    single_checkpoint = serial.save()
    migrated_parallel = _trainer(tmp_path, num_devices, mode, global_batch)
    migrated_parallel.restore(single_checkpoint)
    _tree_allclose(serial.state, migrated_parallel.state, rtol=0, atol=0)
    _tree_allclose(serial.optimizer_states, migrated_parallel.optimizer_states, rtol=0, atol=0)
    np.testing.assert_array_equal(jax.device_get(serial.rng), jax.device_get(migrated_parallel.rng))
    assert migrated_parallel.sampler.state_dict() == serial.sampler.state_dict()
    assert asdict(migrated_parallel.progress) == asdict(serial.progress)
    assert [leaf.shape for leaf in jax.tree.leaves(migrated_parallel.state)] == shapes
    checkpoint = parallel.save()
    restored = _trainer(tmp_path, 1, mode, global_batch)
    restored.restore(checkpoint)
    restored_parallel = _trainer(tmp_path, num_devices, mode, global_batch)
    restored_parallel.restore(checkpoint)
    _tree_allclose(parallel.state, restored.state, rtol=0, atol=0)
    _tree_allclose(parallel.optimizer_states, restored.optimizer_states, rtol=0, atol=0)
    np.testing.assert_array_equal(jax.device_get(parallel.rng), jax.device_get(restored.rng))
    assert restored.sampler.state_dict() == parallel.sampler.state_dict()
    assert asdict(restored.progress) == asdict(parallel.progress)
    assert [leaf.shape for leaf in jax.tree.leaves(restored.state)] == shapes
    # Resume performs another real optimizer update, testing restored moments,
    # key/counter continuity and changing device counts rather than only shapes.
    for trainer in (serial, parallel, migrated_parallel, restored, restored_parallel):
        if mode == "stage1":
            trainer.config.training.stage1_updates = 2
            trainer.progress.complete = False
        else:
            trainer.config.training.alternating_cycles = 2
            trainer.progress.set_cycle_budget(2)
    for _ in range(1 if mode == "stage1" else 2):
        parallel_next = parallel.step()
        for continued in (serial, migrated_parallel, restored, restored_parallel):
            restored_next = continued.step()
            for name in ("step", "stage", "cycle", "phase_step", "task_updates", "task_samples"):
                assert parallel_next[name] == restored_next[name]
            np.testing.assert_allclose(parallel_next["loss"], restored_next["loss"], rtol=1e-5, atol=1e-6)
            _tree_allclose(parallel.state, continued.state)
            _tree_allclose(parallel.optimizer_states, continued.optimizer_states)
            np.testing.assert_array_equal(jax.device_get(parallel.rng), jax.device_get(continued.rng))
            assert parallel.sampler.state_dict() == continued.sampler.state_dict()
    # Canonical checkpoints merge into the ordinary single-device NNX graph,
    # including non-parameter variables, without an extra device dimension.
    inference_model = nnx.merge(restored.graphdef, restored.state)
    inference_model.eval()
    inference_batch = jax.device_put(collate_sequences([_sample(3, 0.4, demonstration=True)]))
    loss, _ = inference_model.compute_sequence_loss(
        restored.rng, T2Observation.from_dict(inference_batch["observation"]), inference_batch["actions"],
        inference_batch["action_mask"], inference_batch["write_mask"], inference_batch["observation_mask"],
        train=False, memory_enabled=mode != "stage1",
    )
    assert np.isfinite(float(jax.device_get(loss)))
