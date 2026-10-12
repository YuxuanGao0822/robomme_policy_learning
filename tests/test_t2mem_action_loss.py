"""JAX action-objective tests, NOT RUN locally; requires Linux JAX environment."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.t2mem.loss import mask_action_domain
from mme_vla_suite.models.t2mem.loss import masked_flow_action_loss
from mme_vla_suite.models.t2mem import model as memory_model


def test_action_loss_normalizes_each_chunk_by_its_valid_elements():
    velocity = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 10.0], [10.0, 10.0]]])
    noise = jnp.zeros_like(velocity)
    actions = jnp.zeros_like(velocity)
    mask = jnp.array([[[True, True], [True, True]], [[True, False], [False, False]]])
    summed_loss, count = masked_flow_action_loss(velocity, noise, actions, mask)
    # Chunk 1: (1+4+9+16)/4; chunk 2: 4/1. Mean is over
    # supervised chunks instead of pooling unequal valid-element counts.
    np.testing.assert_allclose(summed_loss, 7.5 + 4.0)
    assert int(count) == 2
    np.testing.assert_allclose(summed_loss / count, 5.75)


def test_action_padding_has_zero_gradient_including_poisoned_values():
    mask = jnp.array([[[True, False], [False, False]]])
    velocity = jnp.array([[[2.0, float("nan")], [float("inf"), -1e20]]])
    noise = jnp.array([[[1.0, float("nan")], [float("inf"), 1e20]]])
    actions = jnp.array([[[0.0, float("nan")], [float("inf"), -1e20]]])
    loss, count = masked_flow_action_loss(velocity, noise, actions, mask)
    np.testing.assert_allclose(loss, 1.0)
    assert int(count) == 1
    gradient = jax.grad(lambda predicted: masked_flow_action_loss(predicted, noise, actions, mask)[0])(velocity)
    np.testing.assert_allclose(gradient, jnp.array([[[2.0, 0.0], [0.0, 0.0]]]))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_observation_padding_and_no_action_supervision_contribute_zero_loss():
    velocity = jnp.ones((3, 2, 4))
    mask = jnp.ones_like(velocity, dtype=bool).at[1].set(False)
    # First frame supervised; second has no actions; third is observation pad.
    loss, count = masked_flow_action_loss(
        velocity, jnp.zeros_like(velocity), jnp.zeros_like(velocity), mask, jnp.array([True, True, False])
    )
    np.testing.assert_allclose(loss, 1.0)
    assert int(count) == 1
    gradient = jax.grad(lambda x: masked_flow_action_loss(
        x, jnp.zeros_like(x), jnp.zeros_like(x), mask, jnp.array([True, True, False])
    )[0])(velocity)
    np.testing.assert_allclose(gradient[0], 0.25)
    np.testing.assert_array_equal(gradient[1:], 0)


def test_empty_action_sequence_has_finite_zero_loss_and_gradients():
    prediction = jnp.full((2, 3, 4), float("nan"))
    mask = jnp.zeros_like(prediction, dtype=bool)
    loss, count = masked_flow_action_loss(prediction, prediction, prediction, mask)
    np.testing.assert_array_equal(loss, 0)
    assert int(count) == 0
    gradient = jax.grad(lambda x: masked_flow_action_loss(x, prediction, prediction, mask)[0])(prediction)
    np.testing.assert_array_equal(gradient, 0)


def test_flow_target_is_noise_minus_actions():
    actions = jnp.array([[[1.0, 2.0], [-3.0, 4.0]]])
    noise = jnp.array([[[0.3, -0.5], [0.1, -0.7]]])
    exact_velocity = noise - actions
    loss, count = masked_flow_action_loss(exact_velocity, noise, actions, jnp.ones_like(actions, bool))
    np.testing.assert_array_equal(loss, 0)
    assert int(count) == 1


def test_dimension_padding_and_bf16_predictions_use_float32_loss():
    predicted = jnp.ones((1, 64, 32), dtype=jnp.bfloat16)
    mask = jnp.zeros_like(predicted, dtype=bool).at[:, :10, :8].set(True)
    loss, count = masked_flow_action_loss(predicted, jnp.zeros_like(predicted), jnp.zeros_like(predicted), mask)
    assert loss.dtype == jnp.float32
    np.testing.assert_allclose(loss, 1.0)
    assert int(count) == 1


def test_sampling_action_domain_initialization_and_every_euler_step():
    raw_noise = jax.random.normal(jax.random.key(1), (1, 64, 32))
    initial = mask_action_domain(raw_noise, robot_action_dim=8)
    np.testing.assert_array_equal(initial[..., :8], raw_noise[..., :8])
    np.testing.assert_array_equal(initial[..., 8:], 0)

    def denoise(actions, _):
        # Artificial velocity activates padded channels; the actual sampling
        # helper must keep those inputs at zero for the next action projection.
        velocity = jnp.ones_like(actions)
        updated = mask_action_domain(actions - 0.1 * velocity, robot_action_dim=8)
        return updated, updated

    final, trajectory = jax.lax.scan(denoise, initial, jnp.arange(10))
    np.testing.assert_array_equal(trajectory[..., 8:], 0)
    np.testing.assert_allclose(final[..., :8], raw_noise[..., :8] - 1.0, atol=2e-6)
    np.testing.assert_array_equal(final[..., 8:], 0)


def test_sampling_domain_mask_has_zero_padded_gradient_and_preserves_dtype():
    actions = jnp.ones((2, 32), dtype=jnp.bfloat16)
    masked = mask_action_domain(actions)
    assert masked.dtype == jnp.bfloat16
    gradient = jax.grad(lambda x: jnp.sum(mask_action_domain(x).astype(jnp.float32)))(actions)
    np.testing.assert_array_equal(gradient[..., :8], 1)
    np.testing.assert_array_equal(gradient[..., 8:], 0)


@pytest.mark.parametrize("robot_action_dim", [0, -1, 33])
def test_sampling_domain_rejects_invalid_robot_width(robot_action_dim):
    with pytest.raises(ValueError, match="fit inside"):
        mask_action_domain(jnp.ones((1, 64, 32)), robot_action_dim)


@pytest.mark.parametrize("experiment", ["baseline", "full", "no_write"])
def test_actual_sampling_method_zeros_padding_and_encodes_cache_once(monkeypatch, experiment):
    """Exercise actual sampler logic without constructing the 3B VLA.

    A concrete loop replaces only the scan primitive so every velocity input
    can be inspected without JAX tracing side effects. The method itself still
    initializes noise, constructs one observation cache, and executes its real
    Euler body with the same A/B/C action domain.
    """
    config = SimpleNamespace(
        memory_enabled=experiment != "baseline",
        memory=SimpleNamespace(online_write=experiment == "full"),
        robot_action_dim=8,
    )
    observation = SimpleNamespace(state=jnp.zeros((1, 32)))
    cache_marker = object()
    valid = jnp.ones((1, 3), dtype=bool)
    encode_calls, velocity_inputs, times = [], [], []
    initial_state = jnp.array(23.0)

    def preprocess(rng, obs, *, train):
        assert rng is None and not train
        assert obs is observation
        return obs

    def encode(obs, state, write_mask, memory_enabled):
        assert obs is observation
        assert memory_enabled == config.memory_enabled
        encode_calls.append(np.asarray(write_mask))
        commit = bool(write_mask[0]) and memory_enabled and config.memory.online_write
        return cache_marker, state + int(commit), {}, valid

    def velocity(obs, actions, time, cache, vl_valid, memory_enabled):
        assert obs is observation and cache is cache_marker and vl_valid is valid
        assert memory_enabled == config.memory_enabled
        velocity_inputs.append(np.asarray(actions))
        times.append(np.asarray(time))
        np.testing.assert_array_equal(actions[..., 8:], 0)
        # A padded velocity deliberately tries to move the extra channels.
        return jnp.ones_like(actions)

    def concrete_scan(step, carry, indices):
        for index in np.asarray(indices):
            carry, output = step(carry, jnp.asarray(index))
            assert output is None
        return carry, None

    monkeypatch.setattr(memory_model.jax.lax, "scan", concrete_scan)
    model = SimpleNamespace(
        config=config, action_horizon=64, action_dim=32,
        _preprocess=preprocess, _encode=encode, _velocity=velocity,
    )
    rng = jax.random.key(9)
    actions, returned_state = memory_model.T2Mem.sample_actions_with_memory(
        model, rng, observation, initial_state, num_steps=5, write=True
    )
    expected_noise = jax.random.normal(rng, (1, 64, 32))
    assert len(encode_calls) == 1
    assert len(velocity_inputs) == 5
    np.testing.assert_array_equal(encode_calls[0], [True])
    np.testing.assert_array_equal(velocity_inputs[0][..., :8], expected_noise[..., :8])
    np.testing.assert_array_equal(actions[..., 8:], 0)
    np.testing.assert_allclose(actions[..., :8], expected_noise[..., :8] - 1.0, atol=2e-6)
    np.testing.assert_allclose(np.concatenate(times), [1.0, 0.8, 0.6, 0.4, 0.2], atol=1e-7)
    np.testing.assert_allclose(returned_state, 24.0 if experiment == "full" else 23.0)


@pytest.mark.parametrize("num_steps", [0, -1])
def test_actual_sampling_method_rejects_nonpositive_denoising_steps(num_steps):
    with pytest.raises(ValueError, match="must be positive"):
        memory_model.T2Mem.sample_actions_with_memory(
            SimpleNamespace(), jax.random.key(0), None, None, num_steps=num_steps
        )
