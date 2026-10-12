"""Server-side JAX tests; do not execute on the dependency-free development Mac."""

import dataclasses
import math

from flax.core import unfreeze
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.t2mem.interface import channel_fusion
from mme_vla_suite.models.t2mem.interface import insert_interface_mask
from mme_vla_suite.models.t2mem.interface import masked_cross_attention
from mme_vla_suite.models.t2mem import gemma as memory_gemma
from mme_vla_suite.models.t2mem.memory import FastWeights
from mme_vla_suite.models.t2mem.memory import GATE_ALPHA_INIT
from mme_vla_suite.models.t2mem.memory import MemoryConfig
from mme_vla_suite.models.t2mem.memory import RHO_INIT
from mme_vla_suite.models.t2mem.memory import adaptive_scale
from mme_vla_suite.models.t2mem.memory import calibrated_step
from mme_vla_suite.models.t2mem.memory import fast_mlp
from mme_vla_suite.models.t2mem.memory import head_losses
from mme_vla_suite.models.t2mem.memory import init_fast_weights
from mme_vla_suite.models.t2mem.memory import interleaved_rope
from mme_vla_suite.models.t2mem.memory import memory_step
from mme_vla_suite.models.t2mem.memory import project_bindings
from mme_vla_suite.models.t2mem.memory import reset_fast_state
from mme_vla_suite.models.t2mem.sequence import differentiable_scan
from openpi.models import gemma


def tiny_state():
    config = MemoryConfig(num_heads=2, head_dim=2, hidden_dim=3, num_tokens=2, width=4, adapter_rank=1)
    return init_fast_weights(jax.random.key(8), config)


def bindings():
    keys = jnp.array([[[0.8, -0.2], [0.1, 0.7]], [[-0.4, 0.6], [0.5, 0.2]]], dtype=jnp.float32)
    values = jnp.array([[[0.7, -0.3], [-0.2, 0.5]], [[0.4, 0.6], [-0.5, 0.3]]], dtype=jnp.float32)
    return keys, values


def assert_tree_close(actual, expected, **kwargs):
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, **kwargs)


def subtract_by_head(state, gradient, scale):
    return jax.tree.map(
        lambda weight, direction: weight - scale.reshape((scale.shape[0],) + (1,) * (weight.ndim - 1)) * direction,
        state, gradient,
    )


def test_paper_shapes_and_fp32_initialization():
    initial = init_fast_weights(jax.random.key(1))
    assert initial.w1.shape == (16, 64, 256)
    assert initial.w2.shape == (16, 256, 64)
    assert initial.b1.shape == (16, 256)
    assert initial.b2.shape == (16, 64)
    assert sum(x.size for x in initial) * 18 == 9_529_344
    assert all(x.dtype == jnp.float32 for x in initial)
    np.testing.assert_array_equal(initial.b1, 0)
    np.testing.assert_array_equal(initial.b2, 0)
    assert 0.019 < float(jnp.std(initial.w1)) < 0.021
    assert fast_mlp(initial, jnp.ones((16, 16, 64), dtype=jnp.bfloat16)).shape == (16, 16, 64)


def test_fast_mlp_exact_gelu_and_output_shape():
    identity = jnp.eye(2)[None, ...]
    state = FastWeights(identity, jnp.zeros((1, 2)), identity, jnp.zeros((1, 2)))
    inputs = jnp.array([[[-3.0, 1.25], [0.8, -1.8]]], dtype=jnp.float32)
    expected = 0.5 * inputs * (1.0 + jax.lax.erf(inputs / math.sqrt(2.0)))
    np.testing.assert_allclose(fast_mlp(state, inputs), expected, atol=1e-7)


def test_head_loss_uses_global_eq10_normalization():
    state = tiny_state()
    keys, values = bindings()
    predictions = fast_mlp(state, keys)
    losses = head_losses(state, keys, values)
    np.testing.assert_allclose(jnp.sum(losses), jnp.mean((predictions - values) ** 2), rtol=1e-6)
    gradient = jax.grad(lambda weights: jnp.sum(head_losses(weights, keys, values)))(state)
    expected_bias_gradient = 2.0 * jnp.sum(predictions - values, axis=1) / keys.size
    np.testing.assert_allclose(gradient.b2, expected_bias_gradient, rtol=1e-6)


def test_read_before_write_and_update_equation():
    initial = tiny_state()
    keys, values = bindings()
    readout, updated, stats = memory_step(initial, initial, keys, keys, values)
    gradient = jax.grad(lambda weights: jnp.sum(head_losses(weights, keys, values)))(initial)
    expected_update = subtract_by_head(initial, gradient, stats["beta"] * stats["eta"])
    np.testing.assert_array_equal(readout, fast_mlp(initial, keys))
    assert_tree_close(updated, expected_update, atol=1e-7)
    assert np.linalg.norm(np.asarray(updated.b2 - initial.b2)) > 0
    np.testing.assert_allclose(stats["eta"], 0.5 * stats["calibrated_step"], rtol=1e-6)
    assert not np.allclose(np.asarray(readout), np.asarray(fast_mlp(updated, keys)))


def test_curvature_matches_independent_jvp_and_nonlinear_probe():
    initial = tiny_state()
    keys, values = bindings()
    gradient = jax.grad(lambda weights: jnp.sum(head_losses(weights, keys, values)))(initial)
    step, stats = calibrated_step(initial, gradient, keys, values)
    _, projected_direction = jax.jvp(lambda weights: fast_mlp(weights, keys), (initial,), (gradient,))
    squared_norm = sum(jnp.sum(leaf**2, axis=tuple(range(1, leaf.ndim))) for leaf in gradient)
    curvature = 2.0 * jnp.sum(projected_direction**2, axis=(-1, -2)) / keys.size
    probe = jnp.clip(squared_norm / jnp.maximum(curvature, 1e-30), 1e-12, 1e3)
    quadratic = (
        head_losses(subtract_by_head(initial, gradient, probe), keys, values)
        - head_losses(initial, keys, values)
        + squared_norm * probe
    ) / probe**2
    expected = jnp.minimum(
        1e3, jnp.where(quadratic > 0, squared_norm / jnp.maximum(2 * quadratic, 1e-30), probe)
    )
    np.testing.assert_allclose(stats["curvature"], curvature, rtol=1e-6)
    np.testing.assert_allclose(stats["probe_step"], probe, rtol=1e-6)
    np.testing.assert_allclose(step, expected, rtol=1e-6)
    assert step.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(step)))


def test_curvature_cap_precedes_learned_multiplier():
    initial = tiny_state()
    keys, values = bindings()
    _, _, stats = memory_step(initial, initial, keys, keys, values, rho=20.0, adaptive_write=False)
    assert bool(jnp.all(stats["calibrated_step"] <= 1e3))
    np.testing.assert_allclose(stats["eta"], 0.5 * jax.nn.softplus(20.0) * stats["calibrated_step"])
    assert bool(jnp.all(stats["eta"] > stats["calibrated_step"]))


def test_adaptive_scaling_zero_direction_displacement_and_alignment():
    initial = tiny_state()
    keys, values = bindings()
    gradient = jax.tree.map(jnp.ones_like, initial)
    zero = jax.tree.map(jnp.zeros_like, gradient)
    residual = jnp.ones_like(values) * 100.0
    np.testing.assert_allclose(adaptive_scale(initial, initial, gradient, residual, values), 1.0)
    np.testing.assert_allclose(adaptive_scale(initial, initial, zero, residual, values), 0.1)
    # Exactly aligned accumulated descent attenuates subsequent writes.
    descended = jax.tree.map(lambda weight, direction: weight - direction, initial, gradient)
    np.testing.assert_allclose(adaptive_scale(descended, initial, gradient, residual, values), 0.1, atol=1e-6)
    opposite = jax.tree.map(lambda weight, direction: weight + direction, initial, gradient)
    np.testing.assert_allclose(adaptive_scale(opposite, initial, gradient, residual, values), 1.0, atol=1e-6)


def test_zero_gradient_keeps_state_and_is_finite():
    initial = tiny_state()
    keys, _ = bindings()
    values = fast_mlp(initial, keys)
    _, updated, stats = memory_step(initial, initial, keys, keys, values)
    assert_tree_close(updated, initial, atol=0)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(stats))
    np.testing.assert_allclose(stats["beta"], 0.1)


@pytest.mark.parametrize("write_mask,online_write", [(False, True), (True, False)])
def test_skip_write_retains_learned_initialization_read(write_mask, online_write):
    initial = tiny_state()
    keys, values = bindings()
    readout, updated, stats = memory_step(
        initial, initial, keys, keys, values, write_mask=write_mask, online_write=online_write
    )
    assert_tree_close(updated, initial, atol=0)
    np.testing.assert_array_equal(readout, fast_mlp(initial, keys))
    assert not bool(jnp.any(stats["write_applied"]))


def test_nonadaptive_control_retains_curvature_calibration():
    initial = tiny_state()
    keys, values = bindings()
    _, _, adaptive = memory_step(initial, initial, keys, keys, values)
    _, _, control = memory_step(initial, initial, keys, keys, values, adaptive_write=False)
    np.testing.assert_array_equal(control["beta"], 1)
    np.testing.assert_allclose(control["eta"], adaptive["eta"])
    assert bool(jnp.all(control["calibrated_step"] > 0))


def test_episode_reset_no_batch_leakage_and_write_mask():
    initial = tiny_state()
    state = reset_fast_state(initial, 2)
    keys, values = bindings()
    batched_keys = jnp.stack((keys, keys))
    batched_values = jnp.stack((values, values * -2))
    _, updated, _ = memory_step(state, initial, batched_keys, batched_keys, batched_values, write_mask=jnp.array([True, False]))
    untouched = jax.tree.map(lambda leaf: leaf[1], updated)
    assert_tree_close(untouched, initial, atol=0)
    _, alone, _ = memory_step(initial, initial, keys, keys, values)
    assert_tree_close(jax.tree.map(lambda leaf: leaf[0], updated), alone, rtol=1e-6, atol=1e-7)
    assert_tree_close(reset_fast_state(initial), initial, atol=0)


def test_qkv_normalization_biased_projection_and_interleaved_token_rope():
    summary = jnp.array([[[1.0, -2.0, 3.0, 0.5], [2.0, 1.0, -1.0, 4.0]]])
    identity = jnp.eye(4)
    zeros = jnp.zeros((4,))
    q, k, v = project_bindings(summary, identity, zeros, identity, zeros, identity, jnp.ones((4,)), num_heads=2)
    assert q.shape == k.shape == v.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(jnp.mean(q**2, axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(np.asarray(q), np.asarray(v))
    # Adjacent pair [1,0] at token 1 rotates to [cos(1),sin(1)].
    pair = jnp.array([[[1.0, 0.0], [1.0, 0.0]]])
    expected = np.array([[[1.0, 0.0], [math.cos(1.0), math.sin(1.0)]]])
    np.testing.assert_allclose(interleaved_rope(pair), expected, atol=1e-7)


def test_interface_mask_preserves_direct_path_and_excludes_suffix_keys():
    # Two valid VL tokens, one invalid; two valid suffix tokens.
    valid = jnp.array([[True, True, False]])
    native = jnp.array([[
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, False, False, False],
        [True, True, False, True, False],
        [True, True, False, True, True],
    ]])
    mask = insert_interface_mask(native, valid, num_tokens=2)
    assert mask.shape == (1, 7, 7)
    np.testing.assert_array_equal(mask[:, 3:5, :3], jnp.broadcast_to(valid[:, None, :], (1, 2, 3)))
    assert not bool(jnp.any(mask[:, 3:5, 3:]))
    assert not bool(jnp.any(mask[:, :3, 3:]))
    np.testing.assert_array_equal(mask[:, 5:, :3], native[:, 3:, :3])
    np.testing.assert_array_equal(mask[:, 5:, 5:], native[:, 3:, 3:])
    assert bool(jnp.all(mask[:, 5:, 3:5]))


def test_cross_attention_ignores_invalid_tokens_and_all_masked_is_zero():
    query = jnp.ones((1, 2, 4, 2))
    keys = jnp.ones((1, 3, 1, 2))
    values = jnp.array([[[[1.0, 2.0]], [[1000.0, 3000.0]], [[5.0, 6.0]]]])
    actual = masked_cross_attention(query, keys, values, jnp.array([[True, False, True]]))
    np.testing.assert_allclose(actual, jnp.broadcast_to(jnp.array([3.0, 4.0]), actual.shape))
    empty = masked_cross_attention(query, keys, values, jnp.zeros((1, 3), dtype=bool))
    np.testing.assert_array_equal(empty, 0)


def test_gate_initializes_to_point_zero_one_per_channel():
    summary = jnp.zeros((1, 2, 4))
    np.testing.assert_allclose(channel_fusion(summary, jnp.ones_like(summary), jnp.full((4,), GATE_ALPHA_INIT)), 0.01)


def test_late_action_loss_has_gradient_through_earlier_write_and_finite_difference():
    """Finite differences freeze only Eq. (11)–(15)'s detached quantities.

    Ordinary finite differences that recompute beta/calibration would test a
    different derivative than the paper prescribes. The inner gradients and
    second write's dependence on its pre-write state remain recomputed here.
    """
    initial = tiny_state()
    keys, values = bindings()
    late_query = keys * 0.6 + 0.2

    def forward(scale, detach=False, return_stats=False):
        _, first, first_stats = memory_step(initial, initial, keys, keys, scale * values, detach_fast_state=detach)
        _, second, second_stats = memory_step(first, initial, keys, keys * 0.7, -values * 0.4, detach_fast_state=detach)
        prediction = fast_mlp(second, late_query)
        loss = jnp.mean((prediction - values * 0.25) ** 2)
        return (loss, (first_stats, second_stats)) if return_stats else loss

    gradient = jax.grad(forward)(jnp.array(1.0))
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 1e-6
    detached = jax.grad(lambda scale: forward(scale, detach=True))(jnp.array(1.0))
    np.testing.assert_allclose(detached, 0.0, atol=1e-8)
    _, constants = forward(jnp.array(1.0), return_stats=True)

    def replay(scale):
        state = initial
        for binding_keys, binding_values, stats in (
            (keys, scale * values, constants[0]),
            (keys * 0.7, -values * 0.4, constants[1]),
        ):
            gradient = jax.grad(lambda w: jnp.sum(head_losses(w, binding_keys, binding_values)))(state)
            state = subtract_by_head(state, gradient, stats["beta"] * stats["eta"])
        return jnp.mean((fast_mlp(state, late_query) - values * 0.25) ** 2)

    epsilon = 1e-3
    difference = (replay(1.0 + epsilon) - replay(1.0 - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(gradient, difference, rtol=0.02, atol=2e-5)


def test_meta_gradient_reaches_initialization_and_learned_multiplier():
    initial = tiny_state()
    keys, values = bindings()

    def loss(w0, rho):
        _, state, _ = memory_step(w0, w0, keys, keys, values, rho)
        return jnp.mean((fast_mlp(state, keys * 0.3) - values) ** 2)

    initialization_gradient, multiplier_gradient = jax.grad(loss, argnums=(0, 1))(initial, jnp.array(RHO_INIT))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in initialization_gradient)
    assert sum(float(jnp.linalg.norm(x)) for x in initialization_gradient) > 0
    assert bool(jnp.isfinite(multiplier_gradient))
    assert abs(float(multiplier_gradient)) > 1e-6


def test_scan_retains_gradient_across_chronological_writes():
    initial = tiny_state()
    keys, values = bindings()

    def objective(early_value):
        writes = jnp.stack((early_value, values * 0.5))

        def transition(state, current_value):
            read, state, _ = memory_step(state, initial, keys, keys, current_value)
            return state, read

        final, _ = jax.lax.scan(jax.checkpoint(transition), initial, writes)
        return jnp.mean(fast_mlp(final, keys)**2)

    gradient = jax.grad(objective)(values)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 1e-6


@pytest.mark.parametrize("remat,block_size,length", [
    (True, 2, 4),
    (True, 4, 6),  # Requested block size does not divide the sequence length.
    (True, 3, 5),  # Prime length falls back to a divisor, without synthetic frames.
    (True, 16, 1),
    (False, 2, 4),
    (False, 3, 5),
])
def test_block_rematerialization_retains_full_sequence_gradient(remat, block_size, length):
    """The actual helper preserves values, outputs, and earlier-write gradients."""
    initial = tiny_state()
    keys, values = bindings()

    def transition(state, target):
        read, updated, stats = memory_step(state, initial, keys, keys, target)
        return updated, (read, {"beta": stats["beta"], "write": stats["write_applied"]})

    def loss(early, blocked):
        rest = values[None, ...] * jnp.linspace(0.7, 0.3, max(length - 1, 0))[:, None, None, None]
        sequence = jnp.concatenate((early[None, ...], rest), axis=0)
        if blocked:
            final, outputs = differentiable_scan(
                transition, initial, sequence, remat=remat, block_size=block_size
            )
        else:
            final, outputs = jax.lax.scan(transition, initial, sequence)
        return jnp.mean((fast_mlp(final, keys) - values)**2), (final, outputs)

    (direct_loss, direct_state), direct_grad = jax.value_and_grad(
        lambda early: loss(early, False), has_aux=True
    )(values)
    (blocked_loss, blocked_state), blocked_grad = jax.value_and_grad(
        lambda early: loss(early, True), has_aux=True
    )(values)
    np.testing.assert_allclose(blocked_loss, direct_loss, rtol=1e-6, atol=1e-7)
    assert_tree_close(blocked_state, direct_state, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(blocked_grad, direct_grad, rtol=1e-5, atol=1e-7)
    assert float(jnp.linalg.norm(blocked_grad)) > 1e-6


def test_temporal_scan_handles_pytree_inputs_and_none_outputs():
    # Exercises the same input tuple / nested observation layout as the model,
    # separately from the fast-weight test's array inputs and structured outputs.
    inputs = (jnp.arange(6, dtype=jnp.float32), {"observation": jnp.ones((6, 2))})

    def step(state, item):
        value, observation = item
        return state + value + jnp.sum(observation["observation"]), None

    for remat in (False, True):
        final, outputs = differentiable_scan(step, jnp.array(0.0), inputs, remat=remat, block_size=4)
        np.testing.assert_allclose(final, 27.0)
        assert outputs is None


@pytest.mark.parametrize("block_size", [0, -1])
def test_temporal_scan_rejects_nonpositive_block_size(block_size):
    with pytest.raises(ValueError, match="positive"):
        differentiable_scan(lambda state, x: (state + x, x), jnp.array(0.0), jnp.ones((2,)), block_size=block_size)


def test_rematerialized_scan_rejects_empty_episode():
    with pytest.raises(ValueError, match="at least one"):
        differentiable_scan(lambda state, x: (state + x, x), jnp.array(0.0), jnp.ones((0,)))


@pytest.fixture(scope="module")
def tiny_transformer():
    """Native tiny Gemma experts, with no vision encoder/checkpoint download."""
    policy_config = gemma.get_config("dummy")
    memory_config = MemoryConfig(num_heads=4, head_dim=16, hidden_dim=32, num_tokens=2, width=64, adapter_rank=2)
    module = memory_gemma.Module((policy_config, policy_config), "float32", memory_config, remat=False)
    # This class has the same convenience init method as native OpenPI Gemma;
    # call Linen's inherited initializer explicitly to execute that method.
    variables = nn.Module.init(module, jax.random.key(4), method="init")
    variables = unfreeze(jax.tree.map(jnp.asarray, variables))
    # AdaRMS gates initialize at zero in OpenPI. Enable them for an action-path
    # test so gradients can reach attention instead of testing a closed gate.
    parameters = variables["params"]["layers"]
    for name in ("pre_attention_norm_1", "pre_ffw_norm_1"):
        bias = parameters[name]["Dense_0"]["bias"]
        parameters[name]["Dense_0"]["bias"] = bias.at[..., 2 * policy_config.width:].set(1.0)
    return module, variables


def test_integrated_transformer_layer_delayed_cache_and_fp32_fast_state(tiny_transformer):
    module, variables = tiny_transformer
    state = module.apply(variables, 2, method="initial_memory_state")
    prefix = jax.random.normal(jax.random.key(5), (2, 3, 64))
    valid = jnp.array([[True, True, False], [True, True, True]])
    cache, updated, stats = module.apply(variables, prefix, valid, state, jnp.array([True, False]), method="encode")
    assert cache[0].shape == cache[1].shape == (4, 2, 5, 1, 16)
    assert updated.w1.shape == (4, 2, 4, 16, 32)
    assert all(leaf.dtype == jnp.float32 for leaf in updated)
    assert stats["beta"].shape == (4, 2, 4)
    np.testing.assert_array_equal(cache[0][0, :, -2:], 0)
    np.testing.assert_array_equal(cache[1][0, :, -2:], 0)
    assert bool(jnp.any(cache[0][1:, :, -2:] != 0))
    assert_tree_close(jax.tree.map(lambda leaf: leaf[:, 1], updated), jax.tree.map(lambda leaf: leaf[:, 1], state), atol=0)
    # Four tiny layers have three effective interface updates. The paper's
    # eighteen-layer config uses the identical rule and has seventeen.
    assert module.configs[0].depth - 1 == 3
    assert gemma.get_config("gemma_300m").depth - 1 == 17


def test_integrated_last_memory_layer_has_no_effect_on_actions(tiny_transformer):
    module, variables = tiny_transformer
    state = module.apply(variables, 1, method="initial_memory_state")
    prefix = jax.random.normal(jax.random.key(6), (1, 3, 64))
    valid = jnp.ones((1, 3), dtype=bool)
    original, _, _ = module.apply(variables, prefix, valid, state, jnp.array([False]), method="encode")
    last_changed = state._replace(b2=state.b2.at[-1].add(5.0))
    last_cache, _, _ = module.apply(variables, prefix, valid, last_changed, jnp.array([False]), method="encode")
    assert_tree_close(last_cache, original, atol=0)
    first_changed = state._replace(b2=state.b2.at[0].add(5.0))
    first_cache, _, _ = module.apply(variables, prefix, valid, first_changed, jnp.array([False]), method="encode")
    # Memory cannot modify the direct current-VL pathway.
    assert_tree_close(tuple(leaf[:, :, :3] for leaf in first_cache), tuple(leaf[:, :, :3] for leaf in original), atol=0)
    assert bool(jnp.any(jnp.abs(first_cache[1][1, :, 3:] - original[1][1, :, 3:]) > 1e-7))
    suffix = jax.random.normal(jax.random.key(7), (1, 3, 64))
    cond = jnp.zeros((1, 64))
    action_hidden = module.apply(variables, suffix, valid, original, cond, method="decode")
    unchanged = module.apply(variables, suffix, valid, last_cache, cond, method="decode")
    assert action_hidden.shape == (1, 3, 64)
    np.testing.assert_array_equal(unchanged, action_hidden)


def test_integrated_late_action_gradient_reaches_earlier_observation_write(tiny_transformer):
    module, variables = tiny_transformer
    state = module.apply(variables, 1, method="initial_memory_state")
    early_prefix = jax.random.normal(jax.random.key(12), (1, 3, 64))
    late_prefix = jax.random.normal(jax.random.key(13), (1, 3, 64))
    suffix = jax.random.normal(jax.random.key(14), (1, 3, 64))
    target = jax.random.normal(jax.random.key(15), suffix.shape)
    valid = jnp.ones((1, 3), dtype=bool)
    cond = jnp.zeros((1, 64))

    def action_loss(early, model):
        _, written, _ = model.apply(variables, early, valid, state, jnp.array([True]), method="encode")
        cache, _, _ = model.apply(variables, late_prefix, valid, written, jnp.array([False]), method="encode")
        prediction = model.apply(variables, suffix, valid, cache, cond, method="decode")
        return jnp.mean((prediction - target)**2)

    gradient = jax.grad(lambda early: action_loss(early, module))(early_prefix)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 1e-10
    detached_module = memory_gemma.Module(
        module.configs, module.embed_dtype, dataclasses.replace(module.memory_config, detach_fast_state=True), remat=False
    )
    detached = jax.grad(lambda early: action_loss(early, detached_module))(early_prefix)
    np.testing.assert_allclose(detached, 0.0, atol=1e-10)


def test_integrated_bf16_cache_preserves_fp32_recurrence(tiny_transformer):
    original, variables = tiny_transformer
    module = memory_gemma.Module(original.configs, "bfloat16", original.memory_config, remat=True)
    state = module.apply(variables, 1, method="initial_memory_state")
    prefix = jax.random.normal(jax.random.key(16), (1, 3, 64))
    valid = jnp.ones((1, 3), dtype=bool)
    cache, updated, _ = module.apply(variables, prefix, valid, state, jnp.array([True]), method="encode")
    assert all(leaf.dtype == jnp.bfloat16 for leaf in cache)
    assert all(leaf.dtype == jnp.float32 for leaf in updated)
    suffix = jax.random.normal(jax.random.key(17), (1, 3, 64))
    prediction = module.apply(variables, suffix, valid, cache, jnp.zeros((1, 64)), method="decode")
    assert prediction.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(prediction)))


@pytest.mark.parametrize("valid_bits", [(True, True, True), (True, True, False)], ids=["all-valid", "padded-vl"])
@pytest.mark.parametrize("remat", [False, True], ids=["no-remat", "remat"])
def test_memory_free_path_matches_native_gemma_with_identical_policy_weights(tiny_transformer, valid_bits, remat):
    """Checkpoint names and separated KV decode retain native pi05 behavior.

    GPU float32 array storage does not guarantee full float32 dot-product
    precision. The joint native attention and the split/padded cache use
    different matrix and reduction shapes, so test semantic equivalence at
    highest matmul precision while retaining the original 2e-5 tolerance.
    Per-layer cache and suffix comparisons locate any remaining discrepancy;
    this context does not change production model precision.
    """
    original, variables = tiny_transformer
    module = memory_gemma.Module(original.configs, original.embed_dtype, original.memory_config, remat=remat)
    with jax.default_matmul_precision("highest"):
        state = module.apply(variables, 1, method="initial_memory_state")
        prefix = jax.random.normal(jax.random.key(20), (1, 3, 64))
        suffix = jax.random.normal(jax.random.key(21), (1, 2, 64))
        valid = jnp.array([valid_bits])
        cond = jnp.zeros((1, 64))
        cache, unchanged_state, _ = module.apply(
            variables, prefix, valid, state, jnp.array([False]), memory_enabled=False, method="encode"
        )
        actual = module.apply(variables, suffix, valid, cache, cond, memory_enabled=False, method="decode")
        assert_tree_close(unchanged_state, state, atol=0)
        total_valid = jnp.concatenate((valid, jnp.ones(suffix.shape[:2], dtype=bool)), axis=-1)
        positions = jnp.cumsum(total_valid, axis=-1) - 1
        vl_rows = jnp.concatenate(
            (jnp.broadcast_to(valid[:, None, :], (1, 3, 3)), jnp.zeros((1, 3, 2), dtype=bool)), axis=-1
        ) & valid[:, :, None]
        action_rows = jnp.broadcast_to(total_valid[:, None, :], (1, 2, 5))
        native_mask = jnp.concatenate((vl_rows, action_rows), axis=-2)[:, None]

        # Baseline IF slots are allocated but never visible. Build the same
        # split mask explicitly for layer-by-layer suffix diagnostics.
        n = module.memory_config.num_tokens
        cached_valid = jnp.concatenate((valid, jnp.zeros((1, n), dtype=bool)), axis=-1)
        split_prefix = jnp.broadcast_to(cached_valid[:, None, :], (1, suffix.shape[1], 3 + n))
        split_mask = jnp.concatenate((split_prefix, jnp.ones((1, 2, 2), dtype=bool)), axis=-1)[:, None]
        suffix_positions = jnp.sum(valid, axis=-1)[:, None] + jnp.arange(suffix.shape[1])[None]

        tokens, split_suffix, native_cached_suffix = [prefix, suffix], suffix, suffix
        parameters = variables["params"]
        layer_errors = []
        for layer in range(module.configs[0].depth):
            layer_params = jax.tree.map(lambda weights: weights[layer], parameters["layers"])
            tokens, native_cache = gemma.Block(module.configs, parent=None).apply(
                {"params": layer_params}, tokens, None, positions, native_mask, [None, cond], True
            )
            for kind, cached, native in zip(("K", "V"), cache, native_cache, strict=True):
                # Native Gemma averages an all-masked VL query; the T²Mem
                # implementation returns zero. Those invalid tokens are never
                # visible to valid VL/action queries, so compare valid KV only.
                np.testing.assert_allclose(
                    np.asarray(cached[layer])[:, :3][:, np.asarray(valid_bits)],
                    np.asarray(native)[:, :3][:, np.asarray(valid_bits)],
                    rtol=2e-5, atol=2e-5,
                    err_msg=f"Layer {layer} valid-prefix {kind} cache mismatch at highest precision; remat={remat}",
                )
            (split_tokens, _), _ = memory_gemma.Block(module.configs, module.memory_config, parent=None).apply(
                {"params": layer_params}, ([None, split_suffix], None), tuple(part[layer] for part in cache),
                None, layer, suffix_positions, split_mask, cond, jnp.array([False]), "decode",
            )
            split_suffix = split_tokens[1]
            # Same cached keys, including the masked dummy IF slots, isolate
            # attention/FFN semantics from joint-vs-cached matrix shapes.
            cached_tokens, _ = gemma.Block(module.configs, parent=None).apply(
                {"params": layer_params}, [None, native_cached_suffix], tuple(part[layer] for part in cache),
                suffix_positions, split_mask, [None, cond], True,
            )
            native_cached_suffix = cached_tokens[1]
            np.testing.assert_allclose(
                split_suffix, native_cached_suffix, rtol=2e-5, atol=2e-5,
                err_msg=f"Layer {layer} custom/native cached semantics mismatch with identical key layout; "
                        f"precision={jax.config.jax_default_matmul_precision}; remat={remat}",
            )
            layer_errors.append(float(np.max(np.abs(np.asarray(split_suffix) - np.asarray(tokens[1])))))
            np.testing.assert_allclose(
                split_suffix, tokens[1], rtol=2e-5, atol=2e-5,
                err_msg=f"Layer {layer} suffix mismatch at highest precision; max errors={layer_errors}; remat={remat}",
            )
        expected = gemma.RMSNorm(parent=None).apply(
            {"params": parameters["final_norm_1"]}, tokens[1], cond
        )[0]
        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5,
            err_msg=f"Final suffix mismatch; precision={jax.config.jax_default_matmul_precision}; "
                    f"maxabs={float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))}; "
                    f"layer max errors={layer_errors}; remat={remat}",
        )
        if not all(valid_bits):
            poisoned = prefix.at[:, 2, :].set(jnp.linspace(-1e4, 1e4, prefix.shape[-1]))
            poisoned_cache, poisoned_state, _ = module.apply(
                variables, poisoned, valid, state, jnp.array([False]), memory_enabled=False, method="encode"
            )
            poisoned_output = module.apply(
                variables, suffix, valid, poisoned_cache, cond, memory_enabled=False, method="decode"
            )
            assert_tree_close(poisoned_state, state, atol=0)
            np.testing.assert_allclose(
                poisoned_output, actual, rtol=2e-5, atol=2e-5,
                err_msg="An invalid VL token changed baseline actions despite its masked key",
            )
