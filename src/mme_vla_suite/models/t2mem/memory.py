"""Differentiable nonlinear fast-weight memory from T²Mem, Appendix A.1.

Fast state is an ordinary JAX PyTree, separate from the optimizer's slow
parameters. A leaf starts with ``[heads, ...]`` (one episode) or
``[batch, heads, ...]``. The integrated model additionally stacks layers.
Only adaptive scaling and curvature calibration are stop-gradient quantities;
the gradient, learned multiplier, and recurrent fast weights remain connected.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MemoryConfig:
    num_heads: int = 16
    head_dim: int = 64
    hidden_dim: int = 256
    num_tokens: int = 16
    width: int = 1024
    adapter_rank: int = 16
    adaptive_write: bool = True
    online_write: bool = True
    # Explicit approximation: truncates differentiation at each committed write.
    detach_fast_state: bool = False

    def __post_init__(self):
        if self.width != self.num_heads * self.head_dim:
            raise ValueError("Memory width must equal num_heads * head_dim")
        if min(self.num_heads, self.head_dim, self.hidden_dim, self.num_tokens, self.adapter_rank) <= 0:
            raise ValueError("Memory dimensions and adapter rank must be positive")
        if self.head_dim % 2:
            raise ValueError("Interleaved RoPE requires an even memory head width")


class FastWeights(NamedTuple):
    w1: jax.Array  # [..., H, d, m]
    b1: jax.Array  # [..., H, m]
    w2: jax.Array  # [..., H, m, d]
    b2: jax.Array  # [..., H, d]


RHO_INIT = math.log(math.e - 1.0)
GATE_ALPHA_INIT = math.atanh(0.01)


def init_fast_weights(key: jax.Array, config: MemoryConfig = MemoryConfig()) -> FastWeights:
    """Learned W0 initializer: N(0, .02²) matrices and zero biases."""
    first, second = jax.random.split(key)
    h, d, m = config.num_heads, config.head_dim, config.hidden_dim
    return FastWeights(
        0.02 * jax.random.normal(first, (h, d, m), dtype=jnp.float32),
        jnp.zeros((h, m), dtype=jnp.float32),
        0.02 * jax.random.normal(second, (h, m, d), dtype=jnp.float32),
        jnp.zeros((h, d), dtype=jnp.float32),
    )


def reset_fast_state(initial: FastWeights, batch_size: int | None = None) -> FastWeights:
    """Reset to learned W0 without detaching its outer gradient.

    Arrays are immutable; broadcasting does not share mutable episode storage.
    Neither a reset nor this allocation stores state in a model parameter.
    """
    if batch_size is None:
        return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), initial)
    return jax.tree.map(lambda x: jnp.broadcast_to(x.astype(jnp.float32), (batch_size, *x.shape)), initial)


def fast_mlp(state: FastWeights, inputs: jax.Array) -> jax.Array:
    """Apply H independent d→m→d biased MLPs, with *exact* GeLU.

    ``inputs`` is ``[..., H, N, d]``; all fast arithmetic uses FP32.
    """
    x = jnp.asarray(inputs, dtype=jnp.float32)
    hidden = jnp.einsum("...hnd,...hdm->...hnm", x, state.w1.astype(jnp.float32))
    hidden = hidden + state.b1.astype(jnp.float32)[..., :, None, :]
    hidden = jax.nn.gelu(hidden, approximate=False)
    return (
        jnp.einsum("...hnm,...hmd->...hnd", hidden, state.w2.astype(jnp.float32))
        + state.b2.astype(jnp.float32)[..., :, None, :]
    )


def rms_normalize(inputs: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """Parameter-free per-head RMS normalization, Table 2."""
    x = jnp.asarray(inputs, dtype=jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + epsilon)


def interleaved_rope(inputs: jax.Array, positions: jax.Array | None = None, base: float = 10_000.0) -> jax.Array:
    """Rotate adjacent channel pairs at *interface token* positions.

    Input axes are ``[..., H, N, d]``; no environment or denoising time enters
    the rotation. The default positions are 0, …, N−1 on every observation.
    """
    x = jnp.asarray(inputs, dtype=jnp.float32)
    width = x.shape[-1]
    if width % 2:
        raise ValueError("Interleaved RoPE requires even channel width")
    if positions is None:
        positions = jnp.arange(x.shape[-2], dtype=jnp.float32)
    frequency = base ** (-jnp.arange(0, width, 2, dtype=jnp.float32) / width)
    angle = jnp.asarray(positions, dtype=jnp.float32)[..., :, None] * frequency
    # Add the head axis to [N,d/2] or [...,N,d/2] position arrays.
    angle = jnp.expand_dims(angle, axis=-3)
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    even, odd = x[..., 0::2], x[..., 1::2]
    paired = jnp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1)
    return paired.reshape(x.shape)


def project_bindings(
    summary: jax.Array,
    q_kernel: jax.Array,
    q_bias: jax.Array,
    k_kernel: jax.Array,
    k_bias: jax.Array,
    v_kernel: jax.Array,
    v_bias: jax.Array,
    *,
    num_heads: int = 16,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Separate biased Q/K/V, then head RMS normalization and Q/K RoPE."""
    def project(kernel, bias):
        x = jnp.asarray(summary, dtype=jnp.float32) @ kernel.astype(jnp.float32) + bias.astype(jnp.float32)
        x = x.reshape((*x.shape[:-1], num_heads, x.shape[-1] // num_heads))
        return rms_normalize(jnp.swapaxes(x, -3, -2))

    return (
        interleaved_rope(project(q_kernel, q_bias)),
        interleaved_rope(project(k_kernel, k_bias)),
        project(v_kernel, v_bias),
    )


def head_losses(state: FastWeights, keys: jax.Array, values: jax.Array) -> jax.Array:
    """Each head's contribution to Eq. (10), using the *global* H·N·d divisor.

    Summing this array over heads gives the observation's memory MSE. Using
    an independent N·d denominator here changes the curvature and step rule.
    """
    denominator = math.prod(keys.shape[-3:])
    return jnp.sum(jnp.square(fast_mlp(state, keys) - values.astype(jnp.float32)), axis=(-2, -1)) / denominator


def _head_squared_norm(tree: FastWeights) -> jax.Array:
    return sum(jnp.sum(jnp.square(x), axis=tuple(range(1, x.ndim))) for x in tree)


def _head_inner_product(first: FastWeights, second: FastWeights) -> jax.Array:
    return sum(jnp.sum(x * y, axis=tuple(range(1, x.ndim))) for x, y in zip(first, second, strict=True))


def _scaled_subtract(state: FastWeights, gradient: FastWeights, scale: jax.Array) -> FastWeights:
    return jax.tree.map(lambda w, g: w - scale.reshape((scale.shape[0],) + (1,) * (w.ndim - 1)) * g, state, gradient)


def adaptive_scale(
    state: FastWeights,
    initial: FastWeights,
    gradient: FastWeights,
    residual: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Eq. (11)–(12), including both zero-direction/displacement cases."""
    state, initial, gradient, residual, values = jax.tree.map(
        jax.lax.stop_gradient, (state, initial, gradient, residual, values)
    )
    displacement = jax.tree.map(jnp.subtract, state, initial)
    descent = jax.tree.map(jnp.negative, gradient)
    direction_norm2, displacement_norm2 = _head_squared_norm(descent), _head_squared_norm(displacement)
    alignment = _head_inner_product(descent, displacement) / jnp.maximum(
        jnp.sqrt(direction_norm2) * jnp.sqrt(displacement_norm2), 1e-12
    )
    novelty = 1.0 - jnp.clip(alignment, 0.0, 1.0)
    novelty = jnp.where(displacement_norm2 <= 1e-24, 1.0, novelty)
    # Zero descent takes precedence when both direction and displacement vanish.
    novelty = jnp.where(direction_norm2 <= 1e-24, 0.0, novelty)
    residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual), axis=(-2, -1)))
    value_rms = jnp.sqrt(jnp.mean(jnp.square(values), axis=(-2, -1)))
    ratio = jnp.clip(residual_rms / jnp.maximum(value_rms, 1e-8), 0.0, 1.0)
    z = novelty * (1.0 + ratio) / 2.0
    return jax.lax.stop_gradient(0.1 + 0.9 * (3.0 * z**2 - 2.0 * z**3))


def calibrated_step(
    state: FastWeights, gradient: FastWeights, keys: jax.Array, values: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Eq. (13)–(15), JVP curvature followed by a nonlinear loss probe.

    Computations are per-head contributions to the global objective, rather
    than separately normalized head objectives. No inner gradient clipping is
    applied. Calibration is detached while the caller retains the original g.
    """
    state, gradient, keys, values = jax.tree.map(jax.lax.stop_gradient, (state, gradient, keys, values))
    squared_gradient = _head_squared_norm(gradient)
    _, jacobian_gradient = jax.jvp(lambda w: fast_mlp(w, keys), (state,), (gradient,))
    denominator = math.prod(keys.shape[-3:])
    curvature = 2.0 * jnp.sum(jnp.square(jacobian_gradient), axis=(-2, -1)) / denominator
    probe_step = jnp.clip(squared_gradient / jnp.maximum(curvature, 1e-30), 1e-12, 1e3)
    original_loss = head_losses(state, keys, values)
    probe_loss = head_losses(_scaled_subtract(state, gradient, probe_step), keys, values)
    quadratic = (probe_loss - original_loss + squared_gradient * probe_step) / jnp.square(probe_step)
    step = jnp.minimum(
        1e3,
        jnp.where(quadratic > 0.0, squared_gradient / jnp.maximum(2.0 * quadratic, 1e-30), probe_step),
    )
    return jax.lax.stop_gradient(step), {
        "gradient_squared_norm": squared_gradient,
        "curvature": curvature,
        "probe_step": probe_step,
        "probe_quadratic": quadratic,
    }


def _one_episode_step(
    state: FastWeights,
    initial: FastWeights,
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    rho: jax.Array,
    write_mask: jax.Array,
    *,
    adaptive_write: bool,
    detach_fast_state: bool,
) -> tuple[jax.Array, FastWeights, dict[str, jax.Array]]:
    readout = fast_mlp(state, queries)

    def write(_):
        loss, gradient = jax.value_and_grad(lambda w: jnp.sum(head_losses(w, keys, values)))(state)
        residual = fast_mlp(state, keys) - values.astype(jnp.float32)
        beta = (
            adaptive_scale(state, initial, gradient, residual, values)
            if adaptive_write else jnp.ones((keys.shape[0],), dtype=jnp.float32)
        )
        step, stats = calibrated_step(state, gradient, keys, values)
        # Cap the calibrated step above; do not cap this learned product.
        eta = 0.5 * jax.nn.softplus(jnp.asarray(rho, dtype=jnp.float32)) * step
        updated = _scaled_subtract(state, gradient, beta * eta)
        if detach_fast_state:
            updated = jax.tree.map(jax.lax.stop_gradient, updated)
        return updated, {
            **stats,
            "beta": beta,
            "calibrated_step": step,
            "eta": eta,
            "memory_loss": jnp.broadcast_to(loss, (keys.shape[0],)),
            "head_loss": head_losses(state, keys, values),
            "write_applied": jnp.ones((keys.shape[0],), dtype=jnp.bool_),
        }

    def skip(_):
        zeros = jnp.zeros((keys.shape[0],), dtype=jnp.float32)
        return state, {
            "gradient_squared_norm": zeros,
            "curvature": zeros,
            "probe_step": zeros,
            "probe_quadratic": zeros,
            "beta": jnp.ones_like(zeros),
            "calibrated_step": zeros,
            "eta": zeros,
            "memory_loss": zeros,
            "head_loss": zeros,
            "write_applied": jnp.zeros((keys.shape[0],), dtype=jnp.bool_),
        }

    updated, stats = jax.lax.cond(jnp.asarray(write_mask, dtype=jnp.bool_), write, skip, operand=None)
    return readout, updated, stats


def memory_step(
    state: FastWeights,
    initial: FastWeights,
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    rho: jax.Array = RHO_INIT,
    write_mask: jax.Array = True,
    *,
    adaptive_write: bool = True,
    online_write: bool = True,
    detach_fast_state: bool = False,
) -> tuple[jax.Array, FastWeights, dict[str, jax.Array]]:
    """Read pre-write state, then optionally commit one Eq. (12) write.

    An unbatched call takes [H,N,d] bindings. A batched call takes [B,H,N,d]
    and independent [B,H,...] weights; write_mask is scalar or [B]. Batch size
    never enters Eq. (10)'s normalization. Layer iteration belongs to the
    integrated Transformer, and denoising must pass write_mask=False.
    """
    state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
    initial = jax.tree.map(lambda x: x.astype(jnp.float32), initial)
    mask = jnp.asarray(write_mask, dtype=jnp.bool_) & online_write
    if queries.ndim == 3:
        return _one_episode_step(
            state, initial, queries, keys, values, rho, mask,
            adaptive_write=adaptive_write, detach_fast_state=detach_fast_state,
        )
    if queries.ndim != 4:
        raise ValueError(f"Expected [H,N,d] or [B,H,N,d] memory bindings; got {queries.shape}")
    batch_size = queries.shape[0]
    if initial.w1.ndim == 3:
        initial = reset_fast_state(initial, batch_size)
    masks = jnp.broadcast_to(mask, (batch_size,))
    return jax.vmap(
        lambda w, w0, q, k, v, enabled: _one_episode_step(
            w, w0, q, k, v, rho, enabled,
            adaptive_write=adaptive_write, detach_fast_state=detach_fast_state,
        )
    )(state, initial, queries, keys, values, masks)
