"""Paper Eq. (16): valid-element normalized Flow Matching action loss."""

import jax
import jax.numpy as jnp


def mask_action_domain(actions: jax.Array, robot_action_dim: int = 8) -> jax.Array:
    """Keep physical robot dimensions, zeroing padded channels at every flow step.

    The model's 32-dimensional OpenPI representation contains only eight
    RoboMME action dimensions. Training supplies zero noisy inputs outside this
    domain, so inference must initialize and evolve those same channels at zero.
    """
    if actions.ndim < 1 or not 1 <= robot_action_dim <= actions.shape[-1]:
        raise ValueError("robot_action_dim must fit inside the model action width")
    valid = jnp.arange(actions.shape[-1]) < robot_action_dim
    return jnp.where(valid, actions, jnp.zeros((), dtype=actions.dtype))


def masked_flow_action_loss(
    velocity: jax.Array,
    noise: jax.Array,
    actions: jax.Array,
    action_mask: jax.Array,
    observation_valid: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return summed normalized chunk losses and the supervised chunk count.

    Inputs are ``[..., prediction_horizon, action_dim]``. Every chunk is
    normalized by its own valid action-element count, then chunks are averaged
    by the caller over the chronological sequence. Unsupervised demonstration
    frames and sequence padding contribute neither loss nor count. Sanitizing
    operands before subtraction also prevents padded NaNs contaminating outer
    gradients through a nominally zero ``where`` branch.
    """
    if velocity.shape != noise.shape or velocity.shape != actions.shape:
        raise ValueError("Velocity, noise, and expert action shapes must match")
    if velocity.ndim < 2:
        raise ValueError("Action chunks need prediction-horizon and action-dimension axes")
    mask = jnp.broadcast_to(jnp.asarray(action_mask, dtype=jnp.bool_), velocity.shape)
    if observation_valid is not None:
        valid = jnp.broadcast_to(jnp.asarray(observation_valid, dtype=jnp.bool_), velocity.shape[:-2])
        mask = mask & valid[..., None, None]
    predicted = jnp.where(mask, velocity.astype(jnp.float32), 0.0)
    target_noise = jnp.where(mask, noise.astype(jnp.float32), 0.0)
    target_actions = jnp.where(mask, actions.astype(jnp.float32), 0.0)
    squared = jnp.square(predicted - (target_noise - target_actions))
    counts = jnp.sum(mask, axis=(-2, -1), dtype=jnp.int32)
    chunk_loss = jnp.sum(squared, axis=(-2, -1)) / jnp.maximum(counts, 1)
    return jnp.sum(chunk_loss, dtype=jnp.float32), jnp.sum(counts > 0, dtype=jnp.int32)
