"""Exact chronological scan with temporal-block activation rematerialization."""

import jax


def differentiable_scan(step, initial, inputs, *, remat=True, block_size=16):
    """Reduce recurrent snapshot storage without detaching any gradient.

    Checkpointing a frame alone retains its incoming fast state at every time
    step. Checkpointing a whole temporal block retains boundary states and
    recomputes the inner trajectory during backward. Both scans differentiate
    through all writes, including writes in earlier blocks.
    """
    if block_size <= 0:
        raise ValueError("Temporal rematerialization block size must be positive")
    if not remat:
        return jax.lax.scan(step, initial, inputs)
    length = jax.tree.leaves(inputs)[0].shape[0]
    if length < 1:
        raise ValueError("An episode sequence must contain at least one observation")
    block_size = min(block_size, length)
    # Dataset batches have power-of-two padding. For other callers, use a
    # divisor without adding synthetic observations or changing the sequence.
    while length % block_size:
        block_size -= 1
    blocks = jax.tree.map(lambda x: x.reshape(length // block_size, block_size, *x.shape[1:]), inputs)

    def block_step(state, block):
        return jax.lax.scan(jax.checkpoint(step), state, block)

    final, outputs = jax.lax.scan(jax.checkpoint(block_step), initial, blocks)
    outputs = jax.tree.map(lambda x: x.reshape(length, *x.shape[2:]), outputs)
    return final, outputs
