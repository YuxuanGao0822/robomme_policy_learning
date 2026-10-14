"""Data-parallel minibatch gradients with explicit weights for real examples.

Slow parameters and optimizer states keep their ordinary logical shapes. Only
per-device episode inputs and unreduced gradient sums have a leading data axis;
one final collective converts those sums back into replicated logical arrays.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from .dataset import SequenceSample, collate_sequences

DATA_AXIS = "data"


def wave_sizes(global_batch: int, num_devices: int, micro_batch_size: int = 1) -> tuple[int, ...]:
    """Return real sample counts, including a partially occupied last wave."""
    if type(global_batch) is not int or global_batch <= 0:
        raise ValueError("global_batch must be a positive integer")
    if type(num_devices) is not int or num_devices <= 0:
        raise ValueError("num_devices must be a positive integer")
    if type(micro_batch_size) is not int or micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be a positive integer")
    capacity = num_devices * micro_batch_size
    full, tail = divmod(global_batch, capacity)
    return (capacity,) * full + ((tail,) if tail else ())


def collate_parallel_wave(samples: list[SequenceSample], num_devices: int,
                          micro_batch_size: int = 1) -> tuple[dict, np.ndarray]:
    """Pad one wave to ``[D,T,M,...]`` with completely inactive dummy slots.

    All devices use the same power-of-two temporal bucket. Dummy observations
    reuse a real template, while every action/write/observation mask is false.
    They do not represent sampled episodes and do not advance any host RNG.
    """
    if type(num_devices) is not int or num_devices <= 0:
        raise ValueError("num_devices must be a positive integer")
    if type(micro_batch_size) is not int or micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be a positive integer")
    capacity = num_devices * micro_batch_size
    if not 1 <= len(samples) <= capacity:
        raise ValueError("A parallel wave needs between one and num_devices * micro_batch_size real samples")
    real = len(samples)
    batch = collate_sequences(samples + [samples[0]] * (capacity - real))
    for name in ("action_mask", "write_mask", "observation_mask"):
        batch[name][:, real:] = False

    def device_batches(value: np.ndarray) -> np.ndarray:
        flat = np.moveaxis(value, 1, 0)
        grouped = flat.reshape(num_devices, micro_batch_size, *flat.shape[1:])
        return np.moveaxis(grouped, 2, 1)

    batch = jax.tree.map(device_batches, batch)
    counts = (np.arange(capacity).reshape(num_devices, micro_batch_size) < real).sum(axis=1, dtype=np.int32)
    return batch, counts


def make_mesh(num_devices: int) -> Mesh:
    """Select local devices; the CLI independently enforces the visible count."""
    if type(num_devices) is not int or num_devices <= 0:
        raise ValueError("num_devices must be a positive integer")
    devices = jax.local_devices()
    if num_devices > len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are local")
    return Mesh(np.asarray(devices[:num_devices]), (DATA_AXIS,))


def make_parallel_grad(loss_fn: Callable, mesh: Mesh, micro_batch_size: int = 1) -> Callable:
    """Compute independent local gradients without any gradient all-reduce.

    Inputs are ``(params, state, keys[D,2], batch[D,T,M,...], real_counts[D])``.
    The loss function returns a mean over this device's real samples. Gradients,
    loss and mean statistics are weighted by real_count before final reduction.
    Stage 1 uses T=1 batched GEMMs; Stage 2 retains M=1 independent episodes.
    Dummy results use ``where`` so NaNs cannot contaminate the collective.
    """
    replica = NamedSharding(mesh, P())
    slots = NamedSharding(mesh, P(DATA_AXIS))
    if type(micro_batch_size) is not int or micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be a positive integer")
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def local_grad(params: Any, state: Any, keys: Any, batch: dict, valid: Any) -> Any:
        local_batch = jax.tree.map(lambda value: value[0], batch)
        result = value_and_grad(params, state, keys[0], local_batch)
        real_count = valid[0]
        return jax.tree.map(
            lambda value: jnp.where(real_count > 0, value * real_count, jnp.zeros_like(value))[None], result,
        )

    mapped = shard_map(local_grad, mesh=mesh,
                       in_specs=(P(), P(), P(DATA_AXIS), P(DATA_AXIS), P(DATA_AXIS)),
                       out_specs=((P(DATA_AXIS), P(DATA_AXIS)), P(DATA_AXIS)))
    return jax.jit(mapped, in_shardings=(replica, replica, slots, slots, slots),
                   out_shardings=((slots, slots), slots))


def make_parallel_accumulator(mesh: Mesh) -> Callable:
    """Accumulate local sums while preserving their sharded leading D axis."""
    slots = NamedSharding(mesh, P(DATA_AXIS))
    return jax.jit(lambda old, new: jax.tree.map(jnp.add, old, new),
                   in_shardings=(slots, slots), out_shardings=slots)


def make_gradient_reducer(mesh: Mesh) -> Callable:
    """Sum all device-local wave accumulators, dividing by real global batch.

    This accepts a gradient PyTree or a bundle of gradients/loss/statistics.
    The denominator counts real examples: frames in Stage 1 and entire episodes
    in Stage 2. Temporal padding and unused device slots do not add examples.
    In particular the partially occupied final wave receives the same episode
    weight as a complete wave. Returned arrays have no data/replica dimension.
    """
    replica = NamedSharding(mesh, P())
    slots = NamedSharding(mesh, P(DATA_AXIS))

    def reduce(tree: Any, count: Any) -> Any:
        return jax.tree.map(lambda value: jax.lax.psum(value[0], DATA_AXIS) / count, tree)

    mapped = shard_map(reduce, mesh=mesh, in_specs=(P(DATA_AXIS), P()), out_specs=P())
    return jax.jit(mapped, in_shardings=(slots, replica), out_shardings=replica)
