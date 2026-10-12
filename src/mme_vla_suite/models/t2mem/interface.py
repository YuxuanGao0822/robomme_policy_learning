"""Observation-grounded interface utilities; native AE kernels are reused.

The integrated Transformer supplies its native attention and FFN weights.
These utilities add masks, low-rank interface-only attention deltas, and the
channel-wise memory gate; they do not allocate a replacement policy expert.
"""

import jax
import jax.numpy as jnp


def observation_only_mask(vl_valid: jax.Array, num_tokens: int = 16) -> jax.Array:
    """Interface queries can attend only to valid vision/language keys."""
    valid = jnp.asarray(vl_valid, dtype=jnp.bool_)
    return jnp.broadcast_to(valid[..., None, :], (*valid.shape[:-1], num_tokens, valid.shape[-1]))


def insert_interface_mask(native_mask: jax.Array, vl_valid: jax.Array, num_tokens: int = 16) -> jax.Array:
    """Insert interface between VL and suffix, retaining native suffix masking.

    Inputs use [VL, suffix]; the result uses [VL, interface, suffix]. VL cannot
    attend to interface/suffix, interface cannot attend to itself/suffix, and
    valid suffix queries can attend to interface in addition to native keys.
    """
    vl_size = vl_valid.shape[-1]
    total_native = native_mask.shape[-1]
    if native_mask.shape[-2] != total_native or vl_size > total_native:
        raise ValueError("Expected square native mask containing the VL prefix")
    suffix_size = total_native - vl_size
    total = total_native + num_tokens
    result = jnp.zeros((*native_mask.shape[:-2], total, total), dtype=jnp.bool_)
    result = result.at[..., :vl_size, :vl_size].set(native_mask[..., :vl_size, :vl_size])
    result = result.at[..., vl_size:vl_size + num_tokens, :vl_size].set(
        observation_only_mask(vl_valid, num_tokens)
    )
    suffix_start = vl_size + num_tokens
    if suffix_size:
        result = result.at[..., suffix_start:, :vl_size].set(native_mask[..., vl_size:, :vl_size])
        result = result.at[..., suffix_start:, suffix_start:].set(native_mask[..., vl_size:, vl_size:])
        query_valid = jnp.any(native_mask[..., vl_size:, :], axis=-1)
        result = result.at[..., suffix_start:, vl_size:suffix_start].set(
            jnp.broadcast_to(query_valid[..., :, None], (*query_valid.shape, num_tokens))
        )
    return result


def low_rank_projection(inputs: jax.Array, first: jax.Array, second: jax.Array, scale: float = 1.0) -> jax.Array:
    """A flattened rank-r interface attention delta; base weights stay native.

    The paper specifies rank 16 but does not disclose adapter factor layout or
    scaling. Flattening the native output geometry and scale=1 are engineering
    choices, recorded separately from the paper-exact memory update.
    """
    return ((inputs @ first.astype(inputs.dtype)) @ second.astype(inputs.dtype)) * scale


def channel_fusion(summary: jax.Array, readout: jax.Array, alpha: jax.Array) -> jax.Array:
    """Eq. (8). Initialize alpha=atanh(.01), giving tanh(alpha)=.01."""
    return summary + (jnp.tanh(alpha.astype(jnp.float32)) * readout.astype(jnp.float32)).astype(summary.dtype)


def masked_cross_attention(
    queries: jax.Array, keys: jax.Array, values: jax.Array, vl_valid: jax.Array
) -> jax.Array:
    """Grouped-query VL-only attention; [B,N,Q,D] queries, [B,V,K,D] KV.

    Fully masked observations return zero instead of averaging invalid keys.
    This is also useful for standalone mask tests; the integrated model uses
    the existing Gemma attention implementation and its pretrained kernels.
    """
    batch, tokens, heads, width = queries.shape
    kv_heads = keys.shape[-2]
    if heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    grouped = queries.reshape(batch, tokens, kv_heads, heads // kv_heads, width)
    logits = jnp.einsum("bnkgd,bvkd->bkgnv", grouped, keys, preferred_element_type=jnp.float32)
    logits = logits * width**-0.5
    mask = jnp.asarray(vl_valid, dtype=jnp.bool_)[:, None, None, None, :]
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    probabilities = jax.nn.softmax(logits, axis=-1)
    probabilities = jnp.where(mask, probabilities, 0.0).astype(values.dtype)
    output = jnp.einsum("bkgnv,bvkd->bnkgd", probabilities, values)
    return output.reshape(batch, tokens, heads, width)
