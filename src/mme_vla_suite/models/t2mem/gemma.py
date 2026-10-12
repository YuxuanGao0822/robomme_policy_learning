"""Observation-grounded, layer-delayed memory using the native OpenPI experts.

Policy parameter names and tensor shapes are preserved for pi05 checkpoint loading.
Encoding builds one cache per layer from VL and the *incoming* interface. The
updated interface is therefore available at the next layer (17 effective layers).
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from mme_vla_suite.models.t2mem.memory import FastWeights
from mme_vla_suite.models.t2mem.memory import memory_step
from mme_vla_suite.models.t2mem.memory import interleaved_rope
from mme_vla_suite.models.t2mem.memory import rms_normalize
from openpi.models import gemma
from openpi.models import lora


class Adapters(nn.Module):
    rank: int

    @nn.compact
    def __call__(self, x, output_width, name):
        a = self.param(name + "_a", nn.initializers.normal(0.01), (x.shape[-1], self.rank))
        b = self.param(name + "_b", nn.initializers.zeros, (self.rank, output_width))
        return (x @ a.astype(x.dtype)) @ b.astype(x.dtype)


class Attention(gemma.Attention):
    """Native mixed-expert attention; interface-only Q/KV/O rank-16 adapters."""
    adapter_rank: int = 16

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, interface_mode=False):
        dtype = next(x.dtype for x in xs if x is not None)
        qkvs = []
        adapters = Adapters(self.adapter_rank, name="memory") if interface_mode else None
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            q = lora.Einsum(
                shape=(config.num_heads, config.width, config.head_dim),
                name=gemma._name("q_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BTD,NDH->BTNH", x)
            k, v = lora.Einsum(
                shape=(2, config.num_kv_heads, config.width, config.head_dim),
                name=gemma._name("kv_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                lora_config=config.lora_configs.get("attn"),
            )("BSD,2KDH->2BSKH", x)
            if interface_mode and i == 1:
                q = q + adapters(x, config.num_heads * config.head_dim, "q").reshape(q.shape)
                delta = adapters(x, 2 * config.num_kv_heads * config.head_dim, "kv")
                dk, dv = jnp.split(delta, 2, axis=-1)
                k, v = k + dk.reshape(k.shape), v + dv.reshape(v.shape)
            qkvs.append((q, k, v))
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
        q = gemma._apply_rope(q, positions=positions) * self.configs[0].head_dim**-0.5
        k = gemma._apply_rope(k, positions=positions)
        if kv_cache is not None:
            k = jnp.concatenate([kv_cache[0], k], axis=1)
            v = jnp.concatenate([kv_cache[1], v], axis=1)
        cfg = self.configs[0]
        q = q.reshape(*q.shape[:2], cfg.num_kv_heads, cfg.num_heads // cfg.num_kv_heads, cfg.head_dim)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        logits = jnp.where(attn_mask[:, :, None], logits, -2.3819763e38)
        probs = jax.nn.softmax(logits, axis=-1).astype(dtype)
        # An all-invalid query must have a zero summary, not an average of padding.
        probs = jnp.where(jnp.any(attn_mask, axis=-1)[:, :, None, :, None], probs, 0)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = encoded.reshape(*encoded.shape[:2], cfg.num_heads, cfg.head_dim)
        out, start = [], 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]
            z = encoded[:, start:end]
            projected = lora.Einsum(
                shape=(config.num_heads, config.head_dim, config.width),
                name=gemma._name("attn_vec_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                lora_config=config.lora_configs.get("attn"),
            )("BTNH,NHD->BTD", z)
            if interface_mode and i == 1:
                projected += adapters(z.reshape(*z.shape[:2], -1), config.width, "out")
            out.append(projected)
            start = end
        return out, (k, v)


class MemoryLayer(nn.Module):
    config: object

    def setup(self):
        self.pre_attention_scale = self.param("pre_attention_scale", nn.initializers.zeros, (self.config.width,))
        self.pre_ffw_scale = self.param("pre_ffw_scale", nn.initializers.zeros, (self.config.width,))

    def normalize(self, x, name):
        scale = getattr(self, name)
        return (rms_normalize(x.astype(jnp.float32)) * (1 + scale)).astype(x.dtype)

    @nn.compact
    def __call__(self, summary, state, write_mask):
        c = self.config
        initial = FastWeights(
            self.param("W0_w1", nn.initializers.normal(0.02), (c.num_heads, c.head_dim, c.hidden_dim)),
            self.param("W0_b1", nn.initializers.zeros, (c.num_heads, c.hidden_dim)),
            self.param("W0_w2", nn.initializers.normal(0.02), (c.num_heads, c.hidden_dim, c.head_dim)),
            self.param("W0_b2", nn.initializers.zeros, (c.num_heads, c.head_dim)),
        )
        rho = self.param("rho", nn.initializers.constant(math.log(math.e - 1)), ())
        alpha = self.param("alpha", nn.initializers.constant(math.atanh(0.01)), (c.width,))
        bindings = []
        for name in ("q", "k", "v"):
            p = nn.Dense(c.width, use_bias=True, dtype=jnp.float32, name=name)(summary.astype(jnp.float32))
            p = p.reshape(*p.shape[:2], c.num_heads, c.head_dim).transpose(0, 2, 1, 3)
            p = rms_normalize(p)
            if name != "v":
                p = interleaved_rope(p)
            bindings.append(p)
        read, next_state, stats = memory_step(
            state, initial, *bindings, rho, write_mask & c.online_write,
            adaptive_write=c.adaptive_write, detach_fast_state=c.detach_fast_state,
        )
        read = read.transpose(0, 2, 1, 3).reshape(summary.shape)
        return (summary.astype(jnp.float32) + jnp.tanh(alpha) * read).astype(summary.dtype), next_state, stats


class Block(nn.Module):
    configs: tuple
    memory_config: object

    @nn.compact
    def __call__(self, carry, kv_cache, fast_state, layer_index, positions, mask, cond, write_mask, mode):
        xs, interface = carry
        memory = MemoryLayer(self.memory_config, name="memory")
        attn = Attention(self.configs, adapter_rank=self.memory_config.adapter_rank, name="attn")
        if mode in ("encode", "baseline"):
            vl, _ = xs
            vl_norm, _ = gemma.RMSNorm(name="pre_attention_norm")(vl, None)
            if mode == "baseline":
                updates, cache = attn([vl_norm, None], positions[:, :vl.shape[1]], mask[:, :, :vl.shape[1], :vl.shape[1]], None)
                vl = vl + updates[0]
                z, _ = gemma.RMSNorm(name="pre_ffw_norm")(vl, None)
                vl += lora.FeedForward(features=self.configs[0].width, hidden_dim=self.configs[0].mlp_dim,
                                      lora_config=self.configs[0].lora_configs.get("ffn"), name="mlp")(z)
                n = self.memory_config.num_tokens
                cache = tuple(jnp.pad(v, ((0,0),(0,n),(0,0),(0,0))) for v in cache)
                return ([vl, None], interface), (cache, fast_state, {})
            if_norm = memory.normalize(interface, "pre_attention_scale")
            updates, cache = attn([vl_norm, if_norm], positions, mask, None, True)
            vl = vl + updates[0]
            u, next_state, stats = memory(updates[1], fast_state, write_mask)
            interface = interface + u
            if_norm = memory.normalize(interface, "pre_ffw_scale")
            interface += lora.FeedForward(
                features=self.configs[1].width, hidden_dim=self.configs[1].mlp_dim,
                lora_config=self.configs[1].lora_configs.get("ffn"), name="mlp_1",
            )(if_norm)
            vl_norm, _ = gemma.RMSNorm(name="pre_ffw_norm")(vl, None)
            vl += lora.FeedForward(
                features=self.configs[0].width, hidden_dim=self.configs[0].mlp_dim,
                lora_config=self.configs[0].lora_configs.get("ffn"), name="mlp",
            )(vl_norm)
            # Initial interface tokens are queries only; no enriched context at layer zero.
            n = self.memory_config.num_tokens
            cache = tuple(jnp.concatenate([v[:, :-n], jnp.where(layer_index > 0, v[:, -n:], 0)], axis=1) for v in cache)
            return ([vl, None], interface), (cache, next_state, stats)
        suffix = xs[1]
        z, gate = gemma.RMSNorm(name="pre_attention_norm_1")(suffix, cond)
        updates, _ = attn([None, z], positions, mask, kv_cache)
        suffix = gemma._gated_residual(suffix, updates[1], gate)
        z, gate = gemma.RMSNorm(name="pre_ffw_norm_1")(suffix, cond)
        ff = lora.FeedForward(
            features=self.configs[1].width, hidden_dim=self.configs[1].mlp_dim,
            lora_config=self.configs[1].lora_configs.get("ffn"), name="mlp_1",
        )(z)
        suffix = gemma._gated_residual(suffix, ff, gate)
        return ([None, suffix], interface), (kv_cache, fast_state, {})


class Module(nn.Module):
    configs: tuple
    embed_dtype: str
    memory_config: object
    remat: bool = True

    def setup(self):
        if self.configs[0].depth != self.configs[1].depth:
            raise ValueError("VLM and AE depths must match")
        self.embedder = gemma.Embedder(gemma.PALIGEMMA_VOCAB_SIZE, self.configs[0].width, name="embedder")
        self.interface_tokens = self.param(
            "memory_interface_tokens", nn.initializers.normal(0.02),
            (self.memory_config.num_tokens, self.memory_config.width),
        )
        block = nn.remat(Block, prevent_cse=False, static_argnums=(9,)) if self.remat else Block
        self.layers = nn.scan(
            block, variable_axes={"params": 0}, split_rngs={"params": True},
            in_axes=(0, 0, 0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, memory_config=self.memory_config, name="layers")
        self.final_norms = [gemma.RMSNorm(name=gemma._name("final_norm", i)) for i in range(2)]

    def embed(self, tokens):
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def initial_memory_state(self, batch_size=1):
        p = self.variables["params"]["layers"]["memory"]
        return FastWeights(*(jnp.broadcast_to(p[name][:, None], (p[name].shape[0], batch_size, *p[name].shape[1:])).astype(jnp.float32)
                             for name in ("W0_w1", "W0_b1", "W0_w2", "W0_b2")))

    def encode(self, tokens, valid, fast_state, write_mask, memory_enabled=True):
        tokens = tokens.astype(self.embed_dtype)
        b, t = valid.shape
        n = self.memory_config.num_tokens
        interface = jnp.broadcast_to(self.interface_tokens, (b, n, self.memory_config.width)).astype(self.embed_dtype)
        # VL queries only see VL; interface queries only see valid VL keys.
        row = jnp.broadcast_to(valid[:, None, :], (b, t + n, t))
        row = row & jnp.concatenate([valid, jnp.ones((b, n), bool)], axis=-1)[:, :, None]
        mask = jnp.pad(row, ((0, 0), (0, 0), (0, n)))[:, None]
        positions = jnp.concatenate([jnp.cumsum(valid, axis=-1)-1,
                                     jnp.broadcast_to(jnp.arange(n), (b, n))], axis=-1)
        (_, _), (cache, next_state, stats) = self.layers(
            ([tokens, None], interface), None, fast_state, jnp.arange(self.configs[0].depth),
            positions, mask, None, jnp.asarray(write_mask), "encode" if memory_enabled else "baseline",
        )
        return cache, next_state, stats

    def decode(self, suffix, valid, cache, cond, memory_enabled=True):
        b, s = suffix.shape[:2]
        n = self.memory_config.num_tokens
        # Cache is always VL + interface. Mask interfaces for layer 0 and baseline.
        cached_valid = jnp.concatenate([valid, jnp.ones((b, n), bool)], axis=-1)
        allowed = jnp.broadcast_to(cached_valid[:, None, :], (b, s, cached_valid.shape[-1]))
        suffix_mask = jnp.ones((b, s, s), bool)
        if memory_enabled:
            # Native proprio query does not attend to the following action block.
            suffix_mask = suffix_mask.at[:, 0, 1:].set(False)
        mask = jnp.concatenate([allowed, suffix_mask], axis=-1)[:, None]
        depth = self.configs[0].depth
        masks = jnp.broadcast_to(mask, (depth, *mask.shape))
        layer_allowed = (jnp.arange(depth) > 0) & memory_enabled
        masks = masks.at[:, :, :, :, valid.shape[-1]:valid.shape[-1]+n].set(layer_allowed[:, None, None, None, None])
        positions = jnp.sum(valid, axis=-1)[:, None] + jnp.arange(s)[None]
        # Mask is layer-dependent, so decode uses a separate scan transform over
        # the same layer module/parameters rather than broadcasting the encode mask.
        def run(carry, inputs):
            index, layer_cache, layer_mask = inputs
            layer_params = jax.tree.map(lambda p: p[index], self.variables["params"]["layers"])
            (xs, _), _ = Block(self.configs, self.memory_config, parent=None).apply(
                {"params": layer_params}, ([None, carry], None), layer_cache, None, index,
                positions, layer_mask, cond, jnp.zeros((b,), bool), "decode",
            )
            return xs[1], None
        run = jax.checkpoint(run) if self.remat else run
        suffix, _ = jax.lax.scan(run, suffix.astype(self.embed_dtype), (jnp.arange(depth), cache, masks))
        return self.final_norms[1](suffix, cond)[0]

    def init(self):
        b, n = 1, self.memory_config.num_tokens
        c = self.memory_config
        empty = FastWeights(
            jnp.zeros((self.configs[0].depth,b,c.num_heads,c.head_dim,c.hidden_dim)),
            jnp.zeros((self.configs[0].depth,b,c.num_heads,c.hidden_dim)),
            jnp.zeros((self.configs[0].depth,b,c.num_heads,c.hidden_dim,c.head_dim)),
            jnp.zeros((self.configs[0].depth,b,c.num_heads,c.head_dim)),
        )
        self.embed(jnp.zeros((b,1), jnp.int32))
        cache, _, _ = self.encode(jnp.zeros((b,1,self.configs[0].width)), jnp.ones((b,1),bool), empty, jnp.zeros((b,),bool))
        # Initialize AE parameters through the scanned module so Linen registers
        # the same policy parameter tree before the pure decode scan reads it.
        dummy_mask = jnp.ones((b,1,1,1+n+1),bool)
        self.layers(([None,jnp.zeros((b,1,self.configs[1].width),jnp.dtype(self.embed_dtype))], None),
                    cache, empty, jnp.arange(self.configs[0].depth), jnp.zeros((b,1),jnp.int32),
                    dummy_mask, jnp.zeros((b,self.configs[1].width)), jnp.zeros((b,),bool), "decode")
        self.final_norms[0](jnp.zeros((b,1,self.configs[0].width),jnp.dtype(self.embed_dtype)),None)
        self.final_norms[1](jnp.zeros((b,1,self.configs[1].width),jnp.dtype(self.embed_dtype)),jnp.zeros((b,self.configs[1].width)))
