"""T²Mem pi0.5 with differentiable episode-local nonlinear fast weights."""

import dataclasses

from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp

from mme_vla_suite.models.t2mem import gemma as t2gemma
from mme_vla_suite.models.t2mem.memory import MemoryConfig
from mme_vla_suite.models.t2mem.observation import T2Observation
from mme_vla_suite.models.t2mem.loss import mask_action_domain
from mme_vla_suite.models.t2mem.loss import masked_flow_action_loss
from mme_vla_suite.models.t2mem.sequence import differentiable_scan
from openpi.models import gemma
from openpi.models import model as base_model
from openpi.models import pi0
from openpi.models import siglip
from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class T2MemConfig(Pi0Config):
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m"
    pi05: bool = True
    discrete_state_input: bool = False
    action_horizon: int = 64
    robot_action_dim: int = 8
    max_token_len: int = 200
    memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)
    memory_enabled: bool = True
    remat: bool = True
    temporal_remat_block_size: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.temporal_remat_block_size <= 0:
            raise ValueError("temporal_remat_block_size must be positive")
        if self.robot_action_dim != 8 or self.action_dim < self.robot_action_dim:
            raise ValueError("PickXtimes requires eight physical action dimensions within the padded model width")
        if not self.pi05 or self.discrete_state_input:
            raise ValueError("T²Mem requires pi05=True and visual/language-only prefix; state is a suffix token")
        vlm, ae = gemma.get_config(self.paligemma_variant), gemma.get_config(self.action_expert_variant)
        if vlm.depth != ae.depth or ae.width != self.memory.width:
            raise ValueError("Memory width must match AE; VLM and AE depths must match")

    def inputs_spec(self, *, batch_size=1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        return T2Observation(
            images=obs.images, image_masks=obs.image_masks, state=obs.state,
            tokenized_prompt=obs.tokenized_prompt, tokenized_prompt_mask=obs.tokenized_prompt_mask,
            tokenized_pi05_prompt=obs.tokenized_prompt, tokenized_pi05_prompt_mask=obs.tokenized_prompt_mask,
        ), actions

    def create(self, rng):
        return T2Mem(self, nnx.Rngs(rng))


class T2Mem(pi0.Pi0):
    def __init__(self, config, rngs):
        # Reuse Pi0 embedding and projection methods without initializing a second
        # 3B-parameter backbone, which would raise the single-GPU memory peak.
        base_model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = True
        vlm = gemma.get_config(config.paligemma_variant)
        ae = gemma.get_config(config.action_expert_variant)
        llm = bridge.ToNNX(t2gemma.Module((vlm, ae), config.dtype, config.memory, config.remat))
        llm.lazy_init(rngs=rngs, method="init")
        img = bridge.ToNNX(siglip.Module(
            num_classes=vlm.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=config.dtype,
        ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, ae.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(ae.width, config.action_dim, rngs=rngs)
        self.time_mlp_in = nnx.Linear(ae.width, ae.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(ae.width, ae.width, rngs=rngs)
        self.state_proj = nnx.Linear(config.action_dim, ae.width, rngs=rngs)
        self.deterministic = True

    def initial_memory_state(self, batch_size=1):
        return self.PaliGemma.llm(batch_size, method="initial_memory_state")

    def _encode(self, observation, state, write_mask, memory_enabled=True):
        if not memory_enabled:
            native = getattr(observation, "tokenized_pi05_prompt", None)
            if native is None:
                raise ValueError("Native pi05 baseline requires the shared dual-prompt preprocessing")
            observation = dataclasses.replace(observation, tokenized_prompt=native,
                tokenized_prompt_mask=observation.tokenized_pi05_prompt_mask)
        prefix, valid, _ = self.embed_prefix(observation)
        cache, next_state, stats = self.PaliGemma.llm(
            prefix, valid, state, write_mask, memory_enabled, method="encode",
        )
        return cache, next_state, stats, valid

    def _velocity(self, observation, noisy_actions, time, cache, valid, memory_enabled=True):
        suffix, _, _, cond = super().embed_suffix(observation, noisy_actions, time)
        if memory_enabled:
            suffix = jnp.concatenate([self.state_proj(observation.state)[:, None, :], suffix], axis=1)
        hidden = self.PaliGemma.llm(suffix, valid, cache, cond, memory_enabled, method="decode")
        return self.action_out_proj(hidden[:, -self.action_horizon:])

    def _preprocess(self, rng, observation, *, train=False):
        base = base_model.preprocess_observation(rng, observation, train=train)
        if isinstance(observation, T2Observation):
            return dataclasses.replace(observation, images=base.images, image_masks=base.image_masks)
        return base

    def observe_memory(self, observation, state, *, write_mask=True):
        observation = self._preprocess(None, observation, train=False)
        write_mask = jnp.broadcast_to(jnp.asarray(write_mask, bool), observation.state.shape[:1])
        return self._encode(observation, state, write_mask, self.config.memory_enabled)[1]

    def sample_actions_with_memory(self, rng, observation, state, *, num_steps=10, write=False):
        if num_steps <= 0:
            raise ValueError("Flow denoising steps must be positive")
        observation = self._preprocess(None, observation, train=False)
        b = observation.state.shape[0]
        cache, next_state, _, valid = self._encode(
            observation, state, jnp.full((b,), write, bool), self.config.memory_enabled,
        )
        # Cache contains reads of pre-write W. It remains fixed at every flow step;
        # the one write above is returned only after this action prediction.
        noise = mask_action_domain(
            jax.random.normal(rng, (b, self.action_horizon, self.action_dim)), self.config.robot_action_dim,
        )
        dt = -1.0 / num_steps
        def step(x, index):
            time = jnp.full((b,), 1.0 + index * dt)
            velocity = self._velocity(observation, x, time, cache, valid, self.config.memory_enabled)
            # Training zeros noisy inputs in padded channels. Keep the same
            # physical action domain throughout inference for all A/B/C paths.
            return mask_action_domain(x + dt * velocity, self.config.robot_action_dim), None
        actions, _ = jax.lax.scan(step, noise, jnp.arange(num_steps))
        return actions, next_state

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        if noise is not None:
            raise ValueError("Use the stateful API; external noise is currently unsupported")
        actions, _ = self.sample_actions_with_memory(
            rng, observation, self.initial_memory_state(observation.state.shape[0]), num_steps=num_steps,
        )
        return actions

    def compute_sequence_loss(self, rng, observations, actions, action_mask, write_mask,
                              observation_mask, *, train=True, memory_enabled=True):
        """Eq.16 through the complete chronological memory recurrence.

        Inputs are time-major [T,B,...]. Every batch element starts at learned W0.
        Frozen slow groups do not imply detaching activations or fast states.
        """
        memory_enabled = memory_enabled and self.config.memory_enabled
        graph, slow = nnx.split(self)
        b = actions.shape[1]
        state = self.initial_memory_state(b)
        keys = jax.random.split(rng, actions.shape[0])
        def step(fast, inputs):
            # Reconstitute a module inside the pure scan body: bridge mutations are
            # local to the trace and no Tracers escape into the outer NNX object.
            model = nnx.merge(graph, slow)
            obs, target, mask, write, present, key = inputs
            pre_key, noise_key, time_key = jax.random.split(key, 3)
            obs = model._preprocess(pre_key, obs, train=train)
            write = write & present & memory_enabled
            cache, updated, mem_stats, valid = model._encode(obs, fast, write, memory_enabled)
            # Padding must preserve the entire fast-state pytree, for each episode.
            updated = jax.tree.map(
                lambda new, old: jnp.where(present.reshape((1,b)+ (1,)*(new.ndim-2)), new, old), updated, fast,
            )
            mask = mask & present[:, None, None]
            counts = jnp.sum(mask, axis=(-2,-1))
            def supervised(_):
                eps = jax.random.normal(noise_key, target.shape)
                tau = jax.random.beta(time_key, 1.5, 1.0, (b,)) * 0.999 + 0.001
                noisy = jnp.where(mask, tau[:,None,None]*eps + (1-tau[:,None,None])*target, 0)
                velocity = model._velocity(obs, noisy, tau, cache, valid, memory_enabled)
                return masked_flow_action_loss(velocity, eps, target, mask)
            loss, count = jax.lax.cond(jnp.any(counts > 0), supervised, lambda _: (jnp.float32(0), jnp.int32(0)), None)
            return updated, (loss, count)
        _, (losses, counts) = differentiable_scan(
            step, state, (observations, actions, action_mask, write_mask, observation_mask, keys),
            remat=self.config.remat, block_size=self.config.temporal_remat_block_size,
        )
        count = jnp.sum(counts)
        loss = jnp.sum(losses)/jnp.maximum(count, 1)
        return loss, {"action_loss":loss, "supervised_chunks":count,
                      "scheduled_writes":jnp.sum(write_mask & observation_mask) * memory_enabled}

    def compute_loss(self, rng, observation, actions, *, train=False):
        observations = jax.tree.map(lambda x: x[None], observation)
        b = actions.shape[0]
        loss, _ = self.compute_sequence_loss(
            rng, observations, actions[None], jnp.ones_like(actions[None],bool),
            jnp.zeros((1,b),bool), jnp.ones((1,b),bool), train=train, memory_enabled=False,
        )
        return jnp.full((b,self.action_horizon),loss)
