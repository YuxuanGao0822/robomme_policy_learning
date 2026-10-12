"""Run on the server only: pytest tests/test_t2mem_policy.py.

These protocol tests use a tiny JAX state, avoiding pretrained model downloads.
They are authored but intentionally not executed in the local Mac workflow.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.t2mem.policy import PROTOCOL
from mme_vla_suite.models.t2mem.policy import T2MemPolicy
from mme_vla_suite.models.t2mem.policy import WriteSchedule


class SmallModel:
    action_horizon = 64

    def __init__(self, online_write=True):
        self.config = SimpleNamespace(memory_enabled=True, memory=SimpleNamespace(online_write=online_write))
        self.sample_calls = 0
        self.observe_calls = 0
        self.denoising_reads = []

    def initial_memory_state(self, batch_size=1):
        return {"value": jnp.zeros((batch_size,))}

    def observe_memory(self, observation, state, *, write_mask=True):
        self.observe_calls += 1
        return {"value": state["value"] + int(write_mask and self.config.memory.online_write)}

    def sample_actions_with_memory(self, rng, observation, state, *, num_steps=10, write=False):
        self.sample_calls += 1
        self.denoising_reads.extend([np.asarray(state["value"]).copy() for _ in range(num_steps)])
        actions = jnp.broadcast_to(state["value"][:, None, None], (1, 64, 8))
        return actions, {"value": state["value"] + int(write and self.config.memory.online_write)}


def input_transform(raw):
    return {
        "image": {"base_0_rgb": raw["observation/image"]},
        "image_mask": {"base_0_rgb": True},
        "state": raw["observation/state"],
    }


def raw(prompt="target instruction"):
    return {"observation/image": np.zeros((8, 8, 3), np.uint8), "observation/state": np.zeros(8), "prompt": prompt}


def request(op, episode="0", **payload):
    control = payload.pop("control", {})
    return {
        "_t2mem": {"protocol": PROTOCOL, "op": op, "env_id": "env0", "task": "PickXtimes", "episode_id": episode, **control},
        **payload,
    }


def make_policy(online_write=True):
    model = SmallModel(online_write)
    return T2MemPolicy(model, transforms_in=(input_transform,), jit=False), model


def start(policy, episode="0", owner="local"):
    return policy.handle(request("start", episode, prompt="target instruction", seed=7), owner=owner)


def infer(policy, index, episode="0", owner="local"):
    return policy.handle(request("infer", episode, control={"index": index}, observation=raw()), owner=owner)


def observe(policy, phase, indices, episode="0", owner="local"):
    return policy.handle(request("observe", episode, control={"phase": phase}, frames=[{"index": index, "observation": raw()} for index in indices]), owner=owner)


def test_read_before_write_and_denoising_does_not_repeat_write():
    policy, model = make_policy()
    start(policy)
    output = infer(policy, 0)
    np.testing.assert_array_equal(output["actions"], 0)
    assert len(model.denoising_reads) == 10
    assert all(np.array_equal(value, [0]) for value in model.denoising_reads)
    assert model.sample_calls == 1
    assert infer(policy, 0) is output
    assert model.sample_calls == 1
    observe(policy, "execution", [16])
    np.testing.assert_array_equal(infer(policy, 32)["actions"], 2)
    assert policy.handle(request("end"))["writes"] == 3


def test_demo_prefix_is_chronological_incremental_and_idempotent():
    policy, model = make_policy()
    start(policy)
    assert observe(policy, "demo", list(range(34)))["encoded"] == 3
    assert observe(policy, "demo", list(range(34)))["encoded"] == 0
    assert model.observe_calls == 3
    np.testing.assert_array_equal(infer(policy, 0)["actions"], 3)
    with pytest.raises(ValueError, match="after execution"):
        observe(policy, "demo", [40])


def test_reset_episode_and_connections_cannot_leak_memory():
    policy, _ = make_policy()
    start(policy, owner="connection-a")
    observe(policy, "demo", [0], owner="connection-a")
    start(policy, owner="connection-b")
    np.testing.assert_array_equal(infer(policy, 0, owner="connection-b")["actions"], 0)
    np.testing.assert_array_equal(infer(policy, 0, owner="connection-a")["actions"], 1)
    start(policy, episode="new", owner="connection-a")
    np.testing.assert_array_equal(infer(policy, 0, episode="new", owner="connection-a")["actions"], 0)
    with pytest.raises(ValueError, match="Start the episode"):
        infer(policy, 0, owner="connection-a")
    policy.close_owner("connection-a")
    assert all(key[0] != "connection-a" for key in policy._sessions)
    assert any(key[0] == "connection-b" for key in policy._sessions)


def test_terminal_cleanup_and_no_write_control_keep_learned_initialization():
    policy, _ = make_policy(online_write=False)
    start(policy)
    observe(policy, "demo", [0, 16])
    np.testing.assert_array_equal(infer(policy, 0)["actions"], 0)
    assert policy.handle(request("end"))["writes"] == 0
    assert not policy._sessions


def test_missing_midpoint_and_boundary_observe_are_rejected():
    policy, _ = make_policy()
    start(policy)
    infer(policy, 0)
    with pytest.raises(ValueError, match="Missing scheduled"):
        infer(policy, 32)
    with pytest.raises(ValueError, match="read-before-write"):
        observe(policy, "execution", [32])
    observe(policy, "execution", [16])
    infer(policy, 32)


def test_schedule_uses_physical_timelines_and_explicit_diagnostic_offset():
    schedule = WriteSchedule()
    assert [index for index in range(33) if schedule.should_write("execution", index)] == [0, 16, 32]
    diagnostic = WriteSchedule(stride=4, demo_offset=2)
    assert [index for index in range(34) if diagnostic.should_write("demo", index)] == list(range(2, 34, 4))


def test_observation_instruction_cannot_change_target_goal():
    policy, _ = make_policy()
    start(policy)
    with pytest.raises(ValueError, match="target episode instruction"):
        policy.handle(request("observe", control={"phase": "demo"}, frames=[{"index": 0, "observation": raw("other instruction")}]))


def test_pinned_tonnx_param_only_checkpoint_matches_serving_load(tmp_path):
    """Exercise the exact ToNNX RNG ownership and inherited config.load path."""
    from flax import linen
    from flax import nnx
    from flax.nnx import bridge
    import jax
    import orbax.checkpoint as ocp

    from openpi.models.model import BaseModelConfig
    from openpi.models.model import restore_params

    class TinyLinen(linen.Module):
        @linen.compact
        def __call__(self, inputs):
            return linen.Dense(4, name="projection")(inputs)

    def create(rng):
        module = bridge.ToNNX(TinyLinen())
        module.lazy_init(jnp.zeros((1, 4)), rngs=nnx.Rngs(rng))
        return module

    module = create(jax.random.key(0))
    _, state = nnx.split(module)
    assert not state.filter(nnx.Not(nnx.Param)).flat_state()
    path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, args=ocp.args.PyTreeSave({"params": jax.device_get(state.filter(nnx.Param).to_pure_dict())}))
    # BaseModelConfig.load only needs create(). The tiny bridge is the same kind
    # of parameter-only module used by the full VLM and image encoder.
    config = SimpleNamespace(create=create)
    restored = BaseModelConfig.load(config, restore_params(path))
    inputs = jnp.ones((1, 4))
    np.testing.assert_allclose(restored(inputs), module(inputs), rtol=1e-6, atol=1e-6)
