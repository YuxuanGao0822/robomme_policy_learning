"""Incremental episode-local T²Mem serving, with physical-step write scheduling.

The server keeps fast weights; clients send a demonstration prefix once and then
only replanning/midpoint observations. Denoising and write ordering live in the
model's ``sample_actions_with_memory`` implementation, not in an RPC loop.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import logging
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.frames

from openpi import transforms
from mme_vla_suite.models.t2mem.observation import T2Observation as Observation
from openpi.shared import nnx_utils

from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer

TARGET_TASKS = ("PickXtimes",)
PROTOCOL = "t2mem.v1"


@dataclasses.dataclass(frozen=True)
class WriteSchedule:
    """Demo indices and execution steps are separate physical timelines.

    Appendix B specifies stride but does not specify the default demo phase
    offset. Offset zero is an explicit engineering choice, shared with training.
    The diagnostic stride-4 frame-2 schedule can be selected separately.
    """

    stride: int = 16
    execution_horizon: int = 32
    demo_offset: int = 0

    def __post_init__(self) -> None:
        if self.stride <= 0 or self.execution_horizon <= 0:
            raise ValueError("Write stride and execution horizon must be positive.")
        if self.execution_horizon % self.stride:
            raise ValueError("Execution horizon must be divisible by memory write stride.")
        if not 0 <= self.demo_offset < self.stride:
            raise ValueError("Demo write offset must lie in [0, stride).")

    def should_write(self, phase: str, index: int) -> bool:
        offset = self.demo_offset if phase == "demo" else 0
        return index >= offset and (index - offset) % self.stride == 0


@dataclasses.dataclass
class EpisodeSession:
    fast_state: Any
    rng: Any
    prompt: str
    demo_last: int = -1
    execution_last: int = -1
    last_infer: int = -1
    cached_output: dict[str, Any] | None = None
    observations_encoded: int = 0
    writes: int = 0


class T2MemPolicy:
    """Slow parameters are shared; (connection, env, task, episode) fast states are isolated."""

    def __init__(
        self,
        model,
        *,
        transforms_in: Sequence[transforms.DataTransformFn] = (),
        transforms_out: Sequence[transforms.DataTransformFn] = (),
        seed: int = 7,
        schedule: WriteSchedule | None = None,
        num_steps: int = 10,
        max_sessions: int = 8,
        metadata: dict[str, Any] | None = None,
        jit: bool = True,
    ) -> None:
        self._model = model
        self._seed = seed
        self._schedule = schedule or WriteSchedule()
        self._num_steps = num_steps
        self._max_sessions = max_sessions
        self._input_transform = transforms.compose(transforms_in)
        self._output_transform = transforms.compose(transforms_out)
        self._sessions: dict[tuple[str, str, str, str], EpisodeSession] = {}
        self._metadata = {
            **(metadata or {}),
            "protocol": PROTOCOL,
            "experiment": "single-task reproduction",
            "tasks": list(TARGET_TASKS),
            "action_horizon": int(model.action_horizon),
            "execution_horizon": self._schedule.execution_horizon,
            "memory_write_stride": self._schedule.stride,
            "demo_write_offset": self._schedule.demo_offset,
        }
        self._sample = model.sample_actions_with_memory
        self._observe = model.observe_memory
        if jit:
            self._sample = nnx_utils.module_jit(self._sample, static_argnames=("num_steps", "write"))
            self._observe = nnx_utils.module_jit(self._observe)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _key(self, request: dict, owner: str) -> tuple[str, str, str, str]:
        control = request.get("_t2mem", {})
        if control.get("protocol") != PROTOCOL:
            raise ValueError(f"Expected stateful protocol {PROTOCOL}.")
        values = tuple(str(control.get(field, "")) for field in ("env_id", "task", "episode_id"))
        if not all(values) or values[1] not in TARGET_TASKS:
            raise ValueError("env_id, a supported task, and episode_id are required.")
        return (owner, *values)

    def _prepare(self, raw: dict, session: EpisodeSession) -> Observation:
        if raw.get("prompt") != session.prompt:
            raise ValueError("Observation instruction differs from the target episode instruction.")
        # Copy transforms' mutable dictionaries; leave image storage shared until transformed.
        inputs = self._input_transform(dict(raw))
        inputs = jax.tree.map(lambda x: None if x is None else jnp.asarray(x)[None, ...], inputs)
        return Observation.from_dict(inputs)

    def close_owner(self, owner: str) -> None:
        for key in list(self._sessions):
            if key[0] == owner:
                del self._sessions[key]

    def handle(self, request: dict, *, owner: str = "local") -> dict[str, Any]:
        key = self._key(request, owner)
        control = request["_t2mem"]
        op = control.get("op")
        if op == "start":
            if not isinstance(request.get("prompt"), str):
                raise ValueError("Episode start requires a target instruction.")
            # Reset any prior episode for the same connection/environment, never another owner.
            for previous in list(self._sessions):
                if previous[:2] == key[:2]:
                    del self._sessions[previous]
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError("Active episode capacity exceeded; end old episodes first.")
            self._sessions[key] = EpisodeSession(
                fast_state=self._model.initial_memory_state(batch_size=1),
                rng=jax.random.key(int(request.get("seed", self._seed))),
                prompt=request["prompt"],
            )
            return {"reset_finished": True, "protocol": PROTOCOL}
        if op == "end":
            session = self._sessions.pop(key, None)
            return {"end_finished": True, "writes": session.writes if session else 0}
        if key not in self._sessions:
            raise ValueError("Start the episode before sending observations or inference requests.")
        session = self._sessions[key]
        if op == "observe":
            return self._observe_frames(request, session)
        if op == "infer":
            return self._infer(request, session)
        raise ValueError(f"Unsupported T²Mem operation: {op!r}.")

    def _observe_frames(self, request: dict, session: EpisodeSession) -> dict[str, Any]:
        phase = request["_t2mem"].get("phase")
        if phase not in ("demo", "execution"):
            raise ValueError("Observation phase must be demo or execution.")
        if phase == "demo" and session.execution_last >= 0:
            raise ValueError("Demonstration observations cannot arrive after execution begins.")
        encoded = 0
        previous_in_request = -1
        for frame in request.get("frames", []):
            index = int(frame["index"])
            if index < 0 or index <= previous_in_request:
                raise ValueError("Frame indices must be nonnegative and strictly chronological.")
            previous_in_request = index
            last = session.demo_last if phase == "demo" else session.execution_last
            if index <= last:
                continue  # Idempotent delivery: no re-encoding and no duplicate write.
            if phase == "execution":
                if index % self._schedule.execution_horizon == 0:
                    raise ValueError("Replanning observations must use infer to preserve read-before-write.")
                self._check_execution_gap(session, index)
            if self._schedule.should_write(phase, index):
                observation = self._prepare(frame["observation"], session)
                session.fast_state = self._observe(observation, session.fast_state, write_mask=True)
                jax.tree.map(lambda x: x.block_until_ready(), session.fast_state)
                session.observations_encoded += 1
                session.writes += int(self._writes_enabled())
                encoded += 1
            if phase == "demo":
                session.demo_last = index
            else:
                session.execution_last = index
        return {"observe_finished": True, "encoded": encoded, "writes": session.writes}

    def _writes_enabled(self) -> bool:
        return bool(self._model.config.memory_enabled and self._model.config.memory.online_write)

    def _check_execution_gap(self, session: EpisodeSession, index: int) -> None:
        # A client must deliver every scheduled midpoint, rather than silently
        # dropping observations and calling the result a stride-16 policy.
        first = max(0, session.execution_last + 1)
        missed = [i for i in range(first, index) if self._schedule.should_write("execution", i)]
        if missed:
            raise ValueError(f"Missing scheduled execution observations at physical steps {missed}.")

    def _infer(self, request: dict, session: EpisodeSession) -> dict[str, Any]:
        index = int(request["_t2mem"]["index"])
        if index < 0 or index % self._schedule.execution_horizon:
            raise ValueError("Inference index must be an execution chunk boundary in physical steps.")
        if index == session.last_infer and session.cached_output is not None:
            return session.cached_output
        if index <= session.execution_last:
            raise ValueError("Inference observation was already processed, violating read-before-write.")
        self._check_execution_gap(session, index)
        observation = self._prepare(request["observation"], session)
        next_rng, sample_rng = jax.random.split(session.rng)
        start = time.monotonic()
        write = self._schedule.should_write("execution", index)
        actions, next_state = self._sample(
            sample_rng, observation, session.fast_state, num_steps=self._num_steps, write=write
        )
        output = self._output_transform({"state": np.asarray(observation.state[0]), "actions": np.asarray(actions[0])})
        jax.tree.map(lambda x: x.block_until_ready(), next_state)
        output = {**output, "infer_time_ms": (time.monotonic() - start) * 1000, "execution_step": index}
        session.fast_state = next_state
        session.rng = next_rng
        session.execution_last = index
        session.last_infer = index
        session.cached_output = output
        session.observations_encoded += 1
        session.writes += int(write and self._writes_enabled())
        return output


class T2MemWebsocketPolicyServer(WebsocketPolicyServer):
    """Reuse OpenPI msgpack transport/healthcheck with per-connection cleanup."""

    async def _handler(self, websocket) -> None:
        owner = str(id(websocket))
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        try:
            async for message in websocket:
                request = msgpack_numpy.unpackb(message)
                try:
                    # Synchronous model work is deliberately serialized on the single GPU.
                    response = self._policy.handle(request, owner=owner)
                    await websocket.send(packer.pack(response))
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="T²Mem request failed; traceback sent in previous frame.",
                    )
                    break
        except websockets.ConnectionClosed:
            logging.info("T²Mem client disconnected: %s", owner)
        finally:
            self._policy.close_owner(owner)
