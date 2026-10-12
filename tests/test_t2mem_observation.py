"""Shared A/B/C preprocessing and proprioception routing tests for the server."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.t2mem.model import T2Mem
from mme_vla_suite.models.t2mem.observation import T2Observation
from mme_vla_suite.models.t2mem.observation import TokenizeT2MemPrompt


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, state=None):
        self.calls.append((prompt, state))
        # Distinct token sequences model the native state-containing prompt and
        # instruction-only prompt without loading a real tokenizer/checkpoint.
        tokens = np.array([1, 2, 3, 0] if state is None else [1, 7, 8, 9], dtype=np.int32)
        return tokens, tokens != 0


def sample_dict():
    return {
        "prompt": np.array("Pick up the blue cube three times."),
        "state": np.zeros((8,), dtype=np.float32),
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "actions": np.ones((64, 8), dtype=np.float32),
    }


def test_every_experiment_gets_both_prompt_variants_and_identical_robot_preprocessing():
    tokenizer = RecordingTokenizer()
    original = sample_dict()
    transformed = TokenizeT2MemPrompt(tokenizer)(original)
    assert "prompt" in original
    assert "prompt" not in transformed
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[0][1] is original["state"]
    assert tokenizer.calls[1][1] is None
    assert tokenizer.calls[0][0] == tokenizer.calls[1][0]
    for key in ("state", "image", "image_mask", "actions"):
        assert transformed[key] is original[key]
    np.testing.assert_array_equal(transformed["tokenized_pi05_prompt"], [1, 7, 8, 9])
    np.testing.assert_array_equal(transformed["tokenized_prompt"], [1, 2, 3, 0])


def test_observation_preserves_both_prompt_masks_as_pytree_fields():
    transformed = TokenizeT2MemPrompt(RecordingTokenizer())(sample_dict())
    observation = T2Observation.from_dict(transformed)
    np.testing.assert_array_equal(observation.tokenized_pi05_prompt, transformed["tokenized_pi05_prompt"])
    np.testing.assert_array_equal(observation.tokenized_pi05_prompt_mask, transformed["tokenized_pi05_prompt_mask"])
    np.testing.assert_array_equal(observation.tokenized_prompt, transformed["tokenized_prompt"])
    roundtrip = observation.to_dict()
    np.testing.assert_array_equal(roundtrip["state"], transformed["state"])
    np.testing.assert_array_equal(roundtrip["tokenized_pi05_prompt"], transformed["tokenized_pi05_prompt"])


def test_native_prompt_tokens_and_mask_must_be_supplied_together():
    transformed = TokenizeT2MemPrompt(RecordingTokenizer())(sample_dict())
    transformed.pop("tokenized_pi05_prompt_mask")
    with pytest.raises(ValueError, match="together"):
        T2Observation.from_dict(transformed)


def test_baseline_routes_state_prompt_and_memory_interface_routes_instruction_only():
    transformed = TokenizeT2MemPrompt(RecordingTokenizer())(sample_dict())
    observation = T2Observation.from_dict(transformed)
    selected = []

    def embed_prefix(obs):
        selected.append(obs.tokenized_prompt)
        return jnp.zeros((1, 2, 64)), jnp.ones((1, 2), dtype=bool), None

    def transformer(*args, **kwargs):
        return "cache", "next_state", {}

    model = SimpleNamespace(embed_prefix=embed_prefix, PaliGemma=SimpleNamespace(llm=transformer))
    T2Mem._encode(model, observation, "state", False, memory_enabled=False)
    T2Mem._encode(model, observation, "state", False, memory_enabled=True)
    np.testing.assert_array_equal(selected[0], observation.tokenized_pi05_prompt)
    np.testing.assert_array_equal(selected[1], observation.tokenized_prompt)
    # Routing leaves the shared original observation unchanged for experiment C.
    np.testing.assert_array_equal(observation.tokenized_prompt, [1, 2, 3, 0])


def test_image_preprocessing_retains_extended_prompt_fields():
    observation = T2Observation(
        images={"base_0_rgb": jnp.zeros((1, 224, 224, 3))},
        image_masks={"base_0_rgb": jnp.ones((1,), dtype=bool)},
        state=jnp.zeros((1, 8)),
        tokenized_prompt=jnp.array([[1, 2, 3, 0]]),
        tokenized_prompt_mask=jnp.array([[True, True, True, False]]),
        tokenized_pi05_prompt=jnp.array([[1, 7, 8, 9]]),
        tokenized_pi05_prompt_mask=jnp.ones((1, 4), dtype=bool),
    )
    processed = T2Mem._preprocess(SimpleNamespace(), None, observation, train=False)
    assert isinstance(processed, T2Observation)
    np.testing.assert_array_equal(processed.tokenized_pi05_prompt, observation.tokenized_pi05_prompt)
    np.testing.assert_array_equal(processed.tokenized_prompt, observation.tokenized_prompt)
    np.testing.assert_array_equal(processed.state, observation.state)
