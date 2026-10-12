"""Matched PickXtimes result/provenance tests, authored for server pytest."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("t2mem_comparison", Path(__file__).parents[1] / "scripts/t2mem/compare_results.py")
_comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comparison)


@pytest.fixture
def experiments(tmp_path):
    checkpoint_a = str(tmp_path / "stage1" / "20000")
    checkpoint_b = str(tmp_path / "stage2" / "1000")
    records = [
        {"task": "PickXtimes", "seed": seed, "episode_id": index, "difficulty": difficulty,
         "instruction": "pick up the cube three times", "initial_observation_sha256": f"same-{seed}-{index}",
         "status": "success", "success": True}
        for seed in (7, 17) for index, difficulty in enumerate(("easy", "medium", "hard"))
    ]
    directories = []
    for name, ablation, checkpoint, memory, write in (
        ("A", "memory-free", checkpoint_a, False, True),
        ("B", "full", checkpoint_b, True, True),
        ("C", "no-online-write", checkpoint_b, True, False),
    ):
        directory = tmp_path / name
        directory.mkdir()
        server = {
            "checkpoint": checkpoint,
            "initialization_source": checkpoint_a if memory else "/server/pi05_base/params",
            "checkpoint_progress": {"mode": "alternating" if memory else "stage1", "global_step": 1000 if memory else 20000, "complete": True},
            "ablation": ablation, "memory_enabled": memory, "online_write": write,
            "model_config": {"memory_enabled": memory, "memory": {"online_write": write, "adaptive_write": True}},
            "training_config": {"name": "pickxtimes"}, "approximations": [],
            "preprocessing": {"pipeline": "pipeline:v1", "normalization_sha256": "same-stats",
                              "max_token_len": 200, "action_dim": 32, "robot_action_dim": 8},
            "success_predicate": "terminal-success",
            "action_horizon": 64, "execution_horizon": 32, "memory_write_stride": 16, "flow_denoising_steps": 10,
        }
        manifest = {"tasks": ["PickXtimes"], "seeds": [7, 17], "episodes_per_task": 3,
                    "max_steps": 1300, "simulation_mode": "gpu", "server": server}
        for filename, data in (("manifest.json", manifest), ("metrics.json", {"complete": True, "valid_rollouts": 6}), ("episodes.json", records)):
            (directory / filename).write_text(json.dumps(copy.deepcopy(data)))
        directories.append(directory)
    return directories


def test_compare_exports_matched_easy_medium_hard_overall(experiments, tmp_path):
    output = tmp_path / "comparison"
    report = _comparison.compare(*experiments, output)
    assert report["protocol_validation"] == "matched"
    assert (output / "comparison.csv").read_text().splitlines()[0] == "Model,Easy,Medium,Hard,Overall"
    assert "100.00" in (output / "comparison.md").read_text()
    assert len(report["models"]) == 3


@pytest.mark.parametrize("field,value", [("checkpoint", "/another/checkpoint"), ("preprocessing", "another-pipeline")])
def test_compare_rejects_unmatched_no_write_model(experiments, tmp_path, field, value):
    path = experiments[2] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["server"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _comparison.compare(*experiments, tmp_path / "comparison")


def test_compare_rejects_different_stage1_initialization(experiments, tmp_path):
    for directory in experiments[1:]:
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["server"]["initialization_source"] = "/another/stage1/20000"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="initialized"):
        _comparison.compare(*experiments, tmp_path / "comparison")


def test_compare_rejects_different_target_observation(experiments, tmp_path):
    path = experiments[1] / "episodes.json"
    records = json.loads(path.read_text())
    records[0]["initial_observation_sha256"] = "different"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="target episodes"):
        _comparison.compare(*experiments, tmp_path / "comparison")


@pytest.mark.parametrize("field,value", [
    ("normalization_sha256", "other-stats"), ("max_token_len", 64),
    ("action_dim", 16), ("robot_action_dim", 7),
])
def test_compare_checks_actual_preprocessing_settings(experiments, tmp_path, field, value):
    path = experiments[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["server"]["preprocessing"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="preprocessing"):
        _comparison.compare(*experiments, tmp_path / "comparison")


def test_comparison_smoke_protocol_is_explicitly_not_paper_protocol(experiments, tmp_path):
    report = _comparison.compare(*experiments, tmp_path / "comparison")
    assert report["paper_evaluation_episode_seed_counts_match"] is False
    assert report["paper_evaluation_difficulty_counts_match"] is False
    assert report["paper_evaluation_protocol_match"] is False
