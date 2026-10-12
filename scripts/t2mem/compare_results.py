"""Validate matched PickXtimes A/B/C evaluations and export CSV/Markdown tables.

Only standard-library JSON/CSV processing is used. It checks the common target
episodes, seeds, initial observations, preprocessing, H64/E32, and provenance.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
from pathlib import Path


def _load(directory: Path) -> tuple[dict, dict, list[dict]]:
    values = tuple(json.loads((directory / name).read_text()) for name in ("manifest.json", "metrics.json", "episodes.json"))
    manifest, metrics, records = values
    if manifest["tasks"] != ["PickXtimes"] or not metrics.get("complete"):
        raise ValueError(f"Expected a complete PickXtimes-only evaluation: {directory}")
    if any(record["status"] == "error" or record["success"] is None for record in records):
        raise ValueError(f"Infrastructure errors must be resolved before comparison: {directory}")
    keys = [(record["task"], record["seed"], record["episode_id"]) for record in records]
    expected = len(manifest["seeds"]) * manifest["episodes_per_task"]
    expected_keys = {("PickXtimes", seed, episode) for seed in manifest["seeds"] for episode in range(manifest["episodes_per_task"])}
    if len(keys) != len(set(keys)) or len(keys) != expected or set(keys) != expected_keys:
        raise ValueError(f"Duplicate or missing evaluation episodes: {directory}")
    if metrics["valid_rollouts"] != len(records):
        raise ValueError(f"Result summary disagrees with per-episode records: {directory}")
    return values


def _source_checkpoint(source) -> str:
    if isinstance(source, dict):
        source = source.get("checkpoint", source.get("path"))
    if not isinstance(source, str) or not source:
        raise ValueError("Full T²Mem checkpoint is missing its saved Stage-1 initialization source.")
    return str(Path(source).expanduser().resolve())


def compare(baseline: Path, full: Path, no_write: Path, output_dir: Path) -> dict:
    evaluations = [_load(path) for path in (baseline, full, no_write)]
    manifests = [item[0] for item in evaluations]
    servers = [manifest["server"] for manifest in manifests]
    for field in ("tasks", "seeds", "episodes_per_task", "max_steps", "simulation_mode"):
        if not all(manifest[field] == manifests[0][field] for manifest in manifests):
            raise ValueError(f"A/B/C evaluation protocols differ in {field}.")
    for field in ("preprocessing", "success_predicate", "action_horizon", "execution_horizon", "memory_write_stride", "flow_denoising_steps"):
        if not all(server.get(field) == servers[0].get(field) for server in servers) or servers[0].get(field) is None:
            raise ValueError(f"A/B/C serving protocols differ or omit {field}.")
    if servers[0]["action_horizon"] != 64 or servers[0]["execution_horizon"] != 32:
        raise ValueError("Comparison requires paper H64/E32.")
    expected_ablations = ("memory-free", "full", "no-online-write")
    if tuple(server["ablation"] for server in servers) != expected_ablations:
        raise ValueError("Expected Experiment A memory-free, Experiment B full, and Experiment C no-online-write.")
    if servers[0]["memory_enabled"] or not servers[1]["memory_enabled"] or not servers[2]["memory_enabled"]:
        raise ValueError("A must disable memory; B/C must preserve the memory pathway.")
    if not servers[1]["online_write"] or servers[2]["online_write"]:
        raise ValueError("B must enable all writes; C must disable all writes.")
    checkpoint_a, checkpoint_b, checkpoint_c = (str(Path(server["checkpoint"]).resolve()) for server in servers)
    if checkpoint_b != checkpoint_c:
        raise ValueError("Full and No-Write must load the exact same Experiment B checkpoint.")
    if _source_checkpoint(servers[1].get("initialization_source")) != checkpoint_a:
        raise ValueError("Experiment B was not initialized from the evaluated Experiment A checkpoint.")
    if servers[0]["checkpoint_progress"]["mode"] != "stage1":
        raise ValueError("Experiment A must be our independently trained Stage-1 baseline.")
    if any(server["checkpoint_progress"]["mode"] != "alternating" for server in servers[1:]):
        raise ValueError("Experiment B/C must use the alternating Stage-2 training run.")
    if not all(server["checkpoint_progress"].get("complete", False) for server in servers):
        raise ValueError("Finish each configured training budget before reporting the A/B/C comparison.")
    if servers[1]["checkpoint_progress"] != servers[2]["checkpoint_progress"]:
        raise ValueError("Full and No-Write checkpoint progress metadata differs.")
    model_b, model_c = (copy.deepcopy(server["model_config"]) for server in servers[1:])
    model_b["memory"]["online_write"] = model_c["memory"]["online_write"] = True
    if model_b != model_c or servers[1]["training_config"] != servers[2]["training_config"]:
        raise ValueError("Full and No-Write differ beyond the inference-time write switch.")
    reference = {
        (record["task"], record["seed"], record["episode_id"]):
        (record["difficulty"], record["instruction"], record["initial_observation_sha256"])
        for record in evaluations[0][2]
    }
    for _, _, records in evaluations[1:]:
        targets = {
            (record["task"], record["seed"], record["episode_id"]):
            (record["difficulty"], record["instruction"], record["initial_observation_sha256"])
            for record in records
        }
        if targets != reference:
            raise ValueError("A/B/C target episodes, instructions, difficulties, seeds, or initial observations differ.")
    names = ("π0.5 Fine-tuned", "Full T²Mem", "T²Mem No-Write")
    rows, details = [], []
    for name, (manifest, metrics, records), server in zip(names, evaluations, servers, strict=True):
        def rate(difficulty):
            group = [record for record in records if difficulty is None or record["difficulty"] == difficulty]
            return 100 * sum(record["success"] for record in group) / len(group) if group else None

        rows.append([name, *(rate(difficulty) for difficulty in ("easy", "medium", "hard", None))])
        details.append({
            "model": name,
            "checkpoint": server["checkpoint"],
            "initialization_source": server.get("initialization_source"),
            "seeds": manifest["seeds"],
            "checkpoint_progress": server["checkpoint_progress"],
            "training_config": server["training_config"],
            "approximations": server["approximations"],
            "rollouts": metrics["valid_rollouts"],
            "ablation": server["ablation"],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ("Model", "Easy", "Medium", "Hard", "Overall")
    with (output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows([[row[0], *(f"{value:.2f}" if value is not None else "N/A" for value in row[1:])] for row in rows])
    paper_episode_seed_counts_match = manifests[0]["episodes_per_task"] == 50 and len(manifests[0]["seeds"]) == 3
    paper_difficulty_counts_match = all(
        dict(collections.Counter(record["difficulty"] for record in records if record["seed"] == seed))
        == {"easy": 26, "medium": 12, "hard": 12}
        for manifest, _, records in evaluations for seed in manifest["seeds"]
    )
    paper_write_stride_match = servers[0]["memory_write_stride"] == 16
    paper_protocol_match = paper_episode_seed_counts_match and paper_difficulty_counts_match and paper_write_stride_match
    if paper_protocol_match:
        protocol_label = "paper evaluation: 50 episodes × 3 seeds, 26 Easy/12 Medium/12 Hard, stride16"
    elif not paper_episode_seed_counts_match:
        protocol_label = "reduced evaluation protocol (smoke/diagnostic)"
    else:
        protocol_label = "evaluation differs from paper difficulty counts or write stride"
    lines = [f"PickXtimes single-task reproduction; {protocol_label}. Success rates (%).", "", "| " + " | ".join(columns) + " |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([row[0], *(f"{value:.2f}" if value is not None else "N/A" for value in row[1:])]) + " |")
    lines.extend(["", "All rows use matched test episodes, initial observations, evaluation seeds, preprocessing and H64/E32.", ""])
    for detail in details:
        lines.extend([
            f"- {detail['model']}: checkpoint `{detail['checkpoint']}`; initialization `{detail['initialization_source']}`; "
            f"optimizer steps {detail['checkpoint_progress']['global_step']}; seeds {detail['seeds']}; "
            f"approximations {detail['approximations'] or 'none'}.",
        ])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    report = {"experiment": "single-task reproduction", "protocol_validation": "matched",
              "paper_evaluation_episode_seed_counts_match": paper_episode_seed_counts_match,
              "paper_evaluation_difficulty_counts_match": paper_difficulty_counts_match,
              "paper_evaluation_write_stride_match": paper_write_stride_match,
              "paper_evaluation_protocol_match": paper_protocol_match, "models": details}
    (output_dir / "comparison_metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--no-write", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.baseline, args.full, args.no_write, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
