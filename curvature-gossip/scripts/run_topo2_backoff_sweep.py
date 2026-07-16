"""Run topo2 backoff parameter sweeps and summarize curvature-policy gains.

This script expands the base topo2_backoff_v2.yaml into independent experiment
configs for node count, update probability, and curvature weight scans, runs each
config, then writes compact CSV/JSON summaries for comparing curvature gains.
"""

import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml

from curvature_gossip.experiments import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "topo2_backoff_v2.yaml"
GENERATED_CONFIG_DIR = PROJECT_ROOT / "configs" / "generated_sweeps"
RESULT_ROOT = PROJECT_ROOT / "results"
SWEEP_ID = "topo2_backoff_v2_sweep_5seed"

TOPOLOGY_SEEDS = [0, 1, 2, 3, 4]
CHANNEL_SEEDS = [0]
UPDATE_SEEDS = [0]

NODE_VALUES = [80, 100, 120, 150]
UPDATE_PROBABILITY_VALUES = [0.05, 0.1, 0.2, 0.4, 0.6]
CURVATURE_WEIGHT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

BASE_NODES = 100
BASE_UPDATE_PROBABILITY = 0.2
BASE_CURVATURE_WEIGHT = 0.75

LOWER_IS_BETTER = {
    "mean_VAoI",
    "mean_max_VAoI",
    "p95_max_VAoI",
    "mean_tail_VAoI",
    "mean_dissemination_delay",
}


def load_base_config():
    with BASE_CONFIG.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def set_node_count(config, n_nodes):
    """Update topology size fields together so soft_two_community remains valid."""
    params = config["topology"]["params"]
    params["n_nodes"] = n_nodes
    params["cluster_sizes"] = [n_nodes // 2, n_nodes - n_nodes // 2]


def set_curvature_weight(config, weight):
    """Apply the tested curvature weight only to curvature-aware policies."""
    for policy in config["policies"]:
        if policy.get("type") == "curvature_freshness":
            policy.setdefault("params", {})["curvature_weight"] = weight


def build_cases():
    seen = set()
    cases = []

    # One-factor scans keep interpretation clean and avoid a full grid explosion.
    for n_nodes in NODE_VALUES:
        cases.append(("nodes", n_nodes, BASE_UPDATE_PROBABILITY, BASE_CURVATURE_WEIGHT))
    for update_probability in UPDATE_PROBABILITY_VALUES:
        cases.append(("update_probability", BASE_NODES, update_probability, BASE_CURVATURE_WEIGHT))
    for curvature_weight in CURVATURE_WEIGHT_VALUES:
        cases.append(("curvature_weight", BASE_NODES, BASE_UPDATE_PROBABILITY, curvature_weight))

    unique_cases = []
    for scan, n_nodes, update_probability, curvature_weight in cases:
        key = (n_nodes, update_probability, curvature_weight)
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append({
            "scan": scan,
            "n_nodes": n_nodes,
            "update_probability": update_probability,
            "curvature_weight": curvature_weight,
        })
    return unique_cases


def write_case_config(base_config, case):
    config = deepcopy(base_config)
    n_nodes = case["n_nodes"]
    update_probability = case["update_probability"]
    curvature_weight = case["curvature_weight"]

    experiment_id = (
        f"{SWEEP_ID}_{case['scan']}_n{n_nodes}_u{update_probability:g}_cw{curvature_weight:g}"
    )
    config["experiment"]["id"] = experiment_id
    config["experiment"]["topology_seeds"] = TOPOLOGY_SEEDS
    config["experiment"]["channel_seeds"] = CHANNEL_SEEDS
    config["experiment"]["update_seeds"] = UPDATE_SEEDS
    set_node_count(config, n_nodes)
    config["source"]["update_probability"] = update_probability
    set_curvature_weight(config, curvature_weight)

    # Keep core artifacts and summaries, but skip heavy traces/plots for a broad sweep.
    config["output"]["root"] = str(RESULT_ROOT)
    config["output"]["save_per_slot"] = False
    config["output"]["node_diagnostics_stride"] = 0
    config["output"]["make_plots"] = False

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_CONFIG_DIR / f"{experiment_id}.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    return path, experiment_id


def metric_gain(baseline, curvature, metric):
    if baseline is None or curvature is None:
        return None
    if metric in LOWER_IS_BETTER:
        return (baseline - curvature) / baseline if baseline else None
    return (curvature - baseline) / baseline if baseline else None


def read_policy_comparison(experiment_id):
    path = RESULT_ROOT / experiment_id / "policy_comparison.json"
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    return {row["policy"]: row for row in rows}


def case_is_complete(experiment_id):
    return (RESULT_ROOT / experiment_id / "policy_comparison.json").is_file()


def summarize_case(case, experiment_id):
    rows = read_policy_comparison(experiment_id)
    baseline = rows.get("freshness_backoff")
    curvature = rows.get("orc_freshness_backoff")
    if baseline is None or curvature is None:
        raise ValueError(f"missing comparison policy in {experiment_id}")

    summary = {
        "scan": case["scan"],
        "experiment_id": experiment_id,
        "n_nodes": case["n_nodes"],
        "update_probability": case["update_probability"],
        "curvature_weight": case["curvature_weight"],
        "seed_count": curvature.get("independent_runs"),
    }
    for metric in (
        "mean_VAoI",
        "mean_max_VAoI",
        "p95_max_VAoI",
        "mean_tail_VAoI",
        "mean_dissemination_delay",
        "avg_tx_per_slot",
        "successful_decodes_per_tx",
        "innovative_entries_per_tx",
    ):
        summary[f"freshness_{metric}"] = baseline.get(metric)
        summary[f"curvature_{metric}"] = curvature.get(metric)
        summary[f"{metric}_gain"] = metric_gain(baseline.get(metric), curvature.get(metric), metric)
    return summary


def write_summary(rows):
    output_dir = RESULT_ROOT / SWEEP_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_summary.csv"
    json_path = output_dir / "sweep_summary.json"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    return csv_path, json_path


def main():
    base_config = load_base_config()
    summary_rows = []
    for index, case in enumerate(build_cases(), start=1):
        config_path, experiment_id = write_case_config(base_config, case)
        if case_is_complete(experiment_id):
            print(f"[{index}] reusing {experiment_id}")
        else:
            print(f"[{index}] running {experiment_id}")
            run_experiment(str(config_path), overwrite=True, progress=True)
        summary_rows.append(summarize_case(case, experiment_id))
        csv_path, json_path = write_summary(summary_rows)
        print(f"Partial summary saved: {csv_path}")
    print(f"Sweep summary saved: {csv_path}")
    print(f"Sweep summary saved: {json_path}")


if __name__ == "__main__":
    main()
