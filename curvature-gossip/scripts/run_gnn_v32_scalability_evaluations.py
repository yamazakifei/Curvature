"""Run the V3.2 checkpoint on the existing N- and u-scalability scenarios.

功能：复用 0804Scalability_N 和 0804Scalability_u 中已经生成的测试 YAML、
seed 和验证时长，只替换曲率模型为 V3.2 checkpoint，并分别写入
0806ScalabilityV3.2_N 与 0806ScalabilityV3.2_u。结果保留无曲率和
matched-random 对照，并在汇总 CSV 中增加平均广播概率及其 95% 置信区间。
脚本只做推理，不训练或修改任何模型目录。
"""

from __future__ import division

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result_GNN"
V32_MODEL_DIR = RESULT_ROOT / "mpnnV3.2_search_ch1_n100_u0.20_Bmax0.10_stage1_no_center"
SOURCE_N_ROOT = RESULT_ROOT / "0804Scalability_N"
SOURCE_U_ROOT = RESULT_ROOT / "0804Scalability_u"
OUTPUT_N_ROOT = RESULT_ROOT / "0806ScalabilityV3.2_N"
OUTPUT_U_ROOT = RESULT_ROOT / "0806ScalabilityV3.2_u"

SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def copy_config_tree(source_root, output_root, prefix):
    """Copy existing evaluation YAMLs and return topology/key path mappings."""
    config_paths = {}
    for topology_name in ("community", "random"):
        source_dir = source_root / "configs" / topology_name
        output_dir = output_root / "configs" / topology_name
        output_dir.mkdir(parents=True, exist_ok=True)
        config_paths[topology_name] = {}
        for source_path in sorted(source_dir.glob("{}*.yaml".format(prefix))):
            target_path = output_dir / source_path.name
            shutil.copy2(str(source_path), str(target_path))
            key_text = source_path.stem[len(prefix):]
            key = int(key_text) if prefix == "n" else float(key_text) / 100.0
            config_paths[topology_name][key] = target_path
    return config_paths


def configure_model_dirs(scalability_module, update_module):
    """Use V3.2 for both topology evaluations and retain existing baselines."""
    model_dirs = {
        "community": {
            "curvature": V32_MODEL_DIR,
            "no_curvature": RESULT_ROOT / "mpnnV3_no_curvature_heuristic_channel_n100_u0.20_b0.10",
        },
        "random": {
            "curvature": V32_MODEL_DIR,
            "no_curvature": RESULT_ROOT / "mpnnV3_no_curvature_random_n100_u0.20_b0.10",
        },
    }
    scalability_module.MODEL_DIRS = model_dirs
    update_module.MODEL_DIRS = model_dirs
    # Make the generated figures identify the new checkpoint explicitly.
    for module in (scalability_module, update_module):
        module.PLOT_LABELS = dict(module.PLOT_LABELS)
        module.PLOT_LABELS[("curvature", "nn")] = "V3.2 curvature MPNN"
        module.PLOT_LABELS[("curvature", "matched_random")] = "V3.2 matched random"
    return model_dirs


def serialize_model_dirs(model_dirs):
    """Convert nested Path mappings into readable JSON metadata."""
    return {
        topology: {model: str(path) for model, path in models.items()}
        for topology, models in model_dirs.items()
    }


def write_csv(path, rows):
    """Write rows while preserving the first-seen column order."""
    if not rows:
        raise ValueError("cannot write empty CSV: {}".format(path))
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std_ci(values):
    """Return mean, sample standard deviation, and normal-approximation CI."""
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(len(values))
    else:
        std = 0.0
        ci95 = 0.0
    return mean, std, ci95


def add_broadcast_probability_summary(summary_rows, scenario_rows, keys):
    """Add mean broadcast probability statistics to an existing summary."""
    grouped = defaultdict(list)
    for row in scenario_rows:
        grouped[tuple(row[key] for key in keys)].append(row["action_prob_mean"])
    for row in summary_rows:
        group = grouped[tuple(row[key] for key in keys)]
        mean, std, ci95 = mean_std_ci(group)
        row["mean_action_probability"] = mean
        row["std_action_probability"] = std
        row["ci95_action_probability"] = ci95
    return summary_rows


def run_node_scalability(scalability_module, model_dirs):
    """Evaluate V3.2 on the copied N=50..150 scenarios."""
    OUTPUT_N_ROOT.mkdir(parents=True, exist_ok=True)
    config_paths = copy_config_tree(SOURCE_N_ROOT, OUTPUT_N_ROOT, "n")
    scenario_rows, provenance = scalability_module.evaluate_all(
        config_paths, ("community", "random")
    )
    summary_rows = scalability_module.summarize_rows(scenario_rows)
    summary_rows = add_broadcast_probability_summary(
        summary_rows, scenario_rows, ("topology", "node_count", "model", "policy")
    )
    write_csv(OUTPUT_N_ROOT / "scalability_per_scenario.csv", scenario_rows)
    write_csv(OUTPUT_N_ROOT / "scalability_summary.csv", summary_rows)
    plot_paths = {
        topology: scalability_module.plot_topology(
            summary_rows, topology, OUTPUT_N_ROOT
        )
        for topology in ("community", "random")
    }
    metadata = {
        "evaluation": "V3.2 checkpoint on existing N-scalability scenarios",
        "source_config_root": str(SOURCE_N_ROOT),
        "models": provenance,
        "configs": {
            topology: [str(path) for path in paths.values()]
            for topology, paths in config_paths.items()
        },
        "plots": {
            topology: [str(path) for path in paths]
            for topology, paths in plot_paths.items()
        },
        "node_counts": list(scalability_module.NODE_COUNTS),
        "scenarios_per_node_count": scalability_module.SCENARIOS_PER_NODE_COUNT,
        "slots_per_scenario": 200,
        "evaluation_master_seed": scalability_module.EVALUATION_MASTER_SEED,
        "density_scaling": "reused from 0804Scalability_N",
        "model_dirs": serialize_model_dirs(model_dirs),
    }
    with (OUTPUT_N_ROOT / "scalability_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)


def run_update_probability(update_module, scalability_module, model_dirs):
    """Evaluate V3.2 on the existing u=0.05..0.30 scenarios."""
    OUTPUT_U_ROOT.mkdir(parents=True, exist_ok=True)
    config_paths = copy_config_tree(SOURCE_U_ROOT, OUTPUT_U_ROOT, "u")
    scenario_rows, provenance = update_module.evaluate_all(
        config_paths, ("community", "random"), 500
    )
    summary_rows = update_module.summarize_rows(scenario_rows)
    summary_rows = add_broadcast_probability_summary(
        summary_rows,
        scenario_rows,
        ("topology", "update_probability", "model", "policy"),
    )
    write_csv(OUTPUT_U_ROOT / "update_probability_per_scenario.csv", scenario_rows)
    write_csv(OUTPUT_U_ROOT / "update_probability_summary.csv", summary_rows)
    plot_paths = {
        topology: update_module.plot_topology(
            summary_rows, topology, OUTPUT_U_ROOT
        )
        for topology in ("community", "random")
    }
    metadata = {
        "evaluation": "V3.2 checkpoint on existing update-probability scenarios",
        "source_config_root": str(SOURCE_U_ROOT),
        "models": provenance,
        "configs": {
            topology: [str(path) for path in paths.values()]
            for topology, paths in config_paths.items()
        },
        "plots": {
            topology: [str(path) for path in paths]
            for topology, paths in plot_paths.items()
        },
        "node_count": update_module.NODE_COUNT,
        "update_probabilities": list(update_module.UPDATE_PROBABILITIES),
        "scenarios_per_update_probability": update_module.SCENARIOS_PER_UPDATE_PROBABILITY,
        "slots_per_scenario": 500,
        "evaluation_master_seed": update_module.EVALUATION_MASTER_SEED,
        "model_dirs": serialize_model_dirs(model_dirs),
    }
    with (OUTPUT_U_ROOT / "update_probability_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)


def read_summary_rows(path, numeric_fields, integer_fields=()):
    """Read a generated summary CSV and restore numeric columns for plotting."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
        for field in integer_fields:
            row[field] = int(row[field])
    return rows


def refresh_plots_and_metadata(scalability_module, update_module, model_dirs):
    """Refresh labels and metadata without repeating simulator evaluations."""
    node_summary = read_summary_rows(
        OUTPUT_N_ROOT / "scalability_summary.csv",
        ("mean_vaoi", "ci95_vaoi"),
        ("node_count",),
    )
    for topology in ("community", "random"):
        scalability_module.plot_topology(node_summary, topology, OUTPUT_N_ROOT)
    node_metadata_path = OUTPUT_N_ROOT / "scalability_metadata.json"
    with node_metadata_path.open("r", encoding="utf-8") as stream:
        node_metadata = json.load(stream)
    node_metadata["model_dirs"] = serialize_model_dirs(model_dirs)
    with node_metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(node_metadata, stream, indent=2, sort_keys=True)

    update_summary = read_summary_rows(
        OUTPUT_U_ROOT / "update_probability_summary.csv",
        ("mean_vaoi", "ci95_vaoi", "update_probability"),
    )
    for topology in ("community", "random"):
        update_module.plot_topology(update_summary, topology, OUTPUT_U_ROOT)
    update_metadata_path = OUTPUT_U_ROOT / "update_probability_metadata.json"
    with update_metadata_path.open("r", encoding="utf-8") as stream:
        update_metadata = json.load(stream)
    update_metadata["model_dirs"] = serialize_model_dirs(model_dirs)
    with update_metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(update_metadata, stream, indent=2, sort_keys=True)


def main():
    """Restore the V3.2 checkpoint and run both existing scalability tests."""
    parser = argparse.ArgumentParser(
        description="Evaluate the V3.2 GNN on existing N and u scalability scenarios"
    )
    parser.add_argument(
        "--only",
        choices=("both", "N", "u"),
        default="both",
        help="run both tests or only one test family (default: both)",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="refresh generated labels and metadata without rerunning simulations",
    )
    args = parser.parse_args()
    if not V32_MODEL_DIR.is_dir():
        raise FileNotFoundError("missing V3.2 model directory: {}".format(V32_MODEL_DIR))

    # Import the established evaluators so their simulator and statistics paths
    # remain identical to the original 0804 scalability experiments.
    import run_gnn_scalability_generalization as scalability_module
    import run_gnn_update_probability_generalization as update_module

    model_dirs = configure_model_dirs(scalability_module, update_module)
    if args.refresh_only:
        refresh_plots_and_metadata(scalability_module, update_module, model_dirs)
        print("Refreshed V3.2 plot labels and metadata")
        return
    if args.only in ("both", "N"):
        run_node_scalability(scalability_module, model_dirs)
    if args.only in ("both", "u"):
        run_update_probability(update_module, scalability_module, model_dirs)
    print("Saved V3.2 scalability outputs under 0806ScalabilityV3.2_N and 0806ScalabilityV3.2_u")


if __name__ == "__main__":
    main()
