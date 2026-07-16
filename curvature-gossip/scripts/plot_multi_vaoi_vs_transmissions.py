"""Build a multi-configuration VAoI-vs-transmissions plot.

This script creates a small, isolated sweep under
results/test_vaoi_vs_transmissions, runs one policy configuration per case,
then plots all cases on the same VAoI-vs-transmissions figure with compact
labels and a parameter table.
脚本会复用已经完成的配置；新增配置会自动跑，删除配置不会再画到新图里。
"""

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.experiments import run_experiment


BASE_CONFIG = PROJECT_ROOT / "configs" / "topo2_backoff_v2.yaml"
GENERATED_CONFIG_DIR = PROJECT_ROOT / "configs" / "generated_sweeps"
RESULT_ROOT = PROJECT_ROOT / "results" / "test_vaoi_vs_transmissions_u0.05_af3"
SWEEP_ID = "test_vaoi_vs_transmissions"

TOPOLOGY_SEEDS = [0, 1, 2, 3, 4]
CHANNEL_SEEDS = [0]
UPDATE_SEEDS = [0]

POLICY_STYLE = {
    "R": {
        "name": "random",
        "short_name": "R-random",
        "color": "#4C78A8",
        "marker": "o",
    },
    "F": {
        "name": "freshness_backoff",
        "short_name": "F-freshness-backoff",
        "color": "#F58518",
        "marker": "s",
    },
    "C": {
        "name": "orc_freshness_backoff",
        "short_name": "C-orc_freshness-backoff",
        "color": "#54A24B",
        "marker": "^",
    },
}

RANDOM_CASES = [
    {"tx_probability": 0.07},
    {"tx_probability": 0.10},
    {"tx_probability": 0.12},
    {"tx_probability": 0.15},
]

FRESHNESS_CASES = [
    {"q_max": 0.30, "threshold": 0.50, "congestion_backoff_weight": 0.04, "attempt_backoff_factor": 0.90},
    {"q_max": 0.35, "threshold": 0.50, "congestion_backoff_weight": 0.04, "attempt_backoff_factor": 0.90},
    {"q_max": 0.40, "threshold": 0.50, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.40, "threshold": 0.60, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.45, "threshold": 0.50, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.45, "threshold": 0.60, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.50, "threshold": 0.60, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.80},
]

CURVATURE_CASES = [
    {"q_max": 0.35, "threshold": 0.50, "curvature_weight": 0.50, "congestion_backoff_weight": 0.04, "attempt_backoff_factor": 0.90},
    {"q_max": 0.40, "threshold": 0.60, "curvature_weight": 0.75, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.40, "threshold": 0.60, "curvature_weight": 1.00, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.45, "threshold": 0.60, "curvature_weight": 0.75, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.85},
    {"q_max": 0.45, "threshold": 0.60, "curvature_weight": 1.00, "congestion_backoff_weight": 0.08, "attempt_backoff_factor": 0.80},
    {"q_max": 0.50, "threshold": 0.70, "curvature_weight": 1.25, "congestion_backoff_weight": 0.12, "attempt_backoff_factor": 0.80},
    {"q_max": 0.55, "threshold": 0.70, "curvature_weight": 1.25, "congestion_backoff_weight": 0.12, "attempt_backoff_factor": 0.80},
]


def load_base_config():
    with BASE_CONFIG.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def format_float(value):
    if value is None or value == "":
        return ""
    return "{:g}".format(float(value))


def find_base_policy(base_config, policy_name):
    for policy in base_config["policies"]:
        if policy["name"] == policy_name:
            return deepcopy(policy)
    raise ValueError("base config does not contain policy: {}".format(policy_name))


def build_cases():
    cases = []
    for index, params in enumerate(RANDOM_CASES, start=1):
        cases.append({"tag": "R", "index": index, "params": params})
    for index, params in enumerate(FRESHNESS_CASES, start=1):
        cases.append({"tag": "F", "index": index, "params": params})
    for index, params in enumerate(CURVATURE_CASES, start=1):
        cases.append({"tag": "C", "index": index, "params": params})
    return cases


def policy_for_case(base_config, case):
    style = POLICY_STYLE[case["tag"]]
    policy = find_base_policy(base_config, style["name"])
    params = policy.setdefault("params", {})

    # Only change the scanned knobs; all other behavior stays inherited from topo2_backoff_v2.yaml.
    if case["tag"] == "R":
        params["tx_probability"] = case["params"]["tx_probability"]
    else:
        for key, value in case["params"].items():
            params[key] = value
    return policy


def experiment_id(case):
    params = case["params"]
    if case["tag"] == "R":
        suffix = "p{}".format(format_float(params["tx_probability"]))
    elif case["tag"] == "F":
        suffix = "q{}_th{}_cb{}_ab{}".format(
            format_float(params["q_max"]),
            format_float(params["threshold"]),
            format_float(params["congestion_backoff_weight"]),
            format_float(params["attempt_backoff_factor"]),
        )
    else:
        suffix = "q{}_th{}_cw{}_cb{}_ab{}".format(
            format_float(params["q_max"]),
            format_float(params["threshold"]),
            format_float(params["curvature_weight"]),
            format_float(params["congestion_backoff_weight"]),
            format_float(params["attempt_backoff_factor"]),
        )
    return "{}_{}{:02d}_{}".format(SWEEP_ID, case["tag"], case["index"], suffix).replace(".", "p")


def write_case_config(base_config, case):
    config = deepcopy(base_config)
    case_id = experiment_id(case)
    config["experiment"]["id"] = case_id
    config["experiment"]["topology_seeds"] = TOPOLOGY_SEEDS
    config["experiment"]["channel_seeds"] = CHANNEL_SEEDS
    config["experiment"]["update_seeds"] = UPDATE_SEEDS
    config["policies"] = [policy_for_case(base_config, case)]

    # Keep the sweep self-contained and skip heavy per-slot traces for speed and disk usage.
    config["output"]["root"] = str(RESULT_ROOT)
    config["output"]["save_per_slot"] = False
    config["output"]["node_diagnostics_stride"] = 0
    config["output"]["make_plots"] = False

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = GENERATED_CONFIG_DIR / "{}.yaml".format(case_id)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    return config_path, case_id


def case_is_complete(case_id):
    return (RESULT_ROOT / case_id / "policy_comparison.json").is_file()


def read_case_result(case, case_id):
    path = RESULT_ROOT / case_id / "policy_comparison.json"
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if len(rows) != 1:
        raise ValueError("expected one policy row in {}".format(path))
    row = rows[0]
    style = POLICY_STYLE[case["tag"]]
    params = case["params"]
    return {
        "point_id": "{}{}".format(case["tag"], case["index"]),
        "policy": row["policy"],
        "legend": style["short_name"],
        "experiment_id": case_id,
        "tx_probability": params.get("tx_probability", ""),
        "q_max": params.get("q_max", ""),
        "threshold": params.get("threshold", ""),
        "curvature_weight": params.get("curvature_weight", ""),
        "congestion_backoff_weight": params.get("congestion_backoff_weight", ""),
        "attempt_backoff_factor": params.get("attempt_backoff_factor", ""),
        "independent_runs": row["independent_runs"],
        "mean_VAoI": row["mean_VAoI"],
        "mean_VAoI_ci95": row["mean_VAoI_ci95"],
        "avg_tx_per_slot": row["avg_tx_per_slot"],
        "avg_tx_per_slot_ci95": row["avg_tx_per_slot_ci95"],
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "point_id",
        "legend",
        "policy",
        "tx_probability",
        "q_max",
        "threshold",
        "curvature_weight",
        "congestion_backoff_weight",
        "attempt_backoff_factor",
        "independent_runs",
        "avg_tx_per_slot",
        "avg_tx_per_slot_ci95",
        "mean_VAoI",
        "mean_VAoI_ci95",
        "experiment_id",
    ]
    write_csv(RESULT_ROOT / "points.csv", rows, fields)
    table_fields = [
        "point_id",
        "legend",
        "tx_probability",
        "q_max",
        "threshold",
        "curvature_weight",
        "congestion_backoff_weight",
        "attempt_backoff_factor",
    ]
    write_csv(RESULT_ROOT / "config_table.csv", rows, table_fields)


def plot(rows):
    figure = plt.figure(figsize=(13.5, 10))
    grid = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[3.25, 1.75], hspace=0.36)
    axis = figure.add_subplot(grid[0])
    table_grid = grid[1].subgridspec(nrows=1, ncols=2, wspace=0.08)
    table_axes = [figure.add_subplot(table_grid[0]), figure.add_subplot(table_grid[1])]
    for table_axis in table_axes:
        table_axis.axis("off")

    for tag, style in POLICY_STYLE.items():
        tag_rows = [row for row in rows if row["point_id"].startswith(tag)]
        if not tag_rows:
            continue
        x_values = [row["avg_tx_per_slot"] for row in tag_rows]
        y_values = [row["mean_VAoI"] for row in tag_rows]
        x_errors = [row["avg_tx_per_slot_ci95"] or 0.0 for row in tag_rows]
        y_errors = [row["mean_VAoI_ci95"] or 0.0 for row in tag_rows]
        axis.errorbar(
            x_values,
            y_values,
            xerr=x_errors,
            yerr=y_errors,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=3,
            markersize=7,
            label=style["short_name"],
            alpha=0.9,
        )
        for row in tag_rows:
            axis.annotate(
                row["point_id"],
                (row["avg_tx_per_slot"], row["mean_VAoI"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                weight="bold",
            )

    axis.set_title("VAoI vs transmissions across policy configurations")
    axis.set_xlabel("Average transmissions per slot")
    axis.set_ylabel("Mean VAoI")
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend(loc="best", frameon=True)

    # Keep low-VAoI points visually separated from the parameter table.
    y_lows = [
        row["mean_VAoI"] - (row["mean_VAoI_ci95"] or 0.0)
        for row in rows
    ]
    y_highs = [
        row["mean_VAoI"] + (row["mean_VAoI_ci95"] or 0.0)
        for row in rows
    ]
    y_span = max(y_highs) - min(y_lows)
    y_padding = max(0.15, 0.16 * y_span)
    axis.set_ylim(max(0.0, min(y_lows) - y_padding), max(y_highs) + 0.08 * y_span)
    axis.margins(x=0.04)

    table_rows = []
    for row in rows:
        table_rows.append([
            row["point_id"],
            row["legend"].split("-")[0],
            format_float(row["tx_probability"]),
            format_float(row["q_max"]),
            format_float(row["threshold"]),
            format_float(row["curvature_weight"]),
            format_float(row["congestion_backoff_weight"]),
            format_float(row["attempt_backoff_factor"]),
        ])
    # Split the parameter table so added sweep points remain readable.
    midpoint = (len(table_rows) + 1) // 2
    for table_axis, block_rows in zip(table_axes, [table_rows[:midpoint], table_rows[midpoint:]]):
        table = table_axis.table(
            cellText=block_rows,
            colLabels=["ID", "type", "p", "qmax", "th", "cw", "cbw", "abf"],
            bbox=[0.0, 0.0, 1.0, 1.0],
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)

    output_path = RESULT_ROOT / "vaoi_vs_transmissions.png"
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    figure.savefig(str(RESULT_ROOT / "plot_multi_vaoi_vs_transmissions.png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    base_config = load_base_config()
    rows = []
    for number, case in enumerate(build_cases(), start=1):
        config_path, case_id = write_case_config(base_config, case)
        if case_is_complete(case_id):
            print("[{}] reusing {}".format(number, case_id))
        else:
            print("[{}] running {}".format(number, case_id))
            run_experiment(str(config_path), overwrite=True, progress=True)
        rows.append(read_case_result(case, case_id))

    write_outputs(rows)
    output_path = plot(rows)
    print("Saved points: {}".format(RESULT_ROOT / "points.csv"))
    print("Saved config table: {}".format(RESULT_ROOT / "config_table.csv"))
    print("Saved plot: {}".format(output_path))


if __name__ == "__main__":
    main()
