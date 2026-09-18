"""独立后处理脚本：生成节点最小相邻边曲率和边曲率分布图。

本脚本只读取 result_mask/outputs 下已有的单节点静默故障结果，不重新运行
仿真，也不修改原有实验脚本。对每个场景，它会：
1. 按已有结果中的平均节点曲率直接生成 Curv_vs_aoi_NodeMean.png；
2. 用节点相邻边曲率的最小值生成 Curv_vs_aoi_NodeMin.png；
3. 统计所有已保存边曲率，生成 EdgeCurv_Distribution.png；
4. 保存节点最小曲率明细和后处理元数据，便于复核定义。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "result_mask" / "outputs"

AOI_METRICS: Tuple[Tuple[str, str], ...] = (
    ("delta_mean_aoi", "Delta mean AoI"),
    ("delta_mean_max_aoi", "Delta time-averaged max AoI"),
    ("delta_mean_min_aoi", "Delta time-averaged min AoI"),
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """读取 UTF-8 CSV，并给出清晰的缺失文件错误。"""
    if not path.is_file():
        raise FileNotFoundError("missing result file: {}".format(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _topology_seed(topology_directory: Path) -> int:
    match = re.fullmatch(r"topology_(\d+)", topology_directory.name)
    if match is None:
        raise ValueError("unexpected topology directory name: {}".format(topology_directory))
    return int(match.group(1))


def _load_node_min_curvature(
    scenario_directory: Path,
) -> Tuple[Dict[Tuple[int, int], float], List[float], List[Dict[str, object]]]:
    """从边曲率表计算 min_{u in N(v)} kappa(v,u)。"""
    node_min: Dict[Tuple[int, int], float] = {}
    node_min_rows: List[Dict[str, object]] = []
    all_edge_values: List[float] = []

    topology_directories = sorted(
        path for path in scenario_directory.glob("topology_*") if path.is_dir()
    )
    if not topology_directories:
        raise FileNotFoundError("no topology_* directory found in {}".format(scenario_directory))

    for topology_directory in topology_directories:
        topology_seed = _topology_seed(topology_directory)
        edge_rows = _read_csv(topology_directory / "topology_edges.csv")
        incident: Dict[int, List[Tuple[float, int, int]]] = {}

        for row in edge_rows:
            node_i = int(row["node_i"])
            node_j = int(row["node_j"])
            curvature = float(row["edge_curvature"])
            if not np.isfinite(curvature):
                raise ValueError("non-finite edge curvature in {}".format(topology_directory))
            all_edge_values.append(curvature)
            incident.setdefault(node_i, []).append((curvature, node_i, node_j))
            incident.setdefault(node_j, []).append((curvature, node_i, node_j))

        node_rows = _read_csv(topology_directory / "topology_nodes.csv")
        output_rows: List[Dict[str, object]] = []
        for node_row in node_rows:
            node = int(node_row["node"])
            incident_edges = incident.get(node, [])
            if not incident_edges:
                raise ValueError("node {} has no incident edges in {}".format(node, topology_directory))
            minimum_edge = min(incident_edges, key=lambda item: (item[0], item[1], item[2]))
            minimum_curvature, edge_i, edge_j = minimum_edge
            node_min[(topology_seed, node)] = minimum_curvature
            output_rows.append(
                {
                    "topology_seed": topology_seed,
                    "node": node,
                    "node_min_curvature": minimum_curvature,
                    "min_edge_node_i": edge_i,
                    "min_edge_node_j": edge_j,
                    "degree": len(incident_edges),
                }
            )

        # 将明细放在 topology 目录下，不覆盖原始 topology_nodes.csv。
        output_path = topology_directory / "topology_nodes_node_min.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
        node_min_rows.extend(output_rows)

    return node_min, all_edge_values, node_min_rows


def _broadcast_probability(rows: Iterable[Mapping[str, str]]) -> float | None:
    """提取结果中的平均广播概率，用于图标题标注。"""
    candidates: List[float] = []
    for row in rows:
        value = _float_or_none(row.get("average_broadcast_probability"))
        if value is None:
            value = _float_or_none(row.get("configured_broadcast_probability"))
        if value is not None:
            candidates.append(value)
    if not candidates:
        return None
    return float(np.mean(candidates))


def _title_suffix(probability: float | None) -> str:
    if probability is None:
        return ""
    return " (mean broadcast probability p={:.3f})".format(probability)


def _plot_curvature_impact(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    x_values: Sequence[float],
    x_label: str,
    title: str,
    probability: float | None,
) -> None:
    """绘制三类 AoI 退化与指定节点曲率定义的散点关系。"""
    fault_rows = [row for row in rows if int(row["failure_node"]) != -1]
    if not fault_rows:
        raise ValueError("no single-node failure rows found")
    if len(fault_rows) != len(x_values):
        raise ValueError("curvature values and failure rows have different lengths")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)
    x = np.asarray(x_values, dtype=float)
    seeds = np.asarray([int(row["topology_seed"]) for row in fault_rows])
    unique_seeds = sorted(set(int(seed) for seed in seeds))

    for axis, (metric, label) in zip(axes, AOI_METRICS):
        y = np.asarray([float(row[metric]) for row in fault_rows], dtype=float)
        scatter = axis.scatter(x, y, c=seeds, cmap="viridis", s=42, alpha=0.85)
        if len(x) >= 2 and np.ptp(x) > 1e-12:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            axis.plot(grid, slope * grid + intercept, color="black", linewidth=1.2, alpha=0.75)
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_xlabel(x_label)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)

    if len(unique_seeds) > 1:
        figure.colorbar(scatter, ax=axes, label="topology seed")
    figure.suptitle(title + _title_suffix(probability))
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_edge_distribution(
    path: Path,
    edge_values: Sequence[float],
    probability: float | None,
    bins: int,
) -> None:
    """绘制边 ORC 曲率的计数直方图，风格对应用户提供的参考图。"""
    if not edge_values:
        raise ValueError("no edge curvature values found")
    figure, axis = plt.subplots(figsize=(8, 5.3))
    axis.hist(
        np.asarray(edge_values, dtype=float),
        bins=bins,
        color="#4C84B5",
        edgecolor="white",
        linewidth=0.8,
    )
    axis.set_title("Edge-curvature distribution" + _title_suffix(probability))
    axis.set_xlabel("edge curvature")
    axis.set_ylabel("count")
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def postprocess_scenario(scenario_directory: Path, bins: int = 20) -> None:
    """为一个已完成的实验目录生成节点最小曲率和边分布结果。"""
    result_path = scenario_directory / "single_node_failure_results.csv"
    rows = _read_csv(result_path)
    fault_rows = [row for row in rows if int(row["failure_node"]) != -1]
    node_min, edge_values, node_min_rows = _load_node_min_curvature(scenario_directory)

    # 平均曲率图直接重绘，避免依赖已经取消的旧图文件。
    plots_directory = scenario_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)
    mean_plot = plots_directory / "Curv_vs_aoi_NodeMean.png"
    mean_x = [float(row["node_curvature"]) for row in fault_rows]
    _plot_curvature_impact(
        mean_plot,
        fault_rows,
        mean_x,
        "mean incident-edge curvature",
        "Single-node silence impact versus node mean curvature",
        _broadcast_probability(rows),
    )

    # 根据 (topology_seed, failed_node) 查找最小相邻边曲率。
    min_x = []
    for row in fault_rows:
        key = (int(row["topology_seed"]), int(row["failure_node"]))
        if key not in node_min:
            raise KeyError("missing node-min curvature for {} in {}".format(key, scenario_directory))
        min_x.append(node_min[key])
    probability = _broadcast_probability(rows)
    _plot_curvature_impact(
        plots_directory / "Curv_vs_aoi_NodeMin.png",
        fault_rows,
        min_x,
        "minimum incident-edge curvature",
        "Single-node silence impact versus node minimum curvature",
        probability,
    )
    _plot_edge_distribution(
        plots_directory / "EdgeCurv_Distribution.png",
        edge_values,
        probability,
        bins,
    )

    # 写入可审计的后处理说明，不覆盖原 experiment_manifest.json。
    manifest = {
        "scenario_directory": str(scenario_directory),
        "node_curvature_definition": "min incident edge curvature",
        "node_curvature_formula": "kappa_node(v) = min_{u in N(v)} kappa_edge(v,u)",
        "edge_curvature_source": "topology_*/topology_edges.csv",
        "edge_curvature_method": "existing experiment curvature output (global ORC in current config)",
        "mean_plot_copy": str(mean_plot.name),
        "min_plot": "Curv_vs_aoi_NodeMin.png",
        "edge_distribution_plot": "EdgeCurv_Distribution.png",
        "edge_histogram_bins": bins,
        "mean_broadcast_probability": probability,
        "node_min_detail_files": [
            "topology_{}/topology_nodes_node_min.csv".format(seed)
            for seed in sorted({int(row["topology_seed"]) for row in node_min_rows})
        ],
    }
    with (scenario_directory / "curvature_postprocess_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        "processed {}: {} failure rows, {} edge curvatures".format(
            scenario_directory.name, len(fault_rows), len(edge_values)
        )
    )


def _discover_scenarios(output_root: Path) -> List[Path]:
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "single_node_failure_results.csv").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root containing scenario result directories",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="scenario directory name; repeat this option to select multiple scenarios",
    )
    parser.add_argument("--bins", type=int, default=20, help="number of edge-curvature histogram bins")
    args = parser.parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    scenario_directories = (
        [output_root / name for name in args.scenario]
        if args.scenario
        else _discover_scenarios(output_root)
    )
    if not scenario_directories:
        raise FileNotFoundError("no scenario result directory found under {}".format(output_root))
    for scenario_directory in scenario_directories:
        postprocess_scenario(scenario_directory, bins=args.bins)


if __name__ == "__main__":
    main()
