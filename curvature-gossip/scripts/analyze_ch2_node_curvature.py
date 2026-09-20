"""功能：复现 ch2 固定验证场景，统计节点曲率分布并比较三种节点排名。

三种排名均在同一批拓扑上计算：
1. 节点关联 AF3 边曲率的最小值；
2. 先取该最小 AF3 边，再取其 local_degree_bound 归一化分数；
3. 节点关联 ORC 边曲率的最小值。

关键输出包括节点明细 CSV、分布统计 CSV、排名相关性/Top-K 重合度，以及 PNG 图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.curvature import (  # noqa: E402
    DistributedAF3Curvature,
    GlobalORCCurvature,
    bottleneck_importance,
    canonical_edge,
)
from curvature_gossip.learning.validation import validation_scenarios  # noqa: E402
from curvature_gossip.random_streams import make_rng  # noqa: E402
from curvature_gossip.topology import get_topology_generator  # noqa: E402


def read_yaml(path: Path):
    """读取并返回 YAML 配置。"""
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_csv(path: Path, rows):
    """将字典行写成 UTF-8 CSV。"""
    rows = list(rows)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_topology(raw, scenario):
    """使用项目 validation 的随机流复现一个固定 ch2 拓扑。"""
    generator = get_topology_generator(raw["topology"]["type"])
    params = dict(raw["topology"].get("params", {}))
    params["n_nodes"] = int(scenario.n_nodes)
    if "cluster_sizes" in params:
        params["cluster_sizes"] = [scenario.n_nodes // 2, scenario.n_nodes - scenario.n_nodes // 2]
    master_seed = int(raw.get("experiment", {}).get("master_seed", 0))
    return generator.generate(
        make_rng(master_seed, "fixed_validation", "topology", scenario.topology_seed),
        params,
    )


def node_scores(topology, af3_result, orc_result):
    """计算三种节点分数；分数越大表示越应该排在前面。"""
    importance = bottleneck_importance(topology, af3_result, "local_degree_bound")
    rows = []
    for node in sorted(topology.graph.nodes):
        edges = [canonical_edge(node, neighbor) for neighbor in topology.graph.neighbors(node)]
        af3_edge = min(edges, key=lambda edge: af3_result.edge_values[edge])
        orc_edge = min(edges, key=lambda edge: orc_result.edge_values[edge])
        af3_raw = float(af3_result.edge_values[af3_edge])
        orc_raw = float(orc_result.edge_values[orc_edge])
        rows.append({
            "node": int(node),
            "af3_min_edge": "{}-{}".format(*af3_edge),
            "af3_min_curvature": af3_raw,
            "af3_min_normalized_score": float(importance[af3_edge]),
            "orc_min_edge": "{}-{}".format(*orc_edge),
            "orc_min_curvature": orc_raw,
            # 排名分数统一为越大越靠前，等价于原始曲率越小越靠前。
            "rank_score_af3_min": -af3_raw,
            "rank_score_af3_normalized": float(importance[af3_edge]),
            "rank_score_orc_min": -orc_raw,
        })
    return rows


def rank(values):
    """返回无并列时的 1-based 降序排名；并列使用平均名次。"""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    output = np.empty(values.size, dtype=float)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        output[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return output


def distribution_rows(rows):
    """输出三种节点分数的均值、方差、分位数等统计量。"""
    definitions = {
        "af3_min_curvature": "AF3最小边原始曲率",
        "af3_min_normalized_score": "AF3最小边对应归一化分数",
        "orc_min_curvature": "ORC最小边原始曲率",
    }
    output = []
    for key, label in definitions.items():
        values = np.asarray([float(row[key]) for row in rows])
        output.append({
            "metric": key,
            "label": label,
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "variance_population": float(np.var(values)),
            "std_population": float(np.std(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return output


def correlation_rows(rows):
    """计算排名相关性及 Top-K 重合率，衡量归一化估计与基准排序的一致性。"""
    names = {
        "af3_min": "rank_score_af3_min",
        "af3_normalized": "rank_score_af3_normalized",
        "orc_min": "rank_score_orc_min",
    }
    output = []
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    for left_name, left_key in names.items():
        for right_name, right_key in names.items():
            if left_name >= right_name:
                continue
            scenario_rho, scenario_tau, scenario_overlap = [], [], []
            for scenario_id in scenario_ids:
                scenario_rows = [row for row in rows if row["scenario_id"] == scenario_id]
                left = np.asarray([row[left_key] for row in scenario_rows], dtype=float)
                right = np.asarray([row[right_key] for row in scenario_rows], dtype=float)
                scenario_rho.append(float(spearmanr(left, right).statistic))
                scenario_tau.append(float(kendalltau(left, right).statistic))
                left_top = set(np.argsort(-left)[:10])
                right_top = set(np.argsort(-right)[:10])
                scenario_overlap.append(len(left_top & right_top))
            output.append({
                "method_a": left_name,
                "method_b": right_name,
                "scenario_count": len(scenario_ids),
                "spearman_rho_mean": float(np.mean(scenario_rho)),
                "spearman_rho_std": float(np.std(scenario_rho, ddof=1)),
                "kendall_tau_mean": float(np.mean(scenario_tau)),
                "kendall_tau_std": float(np.std(scenario_tau, ddof=1)),
                "top10_overlap_count_mean": float(np.mean(scenario_overlap)),
                "top10_overlap_count_std": float(np.std(scenario_overlap, ddof=1)),
                "top10_overlap_fraction_mean": float(np.mean(scenario_overlap) / 10.0),
            })
    return output


def plot_results(all_rows, output_dir):
    """绘制分布、节点排名散点和 Top-K 重合率。"""
    labels = ["AF3-min curvature", "AF3-min normalized", "ORC-min curvature"]
    keys = ["af3_min_curvature", "af3_min_normalized_score", "orc_min_curvature"]
    values = [np.asarray([row[key] for row in all_rows], dtype=float) for key in keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    axes[0].boxplot(values, tick_labels=labels, showmeans=True)
    axes[0].set_title("ch2 node-score distribution")
    axes[0].set_ylabel("node score")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.asarray([row["rank_score_af3_min"] for row in all_rows])
    y = np.asarray([row["rank_score_af3_normalized"] for row in all_rows])
    axes[1].scatter(x, y, s=10, alpha=0.35, label="节点")
    axes[1].set_xlabel("AF3-min severity (-curvature)")
    axes[1].set_ylabel("corresponding normalized score")
    axes[1].set_title("AF3 raw vs normalized estimate")
    axes[1].grid(alpha=0.25)

    top10 = []
    method_names = ["AF3-min", "AF3-normalized", "ORC-min"]
    score_keys = ["rank_score_af3_min", "rank_score_af3_normalized", "rank_score_orc_min"]
    # 先在每个拓扑内取 Top-10，再对 20 个拓扑取平均，避免跨场景节点编号混淆。
    scenario_ids = sorted({r["scenario_id"] for r in all_rows})
    for i in range(3):
        row_values = []
        for j in range(3):
            overlaps = []
            for scenario_id in scenario_ids:
                scenario_rows = [r for r in all_rows if r["scenario_id"] == scenario_id]
                sets_i = set(np.argsort(-np.asarray([r[score_keys[i]] for r in scenario_rows]))[:10])
                sets_j = set(np.argsort(-np.asarray([r[score_keys[j]] for r in scenario_rows]))[:10])
                overlaps.append(len(sets_i & sets_j))
            row_values.append(float(np.mean(overlaps)))
        top10.append(row_values)
    image = axes[2].imshow(top10, vmin=0, vmax=10, cmap="Blues")
    axes[2].set_xticks(range(3), method_names, rotation=25, ha="right")
    axes[2].set_yticks(range(3), method_names)
    axes[2].set_title("Top-10 node overlap")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, "{:.1f}".format(top10[i][j]), ha="center", va="center")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "ch2_node_curvature_analysis.png", dpi=220)
    plt.close(fig)

    # 归一化分数单独画直方图，便于直接检查分布形状。
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values[1], bins=25, color="#2f6f9f", alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(values[1]), color="#c0392b", linestyle="--", label="均值 {:.4f}".format(np.mean(values[1])))
    ax.set_xlabel("node AF3-min normalized score")
    ax.set_ylabel("node count")
    ax.set_title("ch2 normalized curvature-score distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ch2_normalized_curvature_histogram.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze node curvature rankings for the ch2 result.")
    parser.add_argument("--result-dir", type=Path, default=PROJECT_ROOT / "result_GNN" / "mpnnV3.5_CurvAttn_ch2_n100_u0.20_b0.30_4EdgeFeature_rawBmax")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT.parent / "ch2_node_curvature_analysis")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    raw = read_yaml(result_dir / "source_config.yaml")
    scenarios = validation_scenarios(raw)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    scenario_summary = []
    for scenario in scenarios:
        topology = make_topology(raw, scenario)
        af3 = DistributedAF3Curvature().compute(topology)
        orc = GlobalORCCurvature(alpha=0.5).compute(topology)
        rows = node_scores(topology, af3, orc)
        for row in rows:
            row.update({"scenario_id": scenario.scenario_id, "topology_seed": scenario.topology_seed})
        all_rows.extend(rows)
        scenario_summary.append({
            "scenario_id": scenario.scenario_id,
            "topology_seed": scenario.topology_seed,
            "n_nodes": scenario.n_nodes,
            "af3_normalized_mean": float(np.mean([r["af3_min_normalized_score"] for r in rows])),
            "af3_normalized_std": float(np.std([r["af3_min_normalized_score"] for r in rows])),
        })

    # 聚合排名用于跨 20 个固定场景的总体比较；节点位置按 scenario 内部独立编号。
    write_csv(output_dir / "node_curvature_scores.csv", all_rows)
    write_csv(output_dir / "scenario_distribution_summary.csv", scenario_summary)
    write_csv(output_dir / "distribution_summary.csv", distribution_rows(all_rows))
    write_csv(output_dir / "ranking_comparison.csv", correlation_rows(all_rows))

    # 为每个场景保存 1-based 节点排名，避免跨场景同编号造成误读。
    ranking_rows = []
    for scenario_id in sorted({r["scenario_id"] for r in all_rows}):
        rows = [r for r in all_rows if r["scenario_id"] == scenario_id]
        for method, key in (("af3_min", "rank_score_af3_min"), ("af3_normalized", "rank_score_af3_normalized"), ("orc_min", "rank_score_orc_min")):
            ranks = rank([r[key] for r in rows])
            for row, value in zip(rows, ranks):
                ranking_rows.append({"scenario_id": scenario_id, "node": row["node"], "method": method, "rank": float(value), "score": float(row[key])})
    write_csv(output_dir / "node_rankings_by_scenario.csv", ranking_rows)
    plot_results(all_rows, output_dir)

    metadata = {
        "source_config": str(result_dir / "source_config.yaml"),
        "scenario_count": len(scenarios),
        "node_count_per_scenario": int(scenarios[0].n_nodes),
        "orc_alpha": 0.5,
        "af3_normalization": "local_degree_bound",
        "ranking_direction": "smaller raw curvature means higher rank; normalized score uses larger-is-higher",
        "outputs": [p.name for p in sorted(output_dir.iterdir())],
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "scenario_count": len(scenarios), "node_rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
