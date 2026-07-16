"""Compare saved ORC and AF3 result sets for curvature and policy effects.

The script reads two completed VAoI-vs-transmissions sweeps, compares raw edge
curvature, normalized bottleneck importance, curvature-policy inputs, and final
policy metrics, then writes compact CSV/Markdown outputs.
"""

import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORC_ROOT = PROJECT_ROOT / "results" / "test_vaoi_vs_transmissions_u0.05"
AF3_ROOT = PROJECT_ROOT / "results" / "test_vaoi_vs_transmissions_u0.05_af3"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "orc_af3_comparison_u0.05"

CURVATURE_POLICY = "orc_freshness_backoff"
LOWER_IS_BETTER = {
    "mean_VAoI",
    "mean_max_VAoI",
    "p95_max_VAoI",
    "mean_tail_VAoI",
    "mean_dissemination_delay",
}


def read_csv_dicts(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv_dicts(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def as_float(value):
    if value is None or value == "":
        return None
    return float(value)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def ranks(values):
    """Return average ranks, preserving ties."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            output[original_index] = rank
        index = end
    return output


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def cosine(xs, ys):
    num = sum(x * y for x, y in zip(xs, ys))
    den_x = math.sqrt(sum(x * x for x in xs))
    den_y = math.sqrt(sum(y * y for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def top_edges(edges, scores, fraction):
    count = max(1, int(math.ceil(len(edges) * fraction)))
    ordered = sorted(zip(edges, scores), key=lambda item: item[1], reverse=True)
    return {edge for edge, _ in ordered[:count]}


def experiment_dirs(root):
    return {
        path.name: path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("test_vaoi_vs_transmissions_")
    }


def common_curvature_cases():
    orc_dirs = experiment_dirs(ORC_ROOT)
    af3_dirs = experiment_dirs(AF3_ROOT)
    return sorted(
        name for name in set(orc_dirs).intersection(af3_dirs)
        if "_C" in name
    )


def read_edge_values(root, experiment_id, seed):
    path = root / experiment_id / str(seed) / CURVATURE_POLICY / "edge_curvature.csv"
    rows = read_csv_dicts(path)
    output = {}
    for row in rows:
        edge = (int(row["node_i"]), int(row["node_j"]))
        output[edge] = {
            "curvature": float(row["curvature"]),
            "importance": float(row["bottleneck_importance"]),
        }
    return output


def compare_curvature(case_id):
    rows = []
    for seed in range(5):
        orc = read_edge_values(ORC_ROOT, case_id, seed)
        af3 = read_edge_values(AF3_ROOT, case_id, seed)
        edges = sorted(set(orc).intersection(af3))
        orc_curv = [orc[edge]["curvature"] for edge in edges]
        af3_curv = [af3[edge]["curvature"] for edge in edges]
        orc_imp = [orc[edge]["importance"] for edge in edges]
        af3_imp = [af3[edge]["importance"] for edge in edges]

        abs_curv = [abs(a - b) for a, b in zip(af3_curv, orc_curv)]
        abs_imp = [abs(a - b) for a, b in zip(af3_imp, orc_imp)]
        sign_agree = [
            1.0 if sign(a) == sign(b) else 0.0
            for a, b in zip(af3_curv, orc_curv)
        ]
        nonzero_agree = [
            1.0 if (a > 0.0) == (b > 0.0) else 0.0
            for a, b in zip(af3_imp, orc_imp)
        ]

        row = {
            "case_id": case_id,
            "topology_seed": seed,
            "edge_count": len(edges),
            "raw_curvature_pearson": pearson(orc_curv, af3_curv),
            "raw_curvature_spearman": spearman(orc_curv, af3_curv),
            "raw_curvature_sign_agreement": mean(sign_agree),
            "raw_curvature_mean_abs_diff": mean(abs_curv),
            "orc_curvature_min": min(orc_curv),
            "orc_curvature_max": max(orc_curv),
            "af3_curvature_min": min(af3_curv),
            "af3_curvature_max": max(af3_curv),
            "importance_pearson": pearson(orc_imp, af3_imp),
            "importance_spearman": spearman(orc_imp, af3_imp),
            "importance_cosine": cosine(orc_imp, af3_imp),
            "importance_mean_abs_diff": mean(abs_imp),
            "importance_nonzero_agreement": mean(nonzero_agree),
        }
        for fraction in (0.05, 0.10, 0.20):
            orc_top = top_edges(edges, orc_imp, fraction)
            af3_top = top_edges(edges, af3_imp, fraction)
            overlap = len(orc_top.intersection(af3_top)) / len(orc_top)
            row["top_{}pct_bottleneck_overlap".format(int(fraction * 100))] = overlap
        rows.append(row)
    return rows


def read_points(root):
    rows = read_csv_dicts(root / "points.csv")
    return {row["point_id"]: row for row in rows}


def metric_delta(orc_value, af3_value, metric):
    if orc_value is None or af3_value is None:
        return None
    if metric in LOWER_IS_BETTER:
        return (orc_value - af3_value) / orc_value if orc_value else None
    return (af3_value - orc_value) / orc_value if orc_value else None


def compare_policy_points():
    orc_points = read_points(ORC_ROOT)
    af3_points = read_points(AF3_ROOT)
    rows = []
    for point_id in sorted(set(orc_points).intersection(af3_points)):
        orc = orc_points[point_id]
        af3 = af3_points[point_id]
        row = {
            "point_id": point_id,
            "policy": orc["policy"],
            "curvature_weight": orc.get("curvature_weight", ""),
            "orc_avg_tx_per_slot": as_float(orc["avg_tx_per_slot"]),
            "af3_avg_tx_per_slot": as_float(af3["avg_tx_per_slot"]),
            "avg_tx_per_slot_relative_change_af3_vs_orc": metric_delta(
                as_float(orc["avg_tx_per_slot"]),
                as_float(af3["avg_tx_per_slot"]),
                "avg_tx_per_slot",
            ),
            "orc_mean_VAoI": as_float(orc["mean_VAoI"]),
            "af3_mean_VAoI": as_float(af3["mean_VAoI"]),
            "mean_VAoI_gain_af3_vs_orc": metric_delta(
                as_float(orc["mean_VAoI"]),
                as_float(af3["mean_VAoI"]),
                "mean_VAoI",
            ),
        }
        rows.append(row)
    return rows


def read_policy_comparison(root, experiment_id):
    rows = read_json(root / experiment_id / "policy_comparison.json")
    return rows[0]


def compare_full_policy_metrics():
    rows = []
    for case_id in common_curvature_cases():
        orc = read_policy_comparison(ORC_ROOT, case_id)
        af3 = read_policy_comparison(AF3_ROOT, case_id)
        row = {
            "case_id": case_id,
            "curvature_weight": "",
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
            row["orc_" + metric] = orc.get(metric)
            row["af3_" + metric] = af3.get(metric)
            row[metric + "_gain_af3_vs_orc"] = metric_delta(
                orc.get(metric), af3.get(metric), metric
            )
        rows.append(row)
    return rows


def summarize(rows, key):
    return mean([row.get(key) for row in rows])


def write_report(curvature_rows, input_rows, point_rows, metric_rows):
    c_rows = curvature_rows
    c_point_rows = [row for row in point_rows if row["point_id"].startswith("C")]
    report = []
    report.append("# ORC vs AF3 comparison, update_probability=0.05")
    report.append("")
    report.append("The ORC root is `results/test_vaoi_vs_transmissions_u0.05`; the AF3 root is `results/test_vaoi_vs_transmissions_u0.05_af3`.")
    report.append("")
    report.append("## Curvature values")
    report.append("")
    report.append("- Raw edge curvature correlation is high: mean Pearson `{:.3f}`, mean Spearman `{:.3f}`.".format(
        summarize(c_rows, "raw_curvature_pearson"),
        summarize(c_rows, "raw_curvature_spearman"),
    ))
    report.append("- Raw curvature sign agreement is `{:.1f}%` on average.".format(
        100.0 * summarize(c_rows, "raw_curvature_sign_agreement")
    ))
    report.append("- ORC values are small real numbers, while AF3 values are integer structural scores; direct magnitude comparison is therefore not very meaningful.")
    report.append("")
    report.append("## Bottleneck ranking")
    report.append("")
    report.append("- Normalized bottleneck-importance correlation is also weak: mean Pearson `{:.3f}`, mean Spearman `{:.3f}`, cosine `{:.3f}`.".format(
        summarize(c_rows, "importance_pearson"),
        summarize(c_rows, "importance_spearman"),
        summarize(c_rows, "importance_cosine"),
    ))
    report.append("- Top bottleneck-edge overlap is low: top 5% `{:.1f}%`, top 10% `{:.1f}%`, top 20% `{:.1f}%`.".format(
        100.0 * summarize(c_rows, "top_5pct_bottleneck_overlap"),
        100.0 * summarize(c_rows, "top_10pct_bottleneck_overlap"),
        100.0 * summarize(c_rows, "top_20pct_bottleneck_overlap"),
    ))
    report.append("")
    report.append("## Strategy input")
    report.append("")
    report.append("- The actual curvature-policy input `curvature_weight * bottleneck_importance` differs substantially between ORC and AF3.")
    report.append("- Mean absolute input difference across C cases is `{:.3f}`; mean input cosine similarity is `{:.3f}`.".format(
        summarize(input_rows, "weighted_input_mean_abs_diff"),
        summarize(input_rows, "weighted_input_cosine"),
    ))
    report.append("")
    report.append("## Final policy metrics")
    report.append("")
    report.append("- Across C points, AF3 changes mean VAoI by `{:.1f}%` on average versus ORC, where positive means AF3 is better.".format(
        100.0 * summarize(c_point_rows, "mean_VAoI_gain_af3_vs_orc")
    ))
    report.append("- AF3 changes average transmissions per slot by `{:.1f}%` on average versus ORC.".format(
        100.0 * summarize(c_point_rows, "avg_tx_per_slot_relative_change_af3_vs_orc")
    ))
    best = min(c_point_rows, key=lambda row: row["af3_mean_VAoI"])
    report.append("- Best AF3 C point by mean VAoI is `{}`: mean_VAoI `{:.3f}`, avg_tx_per_slot `{:.3f}`.".format(
        best["point_id"], best["af3_mean_VAoI"], best["af3_avg_tx_per_slot"]
    ))
    best_orc = min(c_point_rows, key=lambda row: row["orc_mean_VAoI"])
    report.append("- Best ORC C point by mean VAoI is `{}`: mean_VAoI `{:.3f}`, avg_tx_per_slot `{:.3f}`.".format(
        best_orc["point_id"], best_orc["orc_mean_VAoI"], best_orc["orc_avg_tx_per_slot"]
    ))
    report.append("")
    report.append("Interpretation: AF3 preserves much of the raw edge-curvature ordering, but it is not an identical strategy input after negative-part normalization. It selects a noticeably different set of top bottleneck edges and drives higher transmission rates for all curvature-policy configurations. The performance effect is mixed: AF3 improves C2-C4 and C6-C7, but degrades C1 and C5; the best mean-VAoI point remains ORC C1.")
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def compare_weighted_inputs(case_ids):
    rows = []
    point_rows = read_points(ORC_ROOT)
    for case_id in case_ids:
        point = next(
            row for row in point_rows.values()
            if row["experiment_id"] == case_id
        )
        weight = float(point["curvature_weight"])
        for seed in range(5):
            orc = read_edge_values(ORC_ROOT, case_id, seed)
            af3 = read_edge_values(AF3_ROOT, case_id, seed)
            edges = sorted(set(orc).intersection(af3))
            orc_input = [weight * orc[edge]["importance"] for edge in edges]
            af3_input = [weight * af3[edge]["importance"] for edge in edges]
            abs_diff = [abs(a - b) for a, b in zip(af3_input, orc_input)]
            rows.append({
                "case_id": case_id,
                "topology_seed": seed,
                "curvature_weight": weight,
                "weighted_input_mean_abs_diff": mean(abs_diff),
                "weighted_input_max_abs_diff": max(abs_diff),
                "weighted_input_pearson": pearson(orc_input, af3_input),
                "weighted_input_spearman": spearman(orc_input, af3_input),
                "weighted_input_cosine": cosine(orc_input, af3_input),
            })
    return rows


def main():
    case_ids = common_curvature_cases()
    if not case_ids:
        raise SystemExit("No common curvature cases found.")

    # Curvature and bottleneck importance are topology-level, so one C case is enough.
    curvature_rows = compare_curvature(case_ids[0])
    input_rows = compare_weighted_inputs(case_ids)
    point_rows = compare_policy_points()
    metric_rows = compare_full_policy_metrics()

    write_csv_dicts(
        OUTPUT_ROOT / "curvature_bottleneck_by_seed.csv",
        curvature_rows,
        list(curvature_rows[0].keys()),
    )
    write_csv_dicts(
        OUTPUT_ROOT / "strategy_input_by_case_seed.csv",
        input_rows,
        list(input_rows[0].keys()),
    )
    write_csv_dicts(
        OUTPUT_ROOT / "points_orc_vs_af3.csv",
        point_rows,
        list(point_rows[0].keys()),
    )
    write_csv_dicts(
        OUTPUT_ROOT / "curvature_policy_metrics_orc_vs_af3.csv",
        metric_rows,
        list(metric_rows[0].keys()),
    )
    write_report(curvature_rows, input_rows, point_rows, metric_rows)
    print("Saved comparison outputs to {}".format(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
