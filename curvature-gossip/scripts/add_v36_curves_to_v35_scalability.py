"""Add the two V3.6 node-curvature curves to the existing scalability plot.

功能：复用 V3.5 ch2 的 community N-generalization 评估配置，对两个
V3.6 checkpoint 执行相同的 N=50..150 评估，并更新原有 CSV、PNG、PDF、
metadata 和 README。图中每个 VAoI 点继续标注平均 Actor 广播概率。
"""

from __future__ import division

import argparse
import csv
import json
from pathlib import Path

import compare_v35_ch2_scalability as comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result_GNN"
DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "0917ScalabilityV35_ch2_b0.30"

V36_MODELS = (
    (
        "v36_no_min_edge_curvature",
        RESULT_ROOT / "mpnnV3.6_NodeCurv_CurvAttn_ch2_n100_u0.20_b0.30_noMinEdgeCurv",
        "V3.6 node-curvature MPNN (no min-edge curvature)",
        "#7B2CBF",
        "--",
        "D",
        (1, 30),
    ),
    (
        "v36_raw_bmax",
        RESULT_ROOT / "mpnnV3.6_NodeCurv_CurvAttn_ch2_n100_u0.20_b0.30_rawBmax",
        "V3.6 node-curvature MPNN (raw Bmax)",
        "#E76F51",
        (0, (5, 2)),
        "P",
        (1, -30),
    ),
)


def read_csv(path):
    """Read rows from an existing UTF-8 CSV."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configure_plot():
    """Register the six curve styles used by the regenerated figure."""
    # Re-register the existing V3.3 overlay because this script runs in a
    # fresh Python process rather than through add_v33_curve_to_v35_scalability.
    comparison.MODEL_LABELS["v33_curvature_stage2"] = "V3.3 ch2 curvature MPNN (mean-max)"
    comparison.MODEL_COLORS["v33_curvature_stage2"] = "#8E5EA2"
    comparison.MODEL_LINESTYLES["v33_curvature_stage2"] = ":"
    comparison.MODEL_MARKERS["v33_curvature_stage2"] = "^"
    comparison.ANNOTATION_OFFSETS["v33_curvature_stage2"] = (1, 16)

    comparison.MODEL_ORDER = (
        "v35_curvature_stage2",
        "v33_curvature_stage2",
        "v36_no_min_edge_curvature",
        "v36_raw_bmax",
        "v35_no_curvature_stage2",
        "v35_curvature_stage1_only",
    )
    for key, _, label, color, linestyle, marker, offset in V36_MODELS:
        comparison.MODEL_LABELS[key] = label
        comparison.MODEL_COLORS[key] = color
        comparison.MODEL_LINESTYLES[key] = linestyle
        comparison.MODEL_MARKERS[key] = marker
        comparison.ANNOTATION_OFFSETS[key] = offset

    # Spread the existing labels around the denser six-curve plot.
    comparison.ANNOTATION_OFFSETS["v35_curvature_stage2"] = (1, 48)
    comparison.ANNOTATION_OFFSETS["v33_curvature_stage2"] = (1, 16)
    comparison.ANNOTATION_OFFSETS["v35_no_curvature_stage2"] = (1, -48)
    comparison.ANNOTATION_OFFSETS["v35_curvature_stage1_only"] = (-2, 12)


def write_readme(output_root):
    """Document the six curves and the common evaluation protocol."""
    text = (
        "# V3.5 ch2 Bmax=0.30 community scalability\n\n"
        "This result compares the existing V3.5/V3.3 curves with two V3.6 "
        "node-curvature MPNN variants: `noMinEdgeCurv` and `rawBmax`. All "
        "curves use the same community evaluation YAMLs (`N=50..150`, five "
        "scenarios per N, 100 slots per scenario). Every VAoI point is "
        "annotated with the mean Actor broadcast probability, and shaded "
        "regions show normal-approximation 95% confidence intervals.\n"
    )
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main():
    """Evaluate both V3.6 checkpoints and regenerate the combined artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_root
    existing_path = output_root / "scalability_per_scenario.csv"
    metadata_path = output_root / "scalability_metadata.json"
    if not existing_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("existing scalability result is required: {}".format(output_root))

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    config_paths = {
        int(node_count): output_root / "configs" / "community" / "n{:03d}.yaml".format(node_count)
        for node_count in metadata["node_counts"]
    }
    for path in config_paths.values():
        if not path.is_file():
            raise FileNotFoundError("missing shared evaluation YAML: {}".format(path))

    configure_plot()
    scenario_rows = [
        row for row in read_csv(existing_path)
        if row["model"] not in {item[0] for item in V36_MODELS}
    ]
    metadata.setdefault("models", {})
    metadata.setdefault("checkpoint_paths", {})
    metadata.setdefault("checkpoint_episodes", {})

    for model_key, model_dir, _, _, _, _, _ in V36_MODELS:
        checkpoint = comparison.checkpoint_episode(model_dir)
        model, model_raw = comparison.build_model(model_dir)
        try:
            rows = comparison.evaluate_model(
                model,
                model_raw,
                model_key,
                checkpoint,
                config_paths,
                stage1_only=False,
            )
        finally:
            model.close()
        scenario_rows.extend(rows)
        metadata["models"][model_key] = str(model_dir)
        metadata["checkpoint_paths"][model_key] = str(comparison.bmax_checkpoint_path(model_dir))
        metadata["checkpoint_episodes"][model_key] = checkpoint

    summary_rows = comparison.summarize_rows(scenario_rows)
    comparison.write_csv(existing_path, scenario_rows)
    comparison.write_csv(output_root / "scalability_summary.csv", summary_rows)
    plot_paths = comparison.plot_scalability(summary_rows, output_root)

    metadata["evaluation"] = "V3.5 ch2 community N-generalization with V3.3 and V3.6 overlays"
    metadata["line_order"] = list(comparison.MODEL_ORDER)
    metadata["plots"] = [str(path) for path in plot_paths]
    metadata["mean_broadcast_probability"] = "summary mean of per-scenario action_prob_mean"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    write_readme(output_root)
    print("Added V3.6 curves to {}".format(output_root))


if __name__ == "__main__":
    main()
