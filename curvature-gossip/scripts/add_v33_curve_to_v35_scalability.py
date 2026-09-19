"""Add the V3.3 ch2 mean-max curve to an existing V3.5 scalability result.

功能：读取已经完成的 V3.5 ch2 Bmax=0.30 scalability 结果，只对
V3.3 Bpen 的 ``best_validation_Bmax`` 检查点执行同一批社区场景评估，
然后合并逐场景 CSV、汇总 CSV、PNG/PDF 图和元数据。

The script deliberately reuses the existing evaluation YAMLs and the three
already evaluated V3.5 curves, avoiding an expensive full re-evaluation.
"""

from __future__ import division

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_v35_ch2_scalability as comparison


RESULT_ROOT = PROJECT_ROOT / "result_GNN"
DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "0917ScalabilityV35_ch2_b0.30"
DEFAULT_V33_MODEL = RESULT_ROOT / "mpnnV3.3_Bpen_ch2_n100_u0.20_b0.30_curvBmax20"
V33_MODEL_KEY = "v33_curvature_stage2"


def read_csv(path):
    """Read a CSV into dictionaries while preserving numeric conversion downstream."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_yaml(path):
    """Load one existing evaluation YAML through the shared comparison helper."""
    return comparison.read_yaml(path)


def configure_four_curve_plot():
    """Extend the shared plot style with a distinct V3.3 mean-max curve."""
    comparison.MODEL_ORDER = (
        "v35_curvature_stage2",
        V33_MODEL_KEY,
        "v35_no_curvature_stage2",
        "v35_curvature_stage1_only",
    )
    comparison.MODEL_LABELS[V33_MODEL_KEY] = "V3.3 ch2 curvature MPNN (mean-max)"
    comparison.MODEL_COLORS[V33_MODEL_KEY] = "#8E5EA2"
    comparison.MODEL_LINESTYLES[V33_MODEL_KEY] = ":"
    comparison.MODEL_MARKERS[V33_MODEL_KEY] = "^"
    # Keep the four probability annotations separated around each VAoI point.
    comparison.ANNOTATION_OFFSETS[V33_MODEL_KEY] = (1, 2)
    comparison.ANNOTATION_OFFSETS["v35_curvature_stage2"] = (1, 20)
    comparison.ANNOTATION_OFFSETS["v35_no_curvature_stage2"] = (1, -20)
    comparison.ANNOTATION_OFFSETS["v35_curvature_stage1_only"] = (-2, 12)


def write_readme(output_root):
    """Document the four curves and the reused 100-slot evaluation protocol."""
    text = (
        "# V3.5 ch2 Bmax=0.30 community scalability\n\n"
        "This result compares the V3.5 curvature-attention Stage-2 model, the "
        "V3.3 curvature mean-max model, the V3.5 no-curvature Stage-2 model, "
        "and the V3.5 curvature Stage-1-only baseline. All curves use the "
        "same existing community evaluation YAMLs (`N=50..150`, five "
        "scenarios per N, 100 slots per scenario).\n\n"
        "The V3.3 curve is evaluated from its `best_validation_Bmax` checkpoint "
        "and is merged with the previously generated V3.5 rows. Every point "
        "is annotated with mean Actor broadcast probability; shaded regions "
        "are normal-approximation 95% confidence intervals over five scenarios.\n"
    )
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main():
    """Evaluate only V3.3 and regenerate the combined four-curve artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--v33-model", type=Path, default=DEFAULT_V33_MODEL)
    args = parser.parse_args()

    output_root = args.output_root
    existing_path = output_root / "scalability_per_scenario.csv"
    metadata_path = output_root / "scalability_metadata.json"
    if not existing_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "existing V3.5 scalability result is required: {}".format(output_root)
        )
    comparison.bmax_checkpoint_path(args.v33_model)

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    config_paths = {
        int(node_count): output_root / "configs" / "community" / "n{:03d}.yaml".format(node_count)
        for node_count in metadata["node_counts"]
    }
    for path in config_paths.values():
        if not path.is_file():
            raise FileNotFoundError("missing shared evaluation YAML: {}".format(path))

    # Register the new line before evaluate_model builds labeled scenario rows.
    configure_four_curve_plot()
    model, model_raw = comparison.build_model(args.v33_model)
    try:
        checkpoint = comparison.checkpoint_episode(args.v33_model)
        v33_rows = comparison.evaluate_model(
            model,
            model_raw,
            V33_MODEL_KEY,
            checkpoint,
            config_paths,
            stage1_only=False,
        )
    finally:
        model.close()

    # Replace an older V3.3 append, if present, while preserving the three V3.5 curves.
    scenario_rows = [row for row in read_csv(existing_path) if row["model"] != V33_MODEL_KEY]
    scenario_rows.extend(v33_rows)

    summary_rows = comparison.summarize_rows(scenario_rows)
    comparison.write_csv(existing_path, scenario_rows)
    comparison.write_csv(output_root / "scalability_summary.csv", summary_rows)
    plot_paths = comparison.plot_scalability(summary_rows, output_root)

    metadata["evaluation"] = "V3.5 ch2 community N-generalization with V3.3 mean-max overlay"
    metadata["models"][V33_MODEL_KEY] = str(args.v33_model)
    metadata.setdefault("checkpoint_paths", {})[V33_MODEL_KEY] = str(
        comparison.bmax_checkpoint_path(args.v33_model)
    )
    metadata.setdefault("checkpoint_episodes", {})[V33_MODEL_KEY] = checkpoint
    metadata["line_order"] = list(comparison.MODEL_ORDER)
    metadata["plots"] = [str(path) for path in plot_paths]
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    write_readme(output_root)
    print("Added V3.3 curve to {}".format(output_root))


if __name__ == "__main__":
    main()
