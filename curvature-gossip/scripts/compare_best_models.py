"""Compare any number of restored best-validation models on common fixed scenarios.

Each model is reconstructed from its own ``training_config.yaml`` and restored
from ``checkpoints/best_validation/model``.  The script rejects mismatched
evaluation settings so a summary never silently mixes incomparable runs.
"""

import argparse
import csv
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates
from curvature_gossip.learning.trainer import _critic_config, _stage1_actor_config
from curvature_gossip.learning.validation import evaluate_fixed_validation


DEFAULT_CHECKPOINT = Path("checkpoints") / "best_validation" / "model"
EVALUATION_KEYS = ("topology", "source", "channel", "curvature", "constraints", "observation", "validation")

# 指定要比较的模型目录，按优先顺序排列
DEFAULT_MODEL_DIRS = (
    PROJECT_ROOT / "result_GNN/mpnn_heuristic_channel_n100_u0.20_b0.10_a1.5_lr1e-3",
    PROJECT_ROOT / "result_GNN/mpnn_heuristic_channel_n100_u0.20_b0.10_a1.5",
    PROJECT_ROOT / "result_2layers/nn_2layers_stage2_heuristic_channel_n100_u0.20_b0.10_a1.50_v2",
)

# 输出目录
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result_compare"
DEFAULT_OVERWRITE = True

# 指定验证场景配置文件
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "configs/compare/compare_n100_heuristic_channel.yaml"


def read_yaml(path: Path) -> Mapping:
    """Read a nonempty YAML mapping with a clear path-specific error."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return dict(value)


def with_evaluation_settings(model_config: Mapping, evaluation_config: Mapping) -> Mapping:
    """Overlay common simulator settings without changing a restored model's Actor definition."""
    missing = [key for key in EVALUATION_KEYS if key not in evaluation_config]
    if missing:
        raise ValueError("evaluation config is missing required sections: {}".format(", ".join(missing)))
    if "experiment" not in evaluation_config or "master_seed" not in evaluation_config["experiment"]:
        raise ValueError("evaluation config must define experiment.master_seed")
    merged = deepcopy(dict(model_config))
    # The evaluation seed is part of the paired external randomness definition.
    merged["experiment"] = dict(merged.get("experiment", {}))
    merged["experiment"]["master_seed"] = int(evaluation_config["experiment"]["master_seed"])
    for key in EVALUATION_KEYS:
        merged[key] = deepcopy(evaluation_config[key])
    return merged


def checkpoint_episode(model_dir: Path) -> int:
    """Read the saved best-checkpoint episode when available for provenance."""
    info_path = model_dir / "checkpoints" / "best_validation" / "checkpoint_info.json"
    if not info_path.exists():
        return -1
    with info_path.open("r", encoding="utf-8") as stream:
        return int(json.load(stream).get("episode", -1))


def summary_row(model_dir: Path, raw: Mapping, summary: Mapping) -> Mapping:
    """Flatten one model's validation result into a compact comparison row."""
    actor = raw.get("actor", {})
    return {
        "model_id": model_dir.name,
        "nn_mean_VAoI": float(summary["nn_mean_VAoI"]),
        "stage1_only_mean_VAoI": summary.get("stage1_only_mean_VAoI"),
        "matched_random_mean_VAoI": float(summary["matched_random_mean_VAoI"]),
        "fixed_random_mean_VAoI": float(summary["fixed_random_mean_VAoI"]),
        "nn_mean_action_probability": float(summary["nn_mean_action_probability"]),
        "stage1_only_mean_action_probability": summary.get("stage1_only_mean_action_probability"),
        "matched_random_mean_action_probability": float(summary["matched_random_mean_action_probability"]),
        "fixed_random_mean_action_probability": float(summary["fixed_random_mean_action_probability"]),
        "nn_min_action_probability": float(summary["action_prob_min"]),
        "nn_max_action_probability": float(summary["action_prob_max"]),
        "nn_actual_tx_ratio": float(summary["nn_actual_tx_ratio"]),
        "stage1_only_actual_tx_ratio": summary.get("stage1_only_actual_tx_ratio"),
        "matched_random_actual_tx_ratio": float(summary["matched_random_actual_tx_ratio"]),
        "fixed_random_actual_tx_ratio": float(summary["fixed_random_actual_tx_ratio"]),
        "delta_vaoi_matched_mean": float(summary["delta_vaoi_matched_mean"]),
        "delta_vaoi_fixed_mean": float(summary["delta_vaoi_fixed_mean"]),
        # Keep provenance columns at the right edge for a compact metric view.
        "model_directory": str(model_dir),
        "checkpoint": str(model_dir / DEFAULT_CHECKPOINT),
        "checkpoint_episode": checkpoint_episode(model_dir),
        "architecture_version": actor.get("architecture_version", "unknown"),
        "residual_architecture": actor.get("residual_architecture", "none"),
        "n_scenarios": int(summary["n_scenarios"]),
    }


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    """Write a compact, stable CSV while keeping Python-side values unrounded."""
    if not rows:
        raise ValueError("cannot write an empty comparison CSV")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "{:.5f}".format(float(value))
                if isinstance(value, (float, np.floating)) and np.isfinite(value) else value
                for key, value in row.items()
            })


def compare_models(
    model_dirs: Sequence[Path], output_dir: Path, evaluation_config_path: Path, overwrite: bool = False,
) -> Path:
    """Restore, evaluate, and persist a shared fixed-validation comparison."""
    model_dirs = [Path(path).resolve() for path in model_dirs]
    if len(set(model_dirs)) != len(model_dirs):
        raise ValueError("model directories must be unique")
    raw_configs = []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            raise FileNotFoundError("model directory does not exist: {}".format(model_dir))
        config_path, checkpoint = model_dir / "training_config.yaml", model_dir / DEFAULT_CHECKPOINT
        if not config_path.is_file() or not checkpoint.with_suffix(".index").is_file():
            raise FileNotFoundError("model directory needs training_config.yaml and best_validation/model: {}".format(model_dir))
        raw_configs.append(read_yaml(config_path))
    evaluation_config_path = Path(evaluation_config_path).resolve()
    if not evaluation_config_path.is_file():
        raise FileNotFoundError("evaluation config does not exist: {}".format(evaluation_config_path))
    evaluation_config = read_yaml(evaluation_config_path)
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "comparison_summary.csv"
    if summary_path.exists() and not overwrite:
        raise FileExistsError("comparison output already exists: {}; use --overwrite".format(summary_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, scenario_rows = [], []
    for model_dir, raw in zip(model_dirs, raw_configs):
        # Reconstruct precisely from the model's persisted actor configuration.
        actor = _stage1_actor_config(raw)
        critic = _critic_config(raw)
        actor_learning_rate, critic_learning_rate = resolve_learning_rates(raw.get("training", {}))
        model = CTDEPPO(
            learning_rate=float(raw.get("training", {}).get("learning_rate", 3e-4)),
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            clip_ratio=float(raw.get("training", {}).get("clip_ratio", 0.2)),
            entropy_coefficient=float(raw.get("training", {}).get("entropy_coefficient", 0.0)),
            seed=int(raw.get("experiment", {}).get("master_seed", 0)),
            actor_config=actor, critic_config=critic,
        )
        try:
            model.restore(str(model_dir / DEFAULT_CHECKPOINT))
            # Use the shared YAML only for the simulator and fixed scenarios.
            evaluation_raw = with_evaluation_settings(raw, evaluation_config)
            summary, per_scenario = evaluate_fixed_validation(
                model, evaluation_raw, checkpoint_episode(model_dir), model_dir.name
            )
        finally:
            model.close()
        summary_rows.append(summary_row(model_dir, raw, summary))
        for row in per_scenario:
            tagged = dict(row)
            tagged.update({"model_id": model_dir.name, "model_directory": str(model_dir)})
            scenario_rows.append(tagged)
    summary_rows.sort(key=lambda row: row["nn_mean_VAoI"])
    write_csv(summary_path, summary_rows)
    write_csv(output_dir / "comparison_per_scenario.csv", scenario_rows)
    with (output_dir / "comparison_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "models": [str(path) for path in model_dirs],
            "evaluation_config": str(evaluation_config_path),
            "checkpoint_relative_path": str(DEFAULT_CHECKPOINT),
            "comparison_summary_csv": "comparison_summary.csv",
            "comparison_per_scenario_csv": "comparison_per_scenario.csv",
            "note": "Every model uses its own training_config.yaml for architecture and weights; the shared evaluation config defines all validation scenarios.",
        }, stream, indent=2, sort_keys=True)
    return output_dir


def main() -> None:
    """Compare configured defaults, or command-line overrides, and write the shared table."""
    parser = argparse.ArgumentParser(description="Compare best_validation checkpoints on one shared evaluation YAML")
    parser.add_argument("--model-dirs", nargs="+", help="one or more result directories; overrides DEFAULT_MODEL_DIRS")
    parser.add_argument("--output-dir", help="comparison output directory; overrides DEFAULT_OUTPUT_DIR")
    parser.add_argument(
        "--evaluation-config", help="common simulator and scenario YAML; overrides DEFAULT_EVALUATION_CONFIG"
    )
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true", default=DEFAULT_OVERWRITE,
        help="replace existing comparison CSV files (the no-argument default)",
    )
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="fail instead of replacing an existing comparison CSV",
    )
    args = parser.parse_args()
    model_dirs = args.model_dirs if args.model_dirs else DEFAULT_MODEL_DIRS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    evaluation_config = Path(args.evaluation_config) if args.evaluation_config else DEFAULT_EVALUATION_CONFIG
    output = compare_models(model_dirs, output_dir, evaluation_config, args.overwrite)
    print("Saved comparison artifacts to {}".format(output))


if __name__ == "__main__":
    main()
