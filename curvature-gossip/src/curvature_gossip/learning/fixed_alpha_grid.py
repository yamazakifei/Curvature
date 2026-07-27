"""Evaluate fixed global Stage-1 curvature alphas on paired validation cases.

The module deliberately bypasses PPO updates.  Each candidate uses the same
frozen topology, source, channel, and action random streams as its paired
random baselines, so differences are attributable to the fixed alpha only.
"""

import copy
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from .ctde_ppo import CTDEPPO
from .trainer import _freeze_stage1_center, _stage1_actor_config
from .validation import evaluate_fixed_validation


DEFAULT_UPDATE_PROBABILITIES = (0.05, 0.10, 0.20)
DEFAULT_ALPHA_VALUES = (0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00)


def _percentage_improvement(candidate: float, baseline: float) -> float:
    """Return the lower-is-better VAoI improvement percentage over a baseline."""
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("VAoI values must be finite and the baseline positive")
    return float(100.0 * (baseline - candidate) / baseline)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    """Write rows with a stable union of fields, including an empty-grid header."""
    if not rows:
        raise ValueError("cannot write an empty fixed-alpha grid result")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_update_probability(raw: Mapping, update_probability: float) -> Mapping:
    """Clone a config and apply one update rate to both training and validation."""
    configured = copy.deepcopy(dict(raw))
    probability = float(update_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("grid update probabilities must lie in [0, 1]")
    configured.setdefault("source", {})["update_probability"] = probability
    configured.setdefault("training", {})["update_probabilities"] = [probability]
    scenarios = configured.get("validation", {}).get("scenarios", [])
    if not scenarios:
        raise ValueError("fixed-alpha grid search requires validation.scenarios")
    # A candidate and both random baselines must face the same update process.
    for scenario in scenarios:
        scenario["update_probability"] = probability
    return configured


def _summary_row(summary: Mapping, update_probability: float, alpha: float, center: float) -> Mapping:
    """Flatten one fixed-validation summary into the user-facing grid CSV schema."""
    per_node_std = float(summary["action_prob_per_node_mean_std"])
    global_std = float(summary["action_prob_global_std"])
    return {
        "update_probability": float(update_probability),
        "alpha": float(alpha),
        "curvature_center": float(center),
        "n_validation_scenarios": int(summary["n_scenarios"]),
        "mean_VAoI": float(summary["nn_mean_VAoI"]),
        "matched_random_mean_VAoI": float(summary["matched_random_mean_VAoI"]),
        "fixed_random_mean_VAoI": float(summary["fixed_random_mean_VAoI"]),
        "improvement_vs_matched_random_pct": _percentage_improvement(
            summary["nn_mean_VAoI"], summary["matched_random_mean_VAoI"]
        ),
        "improvement_vs_fixed_random_pct": _percentage_improvement(
            summary["nn_mean_VAoI"], summary["fixed_random_mean_VAoI"]
        ),
        "mean_broadcast_probability": float(summary["nn_mean_action_probability"]),
        "actual_tx_ratio": float(summary["nn_actual_tx_ratio"]),
        "probability_min": float(summary["action_prob_min"]),
        "probability_max": float(summary["action_prob_max"]),
        "probability_global_std": global_std,
        "probability_global_variance": float(global_std ** 2),
        "node_mean_probability_std": per_node_std,
        "node_mean_probability_variance": float(per_node_std ** 2),
        "mean_within_slot_probability_std": float(summary["action_prob_node_std_mean"]),
        "delta_vaoi_matched_mean": float(summary["delta_vaoi_matched_mean"]),
        "delta_vaoi_matched_ci95_low": float(summary["delta_vaoi_matched_ci95_low"]),
        "delta_vaoi_matched_ci95_high": float(summary["delta_vaoi_matched_ci95_high"]),
        "nn_beats_matched_fraction": float(summary["nn_beats_matched_fraction"]),
    }


def run_fixed_alpha_grid(
    raw: Mapping,
    update_probabilities: Iterable[float] = DEFAULT_UPDATE_PROBABILITIES,
    alpha_values: Iterable[float] = DEFAULT_ALPHA_VALUES,
):
    """Return aggregate and per-scenario rows for all fixed-alpha candidates."""
    update_probabilities = tuple(float(value) for value in update_probabilities)
    alpha_values = tuple(float(value) for value in alpha_values)
    if not update_probabilities or not alpha_values:
        raise ValueError("the fixed-alpha grid must contain update probabilities and alpha values")
    if any(not 0.0 <= value <= 1.0 for value in update_probabilities):
        raise ValueError("grid update probabilities must lie in [0, 1]")
    if any(not np.isfinite(value) or value < 0.0 for value in alpha_values):
        raise ValueError("grid alpha values must be finite and nonnegative")

    base_actor = _stage1_actor_config(raw)
    if int(base_actor.get("stage", 0)) != 1:
        raise ValueError("fixed-alpha grid search requires actor.stage=1")
    # The center is topology-only; calculate it once so every grid point shares it.
    center = _freeze_stage1_center(raw, base_actor)
    aggregate_rows, scenario_rows = [], []
    seed = int(raw.get("experiment", {}).get("master_seed", 0))

    for update_probability in update_probabilities:
        rate_raw = _configure_update_probability(raw, update_probability)
        for alpha in alpha_values:
            actor = copy.deepcopy(dict(base_actor))
            curvature = dict(actor["curvature"])
            curvature.update({"center": center, "alpha_override": alpha})
            actor["curvature"] = curvature
            rate_raw["actor"] = actor
            # alpha_override makes the actor deterministic with respect to the candidate alpha.
            model = CTDEPPO(seed=seed, actor_config=actor)
            try:
                label = "u={:.2f}_alpha={:.2f}".format(update_probability, alpha)
                summary, per_scenario = evaluate_fixed_validation(model, rate_raw, 0, label)
            finally:
                model.close()
            aggregate_rows.append(_summary_row(summary, update_probability, alpha, center))
            for row in per_scenario:
                scenario_row = dict(row)
                scenario_row.update({
                    "grid_update_probability": update_probability,
                    "grid_alpha": alpha,
                    "curvature_center": center,
                })
                scenario_rows.append(scenario_row)
    return aggregate_rows, scenario_rows, center


def save_fixed_alpha_grid(
    config_path: str,
    output_directory: str,
    update_probabilities: Iterable[float] = DEFAULT_UPDATE_PROBABILITIES,
    alpha_values: Iterable[float] = DEFAULT_ALPHA_VALUES,
):
    """Load YAML, run the exact grid, and persist aggregate and scenario-level CSVs."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("fixed-alpha grid configuration must be a YAML mapping")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    aggregate_rows, scenario_rows, center = run_fixed_alpha_grid(
        raw, update_probabilities, alpha_values
    )
    _write_csv(output / "fixed_alpha_grid_summary.csv", aggregate_rows)
    _write_csv(output / "fixed_alpha_grid_per_scenario.csv", scenario_rows)
    with (output / "fixed_alpha_grid_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "source_config": str(config_path),
            "curvature_center": center,
            "update_probabilities": list(update_probabilities),
            "alpha_values": list(alpha_values),
            "summary_csv": "fixed_alpha_grid_summary.csv",
            "per_scenario_csv": "fixed_alpha_grid_per_scenario.csv",
        }, stream, indent=2, sort_keys=True)
    return output
