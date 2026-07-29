"""Evaluate a trained Stage-2 model on a saved VAoI-transmissions sweep.

The script reuses the sweep's topology, source-update, channel-fading, and
policy-action seed convention for topology seeds 0--4.  It writes the neural
point, per-seed records, evaluation provenance, and an overlaid plot into the
selected model-result directory; it never overwrites the source sweep figure.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.channel import ChannelParameters, PropagationModel
from curvature_gossip.curvature import bottleneck_importance
from curvature_gossip.experiments.runner import _curvature_provider
from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import encode_stage2_observations
from curvature_gossip.random_streams import make_rng
from curvature_gossip.simulator import GossipSimulator, SimulationParameters
from curvature_gossip.topology import get_topology_generator


NN_STYLE = {"color": "#E45756", "marker": "*", "label": "N-distributed NN"}


def read_yaml(path: Path):
    """Load one YAML mapping and fail early for missing or malformed inputs."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


def ci95(values):
    """Return a mean and a normal 95 percent half-width for independent seeds."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        raise ValueError("cannot aggregate an empty result set")
    return float(np.mean(values)), (
        0.0 if values.size == 1 else float(1.96 * np.std(values, ddof=1) / math.sqrt(values.size))
    )


def evaluate_seed(model, sweep_raw, model_raw, topology_seed, channel_seed, update_seed):
    """Evaluate the frozen NN on one sweep seed with the sweep's external streams."""
    experiment = sweep_raw["experiment"]
    master_seed = int(experiment["master_seed"])
    topology = get_topology_generator(sweep_raw["topology"]["type"]).generate(
        # This exactly matches experiments.runner's topology generation stream.
        make_rng(master_seed, "topology", topology_seed), sweep_raw["topology"].get("params", {})
    )
    curvature_raw = model_raw["curvature"]
    curvature = _curvature_provider(curvature_raw).compute(topology)
    importance = bottleneck_importance(
        topology, curvature, curvature_raw.get("normalization", "local_degree_bound")
    )
    propagation = PropagationModel(
        topology.positions,
        ChannelParameters.from_mapping(sweep_raw.get("channel", {})),
        make_rng(master_seed, "shadowing", topology_seed, channel_seed),
    )
    constraints = model_raw.get("constraints", {})
    observation = model_raw.get("observation", {})
    simulator = GossipSimulator(
        topology, curvature, importance, propagation, None,
        SimulationParameters(
            slots=int(experiment["slots"]),
            update_probability=float(sweep_raw["source"]["update_probability"]),
            warmup_slots=int(experiment.get("warmup_slots", 0)),
            trace_stride=int(experiment.get("trace_stride", 1)),
            node_diagnostics_stride=0,
            target_tx_ratio=float(constraints["target_tx_ratio"]),
            per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.5)),
            congestion_ewma_beta=float(observation.get("congestion_ewma_beta", 0.8)),
            congestion_feature_scale=float(observation.get("congestion_feature_scale", 5.0)),
        ),
        # These streams exactly match the source sweep except for policy name.
        make_rng(master_seed, "updates", topology_seed, update_seed),
        make_rng(master_seed, "fading", topology_seed, channel_seed, update_seed),
        make_rng(master_seed, "policy", topology_seed, channel_seed, update_seed, "stage2_nn"),
    )
    action_rng = simulator.policy_rng
    training = model_raw.get("training", {})
    residual = model_raw["actor"]["residual"]
    probability_steps = []
    for slot in range(simulator.parameters.slots):
        observations = simulator.begin_step(slot)
        encoded = encode_stage2_observations(
            observations,
            simulator.parameters.target_tx_ratio,
            simulator.parameters.update_probability,
            float(model_raw.get("observation", {}).get("consecutive_tx_scale", 3.0)),
            float(model_raw.get("observation", {}).get("neighbor_confidence_time_constant", 20.0)),
            float(observation.get("congestion_feature_scale", 5.0)),
            bool(residual.get("include_scenario_context", False)),
        )
        probabilities = np.asarray(model.predict_probabilities(encoded), dtype=float)
        if probabilities.shape != (topology.graph.number_of_nodes(),) or not np.isfinite(probabilities).all():
            raise RuntimeError("model returned invalid probabilities on topology seed {}".format(topology_seed))
        simulator.complete_step(action_rng.random(topology.graph.number_of_nodes()) < probabilities)
        probability_steps.append(probabilities)
    summary = simulator.metrics.summary()
    summary.update(simulator.tracker.summary())
    return {
        "topology_seed": int(topology_seed),
        "channel_seed": int(channel_seed),
        "update_seed": int(update_seed),
        "mean_VAoI": float(summary["mean_VAoI"]),
        "avg_tx_per_slot": float(summary["avg_tx_per_slot"]),
        "actual_tx_ratio": float(summary["avg_tx_per_slot"] / topology.graph.number_of_nodes()),
        "mean_action_probability": float(np.mean(probability_steps)),
        "slots": int(experiment["slots"]),
        "warmup_slots": int(experiment.get("warmup_slots", 0)),
    }


def write_csv(path: Path, rows):
    """Persist nonempty dictionary rows with their first row's stable field order."""
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overlay(source_points, nn_point, output_path):
    """Recreate the source scatter styles and add the Stage-2 point prominently."""
    styles = {
        "R": {"label": "R-random", "color": "#4C78A8", "marker": "o"},
        "F": {"label": "F-freshness-backoff", "color": "#F58518", "marker": "s"},
        "C": {"label": "C-orc_freshness-backoff", "color": "#54A24B", "marker": "^"},
    }
    figure, axis = plt.subplots(figsize=(13.5, 8.2))
    for tag, style in styles.items():
        rows = [row for row in source_points if row["point_id"].startswith(tag)]
        if not rows:
            continue
        axis.errorbar(
            [float(row["avg_tx_per_slot"]) for row in rows],
            [float(row["mean_VAoI"]) for row in rows],
            xerr=[float(row["avg_tx_per_slot_ci95"]) for row in rows],
            yerr=[float(row["mean_VAoI_ci95"]) for row in rows],
            fmt=style["marker"], color=style["color"], ecolor=style["color"],
            capsize=3, markersize=7, label=style["label"], alpha=0.9,
        )
        for row in rows:
            axis.annotate(row["point_id"], (float(row["avg_tx_per_slot"]), float(row["mean_VAoI"])),
                          xytext=(5, 5), textcoords="offset points", fontsize=10,
                          color=style["color"], weight="bold")
    axis.errorbar(
        nn_point["avg_tx_per_slot"], nn_point["mean_VAoI"],
        xerr=nn_point["avg_tx_per_slot_ci95"], yerr=nn_point["mean_VAoI_ci95"],
        fmt=NN_STYLE["marker"], color=NN_STYLE["color"], ecolor=NN_STYLE["color"],
        capsize=4, markersize=14, label=NN_STYLE["label"], zorder=5,
    )
    axis.annotate("NN", (nn_point["avg_tx_per_slot"], nn_point["mean_VAoI"]),
                  xytext=(7, 7), textcoords="offset points", fontsize=12,
                  color=NN_STYLE["color"], weight="bold")
    # Enlarge all reader-facing chart typography for paper and slide use.
    axis.set_title("VAoI vs transmissions across policy configurations", fontsize=20, pad=12)
    axis.set_xlabel("Average transmissions per slot", fontsize=16)
    axis.set_ylabel("Mean VAoI", fontsize=16)
    axis.tick_params(axis="both", which="major", labelsize=14)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend(loc="best", frameon=True, fontsize=14)
    axis.margins(x=0.05, y=0.12)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    """Restore the best Stage-2 checkpoint, evaluate the five sweep seeds, and plot it."""
    parser = argparse.ArgumentParser(description="Overlay a trained Stage-2 model on a saved VAoI sweep")
    parser.add_argument("--model-dir", required=True, help="Stage-2 result directory containing training_config.yaml")
    parser.add_argument("--sweep-dir", required=True, help="source plot directory containing points.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_validation/model", help="checkpoint relative to model-dir")
    args = parser.parse_args()
    model_dir, sweep_dir = Path(args.model_dir), Path(args.sweep_dir)
    model_raw = read_yaml(model_dir / "training_config.yaml")
    # Any existing sweep resolved config carries the shared external seed and channel setup.
    source_config = sweep_dir / "test_vaoi_vs_transmissions_R02_p0p1" / "0" / "random" / "resolved_config.yaml"
    sweep_raw = read_yaml(source_config)
    topology_seeds = list(sweep_raw["experiment"]["topology_seeds"])
    channel_seed = int(sweep_raw["experiment"]["channel_seeds"][0])
    update_seed = int(sweep_raw["experiment"]["update_seeds"][0])
    actor = model_raw["actor"]
    model = CTDEPPO(
        learning_rate=float(model_raw.get("training", {}).get("learning_rate", 3e-4)),
        clip_ratio=float(model_raw.get("training", {}).get("clip_ratio", 0.2)),
        entropy_coefficient=float(model_raw.get("training", {}).get("entropy_coefficient", 0.0)),
        seed=int(model_raw["experiment"]["master_seed"]), actor_config=actor,
    )
    try:
        model.restore(str(model_dir / args.checkpoint))
        per_seed_rows = [evaluate_seed(model, sweep_raw, model_raw, seed, channel_seed, update_seed)
                         for seed in topology_seeds]
    finally:
        model.close()
    mean_vaoi, vaoi_ci95 = ci95([row["mean_VAoI"] for row in per_seed_rows])
    mean_tx, tx_ci95 = ci95([row["avg_tx_per_slot"] for row in per_seed_rows])
    nn_point = {
        "point_id": "NN", "legend": NN_STYLE["label"], "policy": "stage2_nn",
        "independent_runs": len(per_seed_rows), "avg_tx_per_slot": mean_tx,
        "avg_tx_per_slot_ci95": tx_ci95, "mean_VAoI": mean_vaoi, "mean_VAoI_ci95": vaoi_ci95,
    }
    output_dir = model_dir / "vaoi_vs_transmissions_u0.2_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stage2_nn_per_seed.csv", per_seed_rows)
    write_csv(output_dir / "stage2_nn_point.csv", [nn_point])
    with (sweep_dir / "points.csv").open("r", newline="", encoding="utf-8") as stream:
        source_points = list(csv.DictReader(stream))
    write_csv(output_dir / "points_with_stage2_nn.csv", source_points + [nn_point])
    provenance = {
        "model_checkpoint": str(model_dir / args.checkpoint),
        "model_training_config": str(model_dir / "training_config.yaml"),
        "source_sweep_config": str(source_config),
        "topology_seeds": topology_seeds,
        "channel_seed": channel_seed,
        "update_seed": update_seed,
        "topology_rng_labels": ["topology", "topology_seed"],
        "source_update_rng_labels": ["updates", "topology_seed", "update_seed"],
        "channel_fading_rng_labels": ["fading", "topology_seed", "channel_seed", "update_seed"],
        "policy_action_rng_labels": ["policy", "topology_seed", "channel_seed", "update_seed", "stage2_nn"],
        "note": "Topology/source/channel streams match the saved sweep; NN retains distributed_af3 curvature and b=0.10 from its training config.",
    }
    with (output_dir / "evaluation_provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
    plot_overlay(source_points, nn_point, output_dir / "plot_multi_vaoi_vs_transmissions_with_stage2_nn.png")
    print("NN point: tx_per_slot={:.6f}, VAoI={:.6f}".format(mean_tx, mean_vaoi))
    print("Saved overlay artifacts to {}".format(output_dir))


if __name__ == "__main__":
    main()
