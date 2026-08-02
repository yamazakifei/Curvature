# Curvature-Gossip Project Status

## Independent Actor/Critic learning rates (2026-07-31)

CTDE-PPO now accepts `training.actor_learning_rate` and
`training.critic_learning_rate` independently. Each missing key falls back to
`training.learning_rate`, so existing YAML configurations remain unchanged.
The training entry point and checkpoint-reconstruction evaluation scripts use
the same resolution logic, and resolved values are written to
`model_metadata.json`.

## MPNN curvature edge feature (2026-08-01)

The Stage-2 MPNN edge vector is now
`neighbor_freshness_gain`, `neighbor_estimate_confidence`, and
`clipped_bottleneck_score`. The latter is
`min(max(-kappa_ij, 0), bmax) / bmax`, where `bmax` is configured under
`actor.residual.mpnn`. The full-curvature MPNN enables this feature; the
no-curvature paired ablation disables it and supplies zero in that slot.

## Local edge-message MPNN Stage-2 (2026-07-30)

Stage-2 residuals now support `architecture: mpnn` alongside the default MLP.
The MPNN uses five self features and three locally cached directed-edge
features, vectorized mean/max segment aggregation, and a zero-initialized
residual update. It has no neighbor-current-state input, hidden-state exchange,
or extra communication. All new MPNN configurations write under
`curvature-gossip/result_GNN/`.

Verified smoke configurations are
`configs/nn_2layers_stage2_mpnn_smoke.yaml` and
`configs/nn_2layers_stage2_mpnn_no_curvature_smoke.yaml`. Formal paired full
and no-curvature MPNN configurations, plus a paired no-curvature MLP baseline,
are under `curvature-gossip/configs/`.

### Fixed validation comparison update (2026-07-31)

For Stage-2 Actors, fixed validation now records a paired `stage1_only` policy
that samples the frozen curvature-base (Stage-1) probability at every node.
The validation history orders mean VAoI and mean broadcast probability as
Stage-2 NN, Stage-1-only, matched-rate random, then fixed-rate random.
`checkpoint_label` is removed from both validation CSVs, and CSV floating
values are written with five decimal places for readability.

### Batch best-checkpoint comparison (2026-07-31)

`scripts/compare_best_models.py` restores any number of
`best_validation/model` checkpoints from their own result directories and
training configurations. It verifies that all fixed-validation settings match,
then writes ranked aggregate and per-scenario CSVs to `result_compare/`,
including each NN's mean, minimum, and maximum broadcast probability.
The script's top-level `DEFAULT_MODEL_DIRS` and `DEFAULT_OUTPUT_DIR` provide
the editable no-argument batch configuration. Its common
`configs/compare/compare_n100_heuristic_channel.yaml` supplies simulator
settings and `validation.scenarios` for every restored model, while each
model's saved `training_config.yaml` still supplies its network architecture.
The public comparison YAML centralizes target transmission ratio, validation
slots, node count, and update probability as YAML anchors under
`comparison_defaults`.
Default input, output, and evaluation paths are anchored to the project root,
so direct IDE execution is independent of the current working directory. A
no-argument run refreshes the default comparison output; `--no-overwrite`
restores the protective fail-on-existing-output behavior.

## NN_2layers: Stage 1 complete

Stage 0 establishes evaluation-only baselines for the staged two-layer Actor
study.  It introduces a strictly local, fixed-alpha AF3 policy, a strict
`alpha_override=0.0` uniform degeneration, and realized-rate-matched random
evaluation through the normal experiment runner.  It also records target,
mean intended, and actual broadcast rates in every run summary.

The executable smoke configuration is
`curvature-gossip/configs/nn_2layers_stage0_smoke.yaml`.  Run it with the
project's `GRL_AoI_cpu37` environment as documented in `curvature-gossip/README.md`.

Stage 1 adds the shared, single-parameter curvature Actor and moves EWMA
settings to the observation configuration.  The verified smoke artifacts are
under `curvature-gossip/result_NN/nn_2layers_stage1_smoke/`.  Stage 2 residual
MLP work remains out of scope.

The formal Stage 1 configuration is
`curvature-gossip/configs/nn_2layers_stage1_single.yaml`.  It reuses the
100-node two-community topology and channel settings from V3 single-condition
training, and computes the fixed curvature center over all 300 training
topology seeds before optimization begins.

Its validation uses the fixed 10-scenario V3 set once every 10 completed
training episodes.  `checkpoints/best_validation/` is overwritten only when
that validation mean VAoI sets a new minimum; it deliberately does not retain
`latest` or periodic episode checkpoints.

Fixed-alpha baselines can now be evaluated without PPO training through
`curvature-gossip/scripts/search_stage1_fixed_alpha.py`.  The tool evaluates
paired fixed-validation cases and writes aggregate and per-scenario CSV files,
including matched-random and fixed-random improvements plus probability
dispersion metrics.

Stage 2 is implemented as a local-context residual MLP.  Its clean context
contains eight non-curvature local features and explicitly appends one
detached Stage-1 probability reference, yielding a nine-dimensional MLP
input.  The Stage-2 smoke configuration fixes the grid-search-selected
curvature alpha at 1.5 and trains only the residual MLP.
The formal curvature Stage-2 YAML now uses anchors and aliases to keep its
training and fixed-validation values for slots, node count, update rate, and
target rate synchronized.

## Strict no-curvature Stage-2 ablation (2026-07-29)

`configs/nn_2layers_stage2_no_curvature.yaml` is the formal ablation paired
with the heuristic-channel Stage-2 setup: it shares its `b=0.15`, physical
channel, 300-episode/400-slot training schedule, and ten fixed validation
scenarios. It uses YAML anchors and aliases for the common evaluation slots,
node count, update probability, and target rate. Its smoke counterpart is
`configs/nn_2layers_stage2_no_curvature_smoke.yaml`.  This is not merely
`alpha_override: 0`: it sets that override to zero and removes the detached
Stage-1 probability from the residual MLP, leaving the eight non-curvature
dynamic context features as its only input.  Checkpoint metadata records
`no_curvature_stage2_residual_v1`, and diagnostics record the effective alpha,
reference flag, and residual input feature list so the ablation is auditable.

Run the smoke experiment with:

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2layers_stage2_no_curvature_smoke.yaml
```

## Stage 2 heuristic-channel controlled experiment

`curvature-gossip/configs/nn_2layers_stage2_heuristic_channel.yaml` changes
only the physical channel to the one used by the tuned
`test_vaoi_vs_transmissions_u0.2` heuristic experiment.  It deliberately keeps
distributed AF3, the Stage-2 architecture, training and validation seeds, and
200 episodes unchanged so that its validation history isolates the channel
effect from curvature-backend and optimization changes.

Run the controlled experiment with:

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2layers_stage2_heuristic_channel.yaml
```

## Heuristic-channel fixed-alpha baseline (2026-07-28)

The evaluation-only configuration
`curvature-gossip/configs/nn_2layers_stage1_fixed_alpha_heuristic_channel.yaml`
holds the Stage-2 heuristic physical channel, validation seeds, `b=0.10`, and
curvature center fixed while disabling the residual layer.  The completed
15-point grid covers `u=0.05/0.10/0.20` and fixed
`alpha=0.5/1.0/1.2/1.5/1.8`.  Its canonical CSV artifacts and the direct
comparison to the Stage-2 v2 best-validation checkpoint are in
`curvature-gossip/result_2layers/fixed_alpha_grid_heuristic_channel_b0.10/`.

## Stage-2 NN point on the heuristic VAoI-transmission plot (2026-07-28)

`curvature-gossip/scripts/evaluate_stage2_on_vaoi_transmission_plot.py` restores
the Stage-2 v2 `best_validation` checkpoint and evaluates it on the five
external seed tuples used by `results/test_vaoi_vs_transmissions_u0.2`.  It
preserves that sweep's topology, source-update, channel, and fading streams,
while retaining the NN's distributed-AF3 curvature input and `b=0.10`.
The generated red-star overlay, per-seed metrics, aggregate point CSV, and
provenance record are stored under the Stage-2 result directory's
`vaoi_vs_transmissions_u0.2_overlay/` subdirectory.
