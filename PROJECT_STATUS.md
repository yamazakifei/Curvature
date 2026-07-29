# Curvature-Gossip Project Status

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
