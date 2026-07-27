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
