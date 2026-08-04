# Project Status

## Dual no-curvature ablation

The Stage-2 paired ablation is implemented for both the MLP Actor and the MPNN
Actor. The Actor and node-conditioned Critic use independent no-curvature
encoding paths. The simulator still computes physical curvature so that the
paired environment remains unchanged; those curvature values are not passed to
the Actor or Critic.

Configuration files:

- `configs/nn_2_layers/nn_2layers_newC_no_curvature_heuristic_channel.yaml`
- `configs/GNN/mpnn_newC_no_curvature_heuristic_channel.yaml`
- `configs/nn_2_layers/nn_2layers_newC_no_curvature_smoke.yaml`
- `configs/GNN/mpnn_newC_no_curvature_smoke.yaml`

The no-curvature Actor uses `actor.curvature.enabled: false`,
`alpha_override: 0.0`, and `residual.use_stage1_reference: false`. The MLP
residual receives only the eight non-curvature dynamic features. The MPNN keeps
the original three-column edge tensor and uses the following fixed schema:

```text
neighbor_freshness_gain
neighbor_estimate_confidence
disabled_zero_placeholder
```

The third edge column is always zero. Its tensor shape and MPNN hidden-layer
capacity are unchanged.

The no-curvature node-conditioned Critic has 22 inputs:

```text
4 global + 3 scenario + 6 local + 7 exact + 2 channel
```

The full-curvature Critic remains 25-dimensional. No-curvature encoding does
not read curvature fields and then discard them. Rollout critic inputs,
rollout-end bootstrap inputs, validation encoding, policy inference, checkpoint
reconstruction, and comparison scripts all use the same feature flags.

Metadata records:

```text
actor_curvature_enabled: false
critic_curvature_enabled: false
critic_input_dim: 22
critic_input_feature_names
mpnn_disabled_edge_feature: disabled_zero_placeholder  # MPNN only
```

No-curvature experiments train from scratch. Curvature Actor checkpoints and
25-dimensional Critic checkpoints are intentionally incompatible with the
dual no-curvature configuration and are rejected when supplied as an initial
checkpoint.

Smoke validation:

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2_layers/nn_2layers_newC_no_curvature_smoke.yaml
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/GNN/mpnn_newC_no_curvature_smoke.yaml
```

Both smoke runs completed successfully on 2026-08-03 and produced
best-validation checkpoints and metadata under `result_2layers/` and
`result_GNN/`. The full test run is currently blocked only by four pre-existing
tests that reference the already-deleted `configs/smoke_test.yaml`; all other
tests pass when those legacy smoke tests are excluded.

## GNN curvature performance comparison (2026-08-04)

Added `scripts/plot_gnn_curvature_comparison.py` to compare the paired MPNN
curvature and no-curvature experiments in `result_GNN/` for the community and
random-geometric topologies. The script selects the checkpoint with the lowest
`nn_mean_VAoI` in each model's fixed-validation history, then summarizes the
per-scenario `nn` and curvature-run `matched_random` policies over 10 scenarios.

The generated figure contains the requested three-bar comparison per topology,
plus a second panel showing the paired percentage reduction from curvature
relative to no-curvature and matched random. Error bars are normal-approximation
95% confidence intervals across validation scenarios. Outputs are written to
`result_GNN/curvature_comparison/`:

- `gnn_curvature_performance_comparison.png`
- `gnn_curvature_performance_comparison.pdf`
- `curvature_comparison_data.csv`
- `curvature_relative_gains.csv`

Run it with:

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/plot_gnn_curvature_comparison.py
```

The comparison figure now uses larger presentation-oriented fonts for the
title, axes, ticks, legend, and bar annotations. Slide-ready Chinese
conclusion text is stored in
`result_GNN/curvature_comparison/ppt_conclusion.md`.

## GNN scalability generalization (2026-08-04)

Added `scripts/run_gnn_scalability_generalization.py` for paired inference of
the current curvature and no-curvature MPNN checkpoints at node counts
50, 60, ..., 150. Community and random-geometric evaluations each use five
fixed scenarios per node count. The area and community geometry lengths scale
with `sqrt(N/100)` while the communication radius remains fixed, preserving
the approximate areal node density and radio range.

The script writes one YAML per topology and node count under
`result_GNN/0804Scalability/configs/`, evaluates both restored models, and
outputs `scalability_per_scenario.csv`, `scalability_summary.csv`, metadata, and
separate PNG/PDF line plots for the two topologies.
