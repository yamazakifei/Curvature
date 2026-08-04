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

The generated scan uses 200 slots per scenario by default to keep the full
scale sweep practical; pass `--slots 500` to align the evaluation horizon with
the original fixed-validation configurations.

The script writes one YAML per topology and node count under
`result_GNN/0804Scalability/configs/`, evaluates both restored models, and
outputs `scalability_per_scenario.csv`, `scalability_summary.csv`, metadata, and
separate PNG/PDF line plots for the two topologies.

## GNN update-probability generalization (2026-08-04)

Added `scripts/run_gnn_update_probability_generalization.py` for the paired
N=100 sweep at `u=0.05, 0.10, 0.15, 0.20, 0.25, 0.30`. Community and
random-geometric topologies each use five fixed scenarios per update
probability; the original N=100 geometry is kept unchanged. Results and the
12 generated YAML files are stored under `result_GNN/0804Scalability_u/`.

The default horizon is 200 slots per scenario, with `--slots 500` available for
strict alignment with the original fixed-validation horizon. The output has
one combined per-scenario CSV and one combined summary CSV, plus separate
community/random PNG/PDF update-probability curves.
The reported sweep in `result_GNN/0804Scalability_u/` was run with
`--slots 500` to match the original N=100 validation horizon.

## Stage-1 fixed c_kappa center analysis (2026-08-04)

Added `scripts/run_stage1_center_kappa_analysis.py`. It reuses the exact
N=50, 60, ..., 150 configurations from `result_GNN/0804Scalability_N/` and
computes the node-level `c_kappa` distribution for five scenarios at each
node count. It compares those distributions with each model's frozen
training center, then evaluates the same N=150 scenarios after replacing
only the curvature center with the pooled N=150 test center. The paired
evaluation keeps the restored checkpoint and all scenario seeds unchanged;
the existing 200-slot scalability rows provide the fixed-center baseline.

Outputs are stored under `result_GNN/0804Stage1_Center_kappa_analysis/`,
including center statistics CSVs, N=150 paired comparison CSVs, PNG/PDF
figures, copied YAML configurations, and metadata.

## Community-trained models on random N=100 scenarios (2026-08-04)

Added `scripts/compare_community_models_on_random_n100.py` to compare the
community-trained curvature MPNN, its community-trained no-curvature ablation,
and the corresponding random-trained curvature MPNN on the same five random
geometric N=100 scenarios. The evaluation uses the existing 200-slot
`result_GNN/0804Scalability_N/configs/random/n100.yaml` configuration and
stores the three-model comparison under
`result_GNN/mpnnV3_heuristic_channel_n100_u0.20_b0.10_a1.5/random_n100_generalization/`.
