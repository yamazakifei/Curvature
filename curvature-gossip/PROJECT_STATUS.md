# Project Status

## Cross-N training path and validation (2026-08-06)

The cross-N YAML files under `configs/GNN/` use the project-local
`result_cross/` container for future runs.  The non-resume YAML performs the
pooled SearchBase calibration/search; the resume YAML reuses its selected
cross-N intercept and alpha and starts PPO from a fresh model.  Fixed
validation aggregates probability statistics per scenario, so N-dependent
matrix widths are supported.

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

## Ch1 model on u=0.10 scalability scenarios (2026-08-05)

Added `scripts/compare_ch1_model_on_update_probability_u10.py` to evaluate
`mpnnV3_ch1_budget_n100_u0.10_b0.10_a1.5_lowLR` on the existing community and
random-geometric u=0.10 configurations and seeds. Each topology uses five
500-slot scenarios. The comparison includes the new ch1 curvature model,
the existing topology-specific curvature model, the no-curvature model, and
the existing matched-random rows. Outputs are stored under
`result_GNN/0804Scalability_u/u0.1_gap_comparison_budget_model/`.

## u=0.10 retraining configuration consistency check (2026-08-05)

Compared `configs/GNN/mpnn_newC_heuristic_channel.yaml` with the saved
community-structure `result_GNN/0804Scalability_u/configs/community/u10.yaml`.
The topology, source update probability, channel, curvature definition,
transmission constraints, observation features, and 500-slot horizon are
consistent. The validation seeds are intentionally different: the training
configuration uses ten `v00`-`v09` scenarios, while the scalability test uses
five `u0.10_s00`-`s04` scenarios.

The `0804Scalability_u` sweep itself does not use the budget model: its
curvature baselines are `mpnnV3_heuristic_channel_n100_u0.20_b0.10_a1.5_lowLR`
and `mpnnV3_random_n100_u0.20_b0.10_a1.5_lowLR`; both saved training
configurations use `use_budget_advantage: false` and `update_multiplier: false`.
The separate `mpnnV3_ch1_budget_n100_u0.10_b0.10_a1.5_lowLR` comparison is a
different experiment and should not be used to characterize the u-sweep
baseline.

## Ch1 no-budget model comparison at u=0.10 (2026-08-05)

Extended `scripts/compare_ch1_model_on_update_probability_u10.py` with
`--model-dir` and `--output-dir` overrides so the same paired evaluation can
be reused for different Ch1 checkpoints. The script now also annotates the
aggregate VAoI bars with the mean actor broadcast probability `p`.

Evaluated `mpnnV3_ch1_n100_u0.10_b0.10_a1.5_lowLR` at its best-validation
checkpoint (episode 160) on the same five community and five random-geometric
u=0.10 scenarios used by `0804Scalability_u`. The new results are stored in
`result_GNN/0804Scalability_u/u0.1_gap_comparison_NObudget_model/`, including
PNG/PDF figures, per-scenario and summary CSVs, copied test YAMLs, and
metadata.

The no-budget Ch1 model obtains mean VAoI 2.848 (mean p=0.128) on community
topologies and 2.472 (mean p=0.132) on random-geometric topologies. Relative
to the no-curvature baseline, the VAoI reduction is 4.98% and 0.02%,
respectively; the existing curvature models reduce VAoI by 8.31% and 3.00%.

## Stage-1 Bmax-decoupled base search (2026-08-04)

Implemented the Stage-1 curvature-only coarse-to-fine search described in
`CODEX_STAGE1_BASE_SEARCH_MODIFICATION_PLAN_0804.md`. The new
`learning/stage1_search.py` calibrates one pooled `beta0` per `(b, alpha)`
candidate using a public `c_kappa` center, evaluates each alpha's local b
interval on paired simulator realizations, and injects the selected frozen
parameters into the Stage-2 Actor. `constraints.max_tx_ratio` is now Bmax and
does not directly determine the Stage-1 logit when `base_tx_ratio` or
`calibrated_intercept` is present.

`learning/seed_plan.py` writes the four-pool namespace/index manifest. Automatic
validation scenarios are generated once from `fixed_validation` and are reused
for every checkpoint; explicit `validation.scenarios` remains supported. The
recommended entry point is `configs/GNN/mpnn_newC_ch1_search.yaml`. Search
outputs include calibration JSON, aggregate/per-scenario CSVs, a search summary,
`resolved_seed_manifest.json`, and `resolved_training_config.yaml`.

Added unit coverage in `tests/test_stage1_base_search.py` and
`tests/test_seed_plan.py`, plus an Actor assertion that changing the Bmax
feature does not alter a fixed Stage-1 base probability. Existing Stage-1,
Stage-2 Actor and MPNN tests continue to pass; TensorFlow emits its existing
deprecation warnings in this Python 3.7 environment.

After Stage-1 search completes, the terminal now prints the selected `b`,
`alpha`, `beta0`, `c_kappa`, search mean probability, actual transmission
ratio, mean VAoI, feasibility, and the output directory.

## V3.2 scalability evaluations (2026-08-06)

Added `scripts/run_gnn_v32_scalability_evaluations.py` to evaluate
`mpnnV3.2_search_ch1_n100_u0.20_Bmax0.10_stage1_no_center` on the existing
`0804Scalability_N` and `0804Scalability_u` scenarios. The runner copies the
existing YAMLs instead of regenerating them, so all topology, seed, density,
and validation-horizon settings remain unchanged. It uses 200 slots for the N
sweep and 500 slots for the u sweep, while retaining the no-curvature and
matched-random comparison rows.

Outputs are stored in `result_GNN/0806ScalabilityV3.2_N/` and
`result_GNN/0806ScalabilityV3.2_u/`. Summary CSVs include mean actor broadcast
probability and its normal-approximation 95% confidence interval. The plotted
curvature legend explicitly identifies the V3.2 checkpoint.

At N=100, V3.2 mean VAoI is 5.716 (mean p=0.109) on community topologies and
4.661 (mean p=0.114) on random-geometric topologies. At u=0.10, the values
are 2.877 (mean p=0.107) and 2.472 (mean p=0.111), respectively.
