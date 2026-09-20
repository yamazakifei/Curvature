# Curvature-Gossip Project Status

## Single-node silence fault evaluation (2026-09-15)

The independent `result_mask/` experiment now evaluates single-node silence
faults without changing the existing `curvature-gossip/src` implementation.
`result_mask/run_single_node_silence.py` reuses the existing topology, curvature,
channel, random policy, and simulator components.  It samples every active
node with a fixed Bernoulli broadcast probability of 0.1 and forces the selected
fault node's action to `False` at every slot.  The fault node remains in the
simulator and can receive information; it cannot broadcast or forward packets.

The baseline and each single-node counterfactual share topology, shadowing,
source-update, fading, and random-action streams.  Results include mean AoI,
time-averaged maximum/minimum AoI, observed maximum/minimum AoI, deltas against
the paired baseline, node curvature, and Pearson/Spearman curvature-impact
statistics.  The output also records configured/mean policy broadcast
probability, actual network broadcast ratio, active-node actual ratio, and the
fault-node activity ratio.  Plots are written under `result_mask/outputs/.../plots/`.

Default run:

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python result_mask/run_single_node_silence.py --config result_mask/config_single_node_silence.yaml
```

The formal run completed with 101 scenarios (one baseline plus all 100 node
failures) at 5000 slots and 100 warm-up slots.  The paired baseline had
`mean_aoi=5.59837`, `mean_max_aoi=21.67163`, and an actual network broadcast
ratio of `0.099884`; the configured and mean policy broadcast probability was
`0.1`.  Across the 100 silent-node cases, the mean increase in `mean_aoi` was
`2.53262`, while the mean increase in time-averaged maximum AoI was `232.90246`.
The time-averaged minimum AoI was zero in every case, so its change is not
informative for this topology and traffic setting.  The raw node-curvature
Spearman correlation was `-0.33535` for mean-AoI degradation and `0.02902` for
time-averaged maximum-AoI degradation.  These are single-topology,
single-channel-seed, single-update-seed results and should be treated as an
initial diagnostic rather than a general conclusion about curvature.

A paired formal run with `source.update_probability=0.2` was then completed
under `result_mask/outputs/single_node_silence_p01_update_u02/`, using the same
topology, broadcast probability, slots, and random seeds.  It also contains
101 scenarios and has zero fault-node broadcast activity in all 100 fault
cases.  Its baseline was `mean_aoi=11.23061`, `mean_max_aoi=38.48633`, with
actual broadcast ratio `0.099884` and configured/mean policy broadcast
probability `0.1`.  The mean fault-induced increases were `5.06193` for mean
AoI and `470.25135` for time-averaged maximum AoI.  Raw node-curvature
Spearman correlations were `-0.39719` for mean-AoI degradation and `-0.03233`
for maximum-AoI degradation; minimum AoI remained zero for all scenarios.

## Cross-N community-topology training (2026-08-06)

The new cross-N configuration is
`curvature-gossip/configs/GNN/mpnnV3.2_ch1_cross_n80_120.yaml`.  It trains the
Stage-2 MPNN on the same soft two-community family at N=80/90/100/110/120,
with only the N condition changing.  The community geometry scales with
sqrt(N/100) while the physical communication radius stays fixed, preserving
node density, local edge distances, and mean degree approximately across N.
The configured cross-community edge range is N-dependent (2--4 through
4--6), and generated topology metadata records density, mean degree,
conductance, and normalized-cut strength for audit.

Stage-1 SearchBase now uses balanced N schedules independently for its
calibration and search pools.  The generated `stage1_calibration.json`,
`stage1_search_per_scenario.csv`, and runtime profile record the N schedules,
so intercept calibration is pooled across N instead of being implicitly fixed
at one topology size.  The training command is:

Fixed validation also pools probability statistics across unequal node widths
per scenario, so N=80/90/100/110/120 validation no longer assumes a single
matrix width.

The completed run is canonically organized at
`curvature-gossip/result_cross/mpnnV3.2_ch1_cross_n80_120/`.  Its Stage-1
SearchBase artifacts are retained from the pooled search, while its complete
300-episode PPO history, cross-N validation history, and best checkpoint are
from the fixed-parameter run.  `integration_manifest.json` records the two
execution sources; this is one logical experiment directory, although the
SearchBase and PPO phases were executed as separate processes.

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/GNN/mpnnV3.2_ch1_cross_n80_120.yaml
```

## Stage-1 base search V3.2 runtime compression (2026-08-05)

`src/curvature_gossip/learning/stage1_search.py` now implements the V3.2
search acceleration plan. Calibration and search topologies cache their
immutable topology, AF3 result, bottleneck importance, node score vector, and
per-repeat static `PropagationModel`; every candidate still receives a fresh
`GossipSimulator` and deterministic source/fading/action RNG streams.

Coarse and full evaluations resolve `coarse_slots` and `full_slots` with
fallback to the legacy `evaluation.slots` field.  After coarse evaluation,
only `candidate_selection.coarse_keep_top_k_per_alpha` candidates per alpha,
plus unique refine candidates, enter full evaluation.  Coarse rows are
retained in `stage1_search_results.csv`; promoted candidates appear once for
their coarse measurement and once for their full measurement.  Every row has
`coarse_rank_within_alpha`, `evaluated_on_full`, and `pruned_after_coarse`.

When enabled, candidate-level `ProcessPoolExecutor` parallelism uses a
worker-local immutable scenario cache and ordered result collection, so worker
identity and completion order do not affect RNG or CSV order.  The optional
`stage1_search_runtime.json` records phase timings, simulation counts, cache
status, actual worker count, slots, and candidate counts.  V3.2's explicit
`center_mode: none` uses `q_base=sigmoid(beta0 + alpha_kappa * score)`;
omitted center fields retain the legacy fixed/auto resolution.  The trainer
also skips the old center recomputation after a successful search.

The focused regression coverage is in
`curvature-gossip/tests/test_stage1_search_v32.py`.

## Stage-2 residual common-mode stabilization (2026-08-02)

Stage-2 residual training now supports `training.common_mode_coefficient`.
The penalty uses TensorFlow unsorted-segment means grouped by rollout time
step, so residuals from different slots or episodes cannot cancel each other.
It is applied only in the centralized training loss; Actor inference remains
strictly local and does not require `step_ids`.  The Stage-2 MLP and MPNN
`delta_logit` output layers are bias-free, while all intermediate layer biases
remain unchanged.  Training history records common-mode loss, residual sign
and saturation statistics, and PPO/common/total Actor gradient norms.

## Node-conditioned Critic deduplication for MPNN V3 (2026-08-02)

The new `configs/GNN/mpnn_newC_heuristic_channel.yaml` keeps the existing
heuristic physical channel and Stage-2 MPNN Actor, while selecting the new
training-only `node_conditioned` Critic.  It outputs one value per node and
uses the revised 25-dimensional feature layout from the design: four dynamic
global features, three scenario features, eight retained Actor-visible node
features, eight exact version-age/cache and innovation features, and two
static mean-channel features.  The Actor still retains its local freshness
estimates; only the three duplicated freshness estimates are removed from the
Critic, while `neighbor_confidence_mean` remains.

Training computes `[T,N]` node GAE targets, normalizes VAoI advantages
over the flattened `[T*N]` rollout, and bootstraps the final rollout state.
The legacy `scalar_global` Critic remains the default for old configurations;
node Critic inputs are never required by Actor inference or fixed validation.
Training diagnostics and `model_metadata.json` record the Critic architecture,
input dimension, value dispersion, bootstrap statistics, and explained
variance.  The requested full run command is:

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/GNN/mpnn_newC_heuristic_channel.yaml
```

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

## ch2 节点曲率分布与排名分析（2026-09-19）

新增 `curvature-gossip/scripts/analyze_ch2_node_curvature.py`，按
`mpnnV3.5_CurvAttn_ch2_n100_u0.20_b0.30_4EdgeFeature_rawBmax/source_config.yaml`
复现 20 个固定验证拓扑，统计 2000 个节点的 AF3/归一化 AF3/ORC 节点分数，
并按场景比较三种节点排名。结果和图位于项目根目录
`ch2_node_curvature_analysis/`；训练结果目录中的 README 记录了指标定义。
