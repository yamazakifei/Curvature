# topo2_backoff_v2 5-seed sweep report

This report summarizes the saved sweep results in `sweep_summary.csv` and
`sweep_summary.json`. The baseline is `freshness_backoff`; the curvature policy
is `orc_freshness_backoff`. Positive gains mean improvement over the baseline.

## Experiment scope

- Seeds: topology seeds `0,1,2,3,4`; channel seed `0`; update seed `0`.
- Node scan: `n_nodes = 80, 100, 120, 150`, with `update_probability = 0.2`,
  `curvature_weight = 0.75`.
- Update scan: `update_probability = 0.05, 0.1, 0.4, 0.6`, with
  `n_nodes = 100`, `curvature_weight = 0.75`. The `0.2` case is shared with the
  node scan.
- Curvature-weight scan: `curvature_weight = 0, 0.25, 0.5, 1.0, 1.5`, with
  `n_nodes = 100`, `update_probability = 0.2`. The `0.75` case is shared with
  the node scan.

## Main observations

- Node scale: at `curvature_weight = 0.75`, mean VAoI improves by `6.53%`,
  `5.40%`, and `6.47%` for `80`, `100`, and `120` nodes, respectively. At
  `150` nodes the gain becomes `-0.39%`, so this parameter setting does not
  scale cleanly to the largest tested graph.
- Update probability: curvature helps most when updates are sparse. Mean VAoI
  gain drops from `10.13%` at `update_probability = 0.05` to `1.62%` at
  `update_probability = 0.6`.
- Curvature weight: at `n_nodes = 100` and `update_probability = 0.2`, mean VAoI
  gain improves from near zero at `0.0`, to `5.00%` at `0.5`, `5.40%` at `0.75`,
  `6.39%` at `1.0`, and `7.03%` at `1.5`.
- Tradeoff: larger curvature weights increase transmission rate. At
  `curvature_weight = 1.5`, mean VAoI improves `7.03%`, but average
  transmissions per slot increase by `11.58%` and successful decodes per
  transmission drop by `6.85%`.

## Recommendation

For the current topo2 backoff configuration, use `curvature_weight = 1.0` as the
default practical setting. It gives strong gains on mean VAoI (`6.39%`), p95 max
VAoI (`7.19%`), and dissemination delay (`7.20%`) while being less aggressive
than `1.5` in transmission overhead.

If the objective is purely minimizing mean VAoI and extra transmissions are
acceptable, `curvature_weight = 1.5` is the best tested value. If transmission
efficiency is important, keep `curvature_weight` in the `0.75-1.0` range.

The current curvature strategy is most useful under sparse or moderate update
loads (`update_probability <= 0.2`). At high update rates, gains remain positive
but become small.
