# ORC vs AF3 comparison, update_probability=0.05

The ORC root is `results/test_vaoi_vs_transmissions_u0.05`; the AF3 root is `results/test_vaoi_vs_transmissions_u0.05_af3`.

## Curvature values

- Raw edge curvature correlation is high: mean Pearson `0.907`, mean Spearman `0.936`.
- Raw curvature sign agreement is `72.0%` on average.
- ORC values are small real numbers, while AF3 values are integer structural scores; direct magnitude comparison is therefore not very meaningful.

## Bottleneck ranking

- Normalized bottleneck-importance correlation is also weak: mean Pearson `0.707`, mean Spearman `0.696`, cosine `0.731`.
- Top bottleneck-edge overlap is low: top 5% `59.6%`, top 10% `66.5%`, top 20% `72.6%`.

## Strategy input

- The actual curvature-policy input `curvature_weight * bottleneck_importance` differs substantially between ORC and AF3.
- Mean absolute input difference across C cases is `0.084`; mean input cosine similarity is `0.731`.

## Final policy metrics

- Across C points, AF3 changes mean VAoI by `0.9%` on average versus ORC, where positive means AF3 is better.
- AF3 changes average transmissions per slot by `13.4%` on average versus ORC.
- Best AF3 C point by mean VAoI is `C1`: mean_VAoI `1.976`, avg_tx_per_slot `12.186`.
- Best ORC C point by mean VAoI is `C1`: mean_VAoI `1.905`, avg_tx_per_slot `11.268`.

Interpretation: AF3 preserves much of the raw edge-curvature ordering, but it is not an identical strategy input after negative-part normalization. It selects a noticeably different set of top bottleneck edges and drives higher transmission rates for all curvature-policy configurations. The performance effect is mixed: AF3 improves C2-C4 and C6-C7, but degrades C1 and C5; the best mean-VAoI point remains ORC C1.
