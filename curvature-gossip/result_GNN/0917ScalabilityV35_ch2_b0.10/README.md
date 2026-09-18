# V3.5 ch2 community scalability

This result evaluates the `best_validation_Bmax` checkpoint from the V3.5 ch2 curvature-attention MPNN and the V3.5 ch2 no-curvature MPNN on a fixed community N-generalization sweep (`N=50..150`, step 10, five scenarios per N, 200 slots per scenario). The scenario physics are inherited from the V3.5 ch2 training configuration, while the community geometry is scaled with `sqrt(N/100)` to keep density stable.

The third curve uses the curvature model checkpoint's Stage-1 base probability output directly; Stage-2 residual inference is disabled for that curve. Every point is annotated with the mean Actor broadcast probability, and the shaded regions are normal-approximation 95% CIs over the five scenarios.

## N=100 maximum VAoI distribution

`max_vaoi_n100_distribution.png` compares the per-slot maximum VAoI over five N=100 community scenarios and 200 slots per scenario for the three V3.5 ch2 modes. The x-axis labels and annotations include the mean Actor broadcast probability. Raw samples and percentile statistics are stored in the two CSV files.

## N=100 probability versus node curvature

`broadcast_probability_vs_node_curvature_n100.png` contains six panels for the three V3.5 ch2 modes. The first row uses a fixed reproducibly random slot from a 500-slot N=100 scene; the second row uses each node's mean output probability over all 500 slots. The x-axis is the minimum incident-edge curvature, and the panel titles report the mean broadcast probability.
