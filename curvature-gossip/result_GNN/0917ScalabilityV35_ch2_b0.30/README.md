# V3.5 ch2 community scalability

This result evaluates the `best_validation_Bmax` checkpoint from the V3.5 ch2 curvature-attention MPNN and the V3.5 ch2 no-curvature MPNN on a fixed community N-generalization sweep (`N=50..150`, step 10, five scenarios per N, 200 slots per scenario). The scenario physics are inherited from the V3.5 ch2 training configuration, while the community geometry is scaled with `sqrt(N/100)` to keep density stable.

The third curve uses the curvature model checkpoint's Stage-1 base probability output directly; Stage-2 residual inference is disabled for that curve. Every point is annotated with the mean Actor broadcast probability, and the shaded regions are normal-approximation 95% CIs over the five scenarios.
