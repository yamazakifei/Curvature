# V3.5 ch2 Bmax=0.30 community scalability

This result compares the V3.5 curvature-attention Stage-2 model, the V3.3 curvature mean-max model, the V3.5 no-curvature Stage-2 model, and the V3.5 curvature Stage-1-only baseline. All curves use the same existing community evaluation YAMLs (`N=50..150`, five scenarios per N, 100 slots per scenario).

The V3.3 curve is evaluated from its `best_validation_Bmax` checkpoint and is merged with the previously generated V3.5 rows. Every point is annotated with mean Actor broadcast probability; shaded regions are normal-approximation 95% confidence intervals over five scenarios.
