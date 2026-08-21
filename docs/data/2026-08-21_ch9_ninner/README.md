# N_inner figure -- the numbers, with provenance

Source: `results/ch9_ninner/converged/20260820-140643/steps.csv`
(experiment B1, isolated subordinate loop, frozen supervisory OFO,
final tuned weights `stage1/fe010aa3ead1`, commit `e5d1602`, clean tree).
No re-simulation: every number below is read off that finished run.

Criterion: `N_inner` = first subordinate iteration from which
`|q_pcc(k) - q_pcc(k-1)|` stays below **1.0 Mvar**, the same band
`settling_metrics` uses as `abs_floor` on interface flows in the
open-loop battery. Observation horizon 900 s / 20 s = 45 iterations.

| population | n | median | p90 | p95 | never flat | P(N<=9) |
|:--|--:|--:|--:|--:|--:|--:|
| capability-free (all) | 207 | 4 | 18.7 | 25.0 | 0.014 | 0.69 |
| capability-free, voltage-unconstrained | 114 | 2 | 9.8 | 14.4 | 0.009 | 0.89 |
| capability-free, voltage-constrained | 93 | 11 | 25.0 | 30.0 | 0.022 | 0.44 |

Capability filter removed 33 of 240 steps
(reported headroom < 5 Mvar, or residual beyond reported headroom).
Those converge trivially fast (median 0) because a step into a rail
stops moving at once, so excluding them RAISES the median.

## The contrast the figure has to carry

| population | arrival at `q_set`: censored | steady-state offset [Mvar] |
|:--|--:|--:|
| capability-free, voltage-unconstrained | 0.921 | 11.44 |
| capability-free, voltage-constrained | 0.871 | 11.82 |

Failure to REACH `q_set` is universal and near-identical in both
populations, with the same ~11.5 Mvar offset: voltage constraints do
NOT explain it. The supervisory parent was frozen for this run, so
`q_set` is constant and the offset cannot be a moving target -- it is
the subordinate layer's own multi-objective optimum.

What voltage constraints DO explain is speed to flatness:
median 2 vs 11 iterations, p95 14.4 vs 30.0.
Only 2.2% of voltage-constrained steps never go flat,
so a larger `N_inner` WOULD capture them -- 'no matter how big
`N_inner`' is true of the tracking criterion, not of flatness.

## Files

- `ninner_cdf.dat`  -- empirical CDF, columns `n all_free unconstrained vconstrained`;
  asymptotes below 1.0 by exactly the never-flat fraction, which is intended.
- `ninner_hist.dat` -- binned density, columns `lo hi mid all_free unconstrained vconstrained`.
- `ninner_steps.csv` -- one row per step (240), for recomputation.
