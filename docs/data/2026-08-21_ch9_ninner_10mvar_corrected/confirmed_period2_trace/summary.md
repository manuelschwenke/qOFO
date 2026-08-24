# N_inner on the isolated subordinate loop (thesis eq. 9.2)

`N_inner` is the first subordinate iteration after an interface-Q setpoint step from which the per-iteration flow change `|q_pcc(k) - q_pcc(k-1)|` stays within the configured flatness tolerance. Right-censored at the end of the observation window; censoring is reported, never dropped.

## This is a check on the guess, not an independent measurement

`G_w` was calibrated with `N_inner = 9` **assumed**, and this measures `N_inner` with those weights in place. That is a fixed-point argument evaluated at one iteration: the guess is used, the weights follow, and this tests whether the guess survives its own consequence. Quote it that way. If the answer exceeds 9 the choice -- raise `T_TS`, or recalibrate the weights and repeat -- is an author decision, not a script output.

## Result

### Capability-limited steps are separated, not mixed in

A step is **capability-limited** when the subordinate layer's own reported headroom in the commanded direction, at the end of the window, is below the settling band -- it is saying it cannot move a further band-width that way. Such a step failing to settle measures the limit, not the loop. `N_inner` is read over the **free** steps.

| DSO | direction | steps | cap-limited | free | free N_inner med | free p95 | free censored | headroom med [Mvar] | step med [Mvar] |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_4 | up | 1 | 0 | 1 | 46.0 | 46.0 | 1.00 | 116.05 | 10.0 |
| __pooled__ | both | 1 | 0 | 1 | 46.0 | 46.0 | 1.00 | 116.05 | 10.0 |

### All steps, including capability-limited ones

| DSO | direction | steps | N_inner med | p95 | max | censored | step med [Mvar] | band width med [Mvar] | residual med [Mvar] | taps | V viol |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_4 | up | 1 | 46.0 | 46.0 | 46 | 1.00 | 10.0 | 256.0 | 2.87 | 0 | 0 |
| __pooled__ | both | 1 | 46.0 | 46.0 | 46 | 1.00 | 10.0 | 256.0 | 2.87 | 0 | 0 |

## Provenance

- run: `2026-08-21T14:35:28.780686+02:00`
- script SHA-256: `18e181c93ba5fbee61d9fdbe8ecc1cf2a405807930bf0800bb8b9743347557ea`
- commit: `5d8e0f31c2aa5b4256782f6709ce4516142a2bdf` on `main`  **(working tree dirty -- not reproducible from the commit alone)**
- parent-silent variant: **frozen_ofo** (`frozen_ofo` = supervisory OFO solves once at t=0 and holds; `local` = local Q(V) supervisory control, a DIFFERENT baseline controller and a cross-check only)
- weights: campaign `stage1` candidate `fe010aa3ead1`, rebuilt weights == archived weights
- settle 600.0 s, observe 900.0 s, tracking band 1.0 Mvar, flatness tolerance 1.0 Mvar, raw target 0.95 of the reported band, applied step capped at 10.0 Mvar
- `T_STS` = 20 s, bank `tier1_design_set` on `rural_700`
- command: `experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --workers 1 --label stepcap_10mvar_reversal_spring_dso4_t10_up_trace --max-step-mvar 10 --flat-mvar 1.0 --windows d_reversal_spring --dsos DSO_4 --case-design full --focus-trafo DSO_4|trafo_10 --direction up --save-trajectories`