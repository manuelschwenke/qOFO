# N_inner on the isolated subordinate loop (thesis eq. 9.2)

`N_inner` is the first subordinate iteration after an interface-Q setpoint step from which the per-iteration flow change `|q_pcc(k) - q_pcc(k-1)|` stays within the configured flatness tolerance. Right-censored at the end of the observation window; censoring is reported, never dropped.

## This is a check on the guess, not an independent measurement

`G_w` was calibrated with `N_inner = 9` **assumed**, and this measures `N_inner` with those weights in place. That is a fixed-point argument evaluated at one iteration: the guess is used, the weights follow, and this tests whether the guess survives its own consequence. Quote it that way. If the answer exceeds 9 the choice -- raise `T_TS`, or recalibrate the weights and repeat -- is an author decision, not a script output.

## Result

### Capability-limited steps are separated, not mixed in

A step is **capability-limited** when the subordinate layer's own reported headroom in the commanded direction, at the end of the window, is below the settling band -- it is saying it cannot move a further band-width that way. Such a step failing to settle measures the limit, not the loop. `N_inner` is read over the **free** steps.

| DSO | direction | steps | cap-limited | free | free N_inner med | free p95 | free censored | headroom med [Mvar] | step med [Mvar] |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_4 | down | 15 | 0 | 15 | 2.0 | 46.0 | 0.20 | 38.80 | 10.0 |
| DSO_4 | up | 15 | 0 | 15 | 2.0 | 46.0 | 0.27 | 35.79 | 10.0 |
| __pooled__ | both | 30 | 0 | 30 | 2.0 | 46.0 | 0.23 | 37.37 | 10.0 |

### All steps, including capability-limited ones

| DSO | direction | steps | N_inner med | p95 | max | censored | step med [Mvar] | band width med [Mvar] | residual med [Mvar] | taps | V viol |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_4 | down | 15 | 2.0 | 46.0 | 46 | 0.20 | 10.0 | 107.9 | 0.23 | 0 | 0 |
| DSO_4 | up | 15 | 2.0 | 46.0 | 46 | 0.27 | 10.0 | 85.5 | 0.24 | 0 | 0 |
| __pooled__ | both | 30 | 2.0 | 46.0 | 46 | 0.23 | 10.0 | 96.7 | 0.24 | 0 | 0 |

## Windows excluded (no admissible traversal)

The DER reactive capability is structurally zero in these windows (VDE dead zone), so there is no capability band to traverse and no `N_inner` to measure. They are excluded, not failed: counting them would put a spurious censoring fraction into eq. (9.2).

- `d_quiet_summer`
- `d_ramp_up_winter`

## Provenance

- run: `2026-08-21T15:38:26.248273+02:00`
- script SHA-256: `9791620967fc259c87bc3288238a371eb8457c324964d5cc4c8a77ca54fef28e`
- commit: `None` on `None`
- parent-silent variant: **frozen_ofo** (`frozen_ofo` = supervisory OFO solves once at t=0 and holds; `local` = local Q(V) supervisory control, a DIFFERENT baseline controller and a cross-check only)
- weights: campaign `stage1` candidate `fe010aa3ead1`, rebuilt weights == archived weights
- settle 600.0 s, observe 900.0 s, tracking band 1.0 Mvar, flatness tolerance 1.0 Mvar, raw target 0.95 of the reported band, applied step capped at 10.0 Mvar
- `T_STS` = 20 s, bank `tier1_design_set` on `rural_700`
- command: `C:\Users\mschwenke\AppData\Local\Temp\codex_qofo_relief_ab_401a1ba444754274bded154cd21219ba\tree\experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --workers 16 --label dso4_relief10_paired --max-step-mvar 10 --flat-mvar 1.0 --case-design balanced_half --dsos DSO_4 --dso-v-authority-override 10`