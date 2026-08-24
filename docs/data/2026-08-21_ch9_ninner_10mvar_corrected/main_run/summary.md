# N_inner on the isolated subordinate loop (thesis eq. 9.2)

`N_inner` is the first subordinate iteration after an interface-Q setpoint step from which the per-iteration flow change `|q_pcc(k) - q_pcc(k-1)|` stays within the configured flatness tolerance. Right-censored at the end of the observation window; censoring is reported, never dropped.

## This is a check on the guess, not an independent measurement

`G_w` was calibrated with `N_inner = 9` **assumed**, and this measures `N_inner` with those weights in place. That is a fixed-point argument evaluated at one iteration: the guess is used, the weights follow, and this tests whether the guess survives its own consequence. Quote it that way. If the answer exceeds 9 the choice -- raise `T_TS`, or recalibrate the weights and repeat -- is an author decision, not a script output.

## Result

### Capability-limited steps are separated, not mixed in

A step is **capability-limited** when the subordinate layer's own reported headroom in the commanded direction, at the end of the window, is below the settling band -- it is saying it cannot move a further band-width that way. Such a step failing to settle measures the limit, not the loop. `N_inner` is read over the **free** steps.

| DSO | direction | steps | cap-limited | free | free N_inner med | free p95 | free censored | headroom med [Mvar] | step med [Mvar] |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_1 | down | 15 | 0 | 15 | 2.0 | 6.0 | 0.00 | 38.27 | 10.0 |
| DSO_1 | up | 15 | 0 | 15 | 4.0 | 6.0 | 0.00 | 40.91 | 10.0 |
| DSO_2 | down | 15 | 0 | 15 | 4.0 | 14.0 | 0.00 | 40.31 | 10.0 |
| DSO_2 | up | 15 | 0 | 15 | 4.0 | 17.1 | 0.00 | 35.59 | 10.0 |
| DSO_3 | down | 15 | 0 | 15 | 2.0 | 4.0 | 0.00 | 46.58 | 10.0 |
| DSO_3 | up | 15 | 0 | 15 | 4.0 | 4.0 | 0.00 | 44.04 | 10.0 |
| DSO_4 | down | 15 | 0 | 15 | 4.0 | 46.0 | 0.40 | 39.35 | 10.0 |
| DSO_4 | up | 15 | 0 | 15 | 4.0 | 46.0 | 0.40 | 35.86 | 10.0 |
| __pooled__ | both | 120 | 0 | 120 | 4.0 | 46.0 | 0.10 | 41.10 | 10.0 |

### All steps, including capability-limited ones

| DSO | direction | steps | N_inner med | p95 | max | censored | step med [Mvar] | band width med [Mvar] | residual med [Mvar] | taps | V viol |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_1 | down | 15 | 2.0 | 6.0 | 6 | 0.00 | 10.0 | 102.5 | 0.10 | 0 | 0 |
| DSO_1 | up | 15 | 4.0 | 6.0 | 6 | 0.00 | 10.0 | 102.8 | 0.03 | 0 | 0 |
| DSO_2 | down | 15 | 4.0 | 14.0 | 14 | 0.00 | 10.0 | 102.1 | 0.08 | 0 | 0 |
| DSO_2 | up | 15 | 4.0 | 17.1 | 22 | 0.00 | 10.0 | 92.2 | 0.09 | 0 | 0 |
| DSO_3 | down | 15 | 2.0 | 4.0 | 4 | 0.00 | 10.0 | 114.6 | 0.14 | 0 | 0 |
| DSO_3 | up | 15 | 4.0 | 4.0 | 4 | 0.00 | 10.0 | 114.8 | 0.04 | 0 | 0 |
| DSO_4 | down | 15 | 4.0 | 46.0 | 46 | 0.40 | 10.0 | 107.9 | 0.58 | 0 | 0 |
| DSO_4 | up | 15 | 4.0 | 46.0 | 46 | 0.40 | 10.0 | 85.5 | 0.58 | 0 | 0 |
| __pooled__ | both | 120 | 4.0 | 46.0 | 46 | 0.10 | 10.0 | 102.8 | 0.12 | 0 | 0 |

## Windows excluded (no admissible traversal)

The DER reactive capability is structurally zero in these windows (VDE dead zone), so there is no capability band to traverse and no `N_inner` to measure. They are excluded, not failed: counting them would put a spurious censoring fraction into eq. (9.2).

- `d_quiet_summer`
- `d_ramp_up_winter`

## Provenance

- run: `2026-08-21T13:44:12.448568+02:00`
- script SHA-256: `8fe629959cc19455927e322c91ec9f10806df062a718aaad8a9c3fc01b3c9777`
- commit: `5d8e0f31c2aa5b4256782f6709ce4516142a2bdf` on `main`  **(working tree dirty -- not reproducible from the commit alone)**
- parent-silent variant: **frozen_ofo** (`frozen_ofo` = supervisory OFO solves once at t=0 and holds; `local` = local Q(V) supervisory control, a DIFFERENT baseline controller and a cross-check only)
- weights: campaign `stage1` candidate `fe010aa3ead1`, rebuilt weights == archived weights
- settle 600.0 s, observe 900.0 s, tracking band 1.0 Mvar, flatness tolerance 1.0 Mvar, raw target 0.95 of the reported band, applied step capped at 10.0 Mvar
- `T_STS` = 20 s, bank `tier1_design_set` on `rural_700`
- command: `experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --workers 16 --label stepcap_10mvar_balanced_half_corrected --max-step-mvar 10 --flat-mvar 1.0 --case-design balanced_half`