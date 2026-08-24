# T_TS sweep (thesis Sec. 9.1, selection of the supervisory period)

`rho_k = r_k / delta_k` is the fraction of the requested correction still outstanding when the next dispatch lands. `n_k` is the empirical `N_inner` *in situ*, right-censored at `T_TS / T_STS`.

**Reported as a distribution per `T_TS`, never as one time-aggregated scalar**: at a fixed horizon a short period yields proportionally more dispatches, so a pooled RMS residual mixes residual-per-dispatch with dispatch frequency.

**`N_inner` is not read off this sweep.** This is the closed-loop selection evidence for `T_TS`; the isolated measurement of eq. (9.2) is `ch_9_1_ninner_isolated_sts.py`.

## Pooled over all interfaces

| T_TS [s] | N_inner cfg | intervals | scored | rho med | rho p95 | rho max | n_k med | n_k p95 | censored | lockout occ | taps/ival | V viol frac |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60 | 3 | 13104 | 708 | 0.441 | 2.514 | 8.973 | 0.0 | 3.0 | 0.16 | 0.00 | 0.000 | 0.000 |
| 120 | 6 | 6624 | 779 | 0.309 | 2.778 | 8.632 | 0.0 | 6.0 | 0.16 | 0.00 | 0.000 | 0.000 |
| 180 | 9 | 4464 | 761 | 0.257 | 3.516 | 7.348 | 0.0 | 9.0 | 0.16 | 0.00 | 0.000 | 0.000 |
| 240 | 12 | 3312 | 727 | 0.222 | 2.968 | 6.375 | 0.0 | 12.0 | 0.16 | 0.00 | 0.000 | 0.000 |
| 300 | 15 | 2736 | 695 | 0.191 | 2.598 | 6.749 | 0.0 | 15.0 | 0.17 | 0.00 | 0.000 | 0.000 |

## Per interface (STS)

| T_TS [s] | group | intervals | scored | rho med | rho p95 | rho max | n_k med | n_k p95 | censored | CAIR width med [Mvar] | lockout occ | taps/ival | V viol frac | max z_slack |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60 | DSO_1 | 3276 | 227 | 0.489 | 2.206 | 5.670 | 0.0 | 3.0 | 0.16 | 102.5 | 0.00 | 0.000 | 0.000 | 0 |
| 60 | DSO_2 | 3276 | 182 | 0.437 | 2.114 | 6.128 | 0.0 | 3.0 | 0.18 | 92.3 | 0.00 | 0.000 | 0.000 | 0 |
| 60 | DSO_3 | 3276 | 186 | 0.532 | 3.357 | 8.973 | 0.0 | 3.0 | 0.17 | 106.0 | 0.00 | 0.000 | 0.000 | 0 |
| 60 | DSO_4 | 3276 | 113 | 0.289 | 2.080 | 3.790 | 0.0 | 3.0 | 0.13 | 85.5 | 0.00 | 0.000 | 0.002 | 8.65e-05 |
| 120 | DSO_1 | 1656 | 239 | 0.272 | 2.329 | 5.405 | 0.0 | 6.0 | 0.17 | 102.5 | 0.00 | 0.001 | 0.000 | 0 |
| 120 | DSO_2 | 1656 | 203 | 0.325 | 2.146 | 6.227 | 0.0 | 6.0 | 0.15 | 92.4 | 0.00 | 0.000 | 0.000 | 0 |
| 120 | DSO_3 | 1656 | 203 | 0.527 | 3.122 | 8.632 | 0.0 | 6.0 | 0.18 | 106.4 | 0.00 | 0.000 | 0.000 | 0 |
| 120 | DSO_4 | 1656 | 134 | 0.174 | 2.843 | 5.921 | 0.0 | 6.0 | 0.12 | 85.5 | 0.00 | 0.000 | 0.000 | 0 |
| 180 | DSO_1 | 1116 | 234 | 0.288 | 3.079 | 6.791 | 0.0 | 9.0 | 0.17 | 102.6 | 0.00 | 0.001 | 0.000 | 0 |
| 180 | DSO_2 | 1116 | 196 | 0.238 | 3.115 | 5.863 | 0.0 | 9.0 | 0.15 | 92.4 | 0.00 | 0.000 | 0.000 | 0 |
| 180 | DSO_3 | 1116 | 200 | 0.594 | 4.107 | 7.348 | 0.0 | 9.0 | 0.21 | 106.8 | 0.00 | 0.000 | 0.000 | 0 |
| 180 | DSO_4 | 1116 | 131 | 0.165 | 3.531 | 4.480 | 0.0 | 9.0 | 0.12 | 85.6 | 0.00 | 0.000 | 0.000 | 0 |
| 240 | DSO_1 | 828 | 221 | 0.222 | 2.819 | 5.647 | 0.0 | 12.0 | 0.17 | 102.6 | 0.00 | 0.001 | 0.000 | 0 |
| 240 | DSO_2 | 828 | 183 | 0.205 | 2.241 | 4.898 | 0.0 | 12.0 | 0.15 | 92.5 | 0.00 | 0.000 | 0.000 | 0 |
| 240 | DSO_3 | 828 | 190 | 0.462 | 4.085 | 6.375 | 0.0 | 12.0 | 0.21 | 107.4 | 0.00 | 0.000 | 0.000 | 0 |
| 240 | DSO_4 | 828 | 133 | 0.156 | 2.835 | 4.451 | 0.0 | 12.0 | 0.12 | 85.6 | 0.00 | 0.000 | 0.000 | 0 |
| 300 | DSO_1 | 684 | 202 | 0.204 | 2.090 | 4.065 | 0.0 | 15.0 | 0.17 | 102.6 | 0.00 | 0.000 | 0.000 | 0 |
| 300 | DSO_2 | 684 | 181 | 0.162 | 2.423 | 5.691 | 0.0 | 15.0 | 0.16 | 92.6 | 0.00 | 0.000 | 0.000 | 1 |
| 300 | DSO_3 | 684 | 183 | 0.365 | 3.371 | 6.749 | 0.0 | 15.0 | 0.22 | 107.7 | 0.00 | 0.000 | 0.000 | 0 |
| 300 | DSO_4 | 684 | 129 | 0.119 | 2.298 | 3.506 | 0.0 | 15.0 | 0.12 | 85.6 | 0.00 | 0.000 | 0.000 | 0 |

## What rho can and cannot show

`rho_k` measures how well the subordinate layer executed the correction it was **told** to make. It cannot show whether that correction was still the right one by the time it landed, so a stale-setpoint cost at long `T_TS` does **not** appear here and the absence of a U-shape in `rho` is not evidence against one. That cost lives in the supervisory tracking objective (`f_ts`, `f_q`), which this script does not compute.

## Reading the lockout column

`lockout occ` is the mean fraction of subordinate iterations in a dispatch interval during which the tap changer was unavailable. It is **inferred** from observed tap moves and the two cooldown mechanisms (`oltc_cooldown_s` wall-clock and `int_cooldown` iterations), not logged by the runner: a changer that did not move because the controller chose not to move it is indistinguishable here from one that could not. It matters most at `T_TS = 60 s`, where it separates *the subordinate layer cannot converge in three iterations* from *the changer was unavailable*.

## Provenance

- run: `2026-08-21T18:39:27.389272+02:00`
- commit: `None` on `None`
- weights: campaign `stage1` candidate `fe010aa3ead1`, archived weights reproduced; DSO-4 correction asserted
- archived `rho_emp_p95` of that candidate: `1.378849148072478`
- settling band 1.0 Mvar, delta floor 1.0 Mvar
- `T_STS` = 20 s fixed (`dso_period_s = dt_s`), bank `tier1_design_set` on `rural_700`
- command: `Z:\Python_Projekte\qOFO_GH\results\_codex_ninner_gwd_20260821\experiments\ch_9_parameter_selection\ch_9_1_ts_period_sweep.py --workers 16 --out Z:\Python_Projekte\qOFO_GH\docs\data\2026-08-21_ch9_b1_b2_corrected --label B2_ts_period_sweep`