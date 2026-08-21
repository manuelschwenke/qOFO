# 2026-08-21 — B2 re-run at the final weights: the DSO tap changers stop moving

**Timestamp:** 2026-08-21, 09:25–10:00 (server).
**Run:** `results/ch9_ts_period_sweep/final_weights/20260821-092509`, exit **0**,
60/60 cases, 10 workers.
**Commit:** `29522e5` on `main`, working tree **dirty** — the 7 uncommitted files
of the OLTC-inactivity session. The 1110-line diff was captured at launch, so the
run is reproducible from `29522e5` + that patch.
**Supersedes:** `results/ch9_ts_period_sweep/full/20260819-134011...`, which ran
at candidate `aa4f6d4a8654` (`rho_emp_p95` 1.4480) — the candidate the §9.3 text
was drafted against, since superseded.
**Weights:** campaign `stage1`, candidate `fe010aa3ead1` (`rho_emp_p95` 1.3788),
rebuilt through the campaign's own recipe and asserted equal to the archive, with
the per-area voltage relief on DSO_2/DSO_4 (factor 20) applied and both halves
asserted.

## The sweep result (from `summary.md`, pooled)

| `T_TS` [s] | `N_inner` cfg | scored | `rho` med | `rho` p95 | censored | lockout | taps/ival | V viol |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60 | 3 | 711 | 0.474 | 2.546 | 0.22 | 0.00 | 0.000 | 0.000 |
| 120 | 6 | 786 | 0.373 | 2.769 | 0.23 | 0.00 | 0.000 | 0.000 |
| 180 | 9 | 768 | 0.304 | 3.558 | 0.22 | 0.00 | 0.000 | 0.000 |
| 240 | 12 | 726 | 0.258 | 2.991 | 0.23 | 0.00 | 0.000 | 0.000 |
| 300 | 15 | 698 | 0.222 | 2.662 | 0.22 | 0.00 | 0.000 | 0.000 |

`rho` declines monotonically, 0.474 → 0.222. **Still no U-shape**, and that is
still not evidence against one: `rho_k` measures how well the subordinate layer
executed the correction it was *told* to make, not whether that correction was
still right when it landed. The staleness cost lives in the supervisory objective
(`f_ts`, `f_q`) and this script does not compute it. The selection argument for
`T_TS = 180 s` therefore remains **unmeasured**.

The improvement over the superseded weights is concentrated at the short-period
end — `rho` median 0.662 → 0.474 at `T_TS = 60 s` — and is negligible from 120 s
upward (0.375 → 0.373, 0.308 → 0.304). Censoring improved at every period.

## The actual finding: the subordinate OLTC has gone inert

Counted over all 30 240 intervals of each sweep, unfiltered:

| quantity | superseded weights | **final weights** |
|:--|--:|--:|
| total DSO tap moves | 584 | **3** |
| intervals with any tap move | 584 | **3** |
| total voltage violations | 93 | **0** |
| lockout occupancy (mean) | 0.022–0.025 | **0.000–0.001** |

Voltage violations are eliminated, and the tap changers essentially never move —
three moves in thirty thousand intervals. This is the same phenomenon as
`2026-08-20_dso_oltc_inactivity_at_the_tuned_point.md`, reached independently
from the period sweep, and it is at least partly **by design**: the per-area
relief raises `dso_g_v` and `g_w_dso_oltc` together to hold the OLTC loop gain
while letting the DER shape voltage, precisely so the tap does not limit-cycle.

The open question is whether that is the intended endpoint. An actuator that
moves three times in a full sweep is not being used, and §9.1's `T_mech`/`T_elec`
split and the 60 s / 180 s tap cooldowns are all reasoning about a device the
tuned controller does not operate. B1 saw the same thing from another angle:
DSO_3 stopped 28 Mvar short of setpoint across all 60 of its steps with **zero**
tap moves while reporting 30–36 Mvar of headroom.

## Windup, re-tested at the final weights

`k_t_avt = 0` on this path by author decision (2026-08-21), so the commanded
`q_set` is a pure integrator state, never re-anchored to the achieved value.

| `T_TS` | median slope of `r_k` vs k | frac > 0 | median slope of \|`q_set`\| | frac > 0 |
|--:|--:|--:|--:|--:|
| 60 | −0.005 | 0.40 | 0.18 | 0.84 |
| 120 | +0.005 | 0.63 | 0.36 | 0.92 |
| 180 | **+0.007** | 0.68 | 0.43 | 0.88 |
| 240 | +0.003 | 0.69 | 0.62 | 0.88 |
| 300 | +0.009 | 0.70 | 0.73 | 0.88 |

At the superseded weights the residual slope was negative at *every* period. At
the final weights it turns mildly positive for `T_TS >= 120 s`, in roughly
two-thirds of window/interface groups. **The magnitude is immaterial over these
windows** — at `T_TS = 180 s`, +0.007 Mvar per interval over ~25 intervals is
~0.17 Mvar of drift against a 1 Mvar band. But the sign changed, the windows are
only 90 minutes, and the anti-windup is off by choice. Worth one instrumented
long-horizon run before the thesis claims the interface request is stable; do not
extrapolate this to a day.

## Still outstanding

1. **`n_k` is still the rejected criterion.** Median 0.0 and p95 exactly equal to
   the configured `N_inner` at every period — the censoring signature, not a
   measurement. `--reaggregate` re-derives it from `intervals.csv` without
   re-simulating; it should be reworked to the flatness criterion or relabelled
   as tracking.
2. **The staleness metric** — the same sweep scored on `f_ts`/`f_q`. Without it
   there is no measured selection argument for 180 s.
3. **DSO_4 is asked for less.** It carries 149–171 scored intervals against
   216–271 for the others at the same periods. Consistent with the
   voltage-blind capability message over-reporting its usable export.
