# 2026-08-20 — why the DSO OLTCs never move at the selected point, and which archived candidates do move them

**Type:** analysis of existing campaign output. **No code changed, no re-run.**
Source: `results/tuning_mc/stage1/evals/` (318 Tier-1 candidate records,
bank `e7b7647f8531`, 12 x 90 min).

## 1 — The observation is real and it is quantified

Selected candidate `fe010aa3ead1` (§9.3, the seven-coordinate optimum) executes
**one single DSO tap operation in the entire 12-window Tier-1 bank** — one tap
in `DSO_1`, in `d_ramp_up_winter`. `DSO_2`, `DSO_3` and `DSO_4` never move.
The TSO OLTCs move 26 times in the same bank.

## 2 — Which coordinate parks them

Two of the seven knobs gate DSO tap activity; the rest are near-inert for it.
Medians over the 146 records that share the final scoring fingerprint
`a5778a532360` (`limits_mc_v2_tier1_g6.json`, `ds_criterion=guard`):

| `engage_dso_pu` | n | median DSO taps / bank |
|---|---|---|
| 0.0125 | 2 | 255 |
| 0.025 | 45 | 43 |
| 0.0353 | 2 | 13 |
| 0.0420 | 3 | 4 |
| **0.0499** (selected) | 85 | **3** |
| 0.0593 | 3 | 1 |
| 0.0705 | 2 | 0 |

| `dso_g_v_ratio` | n | median DSO taps / bank |
|---|---|---|
| 0.7079 | 3 | 0 |
| **0.8414** (selected) | 15 | **1** |
| 1.0 | 114 | 3 (mean 44) |
| 1.4125 | 2 | 12 |
| 1.9953 | 3 | 33 |

`engage_dso_pu` is the DSO tap **commit threshold** in pu
(`tuning_mc/stage_1_search.py:28`). The search walked it from the design point
0.025 to 0.0499 — a factor 2 — and that alone accounts for roughly a factor 14
in DSO tap count. `dso_g_v_ratio` walked *down* from 1.0 to 0.841, removing a
further factor 3. The two effects are multiplicative and both point the same
way, so the parked OLTC is not a side effect: it is the direction the search
was pulled in.

## 3 — Why the search was pulled that way

The acceptance filter is two-criterion, `(f_ts, f_q)` — TS voltage and
interface-Q (`tuning_mc/metrics.py`). The DS voltage cost `f_ds` is a
*reported diagnostic*, not a filter criterion. DSO tap motion buys `f_ds` and
costs a little `f_ts` and `f_q`, so under this filter it is pure cost. The
Pareto front of the 97 feasible comparable records contains exactly three
points, and **all three sit at `dso_g_v_ratio = 0.8414` with 0–1 DSO taps**.

The only thing that stops the search going further is the g6 DS-headroom guard
added on 2026-08-18. The selected point clears it by
`g6_ds_headroom = -2.84e-4 pu` against a 1e-2 pu allowance — **2.8 % of the
margin.** The next grid step down, `dso_g_v_ratio = 0.7079`
(`037a949dd59c`), violates it (`+1.51e-3`) and is rejected. So the optimum sits
on the DS-headroom cliff, in the corner where the DSO OLTC is switched off,
and it got there because nothing in the objective pays for using it.

## 4 — Yes: there is a second parameter set, and several

Feasible, same scoring fingerprint, same bank. `d` columns are relative to
`fe010aa3ead1`. "taps" = DSO tap operations over the whole 12 x 90 min bank.

| key | d f_ts | d f_q | f_ds | g6 margin [pu] | rho_p95 | taps | worst DSO ops/h | rev/h |
|---|---|---|---|---|---|---|---|---|
| `fe010aa3ead1` (selected) | — | — | 4.2e-4 | -2.8e-4 | 1.379 | **1** | 0.67 | 0 |
| `ef544fd3f139` | +1.9 % | +5.0 % | 1.8e-4 | -2.8e-3 | 1.336 | **14** | 1.34 | 0 |
| `ac8941a46134` | +2.5 % | +10.8 % | 3e-6 | -1.0e-2 | 1.326 | **43** | 2.01 | 0 |
| `70cd9b8644a0` | +3.6 % | **-1.1 %** | 3e-5 | -5.5e-3 | 1.365 | **29** | 1.34 | 0 |
| `91587ae82942` | +3.3 % | +6.4 % | 2e-6 | -9.1e-3 | 1.289 | 43 | 2.01 | 0 |
| `30d47472848b` | +0.1 % | +6.9 % | 2.5e-4 | -2.5e-3 | 1.379 | 5 | 1.34 | 0 |

Coordinates:

```
ef544fd3f139  lambda_tso 0.150  lambda_dso 1.7957  tau 0.7079
              engage_tso_pu 0.015  engage_dso_pu 0.03531  dso_g_v_ratio 1.0  dso_v_authority 20
              g_w_der 12.0890  g_w_pcc 58.5948  g_w_dso_der 549.882
              g_w_tso_oltc 3783.055  g_w_dso_oltc 269.967   dso_g_v 100000

ac8941a46134  lambda_tso 0.150  lambda_dso 0.900  tau 1.000
              engage_tso_pu 0.015  engage_dso_pu 0.025  dso_g_v_ratio 1.0  dso_v_authority 20
              g_w_der 14.4301  g_w_pcc 54.7788  g_w_dso_der 1097.159
              g_w_tso_oltc 3783.055  g_w_dso_oltc 183.112   dso_g_v 100000

70cd9b8644a0  as ac8941a46134 but tau 0.25 -> g_w_der 7.9818, g_w_pcc 81.2885
```

(per-area relief unchanged in form: `dso_g_v_per_area = dso_g_v x 20` and
`dso_g_w_class[.]["dso_oltc"] = g_w_dso_oltc x 20` on `DSO_2`, `DSO_4`.)

`70cd9b8644a0` is the `f_total` leader of the whole comparable set and is the
only alternative that is **not** worse on interface-Q than the selected point;
it pays 3.6 % on TS voltage instead. `ac8941a46134` is the design point `X0`
plus the authority relief — i.e. the DSO OLTCs move at the *untuned* setting
and the tuning is what switched them off.

## 5 — Where the taps land (`ac8941a46134`, 43 taps)

| stratum (DSO Q-capability) | windows | taps |
|---|---|---|
| `none` | 2 | 6 |
| `partial` | 4 | 36 |
| `full` | 6 | 1 |

| role | taps |
|---|---|
| quiet | 0 |
| gen_trip | 15 |
| ramp_up | 19 |
| ramp_down | 2 |
| reversal | 7 |

The tap is used where and only where the continuous DSO-DER cannot answer:
disturbed windows in the partial/zero-capability strata. No taps in any quiet
window, none worth mentioning in the full-capability stratum, and **zero
reversals in every candidate in the table** — this is not hunting, it is the
discrete actuator covering the capability gap. Worst per-transformer rate is
2.01 ops/h in a 90-min disturbed window, which is inside the Tier-1 chatter
screen (4.0/h) but above the Tier-2 daily budget (1.25/h); the usual
event-density caveat applies and Tier 2 has not been run on these points.

## 6 — Open

* The choice is a *filter* question, not a numerical one: `f_ds` is a
  diagnostic today. Promoting it to a third filter criterion would change
  which of these points is selected. That is an architectural change and is
  not made here.
* 172 of the 318 Tier-1 records were scored before g6 (`0e2b9572d7c6`) and are
  excluded from every comparison above. Some of them are DSO-active; they
  would need re-scoring to enter the comparison.
* No Tier-2 evaluation exists for any alternative in the table.

---

# Addendum, same day — correction to §3, and the cost of changing it

## 7 — Correction: `f_ds` **was** in the filter

§3 above says `f_ds` is a diagnostic outside the filter. That is wrong for the
run that produced `fe010aa3ead1`. `tuning_mc/run_rerun_2026-08-18.sh` passes
`--ds-criterion guard --filter-ds`, so `dominates(..., with_ds=True)` was
active and the filter was three-criterion throughout. It holds 44 members.

The real mechanism is the Phase-B **incumbent-advance rule**,
`tuning_mc/stage_1_search.py:1274`:

```python
improving = [r for r in results if r["feasible"] and not any(dominates(...))]
better    = [r for r in improving if r["f_ts"] < best["f_ts"]]
```

The filter only decides *admissibility*. The incumbent walks downhill on
**`f_ts` alone**. A poll point that spends `f_ds` to buy `f_ts` is
non-dominated — so it is admitted — and it improves `f_ts` — so it becomes the
incumbent. A three-criterion filter therefore cannot, on its own, stop the
search from spending the subordinate layer's voltage. It only records what was
spent, in the filter. The eight polls walked `f_ts` 1.3084 -> 1.2809 and the
DSO tap count 43 -> 1 along the way.

## 8 — The archive is re-scorable offline; verified

`per_scenario[*]["metrics"]` carries the full flat `TrajectoryMetrics`
(locked by `tests/tuning/test_stage1_archive_rescorable.py`), and
`feasibility_constraints` reads **only** `results[*].metrics` — its `cfg`
argument is unused. So the whole hard/soft score is reconstructible without a
simulator.

Checked: rebuilding `fe010aa3ead1` from its stored metric vectors under
`PERF_WEIGHT_PROFILES["ts_voltage_primary"]` reproduces
`f_ts = 1.280885`, `f_q = 0.088986`, `f_ds = 4.170e-4`, `feasible = True` —
exact to the printed digits. All **318** tier-1 records re-score; none is
missing the metric vector. Load + parse of the archive is ~9 s; each re-score
pass over all 318 is ~0.08 s.

## 9 — Offline sweep: which limits force the tap into service

`ds_headroom_pu` (g6) is the lever. `fe010aa3ead1` holds
`min ds_headroom_min_pu = 0.010284` against a 0.010 requirement, so any
requirement above ~0.0103 makes the parked corner infeasible.

| `ds_headroom_pu` | feasible / 318 | 3-crit front | median DSO taps on front |
|---|---|---|---|
| 0.0100 (as run) | 109 | 33 | 3 |
| 0.0125 | 67 | 22 | 3 |
| 0.0150 | 35 | 12 | 42 |
| 0.0175 | 11 | 5 | 43 |

`tap_ops_per_h` (g5a) is the "not excessive" screw, and it has a floor:
g5a is the worst rate over **TSO and DSO** transformers, and the TSO OLTCs
themselves run at 2.007/h, so

* 4.0 (as run) — admits DSO rates up to 3.35/h;
* **2.5 — removes exactly the 10 chattering candidates** (max DSO 2.68–3.35/h,
  all at `engage_dso_pu <= 0.025` with `dso_g_v_ratio` 0.5 or 2.0) and leaves
  the TSO untouched;
* 2.0 — **the TSO's own 2.007/h becomes binding** and only 3 of 318 survive.

There is no DSO-only tap cap in `ConstraintLimits`; adding one is a new
constraint, i.e. a code change.

At `ds_headroom_pu = 0.015, tap_ops_per_h = 2.5`: 26 feasible, 14 on the
three-criterion front, every one of them at `engage_dso_pu = 0.025` and
`dso_g_v_ratio = 1.0`, 26–44 DSO taps per bank, worst DSO rate <= 2.01/h,
**zero DSO reversals**. `fe010aa3ead1` is not in that set, by construction.

**The archive is bimodal in DSO tap count: 26–44, or <= 5, with nothing
between.** That is the shape of an `f_ts`-descent path, not of the design
space, and it is the argument for actually re-running rather than re-selecting.

## 10 — Measured cost of a re-run

Throughput on the 40-core server at `--workers 16`, one tier-1 candidate =
12 x 90 min windows evaluated serially:

* per candidate: median **3993 s** (66.5 min), mean 3920 s, p90 4471 s (n=318);
* sustained **~14 candidates/h**;
* the run that produced `fe010aa3ead1`: 08-19 18:54 -> 08-20 08:58 = **14 h 04**,
  of which phase lam ~1 h 13, phase A (29 evals) ~2 h 19, phase B (8 polls x
  16 points) ~10 h 32.

Any change to the limits file changes `scoring_fingerprint`, which invalidates
the evaluation cache — every polled point re-simulates. No cache-rescoring
tool exists (`preflight_rerun.py` only *checks* the stamp).

---

# Addendum 2, same day — `ac8941a46134` wired into the runner

## 11 — What was changed

`experiments/run_multi_system_ofo.py`, three edits, default path untouched:

1. **New** `make_config_dso_oltc_active()` — `make_config_tuned()` with the five
   Stage-0-designed `g_w_*` scalars and `dso_g_v` of Tier-1 candidate
   `ac8941a46134`. Follows `make_config_per_area`'s idiom: strip
   `dso_g_v_per_area` / `dso_g_w_class`, replace the scalars, re-apply
   `_apply_dso_v_relief` **last**. `apply_dso_v_relief` reads an existing
   per-area `dso_oltc` entry as its base (`configs/config.py:1828`), so calling
   it on an already-relieved config squares the factor — hence the strip.
2. `main()` takes `config_factory=None` (falls back to `make_config_tuned`) and
   an `experiment` name, so an alternative weight set writes its own run series.
3. `__main__` gains a `--dso-oltc-active` branch writing to
   `results/run_multi_system_ofo_dso_oltc_active/`.

```
python experiments/run_multi_system_ofo.py --dso-oltc-active
```

## 12 — Verification

Built config vs `results/tuning_mc/stage1/evals/tier1_ac8941a46134.json`:

| field | built | archived |
|---|---|---|
| `g_w_der` | 14.430127029780047 | ✓ |
| `g_w_pcc` | 54.77878208500574 | ✓ |
| `g_w_dso_der` | 1097.158635547818 | ✓ |
| `g_w_tso_oltc` | 3783.055024849752 | ✓ |
| `g_w_dso_oltc` | 183.1121290126729 | ✓ |
| `dso_g_v` | 100000.0 | ✓ |

All exact to `1e-12` relative. Relief factor exactly 20.000 on `DSO_2`/`DSO_4`;
loop gain `dso_g_v / g_w_dso_oltc` = 546.113469 identically global and per-area,
so the invariant that keeps the integer tap out of a limit cycle holds.
`tests/test_dso_v_relief_pairing.py` — 8 passed.

`lambda_*`, `tau` and both `engage_*` are design-time Stage-0 inputs
(`tuning_mc/stage_0_preconditioning.py:724`), not runtime config fields; they
reach the plant only through those five scalars, so the six numbers above
reproduce the candidate completely.

## 13 — Two things this measurement changes

**(a) `make_config_tuned` is not at the archived optimum.** Its
`g_w_dso_oltc = 150`; the selected candidate's is 392.644. Everything else is
the optimum rounded (`10.2 / 49.3 / 550 / 3783 / 84140`). A *lower* `dso_oltc`
step weight makes taps **cheaper**, so the file in service is already more
tap-permissive than the point it claims to implement.

**(b) The runtime discriminator is the DER:OLTC price ratio, not the OLTC
loop gain.** `g_w_dso_der / g_w_dso_oltc`:

| config | `dso_g_v/g_w_dso_oltc` | `g_w_dso_der/g_w_dso_oltc` |
|---|---|---|
| `fe010aa3ead1` (archived optimum) | 214.2 | **1.40** |
| `make_config_tuned` (in service) | 560.9 | **3.67** |
| `ac8941a46134` (new) | 546.1 | **5.99** |

The OLTC loop gain of the in-service file is already *higher* than the new
candidate's — so if the 36 h case-study run shows no DSO tap motion at the
current weights, the weights are not the whole story and the window itself
(`start_time 2016-01-05 08:00`, its contingency set) may simply not stress the
DSO areas. What actually moves between the three rows is the price of the
continuous DSO-DER relative to the tap: `g_w_dso_der` doubles 550 -> 1097 while
`g_w_dso_oltc` rises only 150 -> 183.

**Open:** `ac8941a46134`'s 43 taps / bank were measured on the Tier-1 bank
(12 x 90 min, `rural_700`), not on the 36 h case-study window. Whether the tap
engages there is the point of the run and is not yet known.

---

# Addendum 3 — how `rho_emp_p95` should be quoted

Recorded because Addendum 2 quoted `9b5d7ecba760`'s `rho = 1.4357` as "1.3 %
of margin", which measures against the innermost of **three** stacked
thresholds and reads as proximity to instability. It is not.

`rho_emp_p95` is the p95 of `zone_contraction_lhs`
(`tuning/metrics.py:832`), which the coordinator defines as

    contraction_lhs = alpha * ( lambda_max(M_ii) + sum_{j!=i} ||M_ij||_2 )

with the sufficient condition **`< 2.0`** and a soft "marginal stability"
warning at `> 1.5` (`controller/multi_tso_coordinator.py:924-934`). Three
different numbers are therefore in play, and they are not the same kind of
quantity:

| threshold | value | what it is |
|---|---|---|
| stability bound | **2.0** | the sufficient condition itself |
| declared ceiling | **1.5** | a chosen 25 % design margin below it; also the coordinator's own warning level |
| search ceiling | **1.4549** | `1.5 / (1 + rho_margin)`, `rho_margin = 0.031` |

`rho_margin` is **not** a safety factor against instability. It is a
*sampling* margin: `rho_emp_p95` is reported as the worst window of the bank,
and a maximum over a small sample is downward-biased as an estimate of the same
quantity on a fresh draw. Measured on 0815, the worst-window figure moved
**3.1 %** between two banks drawn from the same distribution
(1.4743 -> 1.5201), which made the chosen point infeasible on the confirmation
set. `docs/tuning/METHOD_weight_selection.md` §5.1 states this in as many
words: *"The 1.5 ceiling is already a declared 25 % margin below the OFO bound
of 2; this is a second, statistical margin and it serves a different purpose."*

So the two candidates read:

| | rho | vs bound 2.0 | vs ceiling 1.5 | vs search ceiling 1.4549 |
|---|---|---|---|---|
| `ac8941a46134` (lambda_tso 0.15) | 1.3256 | -33.7 % | -11.6 % | -8.9 % |
| `9b5d7ecba760` (lambda_tso 0.225) | 1.4357 | **-28.2 %** | -4.3 % | -1.3 % |
| `fe010aa3ead1` (selected, 0.178) | 1.3788 | -31.1 % | -8.1 % | -5.2 % |

Consistent with `rho` being a function of `lambda_tso` alone (METHOD §3): the
0815 fit `rho = 1.0929 + 1.5446 lambda` predicts 1.3246 and 1.4404 for
`lambda_tso` 0.15 and 0.225 against 1.3256 and 1.4357 measured.

**Consequence.** Neither candidate is near the stability condition, and
`9b5d7ecba760` does not even trip the coordinator's own `> 1.5` warning. The
objection to it is **bookkeeping, not physics**: on a fresh window its
worst-window `rho` would likely read ~1.48 — still under the declared 1.5 —
but it may read over 1.4549 and would then be recorded infeasible under the
campaign's own admissibility rule. For a single-window case-study run in
`run_multi_system_ofo.py` that rule is not being applied, so the sampling
margin does not bear on it.

**For Sec. 9.3: quote `rho` against 2.0 and name the two margins separately.**
A single "1.3 % of margin" figure is not wrong arithmetic but it is the wrong
comparison, and it overstates the risk by conflating a statistical margin with
a stability one.

---

# Addendum 4 — `dQ_tr/ds`, `gamma_oltc_q`, and a diagnostic switch

## 15 — What is actually zeroed (it is not H)

The knob is **`MultiTSOConfig.dso_gamma_oltc_q`** (`configs/config.py:340`,
default **0.0**) -> `DSOControllerConfig.gamma_oltc_q`, validated to `[0, 1]`
(`controller/dso_controller.py:196`). **DSO layer only** -- there is no TSO
analogue; `gamma` does not appear in `controller/tso_controller.py`.

It does **not** zero the `dQ_tr/ds` entries of `H`.
`DSOController._build_gradient` (`controller/dso_controller.py:1216`):

```python
dQ_du = H[:n_interfaces, :]
gamma = self.config.gamma_oltc_q
if gamma < 1.0:
    oltc_slice = slice(n_der, n_der + n_oltc)
    dQ_du_q = dQ_du.copy()            # <- a COPY
    dQ_du_q[:, oltc_slice] *= gamma
grad_f += 2.0 * g_q * (q_error @ dQ_du_q)
```

`H` is untouched, so the full `dQ_tr/ds` remains in the **output rows** and the
MIQP's predicted interface Q always sees the tap -- the physical coupling is
preserved in the constraints. What `gamma = 0` removes is the tap's
**incentive**: with the Q-tracking gradient zeroed on its columns, the DSO OLTC
is driven exclusively by the voltage term
`2 g_v (V - V_set)^T dV/du` (Component 2), and only when `v_setpoints_pu` is
configured.

**This is a third, structural reason the DSO taps are quiet, independent of the
weights in §2 and §13.** At `gamma = 0` the tap cannot respond to interface-Q
error at all, however large. It is not that the tap is priced out of Q
tracking; it is not asked.

## 16 — Turning it on is a diagnostic, not a retune

Three things assume `gamma = 0` and stop holding above it:

1. **`g_w_dso_oltc` was designed at `gamma = 0`.**
   `tuning_mc/stage_0_preconditioning.py:319` applies the *same* gamma to the
   non-voltage rows of the OLTC columns when it computes the tap's self-cost
   `||a_i||^2`, explicitly so "Stage 0 and the MIQP agree". At `gamma > 0` the
   designed weight no longer prices this actuator -- including all three values
   in play (150 in file, 183.11 new, 392.64 selected).
2. **The `dso_v_authority` relief argument breaks.** It holds
   `dso_g_v / g_w_dso_oltc` constant *because* at `gamma = 0` that ratio **is**
   the OLTC loop gain (`configs/config.py:715`). With a second driver it is not,
   and the anti-limit-cycle invariant no longer covers the tap. Neighbouring
   measured failure: **50.5 reversals/h** at an unmatched x6.7 (2026-08-18).
3. **`tests/test_dso_v_relief_pairing.py:50` asserts `gamma == 0.0`**, with the
   note "gamma_oltc_q > 0 the loop-gain argument needs revisiting".

So `--gamma-q` answers *"what would the tap do if it were paid to track Q"*,
not *"what should the tap do"*.

## 17 — The switch

`experiments/run_multi_system_ofo.py`: `main()` gains `gamma_q`, `__main__`
gains `--gamma-q X` (`--gamma-q=X` also accepted). It overrides
`dso_gamma_oltc_q` on whichever config was selected, prints a banner naming the
three invalidations above, and appends `_gammaqX` to the results directory so a
diagnostic run cannot be mistaken for a tuned one. `make_config_tuned` is NOT
edited, so the pairing test still passes (8 passed). `--gamma-q` with
`--compare` raises rather than being ignored: `main_comparison()` builds its own
paired config and would silently report a `gamma=0` run as a `gamma>0` one.

```
python experiments/run_multi_system_ofo.py --dso-oltc-active --gamma-q 0.25
python experiments/run_multi_system_ofo.py --gamma-q 0.25     # default weights
```

Verified: argv parsing for both spellings; the override leaves
`dso_g_v_per_area` / `dso_g_w_class` untouched (the relief does not depend on
gamma); range enforced in [0, 1] before the run starts.

**Suggested sweep: 0.1 -> 0.25 -> 0.5, watching `tap_reversals_per_h_dso`, not
`tap_ops_per_h_dso`.** Ops rising is the intended effect; reversals rising is
the limit cycle. Going straight to 1.0 lands next to the measured failure.

---

# Addendum 5 — the commit thresholds, measured

Method: `tuning_mc.stage_0_preconditioning --from-runner` on the *runner's own*
config (so `H` is the one the plant actually presents at
`start_time 2016-01-05 08:00`), run three times. Stage 0 already computes the
quantity asked for; nothing new was derived. Its rule for integer column `i`:

    engage_Q  = ( g_w_i + ||a_i||^2 ) / ( 2 * g_q  * max|dQ_tr/ds_i| )   [Mvar]
    engage_V  = ( g_w_i + ||a_i||^2 ) / ( 2 * g_v  * |sum_j dV_j/ds_i| ) [pu]

`||a_i||^2` is the column's own curvature self-cost, so the threshold is **not**
`g_w` alone. `engage_V` is quoted for a *systematic* offset across the area (the
case an OLTC exists for); the single-bus reading is ~4x larger and is the
pessimistic one (`stage_0_preconditioning.py:609-620`).

## 18 — At the current weights, no Q deviation triggers any tap. Either layer.

`voltage_share = 1.0000` and `engage_Q = NaN` on **every** OLTC column, TSO and
DSO, for both `make_config_tuned` and `make_config_dso_oltc_active`. Two
separate causes:

* **DSO OLTC** — `dso_gamma_oltc_q = 0` zeroes the Q rows of the OLTC columns in
  the gradient (§15).
* **TSO OLTC** — `tso_g_q_pcc = 0.0` (`configs/config.py:1546`), so a TSO
  controller has no interface-Q rows at all; Stage 0 reports `g_other_typ = nan`
  for all three zones.

So the honest answer to "what Q deviation moves a tap" is **none, at any
magnitude**. The taps are voltage-triggered.

## 19 — The voltage thresholds that do apply

`make_config_dso_oltc_active` (ac8941a46134), systematic offset / single bus:

| loop | cols | `engage_V` uniform | single-bus |
|---|---|---|---|
| TSO zone 1 | 4, 5 | 0.97 – 1.07 % | 2.9 – 3.5 % |
| TSO zone 2 | 12, 13 | 2.81 – 3.95 % | 3.3 % |
| TSO zone 3 | 8 | 0.81 % | 2.8 % |
| DSO_1 | 10–12 | **2.13 – 3.09 %** | 9.9 – 11.3 % |
| DSO_2 | 10–12 | **2.13 – 3.26 %** | 9.2 – 10.4 % |
| DSO_3 | 10–12 | **2.13 – 2.93 %** | 10.8 – 12.6 % |
| DSO_4 | 10–12 | **2.12 – 3.41 %** | 8.8 – 9.9 % |

One tap step moves the strongest bus by 0.8 – 1.2 % (`max_bus_step_pu`), so a
commit needs roughly **2 – 3 tap-steps' worth of standing systematic error**.

`DSO_2`/`DSO_4` carry `g_w = 3662.24` (the x20 relief) against `183.11` on
`DSO_1`/`DSO_3`, yet their voltage thresholds are the same to ~0.1 pp. **That
is the relief's loop-gain invariance working as designed**: `dso_g_v` is x20 as
well, so the ratio that sets `engage_V` is untouched.

Two TSO columns (zone 2 col 11, zone 3 col 7) report `zero_sensitivity` — no
tap authority at all at this operating point.

## 20 — What the Q threshold *would* be at gamma > 0, and an asymmetry

`engage_Q [Mvar]`, same configs, re-run at `gamma_oltc_q = 1.0`:

| loop | `make_config_dso_oltc_active` | `make_config_tuned` |
|---|---|---|
| DSO_1 | **4.6 – 6.1** | 3.8 – 5.0 |
| DSO_3 | **3.5 – 5.5** | 2.9 – 4.6 |
| DSO_2 | **131.8 – 185.3** | 108.4 – 152.3 |
| DSO_4 | **216.7 – 297.5** | 178.2 – 244.4 |
| any TSO | NaN (no Q rows) | NaN |

`engage_Q ~ 1/gamma`, so a diagnostic at `gamma = 0.25` puts DSO_1 at ~18–24
Mvar and `gamma = 0.1` at ~46–61 Mvar.

**The asymmetry is a finding, not a rounding artefact.** The relief holds
`dso_g_v / g_w_dso_oltc` constant, which preserves the *voltage* threshold —
but there is no matching factor on `g_q`, so the *Q* threshold on `DSO_2` /
`DSO_4` rises by the full x20. At `gamma = 1` those two areas would engage at
130–300 Mvar, i.e. never: measured interface-Q RMSE on this plant is ~6 Mvar
(2026-08-18). So turning gamma on would give a **two-speed fleet** — `DSO_1` /
`DSO_3` taps responding at a few Mvar, around their own normal tracking error,
and the two spread-limited areas still Q-inert. That is almost certainly not
the intended behaviour, and it is invisible in the voltage reading.

## 21 — Caveats

* One operating point. `H` is operating-point dependent; these are the
  thresholds at `2016-01-05 08:00`, not a property of the tuning.
* Per-column scalar rule. The MIQP is coupled across columns and carries
  `local_oltc_max_step_per_dt = 1` and `oltc_cooldown_s_mt = 180 s`, so
  exceeding a threshold is **necessary, not sufficient**, for an observed step.
* `engage_V` uniform assumes the offset is the same sign across the area; a
  profile with internal spread (DSO_4: 0.147 pu) does not present that shape,
  which is exactly why the relief moved authority to the DER block instead.

---

# Addendum 6 — `make_config_tuned` now carries `dso_gamma_oltc_q = 1.0`

Manuel set it in the file (line 529). Re-measured on the file **as it stands**,
no `dataclasses.replace`, same Stage-0 rule as §18.

## 22 — The Mvar thresholds

`make_config_tuned` (gamma 1.0, `g_q = 250`, `g_w_dso_oltc = 150`,
`dso_g_v = 84140`), interface-Q deviation at which each tap column commits:

| loop | `g_w` | `engage_Q` [Mvar] | `dQ_tr/ds` [Mvar/tap] |
|---|---|---|---|
| DSO_1 col 10/11/12 | 150 | **5.01 / 3.80 / 4.56** | 0.067 / 0.092 / 0.080 |
| DSO_3 col 10/11/12 | 150 | **3.68 / 2.90 / 3.40** | 0.092 / 0.121 / 0.106 |
| DSO_2 col 10/11/12 | 3000 | **152.3 / 108.4 / 134.6** | 0.044 / 0.064 / 0.054 |
| DSO_4 col 10/11/12 | 3000 | **244.4 / 178.2 / 226.6** | 0.028 / 0.040 / 0.033 |
| any TSO OLTC | 3783 | never (`tso_g_q_pcc = 0`) | -- |

`dQ_tr/ds` recovered by inverting the rule on the stored
`(g_w_current, a_norm_weighted, engage_other_current)` with the loop's own
`g_other_typ = 250.0`; exact algebra, not an estimate.

## 23 — Two reasons the number is smaller than it looks

**(a) The tap is a very weak Q actuator here.** One tap step moves interface Q
by **0.028 - 0.121 Mvar**. So a column that commits at ~4 Mvar of error
corrects ~0.08 Mvar of it per step -- a ratio of roughly **50-90 : 1**. That is
what a constant-power load model plus constant-Q DER gives: changing the ratio
redistributes *voltage*, and the reactive flow only moves by the transformer's
own `X I^2` and the line charging. It is visible in `voltage_share`, which stays
**0.78 - 0.999 even at gamma = 1** -- the Q rows carry between 0.1 % and 22 % of
each column's weighted energy, never more.

So gamma = 1 makes the tap *nominally* Q-aware without giving it Q *authority*.
Expected outcome: taps commit somewhat more often on the mixed V+Q gradient,
interface-Q tracking barely improves, tap wear rises. This is presumably why
"DER-primary, OLTC-backup" was the original setting.

**(b) The x20 relief makes DSO_2 / DSO_4 Q-inert.** 108-244 Mvar against a
measured interface-Q RMSE of ~6 Mvar. The relief scales `g_w_dso_oltc` x20 to
hold the *voltage* loop gain, and there is no matching factor on `g_q`, so the
*Q* threshold rises by the full x20 (§20). At gamma = 1 the fleet is two-speed:
`DSO_1`/`DSO_3` engage at 3-5 Mvar, i.e. around their own normal tracking error;
the two spread-limited areas never do.

## 24 — Two consequences of the edit

1. **`tests/test_dso_v_relief_pairing.py` now fails**, 4 of 8:
   `test_relief_holds_the_oltc_loop_gain[DSO_2|DSO_4 - make_config_tuned|make_config_per_area]`,
   `assert 1.0 == 0.0`. That is the test guarding the anti-limit-cycle
   invariant, and its message is "with gamma_oltc_q > 0 the loop-gain argument
   needs revisiting". Left failing deliberately -- it is reporting a real change
   of premise, not a broken test.
2. **`make_config_dso_oltc_active` was inheriting gamma = 1.0** (it derives from
   `make_config_tuned`), which silently made its docstring false: every weight
   in it was designed by Stage 0 *at gamma = 0*. Now pinned to `0.0` in that
   factory with a comment, so it still reproduces `ac8941a46134` as archived.
   `--gamma-q X` remains the way to run that weight set with a Q incentive, and
   it is labelled a diagnostic and writes to its own results directory.

---

# Addendum 7 — the relief's third leg: per-area `g_q`

Manuel had set `DSO_V_RELIEF_FACTORS = {"DSO_2": 1.0, "DSO_4": 1.0}` -- i.e. the
relief *off* -- as the only available way to stop the x20 on `g_w_dso_oltc`
from pushing those areas' interface-Q commit threshold to 108-244 Mvar. That
also gave up the voltage relief the factor exists for. This addendum adds the
missing leg so both can hold at once, and restores the factor to 20.

## 25 — What was added

* **`MultiTSOConfig.dso_g_q_per_area`** (`configs/config.py`) -- per-area
  override of the interface-Q weight, the exact counterpart of
  `dso_g_v_per_area`.
* **`apply_dso_v_relief(..., scale_q: bool = False)`** -- scales that area's
  `g_q` by the same factor. **Default `False`**, so
  `tuning_mc.stage_1_search.build_config` and therefore the entire 0815/stage1
  campaign is bit-for-bit unchanged. Verified: rebuilding `fe010aa3ead1`
  through `build_config` returns `dso_g_q_per_area = None` and all five weights
  exact to 1e-12.
* **Runner plumbing** (`experiments/runners/multi_tso_dso.py`) -- applies
  `dso_g_q_per_area` to `DSOControllerConfig.g_q` in the same place and the same
  way as `dso_g_v_per_area`, with the same cache invalidation, plus a NOTE when
  it is set at `dso_gamma_oltc_q = 0` where it cannot affect the tap.
* **`DSO_V_RELIEF_SCALE_Q = True`** in `run_multi_system_ofo.py`, and
  `DSO_V_RELIEF_FACTORS` back to `{"DSO_2": 20.0, "DSO_4": 20.0}`.

## 26 — Measured: it does exactly what it should

Stage 0 on the runner's own config, `dso_gamma_oltc_q = 1.0`. `Q` is
`engage_other_current` [Mvar], `V` is `engage_pu_uniform_current` [%].

| loop | col | relief OFF (the workaround) | x20, no `g_q` leg | **x20 + `scale_q`** |
|---|---|---|---|---|
| | | Q / V | Q / V | Q / V |
| DSO_1 | 10/11/12 | 5.01 / 3.05, 3.80 / 2.31, 4.56 / 2.11 | identical | identical |
| DSO_3 | 10/11/12 | 3.68 / 2.91, 2.90 / 2.35, 3.40 / 2.14 | identical | identical |
| **DSO_2** | 10 | 7.65 / 3.19 | **152.25** / 3.18 | **7.65** / 3.19 |
| **DSO_2** | 11 | 5.47 / 2.31 | **108.38** / 2.29 | **5.47** / 2.31 |
| **DSO_2** | 12 | 6.77 / 2.10 | **134.63** / 2.08 | **6.77** / 2.10 |
| **DSO_4** | 10 | 12.24 / 3.33 | **244.41** / 3.33 | **12.24** / 3.33 |
| **DSO_4** | 11 | 8.94 / 2.31 | **178.19** / 2.30 | **8.94** / 2.31 |
| **DSO_4** | 12 | 11.35 / 2.07 | **226.55** / 2.07 | **11.35** / 2.07 |

`scale_q` restores the Q threshold to the **unrelieved value exactly**, while
the voltage threshold stays at its relief-preserved value. Both invariants now
read identically global and per-area:

    dso_g_v / g_w_dso_oltc = 560.9333    (voltage threshold)
    g_w_dso_oltc / g_q     = 0.6000      (interface-Q threshold)

**So: with the relief back at x20 and `scale_q` on, DSO_2 engages at ~5.5-7.7
Mvar and DSO_4 at ~8.9-12.2 Mvar**, against DSO_1's 3.8-5.0 and DSO_3's
2.9-3.7. A fleet uniform to within a factor ~3, with the DSO_4 voltage relief
intact.

## 27 — The cost, and it is not cosmetic

`g_w_dso_der` is deliberately **not** scaled, so this is not a gauge rescaling:
a relieved area's whole objective -- now both channels, not just voltage -- is
x20 against its continuous DER block. Stage 0's designed `g_w_dso_der` moves

| variant | designed `g_w_dso_der` | file runs | shortfall |
|---|---|---|---|
| relief OFF | 1160 | 550 | 2.1x |
| x20, no `g_q` leg | 1170 | 550 | 2.1x |
| **x20 + `scale_q`** | **5189** | 550 | **9.4x** |

(5189/1170 = 4.43 ~ 20^0.5, the geomean over 40 DER columns of which 20 are
scaled -- so the number is the rule, not a surprise.)

Running ~9.4x below the designed DER step weight is the **under-damped**
direction for the continuous block. The taps are now Q-responsive; the thing to
watch on the first run is the DSO-DER trajectories, not the tap counter.

## 28 — Two traps hit while doing this

1. **`apply_dso_v_relief` is not idempotent on the `g_q` leg either**, for the
   same reason as the `dso_oltc` leg: it reads an existing per-area value as its
   base. Measured on a double call: `250 -> 5000 -> 100000`. Both
   `make_config_dso_oltc_active` and `make_config_per_area` strip
   `dso_g_v_per_area` / `dso_g_w_class` before re-applying -- they now strip
   `dso_g_q_per_area` too. Caught by inspection of the built config, not by a
   test; a test for it would be worth having.
2. **`make_config_dso_oltc_active` must not get this leg at all.** It reproduces
   archived candidate `ac8941a46134`, designed before the leg existed and pinned
   at `gamma = 0` where it is inert for the tap but would still move the DER
   block. Now pinned `scale_q=False`. Weight fidelity re-verified: exact.

## 29 — Test change, stated plainly

`tests/test_dso_v_relief_pairing.py` opened with
`assert cfg.dso_gamma_oltc_q == 0.0` and the message "with gamma_oltc_q > 0 the
loop-gain argument needs revisiting". It has been revisited, so that blanket
assert is **replaced** rather than deleted: a new parametrised test
`test_relief_holds_the_oltc_q_threshold_when_the_tap_tracks_q` skips at
`gamma = 0` and, at `gamma > 0`, **requires** the `g_q` leg to be present and
`g_w_dso_oltc / g_q` to be preserved. That is a stronger guard than the one it
replaces, not a weaker one: the old test refused to look at `gamma > 0`; the new
one checks the invariant that regime needs. 12 passed.

---

# Addendum 8 — candidate `ac8941a46134` *with* the `g_q` leg

`make_config_dso_oltc_active` took no arguments and pinned `gamma = 0` /
`scale_q = False`, so the two lines of work could not be combined. It now takes
both as keyword arguments:

```python
make_config_dso_oltc_active(*, gamma_oltc_q: float = 0.0,
                            scale_q: Optional[bool] = None)
```

* `gamma_oltc_q` **defaults to 0.0** -- the no-argument call still reproduces
  the archived candidate exactly (re-verified: five weights to 1e-12,
  `dso_g_q_per_area is None`).
* `scale_q` **defaults to `gamma_oltc_q > 0`**: the leg comes on exactly when
  there is a Q gradient for it to compensate, which is the only regime where it
  affects the tap. Pass it explicitly to override in either direction.

CLI: `--dso-oltc-active --gamma-q X` now routes X into the **factory**, not
through `main()`'s post-hoc override -- the override would have set gamma
without re-deriving the leg coupled to it, i.e. silently produced the
132-297 Mvar case. `main()` still sees `gamma_q`, finds the config already
there (no-op replace), and only prints the banner and tags the directory.

## 30 — Measured

Stage 0, `gamma = 1.0`. Q = `engage_other_current` [Mvar], V = uniform [%].

| loop | col | candidate, no leg | **candidate + leg** | `make_config_tuned` + leg |
|---|---|---|---|---|
| DSO_1 | 10/11/12 | 6.09 / 4.60 / 5.53 | **6.09 / 4.60 / 5.53** | 5.01 / 3.80 / 4.56 |
| DSO_3 | 10/11/12 | 4.46 / 3.51 / 4.12 | **4.46 / 3.51 / 4.12** | 3.68 / 2.90 / 3.40 |
| DSO_2 | 10/11/12 | 185.3 / 131.8 / 163.6 | **9.30 / 6.64 / 8.22** | 7.65 / 5.47 / 6.77 |
| DSO_4 | 10/11/12 | 297.5 / 216.7 / 275.1 | **14.89 / 10.86 / 13.78** | 12.24 / 8.94 / 11.35 |

Voltage thresholds are untouched by the leg (3.26 -> 3.27 % on DSO_2 col 10,
etc. -- rounding only). Invariants at `gamma = 1`:

    dso_g_v / g_w_dso_oltc = 546.1135    global and per-area
    g_w_dso_oltc / g_q     = 0.732449    global and per-area

The candidate's thresholds sit ~20 % above `make_config_tuned`'s throughout,
which is just `g_w_dso_oltc` 183.11 vs 150 -- a more expensive tap, as designed.

## 31 — The same cost applies, slightly worse

Designed `g_w_dso_der`: **1172** without the leg, **5190** with it (against
5189 for `make_config_tuned` -- the same 20^0.5 geomean arithmetic). The
candidate ships `g_w_dso_der = 1097.16`, so with the leg it runs **4.7x below
design**, versus `make_config_tuned`'s 9.4x at 550. So this combination is
the *less* under-damped of the two on the continuous block, by a factor 2.

## 32 — The four combinations, and which answers what

| invocation | tap driver | DSO_2/4 Q threshold | reproduces archive? |
|---|---|---|---|
| `--dso-oltc-active` | voltage only | -- (no Q gradient) | **yes, exactly** |
| `--dso-oltc-active --gamma-q 1` | voltage + Q | **9.3 - 14.9 Mvar** | no (gamma > 0) |
| (no flag) | voltage + Q | 5.5 - 12.2 Mvar | n/a |
| `--dso-oltc-active --gamma-q 1`, `scale_q=False` | voltage + Q | 132 - 297 Mvar | no |

`tests/test_dso_v_relief_pairing.py`: 12 passed.
