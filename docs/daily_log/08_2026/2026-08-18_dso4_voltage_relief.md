# 2026-08-18 — DSO 4 high-voltage relief: what the three candidate weights actually do

**Reason:** After the MC-tuned weights were applied, DSO 4's HV buses sit on the
1.10 p.u. hard bound for most of the day. Feasible but undesired. Three reliefs
were proposed: lower `g_w_pcc` for this DSO in the TS-OFO, raise `dso_g_v`, or
lower `g_w_dso_oltc` to buy more tap operations. This entry records the
measurement of all three plus further variants, and identifies the actual
binding quantity.

**Reproduction:** `experiments/run_multi_system_ofo.py`, `make_config_tuned()`,
`start_time = 2016-05-03 08:00`, `n_total_s = 360 min`, live plots off.
No project code was modified — variants are applied through existing config
fields (`zone_g_w_class`, `dso_g_w_class`) or, where no per-DSO field exists
(`dso_g_v`, `v_setpoints_pu`), through the runner's own `pre_loop_hook`, which
fires after every controller is built and after the per-class `g_w` overrides.
Driver and probes: session scratchpad; records under `results/dso4_relief/`.

## 1. Why DSO 4 and not the others

`SUBNET_DEFS` (`network/ieee39/constants.py:304`) gives DSO 4 `scale = 2.44` on
the shared TUDA 110 kV topology, with the unreinforced 305 mm2 conductor. Same
load and same 700 MW installed DER as the other three.

| | DSO 1 | DSO 2 | DSO 3 | DSO 4 |
| --- | --- | --- | --- | --- |
| line length | 197 km | 336 km | 125 km | **586 km** |
| total X | 74.8 ohm | 127.7 ohm | 43.3 ohm | **222.5 ohm** |
| total C | 1.8 uF | 3.1 uF | 1.4 uF | **5.4 uF** |
| conductor | 305 mm2 | 305 mm2 | 490 mm2 | 305 mm2 |

On a 110 kV / 100 MVA base, 222.5 ohm is 1.84 p.u. of series reactance inside a
single sub-transmission network.

## 2. The binding quantity is the internal spread, not the level

Per-step `V_max - V_min` inside each HV network over the baseline run, and the
sensitivity of that spread to interface-Q duty:

| | spread mean / max [p.u.] | d(spread)/d(Q_PCC) [%V per Mvar] |
| --- | --- | --- |
| DSO 1 | 0.010 / 0.015 | -0.0080 |
| DSO 2 | 0.080 / 0.117 | -0.0984 |
| DSO 3 | 0.024 / 0.037 | -0.0241 |
| **DSO 4** | **0.106 / 0.147** | **-0.3482** |

DSO 4 consumes 73 % of the whole 0.20 p.u. statutory band on internal spread
before any control acts, and is 14x more spread-sensitive per Mvar of interface
duty than DSO 3. With `V_set = 1.03` the arithmetic is forced:
`1.03 + 0.147/2 = 1.103`, i.e. the top of the profile lands on the bound. The
observed value is 1.1001, held there by the MIQP output constraint
(`dso_z_slack_max` non-zero on 57.4 % of steps, magnitude ~1e-4).

DSO 2 is the same defect one step milder (spread 0.117, on the bound 7.3 % of
steps) and is untouched by any DSO-4-specific measure.

## 3. The OLTC is not blocked — it is at its own optimum

`dso_gamma_oltc_q = 0.0` removes the Q-tracking gradient from the OLTC columns
entirely (`controller/dso_controller.py:1211`), so the tap is driven *only* by
`2 * dso_g_v * (V - V_set)^T * dV/ds`. For an integer column the MIQP commits a
tap iff `|grad_f| > g_w + g_u`.

Reproducing the baseline end-state (taps [8,5,2], `Q_DER` = +79.6 Mvar):

| V_set | sum(V - V_set) | grad t9 | grad t10 | grad t11 | threshold | commits |
| --- | --- | --- | --- | --- | --- | --- |
| **1.03** | -0.180 | +490.7 | +123.6 | +77.1 | 183 | 1/3 |
| 1.00 | +0.120 | +219.3 | -285.8 | -404.1 | 183 | 3/3 |

At `V_set = 1.03` the profile mean has already fallen *below* the setpoint, so
the gradient is **positive** — the objective wants the 110 kV side back *up*.
`g_w_dso_oltc` and `dso_g_v` scale or threshold that gradient; neither can
change its sign. The cached dV/ds was checked against a plant finite difference
and agrees to 2–3 %, so this is not a stale-sensitivity artefact.

## 4. The reported PCC capability is voltage-blind

`DSOController.generate_capability_message` maps only the VDE-AR-N 4120 DER Q
rail through the open-loop dQ_iface/dQ_DER. No voltage constraint enters it. The
TSO applies the result as a soft output bound on interface Q
(`controller/tso_controller.py:1273`). Sweeping the DER rail and solving the
power flow at 13:00:

| | voltage-admissible Q_PCC window | reported window | export over-report |
| --- | --- | --- | --- |
| DSO 1 | [-122.7, +232.0] | [-292.5, +232.0] | 2.4x |
| DSO 2 | [-101.3, +203.9] | [-293.4, +235.8] | 2.9x |
| DSO 3 | [-196.7, +227.0] | [-293.4, +227.0] | 1.5x |
| **DSO 4** | **[-57.1, +113.0]** | **[-297.9, +254.3]** | **5.2x** |

In the closed loop the TSO dispatches ~-95 Mvar of export from DSO 4 against a
~-57 Mvar voltage-admissible limit at neutral tap.

## 5. Measured variants (360 min, identical scenario and contingencies)

| case | change (DSO 4 only) | V_max max | V_max mean | % >1.09 | V_min min | spread max | ops/h | rev/h | Q RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | baseline | 1.1010 | 1.0865 | 65.0 % | 0.9531 | 0.147 | 2.33 | 0.00 | 6.08 |
| B | `g_w_pcc(Z3)` 22.1 -> 4.42 | 1.1046 | 1.0886 | 48.3 % | 0.9523 | 0.144 | 4.83 | 1.17 | 3.27 |
| C | `dso_g_v` 1.5e5 -> 1e6 | 1.1012 | 1.0732 | 43.7 % | 0.9841 | 0.110 | 53.67 | 50.50 | 6.15 |
| D | `g_w_dso_oltc` 183 -> 36.6 | 1.1010 | 1.0842 | 62.4 % | 0.9142 | 0.186 | 30.50 | 26.67 | 5.42 |
| E | `V_set` 1.03 -> 1.00 | 1.1010 | 1.0744 | 59.5 % | 0.9324 | 0.168 | 2.67 | 0.00 | 5.92 |
| H | `g_w_pcc(Z3)` 22.1 -> 110.5 | 1.0593 | 1.0477 | 0.0 % | 1.0153 | 0.036 | 0.33 | 0.00 | 0.18 |
| **F** | `dso_g_v` x6.7 + `g_w_dso_oltc` x6.7 | 1.0993 | 1.0745 | 45.5 % | 1.0005 | 0.094 | 0.83 | 0.00 | 6.26 |
| **G** | `dso_g_v` x20 + `g_w_dso_oltc` x20 | **1.0629** | **1.0518** | **0.0 %** | **1.0027** | **0.058** | **0.67** | 0.00 | 6.59 |

F, G and H are derived in §6; they are listed here so the whole set is in one
table. Ranked by upper-bound relief at acceptable switching: G > H > F, with
H carrying the horizon caveat below.

Zone 1/2/3 EHV envelopes and RMS tracking errors, and DSO 1/2/3 max voltages,
are unchanged to four decimals in every case — these are genuinely local
measures. Two exceptions worth naming: D and H both push DSO 3's interface-Q
RMSE 5.79 -> ~10.40 (a discrete change, both landing on the same value, so a
different DSO 3 tap sequence rather than a smooth degradation), and system
losses move within 39.7–41.2 MW against a 40.68 MW baseline.

### B / H — `g_w_pcc` sets how hard the TSO leans on DSO 4, in the direction opposite to the proposal

`g_w` is a proximal penalty on the step `w = du/alpha`
(`optimisation/miqp_solver.py:828`). For an unconstrained quadratic it would set
only the rate of approach, not the fixed point — but the PCC bound is a soft
*output* constraint with slacks (`g_z_q_pcc`), and with that constraint active
the operating point does depend on `g_w`. Measured over this window, with a
run H added at `g_w_pcc(Z3) = 110.5` (x5 the baseline, i.e. the **opposite** of
the proposal):

| `g_w_pcc(Z3)` | Q_PCC mean | V_max max | V_max mean | spread max | DSO 4 Q RMSE |
| --- | --- | --- | --- | --- | --- |
| 4.42 (B) | -124.6 | 1.1046 | 1.0886 | 0.144 | 3.27 |
| 22.1 (A) | -86.7 | 1.1010 | 1.0865 | 0.147 | 6.08 |
| **110.5 (H)** | **-56.0** | **1.0593** | **1.0477** | **0.036** | **0.18** |

The ordering is monotone and the voltage follows it exactly. **Lowering**
`g_w_pcc` as proposed lets the TSO travel *further* into the over-reported
envelope of §4 — B reaches -125 Mvar against a ~-57 Mvar voltage-admissible
limit — and makes the voltage worse. Raising it restrains the TSO to almost
exactly the admissible limit (-56 Mvar) and the problem disappears.

H is therefore a strong **confirmation of the §4 diagnosis**, but it is not
recommended as the fix: whether -56 Mvar is a genuine new equilibrium or just a
slower approach to the same place is not settled by a 6 h window. H's setpoint
is still drifting at -0.54 Mvar/h at t = 360 min (A converged to ~-95 Mvar by
t = 120 min and sits at +0.24 Mvar/h). H also suppresses the TSO's use of a
legitimate actuator, and it degraded DSO 3's interface-Q RMSE 5.79 -> 10.40.
G (§6.1) instead delivers the *same* interface duty as baseline (-99.7 vs
-94.9 Mvar at t = 360) with a better internal profile, so it does not depend on
that unresolved question. A 24 h rerun would settle it.

### D — the commit threshold is a stability parameter

Lowering it commits taps in the direction §3 shows is wrong. Measured: `V_max`
mean -0.0023 p.u., `V_min` min 0.9531 -> 0.9142 with 54 % of steps under 0.95,
and 30.5 ops/h at 26.7 reversals/h. DSO 3's interface-Q RMSE doubles as
collateral (5.79 -> 10.39). Not viable.

### E — a long feeder's spread is not invariant under tap translation

Run to test the §3 prediction that re-centring flips the gradient sign. It does:
taps move [8,5,2] -> [10,7,3] with **zero** reversals. But the **spread grew
0.147 -> 0.168 p.u.**, so `V_max` stayed on the bound (52.8 % of steps) while
`V_min` fell 0.021. Lowering the 110 kV level to deliver the same Mvar export
raises the current, and the I^2 X drops along 586 km stretch the profile. The
OLTC therefore has less effective authority than the linear translation picture
implies. This is the entry's main negative result.

### C — the only mechanism that attacks the spread

C improves **both** bounds simultaneously (`V_max` mean -0.013, `V_min` min
+0.031) because it *shrinks* the spread, 0.147 -> 0.110. Aggregate `Q_DER` is
unchanged (63.3 -> 62.5 Mvar): this is the same reactive power **redistributed
across the ten DER sites**. Raising `dso_g_v` gives the voltage term enough
weight to reshape *which* DER injects, while remaining far too weak (53:1
against the Q-tracking gradient) to change *how much* in total. Interface-Q
tracking is unaffected (6.08 -> 6.15 Mvar).

C's cost is disqualifying: 53.7 ops/h at 50.5 reversals/h — a limit cycle, ~42x
the reversal constraint the tuning campaign enforces.

## 6. Where the hunting comes from, and the separation it implies

`dso_g_v` raises the voltage gradient on the DER block *and* the OLTC block. The
DER is continuous and absorbs gain smoothly; the OLTC is integer with a commit
threshold, and raising the gradient past it produces a limit cycle. The useful
effect (§5, C) is in the DER block; the cost is in the OLTC block.

The two are separable with existing fields, because `dso_g_w_class` sets
`g_w_dso_oltc` per DSO area. Holding the ratio `dso_g_v / g_w_dso_oltc` — the
OLTC loop gain — at its baseline value while raising `dso_g_v` gives the DER
C's voltage authority at the baseline switching rate.

| case | change (DSO 4 only) |
| --- | --- |
| F | `dso_g_v` 1.5e5 -> 1e6 **and** `g_w_dso_oltc` 183 -> 1220 (ratio held) |
| G | `dso_g_v` 1.5e5 -> 3e6 **and** `g_w_dso_oltc` 183 -> 3660 (ratio held) |

### 6.1 Result — the separation works, and G is the recommendation

| case | V_max max | V_max p95 | %>1.09 | V_min min | spread max / mean | ops/h | rev/h | Q RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A baseline | 1.1010 | 1.1001 | 65.0 % | 0.9531 | 0.147 / 0.106 | 2.33 | 0.00 | 6.08 |
| C `dso_g_v` alone | 1.1012 | 1.0973 | 43.7 % | 0.9841 | 0.110 / 0.073 | 53.67 | 50.50 | 6.15 |
| **F** ratio held, x6.7 | 1.0993 | 1.0985 | 45.5 % | **1.0005** | 0.094 / 0.061 | **0.83** | 0.00 | 6.26 |
| **G** ratio held, x20 | **1.0629** | **1.0622** | **0.0 %** | **1.0027** | **0.058 / 0.038** | **0.67** | 0.00 | 6.59 |

Holding the OLTC loop gain `dso_g_v / g_w_dso_oltc` fixed removes the limit
cycle completely (0.00 reversals/h in both) **and** beats C on every voltage
measure. The tap rate *falls below baseline* (2.33 -> 0.83 -> 0.67 ops/h),
because once the DER carries the voltage duty the profile stays centred and the
tap rarely needs to commit at all.

G moves DSO 4 from riding the bound to a comfortable envelope: `V_max` never
exceeds 1.063 (from 1.101), `V_min` never falls below 1.003 (from 0.953), and
the spread drops 61 % (0.147 -> 0.058). F halves the spread but leaves `V_max`
riding just under the bound (p95 = 1.0985, 45.5 % of steps above 1.09), so it is
an improvement rather than a fix.

Cost of G: interface-Q RMSE 6.08 -> 6.59 Mvar (+8 %). No role inversion — at
x20 the DER voltage gradient is ~1.5e4 against a Q-tracking gradient of ~2.6e5,
still 18:1 in favour of interface-Q tracking, so the DSO remains a Q tracker
that also shapes voltage.

**Collateral: none measurable.** Zone 1/2/3 EHV envelopes and RMS tracking
errors, DSO 1/2/3 max voltages, their interface-Q RMSEs and their tap rates are
all identical to four decimals against baseline. System losses *improve*
slightly, 40.68 -> 39.86 MW.

### 6.2 Implemented (G, DSO 4 only)

| file | change |
| --- | --- |
| `configs/config.py` | new `MultiTSOConfig.dso_g_v_per_area: Optional[Dict[str, float]]` — per-DSO `dso_g_v` override, mirroring `dso_g_w_class`. Docstring states the pairing requirement. |
| `experiments/runners/multi_tso_dso.py` | applies it next to `dso_g_w_class`, after controller construction and before the loop. **Warns** when `g_v` is raised without a matching `dso_oltc` raise (the configuration that limit-cycles), naming the value that would restore the ratio. |
| `experiments/run_multi_system_ofo.py` | `DSO4_V_RELIEF_FACTOR = 20.0` and `_apply_dso4_v_relief(cfg, factor)`, applied at the end of `make_config_tuned()` **and** `make_config_per_area()`. |
| `tests/test_dso_v_relief_pairing.py` | 6 tests locking the invariant. |

Both halves are **derived from the config's own `dso_g_v` / `g_w_dso_oltc`**
rather than written as literals. That matters because `make_config_per_area()`
rewrites `dso_g_w_class` wholesale with its analytic per-area block *and*
rescales the whole weight group by `GAUGE = 4e-3`; with literals it would have
left a raised `g_v` against the per-area `dso_oltc` — a loop gain ~5000x too
high, i.e. exactly the hunting configuration. The relief is therefore removed
before that rewrite and re-applied after it, against the per-area value.
Verified: loop-gain change = 1.000000 on both builders, and the per-area design
for DSO 1–3 survives.

Set `DSO4_V_RELIEF_FACTOR = 1.0` to disable.

**End-to-end verification.** A 360 min run driven only by the committed
`make_config_tuned()` — no hook, no overrides — reproduces the hand-applied
run G to machine precision on every metric (`V_max` max 1.0629, `V_min` min
1.0027, spread 0.0576, 0.67 ops/h, 0.00 rev/h, Q RMSE 6.5901; max |diff| = 0).

Do **not** raise `dso_g_v` without the matching `g_w_dso_oltc` raise — that is
case C, which limit-cycles at 50 reversals/h. The runner now warns if you do.

## 6.3 Should this go to the other DSOs, and can the tuner find it?

### It is not a tuning improvement under the current objective — it is *rejected*

Scoring baseline and G with the real Stage-1 machinery
(`tuning.metrics.extract_metrics` + `tuning.objectives_v2.performance_scalar`,
the criteria `tuning_mc.metrics` actually filters on):

| | f_ts (scored) | f_q (scored) | f_ds (reported only) | v_worst_ds (not scored) |
| --- | --- | --- | --- | --- |
| A baseline | **0.901017** | **7.528574** | 0.111848 | 0.076240 |
| G | 0.901039 | 7.713745 | **0.067476** | **0.069380** |

**A dominates G on both filter criteria.** G is not on the Pareto front, so
Stage 1 would reject it at any search budget. The search is not under-powered —
it is optimising a different thing. The 40 % improvement in `f_ds` and the
relief in `v_worst_ds` are recorded and discarded, exactly as
`tuning_mc/metrics.py:28-31` says ("Reported diagnostics — DS voltage ...
never optimised directly").

### Why the DS metric that *is* computed cannot see it

The DS quantity that reaches the objective is
`v_rms_ds = mean(|v_mean_ds - v_set|)` (`tuning/metrics.py:611-613`) — the
distance of the DSO envelope's **centre** from the setpoint. But §3 shows the
OLTC drives the profile centre onto `v_set` *by construction*: that is precisely
what its gradient minimises. So `v_rms_ds` is smallest exactly when the tap has
finished spending its authority, i.e. when the network is most stressed.
Measured on the baseline:

| | `v_rms_ds` (scored) | `v_worst_ds` (computed, unscored) | min headroom to 1.10 | spread max |
| --- | --- | --- | --- | --- |
| DSO 1 | 0.00571 | 0.01703 | +0.04823 | 0.015 |
| DSO 2 | **0.00421** (best!) | 0.06944 | **-0.00122** | 0.117 |
| DSO 3 | 0.00620 | 0.02985 | +0.03921 | 0.037 |
| DSO 4 | 0.01021 | 0.07624 | -0.00101 | 0.147 |

`v_rms_ds` ranks **DSO 2 as the healthiest of the four** while DSO 2 is the
second-worst network. The scored metric is not merely blind here, it is
anti-correlated with the defect. `v_worst_ds` — already computed at
`tuning/metrics.py:630-632` and thrown away — ranks all four correctly.

The hard constraint does not catch it either. g2 (`_voltage_band_excess`) *does*
include `dso_group_v_max_pu`, but it is a zero-margin barrier:
`max(V_max - 1.10, 0)`. Measured, the baseline scores 4.26e-5 pu/step against a
1e-4 limit — **feasible**. A controller that rides the bound at 1.0999 scores
exactly 0. A barrier cannot express "feasible but undesired", which is the
condition being reported.

### What to change, in the order it has to happen

1. **Metric first, or nothing else matters.** Two minimal edits, both using
   quantities that already exist:
   * add `"v_band_excess_ds"` / `"v_worst_ds"` to the scored DS criterion (or
     replace `f_ds = v_rms_ds` with them) — one line in `tuning_mc/metrics.py`;
   * add a **margin** constraint `g6_ds_headroom = h_req - min_t min(v_max_pu -
     V_hi, V_lo - v_min_pu)` with `h_req ~ 0.01 pu`, alongside g5a/g5b in
     `CONSTRAINT_NAMES`. This is the instrument that expresses "stay off the
     bound" as opposed to "do not cross it".

   Calibrate `h_req` from the reference the same way `ConstraintLimits`
   already calibrates `corridor_excess_pu` (`from_reference`, margin 1.5) —
   note the hand-tuned reference is *outside* any positive headroom on two of
   four DSOs, so the limit must be set from the design intent, not from the
   reference.

2. **Then the search dimension, reparameterised so the harmful direction is
   unreachable.** Add one coordinate per DSO area, `dso_v_authority`, that
   multiplies `dso_g_v` **and** that area's `dso_oltc` together — i.e. exactly
   `_apply_dso4_v_relief`. This is the same device the study already uses for
   `tau_der_pcc`, carried as `(sqrt(tau), 1/sqrt(tau))` so the geometric mean is
   pinned and the coordinate moves only the ratio. Here the invariant to pin is
   the OLTC loop gain. Without this reparameterisation the searcher can reach
   case C, which violates g5b by ~40x and is wasted budget.

3. **Better: derive it in Stage 0, do not search it.** The quantity is
   predictable from the network, and a searched coordinate costs polls
   (`tuning_mc` phase B is a compass search: every live direction is evaluated
   at +/- delta on every poll). The relief
   needed tracks the internal spread, which `stage_0_preconditioning` can
   compute from the same cached `H` it already uses for the curvature and
   commit-threshold rules — the DS voltage rows of `H` give
   `dV/dQ_DER` per area, and the spread per Mvar of interface duty follows.
   That fits the existing split (Stage 0 derives per-area weights analytically,
   Stage 1 searches a small shared set) and is where the per-area `der`/`pcc`
   /`tso_oltc` designs already live. Measured slopes for the design rule:
   d(spread)/d(Q_PCC) = -0.0080 / -0.0984 / -0.0241 / -0.3482 %V per Mvar for
   DSO 1/2/3/4.

### Uniformly to all DSOs?  Measured: no — selective by spread

Run I applies the factor-20 pair to all four areas, run J to DSO 2 and DSO 4
only. Per-area, against baseline A:

**min headroom to the nearer bound [p.u.]** (negative = outside the corridor)

| case | DSO 1 | DSO 2 | DSO 3 | DSO 4 |
| --- | --- | --- | --- | --- |
| A none | +0.0482 | **-0.0012** | +0.0392 | **-0.0010** |
| G DSO 4 | +0.0482 | **-0.0012** | +0.0392 | +0.0371 |
| **J DSO 2+4** | +0.0482 | **+0.0181** | +0.0404 | +0.0370 |
| I all four | +0.0498 | +0.0180 | +0.0394 | +0.0370 |

**internal spread max [p.u.]**

| case | DSO 1 | DSO 2 | DSO 3 | DSO 4 |
| --- | --- | --- | --- | --- |
| A none | 0.0147 | 0.1165 | 0.0366 | 0.1469 |
| G DSO 4 | 0.0148 | 0.1165 | 0.0366 | 0.0576 |
| **J DSO 2+4** | 0.0150 | **0.0762** | 0.0365 | 0.0576 |
| I all four | **0.0179** | 0.0759 | 0.0365 | 0.0577 |

**interface-Q RMSE [Mvar]**

| case | DSO 1 | DSO 2 | DSO 3 | DSO 4 |
| --- | --- | --- | --- | --- |
| A none | 0.44 | 8.42 | 5.79 | 6.08 |
| G DSO 4 | 0.44 | 8.43 | 5.78 | 6.59 |
| **J DSO 2+4** | 0.45 | 9.17 | 5.43 | 6.60 |
| I all four | 0.47 | 9.15 | **8.86** | 6.60 |

Three conclusions:

* **DSO 2 should get it.** J moves DSO 2 from -0.0012 (outside the corridor) to
  +0.0181 of headroom, spread 0.117 -> 0.076 (-35 %), and its tap rate *falls*
  2.00 -> 1.50 ops/h. Cost: its own interface-Q RMSE 8.42 -> 9.17 (+9 %), the
  same ~8-9 % trade DSO 4 pays. After J **every** DSO has positive headroom.
* **DSO 1 and DSO 3 should not.** They start with 4.8 and 3.9 p.u.-% of
  headroom, so there is nothing to buy: I improves their `V_max` by 0.0016 and
  0.0002 p.u. Meanwhile I costs DSO 3 an interface-Q RMSE of 5.79 -> **8.86
  (+53 %)** and makes DSO 1's spread *worse* (0.0147 -> 0.0179). Extra voltage
  authority on a network that is not spread-limited is spent competing with
  interface-Q tracking for no return.
* **This validates the §6.3(3) design rule.** The two areas with spread > 0.10
  benefit; the two with spread < 0.04 pay without benefit. A factor scaled to
  the area's spread — derived in Stage 0 from `H` — reproduces this split
  automatically instead of requiring a per-area decision by hand.

Implemented: `DSO_V_RELIEF_FACTORS = {"DSO_2": 20.0, "DSO_4": 20.0}` and
`_apply_dso_v_relief(cfg, factors)`. A config-only run reproduces run J exactly
(DSO 2: `V_max` 1.0819, headroom +0.0181, spread 0.0762, 1.50 ops/h; DSO 4:
1.0630 / +0.0370 / 0.0576 / 0.67 ops/h).

## 7. Guard-band metric: making the defect visible to the tuner

Implemented behind a flag, default off so every study up to today reproduces.

| file | change |
| --- | --- |
| `tuning/metrics.py` | `DS_GUARD_HEADROOM_PU = 0.02`; `_voltage_band_excess(..., groups=)`; new `TrajectoryMetrics.guard_deficit_ds_pu` and `.ds_headroom_min_pu` |
| `tuning_mc/metrics.py` | `score_candidate(..., ds_criterion="v_rms" \| "guard")`; `ds_headroom_min_pu` added to the per-scenario record |

`guard_deficit_ds_pu` is the **existing** ramp `_voltage_band_excess` evaluated
against a corridor shrunk by `h` at both ends — `[0.92, 1.08]` at `h = 0.02` —
restricted to DSO groups and normalised per record. So it charges *before* a
bound is reached. Deliberately an integral over time rather than
`min_t headroom`: a worst-case statistic has between-bank variability larger
than its in-sample margin (the lambda* transfer defect), and the ramp shape is
a ramp keeps every compass poll informative, where a step function would read
as a dead direction until it discontinuously is not. (The existing
`_voltage_band_excess` docstring argues the same shape from TPE's Parzen
kernels — that rationale predates `tuning_mc`, which is a compass search under
an Audet–Dennis filter, not Bayesian optimisation.)

`h = 0.02` is a **design intent, not a calibration**. `ConstraintLimits` derives
`corridor_excess_pu` via `from_reference`, but that is unavailable here: the
hand-tuned reference has *negative* headroom on two of four DSOs, so calibrating
from it would enshrine the defect.

### Re-scored: what the filter would have selected

| case | f_ts | f_q | f_ds (`v_rms`) | f_ds (`guard`) | headroom min |
| --- | --- | --- | --- | --- | --- |
| A baseline | 0.90102 | 7.5286 | 0.11185 | 0.022328 | -0.0012 |
| B `g_w_pcc` lo | 0.90091 | 5.9332 | 0.07982 | 0.019594 | -0.0046 |
| C `dso_g_v` x6.7 | 0.90170 | 7.7611 | 0.06216 | 0.016208 | -0.0014 |
| D `g_w_oltc` lo | 0.90393 | 9.1334 | 0.10825 | 0.022244 | -0.0013 |
| E `V_set` 1.00 | 0.90161 | 7.4643 | 0.11786 | 0.021299 | -0.0013 |
| H `g_w_pcc` hi | 0.90714 | 7.2507 | 0.07237 | 0.009967 | -0.0013 |
| F pair x6.7 | 0.90098 | 7.6277 | 0.07285 | 0.016778 | -0.0012 |
| G pair x20 | 0.90104 | 7.7137 | 0.06748 | 0.009393 | -0.0012 |
| **J pair x20, DSO 2+4** | **0.85973** | 7.8082 | 0.06965 | **0.000002** | **+0.0181** |
| I pair x20, all four | 0.86209 | 9.2263 | 0.07079 | 0.000002 | +0.0180 |

Pareto fronts:

* `(f_ts, f_q)` — today's default: `[B, J]`
* `+ with_ds`, `f_ds = v_rms_ds`: `[B, C, H, F, G, J]`
* `+ with_ds`, `f_ds = guard`: `[B, H, F, G, J]`

The guard criterion separates by three orders of magnitude (J/I at 2e-6 against
0.009–0.022 for everything else) and drops C — the limit-cycling case — from the
front that `v_rms_ds` admits. C and D are barred by g5b anyway (50.5 and 26.7
reversals/h), so the guard front and the admissible set agree.

**J is on the front even under today's two criteria**, because its `f_ts` is
4.6 % better than baseline. That is not a rounding artefact: 81 % of the gain is
`v_band_ts` (raw `v_band_excess_ts` 0.004845 -> 0.004023, -17 %), i.e. relieving
DSO 2 pulls zone-2 EHV buses back inside the quality band. Zone 2's own RMS
error moves 0.00843 -> 0.00822. So the DSO 2 relief partly pays for itself at
the TS layer; only the DSO 4 relief is a pure DS-side gain invisible to the
current filter.

## 8. Structural predictor: attributing the spread to X . Q

Author's hypothesis: the spread should be predictable from where the reactive
infeed sits electrically relative to the coupler — "the spread in the product
X . Q" — which would let the per-area factor be *derived* rather than chosen.

Formalised with the controller's own cached sensitivity. `H_v = dV_j/dQ_der,i`
(voltage rows, DER columns). Column `i` of `H_v` is how far every bus moves when
infeed `i` injects, so its across-bus **range** is the electrical distance from
that infeed to the stiffest point. The range, not the level, is the right
functional: the OLTC translates the profile (measured `dV/ds` is near-uniform
across the ten buses), so the mean of `H_v q` is removable and only its spread
is not.

| DSO | reach = mean col range [pu/Mvar] | \|Q_PCC\| mean [Mvar] | reach x duty | measured spread |
| --- | --- | --- | --- | --- |
| DSO 1 | 0.000456 | 25.5 | 0.0116 | 0.0147 |
| DSO 2 | 0.000686 | 104.5 | 0.0717 | 0.1165 |
| DSO 3 | 0.000291 | 106.4 | 0.0310 | 0.0366 |
| DSO 4 | 0.001080 | 86.7 | 0.0936 | 0.1469 |

**The network term alone is not enough, and the failure is informative.** Every
purely structural predictor (mean/max column range, and the box maximum
`max over q in [q_min, q_max] of spread(H_v q)`) correlates at r ~ +0.90 but
**mis-ranks DSO 1 against DSO 3**: DSO 1 has the *higher* sensitivity yet the
*lower* spread, because the TSO barely uses it (25.5 vs 106.4 Mvar). Spread is
sensitivity x throughput, not sensitivity.

Multiplying by the interface duty fixes it:

| predictor | r | ranks correctly | zero-intercept slope | max residual |
| --- | --- | --- | --- | --- |
| reach_mean (network only) | +0.8992 | no | 132.7 | 0.0458 pu |
| xi_box (network only) | +0.8965 | no | 0.625 | 0.0448 pu |
| **reach_mean x \|Q_PCC\|** | **+0.9964** | **yes** | 1.561 | 0.0118 pu |
| reach_max x \|Q_PCC\| | +0.9947 | yes | 0.851 | 0.0093 pu |
| xi_box x \|Q_PCC\|/100 | +0.9972 | yes | 0.735 | 0.0088 pu |

So the hypothesis holds in the form **spread ~ k . (electrical reach) . (interface
throughput)**, with a one-parameter zero-intercept law giving ~0.01 pu residual
across a 0.015–0.147 pu range.

**Use `reach x |Q_PCC|`, not `xi_box x |Q_PCC|`, despite the latter's marginally
better r.** `xi_box` already has the reactive magnitude folded in (it is
`max over q in the capability box of spread(H_v q)`, so its units are pu volts);
multiplying it by `|Q_PCC|` again gives pu.Mvar and double-counts the reactive
magnitude. `reach` is pu/Mvar, so `reach x |Q_PCC|` is a voltage and reads as a
physical statement. The r difference (0.9972 vs 0.9964) is noise at n = 4.
`reach_max` vs `reach_mean` (0.0093 vs 0.0118 pu residual) cannot be
distinguished at this sample size either; `reach_mean` is the more principled
of the two because the spread is produced by the whole fleet, not one infeed.

### What this does *not* yet support

* **The four "networks" are one topology at four scalings.** Every entry of
  `SUBNET_DEFS` instantiates the *same* `HV_LINE_TOPOLOGY` — same 11 corridors,
  same bus numbering, same DER and load placement — differing only in
  `line_length_scale` (0.82 / 1.40 / 0.52 / 2.44) and one conductor/parallel
  change on DSO 3. So this is a **one-parameter family**, not four independent
  samples, and *any* quantity monotone in impedance x duty will correlate on it.
  This is the weakest point in the evidence, and it is not fixed by adding more
  scale factors — it needs genuinely different topologies (different DER/load
  placement, meshed vs radial, more than one voltage level).
* **n = 4, one free parameter, two clusters.** r = 0.996 over four points that
  fall into a {1,3} low / {2,4} high split is close to measuring "are the two
  clusters separated", which any monotone predictor achieves.
* **The duty is an outcome, not an input.** `|Q_PCC|` is what the TSO ends up
  asking for, which depends on the tuning being designed — circular if used
  naively. Two non-circular options: use the voltage-admissible duty from the
  static probe (§4), or run Stage 0 twice (measure duty at baseline, set the
  factors, re-derive).
* **The factor-to-headroom map is 2–3 points per network.** Measured:
  DSO 4 factor 1 -> -0.0010, 6.7 -> +0.0007, 20 -> +0.0371 pu; DSO 2 factor
  1 -> -0.0012, 20 -> +0.0181. DSO 2 needs a *larger* factor than DSO 4 for the
  same headroom despite its smaller spread, which the spread predictor alone
  does not explain. Fitting "optimum factor per network" needs 3–4 factors per
  network, i.e. ~12 more runs.

### The defensible way to state it in the thesis

Not as an empirical law fitted to four points. **Derive it, then verify.** Given
the linearisation the controller already assumes,

    dV = H_v q ,

and given that the interface OLTC *approximately translates* the profile
(measured: 8 taps moved DSO 4's `V_max` and `V_min` by 0.089 and 0.095 pu, i.e.
the spread moved only 0.005 pu), the removable component of `H_v q` is its mean
across buses and the irreducible component is its across-bus range:

    spread_irreducible  ~  range_j ( (P H_v q)_j ) ,   P = I - 11^T/n .

That is algebra, not a hypothesis. The empirical content is narrow and should be
stated as such: (i) the linearisation holds over the operating range, (ii) the
translation is only approximate — the spread *grew* 0.147 -> 0.168 pu when the
profile was tapped down (case E), so the removable part is not exactly the mean,
and (iii) `|Q_PCC|` is a usable proxy for `||q||`. Presented that way the four
scalings are an illustration of a derived result, and the two clusters are not
carrying the argument.

## 9. Should the tuning be re-run?

### The archive cannot be re-scored — checked, and this is the deciding fact

`stage_1_search._evaluate` receives `(res, records)` per scenario but persists
only the JSON payload; `records` are dropped. The per-scenario dict that *is*
written carries `['f_ds', 'f_q', 'f_total', 'f_ts', 'feasible', 'rho_emp_p95',
'tap_ops_per_h_{tso,dso}', 'tap_reversals_per_h_{tso,dso}']` — no DS voltage
envelope. So `guard_deficit_ds_pu` cannot be recomputed for any of the 110
archived tier-1 trials; adding a DS-voltage criterion forces re-simulation.

**Process fix — implemented (step 0).** `score.per_scenario[name]` now carries a
`"metrics"` sub-dict holding the **whole** flat `TrajectoryMetrics`
(37 fields), not a hand-picked subset. A subset would only move the problem to
whichever metric is invented next; the coverage is the point.

Purely additive: every existing key is kept verbatim, and the three readers
(`report_0815`, `select_windows_v2`, `export_final`, plus `stage_1_search`'s own
phase writers) access `per_scenario` by explicit key name — none iterates or
validates the key set. Checked by running `report_0815.show_windows` against a
new-format payload.

Cost: `per_scenario` for a 12-window tier-1 trial goes 4.0 kB -> 14.9 kB, i.e.
**+1.2 MB across a 110-trial campaign**. Negligible against ~9 h of re-simulation.

Locked by `tests/tuning/test_stage1_archive_rescorable.py` (5 tests). The
coverage test asserts against
`dataclasses.fields(TrajectoryMetrics)` rather than a hand-listed set of names,
so it fails automatically if a future metric is added to `TrajectoryMetrics` but
dropped from the archive — the exact failure mode being prevented. One test also
round-trips through `json.dumps`/`loads` including a NaN field, since that is
how `stage_1_search` actually writes and reads the payload.

### Cost, measured rather than guessed

| bank | windows | sim-h / trial | serial h / trial | 110 trials on 12 cores |
| --- | --- | --- | --- | --- |
| tier1_design | 12 | 18.0 | ~1.0 | **~9 h wall** |
| tier1_confirm | 9 | 13.5 | ~0.7 | ~6 h wall |
| tier2_audit | 4 | 48.0 | ~2.6 | shortlist only |

At the measured 3.2 min/sim-h and the 12-core budget, a full tier-1 Stage-1
re-run at the same trial count is roughly **one working day**, not weeks. That
changes the calculus: the re-run is affordable, so the argument for avoiding it
is only about doing it in the right order.

### Order

Reasons to not simply re-run today, in order:

1. **A re-run today would re-find the same point.** The relief pair is not a
   search dimension, and until §7's guard criterion is switched on the objective
   cannot see the thing being fixed — §6.3 shows A dominates G under the
   two-criterion filter. The search would converge somewhere near the current
   baseline and the areas would still be hand-patched afterwards.
2. **The cheaper and more defensible route is Stage 0, not Stage 1.** The
   project already splits "Stage 0 derives per-area weights analytically from
   `H`; Stage 1 searches a small shared set", and the per-area `der` / `pcc` /
   `gen` / `tso_oltc` block in `make_config_per_area` is exactly that. The
   voltage-authority factor belongs in the same block, derived from §8's
   predictor — which uses the same cached `H` Stage 0 already reads. That needs
   **no Stage-1 re-run at all**: regenerate the Stage-0 block and re-validate.
3. **If it does go into Stage 1**, it must be the reparameterised coordinate
   (one `dso_v_authority` per area multiplying `dso_g_v` and that area's
   `dso_oltc` together), or the searcher can reach case C and burn budget on
   points that violate g5b by ~40x.

What a re-run *would* legitimately buy: the factor 20 is a round number chosen
from two measurements (6.7 and 20) on one 6 h window, and `h = 0.02` is a design
choice. Neither is optimised.

### Recommended sequence

For a thesis that needs "the method produced these numbers" rather than "the
method produced most of them and two areas were designed separately":

0. **Persist the DS diagnostics** (above). **DONE** — the whole
   `TrajectoryMetrics` is archived per scenario, so from now on a criterion
   change is an offline re-score. Note this does *not* retro-fit the existing
   110 trials: they were written before the change and still lack the DS
   envelope, so the re-run below is still required once.
1. **Switch the criterion on**: `ds_criterion="guard"`, `with_ds=True`. Fix
   `h` and justify it as design intent (it cannot be calibrated from a reference
   that is itself outside the corridor).
2. **Put the voltage authority in Stage 0**, derived from §8 against the same
   cached `H` Stage 0 already reads, using the voltage-admissible duty from §4
   to break the circularity in `|Q_PCC|`. Regenerate the per-area block.
3. **Re-run Stage 1 tier-1** (~9 h wall) against the new criterion and the new
   Stage-0 block. This is the step that makes the numbers the method's own.
4. **Re-validate on tier-2 audit** for the shortlist, and on the holdout bank —
   single-window optima do not transfer (`METHOD_weight_selection.md`).

Steps 0–2 are ~a day of work; step 3 is ~a day of compute. Do **not** re-run
before step 1, or the search re-finds the current baseline and the areas get
hand-patched again.

## 7. Open points

* The §5 numbers are one 6 h window (2016-05-03 08:00) with one contingency
  schedule. The ranking must be re-measured on the tuning campaign's scenario
  banks before anything is written into `make_config_tuned` — cf.
  `2026-08-15_campaign_0815_tiered_bank_and_recalibration.md`, where a
  single-window optimum did not transfer.
* Voltage-aware capability reporting (§4) is an **architectural** change to the
  DSO -> TSO message and is deliberately not implemented here — flagged for
  discussion.
* DSO 2 has the same defect one step milder and needs the same treatment; a
  DSO-4-only fix leaves 1.1012 p.u. on the table.
* `V_set = 1.03` uniform across all HV buses is a modelling choice that a
  network with a 0.147 p.u. spread cannot satisfy in the least-squares sense
  without putting one end on a bound. A band/margin objective rather than a
  scalar-tracking objective would state the intent correctly — also
  architectural.
* The spread-vs-`Q_PCC` slopes in §2 are single-variable regressions over a
  narrow operating range with `P_DER` co-varying (corr -0.67 for DSO 4), so the
  cross-DSO *ranking* is the robust statement, not the absolute slope. The
  static probe in §4 is the reliable magnitude.

## 10. The re-run, as executed (2026-08-18 evening)

Four defects were found *while launching*, none of which would have shown up as
an error — each would have produced a plausible-looking but worthless campaign.
All four are now blocked by `tuning_mc/preflight_rerun.py` or by the chain
script `tuning_mc/run_rerun_2026-08-18.sh`.

| # | defect | symptom had it stood |
| --- | --- | --- |
| 1 | `--limits` omitted -> `ConstraintLimits()` defaults (`rho_emp_p95 = 1.0` vs the tier-1 file's 1.5) | every candidate infeasible on g3; `filter_accepts` rejects infeasible outright, so an empty filter for ~9 h |
| 2 | limits were in **no** fingerprint | relaunching with the right file would have replayed cached rows with their stale `feasible` |
| 3 | `X0` left at its shipped *analytic* `lambda_tso = 0.9` | design point at rho 2.487 vs a 1.5 limit — phase A measures identifiability at a point already known infeasible (its own docstring warns of exactly this) |
| 4 | `--rho-margin` defaulted to 0 | lambda* selected against 1.5 instead of 1.4549; lambda = 0.25 (rho 1.4743) would have been chosen — the transfer defect |

`scoring_fingerprint` now hashes the resolved limits as well as the two metric
sources and the criterion; verified live when the 9 rows from the bad launch
were detected and moved to `*.json.scoring_changed`.

### Phase lam — the margin changes the answer

| lambda | rho |
| --- | --- |
| 0.90 | 2.4868 |
| 0.60 | 2.0181 |
| 0.40 | 1.7068 |
| 0.25 | 1.4743 |
| **0.15** | **1.3256** <- lambda* |
| 0.10 | 1.2573 |

Effective target `1.5 / 1.031 = 1.4549`. lambda = 0.25 is under the *declared*
1.5 but over the margined target, so it is correctly rejected — that single step
is the difference between this calibration and the archived one (whose
`calibration` block is `null`). Fitted `lambda_at_boundary = 0.2337`, which
cross-checks the independently measured probe at lambda_tso = 0.225 -> rho 1.436.

Only `lambda_tso` was re-anchored. Measured on this bank, g3 sits at exactly
+0.987 for every probe of `lambda_dso`, `tau`, `engage_*` and `dso_g_v_ratio`
and moves only with `lambda_tso`, so contraction is a one-coordinate property
and detuning the DSO loop as well would cost f_ts for nothing.

### Phase A — feasible anchor, all seven directions live

    design point feasible: True
    g3 -0.1744   g5a -1.9926   g5b -1.3309   (g1/g2/g4 clear)
    base  f_ts = 1.31251   f_q = 0.09861   f_ds = 0.000000
    live: all 7      dead: none

| direction | f_ts | f_q |
| --- | --- | --- |
| engage_tso_pu | 20.68 % | 6.99 % |
| lambda_tso | 12.47 % | 5.00 % |
| dso_g_v_ratio | 1.91 % | 61.56 % |
| engage_dso_pu | 1.91 % | 200.33 % |
| lambda_dso | 1.14 % | 58.50 % |
| tau | 1.10 % | 10.77 % |
| **dso_v_authority** | **0.25 %** | **11.28 %** |

The new coordinate is live, via `f_q` rather than `f_ts` — as expected, since
the relief trades interface-Q tracking for internal voltage margin.

### How to read the phase B result

**`f_ds = 0.000000` at the design point.** With the relief at auth = 20 and
lambda = 0.15, no DSO bus comes within 2 % of a bound at all, so the guard is
fully satisfied and the third criterion cannot *improve* from here — it acts as
a ratchet against degradation rather than as a driver.

Combined with the phase A probes (auth 20 -> 5 improves `f_q` by 6.6 % while
`f_ds` degrades 0 -> 0.00195), the expectation is that phase B pushes
`dso_v_authority` **down** until headroom begins to be consumed, and the filter
settles where that trade balances. If so, the resulting factor is *located by
the method* rather than hand-picked at 20 — which is the whole point of §9.

### Cost

lam 55 min, phase A 110 min (2 batches; evaluations dropped to ~55 min each at
lambda = 0.15, from 73-75 min at 0.9). Phase B started 23:24 at 16 points/poll,
one batch at `--workers 20`.

**Worker count is not "more is better".** Workers are single-threaded (BLAS
pinned to 1, Gurobi already `Threads=1`), so wall time is
`ceil(N/W) * T * max(1, W/cores)` on 20 physical cores. Phase A (N=29):
W=20 -> 2.00 T, W=24 -> **2.40 T**, W=29 -> 1.45 T. Phase B's poll is
`2*live + 2` = 16 points (a coupled lambda pair is appended), so anything from
16 to 20 is a single batch and 24 is again 1.2x slower. Measured at W=18:
CPU/wall = 0.99 per worker — the code's "regresses past 8 (memory-bandwidth
bound)" does not reproduce once BLAS is pinned; that was thread oversubscription.

## 11. Re-run result (converged 2026-08-19 14:24)

12 polls, converged at `delta = 0.0375`. `lambda* = 0.15`, all seven directions
live, none dead. Filter: **79 non-dominated points**.

### The search sells the DSO voltage margin, and the filter cannot stop it

| | f_ts | f_q | worst DS headroom | windows outside [0.90, 1.10] |
| --- | --- | --- | --- | --- |
| design point (auth 20) | 1.312511 | 0.098606 | **+0.0200 pu** | 0 / 12 |
| raw incumbent (auth 5.02) | 1.248436 (-4.88 %) | 0.078699 (-20.19 %) | **-0.0003 pu** | **1 / 12** |

The raw incumbent is the search's own answer and it is **not usable**: it buys
4.9 % of f_ts and 20 % of f_q by spending internal voltage margin until a bus
leaves the statutory corridor — the exact defect this whole entry exists to fix.

`guard_deficit_ds_pu` is a **filter criterion**, so a candidate that improves
`f_ts` and `f_q` while degrading `f_ds` is non-dominated and is accepted.
Nothing in the search bounds how much headroom may be sold. §7 listed
`g6_ds_headroom` as an open item; this is that item materialising.

**A second safeguard eroded the same way.** The incumbent's `lambda_tso` walked
to 0.2518 — past the fitted boundary 0.2337 and well past `lambda* = 0.15` —
giving rho 1.4771: under the *declared* ceiling 1.5 but over the *margined*
target 1.4549. `--rho-margin 0.031` constrains the calibration's **selection**
and nothing else; g3 is evaluated against 1.5, so the search is free to spend
the transfer margin. Same structure: a safeguard applied at one stage and
unenforced downstream.

### Usable result: select from the filter under a headroom requirement

Every filter point carries `ds_headroom_min_pu` (§9 step 0 — the archive change
paid for itself here), so the front can be selected under a stated margin
without re-running anything.

| headroom >= | candidates | f_ts | f_q | auth | lam_tso | lam_dso | gvr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.020 pu | 1 | 1.308868 | 0.110270 | 20.00 | 0.150 | 1.796 | 1.000 |
| 0.015 pu | 9 | 1.308610 | 0.095670 | 10.02 | 0.150 | 1.796 | 0.501 |
| **0.010 pu** | **16** | **1.302146** | **0.089983** | **39.91** | **0.150** | **1.796** | **0.501** |
| 0.005 pu | 24 | 1.300014 | 0.078717 | 20.00 | 0.150 | 1.796 | 0.501 |

**Recommended (headroom >= 0.010 pu):** f_ts -0.79 %, f_q -8.75 % against the
design point, worst headroom +0.0126 pu, and `rho_emp_p95 = 1.3256` — which
respects even the *margined* target 1.4549, so the transfer margin survives too.

Three points about that selection:

* `lambda_dso = 1.796` (2.0x) and `dso_g_v_ratio = 0.501` (0.5x) appear in every
  headroom-respecting row, so those two moves are robust; only
  `dso_v_authority` is contested.
* `lambda_tso` stays at `lambda* = 0.15` in every one of them. The search's move
  to 0.2518 is exactly the transfer margin being spent, and the headroom
  constraint incidentally rejects it.
* **The method wants MORE voltage authority than the hand-picked 20, not less**:
  the selected point sits at `auth = 39.9`. So §6.2's factor of 20 was
  conservative, and the number is now located by the procedure rather than
  chosen — which is what §9 set out to achieve.

The selection was already stable at poll 6 and did not change through poll 12,
so further polling was not buying a better headroom-respecting answer.

### Cost

lam 55 min, phase A 110 min, phase B 12 polls over ~15 h at `--workers 20`
(16 points/poll, one batch each, 53-75 min per evaluation). Total ~17.7 h.

### Carried forward

1. **`g6_ds_headroom` as a hard constraint**, not a filter criterion.
2. **Enforce the rho margin on the search**, not only on the calibration —
   e.g. evaluate g3 against `rho_target / (1 + rho_margin)`.

Both are edits to `CONSTRAINT_NAMES` / `feasibility_constraints` in the shared
`tuning/objectives_v2.py`, which changes the constraint-vector shape for every
existing study and invalidates their archived `hard` fields. Deliberately not
done unilaterally.
