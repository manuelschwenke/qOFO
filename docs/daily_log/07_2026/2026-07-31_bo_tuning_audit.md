# 2026-07-31 — Audit of the BO controller tuning, and Phase-0 repair

**Scope:** `tuning/` package. Audit of why offline Bayesian optimisation of the
cascaded MIQP-OFO weights has not produced usable parameters, plus the first
tranche of corrective changes.

**Trigger:** review of the tuning method ahead of the thesis chapter on
controller synthesis.

---

## 1. What was measured

Evidence comes from the persisted Optuna database
(`tuning/scripts/results/tuning/studies.db`, 19 studies, 1555 scenario-runs) and
from executing the metric code directly. Nothing below is inferred from reading
alone.

### 1.1 The decision variables are not identifiable from the objective

| Diagnostic | Result |
|---|---|
| Parameter spread across the 10 best trials (2 largest studies) | **1.1–3.8 decades in every coordinate** |
| Marginal Spearman \|ρ\| vs. cost, all 9 params | ≤ 0.27; none significant after multiple-comparison correction |
| Random-forest out-of-fold R², log-params → scalarised CVaR-25 cost | **0.09** |
| … → ITAE_v_TS / ITAE_q_PCC / tap switches (individual metrics) | 0.36–0.40 / 0.35–0.43 / 0.28–0.53 |
| … → feasibility (no PF failure), ROC-AUC | 0.67–0.85 |

The physics *is* learnable from the parameters; the scalarisation into a single
9-term weighted cost is what destroys the signal. The optimiser was effectively
solving a feasibility classification problem, not a performance problem.

### 1.2 Structural degeneracy of the search space

The MIQP feasible set contains no weight (`optimisation/miqp_solver.py:429-467`),
so scaling `(∇f, G_w, G_z)` by λ scales the objective by λ and leaves the argmin
unchanged — over the box *and* over the integer lattice. Hence:

- **DSO layer**: exact λ-invariance over `{g_q, dso_g_v, g_w_dso_der,
  g_w_dso_oltc}` (its `G_z` is materially zero, no pinned class) → 4 nominal
  dimensions carry 3 degrees of freedom.
- **TSO layer**: near-exact, broken only by pinned `g_w_gen`, pinned
  `tso_g_q_pcc`, and `g_z_q_pcc=100` where a capability bound binds.
- Each OLTC coordinate is additionally a sigmoid in `log g` with two exactly-flat
  tails (above a finite `ḡ` no tap ever moves; below, `int_cooldown` binds).

**Effective dimension of the nominal 8-dim space: ≈ 4.**

### 1.3 The known-good operating point was never reachable

The hand-tuned `experiments/run_multi_system_ofo.py::make_config()` — reported as
giving good closed-loop behaviour — lies **outside** the search box:

| | hand-tuned | box | |
|---|---|---|---|
| `g_v` | 1e7 | [1e2, 1e5] | 100× above the upper bound |
| `g_w_pcc` | 80 | [1e-1, 30] | 2.7× above the upper bound |

`baseline_002_ieee39.yaml` (`g_w_pcc=100`) and `tests/tuning/conftest.py`
(`g_v=1.2e5`, `g_w_pcc=100`) are outside on the same coordinates. This is why
`--no-warm-start-baseline` was hard-coded in `run_tuning.py`: the workaround
suppressed the symptom of a search space that cannot express the operating point
it is benchmarked against.

Expressed in identifiable ratios, the BO optimum is not a variant of the good
controller but a different one — TSO loop gain `g_v/g_w_der` differs by **1.3e5×**,
`g_w_tso_oltc/g_w_der` by **9.2e4×**, `dso_g_v/g_q` by **1.4e3×**.

### 1.4 The BO tuned a different plant

| | hand-tuned | BO baseline |
|---|---|---|
| `install_tso_tertiary_shunts` | True | False |
| `shunt_dispatch` | integrator | (absent → off) |
| `coordination_mode` | sbx_h | (absent → none) |
| `dt_s` / `dso_period_s` | 20 / 20 | 60 / 10 |

Nothing from the 19-study history transfers.

### 1.5 Concrete metric defects (each reproduced by executing the code)

1. **Divergence earned a discount.** `_itae` dropped non-finite entries and
   returned `0.0`; `_normalise(0.0) = 0.0`. A diverged trajectory therefore
   contributed *zero* to every tracking term and scored exactly `w_pf` = 100 —
   while the median converged scenario-run scored 54–74 and **35–43 % of
   converged runs scored worse than 100**. Divergence was a rewarded search
   direction.
2. **The oscillation term was structurally dead** — `norm_osc ≡ 0` in **100 %** of
   1555 scenario-runs. Not a noise-floor issue: commands are logged per plant
   step but change only on controller ticks, so with `dt_s=60`/`tso_period_s=180`
   each is held for 3 records and `Δu = [0, Δ, 0, 0, Δ, …]`;
   `_count_oscillations` needs two *adjacent* above-floor deltas. Verified: a
   sequence with 4 genuine sign flips counts 0 when held, 4 when not.
3. **The OLTC weights could not be identified by construction.** Taps frozen in
   77 % of clean runs; tap term 0.2 % of mean cost; and
   `norm_tap = (n_tso + n_dso)/5` meant the two weights entered the cost *only
   through their sum*.
4. **Normalisation scales were duplicated** in `cost_components` and
   `extract_metrics` and had to be hand-synchronised.
5. **`validation_set` emitted an invalid network name.** `"base"` is not in
   `SCENARIO_REGISTRY`, so `build_ieee39_net` raised for ~20 % of draws and
   `run_one` converted each into a sentinel cost — a fifth of every validation
   campaign was silently recorded as a power-flow failure.
6. **No timescale separation.** `ScenarioSpec` set `dso_period_s=10` but never
   set `dt_s`, which stayed at the baseline 60. Since the runner tests
   `time_s % period_s < 1`, the DSO fired **every plant step** — one DSO step per
   TSO step, not 18. Every DSO-side weight was fitted under a cascade with no
   timescale separation, which is the premise of the hierarchy.

### 1.6 Corrected claim — the LMI "ceilings" are not an inversion bug

`g_min_required` *is* a stability floor (`analysis/stability_analysis.py:181`,
"Positive = safe"), and `tuning/ceilings.py:299` does assign it to
`Ceilings.g_w_tso_oltc`, which `BOParam.high="ceil"` would use as an upper bound.
I initially recorded this as a bug. It is not: `tuning/ceilings.py:8-17` states
the intent explicitly — above the floor the loop is sufficient-but-sluggish, so
the budget is deliberately spent *below* it, which is the conservatism-gap
argument of `docs/tuning/tuning_strategy.md` §1.

What follows from it is still worth stating: **every sampled point is
non-certified by construction**, so the empirical contraction `rho_emp_p95 < 1`
is the only stability evidence the procedure has. It is currently computed and
recorded but enforced nowhere.

---

## 2. Changes made (Phase 0 — defect repair only)

### `tuning/metrics.py`
- `_itae`: all-NaN input now returns `nan`, not `0.0` (defect 1).
- `_normalise`: non-finite input now returns `inf`, not `1.0` (defect 1).
- `_itae_q_pcc` / `_itae_q_tie`: distinguish *"the network has no such
  interface"* (→ `0.0`) from *"the values are non-finite"* (→ `nan`). Without
  this the repair above marked healthy interface-free runs inadmissible.
- New `_decimate_to_ticks`: restricts command sequences to `tso_active` /
  `dso_active` records before differencing (defect 2).
- New `_count_tap_reversals`, `_tap_wear`: per-transformer operations/day and
  reversals/hour, worst transformer rather than fleet sum (defect 3).
- New `_v_quality`: voltage quality measured against `v_setpoint_pu` —
  spatial-RMS deviation (`zone_v_rms_err_pu`, identical to `rms_v_ts_pu` in
  `cigre_summary_table`), p95 worst-bus deviation, and excess beyond an inner
  `v_ref ± 0.02 pu` quality band. The previous metric used the spatial *mean*,
  under which a zone half at 1.00 pu and half at 1.06 pu scores as perfect.
- New frozen `MetricScales`: single source of truth for the normalisation
  divisors, consumed by both `cost_components` and `extract_metrics` (defect 4).
- New `INFEASIBLE_SENTINEL = 1e6` and `TrajectoryMetrics.infeasible_reason` /
  `.feasible`, so admissibility is reported separately from cost.
- `_voltage_band_excess` and the violation counts now read the corridor from
  `cfg.v_min_pu` / `cfg.v_max_pu` instead of hard-coded 0.9/1.1.
- All `TrajectoryMetrics` fields given defaults.

### `tuning/scenarios.py`
- `ScenarioSpec` gains `dt_s` (default 20.0) and overlays it; defaults are now
  `dt_s=20`, `tso_period_s=180`, `dso_period_s=20` → the intended 9:1 ratio
  (defect 6). A 75-min scenario is 225 plant steps.
- `__post_init__` rejects an unknown network scenario and any
  `dso_period_s < dt_s` (a stated period faster than the plant step is fiction).
- Default `scenario` is now `"base_410"`; `validation_set` draws
  `base_410`/`rural_700` 80/20 instead of `wind_replace`/`"base"` (defect 5).
- Documented the residual `{30, 60, 90}`-minute duration draw in
  `validation_set`, which reintroduces the T² ITAE bias that `design_set` fixes
  its duration to avoid (9× spread).

### `tuning/_types.py`
- `Ceilings` gains a `DIRECTION` class map and a warning that `g_w_*` entries are
  stability **floors** while `g_v` is a **ceiling** (§1.6).

### `tuning/parameters.py`
- `SEARCH_SPACE_VERSION` and `search_space_fingerprint()`.
- `params_from_config` kept a pure extractor; new `out_of_box_params()` reports
  coordinates the space cannot represent (§1.3).

### `tuning/compat.py` (new)
- `LEGACY_PARAM_ALIASES` / `sanitize_legacy_params()`, which *returns* what it
  dropped rather than silently discarding it. Covers the orphaned
  `tso_g_q_tie`, present in every persisted IEEE-39 study but absent from both
  `BO_DIMS` and `MultiTSOConfig`.

### `tuning/tune.py`
- `_guard_search_space`: refuses to resume a study whose fingerprint differs
  (or is absent), unless `--allow-schema-drift`.
- Warm start now fails loudly when the baseline is outside the box, instead of
  being silently skipped.
- `_warn_on_bound_hugging`: flags a best-trial coordinate within 5 % of either
  end of its log-range.
- `_resolve_solver_name`: records the MIQP solver and version in the output
  metadata (`MIQP_SOLVERS` falls back GUROBI→SCIP silently, and the module notes
  the two are not comparable).

### `tuning/scripts/run_tuning.py`
- Paths resolve relative to the file, not the CWD (the default baseline path did
  not exist from the repository root).
- Budget `--n-trials 20 --n-startup-trials 15` (= **5** TPE-guided trials, against
  a docstring claiming "30 random + 120") → `120 / 16`.
- Study name no longer defaults to an existing study; `--no-warm-start-baseline`
  removed.

### Tests
`tests/tuning/test_metrics.py`, `test_scenarios.py`: five new regression guards
(held-command oscillation, per-transformer wear, absent-quantity vs. divergence,
diverged-log admissibility, invalid scenario name, `dso_period_s < dt_s`). Two
existing assertions that encoded the defective behaviour were updated with the
reason recorded inline.

---

## 2b. Measured cost of the `dt_s` correction

`nominal_quiet`, this machine, measured twice with no concurrent load:

| case | wall | records | TSO steps | DSO steps | s/step | taps TS/DS | `itae_q_pcc` |
|---|---|---|---|---|---|---|---|
| `dt=60`, guard on | 62–67 s | 75 | 26 | 75 | 0.82–0.89 | 1 / 0 | 1110.5 |
| `dt=20`, guard on | 104–110 s | 225 | 26 | 225 | 0.46–0.49 | 1 / 0 | **402.4** |
| `dt=20`, guard off | 95–96 s | 225 | 26 | 225 | 0.43 | 1 / 0 | 402.4 |

- `dt_s=20` costs **1.7×**, not the 3× implied by the step count: per-step cost nearly
  halves as fixed setup amortises over more steps.
- **`enable_reachability_guard` costs ~8 %**, not the ~2× estimated during the audit.
  Leave it enabled; the safety signal is worth 8 %.
- **`itae_q_pcc` improves 2.8×** (1110 → 402) purely from the timescale correction. The DSO
  layer was starved of steps, so this is a genuine control-quality result and not only a
  cost.
- The DSO fired on **every** record in both configurations (75/75 and 225/225), confirming
  §1.5 defect 6 directly. What changes is the wall-clock rate: the TSO:DSO step ratio goes
  from 75/26 ≈ 2.9 to 225/26 ≈ 8.7, i.e. the intended ~9:1.
- `nominal_quiet` yields **1 TSO tap and 0 DSO taps**, confirming §1.5 defect 3: the design
  set does not excite the actuators whose weights are being tuned.
- The 12.9 s median recorded in the study database is **not reproducible on this machine**
  even at `dt_s=60` (62 s measured). Those runs predate this hardware; budgets must not be
  derived from them.

**Budget implication.** 5 tune scenarios × 105 s ≈ 9 min/trial serial; 80 trials ≈ 12 h
serial. Measured parallel scaling on this box peaks at ~2.1× (K=6, see
`docs/daily_log/06_2026/2026-06-02_006_cigre_montecarlo.md`), so ~6 h wall.

## 2c. Additional defect found while regenerating the baseline

`save_config_yaml` → `load_config_yaml` **silently changed the type of three fields**:
`measurement_noise` (`MeasurementNoiseConfig` → `dict`), `sbx_config`
(`SBXConfig` → `dict`) and `precondition_exclude_classes` (`tuple` → `list`).
`dataclasses.asdict` flattens nested dataclasses on save and nothing rebuilt them on load,
so any attribute access such as `cfg.sbx_config.k_sched` would raise `AttributeError` deep
inside a run. This was latent only because no baseline had ever included
`coordination_mode="sbx_h"` — it would have fired immediately on the corrected baseline.
Fixed in `tuning/_io.py` via `_NESTED_DATACLASS_FIELDS` / `_TUPLE_FIELDS`, with a
round-trip verification step in `save_baseline.py`.

`tuning/scripts/save_baseline.py` also imported `experiments.002_M_TSO_M_DSO_COMPARE`,
which has since moved to `experiments/archived/`, so the script could not run at all. It now
reads `experiments/run_multi_system_ofo.py::make_config()` and writes
`tuning/scripts/configs/baseline_ieee39.yaml`.

## 2d. Phase-1 diagnostic: curvature attribution (AVR hypothesis refuted)

Measured at the hand-tuned operating point via
`controller.gw_precondition.curvature_spectrum` / `_lambda_max_with`, using the
runner's `pre_loop_hook` to capture the controllers after init (no time loop).

| controller | λ_max(M) | continuous only | integer only | dominant class | `gen` share |
|---|---|---|---|---|---|
| TSO z1 | 1.097 | **0.021** | 1.079 | `tso_oltc` 98.0 % | 0.6 % |
| TSO z2 | 1.789 | **1.775** | 0.164 | `pcc` 74.4 % | 2.3 % |
| TSO z3 | 1.117 | **0.275** | 0.879 | `tso_oltc` 76.1 % | 1.2 % |
| DSO 1 | 0.223 | 0.00095 | 0.222 | `dso_oltc` 99.6 % | — |
| DSO 2 | 0.233 | 0.0019 | 0.232 | `dso_oltc` 99.2 % | — |
| DSO 3 | 0.218 | 0.00054 | 0.217 | `dso_oltc` 99.8 % | — |
| DSO 4 | 0.260 | 0.0051 | 0.257 | `dso_oltc` 98.1 % | — |

**(a) The AVR hypothesis is refuted.** The working hypothesis entering Phase 1 —
that the pinned `g_w_gen` is the de-facto TSO loop gain and therefore the reason
the four TSO weights showed no leverage — is wrong. `gen` contributes **0.6–2.3 %**
of TSO curvature; `g_w_gen = 5e9` genuinely freezes the AVR as intended. The
`avr_band` coordinate is dropped from the Phase-3 design and `g_w_gen` stays
pinned.

**(b) The zones are qualitatively different, so a single `tso_lambda` is the
wrong coordinate.** Zone 2 is continuous-dominated (λ_cont = 1.775, of which
`pcc` is 74 % — so `g_w_pcc` genuinely *is* zone 2's loop gain and is tunable).
Zone 1 is OLTC-dominated with λ_cont = 0.021, i.e. its continuous loop sits ~50×
below the 0.9 target, so a loop-gain coordinate would be inert there. Either make
it per-zone (`zone_g_v` is the existing precedent) or accept that it only bites
in zone 2.

**(c) The `integer_dominated` verdict is very likely a modelling artefact.**
`M = H_V G_w^{-1} H_V^T diag(g_v)` treats every column as a continuous per-tick
move. The OLTC columns are integer with `int_max_step=1` and wall-clock cooldowns
(180 s MT, 60 s NC), so their rank-1 term `‖a‖²/g_w` is an **upper bound** on the
real per-tick effect — which is exactly why `gw_precondition` excludes integer
classes ("their cost is switching frequency, not curvature"). Zone 1's
λ_floor = 1.085 comes almost entirely from that bound. Corroborating evidence:
the hand-tuned configuration operates at λ_max = 1.10–1.79, above the 0.9
"well-damped" target and, in zone 2, approaching the hard OFO stability bound of
2 — and controls well. **Consequence: λ must be defined over the continuous
columns only to be a usable tuning coordinate.**

**(d) The DSO rows are inconclusive and must not be over-read.**
`voltage_curvature_inputs` returns only the *voltage* block, but in priority terms
the DSO's Q-tracking objective dominates its voltage objective by ~500×
(`π_q = g_q·σ_q² = 200 × 5² = 5000` vs `π_v = dso_g_v·σ_v² = 1e5 × 0.01² = 10`).
So the diagnostic is measuring the wrong block for the DSO, and the apparent
result "`g_w_dso_der` contributes < 2 %" is **not** established — it may simply be
invisible here. Settling it requires the `objective_curvature_inputs` extension
(Phase 3), which is the module docstring's own documented next step.

Script: `scratchpad/lambda_floor_attribution.py` (session-local).

## 2e. Phase-1 diagnostic: the weight-scaling invariance is exact

Claim under test: the MIQP feasible set contains no weight
(`optimisation/miqp_solver.py:429-467`), so scaling `(grad_f, G_w, G_z)` by
`lambda > 0` scales the objective by `lambda` and leaves the argmin unchanged —
over the box *and* over the integer lattice, since a positive rescaling of an
objective preserves the argmin over any fixed set. If true, the nominal 8-dim
search space carries a redundant direction per layer and cannot be identifiable
as parameterised.

Measured on one 30-step scenario, comparing full recorded trajectories
(voltages, DER/PCC commands, generator setpoints, OLTC taps, DSO taps and
interface Q) against an unscaled reference:

| case | λ | max abs deviation |
|---|---|---|
| A — every MIQP weight | 0.1 | **3.52e-10** (bit-identical) |
| A — every MIQP weight | 10 | 2.40e+01 Mvar |
| B — all but `g_w_gen` | 10 | 2.39e+01 |
| C — all but `g_z_*` | 10 | 2.40e+01 (identical to A) |
| D — `g_v` alone | 10 | 3.25e+01 |
| **E — every MIQP weight + `shunt_int_g_w`** | **10** | **3.76e-10** (bit-identical) |
| **F — every MIQP weight, shunt dispatch off** | **10** | **3.69e-10** (bit-identical) |

**(a) The invariance is confirmed, and it is exact.** Case A at `lambda = 0.1`
and cases E/F at `lambda = 10` reproduce the reference to ~4e-10 — floating-point
noise. Crucially the OLTC tap trajectories are identical too, confirming the
theoretical argument that the argmin is preserved over the integer lattice, not
only over the box. **The redundant direction is real, so gauge-fixing (searching
dimensionless ratios about a reference point) is the correct remedy rather than
a wider box.**

**(b) Neither candidate breaker in the plan was responsible.** Case C is
bit-for-bit identical to case A, so the slack penalties contribute nothing at
this operating point (`g_z_voltage = 1e-12`, `g_z_current = g_z_interface = 0`,
and the `g_z_q_pcc` capability slack is inactive). Case B differs from A by
0.1 Mvar, so the pinned `g_w_gen` is not responsible either — consistent with
§2d, where the AVR accounts for 0.6–2.3 % of curvature.

**(c) The breaker is the shunt integrator, which was not in the group.**
`controller/shunt_integrator.py:313` advances

    q_eq_aux -= g_h / (2 * shunt_int_g_w),    g_h = grad_g + df_dq_eq

outside the MIQP. `g_h` is a gradient of the same objective and is therefore
homogeneous of degree 1 in the objective weights, but `shunt_int_g_w` is a
separate constant the MIQP scaling never touches — so scaling the objective by
`lambda` scales the shunt integrator's *step* by `lambda`. This explains both
observations: the asymmetry (at `lambda = 0.1` the 10x smaller step never
crosses the 10 Mvar hysteresis band, so nothing commits and the run is exact;
at `lambda = 10` it does) and the magnitude (24.0 Mvar ≈ one
`tso_shunt_msc_q_step_mvar = 25` step). Cases E and F confirm it from both
directions.

**Invariance group (established):**

    {g_v, tso_g_q_pcc, g_q, dso_g_v} u {all g_w_*} u {all g_z_*} u {shunt_int_g_w}

**(d) Consequence for treating the shunt separately.** The shunt integrator
*can* be tuned in its own study, but **only with the MIQP weights frozen**. Its
effective step size is proportional to the objective weights, so a
`shunt_int_g_w` tuned before a change to `g_v` / `g_q` is wrong afterwards by
exactly that factor. Either (i) sequence it strictly after the MIQP weights are
frozen, or (ii) express it in the gauge as the ratio `shunt_int_g_w / g_v`, which
decouples it by construction. Option (ii) is preferable and costs nothing.

Scripts: `scratchpad/invariance_test.py`, `scratchpad/invariance_shunt.py`
(session-local).

## 2f. Phase-3: `objective_curvature_inputs` reverses the DSO reading

§2d flagged the DSO rows as *inconclusive*, because `voltage_curvature_inputs`
exposes only the voltage block while the DSO objective is dominated by
interface-Q tracking. Adding `objective_curvature_inputs` (all weighted output
rows, each with its own weight) and re-running settles it — and reverses the
conclusion:

| DSO | λ_max, V-block only | `dso_der` share | λ_max, full objective | `dso_der` share |
|---|---|---|---|---|
| 1 | 0.223 | 0.4 % | **1.018** | **77.5 %** |
| 2 | 0.233 | 0.8 % | **1.080** | **78.2 %** |
| 3 | 0.218 | 0.2 % | **0.975** | **77.0 %** |
| 4 | 0.260 | 1.9 % | **1.204** | **79.1 %** |

**`g_w_dso_der` is the dominant DSO knob (77–79 %), not a negligible one.** The
mean weighted column energy of the DSO DER block rises from ~0.09 (voltage rows)
to ~106 (full objective) — a factor of ~1000, which is physically what one
expects: DER reactive power sets interface Q directly but moves HV bus voltage
only weakly. The voltage-only diagnostic was measuring the wrong block by three
orders of magnitude.

Two further consequences:

* The DSO continuous loop runs at **λ = 0.91–1.15**, i.e. at or just above the
  0.9 well-damped target. It is a live coordinate, so `dso_lambda` is meaningful
  — unlike the 0.0005–0.005 the voltage-only view suggested.
* Because `voltage_curvature_inputs` returns `None` unless a voltage schedule is
  active with non-zero weight, **the DSO controllers were never preconditioned at
  all**. Tier-2 preconditioning has only ever applied to the TSO layer.

The TSO figures are unchanged, which is the intended backwards-compatibility
check: `tso_g_q_pcc = 0` in the hand-tuned configuration, so the new method
reduces there to exactly the old one.

## 2g. Phase-3: the reparameterised coordinates

**Four coordinates, down from nine nominal (~4 effective).**

| coordinate | meaning | range | drives |
|---|---|---|---|
| `tso_lambda` | TSO loop gain, `λ_max(M)` over continuous columns | `[0.05, 1.20]` uniform | `g_w_der`, `g_w_pcc` |
| `dso_lambda` | DSO loop gain, same definition | `[0.05, 1.20]` uniform | `g_w_dso_der` |
| `tau_der_pcc` | TSO DER-vs-PCC damping ratio, gauge-fixed to geomean 1 | `[1/64, 64]` log | split of the above |
| `dso_v_priority` | DSO voltage vs interface-Q, **relative to the reference** | `[10^-1.5, 10^1.5]` log | `dso_g_v` |

Pinned (with reasons): `g_v`, `g_q` are the gauge — pinning them is what
quotients out the exact scaling redundancy of §2e. `g_w_gen` stays high (AVR is
0.6–2.3 % of curvature, §2d). `tso_g_q_pcc` is zero in the reference.
Bisected, not searched: both OLTC weights (below).

**Bounds are multiplicative windows around the reference, not round numbers.**
A first attempt used absolute ranges and immediately reproduced defect 5 in the
new space: `pi_qpcc`'s reference was **0** (a log coordinate cannot represent
zero) and `pi_dsov`'s reference sat at 7.5 % of a four-decade range. Defining
ratio coordinates relative to the reference puts it at exactly 1.0 — the
geometric centre — *by construction*, so the search space cannot fail to contain
the point it is benchmarked against. Verified: both ratio coordinates now report
50.0 %.

**`precondition_g_w` gained three parameters**, all defaulting to the previous
behaviour (the 17 existing tests pass unchanged):

* `mode='cap'|'set'` — `'cap'` only adds damping, correct for a production
  safety net. `'set'` makes `λ_max` track the target in both directions, which
  is required when the target is a tuning coordinate: under `'cap'` every target
  above the current `λ_max` is the same no-op, so the coordinate is flat over
  much of its range (measured: 3 of 4 test targets were no-ops).
* `class_scale_overrides` — the *shape* knob. With `kappa` fixing the gain, the
  per-class ratios are the only remaining freedom. Verified: a requested 4×
  DER/PCC ratio change came out at exactly 4.000× with `λ` held on target.
* `lambda_scope='all'|'preconditioned'` — measured effect: with integer columns
  dominating, `'all'` returns `integer_dominated` and does nothing (λ stays
  122.8) while `'preconditioned'` reaches the target exactly (0.5000).

**End-to-end verification** (real controllers, `pre_loop_hook`, three λ settings):

| λ target | TSO z1 | TSO z2 | TSO z3 | DSO 1–4 | DSO_1 `g_w_dso_der` |
|---|---|---|---|---|---|
| 0.2 | 0.205 | 0.244 | 0.213 | 0.200 | 3925 |
| 0.5 | 0.505 | 0.544 | 0.513 | 0.500 | 1570 |
| 1.0 | 1.005 | 1.044 | 1.013 | 1.000 | 785 |

All seven controllers track the target; the derived `g_w` scales exactly as
`1/λ` (3925 / 1570 = 2.50 = 0.5/0.2; 1570 / 785 = 2.00). Excluded classes are
untouched (`gen` = 5e9, `tso_oltc` = 5000) as specified. **The DSO is
preconditioned for the first time** — it was skipped entirely before §2f.

Cross-check worth noting: at λ_DSO = 1.0 the derived `g_w_dso_der` is 785 against
a hand-tuned 800, so the hand-tuned value corresponds to λ_DSO ≈ 0.98 —
independently consistent with the 0.91–1.15 measured in §2f.

**Caveat on `lambda_scope='preconditioned'`.** It makes the coordinate
identifiable but no longer bounds the *true* worst-case contraction: in a probe
with `dso_g_v` 50× the reference, `λ_cont` sat on target while `λ_all` reached
19. Constraint **g3** (`rho_emp_p95 < 1`) is what catches those regions
empirically, which is precisely why it is load-bearing rather than decorative.

**OLTC bisection** (`tuning/bisect_switching.py`). Both OLTC weights are
calibrated against an operational taps/day budget by log-space bisection rather
than searched. Justification is structural: `G_w` is diagonal and each OLTC block
is one replicated scalar appearing nowhere else, so the value function is a
pointwise minimum of functions affine in `g` — concave — and its supergradient
`‖w_i*(g)‖²` (with `int_max_step=1`, exactly the number of taps that move) is
monotone non-increasing in `g`. Both tails are *exactly* flat (above a finite
`ḡ` no tap moves; below, the cooldowns bind), and a density-ratio sampler cannot
represent a plateau — a large part of why these two coordinates scored
`|ρ| ≤ 0.27`. The implementation brackets before bisecting, converges to a ±20 %
band rather than a point, takes the median across scenarios, and reports
`plateau_high` / `plateau_low` explicitly so a slack or unreachable budget is
never mistaken for a tuned value. Per-solve monotonicity does not chain into
per-trajectory monotonicity; that caveat is documented in the module rather than
assumed away.

## 2h. Phase-2: the constrained objective, wired behind `--reparam`

`tuning/objectives_v2.py`. Feasibility leaves the cost entirely and becomes six
Optuna constraints (`<= 0` feasible), aggregated worst-case across scenarios:

| | definition |
|---|---|
| g1 | scenarios that diverged or raised |
| g2 | hard-corridor excess vs `cfg.v_min_pu` / `v_max_pu` |
| g3 | `max rho_emp_p95 − 1` — the empirical contraction |
| g4 | worst event-anchored settling time − 900 s |
| g5a | tap operations per day **per transformer** − budget |
| g5b | tap reversals per hour − 4 |

`g2` is expected to be inactive (the corridor is ±7–8 % around a 1.03 pu
setpoint) and that is fine: an always-satisfied *constraint* is free insurance
and exerts no pull on the sampler, whereas an always-zero *cost term* is dead
weight. The discriminating voltage signal lives in the scalar instead (§2a).
`g5` is split because reversals alone miss monotone over-switching while an
absolute budget alone misses chattering inside it; `g5a` is also the Tier-2′
bisection target, so the two mechanisms close a loop — the bisection sets the
OLTC weight to hit the budget and `g5a` stops the search over `λ`/`τ` from
quietly reopening it.

The scalar keeps tracking and utilisation only, all terms O(1) — against the
legacy mix of a binary 0/100 term, a 1000-weighted hinge and an order-10
tracking term.

Wiring in `tuning/tune.py --reparam`:

* `TPESampler(constraints_func=…)`. Constrained TPE partitions trials by
  feasibility *before* fitting its good/bad densities, so an infeasible trial's
  objective value never attracts the sampler. A penalty term cannot do that,
  which is how divergence became a profitable direction.
* `best_feasible_trial` replaces `study.best_trial`, which ignores constraints
  in a single-objective study and would report an infeasible point as the answer.
* A per-constraint violation table prints at the end, so an empty feasible set
  identifies *which* limit binds rather than inviting a blanket relaxation.
* Feasibility short-circuit: a trial abandons after its first diverged scenario.
  No information is lost — one divergence already makes the trial inadmissible —
  and roughly half of all historical trials landed in divergent regions.
* Warm start enqueues the reference point, which is representable by
  construction. Verified in a live smoke run: trial 0 = `tso_lambda 0.5`,
  `dso_lambda 0.5`, `tau_der_pcc 1.0`, `dso_v_priority 1.0`.
* The HTML report is skipped in this mode: its centrepiece is the
  certificate-ratio table, which has no meaning for ratio coordinates.

`interval_settling_table` (`experiments/helpers/rms_replay.py`) gained an
optional `windows` argument so settling can be anchored on contingencies rather
than a fixed grid — a settling time is only meaningful relative to a
disturbance, and a uniform grid mostly measures quiet intervals. Backwards
compatible; verified against an analytic first-order step (τ = 1.2 s → 5.0 s to
the 2 % band).

## 2i. Phase-4: excitation-gated design set

`tune_set_v2()` — five 75-min scenarios, both networks:

| scenario | network | events | intent |
|---|---|---|---|
| `v2_quiet_spring` | base_410 | 0 | keeps a low-excitation case in the set |
| `v2_gen_trip` | base_410 | 2 | impulsive, absorbable by continuous actuators |
| `v2_undervoltage_ramp` | base_410 | 7 | **one-way load steps, no restore** |
| `v2_overvoltage_rural` | rural_700 | 3 | summer-night minimum, DER lifts voltage |
| `v2_dual_rural` | rural_700 | 5 | both layers stressed at once |

The ramps deliberately omit a restore. A trip/restore pair lets the continuous
actuators ride through, whereas sustained one-way drift exhausts reactive
reserve and hands authority to the tap changers — which is the only way the OLTC
weights become identifiable. `rural_700` is covered for the first time.

**Tune/holdout split on ISO-week parity**, not on random days: SimBench profiles
are strongly autocorrelated within a day, so a day-level split leaks. This
caught a real slip in the first draft — the legacy `_T_WINTER` (2016-01-14) is
ISO week 2, an *even* i.e. holdout week, so the v2 winter scenarios initially
sat inside the holdout calendar. Moved to 2016-01-21 (week 3) and pinned by
`test_tune_and_holdout_calendars_are_disjoint`. Tune weeks are now {3, 15, 27},
holdout all even, overlap empty.

`tuning/scripts/audit_design_set.py` — the admission gate. Each candidate runs
once at the reference weights and is admitted only if it moves both OLTC
classes, drives generator reactive reserve below 0.15, and produces ≥ 0.01 pu
peak TS deviation. Exits non-zero on failure. It reports **set-level** coverage
separately, because a weight is identifiable if *some* scenario exercises it,
not if every one does.

## 2j. Smoke run: the reference point failed my own constraints

A 3-trial `--reparam` run against the regenerated baseline validated the whole
pipeline end-to-end — and immediately caught a mistake of mine. Trial 0 is the
hand-tuned reference (warm-started at coordinates `tso_lambda 0.5`,
`dso_lambda 0.5`, `tau_der_pcc 1.0`, `dso_v_priority 1.0`, confirming
representability), and it came out **infeasible on three of six constraints**:

| constraint | reference | my limit | verdict |
|---|---|---|---|
| g1 diverged | 0 | 0 | OK |
| g2 corridor | 0.0022 pu/step | 1e-4 | **violated** |
| g3 contraction | ρ = 0.929 | 1.0 | OK, 7 % margin |
| g4 settling | 1200 s (censored) | 900 s | **violated** |
| g5a tap ops | 96.4 /day | 10 /day | **violated** |
| g5b reversals | 0.80 /h | 4 /h | OK |

Two of the three are the *limit's* fault, not the controller's.

**g5a was a unit error.** 96.4 ops/day is 5.0 actual taps in a 75-minute
disturbance window — 4.0 taps/hour during a contingency, entirely ordinary. The
design scenarios are event-dense windows and a real day is mostly quiet, so
extrapolating one to the other inflates the figure by roughly the stressed-to-
total time ratio (~19×). `_tap_wear` now reports **per hour of simulated
operation**, with the extrapolation caveat documented at the call site; convert
to a daily budget only against a representative daily profile.

**g4 is censored, and the limit is now deliberately inactive.** The value equals
the window width exactly, meaning some signal never entered the 2 % band within
20 minutes. That could be real or it could be the metric applied to a channel
with a persistent offset. Until that is resolved the default sits *above* the
window width, so g4 cannot silently reject every candidate for an instrumental
reason. Resolving which signal is censored is an open item.

**g2 is real, and contradicts my own prediction.** §2h argued g2 would be
inactive because the corridor is ±7–8 % around a 1.03 pu setpoint. Measured, the
reference does touch the band at 0.0022 pu/step. Limit recalibrated with margin.

**The underlying mistake is worth naming.** I invented all six limits as round
numbers — precisely the circular-calibration failure I criticised in the cost
weights (six successive revisions chasing optimiser output) and then committed
myself one layer down. `ConstraintLimits.from_reference(reference, margin=1.5)`
now derives them from a measured reference run, with `rho_emp_p95` held at 1.0
because that one is a stability threshold from theory rather than an operational
preference. Pinned by `test_limits_can_be_calibrated_from_a_reference`, which
asserts the reference passes the limits derived from it.

Also confirmed by the run: ~172 s per scenario (861 s for five), and the
performance scalar is well-conditioned — per-scenario values 1.99 to 3.05,
CVaR-25 2.94, mean 2.67, with no term dominating.

## 2j2. Runbook

Ordered; each step gates the next.

```bash
# 0. Regenerate the baseline from the hand-tuned config (once, after any
#    change to run_multi_system_ofo.make_config).
python -m tuning.scripts.save_baseline

# 1. EXCITATION GATE.  Do not proceed on a non-zero exit: a weight whose
#    actuator never moves cannot be identified, whatever the budget.
python -m tuning.scripts.audit_design_set --set tune_v2 \
    --csv results/tuning/excitation_audit.csv

# 2. Calibrate the two OLTC weights against the switching budget.
#    ~10 evaluations per class.  Check `status == "bracketed"`; a
#    plateau_high/plateau_low result means the budget does not bind, or is
#    unreachable, and must NOT be read as a tuned value.
#    (driver script still to be written; calibrate_switching_price is ready)

# 3. Phase-A calibration sample: ~120 log-uniform draws from the prior, used to
#    set MetricScales non-circularly and to run the no-dead-terms check.
#    (driver script still to be written)

# 4. The tuning run itself.
python -m tuning.tune --reparam \
    --baseline tuning/scripts/configs/baseline_ieee39.yaml \
    --n-trials 80 --n-startup-trials 12 \
    --study-name v5_reparam \
    --storage sqlite:///results/tuning/studies.db \
    --output configs/tuned_params_reparam.yaml

# 5. Holdout, ONCE, on the selected point plus the hand-tuned reference.
#    Acceptance: the BO optimum must beat the hand-tuned point.
```

**Measured cost.** ~175 s per scenario on the real plant (shunts + integrator +
`sbx_h` + preconditioning) — not the 105 s measured against the shunt-less
`baseline_002_ieee39`, and not the 13 s recorded in the historical database.
Five scenarios ≈ 15 min per trial serial; 80 trials ≈ 20 h serial, ≈ 9–10 h wall
at the measured ~2.1× parallel ceiling. **My budget estimate has been wrong
twice; re-measure before committing to anything larger.**

## 3. Status and next steps

Phase 0 (defect repair) is complete. Still open, in order:

1. **Phase 1 — diagnostics.** Empirical invariance test (bit-identical
   trajectories under full-group λ scaling; quantified divergence when `g_w_gen`
   or `g_z` is held out); `lambda_floor` attribution per zone via
   `precondition_g_w`. The latter decides whether `g_w_gen` enters the search
   space: the 2026-06-23 log records `integer_dominated` with `λ_floor = 2.2`
   (zone 1) and `1.4` (zone 3) from the *pinned* columns alone, i.e. above the
   0.9 target and, for zone 1, near the hard stability bound of 2.
2. **Phase 0e — regenerate the baseline** from `make_config()` so the tuning
   targets the plant actually in use (§1.4).
3. **Phase 2 — constrained-scalar objective.** Feasibility (divergence, corridor,
   `rho_emp_p95 > 1`, non-settling, tap wear) moves into Optuna constraints;
   the scalar keeps tracking and utilisation only. Reuse
   `experiments/helpers/rms_replay.py::interval_settling_table` for settling.
4. **Phase 3 — reparameterization** into per-layer loop gain `λ_max(M)`,
   gauge-fixed per-class relative damping, and objective priority ratios, with
   both OLTC weights calibrated by 1-D bisection against a taps/day target.
5. **Phase 4 — design set** with an excitation gate, then the run.

**Acceptance criterion carried forward:** the BO optimum must beat the hand-tuned
`make_config()` on a held-out scenario set. If it does not, the deliverable is
the methodological evidence for a setting already in use — which is still worth
having, and is the honest outcome to report.

## 4. Open questions

- Whether the TSO curvature is AVR-dominated (Phase 1 decides). If it is, no
  choice of `g_w_der`/`g_w_pcc` can move `λ_max(M_TSO)` to target and the TSO
  half of any search over those two weights is inert.
- Per-trajectory monotonicity of tap count in `g_w_*_oltc` is only established
  per-solve; the bisection must bracket rather than assume.
- `has_slack = any(diag(G_z) > 0)` (`optimisation/miqp_solver.py:426`): with
  `g_z_voltage = 1e-12 > 0` the slack branch is active for *all* output rows, so
  rows with `g_z = 0` (`g_z_current`, `g_z_interface`) get an unpenalised slack
  variable and their output constraints are **vacuous, not hard**. Separate
  ticket; not touched here.
