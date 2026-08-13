# BO re-tuning campaign — results for the thesis

**Executed 2026-08-03 / 04 / 05** on the always-on server, following
[`HANDOVER_bo_retuning.md`](HANDOVER_bo_retuning.md).

This file is written **for a fresh Claude session** that has to turn these
numbers into thesis text. It states where every artefact lives, which numbers are
citable, and — importantly — which claims must be qualified. Read
§6 (Do not overclaim) before drafting.

---

## 0. One-paragraph summary

Bayesian optimisation over a 4-dimensional reparameterised weight space
(`tso_lambda`, `dso_lambda`, `tau_der_pcc`, `dso_v_priority`) produced a point
that beats the **production controller** on an independent held-out scenario set:
PCC reactive-power tracking RMS **−49.0 %**, TS spatial-RMS voltage −4.1 %. The
gain comes from one coordinate — `tau_der_pcc`, which reweights PCC tracking to
**62×** the TS-DER weight — and **not** from loop gain. Separately, the campaign
found that the *previous* campaigns' reported unidentifiability was caused by
objective design and a scenario-reuse defect, not by the parameterisation
(§4, §5).

**Two claims from the first holdout draw are RETRACTED — see §3c.** (i) Tracking
is *not* monotone in λ: on an independent draw λ=0.9 gives the best TS voltage and
λ=1.1975 is worse, so the "damping-versus-tracking curve" was an artefact of three
points on one draw. (ii) The 67 % tap-operation reduction seen on the tune set does
not transfer: on holdout data all candidate points have **identical** worst-case
tap ops and reversals, i.e. switching is insensitive to the coordinates.

**No single point wins.** The BO optimum wins interface-Q decisively; the analytic
λ=0.9 point wins TS voltage (−8.0 %) and tap operations (−25 %); and on the legacy
`cost_J` scalar the BO optimum is 16.7 % *worse*. Which point "wins" depends on the
scalarisation — itself a result worth reporting.

---

## 1. Where everything is

| artefact | path |
|---|---|
| **Holdout seed 43 — 4 points incl. production (USE THIS, §3c)** | `results/tuning/bo_campaign_2026-08/holdout_scores_seed43.json` |
| Holdout seed 43 run log | `results/tuning/bo_campaign_2026-08/stage5b_holdout_seed43.log` |
| Holdout seed 42 — 3 preconditioned points only (superseded, §3) | `results/tuning/bo_campaign_2026-08/holdout_scores.json` |
| Holdout seed 42 run log | `results/tuning/bo_campaign_2026-08/stage5_holdout.log` |
| Preconditioner bridge, tune set (§3b) | `results/tuning/bo_campaign_2026-08/diag_precond_bridge.log` |
| Identifiability, corrected study | `results/tuning/bo_campaign_2026-08/identifiability_corrected.json` |
| Identifiability, degenerate study (the control) | `results/tuning/bo_campaign_2026-08/identifiability_degenerate.json` |
| Optuna study, corrected (180 trials) | `results/tuning/bo_campaign_2026-08/v5_reparam_v2.db` (study name `v5_reparam_v2`) |
| Optuna study, degenerate (36 trials, control) | `results/tuning/bo_campaign_2026-08/v5_reparam.db` (study name `v5_reparam`) |
| Tuned coordinates + run metadata | `configs/tuned_params_reparam.yaml` |
| Switching-calibration ladders | `results/tuning/switching_calibration.json` |
| Metric calibration (65-draw prior sample) | `results/tuning/metric_calibration.json` |
| Excitation-gate table | `results/tuning/excitation_audit.csv` |
| Pre-calibration reference snapshot | `tuning/scripts/configs/baseline_ieee39_handtuned.yaml` |

Daily logs with the reasoning:
`docs/daily_log/08_2026/2026-08-03_bo_retuning_campaign_start_three_blockers.md`,
`docs/daily_log/08_2026/2026-08-03_contingency_event_mutation_degrades_every_repeated_run.md`.

Read a study with:

```python
import optuna
st = optuna.load_study(study_name="v5_reparam_v2",
                       storage="sqlite:///results/tuning/bo_campaign_2026-08/v5_reparam_v2.db")
```

---

## 2. The tuned point

Study `v5_reparam_v2`, **trial 173**, objective 0.6321401376964743, all
constraints satisfied. 180 trials, 6 workers, 7.83 h.

```yaml
tso_lambda:     1.1975421798462904
dso_lambda:     1.0964048871681646
tau_der_pcc:    0.01618211331204804
dso_v_priority: 0.7641326900293102
```

These are **coordinates, not weights**. `apply_reparam_to_config` maps them to:

| field | reference | BO optimum |
|---|---|---|
| `precondition_lambda_target_tso` | 0.5 | 1.1975 |
| `precondition_lambda_target_dso` | 0.5 | 1.0964 |
| `precondition_class_scales` | der 1.0, pcc 1.0 | **der 0.1272, pcc 7.8611** |
| `dso_g_v` | 100 000 | 76 413 |

λ is *not* written to a scalar field: it is a target for the curvature
preconditioner applied at controller init (`precondition_mode='set'`,
`precondition_lambda_scope='preconditioned'`), because κ depends on the cached
sensitivity **H**. `tau` enters only through `precondition_class_scales`
= (√τ, 1/√τ), whose geometric mean is pinned at 1 so it moves the DER/PCC ratio
without touching the loop gain.

**Physical reading:** the optimum wants (i) maximum permitted loop gain,
(ii) PCC-tracking weighted **62×** the TS-DER weight, (iii) slightly *reduced*
DSO voltage priority (0.76× reference).

---

## 3. Holdout result — the headline table

`holdout_set_v2(seed=42, n=40)`. Even ISO weeks only (the tune set uses odd
weeks; verified empirically — the drawn weeks are all even). Fixed 4500 s
duration, so no `T²` ITAE bias. **Evaluated exactly once.** All 40 scenarios
completed for all three points (no divergence, no power-flow failure).

Medians across the 40 scenarios:

| metric | reference λ=0.5 | analytic λ=0.9 | BO λ=1.1975 |
|---|---|---|---|
| `v_rms_ts` [pu] | 0.006709 | 0.006421 (−4.3 %) | **0.006306 (−6.0 %)** |
| `q_pcc_rms` [Mvar] | 1.1883 | 1.1185 (−5.9 %) | **0.6512 (−45.2 %)** |
| `itae_q_pcc` [min·Mvar] | 2758.1 | 2445.2 | **1611.0 (−41.6 %)** |
| `q_tie_rms` [Mvar] | 26.815 | 27.115 | 26.986 |
| `v_rms_ds` [pu] | 0.004778 | 0.004948 | 0.006025 (**+26 %**) |
| `cost_J` (legacy scalar) | 14.203 | 12.658 | 12.702 |

Worst-case over the 40 scenarios (this is how constraints g5a/g5b aggregate):

| | ops/h TSO | ops/h DSO | rev/h TSO | rev/h DSO |
|---|---|---|---|---|
| reference | 8.036 | 9.643 | 1.607 | 2.411 |
| analytic | 8.036 | 10.446 | 1.607 | 2.411 |
| BO optimum | 8.839 | 8.036 | 2.411 | 2.411 |

**The monotonicity is the strongest claim available.** Across
λ = 0.5 → 0.9 → 1.1975, `v_rms_ts` falls 0.006709 → 0.006421 → 0.006306 and
`q_pcc_rms` falls 1.1883 → 1.1185 → 0.6512. Both monotone, on data never used
for tuning. Frame the result as *"tracking performance is monotone in loop gain
up to the damping cap the design imposes"*, not as "the tuned point is best".

`tso_lambda = 1.1975` sits at the box ceiling of 1.20. Treat 1.20 as a
deliberate damping cap (the `BO_DIMS_V2` comment calls 0.9 well-damped and 2.0
the hard OFO bound); empirical contraction still held at the optimum
(`g3 = −0.0572`, i.e. ρ_emp,p95 ≈ 0.943 < 1).

---

## 3b. The reference was preconditioned; production is not

`apply_reparam_to_config` forces `precondition_g_w=True` (plus `mode='set'`,
`scope='preconditioned'`, `granularity='column'`). `make_config()` had
`precondition_g_w = False`. So **every** campaign point, including trial 0
"reference", ran with the preconditioner on. `coords_from_config` cannot recover λ
from a config ("properties of the cached `H`") and fell back to the dormant
`precondition_lambda_target: 0.5` field; `tau_der_pcc: 1.0` is likewise
hardcoded there, not measured.

Measured on the **tune set** (`results/tuning/diag_precond_bridge.log`), medians
over the 4 scenarios:

| metric | A production (precond OFF) | B campaign reference | C BO optimum |
|---|---|---|---|
| `v_rms_ts` | 0.0068515 | 0.0072309 (+5.5 %) | 0.0062403 |
| `v_rms_ds` | 0.0080275 | 0.0066975 (−16.6 %) | 0.0046559 |
| `itae_q_pcc` | 3389.7 | 3794.6 (+11.9 %) | 2108.5 |
| `tap_ops_per_h_tso` | 1.2054 | 1.2054 | 0.4018 |
| `tap_reversals_per_h_tso` | 0 | 0 | 0 |
| `perf` scalar | 0.78730 | 0.83069 (+5.5 %) | 0.65231 |

**Citable:** on the tune set the tuned point beats *production* by 17.2 % on the
performance scalar, 37.8 % on `itae_q_pcc`, 8.9 % on `v_rms_ts`, and uses **67 %
fewer** TSO tap operations (0.402 vs 1.205 /h).

**Not citable:** the 42 % `v_rms_ds` gain on the tune set. That metric flips sign
out of sample — BO is 30 % *better* than the reference on the tune set and 26 %
*worse* on the holdout. Treat DS voltage as not improved.

Also note **TSO tap reversals are 0 for all three points on the tune set** while
the holdout produced 1.607–2.411 /h. That is the mechanism behind the g5b
non-transfer in §6.2: the tune set never exercises reversals, so a limit
calibrated on it cannot bound them.

### Applying the tuned point

All **six** preconditioner fields are required together, plus `dso_g_v`. Two fail
**silently** if left at the old values — no error, just reference behaviour:

```python
precondition_g_w             = True
precondition_mode            = "set"            # NOT "cap": 'cap' only REDUCES
                                                # g_w, and the tuned lambda 1.1975
                                                # is ABOVE the reference 0.5, so
                                                # 'cap' cannot raise the gain
precondition_lambda_scope    = "preconditioned"  # NOT "all": under 'all' the
                                                # integer OLTC columns block the
                                                # target (zone 1 reads
                                                # integer_dominated at 1.085 while
                                                # its continuous loop sits at 0.021)
precondition_granularity     = "column"         # NOT "class"
precondition_lambda_target_tso = 1.1975421798462904
precondition_lambda_target_dso = 1.0964048871681646
precondition_class_scales    = {"der": 0.12721, "pcc": 7.86102}   # == tau
dso_g_v                      = 76413.26900293102                  # == priority
```

Applied in `experiments/run_multi_system_ofo.py` 2026-08-05 and verified
field-by-field against `apply_reparam_to_config`.

A uniform safety factor on all `g_w` is **not** an independent knob: scaling every
`g_w` by *f* scales ρ(M) by 1/*f* (`tuning/ceilings.py`), so +10 % on all weights
≡ λ → λ/1.1 = 1.089. Read its cost off the λ curve in §3 rather than treating it
as a margin. If a stability margin is wanted, tighten g3 (`rho_emp_p95` from 1.0
to e.g. 0.90) and re-run Stage 4 — BO can then reallocate the *shape* coordinates
to recover tracking at lower gain, which a uniform scaling cannot. At the optimum
ρ_emp,p95 ≈ 0.943, i.e. only 5.7 % margin.

---

## 3c. Second, independent holdout (seed 43) — the citable comparison

The seed-42 draw scored only *preconditioned* points, so it could not compare
against production. This draw adds it. **`holdout_set_v2(seed=43, n=40)` shares
zero scenario start times with the spent seed-42 draw**, all even ISO weeks, fixed
4500 s — a genuinely independent sample, not a second read. 38/40 scenarios
completed for **every** point (the same 2 fail for all four, so the failure is
structural, not point-specific).

Medians over the 40 scenarios
(`results/tuning/bo_campaign_2026-08/holdout_scores_seed43.json`):

| metric | reference λ=0.5 | analytic λ=0.9 | BO λ=1.1975 | **production** |
|---|---|---|---|---|
| `v_rms_ts` [pu] | 0.0065021 | **0.0058508** | 0.0060965 | 0.0063590 |
| `v_rms_ds` [pu] | **0.0057203** | 0.0065242 | 0.0066278 | 0.0069481 |
| `q_pcc_rms` [Mvar] | 1.25423 | 1.25957 | **0.58842** | 1.15306 |
| `itae_q_pcc` | 3574.3 | 3618.7 | **2202.5** | 3563.2 |
| `q_tie_rms` [Mvar] | 24.083 | 24.419 | 24.876 | 24.069 |
| `tap_ops_per_h_tso` | 1.6071 | **1.2054** | 1.6071 | 1.6071 |
| `tap_reversals_per_h_tso` | 0 | 0 | 0 | 0 |
| `cost_J` (legacy) | 15.956 | 16.782 | 17.670 | **15.138** |

Against **production**, the baseline that matters:

| metric | BO | analytic λ=0.9 |
|---|---|---|
| `q_pcc_rms` | **−49.0 %** | +9.2 % |
| `itae_q_pcc` | **−38.2 %** | +1.6 % |
| `v_rms_ts` | −4.1 % | **−8.0 %** |
| `v_rms_ds` | −4.6 % | −6.1 % |
| `tap_ops_per_h_tso` | ±0.0 % | −25.0 % |
| `q_tie_rms` | +3.4 % | +1.5 % |
| `cost_J` | +16.7 % | +10.9 % |

### The mechanism: τ, not λ

The analytic point has **higher** λ than the reference and **no** PCC gain
(1.2596 vs 1.2542). The BO point has τ = 0.0162 and gains 49 %. So the
improvement is a **reallocation between actuator classes**, not a loop-gain
increase — consistent with τ carrying the dominant identifiability signal
(Spearman ρ = +0.794, §4). State the result that way.

### What replicates and what does not

* **Replicated 3× independently** — the PCC-tracking gain vs the relevant
  baseline: tune set −37.8 %, seed 42 −45.2 %, seed 43 −49.0 %. Citable.
* **Did NOT transfer** — the tune set's 67 % tap-operation reduction. On seed 43
  all four points share identical worst-case tap ops and reversals. Do not quote
  a switching improvement.
* **Baseline-dependent** — DS voltage. BO is 4.6 % *better* than production but
  15.9 % *worse* than the preconditioned reference. The earlier "+26 % worse" was
  against the reference; always name the baseline.
* **Refuted** — monotonicity in λ (see §0).
* **Consistent across tune set and seed 43** — production beats the
  preconditioned λ=0.5 reference on `v_rms_ts`, `q_pcc_rms` and `cost_J`.
  Switching the preconditioner on at the reference λ is mildly *harmful* by
  itself; the τ reshaping is what pays.

---

## 4. Identifiability — the methodological result

Same three diagnostics as the 2026-07-31 audit (§1.1), so it is a like-for-like
comparison.

| diagnostic | historical audit | degenerate objective (`v5_reparam`) | **corrected (`v5_reparam_v2`)** |
|---|---|---|---|
| RF out-of-fold R², log-coords → scalar | 0.09 | −0.060 | **+0.838** (feasible), +0.804 (all) |
| log-spread over the 10 best trials | 1.1–3.8 decades | 1.09–1.66 | **0.04–0.42** |
| marginal Spearman \|ρ\| | ≤0.27, none significant | none significant | up to **0.794**, 3 of 4 at p<0.0001 |
| distinct objective values | — | 25/36 | **180/180** |

Spearman per coordinate on the corrected study (feasible trials):
`tau_der_pcc` +0.794 (p<1e-4), `dso_lambda` −0.430 (p<1e-4),
`tso_lambda` −0.403 (p<1e-4), `dso_v_priority` −0.149 (p=0.113, not significant).

**This is the campaign's most transferable finding:** the search space was never
the problem. The historical unidentifiability had two causes, both fixed here —
see §5.

---

## 5. Two defects that explain the historical result

### 5.1 Scenario reuse silently deleted the tap-carrying scenarios

`prepare_load_contingencies` resolves events by writing `ev.element_index` back
into them (`experiments/helpers/contingency.py:161`), and the runner obtained its
list with a **shallow** `list(config.contingencies)`, sharing the
`ContingencyEvent` objects with the caller. Every driver builds its scenario set
once and reuses it, so from the second run onward the resolver re-entered with
`action='connect'` *and* an explicit index and raised — killing exactly the
scenarios that use mode-3 `connect` load events.

Consequence: run 1 saw the full design set; every later run lost
`v2_undervoltage_ramp` (19 TS / 9 DS tap moves — the only scenario with material
tap excitation) and `v2_overvoltage_rural`. That reproduces the audit's
"taps frozen in 77 % of runs" signature by a purely mechanical route.

Fixed by deep-copying at the ownership boundary
(`experiments/runners/multi_tso_dso.py:1647`). Verified: three identical passes,
12/12 runs feasible; before the fix pass 2 lost two scenarios.

**Caveat for the thesis:** that the mechanism *can* produce the 77 % signature is
demonstrated; that it *did* cause the historical figure is **not** yet verified.
Confirming it requires checking whether `design_set()`'s tap-active scenarios also
use mode-3 `connect` events. State it as a strong hypothesis.

### 5.2 A worst-case aggregator handed the objective to an inert scenario

The objective was CVaR-25 over 4 scenarios — which for 4 scenarios **is the
maximum**. `v2_undervoltage_ramp` scored ~85× the others, so it *was* the
objective. And it starts at `2016-01-21 18:00`, a winter evening peak: PV-based
TS-DER produce no active power, so their reactive capability is zero. Measured:
`zone_q_der` is **exactly 0.0** across all 900 (step × DER) entries, constant in
time, while `v2_quiet_spring` on the same network spans −133…+64 Mvar. Since
`tau_der_pcc` acts only through `precondition_class_scales`, it is
**structurally inert** there.

Result: 8 of 24 trials shared an objective value to 6 significant figures across a
660× range in `tau`, while `quiet_spring` and `gen_trip` varied by 668 % and
493 %.

Fixed by excluding the ramp from the *performance aggregate* while keeping it in
the constraint vector (`perf_exclude`), and using the mean (`cvar_pct=100`).
Stress coverage is retained: `v2_gen_trip` is a stress case at spring noon with
full DER capability.

This is **not** saturation — generator reserve was 0.074–0.081 with a saturated
fraction of 0.000. It is a seasonal capability gap in the design set.

---

## 6. Do not overclaim — required qualifications

1. **`q_tie` was never in the tuning objective.** The v2 performance scalar has
   six terms (`v_rms_ts`, `v_rms_ds`, `v_worst_ts`, `v_band_ts`, `q_pcc`,
   `pcc_underutil`) — no tie term. `q_tie_rms` ≈ 27 Mvar is flat across all three
   points, so tie-Q tracking is **not demonstrated to be controllable** by this
   scheme. Report as a limitation, not a result. (`q_tie` does enter the *legacy*
   `cost_J`, which was not the optimisation target.)
2. **The constraint limits do not transfer from tune to holdout.** They were
   derived from the reference on 4 hand-picked scenarios; the holdout randomises
   over the year and is harsher. The reference itself moves from 0.804 → 2.411
   worst reversals/h and 6.429 → 9.643 worst ops/h. All three points therefore
   violate g5b **identically** (binding quantity: DSO reversals at 2.411/h,
   insensitive to all four coordinates). Do not describe the tuned point as
   "satisfying the switching constraints on the holdout" — none of the three
   does, including the known-good reference.
3. **The optimum is box-constrained**, at the `tso_lambda` ceiling and the
   `tau_der_pcc` floor, with `dso_lambda` at 91 %. Part of the tight top-10
   log-spread is boundary clipping rather than identifiability. R² and Spearman
   are not explainable that way, so the identifiability conclusion holds — but
   the optimum's *location* is set by the box.
4. **Quasi-static only.** `tuning/` never passes a `plant_factory`, so nothing
   here is validated against PowerFactory RMS dynamics — most consequential for
   the OLTC behaviour, given
   `docs/daily_log/07_2026/2026-07-30_rms_oltc_taps_never_fire_midrun.md`.
   An RMS replay of the tuned point is a required follow-up.
5. **`dso_v_priority` is not identified** (Spearman −0.149, p = 0.113). Three of
   four coordinates are; this one is not resolved by 180 trials.
6. **Objective values are not comparable across the campaign.** `MetricScales`
   was recalibrated mid-campaign, so `v5_reparam` values (~220–240) and
   `v5_reparam_v2` values (~0.57–1.18) are different objectives. Never plot them
   on one axis.
7. **`g4_settling` is inactive by design** (limit above the window width) and
   `g3_contraction` never binds in the corrected study. Neither carries
   information; g3 is retained as a safety net.
8. **The holdout comparison is against the *preconditioned* reference, not
   production** (§3b). Production was never evaluated on the holdout. Say
   "beats the preconditioned λ=0.5 reference" for holdout figures, and
   "beats production" only for the tune-set figures in §3b. A fresh-seed holdout
   (`holdout_set_v2(seed=43, n=40)`, 4 points, ~1.3 h) would close this properly
   — the spent set used seed 42, so seed 43 is a genuinely independent draw from
   the same even-ISO-week distribution and avoids a second read.
9. **`sensitivity_update_interval = 1E6` is deliberate, by design** (confirmed
   2026-08-05). `H` is cached once and never refreshed, which is the intended
   architecture: the controllers must never see the plant, only their cached
   sensitivity model. So λ is a loop gain **with respect to the cached model**,
   not with respect to the true plant Jacobian — state it that way. This is a
   property of the control architecture being studied, not a limitation of the
   tuning; the model/plant mismatch it induces is part of what the closed-loop
   simulation is measuring. It does mean λ at the box ceiling should not be read
   as a physical loop gain of 1.1975.

---

## 7. Reproduction commands

```bash
# Stage 1 excitation gate (~12 min)
python -m tuning.scripts.audit_design_set --set tune_v2 --csv results/tuning/excitation_audit.csv

# Stage 2 metric calibration (~15 h, 65-draw Sobol prior sample)
python -m tuning.scripts.calibrate_metrics --n-draws 64 --n-scenarios 3

# Stage 3 switching calibration (~3 h)
python -m tuning.scripts.calibrate_switching --target-tso 6 --target-dso 6 --tol-rel 0.1

# Stage 4 BO (~8 h, 6 workers).  Storage MUST be on local disk: Z: is a
# ~97 %-full SMB share and it filled mid-run, killing 3 of 6 workers.
python -m tuning.scripts.run_tuning_parallel --n-trials 180 --workers 6 \
    --study-name v5_reparam_v2 --cvar-pct 100 --perf-exclude v2_undervoltage_ramp \
    --storage sqlite:///<LOCAL>/v5_reparam_v2.db \
    --output configs/tuned_params_reparam.yaml

# Stage 6 identifiability gate (minutes) — run BEFORE spending the holdout
python -m tuning.scripts.identifiability --storage sqlite:///<LOCAL>/v5_reparam_v2.db \
    --study-name v5_reparam_v2

# Stage 5 holdout (~1 h, 6 jobs) — ONCE ONLY
python -m tuning.scripts.run_holdout --bo-params configs/tuned_params_reparam.yaml \
    --analytic-lambda 0.9 --n-jobs 6
```

Environment: `F:\python_environments\qOFO_clean\python.exe` (3.12.13), GUROBI
academic licence (expires 2027-07-21). If the solver falls back to SCIP the
numbers are **not** comparable — the solver name is stamped into the output
metadata; check it.

Measured throughput (this server, contrary to the handover's "peaks at 6"):
3 workers 8.13 trials/h, 6 workers 13.79, 12 workers 26.2 — near-linear. Choose
workers by **feedback rounds** (`trials / workers`), not throughput: 180/6 = 30
rounds; 80/12 = 6.7 rounds would leave TPE proposing against a stale surrogate.
