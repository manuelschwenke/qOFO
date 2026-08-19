# 2026-08-14 — `tuning_mc`: campaign package, λ calibration, and four corrections

Companion to `2026-08-14_stage0_move_budget_and_per_area.md` (parallel session,
same day). That log covers Stage 0's move budget, per-area refinement and
output weights. **This one covers the Stage-1 campaign package built on top of
it, the λ calibration results, and four defects found along the way.** No
overlap in files except two additive edits to `stage_0_preconditioning.py`,
noted in §6.

---

## 0 — Headline results, for anyone picking this up cold

| finding | number |
|---|---|
| realised vs design contraction (TSO) | `rho_emp_p95 = 1.1700 + 1.6173 · λ_tso`, max resid 0.0066 over λ ∈ [0.1, 0.9] |
| integer floor on `rho` | **1.17** — no continuous weight can go below it; `rho ≤ 1.0` is structurally unreachable on this plant |
| λ_dso optimum (measured, both criteria) | **1.2** → `g_w_dso_der = 823` against the shipped **800** (2.9 %) |
| λ_tso at `rho ≤ 1.5` | 0.15 → `g_w_der = 14.4`, `g_w_pcc = 54.8` against shipped 20 / 60 |
| DER reactive capability | **19 % of the profile year (815/4392 windows) has EXACTLY ZERO** |
| BO Thevenin optimum vs production | tuned 2.7909 vs **production 2.7368** — production is 2.0 % *better* |
| raw-weight pilot (50 trials, preconditioner off) | best 2.6139, **−4.49 %** vs production |

**Net verdict:** the analytic design *reproduces* the hand-tuned controller
(2.9 % on the DSO gain, 8.7–28 % on the TSO block) rather than beating it. Its
value is explanatory and validating, not an improvement in control performance.
Both searches run so far reached at most −4.5 %.

---

## 1 — New package `tuning_mc/`

| file | purpose |
|---|---|
| `scenarios_mc.py` | rural_700-only scenario set, 90-min windows, plus a 12-h wear run and a calendar-disjoint holdout |
| `metrics.py` | three-tier scoring: hard constraints (barrier) / two-criterion filter `(f_ts, f_q)` / diagnostics |
| `stage_1_search.py` | `--phase lam` (λ calibration), `--phase scan` (one knob, others fixed), `--phase a` (OAT probe), `--phase b` (compass search with filter). Candidate-level parallelism, resumable, results cached by knob hash |
| `stage_1a_excitation.py` | `--screen1`: DER reactive capability per candidate window from the profiles alone, using the plant's own capability curve |
| `configs/limits_mc.json` | explicit `ConstraintLimits`; `rho_emp_p95 = 1.5` |

**Design decision worth keeping:** the search moves **the design rule's inputs**
(λ_tso, λ_dso, τ, engage thresholds, `dso_g_v_ratio`), not the weights Stage 0
emits. Stage 0 regenerates the full weight set per candidate through its CLI +
JSON contract, so the search survives refactors of that module and every step
stays interpretable ("we moved X by a factor Y").

`metrics.py` uses a **filter**, not a weighted scalar: interface-Q may be
violated with less violation strictly better, which needs no exchange rate
between pu volts and Mvar. Hard constraints stay a barrier.

---

## 2 — λ_tso calibration (`--phase lam`)

Six points, five 90-min scenarios each, ~17 min per candidate.

| λ | `g_w_der` | `g_w_pcc` | rho | f_ts | feasible |
|---|---|---|---|---|---|
| 0.90 | 2.36 | 9.0 | 2.631 | 1.914 | no |
| 0.60 | 3.61 | 13.7 | 2.134 | 1.930 | no |
| 0.40 | 5.41 | 20.5 | 1.813 | 1.974 | no |
| 0.25 | 8.66 | 32.9 | 1.574 | 2.032 | no |
| 0.15 | 14.4 | 54.8 | 1.415 | 2.104 | yes |
| 0.10 | 21.7 | 82.2 | 1.335 | 2.171 | yes |

**`rho = 1.17 + 1.62 λ`.** The relation is *affine*, not proportional — reporting
a ratio `rho/λ` hides the floor, which is the term that decides feasibility. The
floor of 1.17 is the integer columns' own contribution, consistent with the
independently measured `lambda_floor = 1.085` from zone 1's OLTC columns alone.

**`rho ≤ 1.0` cannot be met by any continuous weight.** This retroactively
explains why the 2026-08-13 study had to re-anchor that threshold to 1.5654 —
the reason was never stated there.

### `rho` semantics — do not misread the criterion

The condition is `0 < rho < 2` (coordinator `check_contraction`); the error mode
decays by `|1 − rho|` per step. `rho ≤ 1` is the *monotone / well-damped*
preference, **not** the stability requirement. A floor of 1.17 means monotone
convergence is unreachable, not that the plant is unstable — it contracts at
0.17/step there. Only λ = 0.9 (rho 2.63, decay 1.63) is genuinely divergent.

Consequence: the λ* choice depends entirely on the margin below 2 that is
declared. `rho ≤ 1.5` → λ ≈ 0.20; `rho ≤ 1.8` (10 % margin) → λ ≈ 0.39, where
f_ts is 1.974 instead of 2.104. **Declare the margin explicitly** — most of the
"10 % tracking penalty" first reported was an artefact of picking 1.5.

---

## 3 — λ_dso scan (`--phase scan --scan-knob lambda_dso --fix lambda_tso=0.15`)

| λ_dso | `g_w_dso_der` | rho(TSO) | f_ts | f_q |
|---|---|---|---|---|
| 0.15 | 6583 | 1.4146 | 2.1040 | 0.1530 |
| 0.30 | 3292 | 1.4146 | 2.0896 | 0.1223 |
| 0.60 | 1646 | 1.4146 | 2.0853 | 0.0990 |
| 0.90 | 1097 | 1.4146 | 2.0823 | 0.0832 |
| **1.20** | **823** | 1.4146 | **2.0803** | **0.0742** |
| 1.60 | 617 | 1.4146 | 2.0821 | 0.1021 |

* `rho(TSO)` is **exactly constant** across all six rows — the layers are
  decoupled in that diagnostic, and the experiment is clean.
* λ_dso = 1.2 is optimal on **both** criteria simultaneously; f_q has a genuine
  interior minimum (worse at 0.9 and at 1.6). First coordinate in this campaign
  that is neither flat nor rail-bound.
* **`g_w_dso_der` = 823 vs the shipped 800.** The shipped value corresponds to
  λ_dso ≈ 1.23 in design coordinates, computed independently — essentially exact
  agreement from opposite directions.
* f_ts spans 1.1 % while f_q spans 106 %: **λ_dso is not identifiable from the
  TS-voltage objective.** A pure TS-voltage scalar would have declared this
  coordinate dead. The filter is what keeps it visible.

**Caveat:** the optimum sits above 1 (oscillatory, decay 0.2/step) and **no DSO
contraction diagnostic exists** — `zone_contraction_lhs` is written per TSO zone
only. Nothing backs λ_dso = 1.2 on stability grounds; it is an objective-only
choice.

---

## 4 — DER reactive capability screen (`stage_1a_excitation --screen1`)

Capability follows `VDE-AR-N-4120-v2`, which has a **hard dead zone**: below
`P/Sn = 0.1` reactive capability is exactly zero. Computed from profiles alone
using the plant's own curve (`controller.der_qv_local_loop._qv_capability`), so
the screen cannot drift from what the simulation enforces.

| window | DER P | Q range | above dead zone |
|---|---|---|---|
| **2016-01-05 08:00** (baseline's own start, N-1 argument) | 2059 MW | **2988 Mvar** | z1/z2/z3 100 %, DSO 60 % |
| 2016-04-15 12:00 | 4440 MW | 3699 Mvar | 100 % everywhere |
| 2016-07-10 12:00 | 378 MW | 710 Mvar | TSO **0 %**, DSO 40 % |
| 2016-07-10 03:00 | 199 MW | **0.0** | 0 % everywhere |
| 2016-01-21 18:00 | 5 MW | **0.0** | 0 % everywhere |

Full year at 2-h stride: **815 / 4392 windows (19 %) have exactly zero DER
reactive capability.** Every reactive-allocation knob (τ, λ_dso, the DSO
objective trade-off) is structurally inert in those — roughly a fifth of the
operating envelope is untunable for them. Summer is *not* a DER-rich condition
for the TSO layer: the TS-connected DER are few (1/1/2) and never clear the
dead zone even at midday.

**Operating-point sensitivity of the design is small**: re-deriving at the N-1
window versus the zero-capability summer night moves the weights by **1–6 %**,
because the curvature rule reads ∂V/∂Q *sensitivities*, not capability *bounds*.
Weight design is robust to the operating point; **identification is not**.

---

## 5 — Four defects found

1. **Preconditioner floor is scoped wrong.** `precondition_g_w` floors columns at
   `floor_frac × max(col_sq)` over *all* columns, including the excluded AVR
   column, which is ~1e7 against ~1 for DER/PCC (`∂V/∂V_ref ≈ 1 pu/pu` vs
   `∂V/∂Q ≈ 3e-4 pu/Mvar`). In TSO zone 2 the floor lands at 21.75 and flattens
   every continuous column to one value — **the conditioning half of the rule has
   been inert on every TSO loop in every study to date**; only κ and τ ever acted.
   DSOs are unaffected (no `gen` class). Verified: scoping the floor to the
   preconditioned columns leaves zone 1 and all DSOs bit-identical and introduces
   spread only where several columns exist. **`controller/gw_precondition.py` was
   NOT modified** — that would change the meaning of every existing study.
   `stage_0` gets `--floor-scope {preconditioned,all}` instead.
2. **`gamma_oltc_q` inconsistency.** `DSOController._build_gradient` zeroes the
   Q-tracking gradient on OLTC columns (γ = 0: "OLTCs receive no Q-tracking
   incentive"), but `objective_curvature_inputs()` returns the *un-attenuated*
   rows. Anything built on it — the curvature `M`, `lambda_floor`, the
   `integer_dominated` verdict — over-counts DSO OLTC columns. Stage 0 now
   applies γ; the shared module does not.
3. **Scenario overlay was unconditional** in `stage_0`, silently relocating the
   design point away from the baseline's own `2016-01-05 08:00`. Fixed with
   `--scenario none` plus `--start-time` / `--network`.
4. **`g_q` drift, uncommitted**: working tree has `g_q = 300` in
   `run_multi_system_ofo.py:267` against the 250 the BO gauge was pinned at.
   Any tuned `dso_g_v` is ~1.2× off until this is reverted or re-anchored.

Also: **TSO z2 col 11 and z3 col 7 are dead actuators** — all-zero response rows,
no threshold definable, the MIQP can never justify moving them. Worth
identifying which transformers those are.

---

## 6 — Edits to `stage_0_preconditioning.py` (additive only)

* `--floor-scope {preconditioned,all}` (default `preconditioned`);
* γ applied to the non-voltage rows of integer OLTC columns;
* engage-threshold reporting under three deviation shapes — single-bus spike,
  systematic offset, other-channel — because the single-bus reading badly
  misdescribes observed tap behaviour (see §7);
* `--scenario none`, `--start-time`, `--network`.

---

## 7 — Tap engage thresholds: read the shape, not just the number

The commit condition is `p_i > ‖a_i‖/2 + g_w/(2‖a_i‖)` on the **projection of the
error onto that tap's response direction**. The threshold splits into a
quantisation floor (half the tap's own step — an overshoot bound no price can
beat) and a price term.

Under a *systematic offset* — the deviation an OLTC exists to correct — the
shipped weights engage at **0.98–1.31 %** (TSO) and **1.83–2.86 %** (DSO), i.e.
already inside the operator's stated 1–2 % / 2–3 % targets. Under a *single-bus
spike* the same weights read 3.4–4.4 % and 7.4–11.1 %. An early conclusion that
DSO taps "effectively never engage" was drawn from the single-bus reading and was
wrong; DSO taps are voltage-driven (not Q-driven — see defect 2) and engage
routinely, as observed in operation.

TSO z2 cols 12/13 are the only genuine outliers (3.60 / 5.06 % systematic), with
per-transformer design weights 1186 / 1837 against 5890–8706 for z1/z3 — a 7×
spread no single scalar reconciles.

---

## Open / next

* Remaining single-knob scans: `dso_g_v_ratio` (the raw-space pilot wanted it
  **29× lower**, its single strongest signal), `tau`, both engage thresholds.
* `mc_reversal_spring` **produced zero reversals** — its whole purpose. The
  hunting criterion stays vacuous until it is strengthened and re-verified.
* `mc_undervolt_ramp_winter` contributes ~60 % of f_ts *and* is a zero-capability
  window: the aggregate that would identify the allocation knobs is governed by
  the one scenario where they are inert. Split its role (constraint case, not
  identification case).
* No DSO contraction diagnostic exists; adding one means the DSO controller
  recording its own the way the coordinator does.
* Wear budget (30 taps/day) is still unmeasured — needs the 12-h run; 90-min
  windows quantise ops/h at 0.667 and cannot resolve a daily budget.
* Decide the declared margin below `rho = 2`. It, not the plant, sets λ_tso.
