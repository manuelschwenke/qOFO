# 2026-08-14 — Stage 0: move budget for `g_w_gen`, per-area refinement, output-weight reading

**File:** `tuning_mc/stage_0_preconditioning.py`
**Reason:** `g_w_gen` had no design rule at all (the `gen` class is in
`precondition_exclude_classes`, so the curvature rule never touches it), and the
config block compressed a per-column, per-area design onto five global scalars
with no intermediate level. Separately: the output weights `g_v` / `g_q` /
`dso_g_v` needed a defensible argument, not a measurement.

## What was verified first (this drives everything below)

`optimisation.miqp_solver.build_miqp_problem` builds

    min_w   w^T G_w w + grad_f^T w + z^T G_z z

with `G_w` diagonal and `grad_f = 2 A^T ytil` (`tso_controller.py:1563`,
`grad_f += 2 g_v (V - V_set) dV/du`). The **tracking term is linear in `w`** —
there is no `||H w||^2` — so the continuous columns *decouple exactly*. The
runner pins `alpha = 1.0` and `g_u = 0` for every TSO and DSO controller
(`multi_tso_dso.py:1087`, `:1415`, `:1644`), so the unconstrained interior step
of continuous column `i` is

    du_i = -(a_i^T ytil) / g_w_i                                          (S)

exactly, not to first order. Bounds and output constraints can only shrink it.

## 1 — Move budget (designs `g_w_gen`)

New per-column block in `_analyse_controller`, reported in its own table.
Projection of a reference error onto column `i`, three shapes:

| shape  | pattern                                   | `|a^T ytil|`              |
|--------|-------------------------------------------|---------------------------|
| `1bus` | only the strongest bus off by `d`         | `g_v max_k|h_ki| d`       |
| `sys`  | every bus off by the same `+d`            | `g_v |sum_k h_ki| d`      |
| `box`  | `|e_k| <= d`, adversarial signs           | `g_v sum_k|h_ki| d`       |

`box` is the max over the whole box `||e||_inf <= d`, hence the l1 norm, hence
the only reading that is a guarantee. Inverting (S):

    g_w_i >= g_v (sum_k |h_ki|) d_ref / du_max                            (S')

CLI: `--max-move-gen-pu` (default 1e-3), `--max-move-q-mvar` (default None —
Mvar-unit classes keep the curvature rule as their design authority and the
budget is diagnostic there), `--dref-pu` (0.02), `--dref-q-mvar` (5.0).
Aggregation for the config scalar is `max`, not geometric mean: it is a bound,
so the binding column fixes it.

**Result at `baseline_ieee39_thevenin.yaml`, `v2_quiet_spring`:** designed
`g_w_gen = 9.617e8` against a shipped `1e9`. The binding column is TSO-z2's AVR.
`d@limit` (the per-bus deviation at which the shipped weight first permits a
full 0.001 pu step) is 2.08 % for that column and 3.4–5.8 % elsewhere — i.e.
outside the operating corridor. The shipped `1e9` is now derived rather than
asserted.

Note `move@sys == move@box` for every continuous TSO column: those `h_ki` are
single-signed (Q injection raises voltage everywhere it reaches), so the
adversarial bound costs nothing over the systematic one. This does **not** hold
for the OLTC columns, which oppose part of their own zone.

## 2 — Per-area refinement (`--per-area`)

Adds the level between "per column" and "one config scalar": the same design
aggregated per TSO zone and per DSO area, plus the single-factor re-gain that
the shipped config can actually apply. The factor is the log-least-squares
optimum, `geomean_cols(designed_col / global_class_scalar)`, which is exactly
what `zone_g_w_scale` realises (it multiplies one zone's whole `g_w` vector).
The residual spread — what a single factor per area cannot absorb — is printed
next to it.

**Result:** `zone_g_w_scale={1: 0.6091, 2: 1.1255, 3: 0.8005}` with residuals
11.8x / 16.8x / 6.1x. The residuals say the TSO zones are *not* well described
by a single per-zone factor (zone 1 wants `g_w_der` at 0.16x global but
`g_w_tso_oltc` at 1.70x). DSO areas are far better behaved (factors 0.94–1.08,
residuals 1.6–2.2x), but `MultiTSOConfig` has no `dso_g_w_scale` hook, so those
numbers are informational only.

## 3 — Output weights `g_v` / `g_q` / `dso_g_v`

Not set by either rule; reported in three parts:

1. **Gauge.** `a_i^T ytil` is linear in `g_y`, so `(g_y, g_w) -> (c g_y, c g_w)`
   leaves (S), (T) and `lambda_max(M)` unchanged. The absolute level of `g_v` is
   not a tuned quantity. Only `g_y/g_w` (loop gain, set by the curvature rule)
   and the ratios *between output blocks* are observable. The one legitimate
   criterion for the gauge is solver conditioning — shipped `G_w` diagonal spans
   `[20, 1e9]`, 5e7x, geometric centre 1133.
2. **Trade-off.** Read the block ratio as an inverse-square-tolerance (Bryson)
   pair `g_block = 1/sigma_block^2`, so `g_q/g_v = (sigma_V/sigma_Q)^2`. The
   report fixes `sigma_V` at `--vtol-pu` and prints the `sigma_Q` the *shipped*
   weights imply. At `g_q=250`, `dso_g_v=1e5`: 1 Mvar of interface-Q error is
   worth 5.00 % on one bus / 1.58 % on all ten, i.e. `sigma_Q = 0.2 Mvar` at
   `sigma_V = 1 %`.
3. **Realised balance.** `E_block = sum_{k in block} g_y_k ||H[k,:]||^2`, the row
   analogue of `||a_i||^2`. DSO: interface-Q carries 93.8–94.1 % of the
   objective curvature over all columns, 99.8–100 % over the preconditioned
   columns. This is where the "~500x priority" claim in the module docstring
   should come from.

## 4 — Gauge normalisation + per-area weights wired to the MIQP (same day, later)

### Files touched
* `configs/config.py` — new `zone_g_w_class` / `dso_g_w_class`
* `experiments/runners/multi_tso_dso.py` — `_apply_class_g_w`, applied before
  `zone_g_w_scale`; plus a cache fix on `zone_g_w_scale` itself
* `tuning/_io.py` — `zone_g_w_class` added to `_INT_KEY_FIELDS`
* `experiments/run_multi_system_ofo.py` — `GAUGE`, `_gauged`, `_gauged_area`,
  `make_config_per_area()`
* `tuning_mc/stage_0_preconditioning.py` — `--from-runner`, λ/τ resolved from
  the config, paste-ready `zone_g_w_class` / `dso_g_w_class` output

### Latent bug found and fixed
`build_miqp_problem` takes `g_w_vector` **in preference to** `g_w` whenever it
is non-None, and `_get_per_variable_weights()` returns `_g_w_vector_cache`,
which `initialise()` materialises from `params.g_w` **once**. Any post-
construction write to `params.g_w` alone — which is all `zone_g_w_scale` did —
is therefore silently ignored on a controller that has a DER mapping. Measured:
at this operating point `_get_der_mapping()` returns `None` for all seven
controllers, so `zone_g_w_scale` happened to work; the defect was latent, not
active. Both paths now write the cache too, mirroring
`apply_preconditioned_g_w`.

### Gauge normalisation
`tuning/reparam.py` already establishes the exact-scaling group
`{g_v, tso_g_q_pcc, g_q, dso_g_v} ∪ {g_w_*} ∪ {g_z_*} ∪ {shunt_int_g_w}` under a
**common** factor (trajectory reproduced to ~4e-10 including the integer tap
sequence). Consequence, and the answer to "normalise g_v, g_q and dso_g_v":
they **cannot** be normalised independently — their ratios are identifiable and
must be preserved; only the one common factor is free. A per-layer rescale is
*not* a gauge transformation.

Convention adopted: fix the factor so `π_v_ts = g_v σ_v_ts² = 1` with
`σ_v_ts = 0.005 pu` from `reparam.PriorityScales`, i.e. `g_v := 1/σ_v_ts²`.
Against the hand-tuned `g_v = 1e7` that is `GAUGE = 4e-3`:

| | before | after | priority π = g σ² |
|---|---|---|---|
| `g_v` | 1e7 | 4e4 | 1.0 (the unit) |
| `g_q` | 250 | 1.0 | 25.0 |
| `dso_g_v` | 1e5 | 400 | 0.04 |

so "interface-Q is priced 625× the DSO's own voltage schedule" — the defensible
form of the ~500× claim. `G_w` also moves from `[13, 1e9]` (spread ~1e8,
entirely above 1) to `[0.0028, 3.8e6]`, geometric centre 5.4 — the only
legitimate criterion for the gauge itself is exactly this conditioning.

Applied programmatically via `_gauged(**weights)` so the un-normalised literals
stay visible at the call site and the invariance cannot drift.

### λ / τ were being ignored
Stage 0 defaulted to `λ = 0.9`, `τ = 1.0` regardless of the config. Against
`make_config_tuned` (`λ_tso = 0.5012`, `class_scales {der: 0.13225,
pcc: 7.5617}` = `τ = 0.017484`) that designs a *different* operating point — the
PCC/DER weight ratio alone moves by ~57×. Both now resolve from the config's
`precondition_*` fields unless given explicitly, with a warning if
`class_scales` has a geometric mean ≠ 1 (i.e. is smuggling gain into the shape).

### Verification
1. **Per-area routing** — for all 7 controllers and all 19 (area, class) pairs,
   the vector `build_miqp_problem` will consume equals the configured value
   exactly (`rtol 1e-12`).
2. **Gauge** — `g_w` ratio between un-normalised and normalised configs is
   `4e-3` on every column of every controller, deviation `0.000e+00`. (The
   trajectory-level claim is reparam.py's, not re-measured here.)
3. **Fixed point** — running stage 0 against the finished
   `make_config_per_area()` reproduces it: `designed == shipped` for every field
   and every area, `d@limit = 2.00 %` on every AVR column.

### Residuals — why per-class and not just per-area
| area | single-factor | residual |
|---|---|---|
| TSO-z1 | 0.743 | 5.3× |
| TSO-z2 | 0.843 | 5.5× |
| TSO-z3 | 1.132 | 6.1× |
| DSO_1–4 | 0.94–1.08 | 1.6–2.2× |

The TSO residuals are the justification for `zone_g_w_class`: zone 1 wants
`tso_oltc` at 1.70× global while `der` sits at 0.43×, which no single factor
absorbs. The DSO areas would have been adequately served by a scalar — but no
`dso_g_w_scale` exists, so the class map is the only hook there anyway.

## 5 — Report readability (same day, later)

Prompted by "I don't see the values for `g_w_pcc` when run per area". The value
was not missing: **TSO zone 1 owns no PCC interface trafo**, so it has no `pcc`
actuator class and nothing to design. The long-format table simply omitted the
row, which is indistinguishable from a bug.

* The per-area section is now a **pivot** (areas down, classes across) with an
  explicit `-` for a class the area does not own, plus a generated legend
  naming every such cell (`TSO-z1/g_w_pcc`). Footer rows: `global`, `shipped`,
  and `designed/shipped`, so the fixed-point check is visible at a glance.
* TSO and DSO get separate pivots — their class sets are disjoint, so one
  combined table would be half empty.
* Section 1 gained the same explanation, plus a note that `gen` never appears
  there because it is in `precondition_exclude_classes` and is designed by the
  move budget (section 2) instead.
* Move-budget table: classes with >3 columns collapse onto their **binding**
  (largest `move@box`) column — the one the design is actually set by — with
  the range of the rest appended. 68 rows → 18. `--all-columns` restores the
  full listing.
* Sections numbered 1–6; paste-ready `zone_g_w_class` / `dso_g_w_class` now
  print one area per line instead of one long line.
* The "shipped g_w too LOW" flag gained a 1e-3 relative tolerance: a config
  derived from this rule sits exactly on the boundary and a 4-significant-digit
  round-trip through the config file lands a hair either side of it.

## 6 — Thesis §9.3 redrafted for the new tuning concept (same day, later)

**File:** `latex_diss_ms/Chapters/Chapter09.tex`, §9.3
`\section{Controller Weights: How Fast Each Loop Converges}`
(`ch:timescales:weights`). +147 / −44 lines.

The section described an offline Bayesian optimisation over the raw weights.
Replaced by the two-part construction this work implements: an analytic rule
that *generates* the weights, plus a small deterministic search over the rule's
inputs. Three arguments the old text could not make, all established in code
above: the exact redundant scaling direction; the per-area weight count; and the
absence of an identifiability step in the old search.

Structure of the new section: why the raw weights are the wrong decision space →
the contraction coordinate (`e_{k+1} = (I − M)e_k`, λ_max(M) ∈ (0,2), ceiling
1.5 = 25 % margin) → λ_TS by sweep against a *measured* gain, λ_DS on the
isolated subordinate loop against `N_inner` → probe-then-pattern-search over the
coordinate table → what is and is not claimed.

Two framing decisions worth recording:

* **The section does not contradict §1.5** (`ch1:sec:scope:empirical`,
  "Empirical Stability Evaluation instead of Formal Guarantees"). λ is presented
  as a *design coordinate on the controller's cached model*, explicitly not a
  certificate; the acceptance test is a measurement and the evidence of
  contraction remains ρ̂ over the Monte-Carlo ensembles. The 25 % margin is
  named as an engineering allowance against the model gap, not a bound derived
  from it. Without this framing the section would silently reopen the
  analytical route that Chapters 2, 6, 8, 10 and 12 all cite §1.5 for closing.
* **No sub-headings.** Commits `60e61ef` ("collapse four paragraph headings into
  three paragraphs") and `7bb9fbc` ("dissolve Ch 9 §…") show the author removing
  sub-structure from this chapter; the three subsections drafted first were
  converted to phantom labels + paragraphs.

Tables: `tab:param:weights:coords` (new — the searched coordinates, with a "live"
column for the identifiability probe) and `tab:param:bo:weights` (kept label,
reframed from "selected by the search" to "emitted by the rule", with the
per-area sets deferred to the appendix). All numbers `[TBD]` per the chapter's
guard (4). λ_TS ≈ 0.15 is recorded in a LaTeX comment only, not in the text.

Verified: clean `latexmk -lualatex` build, exit 0, **0 errors, 0 undefined
references** in the final pass, 240 pages. Section renders on pp. 138–141.

### Follow-up the redraft creates (not done)

* `Appendices/AppendixD_BO_Hyperparams.tex` (`ch:app_bo`) still describes the
  withdrawn method across 8 sections, and §9.3 now points at it for the probe
  design, step rule and per-area weight tables. It needs the same pass.
* `Chapter10.tex:184` calls the BO workflow "the principled route".
* `Glossary.tex:34` `\newabbreviation{BO}{BO}{Bayesian optimisation}` and the
  `ch:*:bo` / `tab:param:bo:*` labels are now misnomers, kept as redirects.
* `ClassicThesis.tex:316` include comment names the appendix by the old title.

## Open / risks

* **`make_config_per_area()` is a substantive re-gain, not a refinement.**
  `make_config_tuned` carries BO-tuned `precondition_*` fields but runs with
  `precondition_g_w=False`, so the tuned DER/PCC shape (~57×) is currently
  **inert** and the run uses `g_w_der=13 / g_w_pcc=20`, ratio 1.5. Writing the
  design into `zone_g_w_class` applies that shape statically: DER gains ~5–19×
  authority, PCC loses ~7–10×. Intended, but it must be validated by a run, not
  assumed. `--tau 1.0` regenerates with the pure analytic shape instead.
* `main()` still calls `make_config_tuned()`. Switching is a one-line change,
  deliberately not made.
* `g_z_voltage` is shared between the TSO and DSO layers (`multi_tso_dso.py`
  lines 1071 / 1405). Harmless under a common-factor gauge, but it means a
  per-layer rescale is not expressible without `zone_g_z_voltage`. Separately,
  the `g_z` warmup restore path (~line 2773) reads `config.g_z_voltage`
  directly and ignores `zone_g_z_voltage` — latent, inactive while
  `g_z_warmup_s = 0`.
* The per-area block is a function of `H`. Regenerate after any change to
  `tie_boundary_equivalent`, the zone partition, `local_sensitivities_*` or
  `start_time`.

* The integer section's commit condition (T) carries a `||a_i||^2` self-cost,
  which follows from modelling the objective as `||ytil - A w||^2`. The MIQP
  linearises the tracking term and has no such quadratic — its exact condition
  is `2|a_i^T ytil| > g_w_i`. The two sections are therefore not using the same
  objective model, and `g_w_for_engage_uniform` is low by `||a_i||^2` relative
  to the MIQP's own test. For the TSO OLTCs the two terms are the same order
  (~1e4), so this is not negligible. **Not changed** — it alters what every
  existing number in that section means. Flagged for a decision.
* `d_ref_pu` is a stated design input, not derived. Every designed weight is
  linear in it.
* All numbers inherit the cached `H` and the operating point
  (`v2_quiet_spring`, `rural_700`, 2016-01-05 08:00).
