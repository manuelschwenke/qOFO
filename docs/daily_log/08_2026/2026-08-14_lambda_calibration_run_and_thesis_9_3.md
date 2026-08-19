# 2026-08-14 — the calibration run end to end: λ_TS, λ_DS, and §9.3 filled

Third log of the day. The first two
(`2026-08-14_stage0_move_budget_and_per_area.md`,
`2026-08-14_tuning_mc_stage1_lambda_calibration.md`) built the machinery and
took a first pass at λ. **This one runs the procedure as a procedure** — Stage 0,
the analytic λ curve, λ_TS against a measured criterion, λ_DS in its own
experiment at the calibrated λ_TS — and writes the result into the thesis.

Two numbers from the earlier passes change. Both changes are explained below and
neither is a correction of an arithmetic error; they follow from running the
steps in the right order and from adding one grid point.

---

## 0 — Headline

| finding | number |
|---|---|
| **λ_TS\*** | **0.20** (was 0.15) at the declared ceiling `rho ≤ 1.5` |
| realised contraction there | `rho_emp_p95 = 1.494` |
| measured relation | `rho = 1.1702 + 1.6172 · λ_TS`, max resid 0.0066 |
| **λ_DS\*** | **1.0** (was 1.2), `g_w_dso_der = 987` |
| analytic relation (cached model, §9) | `rho = 1.1101 + 1.3221 · λ_TS`, resid 0.0015 |
| cached vs measured | intercept **5 %** low, slope **22 %** low |
| analytic TSO floor (coordinator blocks, designed weights) | 1.1137 = **95 % of the measured intercept** |
| analytic DSO floor | 0.31–0.33 — **λ_DS below ~0.35 is inert** |
| inter-zone coupling | **identically 0** (`local_sensitivities_tso=True` zeroes the off-diagonal blocks) |
| designed vs previously shipped weights | **0.54–1.23× on every class** |

**Verdict, unchanged in kind from the earlier pass but much tighter in degree:**
the analytic rule *reproduces* the parametrisation already in service. At λ = 0.9
the earlier pass reported ratios of 0.12–0.15× on the TSO block, which read as a
disagreement; at the calibrated λ_TS = 0.20 the same rule lands within a factor
of two of every shipped value and within 4 % on `g_w_gen`. **The earlier
"disagreement" was an artefact of comparing at an uncalibrated λ.**

---

## 1 — Why λ_TS moved 0.15 → 0.20

Two reasons, both procedural.

1. **The grid was missing the point that matters.** The old grid was
   `0.9, 0.6, 0.4, 0.25, 0.15, 0.10`. The affine fit put the criterion boundary
   at λ = 0.204, which sits in the 0.15–0.25 gap, so the rule "largest *measured*
   point meeting the criterion" had to fall back to 0.15 and gave away 2 % of
   f_ts for nothing. Adding λ = 0.20 measures it: `rho = 1.4940 ≤ 1.5`, feasible.
2. **The old sweep was the joint one.** `--phase lam` moves λ_tso and λ_dso
   together, so its f_ts / f_q columns are not attributable to either layer. The
   run below is `--phase scan --scan-knob lambda_tso` with λ_dso pinned at 0.9.

The pinning also buys a **check on the layer decoupling, in the direction the
earlier log could not test**. The joint sweep gave `rho = 1.1700 + 1.6173 λ`;
this sweep, with λ_dso held constant instead of tracking λ_tso, gives
`rho = 1.1702 + 1.6172 λ`. Same relation to four decimals from a different
experiment — `rho_emp_p95` is a function of λ_tso alone.

### The sweep (5 windows × 90 min, rural_700, λ_dso = 0.9 held)

| λ_TS | g_w_der | g_w_pcc | rho | f_ts | f_q | feasible |
|---|---|---|---|---|---|---|
| 0.10 | 21.6 | 82.2 | 1.3354 | 2.1349 | 0.0966 | yes |
| 0.15 | 14.4 | 54.8 | 1.4146 | 2.0823 | 0.0832 | yes |
| **0.20** | **10.8** | **41.1** | **1.4940** | **2.0444** | **0.0811** | **yes** |
| 0.25 | 8.7 | 32.9 | 1.5736 | 2.0185 | 0.0923 | no |
| 0.40 | 5.4 | 20.5 | 1.8131 | 1.9673 | 0.1252 | no |
| 0.60 | 3.6 | 13.7 | 2.1338 | 1.9443 | 0.1877 | no |
| 0.90 | 2.4 | 9.0 | 2.6312 | 1.9135 | 0.2301 | no |

λ_TS = 0.20 is best on **both** criteria among the admissible rows, so the
contraction criterion and the objective do not conflict at the boundary — worth
recording, because they easily could have.

**Ceiling sensitivity** (the margin, not the plant, sets λ_TS):
`rho ≤ 1.5 → 0.204`, `1.6 → 0.266`, `1.7 → 0.328`, `1.8 → 0.389`. Going to a
10 % margin would buy ~4 % of f_ts. The 25 % margin is kept; the price is now
stated in the thesis rather than left implicit.

---

## 2 — New: the analytic λ curve (`tuning_mc/stage_0_lambda_curve.py`)

The measured intercept 1.17 was previously attributed to "the integer columns".
That is checkable directly and was not being checked, so this module does it:
sweep the design coordinate through Stage 0 and record, per control area, the
target against `lambda_max(M)` over the **full** column set, plus the floor the
non-preconditioned columns produce on their own. No simulation; ~60 s per point.

| loop | floor | λ_full at target 0.05 / 0.2 / 0.9 |
|---|---|---|
| TSO-z1 | **0.551** | 0.573 / 0.656 / 1.253 |
| TSO-z2 | 0.167 | 0.179 / 0.266 / 0.944 |
| TSO-z3 | 0.407 | 0.439 / 0.550 / 1.189 |
| DSO_1–4 | **0.379–0.401** | 0.384 / 0.385 / 0.903 |

Three results:

* A first reading gave a TSO floor of **0.55** against the measured 1.17 and I
  concluded the taps explain only half the intercept. **That conclusion was
  wrong and is retracted in §9** — 0.55 is a per-*controller* quantity, and
  `rho_emp_p95` is computed from the coordinator's per-*zone* blocks, which are
  a different matrix. The correct comparison is in §9.
* **The DSO curve is exactly flat below λ_dso ≈ 0.4** and equals its floor there:
  the tap columns decide the worst mode and the continuous weights cannot touch
  it, even though the rule keeps emitting different `g_w_dso_der` (6583 at 0.15
  vs 3291 at 0.3). **λ_DS below 0.4 is structurally inert on the contraction**,
  which bounds the coordinate from below before any simulation.
* Above ~0.5 the DSO realises its target to within 0.5 % (0.6 → 0.603,
  0.9 → 0.903). The subordinate rule does what it says; the supervisory one does
  not, and now we know by how much and why.

**Defect found and fixed on the way:** `precondition_g_w` returns
`lambda_floor = 0.0` by construction whenever `lambda_scope='preconditioned'`
("the fixed columns are out of scope entirely"), which is the scope Stage 0 uses.
Reading that field as a floor would have reported 0.0 for every loop. Stage 0 now
computes it itself (`_fixed_column_floor`: send the preconditioned weights to
infinity, take `lambda_max`), and the shared module is untouched.

---

## 3 — λ_DS, its own experiment at λ_TS = 0.20

`--phase scan --scan-knob lambda_dso --fix lambda_tso=0.2`, grid widened around
the suspected optimum (1.0 and 1.4 added).

| λ_DS | g_w_dso_der | rho(TSO) | f_ts | f_q |
|---|---|---|---|---|
| 0.15 | 6583 | 1.4940 | 2.0690 | 0.1546 |
| 0.30 | 3291 | 1.4940 | 2.0545 | 0.1097 |
| 0.60 | 1646 | 1.4940 | 2.0467 | 0.0899 |
| 0.90 | 1097 | 1.4940 | 2.0444 | 0.0811 |
| **1.00** | **987** | 1.4940 | 2.0483 | **0.0772** |
| 1.20 | 823 | 1.4940 | 2.0472 | 0.0833 |
| 1.40 | 705 | 1.4940 | **2.0424** | 0.0840 |
| 1.60 | 617 | 1.4940 | 2.0446 | 0.1019 |

* **`rho(TSO)` is exactly constant to all reported digits over an 11× move in the
  DSO weights.** Second independent confirmation of decoupling, now at a finer
  grid and a different λ_TS. This is what licenses fixing the two coordinates in
  sequence rather than jointly.
* f_q has a genuine interior minimum at **λ_DS = 1.0**; f_ts spans 1.3 % over the
  whole range while f_q spans a factor of two. λ_DS is identifiable from the
  subordinate criterion and nearly invisible to the supervisory one — the filter
  is what keeps it visible, a weighted scalar would have declared it dead.
* The two criteria disagree (best f_ts at 1.4, best f_q at 1.0) but the f_ts gap
  between them is 0.29 % against 8.8 % on f_q. **1.0 is selected**: it minimises
  the criterion the subordinate layer is responsible for, in the region where
  f_ts is flat.

### Why λ_DS moved 1.2 → 1.0

The earlier scan held λ_tso = 0.15. At λ_tso = 0.20 the f_q optimum moves to 1.0.
So the *objective* couples the layers weakly even though the *contraction
diagnostic* does not — worth keeping in mind before quoting either optimum
without its λ_TS.

Consequence for the "essentially exact agreement" claim in the previous log:
`g_w_dso_der` designed is now **987 against the shipped 800 (+23 %)**, not 823
(+2.9 %). The earlier 2.9 % was measured at an uncalibrated λ_TS and should not
be quoted.

---

## 4 — The analytical baseline

Stage 0 at (λ_TS = 0.20, λ_DS = 1.0, τ = 1, engage 0.015/0.025 pu,
AVR budget 0.001 pu), `--scenario none`, `--per-area`:

| field | designed | shipped | ratio | per-area range |
|---|---|---|---|---|
| `g_w_der` | 10.82 | 20 | 0.54× | 1.74 – 56.7 |
| `g_w_pcc` | 41.08 | 60 | 0.68× | 18.6 – 88.2 |
| `g_w_gen` | 9.617e8 | 1e9 | **0.96×** | 3.44e8 – 9.62e8 |
| `g_w_dso_der` | 987.4 | 800 | 1.23× | 758 – 1488 |
| `g_w_tso_oltc` | 3783 | 5000 | 0.76× | 1178 – 8732 |
| `g_w_dso_oltc` | 183.1 | 150 | 1.22× | 128 – 224 |

Pinned, not designed: `g_v=1e7`, `g_q=250`, `dso_g_v=1e5`, `tso_g_q_pcc=0`,
`shunt_int_g_w=100`.

Artefacts: `results/tuning_mc/campaign_0814/`
(`stage0_designpoint.*`, `stage0_calibrated.*`, `lambda_curve.*`,
`scan_lambda_tso.log`, `scan_lambda_dso.log`), and
`results/tuning_mc/stage1/scan_lambda_{tso,dso}.json`.

---

## 5 — Reproducibility defect: the design cache had no version

**Found:** the λ = 0.9 row of the first λ_TS sweep reused a Stage-0 design cached
at 13:30, before the day's Stage-0 fixes. Same knobs, different rule, 1.8–6 % off
on every applied field, and **nothing in the output said so**. One row of a
calibration curve was produced by a different design rule from the other six.

**Fixed, in two parts:**

* `stage0_fingerprint()` — sha256 of the Stage-0 source, stamped into every
  design JSON. A cached design whose stamp does not match is set aside (renamed,
  never deleted — it is the only record of what an already-cached evaluation ran
  with) and regenerated. ~60 s, and usually emits the identical block, since most
  Stage-0 edits are reporting rather than rule.
* `_design_is_current()` — before reusing a cached *evaluation*, re-derive the
  design and compare it against the `weights` that evaluation recorded. On
  mismatch the result is renamed `.superseded` and the candidate re-runs. This is
  exact rather than a timestamp heuristic, because every result already records
  the weights it ran with.

The λ = 0.9 row is still the stale one in the table of §1 above (the re-run was
not worth 17 minutes for a row that is infeasible at `rho = 2.63` under any
reading, and its `rho` reproduces the joint sweep exactly). **It is the one
number in this log not produced by the current rule.**

---

## 6 — Other code changes

* `tuning_mc/stage_1_search.py`
  * **`import numpy as np` was missing** while `phase_lam` used `np.linalg.lstsq`
    for the affine fit — that phase would have raised `NameError` at the point
    where it reports its main result.
  * `rho_calibration()` factored out of `phase_lam` and called from `phase_scan`
    as well, so the clean single-knob λ_tso scan performs the calibration the
    confounded joint sweep used to own. It also reports λ at 1.5 / 1.6 / 1.7 /
    1.8 ceilings, because the margin is the decision.
  * `--x0` re-anchors the design point (`lambda_tso=0.2,lambda_dso=1.0`) so
    Phase A and Phase B start from the *calibrated* baseline. `X0` keeps the
    analytic values in source; the override is recorded in the run log.
  * `_launch` deduplicates candidates by knob hash. With λ_dso = 1.0 the ×2 and
    ×4 probes both clamp to 1.90, and two identical candidates raced on the same
    result file and cost a worker slot each (~18 min).
* `tuning_mc/stage_0_preconditioning.py` — `_fixed_column_floor()` and the
  `lambda_floor` field in `continuous_meta` (§2).
* `tuning_mc/stage_0_lambda_curve.py` — new (§2).

---

## 7 — Thesis §9.3 (`latex_diss_ms/Chapters/Chapter09.tex`)

Filled from the above; the section keeps its structure and its no-sub-headings
convention.

* λ_TS paragraph: the sweep grid, the disturbance the gain is measured on, the
  fitted relation as a numbered equation (`eq:param:weights:rhofit`), the
  intercept and why `rho ≤ 1` is unreachable, the 2.3× design/realised gap, and
  the ceiling sensitivity. Closes the `\todo` that asked for exactly these.
* New table `tab:param:weights:lambda`: the seven-row calibration sweep.
* λ_DS paragraph rewritten. **The old text claimed a criterion that was not
  used** — "chosen on the *isolated* DS-OFO, parent silent, against a
  capability-band-traversing step within N_inner". No such instrument exists yet
  (it is the open item in §9.1's own `\todo`, and needs its own entry point in
  `experiments/ch_9_parameter_selection/`). The new text states what was done:
  the reachability bound from the analytic floor, then interface tracking in the
  cascade at the configured N_inner, with both limits — inherited N_inner, no
  measured subordinate contraction — stated in the text.
* `tab:param:bo:weights` filled, with a per-area range column added.
* Design-bank `\todo` closed with the five windows, their excitation roles, the
  15-min stabilisation lead-in and the odd/even calendar-week parity. **One
  `\todo` deliberately left**: parity is enforced inside the tuning package and
  that does not by itself constrain the Ch. 8 Monte-Carlo sampler — someone has
  to confirm the exclusion where that ensemble is defined.
* Symbols aligned to Ch. 8: `\widehat{\rho}_{p95}`, not `\hat{\rho}`.
* Not compiled (author builds in their editor).

---

## 8 — Phase A: the identifiability probe, and a defect in its own criterion

24 candidates (6 knobs × 4 multipliers + the design point), re-anchored to
(λ_TS = 0.20, λ_DS = 1.0). Design point: `f_ts = 2.0483`, `f_q = 0.0772`.

| knob | max \|Δf_ts\| | max \|Δf_q\| | verdict |
|---|---|---|---|
| `engage_tso_pu` | **14.30 %** | 36.8 % | live |
| `lambda_tso` | 9.26 % | 160.1 % | live |
| `engage_dso_pu` | 2.00 % | **320.0 %** | live |
| `dso_g_v_ratio` | 1.61 % | 168.3 % | live |
| `tau` | 0.50 % | 60.2 % | live **via f_q only** |
| `lambda_dso` | 0.44 % | 79.5 % | live **via f_q only** |

**Defect found in `phase_a` and fixed:** liveness was decided on `f_ts` alone.
On that criterion `lambda_dso` and `tau` are dead (0.44 %, 0.50 %) and would have
been excluded from the pattern search — including **`lambda_dso`, the coordinate
§3 above had just selected, on `f_q`, the criterion the screen was ignoring.**
This is precisely the failure mode `metrics.py` uses a two-criterion filter to
avoid, re-introduced one stage earlier in the pipeline. A direction is now live
if *either* criterion responds. With the fix, all six coordinates are live and
Phase B polls the full space.

### Correction to an inference made earlier today

While the λ scans were running I noted that the tap sequence was *bit-identical*
across every candidate and every scenario (TSO and DSO, all five windows) and
inferred that taps are insensitive to the weights. **That inference was wrong,
and the probe disproves it**: `engage_tso_pu` is the single strongest direction
on `f_ts` in the whole probe.

The correct reading: `g_w_tso_oltc = 3783` and `g_w_dso_oltc = 183.1` are
*constant in every row of both λ scans* — λ generates the continuous block only,
so those sweeps never moved the tap price at all. Identical tap sequences were
the expected result, not evidence about tap sensitivity. Only the engage
coordinates move `g_w_oltc`, and when they move it the taps respond strongly.

The zero-reversal observation stands on its own and is unaffected.

## 9 — The intercept, decomposed properly (`stage_0_coupling_decomposition.py`)

§2 compared the measured intercept 1.1702 against a floor of 0.55 and concluded
the taps explain about half of it. **That was wrong.** The 0.55 comes from
`stage_0`'s per-controller `_fixed_column_floor`, built on
`objective_curvature_inputs()` (γ-attenuated, the controller's own row set).
`rho_emp_p95` is not that quantity: it is built from the *coordinator's* zone
blocks, `M_ij = G_w,i^{-½} H_ii^T Q_i H_ij G_w,j^{-½}`, with a different row set
and a different column set. Comparing them was a category error.

New module computes the coordinator's own criterion under two weight policies,
without mutating the coordinator, at **the weights the campaign actually ran**
(designed, not the baseline config's — the floor goes as `1/g_w` on the tap
columns and `g_w_tso_oltc` is 3783 designed against 5000 shipped):

| | λ=0.1 | λ=0.2 | λ=0.4 | λ=0.9 |
|---|---|---|---|---|
| analytic, worst zone | 1.2435 | 1.3742 | 1.6374 | 2.3006 |
| measured (from the fit) | 1.3319 | 1.4936 | 1.8171 | 2.6257 |
| ratio | 1.071 | 1.087 | 1.110 | 1.141 |

```
ANALYTIC : rho = 1.1101 + 1.3221 lam   max resid 0.0015
MEASURED : rho = 1.1702 + 1.6172 lam   max resid 0.0066
floor, continuous -> inf, computed directly = 1.1137
```

**The cached model is far better than §2 suggested.** It gets the intercept to
**5 %** (1.110 vs 1.170; the direct floor 1.1137 confirms the fitted intercept
is genuinely the fixed columns) and the *slope* to **22 %** low. The model gap
is in the **gain**, not the floor — the rule knows almost exactly how much
contraction the loop carries before the continuous actuators are driven, and
understates how much they then add, growing from 7 % at λ=0.1 to 14 % at λ=0.9.

**So the original attribution in the earlier log was right after all:** the
intercept *is* the columns the curvature rule never prices. The taps, the shunts
and the excluded AVR column produce 1.1137 of the measured 1.1702 on their own.

### Coupling is identically zero here, and it is a switch

`Σ_{j≠i}||M_ij||₂ = 0.0000` in every zone under both policies. Not a physical
finding: `multi_tso_dso.py:2473` sets `zero_offdiag=bool(config.local_sensitivities_tso)`
and this baseline runs `local_H tso/dso = True/True`. The supervisory
controllers are given local sensitivities only, so the cross-zone blocks are
zeroed by construction and the coordinator's criterion reduces to
`λ_max(M_ii)`. **Any explanation of the intercept that appeals to inter-zone
coupling is wrong for this configuration**, including the one I put in §9.3 and
have now removed. If `local_sensitivities_tso` is ever turned off, every number
in this section has to be re-derived.

### Knock-on correction to the DSO floor

The DSO floor quoted in §2 (0.379–0.4005) is likewise at the *baseline* tap
weight of 150, while the campaign runs the designed 183.1. A DSO's only fixed
columns are its three OLTC columns at one common weight, so `M_fixed` is exactly
proportional to `1/g_w_dso_oltc` and the floor scales exactly by 150/183.1:
**0.311 to 0.328**. The thesis quotes the corrected range. The claim that the
target is realised closely above the floor is unaffected — the continuous
columns dominate there and the tap weight barely enters.

## 10 — Phase B: converged, and it is mostly a negative result

Compass search, all six directions live, `delta0 = 0.3` decades down to `0.075`.
Converged after **5 polls with 2 accepted moves**.

| coordinate | analytic | converged |
|---|---|---|
| `lambda_tso` | 0.20 | **0.20** |
| `lambda_dso` | 1.0 | **1.4125** |
| `tau` | 1.0 | **1.0** |
| `engage_tso_pu` | 0.015 | **0.015** |
| `engage_dso_pu` | 0.025 | **0.025** |
| `dso_g_v_ratio` | 1.0 | **0.5012** |

Emitted weights: the whole TSO block is **identical** to the analytic design
(`g_w_der` 10.823, `g_w_pcc` 41.084, `g_w_tso_oltc` 3783.1, `g_w_gen` 9.617e8).
Only `g_w_dso_der` 987.44 → **699.06** and `dso_g_v` 1e5 → **50 119** change.

`f_ts` 2.048279 → 2.038270 (**−0.49 %**), `f_q` 0.077233 → 0.060687 (**−21.4 %**),
`rho` unchanged at 1.494 (λ_TS did not move, so the margin is preserved by
construction), taps 2.007/h, reversals still 0.

**The four coordinates that did not move are the interesting half.** Everything
the analytic construction claims to *determine* — the loop gain fixed by a
declared contraction margin, plus the three coordinates that are statements in
engineering units — the search left alone. The two it moved are exactly the two
whose selection rested on an objective rather than a plant property, one of them
the subordinate gain for which no measured contraction criterion exists.

`lambda_dso` moving 1.0 → 1.41 *after* the trade-off halved shows those two are
coupled: the λ_DS scan of §3 picked 1.0 as the f_q minimum **at
`dso_g_v_ratio = 1.0`**, and 1.4 was the f_ts-best point in that same scan. So
the sequential calibration is correct at its stated conditions and not beyond
them. Filter ends with **7 non-dominated points**, `f_q` reaching 0.0587 at
`f_ts` 2.0593 — a small frontier, not a unique optimum.

### Consequence outside this chapter

Halving `dso_g_v` doubles the interface-Q/DSO-voltage priority ratio. The
"interface-Q priced 625× the DSO's own voltage schedule" figure from
`2026-08-14_stage0_move_budget_and_per_area.md` §4 becomes **~1250×** under the
tuned config. Anywhere that ratio is quoted (Ch. 6 accounting) needs to say
which config it refers to. **Not touched today.**

## 11 — Holdout: half the gain survives, and one metric is exposed as unusable

Both candidates re-evaluated on `holdout_set_mc` (3 windows, even calendar
weeks, never tuned on). Artefacts in `results/tuning_mc/campaign_0814/holdout/`.

| | f_ts | f_q | rho | rev/h | feasible |
|---|---|---|---|---|---|
| analytic baseline | 3.976613 | 0.084662 | 1.4442 | 1.338 | **False** |
| incumbent | 3.947193 | 0.083503 | 1.4442 | 0.669 | True |
| delta | **−0.74 %** | **−1.37 %** | — | — | — |
| (in-sample delta) | (−0.49 %) | (−21.4 %) | | | |

**The f_ts gain survives and the f_q gain does not.** −21.4 % in sample becomes
−1.37 % out of sample: the interface-tracking improvement was substantially a
fit to the five design windows. The TS-voltage gain is small but transfers, and
is slightly larger out of sample. Under the rule §9.3 states, the incumbent is
carried forward on the strength of the surviving improvement.

**The design bank is much easier than the confirmation set.** `f_ts` ≈ 2.04
in-sample against ≈ 3.95 on holdout, driven by `ho_gen_trip_spring` (5.0) and
`ho_undervolt_ramp_winter` (6.1). Only candidate-to-candidate comparisons
transfer between the two sets; absolute levels do not.

### g5b is not a usable constraint at 90-minute resolution

The feasibility verdict flips **entirely** on DSO tap reversals in one window,
`ho_gen_trip_spring`: 1.338/h (= 2 reversals) for the analytic baseline against
0.000 for the incumbent. Everything else is identical, including `rho` and
`taps/h`.

The limit is 1.2054/h. One reversal in a 90-min window is 0.6667/h, two is
1.3333/h. **The limit sits between the only two values the window can produce**,
so the constraint is effectively binary at this resolution — 0 or 1 reversal
passes, 2 fails, with nothing in between and a violation margin of 0.2 of a
single tap event. No claim may rest on it, and the thesis says so explicitly
rather than reporting "the search restored feasibility".

Note the irony worth keeping: **the constraint that decides the outcome on
unseen data is the one the design bank cannot exercise at all** (zero reversals
in every tune-set candidate, all campaign long). The search cannot have been
selecting for it; the improvement is incidental.

## Open / next

* **Phase B is running** from the re-anchored baseline, all six directions live,
  14 points per poll, `delta0 = 0.3` decades down to `0.075`. Expect hours.
  Whatever it returns, it does not change §1–§4: those are the analytic baseline
  the search starts *from*.
* The isolated-DSO / `N_inner` instrument still does not exist. It is now cited
  by two sections (§9.1 `\todo`, §9.3 limitation sentence) and is the single
  largest open item in this chapter.
* `AppendixD_BO_Hyperparams.tex` still describes the withdrawn Bayesian method
  across 8 sections while §9.3 points at it for the probe design and the per-area
  weight tables. Unchanged today.
* `mc_reversal_spring` produced **zero reversals** in every candidate of the
  whole campaign (~45 evaluations). `g5b` is vacuous on the tune set while
  being the constraint that decides the holdout verdict (§11) — the single most
  actionable defect found today. Fixing it means both a scenario that actually
  reverses *and* a window long enough to resolve the limit (§11: at 90 min the
  limit falls between adjacent quantisation levels).
* `taps/h` is 2.007 in every row of both λ scans — **explained in §8**: λ moves
  the continuous block only, so `g_w_*_oltc` never changed in those sweeps.
  Not a defect. But it does mean the λ calibrations carry no information about
  tap wear, and no wear claim may be drawn from them.
* Wear budget (30 taps/day) still unmeasured — needs the 12-h `wear_day_set` run.
