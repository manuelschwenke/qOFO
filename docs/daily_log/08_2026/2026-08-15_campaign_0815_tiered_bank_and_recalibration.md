# 2026-08-15 — campaign 0815: a tiered scenario bank, a real wear budget, and the re-calibration on top of it

Continues `2026-08-14_lambda_calibration_run_and_thesis_9_3.md`. That campaign
produced a calibrated coordinate set and, in the process, four defects in its
own experiment design. This one rebuilds the design and re-runs the calibration
on it. Everything from 0814 is left in place; this campaign writes to
`results/tuning_mc/campaign_0815/`.

Environment: `F:\python_environments\qOFO_clean\python.exe` (Python 3.12.13,
pandapower 3.4.0, numpy 2.4.6) on the 40-logical-core server. Project at
`Z:\Python_Projekte\qOFO_GH`.

---

## 0 — Phase 0: what was built, and the two places I departed from the brief

### 0.1 The wear limits (`tuning_mc/configs/limits_mc_v2.json`)

| field | v2 | v1 | basis |
|---|---|---|---|
| `tap_ops_per_h` | **1.25** | 6.0268 | 30 taps/day/transformer ÷ 24 h |
| `tap_reversals_per_h` | **0.25** | 1.2054 | 1.25 / 5, preserving v1's ratio — **a choice** |
| `rho_emp_p95` | 1.5 | 1.5 | unchanged, declared 25 % margin below the OFO bound of 2 |
| `corridor_excess_pu` | 1e-4 | 1e-4 | unchanged |
| `settling_s` | 1500 | 1500 | unchanged (deliberately inactive) |

**Provenance of the v1 pair, as asked.** It is not a wear budget and never was.
Both numbers are `ConstraintLimits.from_reference(margin=1.5)` applied to *one*
measured trajectory — the 2026-08-13 Thevenin reference, whose worst 75-min
window (`v2_gen_trip`) contained **5 tap operations and 1 reversal**:

```
tap_ops_per_h       = 1.5 x 4.01786  = 6.0268      (5 taps      / 75 min)
tap_reversals_per_h = 1.5 x 0.80357  = 1.2054      (1 reversal  / 75 min)
```

Sources: `tuning/objectives_v2.py:151` (`from_reference`),
`docs/tuning/RESULTS_bo_retuning_2026-08.md:163`,
`00_daily_log/2026-08-13_bo_thevenin_study_setup.md:163`.

Two things follow. First, **the exact 5:1 ratio is an artefact** — the same
margin applied to two counts that happened to stand in that ratio in one
window — so the provenance neither dictates nor forbids preserving it. It is
preserved, deliberately, because it is the only empirical relation available
between the two quantities and no operator statement distinguishes reversal
wear from operation wear on this plant. **Reported as a choice, not a
derivation.** Second, note that `0.80357/h` is *exactly one tap event in a
75-minute window* — the quantisation step itself. The v1 hunting limit was
1.5 quantisation steps and never had the resolution to separate a hunting
controller from a quiet one.

### 0.2 Departure 1 — the Tier-1 search cannot run under the Tier-2 budget

The brief specifies `--limits tuning_mc/configs/limits_mc_v2.json` for Phase 1.
Applied to the 90-min Tier-1 bank that setting rejects essentially every
candidate before the search starts. The 0814 campaign measured
`tap_ops_per_h = 2.007` — three taps in a 90-min window — **in every row of both
lambda scans**, which is 61 % above a 1.25/h limit. Phase B's filter rejects
infeasible candidates outright, so the pattern search would poll, find nothing
admissible, shrink `delta`, and converge without moving.

That would not be a wear finding. It is the same unit error the campaign exists
to remove, run in reverse: 1.25/h is a **daily average** over a day that is
mostly quiet, and a 90-min window built around an injected disturbance is not a
day. `ConstraintLimits.tap_ops_per_h`'s own docstring puts the event-density
inflation at roughly 19x.

So there are two limit files:

| field | `limits_mc_v2_tier1.json` | `limits_mc_v2.json` |
|---|---|---|
| `tap_ops_per_h` | 4.0 (= 6 taps/90 min, **2x** what every 0814 candidate did) | 1.25 |
| `tap_reversals_per_h` | 2.0 (= 3 reversals/90 min) | 0.25 |
| everything else | identical | identical |

Both Tier-1 values sit at least three quantisation steps above zero, so both are
*resolvable* at 90 min, unlike the pair they replace. They are a **chatter
screen**, not a budget. This is consistent with the brief's own logic — Phase 5
is described there as "the first time the 30 taps/day/transformer budget is
actually tested", which says in as many words that it is not tested before
then. Tier-1 tap statistics are recorded on every candidate and reported as
diagnostics only.

Full reasoning: `tuning_mc/configs/limits_mc_v2.README.md`.

### 0.3 Departure 2 — the per-zone lambda grid is truncated at 1.9, not 3.5

`controller/gw_precondition.py:458` raises `ValueError: lambda_target must be in
(0, 2) for OFO stability` for any target >= 2. That module may not be edited
(changing it would alter the meaning of every existing study), and the guard is
a stability check, so it is not something to bypass unattended. Verified by
running Stage 0 at `--lambda-tso-zone "1=2.0,..."`: it raises.

Phase 2's zone-1 grid is therefore `{0.2, 0.5, 1.0, 1.5, 1.9}` rather than the
`{0.2, 0.5, 1.0, 2.0, 3.5}` proposed. That is still a 9.5x span on zone 1's
gain. **If zone 1 improves monotonically all the way to 1.9, the finding is that
the guard rather than the plant is binding** — a reportable result and a
well-posed follow-up, not grounds for editing a shared stability check.

### 0.4 The banks (`tuning_mc/scenarios_mc_v2.py`)

Selected from the Screen-1 capability measurement by
`tuning_mc/select_windows_v2.py` (new), not by hand. Strata cut at the VDE
saturation plateau, read off the data at **2988.1 Mvar**:

| stratum | definition | 2016 profile year | Tier-1 design bank |
|---|---|---|---|
| `none` | `q_range_total == 0` | 18.6 % | 2/12 (17 %) |
| `partial` | `0 < q < 2988.1 Mvar` | 33.7 % | 4/12 (33 %) |
| `full` | `q >= 2988.1 Mvar` | 47.7 % | 6/12 (50 %) |

so the zero-capability stratum — where `tau`, `lambda_dso` and `dso_g_v_ratio`
are *structurally* inert — is present at its true share of the year without
dominating. Every window carries its stratum in `WINDOW_META`, and
`stage_1_search` now writes a `by_stratum` aggregate on every candidate
alongside the untouched `f_ts` / `f_q`.

**Tier 1, design — 12 x 90 min, `rural_700`, odd ISO weeks.** The five 0814
excitation roles (quiet; generator outage; load ramp up; load ramp down;
sign-reversing), each replicated across seasons. Event constructions are carried
over *unchanged* so that season and capability are the only differences from
the 0814 counterparts.

| role | season | start | wk | stratum |
|---|---|---|---|---|
| quiet | summer | 2016-06-06 02:00 | 23 | none |
| quiet | spring | 2016-03-03 10:00 | 9 | full |
| gen_trip | spring | 2016-04-16 14:00 | 15 | full |
| gen_trip | summer | 2016-08-31 14:00 | 35 | full |
| gen_trip | winter | 2016-01-04 08:00 | 1 | partial |
| ramp_up | winter | 2016-02-03 20:00 | 5 | none |
| ramp_up | autumn | 2016-11-26 10:00 | 47 | partial |
| ramp_up | spring | 2016-05-12 10:00 | 19 | full |
| ramp_down | summer | 2016-07-18 14:00 | 29 | full |
| ramp_down | spring | 2016-04-02 10:00 | 13 | partial |
| reversal | spring | 2016-03-19 10:00 | 11 | full |
| reversal | autumn | 2016-10-01 14:00 | 39 | partial |

**Tier 1, confirmation — 6 x 90 min, even ISO weeks**, drawn from *the same
cells by the same rule*, which is the property the 0814 holdout lacked (its
`f_ts` was ~2x the design bank's on the same metric).

**Tier 2, audit — 4 x 12 h**, profile-driven, two quiet and two with one
realistic event, odd ISO weeks and on calendar days disjoint from Tier 1.
Quantisation at 12 h is 0.0833 ops/h, so the 1.25/h limit sits 15 steps above
zero and the 0.25/h limit 3 steps — properly resolved, which is the whole point
of the tier.

Cost: Tier 1 is 18 simulated hours per candidate, Tier 2 is 48.

**A defect found and fixed in the selector itself.** The first pass clustered
four of twelve design windows into ISO week 9, three of them on one calendar day
at 10:00, 12:00 and 14:00 — near-identical operating points. Cause: the `full`
stratum is an exact saturation *plateau*, so hundreds of windows carry
bit-identical capability and any deterministic tie-break on timestamp collapses
the bank onto the first days of each season. The selector now picks on
representativeness first (closest quartile to the stratum median) and **maximin
temporal spread** second. The 12 windows now span 11 months on 12 distinct days.

One window worth flagging: the Tier-2 winter audit landed on **2016-12-25**. It
is a data-driven pick with an unremarkable load (4411 MW) and the profile set
carries no holiday model, so nothing about it is anomalous — but it is named
here rather than left to be discovered.

### 0.5 Code changed

* `tuning_mc/scenarios_mc_v2.py` — new. Three banks + `WINDOW_META`.
  `scenarios_mc.py` untouched, so 0814 stays reproducible.
* `tuning_mc/select_windows_v2.py` — new. Stratified selection from `screen1.json`.
* `tuning_mc/configs/limits_mc_v2.json`, `limits_mc_v2_tier1.json`,
  `limits_mc_v2.README.md` — new.
* `tuning_mc/stage_1_search.py` — `--scenario-set` gains `tier1|confirm|audit`;
  per-candidate `by_stratum` and `window_meta` (additive, cannot change any
  acceptance decision); `design_weights` split into `design_payload` +
  `design_weights`; new `zone_class_block()`; `build_config` accepts
  `zone_g_w_class`; `BOUNDS` gains `lambda_tso_z{1,2,3}` — **deliberately not in
  `X0`**, so Phase A/B are unaffected until the hypothesis pays.
* `tuning_mc/stage_0_preconditioning.py` — `--lambda-tso-zone '1=..,2=..,3=..'`,
  recorded in `targets.lambda_tso_by_zone`. Emits nothing different when the
  flag is absent, so cached 0814 designs remain weight-identical (only the
  source fingerprint changes, which the cache handles by regenerating).
* `controller/gw_precondition.py` — **not touched**, per the brief.
* Stage-0 design fingerprint / `_design_is_current` mechanism — **intact**.

Verified end to end before committing the night: all three banks build; both
limit files load; `--lambda-tso-zone "1=1.9,2=0.2,3=0.5"` produces
`zone_g_w_class = {1: {der: 0.183}, 2: {der: 56.71, pcc: 50.04},
3: {der: 4.718, pcc: 9.096}}`. Zone 1 owns **no `pcc` class at all**, which
confirms the 0814 note that its worst mode is nearly pure floor.

Only `der` and `pcc` are carried into `zone_g_w_class`. The per-area block also
holds a per-zone `g_w_tso_oltc`, and applying that as well would move the tap
price at the same time as the loop gain — a Phase-2 row would then answer "does
zone 1 do better with a different design?" instead of the question asked, "does
zone 1 do better at a higher *gain*?".

---

## 1 — Phase 1: lambda_TSO re-calibrated on the new bank

`--phase scan --scan-knob lambda_tso`, `lambda_dso = 1.0` held, 12 x 90 min
Tier-1 bank, `limits_mc_v2_tier1.json`, 7 workers.

| λ_TS | g_w_der | g_w_pcc | rho(TSO) | f_ts | f_q | taps/h | rev/h | feasible |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 21.65 | 82.17 | 1.2573 | 1.3707 | 0.0866 | 2.007 | 0.669 | yes |
| 0.15 | 14.43 | 54.78 | 1.3256 | 1.3112 | 0.0924 | 2.007 | 0.669 | yes |
| 0.20 | 10.82 | 41.08 | 1.3971 | 1.2798 | 0.0902 | 2.007 | 0.669 | yes |
| **0.25** | **8.658** | **32.87** | **1.4743** | **1.2586** | **0.0890** | 2.007 | 0.669 | **yes** |
| 0.40 | 5.411 | 20.54 | 1.7068 | 1.2267 | 0.0837 | 2.007 | 0.669 | no |
| 0.60 | 3.608 | 13.69 | 2.0181 | 1.2073 | 0.0966 | 2.677 | 0.669 | no |
| 0.90 | 2.405 | 9.13 | 2.4868 | 1.2036 | 0.1187 | 4.015 | 0.669 | no |

```
lambda* = 0.25   (largest MEASURED point meeting rho <= 1.5): rho = 1.4743
MEASURED : rho = 1.0929 + 1.5446 lam      max residual 0.0100
boundary : lam = 0.2636
ceiling  : rho<=1.5 -> 0.264   1.6 -> 0.328   1.7 -> 0.393   1.8 -> 0.458
```

**λ_TS moves 0.20 → 0.25.** Against the 0814 bank
(`rho = 1.1702 + 1.6172 λ`) the intercept falls 6.6 % and the slope 4.5 %. Both
move the boundary out, from 0.204 to 0.264. This is a bank effect, not a plant
effect — the contraction is measured over a different set of operating points —
and it is the reason the calibration had to be redone rather than carried over.

The analytic counterpart from the cached model is `1.1101 + 1.3221 λ` (0814 §9,
unchanged since `H` and the operating point did not change). Against the new
measurement the model's **intercept is now within 1.6 %** (1.1101 vs 1.0929,
slightly *high* where it used to be 5 % low) and the slope remains **14 % low**
(1.3221 vs 1.5446, better than the 22 % against the old bank). The qualitative
reading of 0814 §9 survives: the rule knows the floor and understates the gain.

λ_TS = 0.25 is best on `f_ts` among the admissible rows. On `f_q` it is *not* —
0.10 is 2.8 % better — but `f_ts` is 8.2 % better at 0.25, so the two criteria
disagree in the mild direction and the contraction rule decides, as designed.

### 1.1 The hunting constraint is no longer vacuous — the single biggest fix

`rev/h = 0.669` (one reversal in a 90-min window) in **every** candidate, at
three windows: `d_ramp_up_winter`, `d_gen_trip_winter`, `d_gen_trip_summer`.

0814's most actionable open item was that `mc_reversal_spring` produced **zero**
reversals in every candidate of the whole campaign (~45 evaluations), so `g5b`
was vacuous on the tune set while being the constraint that decided the holdout
verdict. **The new bank exercises it.**

With an irony worth recording: the two windows built specifically to reverse —
`d_reversal_spring` and `d_reversal_autumn` — produce **0.000** reversals, while
the impulsive and ramp windows produce them. The sign-reversing construction
(load on, same load off 30 min later) still does not make a tap changer reverse;
seasonal replication of the *other* roles is what did. The construction is
carried over from 0814 unchanged, so this is a clean comparison, and the
inference is that reversals here follow the disturbance's *rate*, not its sign
change.

### 1.2 Both Tier-2 wear constraints would have rejected every candidate

Measured on the Tier-1 bank: `tap_ops_per_h = 2.007` and
`tap_reversals_per_h = 0.669` in every feasible row, against the Tier-2 budget
of 1.25 and 0.25. **Every candidate in this scan is infeasible under
`limits_mc_v2.json`**, exactly as §0.2 predicted. Had the brief's literal
instruction been followed, the `feasible` column would read `False` throughout
and Phase B would have converged without moving. Feasibility here is decided by
`rho_emp_p95 <= 1.5`, which is the constraint that is meaningful at 90 min.

### 1.3 Bank sanity-check (the check the brief asked for before spending a night)

Per-window `f_ts` at λ = 0.9, sorted:

| window | stratum | f_ts | share | f_q | share |
|---|---|---|---|---|---|
| `d_ramp_up_winter` | none | 4.5080 | **31.2 %** | 0.0810 | 5.7 % |
| `d_ramp_up_autumn` | partial | 1.2875 | 8.9 % | 0.1545 | 10.8 % |
| `d_gen_trip_winter` | partial | 1.2539 | 8.7 % | 0.1284 | 9.0 % |
| `d_ramp_down_spring` | partial | 0.9041 | 6.3 % | 0.1932 | 13.6 % |
| `d_ramp_down_summer` | full | 0.8868 | 6.1 % | 0.0533 | 3.7 % |
| `d_gen_trip_summer` | full | 0.8434 | 5.8 % | 0.1828 | 12.8 % |
| `d_gen_trip_spring` | full | 0.8304 | 5.7 % | 0.0283 | 2.0 % |
| `d_reversal_spring` | full | 0.8219 | 5.7 % | 0.0347 | 2.4 % |
| `d_quiet_spring` | full | 0.8197 | 5.7 % | 0.1505 | 10.6 % |
| `d_ramp_up_spring` | full | 0.7845 | 5.4 % | 0.2063 | 14.5 % |
| `d_quiet_summer` | none | 0.7753 | 5.4 % | 0.0088 | 0.6 % |
| `d_reversal_autumn` | partial | 0.7279 | 5.0 % | 0.2024 | 14.2 % |

**Verdict: the bank passes the stated test but not comfortably.** No window
contributes *most* of the aggregate (31.2 % against 60 % on the 0814 bank), and
the remaining eleven are tightly grouped at 5.0–8.9 % against an even share of
8.3 %. But `d_ramp_up_winter` is still 3.7x the even share, which is the same
*relative* concentration the 0814 bank had (60 % against a 20 % even share).
The winter evening under-voltage ramp is simply the hardest TS-voltage case
there is, and it is a genuine 19 %-of-the-year condition. Diluting it further
would be choosing the answer.

### 1.4 A harder finding: 35 % of `f_ts` is a constant that carries no signal

Per-stratum means, across the whole λ_TS grid:

| stratum | n | f_ts at λ=0.1 → 0.9 | f_q at λ=0.1 → 0.9 |
|---|---|---|---|
| full | 6 | 0.9690 → 0.8311 | 0.0120 → 0.1093 |
| partial | 4 | 1.3377 → 1.0433 | 0.2194 → 0.1696 |
| **none** | 2 | **2.6417 → 2.6417** | **0.0449 → 0.0449** |

The `none` stratum is **bit-identical in every row**, over a 9x move in λ_TS.
Not approximately — to every digit recorded. The 0814 log established that
zero-capability windows are structurally inert for the *reactive-allocation*
coordinates (`tau`, `lambda_dso`, `dso_g_v_ratio`), because there is nothing to
allocate. This measurement extends that: they are inert for **`lambda_tso`
too**. With `g_w_gen` pinned as the gauge and the DER columns dead, the only
authority left in those windows is the tap changers, which λ does not price.

Quantified: the two `none` windows contribute a constant **0.4403** to every
`f_ts`, which is **35.0 % of the aggregate** at λ\*. Excluding them, the λ_TS
effect over the grid grows from **13.9 % to 21.9 %** — so the inert stratum
dilutes every measured effect by a factor of about 1.6.

This is not an argument for dropping them, and I have not dropped them: 19 % of
the year genuinely looks like that, and a controller's cost there is a real
cost. It is an argument that **the aggregate must never be quoted without the
stratum split**, which is now recorded on every candidate (`by_stratum`). The
"informative" aggregate over the ten `partial` + `full` windows is reported
alongside it:

| λ_TS | f_ts (all 12) | f_ts (10 informative) | f_q (10 informative) |
|---|---|---|---|
| 0.10 | 1.3707 | 1.1165 | 0.0950 |
| 0.25 | 1.2586 | 0.9820 | 0.0978 |
| 0.90 | 1.2036 | 0.9160 | 0.1334 |

### 1.5 Throughput, measured on this machine

The brief's estimate was 2.1 min wall per simulated hour per worker. Measured
here at 7 concurrent workers, 18 simulated hours per candidate:

| λ_TS | wall | min per simulated hour |
|---|---|---|
| 0.10 | 52.2 min | 2.90 |
| 0.25 | 55.6 min | 3.09 |
| 0.60 | 62.2 min | 3.46 |
| 0.90 | 73.7 min | 4.10 |

Mean **58.4 min per candidate, ~3.2 min/sim-h — about 55 % over the estimate**,
and it *rises with λ* (a higher loop gain moves the actuators more, so the MIQP
does more work). The night's budget is revised accordingly, and Phases 2 and 3
were launched **concurrently** rather than in sequence to absorb it: both depend
only on λ\*, which Phase 1 had just produced, so there was no reason to
serialise them.

**The 8-vs-16-worker scaling check was not run as a separate experiment.** With
one subprocess per *candidate*, two candidates occupy two workers regardless of
`--workers`, so the proposed 2-candidate comparison cannot measure concurrency
scaling at all. The number above is the real measurement, taken from real work
rather than a throwaway batch.

---

## 2 — Phase 2: the per-zone lambda hypothesis is rejected

Gate, as specified: scan zone 1's λ alone with zones 2 and 3 held at the
Phase-1 value; if `f_ts` does not improve meaningfully, drop the idea. Grid
`{0.1, 0.25, 0.5, 1.0, 1.5, 1.9}` — truncated at 1.9 by the
`gw_precondition` guard (§0.3) — with **0.25 as the control row**: all three
zones at λ\*, through the *same* per-area code path as the rows it controls for.

| λ z1 | zone-1 `g_w_der` | rho(TSO) | f_ts | f_q | taps/h | feasible |
|---|---|---|---|---|---|---|
| 0.10 | 3.4773 | 1.4383 | 1.26380 | 0.09646 | 3.346 | yes |
| **0.25** (control) | **1.3909** | **1.4743** | **1.26199** | **0.09658** | 3.346 | **yes** |
| 0.50 | 0.6955 | 1.5077 | 1.26154 | 0.09675 | 3.346 | no |
| 1.00 | 0.3477 | 1.5476 | 1.26410 | 0.09670 | 3.346 | no |
| 1.50 | 0.2318 | 1.5744 | 1.26536 | 0.09700 | 3.346 | no |
| 1.90 | 0.1830 | 1.5913 | 1.27952 | 0.10043 | 3.346 | no |

Zones 2 and 3 are identical in every row
(`2: {der 45.367, pcc 40.031}`, `3: {der 9.437, pcc 18.191}`), so zone 1's
weight is the only thing moving.

**The hypothesis does not pay, and the gate closes it.** Over a **19x** span of
zone 1's gain, `f_ts` moves by 0.14 % between the two *feasible* rows and by
1.4 % over the whole grid — in the wrong direction at the top. `f_q` is flat to
0.4 % and then worsens. The single best `f_ts` (1.26154 at λ z1 = 0.5) is
infeasible on contraction and beats the control by 0.04 %. **`lambda_tso` stays
a single global coordinate and is not promoted in Phase 4.**

The 0814 caution was right: a low analytic slope means those columns do not
drive the worst mode, not that raising their gain helps. Zone 1 owns **one DER
column and no PCC class at all** — confirmed directly here, its
`zone_g_w_class` entry has a `der` key and nothing else — so there is very
little for a larger step to do before it hits an actuator bound.

### 2.1 Two findings that outlast the rejected hypothesis

**(a) The per-area path is itself a worse design on this bank — by more than the
hypothesis could ever have won.** The control row and the Phase-1 global row are
the *same six knobs at the same λ*; they differ only in whether the continuous
weights are applied as global scalars or per-area blocks:

| | f_ts | f_q | taps/h |
|---|---|---|---|
| Phase 1, global scalars (λ_TS = 0.25) | 1.2586 | 0.0890 | 2.007 |
| Phase 2 control, per-area block | 1.2620 | 0.0966 | 3.346 |
| difference | **+0.27 %** | **+8.5 %** | **+67 %** |

This is why the control row was built in. Had the scan been run against the
Phase-1 baseline instead, every row would have looked worse and the cause would
have been unattributable between "per-zone λ" and "per-area vs global".

The mechanism is visible in the weights. Globally every zone gets
`g_w_der = 8.658`; per-area, zone 2 gets **45.37** — its DER is priced 5x higher,
so it moves less and hands work to the tap changers, hence +67 % on `taps/h`.
**The per-area block is a materially different design, not a re-parameterisation
of the same one**, and anything that adopts `zone_g_w_class` inherits that.

**(b) The measured contraction contradicts the analytic per-zone headroom.**
`rho_emp_p95` rises monotonically with zone 1's gain, 1.4383 → 1.5913 (+7.9 %),
and crosses the 1.5 ceiling between λ z1 = 0.25 and 0.5.

It should not. The analytic decomposition
(`stage_0_coupling_decomposition`, 0814 §9) puts zone 1 at
`rho = 0.795 + 0.124 λ`, which at λ = 1.9 is **1.03** — far below the 1.4743 that
zone 2 imposes, so the max over zones should have been pinned by zone 2 and flat
in zone 1's gain. Instead zone 1's gain moves the worst-zone figure by 8 %.

So **the "zone 1 sits ~24x inside its own contraction limit" claim is not borne
out by measurement.** Either zone 1's realised contraction grows far faster than
its analytic slope of 0.124, or `rho_emp_p95` — a p95 over a trajectory of the
coordinator's per-zone blocks — is not the max of three independent per-zone
quantities in the way the decomposition assumes. The 0814 log already flagged
that the measured/analytic derating of 1.087 had been extrapolated from the
global figure to individual zones "which has not been verified per zone". This
is the verification, and it comes out negative.

That does not invalidate the analytic decomposition as a design aid — it got the
global intercept to 1.6 % (§1) — but **the per-zone rows of that table should
not be used to justify a per-zone gain budget** until the discrepancy is
understood. Recorded as the main open question this phase leaves behind.

---

## 3 — Phase 3: lambda_DSO at the calibrated lambda_TSO

`--scan-knob lambda_dso --fix lambda_tso=0.25`, same bank and limits.

| λ_DS | g_w_dso_der | rho(TSO) | f_ts | f_q | taps/h | rev/h |
|---|---|---|---|---|---|---|
| 0.15 | 6583.0 | 1.4743 | 1.2812 | 0.1961 | 2.677 | 0.669 |
| 0.30 | 3291.5 | 1.4743 | 1.2705 | 0.1367 | 2.007 | 0.669 |
| 0.60 | 1645.7 | 1.4743 | 1.2620 | 0.1019 | 2.007 | 0.669 |
| 0.90 | 1097.2 | 1.4743 | 1.2585 | 0.0907 | 2.007 | 0.669 |
| 1.00 | 987.44 | 1.4743 | 1.2586 | 0.0890 | 2.007 | 0.669 |
| 1.20 | 822.87 | 1.4743 | 1.2572 | 0.0854 | 2.007 | 0.669 |
| 1.40 | 705.32 | 1.4743 | 1.2554 | 0.0829 | 2.007 | 0.669 |
| 1.60 | 617.15 | 1.4743 | **1.2550** | **0.0811** | 2.007 | 0.669 |

**`rho(TSO)` is exactly 1.4743 in all eight rows**, over an 11x move in
`g_w_dso_der`. The brief set this as a stop condition — "must come out *exactly*
constant; if it does not, something is wrong with the layer separation and you
should stop and say so". It does. This is the third independent confirmation
(0814 §1 and §3 were the first two), now at a third λ_TS and on a different
scenario bank.

**The scan did not identify an optimum: both criteria are still improving at the
last grid point.** Best `f_ts` and best `f_q` are both at λ_DS = 1.6, the largest
value swept, and the two criteria *agree* — unlike 0814, where `f_q` had an
interior minimum at 1.0 and `f_ts` preferred 1.4.

Marginal returns on `f_q` were shrinking (−4.0 %, −2.9 %, −2.2 % over the last
three steps) but had not turned. Reporting 1.6 as "the optimum" on that evidence
would have been reporting the edge of a box, which is precisely the defect 0814
§1 fixed for λ_TS ("the grid was missing the point that matters"). The scan was
therefore extended to the coordinate's upper bound; `BOUNDS` caps λ_DSO at 1.90
because `controller/gw_precondition.py` requires `lambda_target < 2`.

| λ_DS | g_w_dso_der | rho(TSO) | f_ts | f_q |
|---|---|---|---|---|
| 1.60 | 617.15 | 1.4743 | **1.2550** | **0.0811** |
| 1.80 | 548.58 | 1.4743 | 1.2556 | 0.0900 |
| 1.90 | 519.71 | 1.4743 | 1.2558 | 0.1043 |

**λ_DS\* = 1.6, and it is a genuine interior optimum.** Both criteria turn
there and both worsen beyond it — `f_q` by +11 % at 1.8 and +29 % at 1.9,
`f_ts` by +0.05 % and +0.06 %. The extension was worth the two candidates: on
the eight-point grid alone the honest report would have been "not identified".

`rho(TSO)` remains 1.4743 across all **ten** rows, now over a 12.7x move in
`g_w_dso_der`.

**λ_DS moves 1.0 → 1.6** against 0814. As 0814 §3 warned, the λ_DS optimum is
conditional on λ_TS and on `dso_g_v_ratio`; this one is stated at
**λ_TS = 0.25, `dso_g_v_ratio` = 1.0**, and may not be quoted without them.

Note `f_ts` spans only 0.29 % over the *whole* range while `f_q` spans a factor
of 2.4 — λ_DSO remains identifiable from the subordinate criterion and nearly
invisible to the supervisory one, which is the structural reason the
two-criterion filter exists.

Note also `taps/h = 2.677` at λ_DS = 0.15 against 2.007 everywhere else: a
DSO layer priced too cheaply to act with its continuous actuators hands the work
to its tap changers. That is the only row in either scan where a *DSO* gain
moved a tap count.

---

## 4 — Tier 2, run early: the 30-taps/day budget, measured at last

Two idle cores were available while Phase 1 ran, so the **0814 carry-over
points** were put through the new 4 x 12 h audit set under the real budget
(`limits_mc_v2.json`). This closes the last open item of the 0814 log — "wear
budget (30 taps/day) still unmeasured" — and smoke-tested the Tier-2 path six
hours before Phase 5 needed it.

| | f_ts | f_q | rho | worst ops/h | → taps/day | rev/h | feasible |
|---|---|---|---|---|---|---|---|
| 0814 analytic baseline | 1.0782 | 9.613 | 1.3812 | 1.501 | **36.0** | 0.083 | **False** |
| 0814 incumbent | 1.0767 | 8.786 | 1.3812 | 1.334 | **32.0** | 0.083 | **False** |

**Both exceed the budget, and both fail on exactly one transformer.** The only
violated constraint is `g5a_tap_ops` (+0.2507 and +0.0840); `g3_contraction`
passes at −0.1188, `g5b_tap_reversals` at −0.1666, `g2` and `g4` are inactive.
The offender is `DSO_4|trafo_10` in both cases. The next-worst transformer in
the fleet is at 20 and 18 taps/day, and nine of nineteen never move at all.

### 4.1 The hunting budget passes with margin

0.083 reversals/h is **one reversal per 12-h window**, i.e. 2/day, against a
limit of 0.25/h. Three times inside. At 12 h the quantisation step is
0.083/h, so this is the smallest non-zero value the window can report and the
limit sits 3 steps above it — the resolution the whole tier exists to provide.
Contrast 90 min, where the same limit sits *below* one single reversal.

### 4.2 The tuned point buys a 47 % reduction in DSO tap wear

Fleet-sum tap operations, summed over the four audit windows:

| | DSO fleet | TSO fleet |
|---|---|---|
| analytic baseline | 5.086 ops/h | 2.251 ops/h |
| incumbent | **2.668 ops/h** (−47.5 %) | 2.251 ops/h (identical) |

The TSO figure is *bit-identical* between the two, which is exactly right: the
two points differ only in `lambda_dso` and `dso_g_v_ratio`, both DSO-side. So
this is a clean attribution — **halving `dso_g_v` (the 0814 search's second
accepted move) roughly halves DSO tap wear**, on a measurement the 0814 campaign
never made. That is a much stronger argument for the incumbent than the −1.37 %
`f_q` it survived the 0814 holdout with.

### 4.3 A caveat that has to be stated: the verdict rests on one window

Tap operations by window type, fleet-sum:

| | two quiet windows | two evented windows |
|---|---|---|
| analytic baseline | 0.417 ops/h | 6.920 ops/h |
| incumbent | 0.333 ops/h | 4.585 ops/h |

**94 % of all tap activity occurs in the two windows carrying an injected
event**, and nearly all of that in `a_gen_trip_spring`. The two quiet 12-h
windows produce almost nothing — 0.083 ops/h on a couple of TSO transformers,
which is one tap in twelve hours.

So "36 taps/day" is properly read as *36 taps on a day containing a generator
outage*. A normal day, by this measurement, is **0–2 taps**. If the operator's
30/day budget refers to routine operation, both points are inside it by more
than an order of magnitude and the reported violation is a stress figure. If it
is meant as an envelope covering contingency days, both fail it, narrowly.

**Which of those two readings is intended is not something this campaign can
settle, and it should not be settled by choosing whichever makes the numbers
work.** It is the single most important open question the Tier-2 tier raises,
and it is put to the author rather than answered here. The audit set is 2 quiet
+ 2 evented by the brief's own specification, so the aggregate deliberately
mixes both regimes; the split above is the honest way to report it.

### 4.4 `f_q` is not comparable between tiers

`f_q` on the audit set is **9.61**, against ~0.089 on Tier 1 — a factor of ~100.
This is not a finding about the controller. `f_q` is built on `itae_q_pcc`, a
*time-weighted integral*, so it grows roughly with the square of the window
length: (12 h / 1.5 h)² = 64, and the rest is the longer window's larger
excursions. `f_ts` is an RMS-type quantity and is comparable (1.078 vs 1.259).

**Tier-2 numbers may therefore be used for the wear and hunting verdict and for
`f_ts`, but `f_q` may only be compared between candidates *within* the tier** —
where it still ranks the incumbent ahead (8.79 vs 9.61, −8.7 %).

---

## 5 — Phase 4a: the identifiability probe

Re-anchored to the calibrated point (λ_TS = 0.25, λ_DS = 1.6). Design point:
`f_ts = 1.255025`, `f_q = 0.081062`, feasible — consistent to four decimals with
the λ_DS = 1.6 row of §3, which is a useful cross-check that the re-anchoring
did what it says.

**Probe reduced to ±2x** (12 evaluations + the design point) from the
±{2x, 4x} of 0814 (24 + 1). Reason: measured throughput came in 55 % over the
brief's estimate (§1.5), the full probe was two waves at the worker cap, and
Phase B — which is what actually produces an incumbent — needed the 70 minutes
more. The risk of a smaller probe is declaring a direction dead that only
responds at 4x; the result below shows nothing landed near the threshold, so
the reduction cost nothing here. **It would not be safe to repeat on a bank
where any coordinate came out marginal.**

| knob | x | f_ts | Δf_ts | f_q | Δf_q | feasible |
|---|---|---|---|---|---|---|
| `lambda_tso` | 0.50 | 1.33313 | +6.22 % | 0.08499 | +4.84 % | yes |
| `lambda_tso` | 2.00 | 1.21323 | −3.33 % | 0.08234 | +1.58 % | **no** |
| `lambda_dso` | 0.50 | 1.25931 | +0.34 % | 0.09427 | +16.29 % | yes |
| `lambda_dso` | 2.00 | 1.25579 | +0.06 % | 0.10428 | +28.64 % | yes |
| `tau` | 0.50 | 1.25844 | +0.27 % | 0.08433 | +4.03 % | no |
| `tau` | 2.00 | 1.26310 | +0.64 % | 0.07601 | −6.24 % | yes |
| `engage_tso_pu` | 0.50 | 1.25364 | −0.11 % | 0.08484 | +4.67 % | no |
| `engage_tso_pu` | 2.00 | 1.40477 | +11.93 % | 0.07477 | −7.76 % | yes |
| `engage_dso_pu` | 0.50 | 1.27323 | +1.45 % | 0.11231 | +38.54 % | no |
| `engage_dso_pu` | 2.00 | 1.25280 | −0.18 % | 0.06776 | **−16.41 %** | yes |
| `dso_g_v_ratio` | 0.50 | 1.24946 | **−0.44 %** | 0.06885 | **−15.07 %** | yes |
| `dso_g_v_ratio` | 2.00 | 1.26621 | +0.89 % | 0.09142 | +12.77 % | no |

| knob | max abs Δf_ts | max abs Δf_q | verdict |
|---|---|---|---|
| `engage_tso_pu` | **11.93 %** | 7.76 % | live |
| `lambda_tso` | 6.22 % | 4.84 % | live |
| `engage_dso_pu` | 1.45 % | **38.54 %** | live |
| `dso_g_v_ratio` | 0.89 % | 15.07 % | live **via f_q only** |
| `tau` | 0.64 % | 6.24 % | live **via f_q only** |
| `lambda_dso` | 0.34 % | 28.64 % | live **via f_q only** |

**All six directions live. Three of the six are live via `f_q` only** — against
two in 0814. On `f_ts` alone `dso_g_v_ratio` (0.89 %), `tau` (0.64 %) and
`lambda_dso` (0.34 %) would every one be declared dead at the 1 % threshold and
excluded from the pattern search. That is **half the search space**, and it
includes `lambda_dso`, the coordinate §3 had just calibrated, and
`dso_g_v_ratio`, which the probe shows improving *both* criteria simultaneously.
The 0814 fix to `phase_a` — a direction is live if *either* criterion responds —
is load-bearing here to a degree it was not before, and must not be reverted.

Two directions improve both criteria at once and are the obvious candidates for
Phase B's first accepted move: `dso_g_v_ratio` at 0.5 (−0.44 % / −15.07 %) and
`engage_dso_pu` at 2.0 (−0.18 % / −16.41 %). Both are DSO-side, and
`dso_g_v_ratio` is the same coordinate the 0814 search moved.

Note `lambda_tso` x2 gives the largest `f_ts` gain in the probe (−3.33 %) and is
**infeasible** — the contraction ceiling is doing its job, and the calibration of
§1 is exactly the constraint that keeps the search off that point.

---

## 6 — Phase 4b: converged on one move, and the negative result is sharper than 0814's

Compass search, all six directions live, `delta0 = 0.3` decades, `delta_min`
0.15 (set to bound the poll count against the measured throughput). Converged
after **3 polls with 1 accepted move**.

| coordinate | analytic | converged |
|---|---|---|
| `lambda_tso` | 0.25 | **0.25** |
| `lambda_dso` | 1.6 | **1.6** |
| `tau` | 1.0 | **1.0** |
| `engage_tso_pu` | 0.015 | **0.015** |
| `engage_dso_pu` | 0.025 | **0.025** |
| `dso_g_v_ratio` | 1.0 | **0.50119** |

`f_ts` 1.255025 → 1.249463 (**−0.44 %**), `f_q` 0.081062 → 0.068858
(**−15.05 %**). The filter ends with **6 non-dominated points**.

**Five of six coordinates did not move — one more than in 0814.** The reading
0814 gave now holds more strongly: everything the analytic construction claims
to *determine* — the loop gain fixed by a declared contraction margin, and the
three coordinates that are statements in engineering units — the search left
alone. It moved only the one coordinate whose value rests on an objective rather
than on a plant property.

And it is the **same** coordinate the 0814 search moved, to the **same value**:
`dso_g_v_ratio` = 0.50119 here, 0.5012 there. That is one compass step of 0.3
decades down from 1.0, so the agreement is not a deep coincidence — but the two
campaigns reached it from different λ calibrations on different scenario banks,
and 0814 *also* moved `lambda_dso` (1.0 → 1.4125) where this one did not. Here
λ_DS was already calibrated to 1.6 in §3, so the search had nothing left to
recover on that axis. **The sequential calibration absorbed a move the 0814
pattern search had to make itself**, which is the behaviour the two-stage design
predicts and 0814 could not demonstrate.

The `f_q` gain (−15.05 %) is smaller than 0814's in-sample −21.4 %, and for the
same reason: it starts from a better-calibrated λ_DS, so there is less left on
the table. Whether it survives out of sample is §7 — and 0814's did not.

---

## 7 — Phase 6: the confirmation set, and the campaign's most consequential result

Filter (6 points) + incumbent evaluated on the Tier-1 confirmation set
(6 x 90 min, even ISO weeks, same cells and same selection rule as the design
bank).

### 7.1 The contraction calibration does not transfer — the incumbent is infeasible out of sample

| | worst rho | feasible |
|---|---|---|
| incumbent, design bank | 1.4743 | yes |
| incumbent, confirmation set | **1.5201** | **no** |

Per-window `rho_emp_p95` for the incumbent, both banks:

| design bank | | confirmation set | |
|---|---|---|---|
| `d_ramp_up_winter` | 1.4743 | `c_reversal_spring` | **1.5201** |
| `d_ramp_up_autumn` | 1.4713 | `c_ramp_up_winter` | 1.4980 |
| `d_ramp_down_summer` | 1.4627 | `c_gen_trip_winter` | 1.4706 |
| `d_gen_trip_spring` | 1.4541 | `c_quiet_summer` | 1.4430 |
| `d_gen_trip_winter` | 1.4517 | `c_gen_trip_spring` | 1.4418 |

**This is a defect in the calibration procedure, not in the controller.**
λ\* is selected as the largest measured λ whose *worst-window* `rho_emp_p95`
meets the 1.5 ceiling. But `rho_worst` is a **maximum over windows**, so its
in-sample value is downward-biased as an estimate of the same quantity on a
fresh draw from the same distribution — the design bank simply did not happen to
contain a window as demanding as `c_reversal_spring`.

The numbers make the size of the problem exact: λ\* = 0.25 was chosen with
`rho = 1.4743`, a **1.7 % margin** below the ceiling. The worst-window figure
moves by **3.1 %** between two banks drawn from the same distribution. **The
calibration margin is smaller than the sampling variability of the statistic it
is calibrated on**, so the procedure cannot be expected to transfer, and here it
did not.

Two remedies, neither of which this campaign can choose between on its own
evidence:

* calibrate λ\* against a ceiling reduced by an explicit allowance for that
  variability (on this measurement, ~3 % — which would move λ\* from 0.25 back
  to roughly 0.20, i.e. the 0814 value); or
* calibrate against a statistic more stable than a max over a small bank — the
  p95 over windows, or the mean of the worst two.

**Recorded as the principal methodological finding of the campaign.** It was
invisible to 0814 because its holdout was not comparable in difficulty, so a
contraction difference there could always be blamed on the bank.

### 7.2 The confirmation set is 1.96x harder — and it is one window, not the design

Aggregate `f_ts`: 1.2495 on the design bank against 2.4486 on the confirmation
set. That looks like a repeat of 0814's "the design bank is much easier than the
confirmation set" (2.04 vs 3.95). It is not the same failure, and the per-stratum
split says so:

| stratum | n (design) | f_ts design | n (confirm) | f_ts confirm | difference |
|---|---|---|---|---|---|
| `none` | 2 | 2.5918 | 2 | 2.4779 | **−4.4 %** |
| `full` | 6 | 0.8696 | 3 | 0.8421 | **−3.2 %** |
| `partial` | 4 | 1.1235 | **1** | **7.2097** | +542 % |

**Two of the three strata match to within 4.4 %.** The "same cells, same rule"
construction did what it was built to do — this is a real improvement over the
0814 holdout, where nothing matched. The aggregate is wrecked by a single
window:

| window | stratum | f_ts | share of aggregate |
|---|---|---|---|
| `c_gen_trip_winter` | partial | **7.2097** | **49.1 %** |
| `c_ramp_up_winter` | none | 4.0368 | 27.5 % |
| `c_quiet_summer` | none | 0.9190 | 6.3 % |
| `c_ramp_down_summer` | full | 0.8485 | 5.8 % |
| `c_reversal_spring` | full | 0.8426 | 5.7 % |
| `c_gen_trip_spring` | full | 0.8351 | 5.7 % |

`c_gen_trip_winter` (2016-01-27 08:00, wk 4, Q = 848 Mvar) is **6.4x the mean of
the four `partial` windows in the design bank** and alone is half the
confirmation aggregate.

**The defect is `n = 1`.** With one window in the `partial` stratum there is no
averaging, so one extreme draw becomes half the answer. The design bank has
n = 4 there and is insulated. This is a property of the six-window size the
brief specified, not of the selection rule: the rule picked a window at the
*median* capability of its cell, and capability does not predict TS-voltage cost.

**Consequence for how the confirmation is read: per stratum, never in
aggregate**, until the confirmation set carries at least two windows per
stratum. That is a concrete change to propose for the next campaign, and it is
cheap — going from 6 to 9 windows (3 per stratum) costs ~50 % more per
confirmation candidate and there are only a handful of them.

### 7.3 Survival: the mirror image of 0814

Incumbent against the **calibrated** analytic baseline (λ_TS = 0.25,
λ_DS = 1.6, `dso_g_v_ratio` = 1.0), the point Phase B started from:

| bank | f_ts base | f_ts inc | Δ | f_q base | f_q inc | Δ |
|---|---|---|---|---|---|---|
| Tier-1 design | 1.25502 | 1.24946 | **−0.44 %** | 0.08106 | 0.06886 | **−15.06 %** |
| Tier-1 confirmation | 2.44197 | 2.44864 | **+0.27 %** | 0.05957 | 0.05404 | **−9.28 %** |

**The `f_ts` gain does not survive; the `f_q` gain does.** −0.44 % in sample
becomes **+0.27 %** out of sample — the incumbent is marginally *worse* on TS
voltage on unseen windows, so the `f_ts` improvement was fit to the design bank
and, under the rule stated in §9.3, is not carried forward. The `f_q`
improvement retains **62 %** of its in-sample size (−15.06 % → −9.28 %) and *is*
carried forward.

**This is precisely the opposite of 0814**, where the TS-voltage gain survived
(−0.49 % → −0.74 %) and the interface-tracking gain did not
(−21.4 % → −1.37 %). The two campaigns agree that **only one of the two criteria
transfers, and disagree about which** — so neither result should be read as a
property of the controller. The mechanism is visible in what each search moved:
0814 moved `lambda_dso` *and* `dso_g_v_ratio`, this one moved
`dso_g_v_ratio` alone (λ_DS having been calibrated first, §3/§6). A pure
`dso_g_v_ratio` move is a re-pricing of interface-Q against the DSO's own
voltage schedule, which is exactly an `f_q` intervention; its 0.44 % `f_ts`
effect was never more than noise on a bank whose `f_ts` is 35 % constant (§1.4).

Per stratum on the confirmation set, which is the only valid read (§7.2):

| stratum | n | Δf_ts | Δf_q |
|---|---|---|---|
| `full` | 3 | −0.01 % | −1.02 % |
| `none` | 2 | +0.60 % | **−20.20 %** |
| `partial` | 1 | +0.15 % | −6.76 % |

`f_ts` is flat to within 0.6 % in every stratum — the aggregate +0.27 % is not
concentrated anywhere, it is simply absent. The `f_q` gain is real in all three
and largest in `none`, which is worth a note of caution: that is the stratum
where DER reactive capability is zero, so a 20 % `f_q` improvement there is a
change in how the *taps and the pinned AVR columns* serve the interface, not a
reallocation of DER reactive power.

### 7.4 Two findings that strengthen the incumbent independently

**(a) Infeasibility is the calibration's, not the incumbent's.** `rho` on the
confirmation set is **1.5201 for both** the baseline and the incumbent, to four
decimals. Phase B did not move `lambda_tso`, so the out-of-sample contraction
failure of §7.1 attaches to λ\* itself and says nothing about the search.

**(b) The incumbent halves tap wear out of sample, on both metrics.**

| confirmation set | baseline | incumbent |
|---|---|---|
| worst `tap_ops_per_h` | 5.353 | **2.677** (−50.0 %) |
| worst `tap_reversals_per_h` | 1.338 | **0.669** (−50.0 %) |

Exactly one halving each — 8 taps per window to 4, 2 reversals to 1. This
reproduces out of sample the 47.5 % DSO wear reduction measured independently on
the Tier-2 audit (§4.2), from a different bank, different window length and
different candidate pair. **That is the most robust result in the campaign**: the
same intervention, `dso_g_v_ratio` 1.0 → 0.5, cuts tap wear roughly in half on
every measurement made of it.

**Verdict under the author's rule.** The incumbent is carried forward — on
`f_q` (−9.28 %, survives) and on tap wear (−50 %, survives and is corroborated),
**not** on `f_ts` (+0.27 %, does not survive). And it is carried forward
*subject to* §7.1: at λ_TS = 0.25 neither point meets the contraction ceiling
out of sample, which is a question about λ\*, to be resolved before either is
quoted as feasible.

**Superseded by §8.** The wear reduction reported above is real, but §8 shows
what paid for it, and the verdict does not survive that.

---

## 8 — The DSO tap changers do too little, and the filter cannot see it

Raised by the author on inspecting the result: *the DSO OLTCs tap too
little — this could be due to the reduced `dso_g_v`*. Checked against data
already on disk, and **correct**.

`f_ds` — the DSO voltage RMS cost — is computed on every candidate by
`score_candidate` and recorded in every result file. It is classed in
`metrics.py` as a **reported diagnostic, "never optimised directly"**. Holding
every other coordinate at the calibrated point:

| `dso_g_v_ratio` | `dso_g_v` | f_ts | f_q | **f_ds (DSO V)** | DSO taps/h |
|---|---|---|---|---|---|
| 0.2512 | 25 119 | 1.25357 | **0.06590** | **0.45147** | 0.669 |
| 0.3548 | 35 481 | 1.25358 | 0.06594 | 0.45034 | 0.669 |
| **0.5012** (incumbent) | 50 119 | **1.24946** | 0.06886 | **0.43056** | 1.338 |
| 0.7079 | 70 795 | 1.25439 | 0.07486 | 0.36180 | 2.007 |
| **1.0000** (analytic) | 100 000 | 1.25502 | 0.08106 | **0.29252** | 2.007 |
| 1.9953 | 199 526 | 1.26635 | 0.09213 | **0.15385** | 5.353 |

**The search bought its `f_q` gain and its wear reduction by degrading DSO
voltage regulation by 47 %** (0.2925 → 0.4306 going from the analytic value to
the incumbent). DSO nodal voltage is a **stated controlled output** of the
subordinate layer, and it is not a filter criterion — so the search traded it
away without anything in the procedure objecting.

The "47 % reduction in DSO tap wear" of §4.2 and the "halved tap wear" of §7.4
are therefore **not free**. They are the same fact as this one, reported from
the side that flatters it: the DSO taps less because it has been told to care
less about its own voltages.

### 8.1 The mechanism, and a redundancy between two coordinates

`engage_dso_pu` moves the same trade-off, through the tap price rather than the
objective weight:

| `engage_dso_pu` | `g_w_dso_oltc` | f_ts | f_q | **f_ds** | DSO taps/h |
|---|---|---|---|---|---|
| 0.0125 | 77.8 | 1.27323 | 0.11231 | **0.12476** | 5.353 |
| 0.0250 | 183.1 | 1.25502 | 0.08106 | 0.29252 | 2.007 |
| 0.0500 | 392.6 | 1.25280 | **0.06776** | **0.43124** | 1.338 |

Two observations.

**(a) `f_ds` and `f_q` are in direct opposition, mediated by DSO tap activity.**
More DSO tapping regulates DSO voltage better and interface-Q worse. That is
physically expected — the OLTC moves voltage, which moves reactive flow through
the interface — but it had never been measured on this plant, because nothing in
the procedure was watching `f_ds`.

**(b) `dso_g_v_ratio` and `engage_dso_pu` look redundant near the incumbent.**
Compare `dso_g_v_ratio = 0.5012` against `engage_dso_pu = 0.05`:

| | f_ts | f_q | f_ds | DSO taps/h |
|---|---|---|---|---|
| `dso_g_v_ratio` 0.5012 | 1.24946 | 0.06886 | 0.43056 | 1.338 |
| `engage_dso_pu` 0.05 | 1.25280 | 0.06776 | 0.43124 | 1.338 |

Two different coordinates, two different mechanisms, the same operating point to
within 0.3 % on every criterion.

**This is a local coincidence and the inference drawn from it was wrong** — see
§8.4, which measures the wider grid. In the low-tapping corner the two knobs do
coincide; over the full range they are strongly *non*-equivalent, and the
difference is the practical result of this phase. The claim that the space is
"effectively five-dimensional" is retracted.

### 8.2 The Tier-1 chatter screen blocks the tight-voltage region

`dso_g_v_ratio = 2.0` is marked infeasible — not on contraction, but on
`tap_ops_per_h = 5.353` against the Tier-1 screen of 4.0 set in §0.2. That
screen was calibrated as "2x what every 0814 candidate did", i.e. against a
campaign that had already given DSO voltage away. **A screen set from a
degenerate baseline forbids the region that fixes the degeneracy.** It did not
affect any result above — the scans report all rows regardless — but it would
have blocked a pattern search from ever going there.

### 8.3 Tier-2 audit of the 0815 points: the budget is met, and the voltage penalty is much smaller on realistic days

The six non-dominated filter points, on the 4 x 12 h audit set under the real
budget (`limits_mc_v2.json`):

| `dso_g_v_ratio` | `engage_dso_pu` | f_ts | f_q | f_ds | worst ops/h | **taps/day** | rev/h | feasible |
|---|---|---|---|---|---|---|---|---|
| 0.2512 | 0.0250 | 1.06099 | 9.038 | 0.27202 | 0.834 | **20.0** | 0.083 | yes |
| 0.5012 | 0.0250 | 1.06046 | 9.092 | 0.23770 | 1.084 | **26.0** | 0.083 | yes |
| 0.5012 | 0.0250 (λ_TS 0.125) | 1.09080 | 6.811 | 0.23095 | 1.167 | **28.0** | 0.083 | yes |
| 0.5012 | 0.0499 | 1.06061 | 9.118 | 0.26636 | 0.834 | **20.0** | 0.083 | yes |
| 1.0000 | 0.0499 | 1.06029 | 9.103 | 0.23800 | 1.084 | **26.0** | 0.083 | yes |

**Every 0815 point meets the 30 taps/day budget**, at 20–28, where both 0814
carry-over points failed it at 36 and 32 (§4). The re-calibration — λ_TS
0.20 → 0.25 and λ_DS 1.0 → 1.6 — moved the fleet inside the wear budget for the
first time in either campaign. Hunting passes everywhere with a factor of three
in hand (0.083/h = 2/day against 6/day).

**And the DSO voltage penalty is far smaller here than the design bank
implies.** `f_ds` spans 0.231–0.272 across these points, a range of
**18 %** — against **3x** (0.15–0.45) on the event-dense Tier-1 bank. The
*direction* is unchanged (at matched `engage_dso_pu = 0.025`, ratio 0.2512 gives
`f_ds` 0.272 against 0.5012's 0.238), but the magnitude is not. On realistic
profile-driven days the DSO's voltage regulation is much less sensitive to this
coordinate than a bank of injected disturbances suggests.

That does not dismiss §8 — selection happens on the Tier-1 bank, and there the
penalty is 47 % — but it bounds the operational consequence, and it means the
`f_ds` criterion of §8 should be **read on Tier 2 as well as Tier 1** before a
final value is fixed. The audit does not contain a `dso_g_v_ratio = 2.0` point,
because that point was never in the filter (§8.2), so the tight-voltage end of
the axis is unmeasured at 12 h.

### 8.4 The 3 x 3 grid: the two levers are not equivalent

At λ_TS = 0.25, λ_DS = 1.6, twelve design windows. `f_ds` lower is better.

| `dso_g_v_ratio` | `engage_dso_pu` | f_ts | f_q | **f_ds** | DSO taps/h | **rev/h** |
|---|---|---|---|---|---|---|
| 1.0 | 0.0125 | 1.27323 | 0.11231 | 0.12476 | 5.353 | 2.007 |
| 1.0 | 0.0250 | 1.25502 | 0.08106 | 0.29252 | 2.007 | 0.669 |
| 1.0 | 0.0500 | 1.25280 | **0.06776** | 0.43124 | 1.338 | 0.669 |
| 2.0 | 0.0125 | 1.27040 | 0.13356 | 0.05652 | 20.074 | **18.067** |
| 2.0 | 0.0250 | 1.26621 | 0.09142 | 0.15394 | 5.353 | 2.007 |
| 2.0 | 0.0500 | 1.25581 | 0.07929 | 0.30448 | 2.677 | 0.669 |
| 4.0 | 0.0125 | 1.28129 | 0.33757 | 0.08194 | 25.428 | **24.758** |
| **4.0** | **0.0250** | 1.27161 | 0.12084 | **0.05626** | 9.368 | 7.361 |
| **4.0** | **0.0500** | 1.26235 | 0.09006 | **0.17430** | 4.684 | 2.007 |

**The two levers are strongly non-equivalent, and the objective weight is the
right one.** Two points reach essentially the same DSO voltage cost:

| | f_ds | DSO taps/h | rev/h | f_q |
|---|---|---|---|---|
| ratio 4.0, engage 0.025 | 0.05626 | **9.368** | **7.361** | 0.12084 |
| ratio 2.0, engage 0.0125 | 0.05652 | **20.074** | **18.067** | 0.13356 |

Identical voltage regulation, at **less than half the tap operations and 40 % of
the reversals**. Raising the DSO's own voltage weight buys regulation
efficiently; cheapening the taps buys the same regulation by making the tap
changer hunt.

**Cheapening the taps causes hunting, monotonically and severely.** At
`engage_dso_pu = 0.0125` the reversal rate goes 2.007 → 18.067 → 24.758 as the
ratio rises, against 0.669 → 0.669 → 2.007 at `engage_dso_pu = 0.05`. Twenty-five
reversals per hour is not a tuning trade-off, it is a chattering tap changer.

**Of the author's two proposed remedies — "tighter voltage tracking in DSO, or
smaller OLTC weights" — the measurement supports the first and rejects the
second.** `engage_dso_pu` should if anything move *up*, not down, and the
voltage recovery should come from `dso_g_v`.

Two candidate operating points, against the analytic reference (ratio 1.0,
engage 0.025, `f_ds` 0.29252):

* **ratio 4.0, engage 0.050** — `f_ds` 0.17430 (**−40 %**), `f_q` 0.09006
  (+11 %), 4.684 taps/h, 2.007 rev/h. Raises *both* the objective weight and the
  tap price: better voltage with the tap activity held near the analytic point.
* **ratio 4.0, engage 0.025** — `f_ds` 0.05626 (**−81 %**), `f_q` 0.12084
  (+49 %), 9.368 taps/h, 7.361 rev/h. Much better voltage, at tap activity that
  must be checked against the real budget.

`f_ts` is flat to 2.3 % across the entire grid, so the supervisory criterion has
no opinion about any of this — which is exactly why the choice cannot be made on
the current filter.

Both, plus the conservative (2.0, 0.050) and the analytic reference, are now
running on the 4 x 12 h audit set: the Tier-1 rates above are event-dense and
must not be read as daily wear (§0.2), and §8.3 has already shown the `f_ds`
spread to be much narrower on realistic windows.

### 8.5 On realistic days: the wear budget, not the tuning, is what limits DSO voltage

The four candidates on the 4 x 12 h audit set, against the real budget
(30 taps/day, 6 reversals/day per transformer):

| `dso_g_v_ratio` | `engage_dso_pu` | **f_ds** | f_q | taps/day | rev/day | binding constraint |
|---|---|---|---|---|---|---|
| 0.5012 (incumbent) | 0.0250 | 0.23770 | 9.092 | 26.0 | 2.0 | none — feasible |
| **1.0 (analytic)** | **0.0250** | **0.21806** | 9.073 | **30.0** | 2.0 | g5a by 0.0006 |
| 2.0 | 0.0500 | 0.21865 | 9.164 | 30.0 | 2.0 | g5a by 0.0006 |
| 4.0 | 0.0500 | **0.11766** | 10.864 | **38.0** | 6.0 | g5a by 0.334 |
| 4.0 | 0.0250 | **0.03767** | 11.226 | **52.0** | **32.0** | g5a 0.918, g5b 1.084 |

Three results, and the third is the one that matters.

**(a) The author's observation is confirmed, at 9 % rather than 47 %.** On
realistic windows the incumbent's DSO voltage cost is 0.23770 against the
analytic point's 0.21806 — **9 % worse**, not the 47 % the event-dense Tier-1
bank showed. Tier 1 exaggerates this coordinate's effect by roughly 5x. The
direction of §8 stands; the magnitude quoted there is a design-bank artefact and
should not be carried into the thesis.

**(b) The incumbent is giving away voltage for wear headroom that buys
nothing.** It sits at 26 taps/day under a 30 taps/day budget. There is no reward
for coming in under budget, and the 4 taps/day of unused headroom cost 9 % of
DSO voltage regulation. **The single move Phase B made is not worth making.**

**(c) The binding constraint on DSO voltage regulation is the wear budget
itself, not the choice of weights.** The analytic point sits *exactly* at the
budget — the violation is 0.0006 ops/h, i.e. 0.014 taps/day, which is a rounding
artefact against a quantisation step of 2 taps/day. Every materially better
`f_ds` requires exceeding 30 taps/day:

| f_ds | taps/day | over budget |
|---|---|---|
| 0.21806 | 30.0 | at the limit |
| 0.11766 | 38.0 | +27 % |
| 0.03767 | 52.0 | +73 % |

The measured exchange rate is roughly **5 % of DSO voltage cost per additional
tap operation per day**, over the range 30–52. That is an operator-facing
number and it is the useful output of this phase: *the DSO tap changers do
little because the switching budget permits little*, and buying better voltage
means buying it from the maintenance budget.

Note also that within the budget the frontier is **flat**: (1.0, 0.025) and
(2.0, 0.050) give `f_ds` 0.21806 and 0.21865 at the same 30 taps/day. Two quite
different weight sets, indistinguishable in outcome. There is nothing to gain by
tuning inside the budget; the only lever is the budget.

### 8.6 Recommendation

**Set `dso_g_v_ratio = 1.0` — the analytic value — and leave
`engage_dso_pu = 0.025`.** That is: revert the one move Phase B made, and keep
the rest of the calibration.

Grounds: it is the best DSO voltage regulation obtainable within the stated wear
budget (§8.5c); the frontier inside the budget is flat, so no other weight set
does better (§8.5c); and Phase B's move to 0.5012 was selected by a filter that
does not observe `f_ds` (§8), buying `f_q` and unused wear headroom with a
controlled output.

What is given up by reverting: `f_q` 0.06886 → 0.08106 in sample (+18 %), and
the out-of-sample `f_q` gain of §7.3. What is recovered: 9 % of DSO voltage
regulation on realistic days, and a defensible statement that every controlled
output was accounted for.

**If the operator will accept more wear**, (4.0, 0.050) at 38 taps/day is the
next point on the frontier and halves `f_ds`. That is a decision about the
maintenance budget, not about control tuning, and it should be put as such.

**Two prerequisites before any of this is final**, both from §7.1 and §8: λ\*
still fails to transfer out of sample, and `f_ds` still is not in the filter. The
recommendation above is what the *current* evidence supports; it is not a
substitute for putting the third criterion in the filter and re-running the
selection.

---

## 9 — Night of 2026-08-16/17: the soundness programme, and a coordinate nobody had swept

Brief: fix the soundness defects and produce tuned values, autonomously, 12
workers.

### 9.1 Code changes

* `--rho-margin` — λ\* is selected against `rho_target/(1+δ)` while the
  *declared* ceiling stays 1.5. The criterion is unchanged ("no window may
  exceed 1.5"); the margin is an explicit allowance for the instability of the
  statistic used to check it (§7.1). `rho_worst3_mean` added as a diagnostic.
* `--filter-ds` — `f_ds` as a third filter criterion in `dominates()` /
  `filter_accepts()`. Defaults off so 0814 re-runs reproduce what they ran.
* Confirmation bank **6 → 9 windows**, ≥2 per stratum (none 2 / partial 3 /
  full 4), fixing the n=1 defect of §7.2.
* `limits_mc_v2_scan.json` — wear screens inactive during *scans*, since the
  Tier-1 screen was derived from a baseline that had already given DSO voltage
  away (§8.2) and was hiding rows rather than protecting anything.
* **`bank_fingerprint()`** — a third unversioned-artefact defect, found the hard
  way. Evaluations are cached on `(scenario_set, knob hash)`, but
  `--scenario-set confirm` names a *function*, and extending it from six windows
  to nine left eight cached results whose knob hashes still matched. A re-run
  would have reported six-window numbers as nine-window ones.
  `_design_is_current` cannot catch it: the weights are right, the *ensemble*
  changed. Stale files renamed `.6window`.
* `--tag` on `--phase eval` — two concurrent eval runs on one scenario set were
  both writing `eval_tier1.json`, and the second silently overwrote the first.
  Per-candidate files were unaffected, so nothing was lost.

### 9.2 The sampling distribution of the contraction statistic

Pooling all 18 windows (12 design + 9 confirmation share 12) at the recommended
point, and resampling all C(18,12) = 18564 twelve-window banks:

| statistic | p50 | worst | bank-to-bank spread |
|---|---|---|---|
| **max** (the criterion) | 1.5201 | 1.5201 | bimodal, 3–4.5 % jump |
| mean of worst 2 | 1.4972 | 1.5091 | 0.79 % |
| mean of worst 3 | 1.4886 | 1.4975 | **0.60 %** |
| p75 over windows | 1.4713 | 1.4743 | 0.20 % |

The top of the pooled distribution is `1.5201, 1.4980, 1.4743, 1.4713` — one
window sits 1.47 % above the next, and a random twelve-window bank contains it
with probability **0.667**. So `rho_worst` is not smoothly variable, it is
**bimodal**: the calibration depends on whether the bank happened to draw that
window. No margin can be estimated from a single bank, which is why the 0815
calibration failed out of sample.

Adopted: keep the criterion on the max (it is the physically meaningful
statement) and select against `1.5/1.031`. The mean-of-worst-3 is recorded so a
future campaign can move the criterion onto it, which would need a ceiling
restated for a mean rather than a max.

### 9.3 `tau` is confirmed static; `engage_tso_pu` is not what it was thought to be

`tau` swept over {0.25, 0.5, 1, 2, 4}: `f_ts` minimises at **exactly the
analytic value 1.0** (1.2550, against 1.2670 / 1.2584 / 1.2631 / 1.2680).
`f_q` prefers 2.0 by 6 %. The coordinate is confirmed, not merely unmoved.

`engage_tso_pu` swept over {0.005 … 0.030}, at λ_TS = 0.25:

| `engage_tso_pu` | `g_w_tso_oltc` | rho | implied floor | f_ts |
|---|---|---|---|---|
| 0.005 | 565 | 7.7111 | 7.325 | 1.28308 |
| 0.010 | 2186 | 2.2629 | 1.877 | **1.24448** |
| 0.015 | 3783 | 1.4743 | 1.088 | 1.25502 |
| 0.020 | 5378 | 1.1599 | 0.774 | 1.29608 |
| 0.030 | 8566 | **0.8829** | 0.497 | 1.40477 |

**`rho ≤ 1` is reachable, and the thesis claim that it is not is wrong as
written.** The floor of ~1.09 is a property of `engage_tso_pu = 0.015`, not of
the plant: the commit threshold sets `g_w_tso_oltc`, the tap columns' weight,
and the floor goes as its inverse. The corollary in the earlier text — "any
tightening requires re-pricing the OLTCs" — was right, and `engage_tso_pu` *is*
the coordinate that re-prices them. Corrected in `docs/ch9_weight_selection_method.tex`
and `docs/tuning/METHOD_weight_selection.md`.

A second corollary: `engage_tso_pu` has a **lower bound set by the contraction
criterion**, not by the plant corridor. At 0.010 the floor alone is 1.877, above
the ceiling at any λ, so that threshold is inadmissible however the continuous
weights are chosen — even though it gives the best `f_ts` in the scan. On this
plant the criterion bounds it below at ≈ 0.0135.

### 9.4 Why the pattern search missed this — a correction to §6

§6 read Phase B's five unmoved coordinates as the analytic construction being
vindicated. **That reading is wrong.** A compass search polls ±δ in one
coordinate at a time, so it cannot find a direction whose gain needs two
coordinates to move together:

* raising `engage_tso_pu` alone worsens `f_ts` → rejected;
* raising `lambda_tso` alone breaks the contraction ceiling → rejected;
* raising **both** improves `f_ts` *and* `f_q` and quadruples the margin →
  never polled.

Phase B carries one coupled probe, for (`lambda_tso`, `lambda_dso`) — not for
the pair that mattered. **The coordinates a local search leaves alone are
exactly the ones that need a dedicated sweep**, and "the search did not move it"
is evidence about the search, not about the coordinate.

### 9.5 The joint sweep, and the new operating point

| `engage_tso` | λ_TS | f_ts | f_q | rho | margin to 1.5 |
|---|---|---|---|---|---|
| 0.015 | 0.25 | 1.25502 | 0.08106 | 1.4743 | 1.7 % |
| **0.017** | **0.30** | **1.25139** | **0.07445** | **1.3987** | **7.2 %** |
| 0.018 | 0.38 | 1.23821 | 0.07422 | 1.4620 | 2.6 % |
| 0.020 | 0.44 | 1.24882 | 0.07559 | 1.4555 | 3.1 % |

(0.017, 0.30) improves both filter criteria at identical tap activity and
**quadruples the contraction margin**. (0.018, 0.38) is better on `f_ts` but
carries 2.6 % margin against the 3.1 % §9.2 requires, and is rejected on that
ground — declining a better in-sample number for a stated reason is the whole
point of the corrected rule.

λ_DSO re-swept at the new point: still an interior minimum at **1.6**
(`f_q` 0.07445 against 0.07591 / 0.07876 / 0.09723 at 1.4 / 1.2 / 1.9). The
round trip closes: λ_TS moved and λ_DS did not follow. `rho` is **exactly
1.3987 in all six rows** — a fourth confirmation of layer decoupling.

### 9.6 Confirmation on the 9-window bank: the corrected rule works

| | f_ts | f_q | f_ds | rho | feasible |
|---|---|---|---|---|---|
| **A (new)** | 1.75307 | 0.07249 | 0.32605 | **1.4379** | **yes** |
| D (old baseline) | 1.77045 | 0.07755 | 0.33553 | **1.5201** | **no** |
| Δ | **−0.98 %** | **−6.52 %** | **−2.83 %** | | |

**The new point is admissible out of sample where the old one is not, and it
improves all three criteria.** In sample rho was 1.3987; out of sample 1.4379,
a rise of 2.8 % — within the 3.1 % §9.2 predicted, and absorbed by the margin.
The correction is validated end to end.

The `f_ts` gain **grows** out of sample (−0.28 % → −0.98 %) and improves in
every stratum (−0.96 % full, −0.40 % none, −2.05 % partial), so it is not one
window carrying the result. The 9-window bank is 1.4x harder than the design
bank against the 6-window bank's 1.96x: raising `partial` from n=1 to n=3 fixed
the comparability defect of §7.2.

### 9.7 The wear budget is the binding constraint, and the new point exceeds it

Tier-2 audit, 4 x 12 h, budget 30 taps/day and 6 reversals/day:

| | f_ts | f_ds | **taps/day** | rev/day | rho | feasible |
|---|---|---|---|---|---|---|
| A (balanced) | 1.07838 | 0.20492 | **36.0** | 6.0 | 1.3563 | **no** |
| B (DSO-voltage) | 1.07897 | **0.15970** | **42.0** | 6.0 | 1.3563 | **no** |
| C (reference) | 1.07694 | 0.21365 | **40.0** | 6.0 | 1.3563 | **no** |
| D (old baseline) | 1.06651 | 0.21806 | **30.0** | 2.0 | 1.4474 | no (contraction) |

`DSO_4|trafo_10` is the offender in every case: 30 → 36 → 40 → 42 taps/day.
Everything else in the fleet is well inside.

**No point yet satisfies both constraints.** D meets the wear budget and fails
the contraction ceiling out of sample; A/B/C pass contraction with margin and
exceed the wear budget by 20–40 %. That is a real result about the plant, not a
tuning failure — **the switching budget and a transferable contraction margin
are in tension**, and neither previous campaign could see it because neither
constraint was being measured correctly.

The binding coordinate is DSO-side, so the last wave raises `engage_dso_pu` to
0.050–0.065. Tier-1 for those (all at rho 1.3987, reversals 0.669):

| ratio | `engage_dso` | f_ts | f_q | f_ds | DSO taps/h |
|---|---|---|---|---|---|
| 1.0 | 0.050 | **1.24756** | **0.06933** | 0.44070 | 1.338 |
| 1.5 | 0.050 | 1.24818 | 0.07330 | 0.37135 | 2.007 |
| 2.0 | 0.050 | 1.25001 | 0.07726 | **0.30826** | 2.677 |
| 2.0 | 0.065 | 1.24814 | 0.07323 | 0.36608 | 2.007 |

All four beat A on `f_ts`. Their audit numbers decide the recommendation.

---

## 10 — What lambda and rho are actually computed on, and what that costs

Raised by the author: *was the calculation of lambda and rho done on the
submodels, local only, neglecting the physical connection to neighbouring
areas?* Verified in the source, and yes.

### 10.1 The mechanism, verified

`tuning/scripts/configs/baseline_ieee39_thevenin.yaml` sets
`local_sensitivities_tso: true` and `local_sensitivities_dso: true`.
`multi_tso_dso.py:2473` then calls
`coordinator.compute_cross_sensitivities(zero_offdiag=bool(config.local_sensitivities_tso))`,
so **every off-diagonal block `H_ij` (i != j) is set to zero before the
criterion is formed**. Both quantities inherit that:

* the **analytic target** `lambda_max(M)` that Stage 0 designs against, and
* the **measured** `rho_emp_p95`, which is the p95 of
  `zone_contraction_lhs` over the run -- the coordinator's *own* per-zone
  quantity, formed from the same zeroed blocks.

What is *not* local is the trajectory: pandapower solves the whole
interconnected network, so the operating points at which the local criterion is
evaluated are physically correct. **The matrix is local; the trajectory is
coupled.**

### 10.2 What the neglected coupling is worth

`stage_0_coupling_decomposition.py` gained `--with-coupling`, which recomputes
the cross-sensitivities with the off-diagonals **retained** and re-evaluates the
criterion for the *same, locally-designed* controller. It does not change the
control law; it evaluates the same loop against a less restricted model, so the
difference is the model gap. At the recommended point:

| zone | `lambda_max(M_ii)` | coupling `sum_j!=i ||M_ij||` | criterion |
|---|---|---|---|
| 1 | 0.7188 | 0.3549 | 1.0737 |
| **2** | **1.3488** | **0.8285** | **2.1773** |
| 3 | 1.0193 | 0.7184 | 1.7377 |

and the floor (continuous weights to infinity) moves 0.9564 → 1.5872.

**The coupling term is not a correction, it is comparable to the local term.**
The worst-zone criterion goes 1.3488 → **2.1773, a factor of 1.61**, and the
figure the whole campaign has been calibrating against is the smaller one.

### 10.3 Two consequences

**(a) The "25 % margin below the OFO bound of 2" does not survive contact with
the coupling.** Scaling the measured `rho_emp_p95` by the same factor puts the
recommended point at roughly 1.40 x 1.61 ~ **2.26**, i.e. *above* the nominal
bound rather than 25 % below it. Every candidate in both campaigns is in the
same position.

**This does not mean the controller is unstable, and it must not be reported as
if it did.** `lambda_max(M_ii) + sum_j ||M_ij||_2` is a **sufficient** condition
built on a triangle-inequality bound over the off-diagonal blocks; the true
spectral radius of the assembled `M` is bounded above by it and can be far
smaller. Exceeding it proves nothing. Empirically nothing diverged:
`g1_diverged = 0` in every one of the ~130 evaluations of this campaign, across
both tiers.

What it does mean is that **`rho_emp_p95 <= 1.5` is an orientation value, not a
stability certificate**, and the margin below 2 is not the margin it claims to
be.

**(b) It explains the Phase-2 anomaly (§2.1b).** Raising zone 1's gain moved the
measured worst-zone `rho` by 7.9 % when the local per-zone decomposition said
zone 2 pins the maximum and zone 1 has 24x headroom. With coupling that is
immediate: `M_21 = G_w,1^{-1/2} H_22^T Q_2 H_21 G_w,2^{-1/2}` carries **zone 1's
weights**, so raising zone 1's lambda lowers `g_w,1`, raises `||M_21||`, and
raises *zone 2's* criterion. The zones interact through the physical network
whether or not the controller's model admits it. §2.1b recorded this as an
unexplained contradiction; it is now explained, and the explanation is the
coupling the model zeroes.

### 10.4 How this should be stated

The procedure is unchanged and, if anything, better justified:

1. The analytic `lambda` target is a **design-rule input** computed on the
   local submodel. It is where the search starts, not a prediction of the
   closed loop.
2. `rho_emp_p95` is a **measured proxy** for contraction, formed from the same
   local model along fully-coupled trajectories. It is monotone in the loop gain
   and reproducible, which is what a calibration coordinate needs -- and it is
   *not* a certificate.
3. Because neither is exact, the loop gain is **calibrated empirically** against
   a declared ceiling and then confirmed out of sample. That is not a fallback;
   it is the only defensible route, and §9.6 shows it working.

So: state the locality explicitly, state that the criterion is sufficient rather
than necessary, quote the 1.61x coupling factor measured here as the size of
what is neglected, and present the ceiling as an orientation value calibrated
by measurement. The grid search is then not an admission of weakness -- it is
what makes the number mean anything.

**Open, and now well-posed:** the true contraction of the coupled loop is
bounded above by 2.18 and below by `max_i lambda_max(M_ii)` = 1.35 at this
point. Narrowing that requires the spectral radius of the assembled `M`, not
the block bound -- one eigenvalue computation on a matrix the decomposition
module already assembles.

### 8.7 What was run

A 3 x 3 grid over `dso_g_v_ratio` in {1.0, 2.0, 4.0} and `engage_dso_pu` in
{0.0125, 0.025, 0.050}, at the calibrated λ's, reporting `f_ts`, `f_q`, **`f_ds`**
and DSO tap activity. The point of the grid is to establish whether the trade-off
really is one-dimensional (§8.1(b)) or whether some corner buys DSO voltage
without paying the full interface-Q price.

The selection cannot be made on the two-criterion filter, since the whole
finding is that the filter is under-specified. **The recommendation this campaign
will make is that `f_ds` becomes a third filter criterion, or a constraint with
a stated bound** — a decision for the author, since it changes what the
procedure optimises.

