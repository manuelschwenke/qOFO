# Handoff prompt — overnight tuning campaign (write-up for a fresh Claude session)

Paste everything below the line into the new session on the 40-core machine.

---

You are picking up a PhD tuning campaign mid-stream. The project lives on a
network share that this machine can reach:

    \\130.83.232.108\homefolders$\mschwenke\Python_Projekte\qOFO_GH

**First, orient yourself — do not skip this:**

1. Locate the Python interpreter. On the author's laptop it is
   `C:\Users\Manuel Schwenke\.conda\envs\qOFO_clean\python.exe` (Python 3.12,
   conda env `qOFO_clean`). **On this machine the path is probably different** —
   find the `qOFO_clean` env, verify `python -c "import pandapower, numpy"`
   works, and use that interpreter for everything. Do not create a new env.
2. Read, in this order:
   - `docs/daily_log/08_2026/2026-08-14_lambda_calibration_run_and_thesis_9_3.md`
     — the campaign this continues. Sections 9, 10 and 11 are the ones that
     matter most; §5 and §8 describe two defects that were fixed and must not be
     reintroduced.
   - `tuning_mc/stage_1_search.py` and `tuning_mc/metrics.py` docstrings.
   - `tuning_mc/scenarios_mc.py` — the scenario set you are going to replace.
3. Everything from the previous run is archived under
   `results/tuning_mc/campaign_0814/` and `results/tuning_mc/stage1/`. **Do not
   delete or overwrite it.** Write this campaign to a new directory,
   `results/tuning_mc/campaign_0815/`.

## Where the previous campaign got to

Calibrated coordinates, all on the 5-window / 90-min design bank:

    lambda_tso 0.20   lambda_dso 1.4125   tau 1.0
    engage_tso_pu 0.015   engage_dso_pu 0.025   dso_g_v_ratio 0.5012

Measured contraction `rho_emp_p95 = 1.1702 + 1.6172*lambda_tso` (resid 0.0066),
ceiling 1.5 declared as a 25 % margin below the OFO bound of 2. The analytic
counterpart from the cached model is `1.1101 + 1.3221*lambda_tso` — intercept
within 5 %, slope 22 % low.

Phase B converged after 5 polls, accepting only 2 moves; four of six coordinates
never left their analytic values. On the holdout the TS-voltage gain survived
(−0.7 %) and the interface-tracking gain did not (−21 % → −1.4 %).

## Why this campaign exists — four defects in the experiment design

1. **The wear and hunting constraints cannot be measured on 90-minute windows.**
   One tap in a 90-min window is 0.667 ops/h, so the reversal limit of 1.2054/h
   falls *between* the one- and two-reversal levels. The whole holdout
   feasibility verdict turned on a single tap reversal.
2. **The limits are inconsistent with the stated budget.** `tap_ops_per_h =
   6.0268` is 145 taps/day, against a budget of 30/day per transformer.
3. **The design bank is too small and too easy.** 5 windows; holdout costs are
   ~2x design-bank costs on the same metric; the interface-tracking gain did not
   generalise.
4. **19 % of the profile year has exactly zero DER reactive capability**
   (measured, `tuning_mc/stage_1a_excitation.py --screen1`), where `tau` and
   `lambda_dso` are structurally inert. The bank does not control for this.

## Decisions already made by the author — do not relitigate these

* **Tier-2 audit windows are 12 hours.**
* **The wear budget is 30 tap operations per day per transformer** (the existing
  `tap_ops_per_h` metric is already worst-transformer, not fleet sum), i.e.
  **1.25 ops/h**. Re-derive `ConstraintLimits` from this.
* **Use 16 workers.** The machine has 40 cores and ~300 GB RAM, but throughput
  on this simulator is memory-bandwidth bound on the sparse Newton power flow —
  it was measured to peak at 6 workers and regress past 10 on the laptop. Cap at
  16 and do not raise it even if cores look idle. A brief scaling check
  (2 candidates at 8 vs 16) is worth 20 minutes before committing the night.

## The work, in priority order

Checkpoint after every phase: write results to disk and append to the daily log
as you go. **If you run out of night, the early phases must already be banked.**

### Phase 0 — limits and scenario sets (no simulation)

* New `tuning_mc/configs/limits_mc_v2.json`: `tap_ops_per_h = 1.25` from the
  budget above. For `tap_reversals_per_h`, first try to find the provenance of
  the current pair (`6.0268` / `1.2054`, an exact 5:1 ratio — search the git
  history and `docs/`). If the provenance does not dictate otherwise, preserve
  the 5:1 ratio, giving **0.25/h**, and *document the choice as a choice*. At a
  12-h window that is 3 reversals against a quantisation step of 0.083/h, i.e.
  properly resolved. Keep `rho_emp_p95 = 1.5`, `corridor_excess_pu` and
  `settling_s` unchanged.
* New scenario module `tuning_mc/scenarios_mc_v2.py`:
  - **Tier 1 (search): ~12 windows x 90 min**, on `rural_700`. Keep the
    excitation-role design of `scenarios_mc.py` (quiet control; impulsive
    generator outage; load ramp up; load ramp down; sign-reversing) but
    replicate each role across seasons, and **stratify by DER reactive
    capability** using `stage_1a_excitation --screen1`, recording each window's
    stratum so that `f_q` can be reported per stratum. Do not let
    zero-capability windows dominate the aggregate.
  - **Tier 1 confirmation: ~6 windows**, drawn from the *same* distribution as
    the design bank, so the two are comparable in difficulty — the current pair
    is not. Preserve the disjointness convention: design windows in odd calendar
    weeks, confirmation in even ones.
  - **Tier 2 (audit): 4 windows x 12 h**, profile-driven, spanning seasons; two
    quiet and two carrying one realistic event. This is where wear and hunting
    are actually measured, and it subsumes the old `wear_day_set`.
* Sanity-check the bank before spending a night on it: report per-window `f_ts`
  for one candidate and confirm no single window contributes most of the
  aggregate (in the old bank `mc_undervolt_ramp_winter` contributed ~60 % *and*
  was a zero-capability window).

### Phase 1 — re-calibrate lambda_TSO on the new bank (~7 candidates)

    --phase scan --scan-knob lambda_tso --scan-values 0.10,0.15,0.20,0.25,0.40,0.60,0.90
    --fix lambda_dso=1.0 --limits tuning_mc/configs/limits_mc_v2.json --rho-target 1.5

`rho_emp_p95` is a TSO-only diagnostic and is invariant to `lambda_dso`
(confirmed twice), so holding it is safe. Expect the affine fit to move somewhat
— the bank changed. Report the new fit and `lambda*`.

### Phase 2 — test the per-zone lambda hypothesis (~6 candidates, gated)

This is the most promising new coordinate and the reason a bigger search space
was requested. Analytic per-zone contraction, from
`tuning_mc/stage_0_coupling_decomposition.py` (fits over lambda in
{0.1,0.2,0.4,0.9}, residuals <= 0.005):

| zone | floor | slope | lambda its own ceiling permits |
|---|---|---|---|
| 1 | 0.795 | 0.124 | ~4.7 |
| 2 | 1.110 | 1.322 | ~0.20 |
| 3 | 0.899 | 0.836 | ~0.58 |

**A single global `lambda_tso` is set entirely by zone 2**, leaving zones 1 and 3
roughly 24x and 3x inside their own contraction limit. Zone 1's worst mode is
almost pure floor, so its continuous columns barely move it.

**Treat this as a hypothesis, not a win.** A low slope means those columns do not
drive the *worst* mode — not that raising their gain helps. Zone 1 owns one DER
column and no PCC class at all, so it may simply be a weak zone whose larger
steps hit actuator bounds. The "lambda at measured 1.5" column above also
extrapolates a *global* measured/analytic derating of 1.087 to individual zones,
which has not been verified per zone.

Gate it: **scan zone 1's lambda alone** over ~{0.2, 0.5, 1.0, 2.0, 3.5} with
zones 2 and 3 held at the Phase-1 value. If `f_ts` does not improve
meaningfully, drop the idea and keep the global coordinate. Only if it pays,
promote `lambda_tso` to three per-zone coordinates in Phase 4.

Implementation note: **the config hook already exists.** `zone_g_w_class`
(per zone, per actuator class) can express a per-zone continuous scaling exactly,
and `stage_0_preconditioning.py` already emits a paste-ready `zone_g_w_class`
block. Add `--lambda-tso-zone "1=...,2=...,3=..."` to Stage 0 rather than
inventing a new mechanism. Re-run the analytic decomposition after any change to
`H`, the zone partition or the operating point — the table above is not portable.

### Phase 3 — re-calibrate lambda_DSO (~8 candidates)

`--scan-knob lambda_dso --fix lambda_tso=<phase 1 result>`, grid
`0.15,0.3,0.6,0.9,1.0,1.2,1.4,1.6`. Note from last time: the `f_q` optimum moves
with `dso_g_v_ratio` (the two are coupled), so state the conditions with the
value. `rho(TSO)` must come out *exactly* constant across this scan — if it does
not, something is wrong with the layer separation and you should stop and say so.

### Phase 4 — probe and search

    --phase a --x0 "lambda_tso=<p1>,lambda_dso=<p3>" --limits ...
    --phase b --x0 "..." --limits ...

Liveness is judged on **both** filter criteria — a direction is live if either
`f_ts` or `f_q` responds by >= 1 %. Do not revert this to `f_ts` alone; doing so
last time would have discarded `lambda_dso`, the coordinate the preceding sweep
had just selected on `f_q`.

### Phase 5 — Tier-2 audit (~10 candidates, 4 x 12 h each)

Run the final non-dominated filter points through the 12-h audit set. **This is
the authoritative wear and hunting measurement** and the first time the 30
taps/day/transformer budget is actually tested. Report ops/h and reversals/h per
transformer, not just the worst.

### Phase 6 — confirmation

Evaluate the chosen incumbent *and* the analytic baseline on the Tier-1
confirmation set. Report both, and apply the author's rule: an in-sample
improvement that does not survive is not carried forward.

## Guardrails

* **Do not compile the LaTeX thesis.** The author builds it in their editor. You
  may edit `latex_diss_ms/Chapters/*.tex`; you may not run `latexmk`/`lualatex`.
* **Do not edit `controller/gw_precondition.py`.** Changing it would alter the
  meaning of every existing study. Stage 0 carries its own overrides.
* Keep the Stage-0 design fingerprint mechanism intact (`stage0_fingerprint()` /
  `_design_is_current()` in `stage_1_search.py`). It exists because a design
  cached before a rule change was silently reused and put one row of a
  calibration curve on a different rule from the other six.
* Prose convention in the thesis: en dashes `--`, never `---`. Max
  `\section` + `\subsection`, no subsubsections.
* Log the work to `docs/daily_log/08_2026/` — what changed, the method, the
  reason, the numbers, and anything you had to retract.
* **Report faithfully.** If a phase fails, say so with the output. If you run out
  of time, say which phases did not run. Do not extrapolate Tier-1 numbers to a
  daily budget — that is the mistake this whole campaign exists to fix.

## Rough budget

At the measured rate (~2.1 min wall per simulated hour per worker), Tier 1 at 12
windows x 90 min is ~38 min/candidate; Tier 2 at 4 x 12 h is ~100 min/candidate.
Phases 1-4 are roughly 170 candidates ≈ 110 worker-hours ≈ **7 h at 16 workers**;
Phase 5 adds ~1 h. Re-measure on this machine before trusting that.

If the night is running short, the order above is already the priority order:
Phases 0-3 are the ones that must land.
