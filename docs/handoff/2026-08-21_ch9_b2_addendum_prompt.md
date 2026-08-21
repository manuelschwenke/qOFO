# Handoff prompt — what §9.1 may and may not claim about the supervisory period

Written 2026-08-21 on the server session that ran the sweep. Companion to
`2026-08-21_ch9_table91_fill_prompt.md` (Table 9.1) and
`2026-08-21_ch9_ninner_figure_prompt.md` (the `N_inner` figure). The target
session owns the thesis repo and has `Z:\` mounted.

Everything between the rules is the prompt.

---

You are my research assistant on a PhD project on hierarchical multi-zone
reactive-power control. This session owns the dissertation LaTeX repo
(`...\12_Dissertation\latex_diss_ms`) and can read `Z:\Python_Projekte\qOFO_GH`.

**Your job is a correction, not an addition.** An experiment was run to justify
the selection of the supervisory period `T_TS = 180 s`. It does not justify it,
and §9.1 must not be written as though it does. Read the chapter's current
wording around the choice of `T_TS` and bring it in line with what was actually
measured.

**Do not invent numbers.** Everything is in the run directory below. If a file
disagrees with this brief, **trust the file and tell me**.

```
Z:\Python_Projekte\qOFO_GH\results\ch9_ts_period_sweep\final_weights\20260821-092509\
    summary.md    summary_by_period.csv    intervals.csv    run_meta.json

Z:\Python_Projekte\qOFO_GH\docs\daily_log\08_2026\
    2026-08-21_ch9_b2_sweep_at_final_weights.md
```

## What the experiment was

A sweep of the supervisory period over `T_TS ∈ {60, 120, 180, 240, 300} s` at a
fixed subordinate period `T_STS = 20 s` — i.e. configured ratios
`N_inner ∈ {3, 6, 9, 12, 15}` — across the same 12 design windows the weights
were tuned on, with identical profile, contingency schedule and seed at every
point. Nothing but `tso_period_s` changes. 60 runs, exit 0.

Per dispatch interval it scores `ρ_k = r_k / Δ_k`, the fraction of the requested
interface-Q correction still outstanding when the next supervisory dispatch
lands, plus censoring, tap-lockout occupancy, tap moves and voltage violations.
Reported as a distribution per period and never time-aggregated, because a short
period yields proportionally more dispatches and a pooled residual would mix
residual-per-dispatch with dispatch frequency.

## The result

| `T_TS` [s] | `N_inner` cfg | scored | `ρ` med | `ρ` p95 | censored | lockout | taps/interval | V viol |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60 | 3 | 711 | 0.474 | 2.546 | 0.22 | 0.00 | 0.000 | 0.000 |
| 120 | 6 | 786 | 0.373 | 2.769 | 0.23 | 0.00 | 0.000 | 0.000 |
| 180 | 9 | 768 | 0.304 | 3.558 | 0.22 | 0.00 | 0.000 | 0.000 |
| 240 | 12 | 726 | 0.258 | 2.991 | 0.23 | 0.00 | 0.000 | 0.000 |
| 300 | 15 | 698 | 0.222 | 2.662 | 0.22 | 0.00 | 0.000 | 0.000 |

## The correction — this is the deliverable

`ρ_k` measures how well the subordinate layer **executed the correction it was
told to make**. It cannot see whether that correction was still the right one by
the time it landed. The staleness cost of a long supervisory period — the thing
that would produce a minimum and single out 180 s — is therefore **invisible to
this metric by construction**.

And the data behaves accordingly: `ρ` declines monotonically, 0.474 → 0.222. On
this metric alone, longer is always better, which is plainly not a selection
argument for 180 s.

So the chapter may claim this, and no more:

> Over the measured range 60–300 s, the supervisory period is **admissible**:
> the subordinate layer's residual tracking error, its censoring fraction, its
> tap-lockout occupancy and its voltage-violation count show no failure at any
> period. The choice of `T_TS = 180 s` within that range is **not** established
> by this measurement; the stale-setpoint cost that would distinguish the
> candidates lives in the supervisory tracking objective (`f_ts`, `f_q`), which
> this experiment does not evaluate.

**Find and fix any sentence that implies otherwise.** Specifically, do not write
that the sweep *selects*, *justifies*, *confirms* or *optimises* 180 s. If the
draft currently leans on a case study that was cut from the thesis, that leaning
has to go too — this experiment replaces it and reaches a weaker conclusion.

The absence of a U-shape is **not** evidence against one. Say that explicitly if
the text discusses the shape at all; otherwise say nothing about shape.

## A prediction in the original design that did not survive

The experiment was designed expecting `T_TS = 60 s` to be confounded: with
`N_inner = 3` and a 60 s coupler-tap cooldown, the changer should have been
locked out for essentially the whole interval, and lockout occupancy was
instrumented specifically to separate *the subordinate layer cannot converge in
three iterations* from *the changer was unavailable*.

**Measured lockout occupancy is 0.000 at every period.** The confound does not
arise, so `T_TS = 60 s` is not excluded on that ground. If the draft argues
against short periods via tap lockout, that argument is unsupported.

Read the "Reading the lockout column" section of `summary.md` before using this:
the occupancy is *inferred* from observed tap moves and the two cooldown
mechanisms, not logged, so a changer that did not move because the controller
chose not to move it is indistinguishable from one that could not. Which brings
us to the next point.

## Why lockout is zero: the tap changers have gone inert

Counted over all 30 240 intervals of each sweep, unfiltered, against the same
sweep run at the superseded weights (`campaign_0815 / aa4f6d4a8654`):

| | superseded weights | final weights |
|:--|--:|--:|
| total DSO tap moves | 584 | **3** |
| voltage violations | 93 | **0** |
| lockout occupancy | 0.022–0.025 | **0.000–0.001** |

At the tuned point the subordinate tap changers move three times in thirty
thousand intervals. This is partly **by design**: the per-area voltage relief
raises `dso_g_v` and `g_w_dso_oltc` together, holding the OLTC loop gain while
letting the DER shape voltage, precisely so the tap does not limit-cycle — and
it eliminated all 93 voltage violations. The independent `N_inner` experiment
saw the same thing from another angle: one subordinate network stopped 28 Mvar
short of setpoint across all 60 of its steps with **zero** tap moves while
reporting 30–36 Mvar of headroom.

⚠ **Consequence for §9.1, and it is not yet resolved.** The chapter reasons about
tap mechanics — `T_mech`/`T_elec`, and the 60 s / 180 s cooldowns expressed as 3
and 9 dispatch intervals — for an actuator the tuned controller barely operates.
A separate open question is under review (whether the settling battery applies a
non-physical 5 s command delay on top of the tap's 5 s mechanical time constant;
see the HOLD block in `2026-08-21_ch9_table91_fill_prompt.md`). **Do not rewrite
the tap discussion until that clears.** Raise the tension in your reply so the
author can decide whether the inertness belongs in §9.1 as a caveat on the tap
reasoning, or in §9.3 as a property of the weight design — my view is that it is
a §9.3 finding with a one-sentence forward reference from §9.1, but that is the
author's call.

## Do not write

- **`N_inner` is not readable from this sweep.** The `n_k` column still uses the
  arrival-in-band criterion that was rejected for the isolated experiment: its
  median is 0.0 and its p95 is exactly the configured ratio at every period,
  which is the censoring signature and not a measurement. `N_inner` comes from
  `ch_9_1_ninner_isolated_sts.py` and its own figure. Ignore `n_k` entirely.
- **`T_TS / T_STS = 9` is the configured ratio**, not a measured quantity.
- Nothing about closed-loop plant settling. This sweep is quasi-steady-state:
  "converged" means the QSS iteration converged, not that the plant settled. The
  closed-loop RMS chapter is where that is tested.

## Provenance for any citation

- `ch_9_1_ts_period_sweep.py --workers 10 --label final_weights`, run
  `20260821-092509`, exit 0, 60/60 cases.
- Commit `29522e5` on `main`, working tree **dirty**; the 1110-line diff is
  archived as `worktree_at_launch.patch` in the run directory, so the run is
  reproducible from the commit plus that patch. Say so if you cite it.
- Weights: campaign `stage1`, candidate `fe010aa3ead1` (`rho_emp_p95` 1.3788),
  rebuilt through the campaign's own recipe and asserted equal to the archive,
  with the per-area relief on DSO_2/DSO_4 applied and both halves asserted.
- Settling band 1.0 Mvar, delta floor 1.0 Mvar, `T_STS = 20 s` fixed, bank
  `tier1_design_set` on `rural_700`.
- The earlier sweep `results/ch9_ts_period_sweep/full/20260819-134011...` ran at
  the superseded candidate and must not be cited except as the comparison in the
  tap-inertness table above.

## Working rules

1. Distinguish measured facts, hypotheses and open questions. Label anything
   projected or carried over. No invented numbers.
2. If something contradicts this brief, trust the run directory and say so.
3. Answer as: short answer; assumptions; details; risks / open points.
4. The deliverable is corrected wording around the choice of `T_TS`, plus a
   flagged question about where the tap-inertness finding belongs. Do not
   restructure §9.1 and do not touch the tap discussion yet.
