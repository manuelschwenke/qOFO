# Chapter folders vs the dissertation — what to change

Updated 2026-10-23 after the chapter split, and after three corrections from
the thesis side. **Applied the same afternoon** to four of the five folders,
after the author settled the naming question (see the end). The layout, the
run folders and the archive rule are documented in `results/README.md` and
`experiments/thesis_runs.py`; the change record is
`docs/daily_log/10_2026/2026-10-23_chapter_results_layout.md`.

Authoritative source: the `\include` list in `ClassicThesis.tex` and the
`\chapter{...}` / `\label{ch:...}` lines in each `Chapters/ChapterNN.tex`, at
`C:\Users\Manuel Schwenke\Desktop\Daten\01_Forschung\12_Dissertation\latex_diss_ms`.
Change record on the thesis side:
`00_daily_log/2026-10-23_dad_chapter_split_regframe_trim.md`.

## The live map

The dispatch-anchored droop left Ch 5 (where it was §5.6) and is its own
chapter, so everything after it moved up by one. All `\label` names are
unchanged — only numbers moved.

| Ch | Label | Title (short) | was |
|---:|---|---|---:|
| 4 | `ch:qvofo` | Mixed-Integer OFO for Reactive Power Control | 4 |
| 5 | `ch:architectures:multisystem` | Multi-System OFO | 5 |
| 6 | `ch:droop` | Dispatch-Anchored Droop as the Local Layer | §5.6 |
| 7 | `ch:regframe` | Proposal for a Regulatory Frame | 6 |
| 8 | `ch:setup` | Benchmark Networks, Reference Models, Evaluation Methodology | 7 |
| 9 | `ch:timescales` | Timescale Separation of the Control Layers | 8 |
| 10 | `ch:case_svr` | Case Study: OFO in Place of Pilot-Node SVR | 9 |
| 11 | `ch:case3` | Case Study: Multi-System OFO, Cascaded and Multi-Area | 10 |
| 12 | `ch:case_sts` | Sensitivity Study: Voltage Support from Sub-Transmission | 11 |
| 13 | `ch:conclusions` | Conclusions and Outlook | 12 |

## Corrections to the first version of this note

1. **Ch 5 is `ch:architectures:multisystem`, not `ch:droop`.** The first
   version said `ch:droop`, from a label-extracting regex that excluded colons
   and so fell through to a later, simpler label. `ch:droop` was only §5.6 and
   is now Ch 6.
2. **`ch:timescales:plant` is LIVE, not cut.** It is the modal-analysis section
   of `Chapter09.tex` ("Plant Time Constants and the Subordinate Period"). The
   archived `Ch9_sec1_step_tests_CUT.tex` carries the same label but is not
   compiled, and the first version reported the archived copy because it took
   the first grep hit. **`analysis/build_modal_bundle.py`,
   `modal_observability.py` and `modal_reduce.py` cite a live section — do not
   retire them.**
3. **"The thesis has moved off the RMS path" was too broad.** Only the
   *closed-loop* RMS verification is archived (`ch:case_rms`). The modal
   analysis still linearises the RMS model of `ch:setup:multi:dynamic` at
   W1–W6, and that section carries the post-outage check, both PowerFactory
   runs. So `pf/` and `export/` remain load-bearing, and restoring them on
   2026-10-22 was right.

## Folder names against the live numbering

Done on 2026-10-23 (plain moves, not `git mv`, so the shared index was not
touched; git sees deletions plus untracked folders until they are staged):

| was | now | serves |
|---|---|---|
| `experiments/reference_period/` | `experiments/ch_08_setup/reference_period/` | `ch:setup:scenarios:reference` |
| `experiments/case1_svr/` | `experiments/ch_10_case1_svr/` | `ch:case_svr` |
| `experiments/part3_mc/` | `experiments/ch_11_case2_multisystem/mc/` | `ch:case3:montecarlo`, `ch:setup:montecarlo` |
| `experiments/ch_10_case_study/ch_10_N_*.py` | `experiments/ch_11_case2_multisystem/ladder/*.py` (prefixes dropped) | the superseded seven-rung ladder |
| — | `experiments/ch_12_case3_sts_support/` (README only) | `ch:case_sts` |

Still open:

| folder | serves | now | note |
|---|---|---:|---|
| `experiments/ch_9_parameter_selection/` | `ch:timescales` | **Ch 9** | rename to `ch_09_parameter_selection/` **deferred**: the Experiment B bank campaign on another machine (`results/THESIS_ch8_expB_banks_20261022/driver.py`) calls `ch_9_3_stage2_screen.py`, `ch_9_3_select.py`, `ch_9_1_ninner_isolated_sts.py` by path. Call sites to edit then: the folder's modules, `tuning/stage1_screen.py`, `pf/timescale_study.py`, `tests/experiments/test_ch9_*.py`, `configs/paramsets/ch9_selected.json`, `README.md`, `RUNBOOK_ch9.md` |
| `analysis/e1_interface_column_norms.py` | `ch:architectures:cascade:aggregation` | **Ch 5**, a section | leave in `analysis/` |

Stale path *comments* left on purpose in files the running campaign imports:
`configs/config.py:361` (`experiments/part3_mc/...`), `tuning/stage2_search.py:258`
(`experiments/ch_10_case_study/...`), `experiments/run_multi_system_ofo.py` (two
mentions; that file also carries uncommitted edits of another session). Fix them
with the deferred rename.

Dead label, fixed in code 2026-10-23: `ch:param:timescales` exists only in
`Chapters/archive/Chapter09_Parameterisation_SUPERSEDED.tex`. The citations in
`ch_9_parameter_selection/__init__.py`, `ch_9_1_timescale_seperation.py` and
`pf/probes/probe_modal_residue.py` now point at `ch:timescales`.

## The step battery is a historical instrument

Resolved 2026-10-23 with the thesis side. `tab:param:timescales:settling` held
the open-loop RMS step-battery results — max settling at W1 per actuator class
and output family — and was **retired with the step tests at the 2026-09-18
colloquium**, archived in `Chapters/archive/Ch9_sec1_step_tests_CUT.tex`. So
`ch_9_1_timescale_seperation.py` produces no live table, and neither does
`pf/probes/probe_modal_residue.py`, a read-only precursor probe. Both now say
so, naming their successor rather than claiming production.

The three live tables and what produces each:

| table | is | produced by |
|---|---|---|
| `tab:param:timescales:actuators` | the conceptual successor — longest settling per actuator class plus the mode that sets it | modal residue settling on the linearised RMS model at W1–W6, two droop states: `pf.modal_campaign <W>` → `pf.modal_window` → `analysis.modal_observability` → `analysis.build_modal_bundle` |
| `tab:param:timescales:ninner` | closed-loop subordinate settling, 120 interface set-point steps with the supervisory controllers held (Experiment C) | not the step battery; source data at `qOFO_2027/results/THESIS_ch8_D_settling_20260929/` |
| `tab:timescales:separations` | the chapter's overview of experiments A–D2 and what each fixes | not a data table |

**What this corrects in the code:** the package docstring claimed the battery
"fixes `T_DS` and `T_TS`". It does not. How they are actually fixed, all of it
verified against `Chapters/Chapter09.tex` on 2026-10-23:

- `T_TS` = **180 s by definition**, not measured — "the supervisory period is
  set by definition rather than derived from an experiment" (l. 218), chosen
  because three minutes lies inside the classical secondary-voltage-control
  range of tens of seconds to minutes (l. 225, `paul.1987`, `corsi.2004a`).
  The overview row at l. 33 reads "Supervisory period (by definition)".
- `T_STS` = **20 s from the modal bound** of Experiment A, `T_STS >= 17.7 s`
  (l. 168) — i.e. `tab:param:timescales:actuators`, the modal pipeline.
- `N_inner` = **9**, as `T_TS / T_STS` (`eq:param:timescales:ninner`, ll. 219–224).
  Experiment C only checks that the selected weights settle inside that budget
  (`n_s,95` = 23.05 ≤ 27, l. 433); it does not move the period.

(My first attempt to confirm this found nothing because the `\mathrm` patterns
broke in shell quoting — the lines were there all along. Worth remembering:
a grep that returns nothing against a file you have not opened is not evidence.)

The thesis names the code side at ll. 216–217: `tuning/parameters.py`
`TIMING_FIELDS`, with `refine_evented` candidates `tso_period_s` 180 and
`dso_period_s` 20. `TIMING_FIELDS` is confirmed at `tuning/parameters.py:117`,
so that link holds in both directions.

**Resolved: `N_inner` is derived, and the code comment was stale.** It was not
shorthand, as I had guessed — `tuning/parameters.py:115` was describing the
*superseded* chapter, where `Chapter09_Parameterisation_SUPERSEDED.tex:128` has
the multiplier "measured rather than assumed" and a `\todo` to measure it on
the isolated subordinate loop. The 2026-09-29 restructure
(`Chapter09.tex:195–200`) made it derived: "no experiment in this section.
`T_TS` is set by definition (3 min), `N_inner` = 9 follows, and the section
states the requirement the final weights must meet; Experiment C checks it."

That comment also **remapped the experiment letters** — "Former letters:
B + D → B, E → C, F1/F2 → D1/D2" — so the code's "Experiment B" pointed at the
weight selection, which measures nothing about `N_inner`. Experiment C
(`ch:param:experiment:e`, ll. 423–433) runs 120 ±10 Mvar interface steps with
the supervisory controllers held and checks `n_s,95 <= 27 = 3·N_inner` plus a
median residual ≤ 1 Mvar; it got 23.05 and 0.39 Mvar, and "the ratio
`N_inner` = 9, and thus `T_TS` = 180 s, is therefore kept". A check against a
budget, not a measurement of the budget.

Fixed in code 2026-10-23: the `parameters.py` comment, and in
`ch_9_1_ninner_isolated_sts.py` both the headline claim ("Measure `N_inner`
… thesis eq. (9.2)") and two retired "Experiment E" letters.

**And the answer to the question the thesis side could not see.** That script
computes `n_settle_final_1mvar` and `n_settle_censored` — i.e. it produces
`n_s`, not `N_inner`. So it *is* still the producer behind
`tab:param:timescales:ninner` and `fig:param:timescales:settling_cdf`; only
the interpretation changed, from measuring the multiplier to supplying the
settling counts checked against it. Its docstring now says so and cites both
labels. Note the data itself lives at
`qOFO_2027/results/THESIS_ch8_D_settling_20260929/` — in the other clone, and
untracked.

## The experiment-label trap

**The letters were remapped in 2026-09-29; the labels were not.** So the letter
inside a label is not the experiment's letter, and citing by letter gets it
wrong. Verified against `Chapter09.tex`:

| experiment | cite this label | what the label is attached to |
|---|---|---|
| A — modal analysis, `T_STS` = 20 s | `ch:param:experiment:a` | l. 128 |
| B — weight selection | `ch:param:experiment:d` | l. 260, `\paragraph{Experiment B: weight selection.}` |
| C — settling confirmation | `ch:param:experiment:e` | l. 423 |
| D1 | `ch:param:experiment:f1` | l. 479 |
| D2 | `ch:param:experiment:f2` | l. 517 |

The sharp edge is `ch:param:experiment:b`, which **does exist** and is
therefore not a dangling reference — it is attached to
`\paragraph{Analytical starting weights.}` at l. 264, a sub-paragraph *inside*
Experiment B. Citing `:b` for "Experiment B" lands in the right chapter one
level too deep, and compiles cleanly, so nothing flags it. There is no
`ch:param:experiment:c` at all.

Current code citations check out: `ch:param:experiment:e` in
`tuning/parameters.py:120`, `ch_9_1_ninner_isolated_sts.py:6` and this note
(all Experiment C), and `ch:param:experiment:d` in
`docs/tuning/2026-10-16_protocol_experiment_B_rerun.md` (Experiment B).

### Letter audit, 2026-10-23

Letters A, B and C in `experiments/ch_9_parameter_selection/` match the live
chapter and each claim holds against what that experiment now contains. Three
did not, fixed in `6b2e571`: three retired **F1/F2** references (the live
chapter uses them nowhere — they became D1/D2), and
`ch_9_1_actuator_location_sweep.py`, which opened with "open-loop settling
experiment A". That was wrong twice: A is now the modal analysis via a
different pipeline, and the open-loop settling battery the sweep belongs to was
retired at the 2026-09-18 colloquium. Whether the sweep still has a reported
role is the author's call, so the docstring says that rather than declaring it
dead.

## Hard-coded thesis numbers — the same problem as the folder names

**29 of them in `experiments/ch_9_parameter_selection/` alone:** `Table 9.1`
fourteen times, `eq. (9.1)` three, `eq. (9.2)` five, plus bare `§9.1` / `§9.2`.
Most are already stale — the settling table is retired and the `N_inner`
equation is no longer the second in the chapter — and all of them go stale
again at every renumbering, of which there have now been three.

Not fixed, because each site needs its own decision: some should cite a live
label, some describe retired material and should say so, and a few are
historical notes that are correct as written. The durable answer is the same as
for the folder names — **cite labels, not numbers**. Labels survived both
renumberings untouched; numbers have not survived any of them.

A concrete starting list, worst first: `ch_9_1_timescale_seperation.py` holds
twelve `Table 9.1` references and three `eq. (9.1)`, and it is the script whose
table is retired, so it is both the largest and the clearest case.

**Cross-repo point worth a decision.** `graphics/Ch9/data/2026-09-23_modal/README.md`
records that the modal bundle was produced **in `qOFO_2027` on the PowerFactory
machine**. If `qOFO` is meant to own that chain now, the four-stage pipeline
above and its outputs need a home decision; at present the chapter's actuator
settling table is produced from the other clone.

## The naming question

**This is the third renumbering** (2026-10-06, and the split on 2026-10-23).
Each one leaves every chapter-numbered folder name wrong, and the folder names
are what lets a reported number be traced back to its producer, so a stale one
is worse than no number at all.

Naming folders by **label or case** instead would stop the churn:
`ch_timescales/`, `case_svr/`, `case3/`, `case_sts/`, `setup/`. The labels are
already what the scripts carry in their comments, so the folder name and the
thesis reference would stay in lockstep without maintenance. The cost is a
one-off rename of the three existing folders and their READMEs.

**Decided by the author on 2026-10-23: chapter numbers, zero-padded**
(`ch_NN_<description>`), because a reader browses by chapter. The churn is
contained instead of avoided: every run README, manifest and thesis
`RESULT PROVENANCE` block carries the LaTeX label as the stable key, and
records the number only "as of" the run date (from the thesis `.aux`).
`python -m experiments.thesis_runs check` reports a block whose run or label
no longer resolves. A future renumbering therefore means renaming folders and
updating their header paths, not reconstructing what came from where.

## Two things a strict sort should not do

- **Do not move the `analysis/` modal scripts into a chapter folder.**
  `modal_reduce` is imported by a test and the other two are documented as
  `python -m analysis.X`; they are libraries and entry points. Point the
  chapter README at them instead.
- **Two scripts carry several chapter labels** — `ch_11_case2_multisystem/mc/run_mc.py`
  has three (setup *and* case 3) and `ch_9_1_timescale_seperation.py` two. One
  directory cannot express that; the script sits in its primary chapter and its
  run README names every label (`target` in `thesis_runs.start_thesis_run`).

## Also pending

`experiments/CIGRE_2026/005_CIGRE_MULTI.py` is the last file in that folder and
wants a better home, but moving it means editing four call sites — three are
`importlib.import_module("experiments.CIGRE_2026.005_CIGRE_MULTI")` in
`comparison_none_vs_sbxh.py`, `demonstrate_sbxv.py` and
`tests/sbx_v/smoke_sbxv_closed_loop.py`, and one is a by-path read in
`analysis/e1_interface_column_norms.py`. No static check flags those.
