# Chapter folders vs the dissertation — what to change

Updated 2026-10-23 after the chapter split, and after three corrections from
the thesis side. Still **not applied** to the folder names: see the naming
question at the end, which is worth settling before a fourth rename.

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

| folder | serves | now | note |
|---|---|---:|---|
| `experiments/ch_9_parameter_selection/` | `ch:timescales` | **Ch 9** | name is the chapter's *old* title, "Parameterisation of the Control Hierarchy" |
| `experiments/case1_svr/` | `ch:case_svr` | **Ch 10** | "case1" is the case number, not the chapter |
| `experiments/ch_10_case_study/` | `ch:case3` | **Ch 11** | number now wrong |
| `experiments/reference_period/`, `experiments/part3_mc/` | `ch:setup:*` | **Ch 8** | |
| `analysis/e1_interface_column_norms.py` | `ch:architectures:cascade:aggregation` | **Ch 5**, a section | leave in `analysis/` |
| — | `ch:case_sts` | **Ch 12** | no folder; the switched-shunt / STS work belongs to it |

Dead label, fixed in code 2026-10-23: `ch:param:timescales` exists only in
`Chapters/archive/Chapter09_Parameterisation_SUPERSEDED.tex`. The citations in
`ch_9_parameter_selection/__init__.py`, `ch_9_1_timescale_seperation.py` and
`pf/probes/probe_modal_residue.py` now point at `ch:timescales`. The table
label `tab:param:timescales:settling` is dead for the same reason; the live
chapter carries `tab:param:timescales:actuators`,
`tab:param:timescales:ninner` and `tab:timescales:separations`, and which is
the successor is the author's call, so no table label is named in the code
rather than naming the wrong one.

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

Not done, because it is a decision rather than a repair.

## Two things a strict sort should not do

- **Do not move the `analysis/` modal scripts into a chapter folder.**
  `modal_reduce` is imported by a test and the other two are documented as
  `python -m analysis.X`; they are libraries and entry points. Point the
  chapter README at them instead.
- **Two scripts carry several chapter labels** — `part3_mc/run_mc.py` has three
  (setup *and* case 3) and `ch_9_1_timescale_seperation.py` two. One directory
  cannot express that; the README cross-reference is the honest home.

## Also pending

`experiments/CIGRE_2026/005_CIGRE_MULTI.py` is the last file in that folder and
wants a better home, but moving it means editing four call sites — three are
`importlib.import_module("experiments.CIGRE_2026.005_CIGRE_MULTI")` in
`comparison_none_vs_sbxh.py`, `demonstrate_sbxv.py` and
`tests/sbx_v/smoke_sbxv_closed_loop.py`, and one is a by-path read in
`analysis/e1_interface_column_norms.py`. No static check flags those.
