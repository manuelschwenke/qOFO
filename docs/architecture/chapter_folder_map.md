# Chapter folders vs the dissertation — what to change

Written 2026-10-23, **not applied**: the chapter structure is about to be
rearranged, so renaming now would only have to be redone. This is the
to-do list for afterwards.

Authoritative source: the annotated `\include` list in
`ClassicThesis.tex` (lines 336–385) at
`C:\Users\Manuel Schwenke\Desktop\Daten\01_Forschung\12_Dissertation\latex_diss_ms`.

## The live map

| Ch | Label | Title (short) |
|---:|---|---|
| 5 | `ch:droop` | Multi-System OFO and the Local Control Layer |
| 6 | `ch:regframe` | Regulatory Frame (proposal) |
| 7 | `ch:setup` | Benchmark Networks, Reference Models, Evaluation Methodology |
| 8 | `ch:timescales` | Timescale Separation of the Control Layers |
| 9 | `ch:case_svr` | Case Study 1 — OFO in place of pilot-node SVR |
| 10 | `ch:case3` | Case Study 2 — multi-system OFO, three areas |
| 11 | `ch:case_sts` | Case Study 3 — Voltage Support from Sub-Transmission Systems |

## What is stale in the repo

The 2026-10-06 renumbering moved chapters under the folder names.

| folder | carries | actually | action |
|---|---|---|---|
| `experiments/ch_9_parameter_selection/` | `ch:param:timescales` | **Ch 8** — and that label lives in `Chapters/archive/Chapter09_Parameterisation_SUPERSEDED.tex` | rename to ch 8; the name "parameter_selection" is the chapter's *old* title |
| `experiments/case1_svr/` | `ch:case_svr` | **Ch 9** | rename; "case1" is the case number, not the chapter |
| `experiments/ch_10_case_study/` | `ch:case3` | **Ch 10** | correct already |
| — | `ch:case_sts` | **Ch 11** | no folder yet; the switched-shunt / STS support work belongs here |
| `experiments/reference_period/`, `experiments/part3_mc/` | `ch:setup:*` | **Ch 7** | fold in or cross-reference |
| `analysis/e1_interface_column_norms.py` | `ch:architectures:cascade:aggregation` | **Ch 5**, a section | leave in `analysis/`; fix the comment, which implies a chapter of its own |

## Cross-references pointing at cut material

Three labels the code cites now live in `Chapters/archive/`:

- `ch:case_rms` → `Chapter11_RMS_Verification_CUT.tex`, cited by
  `experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py`
- `ch:timescales:plant` → `Ch9_sec1_step_tests_CUT.tex`, cited by
  `analysis/build_modal_bundle.py`, `modal_observability.py`, `modal_reduce.py`
- and `Ch9_sec5_rms_deadband_evidence_CUT.tex` is cut too

Ch 8's include comment says "step tests retired; **F1/F2 on the QSS plant**",
and Ch 11 mentions neither RMS nor PowerFactory. So the thesis has moved off
the RMS path in the places that used to need it.

**Consequence worth deciding separately:** `pf/` and `export/` were removed on
2026-10-22 and restored on the argument that chapter 9 needed them. Under the
live structure that argument is weaker than it looked — the RMS verification
chapter is cut. Ch 7 still mentions PowerFactory four times, so it is not
simply dead. Re-examine when the rearrangement settles rather than acting on
it now.

## Two things a strict sort should *not* do

- **Do not move the `analysis/` modal scripts into a chapter folder.**
  `modal_reduce` is imported by a test and the other two are documented as
  `python -m analysis.X`; they are libraries and entry points, not chapter
  scripts. Point the chapter README at them instead.
- **Two scripts carry several chapter labels** — `part3_mc/run_mc.py` has three
  (setup *and* case 3) and `ch_9_1_timescale_seperation.py` two. One directory
  cannot express that; the README cross-reference is the honest home.

## Also pending

`experiments/CIGRE_2026/005_CIGRE_MULTI.py` is the last file in that folder.
Moving or renaming it means editing four call sites, three of which are
`importlib.import_module("experiments.CIGRE_2026.005_CIGRE_MULTI")` in
`comparison_none_vs_sbxh`, `demonstrate_sbxv` and `tests/sbx_v`, plus a
by-path read in `analysis/e1_interface_column_norms.py`. No static check will
flag those.
