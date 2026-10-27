# Experiments

Entry points, grouped by what they produce. `helpers/` and `runners/` are
shared implementation infrastructure, not entry points. `thesis_runs.py`
creates the documented run folders of the thesis chapters (below);
`results_io.py` creates the numbered run directories of the non-chapter entry
points; `paths.py` resolves the result roots.

## Thesis chapters

Since 2026-10-23 every thesis producer lives in a chapter folder
`ch_NN_<description>/[<sub-experiment>/]`, and `results/` mirrors that tree.
A producer calls `thesis_runs.start_thesis_run(__file__, target=<label>, ...)`,
which derives the results folder from the script's location and writes
`README.md`, `run_manifest.json` and the thesis's `RESULT PROVENANCE` block
(`latex_header.tex`) into the run. The LaTeX label is the stable key; the
number in the folder name is the chapter number of 2026-10-23. Layout,
archive rule and the thesis check: `results/README.md`.

Each chapter folder carries its own `README.md` mapping the scripts to thesis
labels. Start there rather than here.

| Folder | Label | Contents |
|---|---|---|
| `ch_08_setup/` | `ch:setup` | `reference_period/`: characterisation of the reference period |
| `ch_09_parameter_selection/` | `ch:timescales` | Experiments A–D2 (was `ch_9_parameter_selection/`, renamed 2026-10-27). Its `README.md` is the section map, `RUNBOOK_ch9.md` the procedure. The producers do not write documented runs yet; the weight-selection campaign of 2026-10-25 was adopted as run of record (`thesis_runs.adopt_run`). |
| `ch_10_case1_svr/` | `ch:case_svr` | `run_case1.py`, `report_case1.py`, `figures_case1.py` (was `case1_svr/`) |
| `ch_11_case2_multisystem/` | `ch:case3` | `mc/`: the Monte-Carlo campaign (was `part3_mc/`); `ladder/`: the superseded seven-rung ladder (was `ch_10_case_study/`) |
| `ch_12_case3_sts_support/` | `ch:case_sts` | no producer yet |

## Coordination mechanisms

- `run_multi_system_ofo.py` — manual multi-zone TSO-DSO OFO run. The usual
  starting point, and its `make_config_ch9_frozen()` is what analyses taking a
  config factory are pointed at.
- `comparison_none_vs_sbxh.py` — D0/D1/D2 comparison of no horizontal
  coordination, SBX-H contracts, and SBX-H planned support. The active TSO-TSO
  comparison is `none` versus SBX-H: physical tie-line reactive power stays
  measured and recorded, but is not a separate weight-based objective.
- `014_SBX_SINGLE_DEMO.py` — the 300-minute SBX-H v6 capability demonstration,
  the single thesis-oriented experiment for the retained horizontal ex-post
  remuneration mechanism.
- `017_SBX_PLANNING.py` — the SBX planning pre-pass (DACF/IDCF emulation).
- `demonstrate_sbxv.py` — stressed SBX-V vertical-band demonstrator with live
  metering, request, grant and remuneration visualisation.
- `018_SBXV_E1.py`, `019_SBXV_E2.py` — the SBX-V Phase-5 case studies: parity
  and economics (E1) and the band-width Pareto sweep (E2). `STATUS_SBXV.md`
  documents both as one-command re-runs.

## CIGRE 2026 paper

- `CIGRE_2026/005_CIGRE_MULTI.py` — the V1–V5 control-architecture ladder.
  **Kept although the rest of the folder was archived**, because it is a live
  config source rather than only a paper experiment: `comparison_none_vs_sbxh`,
  `demonstrate_sbxv` and `tests/sbx_v/smoke_sbxv_closed_loop` all load it with
  `importlib.import_module("experiments.CIGRE_2026.005_CIGRE_MULTI")`, and
  `analysis/e1_interface_column_norms.py` (the `ch:architectures` measurement)
  reads it by path. Moving or renaming it means editing those four call sites,
  three of which are dynamic imports that no static check will flag.

## PowerFactory RMS co-simulation

Needs a PowerFactory seat. `pf/` holds the adapter.

- `run_rms_cosim.py` — one closed loop against the RMS plant.
- `run_comparison_rms_cosim_qss.py` — runs the loop twice, once per plant, and
  compares. This is the driver the dead-band sweeps go through.
- `run_openloop_qss_to_rms.py` — the genuine replay: applies a quasi-static
  run's recorded actuator timeline to the RMS plant.
- `postprocess_rms_cosim.py` — trajectory extraction from a finished run.

The PowerShell launchers that drive these are no longer in the tree. The four
§9.2 dead-band producers — `run_qstep_sweep.ps1`,
`run_qstep_falseactivation.ps1`, `run_e1_drift.ps1`, `run_e3_tracking.ps1` —
are parked in the untracked `local_ps_scripts/`, with a README there recording
what each writes. They are chapter producers, not scratch; they are out of the
tree because they are machine-, path- and account-specific, not because they
are disposable.

## Switched shunts

- `shunt_dispatch_check.py` — the `off` versus `miqp` bank-dispatch check.

## Not here any more

The `archived/` folder is gone (2026-10-22). Its pre-refactor entry points
(001–004b) and the `run_rms_phase6_replay` deprecation shim were deleted; the
four current experiments that had been misfiled in it — `014`, `017`, `018`,
`019` — are listed above, where they belong.

Most of `CIGRE_2026/` moved to `_local_archive/experiments/CIGRE_2026/` on
2026-10-22 — untracked, outside the project view, still on disk and in git
history. Fourteen files:

- the `007*` series, eleven scripts measuring the tie-line boundary equivalent
  and the Thévenin sweep. Its conclusion is implemented as the `tie_boundary`
  keyword of `sensitivity/network_reduction.py::build_tso_local_net` and its
  reasoning is in
  `docs/daily_log/08_2026/2026-08-11_tie_boundary_equivalent.md`. If the
  boundary model is ever revisited, those scripts are the re-measurement
  instrument.
- `006_CIGRE_MONTECARLO.py`, the paired Monte-Carlo extension of `005`.
- `008_PILOT_BUS_SELECT.py`, the pilot-bus generator. Its *output* is still
  load-bearing — the pilot buses are pasted into `005` and into `configs/` as
  fixed constants, and `runners/multi_tso_dso.py` names it when they are
  missing — but nothing imports it, so it is a generator you reach for when the
  partition changes, not code on any path.
- `009_S1_VS_O1.py`, the paper's `S1` (classical pilot-node SVR) against `O1`
  (TS-OFO on the continuous actuators only). The thesis makes its own
  comparison in `ch_10_case1_svr/` (and, superseded, in
  `ch_11_case2_multisystem/ladder/`).

None of the four carries a `ch:` label, which is the evidence that they are the
paper's rather than the dissertation's. `005` does not either, but it is kept
for the import reason above.

One open point from that move: the thesis write-up handoff
(`docs/handoff/2026-08-17_thesis_writeup_handoff.md` §8) states the boundary
model as Thévenin, while `tie_boundary` defaults to `"pq"`. Worth reconciling
before that section is written.
