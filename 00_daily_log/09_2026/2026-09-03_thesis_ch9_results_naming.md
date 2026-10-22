# 2026-09-03 — Chapter 9 result folders renamed to the `THESIS_ch<N>_*` convention

**Timestamp:** 2026-09-03
**Scope:** `results/` folder naming, plus the source path literals that point
into it. No experiment was re-run and no result data was modified, moved out of
its folder, or deleted.

**Reason:** a parallel session established the convention that result folders
whose contents are reported in the dissertation carry a `THESIS_ch<chapter>_*`
prefix (`results/THESIS_ch10_variants_single_run`). Chapter 9 evidence did not
yet follow it, so "is this folder in the thesis?" had no answer from the name.

## Which chapter, and which folders

Chapter 9 is *Parameterisation of the Control Hierarchy* (§9.1 timescales,
§9.2 dead band, §9.3 weights, §9.4 shunt threshold), per
`experiments/ch_9_parameter_selection/README.md` and `Chapters/Chapter09.tex`
(`ch:param:*`). The `docs/ch8_deadband_selection.tex` filename predates the
2026-07-26 restructure (`docs/thesis_restructure_handover.md`) and is **not**
evidence that the dead band sits in Chapter 8 — `ch:param:deadband` is a
Chapter 9 label, and `results/deadband_droop/README.md` says so explicitly.

## Renames (six, all in `results/`)

| before | after | § |
|---|---|---|
| `timescale` | `THESIS_ch9_1_timescale_settling` | 9.1 |
| `ch9_ninner` | `THESIS_ch9_1_ninner` | 9.1 |
| `ch9_ts_period_sweep` | `THESIS_ch9_1_ts_period_sweep` | 9.1 |
| `deadband_droop` | `THESIS_ch9_2_deadband_droop` | 9.2 |
| `deadband_droop_e1_drift` | `THESIS_ch9_2_deadband_droop_e1_drift` | 9.2 |
| `deadband_droop_e3_tracking` | `THESIS_ch9_2_deadband_droop_e3_tracking` | 9.2 |

Deliberately **not** renamed:

- `results/deadband_selection/` — the older profiled sweep, chatter-knee study
  and threshold-by-load-step measurement. `deadband_droop/README.md` states
  none of it is reported in the thesis.
- `results/tuning/` — the Bayesian-optimisation campaign of Appendix D, whose
  own header records that the campaign has not been run.
- `results/tuning_mc/` — §9.3's evidence, but read at runtime by
  `_ch9_selected_design.py`, which every §9.1 closed-loop experiment imports,
  and the selected campaign is one of five siblings under it. Author's decision
  (2026-09-03): leave it and record the location in a pointer file,
  `results/THESIS_ch9_3_weight_selection_POINTER.md`. Verified after the
  renaming that `_ch9_selected_design.archived_evaluation()` still resolves —
  `stage1` / `fe010aa3ead1`, `rho_emp_p95` 1.378849148072478.

Author's decision on granularity (2026-09-03): rename the **whole label
folder** in place, keeping probes and superseded runs inside it, rather than
extracting only the run of record. This keeps each `<label>/_latest.txt` and
the relative provenance intact; which run is quoted is stated in each folder's
new `README.md` instead.

## Source edits

Path literals rewritten (sources only — these will be re-run):

- `experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py`
  (docstring, `--out-dir` help, `label_dir`), `ch_9_1_actuator_location_sweep.py`
  (`label_dir`), `ch_9_1_ninner_isolated_sts.py` and `ch_9_1_ts_period_sweep.py`
  (argparse defaults), and the folder `README.md`
- `experiments/run_e1_drift.ps1`, `experiments/run_e3_tracking.ps1`
- every `.py` inside the three renamed dead-band folders (their analysis and
  figure scripts carry absolute `Z:\...\results\...` literals)
- the `Source run:` pointers in `docs/data/2026-08-25_ch9_actuator_location_by_output_family/README.md`
  and `docs/data/2026-08-21_ch9_ninner/README.md`

**Not** rewritten, on purpose: `docs/daily_log/**`, `docs/handoff/**`, and the
`run_meta.json` / `script_snapshot.py` / `*.tex` / `*.csv` artefacts inside the
archived runs. Those are frozen records of what happened at the time; rewriting
them would falsify the provenance they exist to carry. The generated `.tex`
files also feed the thesis directly.

Pre-existing staleness left as it was: `MANIFEST_runs.csv` in the dead-band
folder has a `location` column naming a `runs/` subfolder that no longer exists
(the runs are in `data/`). The rename does not make this worse; it is now noted
in that folder's README.

## New provenance READMEs

`README.md` written into each of `THESIS_ch9_1_timescale_settling`,
`THESIS_ch9_1_ninner`, `THESIS_ch9_1_ts_period_sweep`, in the style of
`THESIS_ch10_variants_single_run/README.md`: run of record, the superseded and
diagnostic labels tabulated with why, and the caveats the run carries. A header
note was prepended to the three dead-band READMEs, which already documented
their studies.

## Finding surfaced by the exercise

**The §9.1 B1/B2 runs of record are not in `results/`.** The final Chapter 9
`N_inner` and `T_TS`-sweep runs — `20260821-180548` and `20260821-183927`, the
ones carrying the post-selection DSO-4 safety correction — exist only as
complete run directories under
`docs/data/2026-08-21_ch9_b1_b2_corrected/`. `results/THESIS_ch9_1_ninner/` and
`results/THESIS_ch9_1_ts_period_sweep/` hold only their superseded lineage
(latest: `stepcap_10mvar_balanced_half_corrected/20260821-134412` and
`final_weights/20260821-092509`). Both new READMEs open with this. Whether
those two folders should keep the `THESIS_` prefix on that basis is an open
question for the author.

## Verification

- `py_compile` clean on all four §9.1 entry points and on every `.py` in the
  three renamed dead-band folders
- `ch_9_1_timescale_seperation.py --self-test` → ALL PASS (no PowerFactory seat
  needed)
- `_ch9_selected_design` imports and reads its archive unchanged
- `grep` over `experiments/ analysis/ visualisation/ tools/ tests/ tuning_mc/
  core/ controller/ pf/ configs/` and over the renamed folders themselves finds
  no live source still naming an old path

## Risks / unresolved

- Any *external* consumer of these paths — the thesis LaTeX tree
  (`latex_diss_ms`, not reachable from this machine) and anything on the server
  — was not checked. `docs/ch8_deadband_selection.tex` still points its
  `\graphicspath` at `results/deadband_selection/` and `results/deadband_n1/`;
  the first is untouched by this change and the second did not exist before it.
- `results/` is gitignored, so the renames are not recorded in git history. This
  log is the only record.
