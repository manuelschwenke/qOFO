# 2026-06-15 — Codebase cleanup: drop GFM/GFL classification, tidy memory/, hoist run configs

**Timestamp:** 2026-06-15 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
Four cleanup items requested by the user:
1. The **grid-forming / grid-following (GFM/GFL) classification** was introduced,
   then abandoned in favour of modelling *all* DER uniformly with a local Q(V)
   droop (the `tag_der_q_modes` path). Dead artifacts remained throughout the
   control stack.
2. The **stability assessment** is not used in day-to-day runs; confirm it is
   dormant and lives in `analysis/`.
3. A stray repo-root **`memory/`** folder (duplicate of a Claude auto-memory).
4. Each **experiment run-script** should expose its run config at the top of the
   file for fast editing.

## Task 1 — Removed the GFM/GFL classification (full strip)

Two layers were removed.

### (A) Dead classification layer (only ever exercised by tests)
- **Deleted** `core/der_classification.py` (`DERMode`, `DERClassification`).
- `network/ieee39/build.py`: removed `apply_der_classification()` (~258 lines, the
  build-time sgen→`pp.gen` promotion) and its imports (`DERClassification`,
  `DERMode`, `dataclasses.replace`). The live build path is
  `build_ieee39_net` + `add_hv_networks` + `tag_der_q_modes` (uniform Q(V)); the
  promotion entry point was called only from `tests/test_tag_der_q_modes.py`.
- `network/ieee39/meta.py`: removed the `der_classification` field and the four
  `tso/dso_grid_forming_gen_indices/buses` registries (plus the now-unused
  `Optional`/`DERClassification` imports).
- `network/ieee39/__init__.py`: dropped the `apply_der_classification` export.
- `tests/test_tag_der_q_modes.py`: removed the two `apply_der_classification`
  coexistence tests; all `tag_der_q_modes` tests kept.
- Stale comment/docstring references to `apply_der_classification` reworded in
  `core/profiles.py` and `network/ieee39/zonal_balancing.py` (the generic
  `profile`-column guard for exogenous wind/PV gens is kept).

### (B) The `gridforming_gen_indices` GF actuator block (V_gf / Q_gf)
This block was woven through the controllers' decision-variable index math but was
**always empty at runtime** (`n_gf == 0`: it was never wired from `meta` into any
zone/controller config), so removing it is numerically a no-op. Key structural
change: the TSO control/output layout collapses from

```
controls: [Q_DER | Q_PCC | V_gen | V_gf | OLTC | shunt]  ->  [Q_DER | Q_PCC | V_gen | OLTC | shunt]
outputs:  [V | Q_PCC | I | Q_gen | Q_gf | Q_tie]         ->  [V | Q_PCC | I | Q_gen | Q_tie]
```
and the DSO layout from `[Q_DER | V_gf | OLTC | shunt]` -> `[Q_DER | OLTC | shunt]`.

Files edited (config fields, `n_gf` terms in every `n_controls`/`n_outputs`/
`q_obj_diagonal`/`gw_diagonal`, bounds, `_build_sensitivity_matrix` V_gf-column /
Q_gf-row fills, and cross-zone `H_ij` offsets):
- `controller/tso_controller.py` (TSOControllerConfig GF fields + `g_w`/`rho`/`vm`
  knobs, validation, `_actuator_class_indices` `"tso_grid_forming"` class,
  `_build_sensitivity_matrix`).
- `controller/dso_controller.py` (config GF fields, `_actuator_class_indices`
  `"dso_grid_forming"`, and the whole `_splice_gridforming_into_H` helper).
- `controller/multi_tso_coordinator.py` (ZoneDefinition GF fields + `g_w_gridforming`,
  the four layout methods, cross-zone `H_ij` column/row offsets).
- `core/measurement.py` (`gridforming_gen_*` kwargs/attrs + the `gf_indices`
  extraction in all four `measure_*` builders).
- `experiments/helpers/plant_io.py` (V_gf write in TSO/DSO/central apply).
- `experiments/helpers/records.py` (`zone_v_gf` field) + its writer in
  `experiments/runners/multi_tso_dso.py`.
- `visualisation/plot_tso_controller.py` (removed the gated `_has_gridforming`
  alternative figure layout, `_zone_v_der`, `n_gridforming_per_zone`, `_redraw_v_der`).
- `tuning/parameters.py` (removed the `g_w_gridforming` BOParam — a vacuous
  coordinate once `n_gf==0`).
- `sensitivity/numerical_h.py` (removed the V_gf/Q_gf blocks from the
  finite-difference H builder, keeping it dimension-consistent).
- `controller/central_controller.py`, `experiments/003_S_DSO_CIGRE_2026.py`:
  stale layout-doc comments referencing `V_gf`/`Q_gf` updated.

## Task 2 — Stability assessment: no code change
Confirmed already dormant: `run_stability_analysis=False` in every experiment
script and all stability code (`analysis/reachability.py`,
`analysis/stability_analysis.py`, `analysis/observer/`) already lives in
`analysis/`. Per the user's choice the nose-curve reachability guard
(`enable_reachability_guard=True`) is **left as-is**.

## Task 3 — Deleted the stray repo-root `memory/` folder
Removed `memory/project_006_cigre_montecarlo.md` and the empty `memory/` dir
(a duplicate of the Claude auto-memory under `.claude/...`, which is untouched).

## Task 4 — Hoisted run config to the top of the experiment scripts
- `experiments/000_M_TSO_M_DSO.py` and `experiments/001_S_TSO_S_DSO.py`: extracted
  the run config out of `main()` into a top-level `make_config()` placed right
  after the imports; `main()` now calls it. (000's `main_comparison()` keeps its
  own paired config; 001's dead local ramp vars were removed.)
- `002`, `003`, `004`, `004b` (`make_base_config()`) and `005`, `006`
  (`make_cigre_config()` + user-facing knobs) already defined their config right
  after the imports — left as-is.

## Verification
- `pytest tests/` — **all non-tuning tests pass** (396 passed, 11 skipped). The
  `tests/tuning/` failures are **pre-existing breakage at HEAD** (confirmed: the
  committed `tuning/parameters.py` produces *more* failures, 19 vs 15; the tuning
  module is out of sync with `MultiTSOConfig`, unrelated to this cleanup).
- The riskiest edits (H-matrix offset arithmetic) are directly covered by
  `test_h_matrix_qcor_transform`, `test_controller`, `test_sensitivity_updater`,
  `test_g_w_adapter` — all green.
- End-to-end smoke: a short `run_multi_tso_dso` (3 TSO steps, full multi-zone
  coordinator + 4 DSOs) ran clean, voltages converging, Q-tracking active,
  reachability guard non-aborting.
- All touched modules + all `experiments/0*.py` byte-compile and import.

---

## 2026-06-16 follow-up — code-quality pass (experiment scripts + runner)

Safe, behaviour-preserving cleanups (no logic changes; verified by `py_compile`,
runner import, `test_controller`/`test_tag_der_q_modes`, and a verbose=2 smoke run).

**Removed unused imports:**
- `experiments/002_M_TSO_M_DSO_COMPARE.py`: `deepcopy`.
- `experiments/004b_REFRESH_PROOF.py`: `Tuple`.
- `experiments/helpers/plant_io.py`: pandapower-internal `_detect_read_write_flag`.
- `experiments/runners/multi_tso_dso.py`: `analyse_multi_zone_stability`,
  `MultiZoneStabilityResult`, `NetworkState`, `write_stability_analysis_markdown`,
  `write_tuned_params_json`, `IEEE39NetworkMeta`, `remove_generators`,
  `ContingencyEvent`, `install_qv_local_loops`, and a dead local `import math as _math`.
  (Verified none are re-exported / monkey-patched externally; the `_multi_tso_helpers`
  imports are **kept** — the comment there marks them as a deliberate patch surface.)

**Removed dead local variables (runner):** `sgens`/`res_sg` (Q-tracking diagnostic),
`contraction_info` (kept the `check_contraction()` call for its side effect/timing),
`stab_result` (init + capture; kept the `_run_delayed_stability_analysis()` call).

**Flagged for the user (not changed — possible latent bugs / half-built code):**
- `004b_REFRESH_PROOF.py:284-285`: `v_avg_steady`/`q_avg_steady` computed but never
  printed; the "Steady-state second-half averages" table header has no data rows.
- `multi_tso_dso.py` (PF-retry path): `pf_converged = True` set but never read.

### Further fixes (user-approved a/b/c)
- **(a) Fixed the 003 latent bug** — `003_S_DSO_CIGRE_2026.py:204-205` now sets
  `cfg.start_time = datetime(2016, 9, 7, 8, 0)` and `cfg.use_profiles = True` (was
  assigning dead locals). **Behaviour change:** 003's `start_time` moves from 002's
  inherited 2016-01-05 to the intended 2016-09-07.
- **(b) De-duplicated 004/004b configs** (like 003 already does):
  - `004_LOCAL_VS_FULL_SENS.make_base_config()` now `= make_002_base_config()` + 6
    explicit overrides (`local_sensitivities_tso/dso=False`, `g_w_pcc=50`,
    `live_plot_controller/cascade=False`, its own 4-event `contingencies`).
  - `004b_REFRESH_PROOF.make_base_config()` now `= make_004_base_config()` +
    `{n_total_s=14400, contingencies=[]}`.
  - **Verified byte-identical**: a field-level diff of the rebuilt configs vs. 002/004
    shows exactly the same delta sets as before the refactor (004↔002: 6 fields;
    004b↔004: 2 fields), so behaviour is preserved.
  - Note: 004 keeps `g_w_pcc=50` while 002's C uses 10 — preserved for parity with
    prior 004 runs (possible historical drift; flagged for review).
- **(c) Reconciled stale config-value comments in 000** `make_config()`:
  `# 300-min` → `# 36-hour (2160-min)` (value is `60*60*36`), `# every 6 min` →
  `# every 3 min`, and `# Live plots OFF for the batch sweep` → an accurate note
  (controller+cascade plots are on). Comments only — no behaviour change. (The 36-hour
  horizon itself is much longer than the old "300-min" comment implied — flagged in
  case the *value* is a leftover.)
