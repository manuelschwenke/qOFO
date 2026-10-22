# 2026-06-25 — Monte-Carlo: re-run a single variant in place (`--rerun-variant`)

**File:** `experiments/006_CIGRE_MONTECARLO.py`
**Author:** Manuel Schwenke / Claude Code

## Motivation
After a full Monte-Carlo batch (all variants V1–V5 simulated on N paired
scenarios — expensive), only the **parameters of one variant** (here V5,
retuned via `VARIANTS["V5"]` / `KAPPA_V5`) changed. Re-running every variant on
every scenario just to refresh V5 wastes the (unchanged) V1–V4 results. We want
to re-simulate **only V5** on the *identical* scenarios V1–V4 already saw, merge
it back in, and rebuild the tables/figures.

## What was changed

### 1. New in-place re-run path
- `_events_from_rows(rows)` — reconstructs the contingency `schedule` (list of
  `ContingencyEvent`) from the per-seed JSON `schedule` rows produced by
  `schedule_to_rows`. Uses the **stored** schedule (the exact one V1–V4 ran on),
  not a regeneration from the seed, so the paired comparison is preserved even if
  schedule-generation constants ever change.
- `_rerun_one_scenario_variant(seed, variant, capture)` — module-level
  (picklable for `ProcessPoolExecutor`). Reloads `(start_time, schedule)` from
  `RUNS_DIR/scenario_<seed>.json`, runs **only** `variant` with the current
  `VARIANTS[variant]` parameters, then overwrites that variant's slice:
  - JSON: drops the old `variant` metric row, appends the new one (if it
    converged), re-sorts by `VARIANT_ORDER`, and records a `rerun` audit entry.
  - `ts_<seed>.npz`: drops every key whose variant component equals `variant`
    (keys are `"<kind>__<variant>__…"`; `t_min` has no `__` and is always kept),
    then writes the new `vrms__/P__/Q__/Vz__` arrays.
  - All other variants are left byte-identical.
- `rerun_variant_only(variant, n_runs, jobs, backup)` — orchestrator. Selects the
  same set `collect_and_finalize` keeps (first `n_runs` accepted scenarios by
  ascending seed, or all when `n_runs<=0`), optionally backs up the staging,
  re-runs the variant serially or in parallel, summarises, then calls
  `collect_and_finalize` to rebuild master CSVs / canonical timeseries / Table 3
  / figures.

### 2. Divergence policy
A scenario was accepted because **all** variants (incl. the *old* variant
params) converged. If the *new* params diverge on a scenario, the scenario is
**kept** (V1–V4 untouched) but the variant is **dropped** for that run — rather
than mixing two parameter sets in one column or discarding the V1–V4 baseline.
The variant then reports n ≤ N runs; the dropped count is summarised. Aggregation
already tolerates per-variant absence (`aggregate_table3`, all figures).

### 3. Safety backup
By default, the per-seed `scenario_<seed>.json` + `ts_<seed>.npz` of the affected
runs are copied to `results/006_cigre_mc/_runs_backup_<variant>_<timestamp>/`
before being overwritten (disable with `--no-backup`). Lets the old variant be
restored.

### 4. CLI
- `--rerun-variant V` — re-simulate only variant `V` (e.g. `V5`) on the existing
  accepted scenarios, then rebuild outputs. Requires the per-seed staging
  (`RUNS_DIR`) from a prior full batch.
- `--no-backup` — skip the safety copy.
- Honours `--runs` (which N to keep) and `--jobs` (parallel workers), mirroring
  `--replot` semantics.

### 5. Incidental bug fix (blocked import)
The `KAPPA_V5` scaling loop (`for _k in _V5_GW_KEYS: VARIANTS["V5"][_k] *= …`)
crashed with `KeyError: 'g_w_der'` because the explicit `g_w_*` overrides in
`VARIANTS["V5"]` had been commented out during tuning. Guarded the loop with
`if _k in VARIANTS["V5"]` so KAPPA only scales keys V5 sets explicitly (when a
key is commented out, V5 inherits it from `make_cigre_config()` and KAPPA does
not apply — consistent with `run_variant`, which only `setattr`s keys present in
`VARIANTS["V5"]`). Without this the whole script failed to import.

## Usage
```bash
# 1) edit VARIANTS["V5"] / KAPPA_V5 in the file to the new parameters
# 2) re-run only V5 over the existing accepted scenarios (parallel), re-merge:
python experiments/006_CIGRE_MONTECARLO.py --rerun-variant V5 --runs 100 --jobs 6
# (add --no-backup to skip the staging backup; use --jobs 1 for serial)
```

## Verification (non-destructive, no simulation)
- `py_compile` passes; `--help` shows the new flags.
- `_events_from_rows` round-trips the stored schedule of `scenario_20260604`
  exactly (4 events, correct types/buses/P/Q).
- The npz key-filter drops exactly the 7 `V5` arrays and keeps `t_min` + all
  V1–V4 arrays (no V5 residue).
- The full V5 re-simulation was **not** executed here (expensive; would mutate
  the existing staging) — to be launched by the user on the real dataset.

## Risks / open points
- Assumes the per-seed staging `RUNS_DIR/scenario_*.json` + `ts_*.npz` from the
  original batch is intact. If only the canonical `timeseries/` + master CSVs
  survive (staging cleared), this path raises a clear error rather than guessing
  — those artefacts also lack `start_time` for zero-event scenarios, so staging
  is the required source.
- If many scenarios that previously accepted now diverge under the new V5
  params, V5's sample size drops below N and the V5 column is no longer fully
  paired with V1–V4 (reported in the summary). Consider a fresh full batch if
  the new V5 is substantially less robust.
