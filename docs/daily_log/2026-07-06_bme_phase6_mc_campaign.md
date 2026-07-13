# 2026-07-06 — BME Phase 6 item 6: H-error axis + Monte-Carlo campaign (012)

**Context.** Continuation after a machine restart on 2026-07-05 that
interrupted the session between MC implementation and validation/launch.
Recovered working-tree state: H-error machinery + D2 edge recalibration
implemented but uncommitted, `experiments/012_BME_MONTECARLO.py` written,
no campaign results staged. This session: validation, one bug fix, launch.

**Authorisation (Manuel, 2026-07-05, recorded in the 012 docstring):**
recalibrated D2 edges (1.02, 1.04); H-error scoped to the H_{b,i} slice;
full ~75-run design; run autonomously.

## What was changed (carried over from 2026-07-05, validated today)

1. **D2 edge recalibration** (`experiments/011_BME_LADDER.py`):
   `BME_V_SOFT_MIN/MAX` (1.01, 1.05) → **(1.02, 1.04)**, w_band = 1e4
   unchanged. Reason: the §6e finding — under the wide band the
   coordinated rungs rode the upper edge for loss harvest and the
   realised hinge cost exceeded the in-scope loss gain (last-hour Φ
   ranking inverted none < bme < oracle). The ±0.01 pu corridor prices
   edge-riding immediately; φ_band returns to a genuine security margin.
   This closes the 6e "calibration-philosophy" question (Manuel's call,
   not silently retuned).

2. **H-error axis** (§6 MC sensitivity-error sweep):
   - `configs/multi_tso_config.py`: `bme_h_error_rel_sigma` (default 0.0
     = bitwise no-op), `bme_h_error_seed` (required when σ > 0,
     fail-fast in the runner's bme setup block).
   - `sensitivity/boundary_sensitivity.py`: `PerturbedZoneBoundaryView`
     — duck-typed `ZoneBoundaryView` stand-in serving
     H̃_{b,i} = H_{b,i} ∘ (1 + σ·Ξ_z), Ξ_z ~ N(0,1) drawn ONCE per run
     per zone (systematic identification error, not per-tick noise);
     magnitude channel `h_b()` perturbed consistently (top |B| rows of
     the same field).
   - `experiments/runners/multi_tso_dso.py`: per-zone error fields
     cached over the run (`seed + zone_id` streams); the SAME wrapped
     view feeds both the price projection H̃ᵀ·μ_total and the
     switch-notice prediction dv_b_pred = H̃_d·Δu_d — the zone's
     (possibly mis-identified) H is one object. Scope deliberately
     EXCLUDES the zones' own MIQP models (experiment 004's trade-off)
     and the μ computation (zone-internal model): the axis isolates the
     robustness of the coordination channel to error in the one
     supra-local quantity a TSO must identify from boundary
     measurements. Metric objective and plant stay exact.
   - `tests/test_bme_h_error.py`: wrapper contract (fixed field,
     elementwise product, consistent channels, shape fail-fast) +
     config-default contract.

3. **MC campaign script** (`experiments/012_BME_MONTECARLO.py`, new):
   - Paired scenarios: seed → deterministic (profile-year start time,
     lightly-constrained contingency schedule), both reusing 006's
     generator (`random_start_time`, `enumerate_elements`,
     `build_random_schedule`); all arms of a scenario share it, paired
     differences remove scenario variance.
   - Phase A: N_BASE = 10 scenarios × {none, bme nominal, oracle},
     drop-and-replace acceptance (all three must converge).
   - Phase B: one-factor sweeps around the bme nominal (d=1, β=0.3,
     drop=0, ε=5.2e3, σ_H=0) on the first 3 accepted seeds:
     d ∈ {0,2,5}, β ∈ {0.1,0.6,1.0}, drop ∈ {0.05,0.2},
     ε ∈ {0,1e3,2.6e4}, σ_H ∈ {0.05,0.15,0.3}, plus the selfish-Φ_i
     ablation (drop = 1.0 — price term never arrives; isolates the
     μ-exchange contribution). 30 + 45 = 75 runs.
   - Horizon 120 min (short-calibration-horizon rule); coordination
     config + metric definitions identical to 011 (same constants,
     imported).
   - Resume-safe staging (`runs/run_<seed>_<arm>.json` is the unit),
     `--jobs` process parallelism, `--smoke`, `--summarize`
     (parquet + generated MC_SUMMARY.md incl. the §3.10.2
     finite-switching premise statistics from the pooled ledgers).

## Fixed today

- `012::summarize()`: ledger entries from a run's FINAL tick never
  receive the deferred realised-ΔΦ fill and round-trip through the CSV
  as empty strings — `astype(float)` would crash the premise
  statistics. Now `pd.to_numeric(..., errors="coerce")` (the subsequent
  `isfinite` filter already handled NaN).

## Validation

- `pytest tests/test_bme_h_error.py tests/test_boundary_sensitivity.py
  tests/test_bme_gradient_identity.py tests/test_discrete_hygiene.py`
  → **40 passed** (H-error wrapper + no regression in the touched
  boundary-sensitivity module, identity hard gate, hygiene).
- `012 --smoke` (1 scenario, 30 min, none/bme/oracle): **all three arms
  converged** (none 34 s, bme 178 s, oracle 551 s); staging
  JSON/CSV/NPZ + parquet + MC_SUMMARY.md generated; 13 pooled ledger
  entries; identical dispatch banner across arms re-confirms the 6d
  scenario-identity fix inside the MC harness. Smoke METRICS are not
  meaningful (30-min horizon = cold-start transient; the "last-hour"
  mask covers the whole run).

## Second fix found by the smoke run

- **Resume-unit horizon collision**: the smoke run staged
  `run_20260705_*.json` at 30 min, and `BASE_SEED = 20260705` is also
  the campaign's first scenario seed — `_done()` would have silently
  absorbed the 30-min smoke artefacts into the 120-min campaign.
  Fixed twice over: smoke artefacts moved to
  `results/012_BME_MC/smoke_30min/`, and `_done()` is now
  horizon-aware (a staged row only counts if `row["minutes"]` matches
  the requested spec).

## Campaign launch

Launched 2026-07-06, background, `--run --jobs 4` (16-core machine;
BLAS pinned to 1 thread per worker by the script's env guards).
Structure: Phase A = 10 accepted scenarios × {none, bme, oracle} with
drop-and-replace; Phase B = 15 sweep arms × first 3 accepted seeds;
75 runs total at the 120-min horizon. Results stage incrementally under
`results/012_BME_MC/runs/` (resume-safe); `--summarize` regenerates
parquet + MC_SUMMARY.md at any time. Expected wall-clock: several hours
(Phase A is gated per scenario by the oracle arm).
