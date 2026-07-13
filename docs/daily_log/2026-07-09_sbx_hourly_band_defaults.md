# 2026-07-09 — SBX: planning-derived hourly band + 014 defaults

**Task:** Manuel: derive the tier-1 band from the day-ahead calculation
too; make 014 use the planning pre-pass and local sensitivities.

**Changed:**

- `experiments/017_SBX_PLANNING.py`: `--band-ensemble N` (default 8) —
  per hour N forecast-error-sampled PFs with taps held at the day-ahead
  plan; hourly per-corridor band = z·σ_ensemble + |s_corr|·ε_track +
  m_gap (floor 5 Mvar); JSON entries gain the band as a 4th element.
  Perfect-mode bands: (1,2) 23–33, (1,3) 14–15, (2,3) 25–36 Mvar —
  within a few Mvar of the 013 empirical calibrations.
- `sbx/contract.py`: `q_band_schedule` + `q_band_at(t)` (+ validation,
  t = 0 consistency rule); `build_default_contract(q_band_schedule=)`.
- `sbx/scheduler.py`: active band resolved per cycle
  (`_CorridorState.q_band_act`), elapsed band captured for settlement
  and the consistency classifier (signature change);
  `CorridorCycleRecord.q_band_mvar`.
- `sbx/settlement.py`: `CycleObservation.q_band_mvar` (validated);
  tier classification uses the windowed observation band instead of the
  contract constant.
- `visualisation/plot_sbx.py`: per-cycle band shading (breathes at hour
  boundaries).
- `sbx/adapter.py` + runner: band schedules parsed from the JSON
  (3- or 4-element entries) and passed into the contracts.
- `experiments/014_SBX_SINGLE_DEMO.py`: DEFAULTS changed — planning
  schedule auto-detected (`--no-schedule` restores the v2 snapshot +
  30-min warmup) and LOCAL sensitivities (`--shared-sens` opts out);
  effective configuration printed; persist block re-creates the output
  directory (a clean-up during a long run had deleted it → the one
  failed validation).
- `tests/sbx`: `make_obs` carries the band; v3 test extended with band
  lookup + validation guards. 58 passed.

**Validation:** 90-min asym_z3 with all defaults (schedule from t = 0,
hourly bands, local sensitivities): exit 0, plan-reality behaviour as
in the F10 fix, payload pickle persisted.

**Analysis session (no code):** Manuel's aggressive-defaults run
(n_need = 1, cap 150) showed ~250 Mvar schedule/measurement divergence
on (1,2) → delivery-gap analysis (soft-tracking priority, offer ≠
instruction, requester pinning, Q locality) + F2 joint-box geometry.
Proposed "v4 — deliverable SBX" package (per-bus w_track, delivery-
conditioned requests, per-corridor capability with priced spillover) —
recorded in STATUS, awaiting Manuel's architecture decision.
