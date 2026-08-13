# 2026-07-07 — SBX Phase 7: demonstration campaign design (session 2)

**Task:** Phase 7 per `HANDOVER_SBX_PHASE7.md` and Manuel's reframing
(mechanism demonstration; Φ/violation exposure descriptive only; BME as
optional orientation context; `sbx_inert` as the published baseline).

**Changed (new files only — `sbx/` and the runner belong to the other
session per the handover):**

- `experiments/013_SBX_LADDER.py`: five scenarios in three families
  (`asym_z3` = validated smoke reference; `asym_z1` border-actuator
  watch F5; `asym_z2` F2 illustration; `sym_z1z2` scarcity; `compl_z1z3`
  mutual-deal demonstration + D-P7-3 exercise), arms
  {none, sbx_inert, sbx[, bme]}, mechanism flags M1–M7, metrics.csv,
  REPORT.md, plots P1–P4, spillover metric (v2.2 item 6), and the
  F6/D-P7-5 tier-1 band calibration (sbx arm's `q_band` = max(5,
  ceil(2·RMS)) of the inert arm's clean-cycle deviation — config-level,
  no `sbx/` edit).
- `STATUS_SBX.md` §7.1–7.2 (appended after the other session closed
  Phase 6): reframing recorded as the governing acceptance, D-P7-1…5
  dispositions.

**Method:** structure copied from the Phase 5/6 smoke (`make_config`
arm construction, `pre_loop_hook` capture, cycle↔time mapping
`t = 30 min + c·15 min`); per-arm pickles for `--evaluate` re-runs;
BME constants imported inside the `bme` branch only, so 013 survives a
future removal of BME from the codebase (Manuel 2026-07-07: likely).

**Reason:** plan v2 §8 definition of done requires one reproducible
experiment script producing the Phase 7 table and plots; the reframing
replaces the J-ordering acceptance with mechanism-behaviour acceptance.

**Status:** 120-min calibration pass for the four new scenarios
running; full campaign (360 min × 3–4 arms × 5 scenarios ≈ 2 h wall)
follows once the stress magnitudes are locked.
