# 2026-07-08 — SBX Phase 7: demonstration campaign executed

**Task:** Phase 7 campaign per the 2026-07-07 design
(`docs/daily_log/07_2026/2026-07-07_sbx_phase7_design.md`).

**Runs:** 120-min calibration passes verified all five scenarios'
stress magnitudes (all need flags fire with correct signs; deal
archetypes as designed). Full campaign: 5 scenarios × {none,
sbx_inert, sbx} × 360 min (15 runs, two parallel thread-capped
processes, ~2 h wall), then the five-scenario merge evaluation.

**Result:** every applicable mechanism flag M1–M7 PASSES on every
scenario; all three deal archetypes demonstrated (unilateral,
scarcity, mutual); caps/dust rejections exercised; supporters exactly
violation-free after every deal (joint-box guarantee); unwind on
budget everywhere; settlement conserved (0 conservation failures,
UNATTRIBUTED only where designed). Key refinements: F1 pinning cost is
freeze-point-dependent (negative on `asym_z2`); F6 bands are
scenario-dependent (26–92 Mvar calibrated); D-P7-3 payer inversion now
evidenced (+456 EUR to the export-need requester in `compl_z1z3`);
spillover ≤ 1 Mvar. Full numbers: STATUS_SBX.md §7.3,
`results/013_SBX_LADDER/{metrics.csv,REPORT.md}` + per-scenario plots
P1–P4 and settlement ledgers.

**Changed:** `STATUS_SBX.md` §7.3 (appended); this log. No code
changes this session beyond `experiments/013_SBX_LADDER.py`
(2026-07-07); `sbx/`, runner, BME paths untouched. `pytest tests/sbx`
57 passed.

**Status:** Phase 7 complete. Open: D-P7-5 final band rule wording for
the dissertation; whether the optional BME orientation arm is wanted.
