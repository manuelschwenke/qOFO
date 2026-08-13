# 2026-07-03 — BME Phase 5: discrete hygiene (slotting, ε-acceptance, ledger, notices)

**Context:** BME Phase 5 (spec §3.8, §5), same day as the Phase 4
closure. Full detail in `docs/BME_STATUS.md` Phase 5 section.

## What was changed

- **NEW `controller/discrete_hygiene.py`**: `SlottingSchedule` (D5 round
  robin, ascending zone ids, slot length in TSO steps),
  `epsilon_accepts()` (§3.8.3 rule on the per-step MIQP vs
  frozen-integer QP objective values; ε = 0 default = pure improvement
  sign test), `SwitchingLedger`/`LedgerEntry` (append-only; one-time
  deferred realised-ΔΦ fill; `to_records`/`from_records` round-trip for
  the Phase 6 parquet export).
- **`controller/base_controller.py`**: step 7b hook
  `_post_solve_gate(result, solve_frozen)` between the MIQP solve and
  the commit — default no-op (mode="none" BITWISE regression vs the
  pre-BME baseline RE-VERIFIED after this change), plus
  `_solve_with_frozen_integers()` (identical per-step problem, integers
  pinned at current values; infeasibility raises).
- **`controller/tso_controller.py`**: gate override under armed hygiene
  (slot-blocked → frozen result; ε-reject → frozen result; accept →
  MIQP result; every decision ledgered; no-move short-circuits without
  the frozen solve). `configure_bme_hygiene()` (requires BME mode) +
  one-shot `set_bme_slot()` context (missing context raises).
- **Config**: `bme_slotting` (True), `bme_slot_length` (1),
  `bme_epsilon_switch` (0.0), `bme_switch_cost_oltc/shunt` (0.0) — D6
  magnitudes are Phase 6 calibration.
- **Runner**: shared ledger + schedule at setup; per tick: deferred
  realised-ΔΦ fill (Φ_global oracle, §3.10.2 premise data), notice
  consumption feeding the §3.8.1(b) estimator-masking hook (documented
  v1 no-op — per-tick re-linearisation leaves no innovation to
  correct), slot-context injection, and post-solve `SwitchNotice`
  publication (dv_b^pred = H_{b,i}^d·Δu_d, stacked coordinates). BME
  internals exposed through the `pre_loop_hook` state dict.
  **Fail-fast carve-out**: bme × `shunt_dispatch="integrator"` raises —
  Q5's integrator-bank notice/ledger emission is not wired yet
  (sign-sensitive and currently untestable; the reserved
  `integrator_commit` ledger reason documents the intent).

## Tests

`tests/test_discrete_hygiene.py` — **14 passed**; full sweep **139**;
mode="none" bitwise regression re-verified; bme smoke with hygiene armed
reproduces the exact pre-hygiene trajectory (no integer moves at the
uncalibrated Φ scale — gate inert, as expected before the Phase 6
gw_precondition calibration). Spec test mapping incl. the two documented
deferrals (closed-loop counter-switch scenario → Phase 6; notice
innovation-correction → requires the online-estimator tie-in) is in
`BME_STATUS.md`.

## Why

Spec §5 Phase 5; targets the sticky-OLTC long-run degradation mechanism
identified 2026-07-01 (ε-acceptance + slotting + ledger are the designed
countermeasures, §0.3).
