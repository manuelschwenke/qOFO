# 2026-07-07 — SBX Phases 3–4: need/capability and messages/matching

**Task:** SBX plan v2/v2.2, Phases 3 and 4.

**Changed (new files unless noted):**

- `sbx/need.py`: `NeedTracker` (per-area consecutive-violation counter,
  direction from the deeper bound violation, gap/direction-change
  resets), `NeedDecision.request_sign(own_end)` (corridor-flow sign
  space), `assert_relieving_sign` (§2.3 sanity assert; sign condition
  direction·(dv per signed request) > 0).
- `sbx/capability.py`: v2.2 D13 joint-box LP — one LP per area/cycle,
  `w = [t | Δu_σ1 … Δu_σm]`, m = 2^|C_i| sign vertices, actuator box +
  margined voltage box (local H) per block + corridor equality rows
  coupling t. Mapped onto the existing `MIQPSolver._solve_qp`
  (G_w = 0, G_z = 0 → hard-constraint pure LP, α = 1) — no solver
  change. Offers (−a, +a), a = min(t, 1)·quantum; skip + (0,0) offers
  when the measured point violates margined limits; `rep1` on any
  non-(near-)optimal status.
- `sbx/messages.py`: frozen versioned `PeerCairMessage` (BME-bus
  dataclass style), reference-end-only `p_sched_mw`, canonical
  serialisation (sorted-key JSON, repr floats) + SHA-256 checksum,
  `assert_checksums_match`.
- `sbx/matching.py`: pure deterministic `match()` per §2.2 Step 3
  (mutual min unpaid / unilateral clip paid / opposite-sign scarcity /
  dust / contract cap / quantum-magnitude guard), frozen `DealRecord`
  with checksum.
- `sbx/contract.py` (edited): `dq_min_deal_mvar` added to
  `CorridorContract` — the dust threshold is matching-relevant and must
  be bilateral for two-sided deterministic evaluation (deliberate
  deviation from plan §2.1's field list, recorded in STATUS_SBX.md).
- `tests/sbx/test_need_capability.py`,
  `tests/sbx/test_messages_matching.py`: full Phase 3/4 acceptance
  (details in STATUS_SBX.md).

**Result:** `pytest tests/sbx` → 38 passed.

**Reason:** Phases 3–4 complete the per-cycle decision pipeline
(need → capability → messages → deal) ahead of the Phase 5 scheduler
that orchestrates it against the plant loop.

**Status:** Continuing with Phase 5 (six-step cycle scheduler, runner
integration, closed-loop smoke test).
