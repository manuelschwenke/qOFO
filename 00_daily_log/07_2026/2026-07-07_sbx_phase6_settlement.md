# 2026-07-07 — SBX Phase 6: three-tier settlement

**Task:** SBX plan v2 §2.5 / §4 Phase 6 — settlement at fixed contract
prices, per elapsed cycle, cycle-averaged measurements only.

**Changed (new files unless noted):**

- `sbx/settlement.py`: `CycleObservation` (frozen; the elapsed cycle's
  averages + the schedule/references that were ACTIVE during it),
  `SettlementEngine` (one per corridor): tier 1 in-band free +
  unmonetised signed netting ledger (beyond-band cycles log the clipped
  in-band portion); tier 2 paid-surplus billing
  `p_surplus × |paid| × t_cycle`, payer = the importing (non-acting)
  side (§2.5 verbatim — note: for an EXPORT-need requester this makes
  the supporter pay; recorded as an open point); tier 3 beyond-band
  excess charged `κ·p_surplus` per Mvar·h, attributed by the dominant
  term of the per-line first-order decomposition
  (C_A | C_B | C_P via `corridor_sensitivities` at the elapsed
  references incl. the acting dv), ΔP-dominant → settlement-neutral,
  residual > max(abs, rel·excess) → `UNATTRIBUTED`, no charge.
  Conservation (payments sum to zero per corridor and cycle) asserted
  on every settlement.  `n_settle_cycles > 1` → rolling-mean window
  (short-cycle ablation).  `write_settlement_outputs` → ledger CSV +
  Markdown summary under `results/<experiment>/`.
- `sbx/scheduler.py` (edited): `record_step` gains per-line terminal
  voltage inputs (`tie_v_a_pu`/`tie_v_b_pu`, sampled per TSO tick);
  at each boundary Step 1 now captures the elapsed cycle's schedule
  (`q_std`, surplus, paid/unpaid, `p_sched`, frozen references) BEFORE
  overwriting and feeds `SettlementEngine.observe`; engines exposed as
  `settlement_engines` / `settlements`.  The elapsed cycle's measured-P
  average doubles as `p_meas` of the settlement and the next cycle's
  persistence forecast (same quantity by construction).
- `sbx/adapter.py` (edited): `_corridor_terminal_voltages` — per-line
  terminal V, each side read from ITS OWN area's measurement.
- `tests/sbx/test_settlement.py`: engine-level acceptance (tier-1
  netting, tier-2 importer-pays both signs, mutual-unpaid not billed,
  partial paid split, tier-3 attribution to side A and B with κ-charge,
  ΔP neutrality, `UNATTRIBUTED`, conservation, `n_settle_cycles=2`
  rolling mean, paid-without-acting-side guard, output files) +
  scheduler-wiring test (unilateral deal billed with the requester as
  the importing payer; paid-first unwind visible as a monotone fall of
  the billed Mvar·h back to zero).  `tests/sbx/conftest.py` shares the
  `plant` fixture.

**Result:** `pytest tests/sbx` → 57 passed.  Closed-loop verification
via the three-arm Phase 5 smoke test re-run with settlement active
(results in STATUS_SBX.md §Phase 6).

**Reason:** Phase 6 completes the operational layer's economic
bookkeeping: deviations become tiered, attributed, conserved transfers
at fixed prices — the quantities Phase 7 compares across
AUTONOMOUS/SBX/BME arms.
