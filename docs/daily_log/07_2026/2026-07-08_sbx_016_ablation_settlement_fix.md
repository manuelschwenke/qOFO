# 2026-07-08 — 016 quantum/cycle ablation + rolling-window settlement fix

**Task:** Manuel activated the deferred D-P7-4 ablation: larger quantum
and a 3-minute cycle.

**Changed:**

- `experiments/016_SBX_ABLATION.py` (new): variants `sbx_fast`
  (k_sched = 1 → 3-min cycle, 2 Mvar/cycle, same 40 Mvar/h ramp,
  rolling 5-cycle settlement window), `sbx_bigq` (rate 30 Mvar/15 min,
  cap raised to 100 — the default 50 cap would saturate after two
  deals), `sbx_fast_bigq` (both). Baselines none/sbx_inert/sbx loaded
  from the 013 pickles; outputs: ABLATION.md table (exposure, deals,
  exchanged Mvar, first-deal latency, tier-2 EUR) +
  `F1_ablation.png` (exposure bars + scheduled-import trajectories).
- `sbx/settlement.py` (**bug fix — F8**): the first closed-loop use of
  `n_settle_cycles > 1` (sbx_fast) hit
  `rep1("paid surplus without an acting side")` at corridor (1,3),
  cycle 38: tier 2 bills the WINDOWED mean paid surplus but took the
  payer direction from the LATEST observation's `acting_end`, which is
  None once the newest cycle is fully unwound while the window still
  carries paid history. Fix: the acting side is derived from the
  windowed mean surplus (§2.5's sign(s) applied to the same window as
  the billed quantity), knife-edge fallback = sign of the windowed paid
  component. Behaviour for `n_settle_cycles = 1` is unchanged (the
  scheduler never produces paid ≠ 0 with s = 0 in a single cycle).
- `tests/sbx/test_settlement.py`: `test_paid_without_acting_side_raises`
  (asserting the old rep1) replaced by
  `test_rolling_window_bills_after_unwind` (regression for the exact
  window state, incl. payer direction and conservation).

**Test state:** `pytest tests/sbx` → 57 passed. Ablation re-launched
(3 × 360 min, asym_z3).

**Note:** first `sbx/` edit by this session — the two-session file
ownership convention is treated as lapsed (the implementing session
closed at Phase 6; Manuel directs this session).
