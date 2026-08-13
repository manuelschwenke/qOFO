# 2026-07-09 — SBX-V Phase 5: runner wiring, closed-loop R1, E1

**What:** Wired SBX-V into the multi-TSO/DSO runner (`coordination_mode="sbxv"`), verified
closed-loop regression R1, and launched the first E1 run.

**Changes:**

- New `sbxv/adapter.py` (`SBXVRunnerAdapter`): derives AggregationAreas from the zone
  controllers' `pcc_dso_controller_ids`, installs `PricingSolver` proxies on the zone
  controllers' `solver` attributes, drives one `CommitScheduler` per zone, meters per plant
  step (`[t−dt, t)`, post-PF state), captures the dispatched netted PCC-Q references per
  window, and settles the scenario in `finalise()` (grants beyond the metered horizon are
  excluded LOUDLY, listed in `dropped_grants`).
- `experiments/runners/multi_tso_dso.py`: five insertions, all guarded by
  `coordination_mode == "sbxv"` (whitelist + validation, pre-loop adapter construction +
  `sbxv_runtime` in the `pre_loop_hook` state, `before_solve` / `after_solve` hooks around
  `coordinator.step`, plant-step metering hook). Every other mode is untouched byte-for-byte.
- `configs/multi_tso_config.py`: `sbxv_config` field + `coordination_mode` docstring update.
- New `tests/sbxv/smoke_sbxv_closed_loop.py` (smoke, not pytest-collected) and
  `experiments/018_SBXV_E1.py` (arms none / sbxv_neutral / sbxv; plan-E1 metrics; settlement
  CSVs + JSON summary under `results/018_SBXV_E1/`).

**Results (45-min smoke):** R1 GREEN — 1263 recorded control arrays byte-identical over 135
steps between `none` and neutral `sbxv`; settlement plane ran (12 windows). Finding: the
BASELINE scenario already operates some netted areas beyond the default ±50 Mvar band with
references beyond the edge → `8.2-4b_adhoc` (ad-hoc Grenzpreis) windows appear without any
grant; E1 quantifies the economics. Unit suite 100/100 green.

**Design decision (documented in STATUS §5.1):** no setpoint feedforward offset in the v1
wiring — there is no plant-side jump for the DSO to counteract (prices move the reference in
micro-steps); the scheduled-envelope lead stays available and E1's commit-instant tracking
metric decides whether it is needed.

**E1 first run (120-min calibration horizon, three arms):** R1 re-confirmed (3363 arrays
byte-identical); the active arm is identical to the baseline because the 005 profile never
violates the transmission voltage band over this horizon (condition B never fires → 0
requests) — the mechanism is non-invasive when unneeded. **DSO commit-instant tracking
invariant HOLDS** (commit-instant mean error 1.70 Mvar < all-steps 1.85 Mvar → no feedforward
lead needed, confirming the v1 wiring decision). Economics: 2 of 32 windows settle ad hoc at
the Grenzpreis (1042 € / 2 h at placeholder prices) — the persistent-exceedance
planning-deficit indicator is live. Grant-path activation in closed loop requires the E2
tighter-band arms or the E3 contingency.

**Why:** Plan §9 Phase 5; completes the Phase-4 deferred closed-loop items (R1 done; the DSO
commit-instant tracking invariant measured and green).
