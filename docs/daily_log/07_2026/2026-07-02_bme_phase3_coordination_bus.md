# 2026-07-02 — BME Phase 3: CoordinationBus, marginal signals, receiver policy

**Context:** BME build, Phase 3 of the spec (`docs/BME_SPEC.md` §5; status
`docs/BME_STATUS.md`). Continues Phase 2 (commit `1a0881e`, same day).

## What was changed

### New: `core/coordination_bus.py`

Placed beside the vertical message classes (`core/message.py`), per the
Phase 0 component mapping. Contents:

- `MarginalSignal(zone_id, step, mu, v_b_meas)` and
  `SwitchNotice(zone_id, step, dv_b_pred, devices)` — spec §4 frozen
  dataclasses, repo-adapted (integer zone ids, `step` index). Vectors are
  validated on construction (1-D, finite, matching lengths) and frozen
  read-only (a published signal cannot be mutated afterwards).
- `CoordinationBus(zone_ids, n_boundary, delay_steps, drop_probability,
  seed)` — in-process pub/sub (§3.9): a message published at step k is
  visible at k + d and not earlier; signals are per-step (no stale
  carry-over at bus level); self-delivery never happens; duplicate
  (zone, step) marginal publishes raise; multiple switch notices per step
  are allowed (one per committed discrete move). Drop simulation: one
  Bernoulli draw per (message, receiver) from a bus-owned seeded RNG,
  drawn AT PUBLISH TIME in ascending receiver order — the drop pattern
  depends only on the publish sequence, never on query order or
  repetition (determinism, spec §8). `drop_probability > 0` without a
  seed raises. Structured `drop_log` of `CoordinationEvent`s.
- `MarginalReceiver(zone_id, bus, beta, start_step, expected_senders)` —
  receiver-side first-order low-pass per SENDER
  (μ^filt = (1−β)·μ^filt + β·μ(k−d), D3 β = 0.3; β = 1 disables
  smoothing, the Phase 4 identity-test configuration). Must be stepped
  consecutively (gaps raise). Returns
  `ReceivedMarginals(step, coordinated, mu_neighbour_sum)`.
- Explicit §3.8 policies (documented, logged as structured events, not
  silent defaults):
  - cold start — exactly d steps `coordinated=False` (`cold_start`
    event per step);
  - missing expected signal after warm-up — RAISES when drops are
    disabled (protocol violation);
  - with drops enabled — hold-last-FILTERED-value (`hold_last` event
    per occurrence);
  - first signal from a sender dropped — nothing to hold: that sender
    contributes exactly zero until its first arrival (`extended_cold`
    event per occurrence);
  - filter initialisation — first received sample (β = 1 once), to
    avoid the ~1/β-step zero-bias of starting the recursion from 0.
- Convention-A scoping: the SELF-marginal μ_i never touches the bus or
  the filter; the receiver sums NEIGHBOUR marginals only and the
  controller adds μ_i locally, undelayed and unfiltered (Phase 4).
- Expected senders default to ALL other zones (not only tie-adjacent):
  the price term H_{b,i}ᵀ·Σ_j μ_j spans all of B — sparsity lives in the
  μ vectors, not the routing.
- A dropped NOTICE is lost and logged, never held (an event, not a
  state).

### `core/__init__.py`

- Exports the six new names.

## Tests

`tests/test_coordination_bus.py` — **15 passed** (pure numpy, ~1 s); full
regression sweep (BME Phases 1–3, output-gradient invariant, tie
coordinator, shunt integrator): **97 passed**. Coverage against the spec
§5 Phase 3 list:

- delay semantics (visible at k+d, not earlier, not later; d = 0
  same-step; notices delayed; no self-delivery);
- cold start logs and runs uncoordinated for exactly d steps; first warm
  step sums first samples; β-recursion checked against manual values;
- missing-signal-after-warm-up raises when drops disabled;
- hold-last-value engages and logs when drops enabled — every warm
  step's neighbour sum is reproduced from event log + filter states
  (held = bit-identical frozen, delivered = exact β-recursion,
  never-arrived = zero); p = 1 gives an all-zero neighbour sum with
  `extended_cold` logged;
- determinism: same seed → identical drop log and sums; different seed
  → different pattern;
- fail-fast validation (constructor bounds, unknown zones, duplicate
  publishes, non-consecutive receiver steps, frozen signal mutation).

## Why

Spec §5 Phase 3. The bus with d ≥ 1 is genuinely new machinery — today
all horizontal exchange happens as same-step direct method calls inside
one runner step (Phase 0 finding, §0.5); BME needs explicit delay, loss
simulation and filtering semantics before the controller integration
(Phase 4) can be tested honestly.
