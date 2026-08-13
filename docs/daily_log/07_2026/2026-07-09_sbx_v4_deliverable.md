# 2026-07-09 — SBX v4 "deliverable SBX": (i) + (iii) + (A)

**Task:** Manuel approved the full v4 package after the delivery-gap
analysis (schedule ratchet to 150 Mvar with ~250 Mvar
schedule/measurement divergence; zone-3 joint-box collapse F2).

**Changed:**

- `controller/tso_controller.py` (FIRST SBX-motivated controller
  change; A4/G5 protection lifted): optional
  `TSOControllerConfig.g_v_per_bus` (validated), `_g_v_vector()`
  helper used at the output-gradient, grad_f and curvature sites
  (scalar case bit-identical), public
  `update_voltage_tracking_weights(bus_indices, weight)`.
- `sbx/config.py`: `w_track_factor = 20.0` (corridor terminals tracked
  at factor × g_v; `w_track` absolute override), `delivery_gate = True`,
  `capability_mode = "auto"` (+ validations).
- `sbx/adapter.py`: applies the terminal priority weights at
  construction; legacy w_track ≡ g_v assertion removed.
- `sbx/scheduler.py`: delivery gate in `_build_message` (returns
  suppression flag; need still tracked, request withheld), non-delivery
  counts toward the unwind dwell, `CorridorCycleRecord.
  request_suppressed_a/b`; capability via the new dispatcher.
- `sbx/capability.py`: `area_capability` dispatcher
  (auto / joint_box / per_corridor), `CapabilityResult.mode`;
  per-corridor 2-vertex LPs with full boxes — cross-corridor side
  effects priced by the existing tier-3 attribution.
- Tests: 3 legacy harness tests pin `delivery_gate=False` (synthetic
  tie_q = 0 IS non-delivery); new `test_v4_deliverable.py` (gate
  suppression + bounded probe + non-delivery unwind; F2 collapse and
  auto fallback; per-bus weight validation). 63 passed.

**Design note (emergent, accepted):** the gate resets after a full
unwind (`no_surplus` carries no evidence) — persistent non-delivery
yields a bounded probe loop (≤ ~1 quantum outstanding) instead of a
hard lockout; deliberate: conditions may change, and periodic probing
at bounded cost is the honest protocol behaviour.

**Status:** closed-loop validation (014, 90 min, all v4 defaults)
running — checks that the acting side now realises the commanded dv
(w_track priority) and that zone 3 offers nonzero support
(per-corridor fallback).
