# 2026-07-03 — BME Phase 4b (part 1): coordination config fields + TSOController BME hook

**Context:** BME Phase 4 wiring (core/hard gate landed 2026-07-02,
commit `f481479`). This entry: config mapping + controller-side hook;
runner wiring and trajectory regressions remain (see `BME_STATUS.md`
Phase 4 "Remaining").

## What was changed

### `configs/multi_tso_config.py`

New flat fields (spec §4 `coordination:` block, no parallel config
system), inserted before the tie-coordination section:
`coordination_mode` ("none" default | "vref" | "bme"),
`bme_delay_steps=1` (D4), `bme_drop_probability=0.0`,
`bme_beta_filter=0.3` (D3), `bme_seed=None` (required when drops > 0),
`bme_w_band=0.0` (D2 — losses-only ablation default),
`bme_v_soft_min_pu=0.97` / `bme_v_soft_max_pu=1.03`,
`bme_vn_kv_min=220.0` (Q7 TS-level scope). Documented fail-fast
exclusions: "bme" × `enable_tie_coordination`, "bme" × non-zero
`g_q_tie` (Q3).

### `controller/tso_controller.py`

Injection-pattern hook (byte-identical behaviour when unused —
class-level `bme_mode=False` default, no per-instance state):

- `enable_bme_mode()` — enrols the controller; raises if `g_q_tie != 0`
  (Q3 double-steering guard).
- `receive_bme_gradient(grad_bus_level)` — one-shot injection of the
  runner-assembled g_i^bme (bus-level DER columns, ZoneInputSpec order
  == controller u order); validated finite 1-D; raises if not in BME
  mode.
- `_bme_objective_gradient()` — per-DER expansion
  ∇_der = [Eᵀ·∇_bus(DER block); rest] using the existing DER mapping
  directly (NOT via `_expand_H_to_der_level`, whose identity-keyed
  cache would be thrashed by a per-step vector); length-checked against
  `n_controls`; raises if no gradient was injected this step (enforces
  the spec §5 per-step sequence).
- One branch at the top of `_compute_objective_gradient`: under
  `bme_mode` the private objective gradient (g_v tracking, reserve,
  loss, tie terms) is fully REPLACED by the injected g_i^bme (D2/Q1).
  Output constraints, CAIR bounds and integer handling are untouched.

## Tests

Controller regressions re-run green: `test_tso_output_gradient`,
`test_tie_coordinator`, `test_tie_coordination_hooks`,
`test_tso_loss_objective` — 32 passed. The hook itself is exercised
end-to-end by the upcoming runner wiring tests (part 2); the gradient
values it will carry are already pinned by
`tests/test_bme_gradient_identity.py` (hard gate, 15 green).

## Why

Spec §5 Phase 4 tasks 1–2 (config + controller integration) of the
wiring stage; the injection pattern keeps the controller free of any
topology/bus knowledge (§3.9: zones interact with bus and plant only —
the runner mediates, mirroring the existing tie-coordination round).
