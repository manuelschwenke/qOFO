# BME Phase 0 — repository reconnaissance (read-only)

**Date:** 2026-07-02
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Phase 0 of the Boundary Marginal Exchange (BME) build plan — the
horizontal multi-TSO coordination spec that supersedes the set-aside
gradient-exchange tie coordinator. Read-only reconnaissance: no source file was
modified; the only files created are `docs/BME_STATUS.md` (the living status
file required by the spec) and this log entry.

## What was done

1. **Component-mapping table** — every abstract spec component mapped to the
   repository with file/line references (full table in `docs/BME_STATUS.md`
   §0.1). Key mappings: multi-area module = `experiments/runners/multi_tso_dso.py`
   + `controller/multi_tso_coordinator.py`; CAIR = `CapabilityMessage` →
   `receive_capability` → PCC input bounds; hysteresis/dwell = base-controller
   integer cooldowns + `shunt_integrator`; DSO feedforward correction =
   `q_itf_sh_offset` + `ShuntDisturbanceMessage` + SMW refresh (runner
   l. 3302–3376).
2. **Audit A1 (gradient convention):** the implemented TSO gradient is
   `∇_u f = ∇_u f_direct + Hᵀ∇_y f` with H built from the **shared full-network
   Jacobian** in the default mode → **Convention B**; the BME price term must use
   J = neighbours only. The `local_sensitivities_tso=True` Ward-PQ reduction is
   neither Convention A nor B (boundary floats as constant-PQ) — proposal:
   `mode="bme"` fail-fasts under that mode.
3. **Audit A2 (v_ref hypothesis):** **not confirmed as stated.** The implemented
   scheme (post 2026-06-28 redesign) exchanges a normalised scalar boundary
   marginal γ per tie and negotiates a bounded ΔV_ref **setpoint** — no price
   term in any gradient. The exchanged signal has the §3.7 quadratic-marginal
   shape, but the mechanism differs; `mode="vref"` will wrap the existing path
   unchanged (spec's fallback branch). The v1 price-in-gradient failure
   (2026-06-25) and the two documented flaws that motivated the pivot
   (incommensurate objective scales; sticky-OLTC long-run degradation,
   2026-07-01) are recorded as positioning points and risk flags.
4. **Audit A3 (boundary topology):** enumerated via a read-only scratch
   diagnostic on `build_ieee39_net()` + `fixed_zone_partition_ieee39`:
   5 tie lines (pp indices 2, 14, 25, 5, 18), boundary set B = 9 tie-endpoint
   buses (IEEE {2, 3, 9, 14, 15, 17, 18, 27, 39}); separator property holds
   (no cross-zone non-line branches). Notables: IEEE 3 serves two ties
   (B_12 ∩ B_23 ≠ ∅ — v_ref cannot compose there, BME's additive price can);
   IEEE 39 is the slack bus (pinned V ⇒ inert μ/band entries); 0-idx bus 19
   absent from all zone lists (interior; verify in Phase 1).

## Key structural findings

- `MultiTSOCoordinator` already computes cross-zone H_ij and the preconditioned
  M_sys with the contraction criterion — most of the Phase 7 non-cooperative
  eigenvalue analysis exists.
- The loss objective (form B, `g_loss`, I-rows of H) is directly reusable for
  the Φ_i loss term; the φ_band hinge as an *objective* term is new.
- No message bus / delay / drop machinery exists today (γ is exchanged by
  direct method calls inside one runner step) — the CoordinationBus is new.

## Reason

Phase 0 acceptance criterion of the BME spec: mapping table + audits A1–A3 with
file/line references + go/no-go **before any code is written**. Outcome: GO for
Phase 1, conditional on Manuel's DECISIONS D1–D8 and open questions Q1–Q3, Q5
(see `docs/BME_STATUS.md` §0.6–0.7, §0.10).
