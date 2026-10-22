# BME — gradient architecture revised to Convention A (decision, no code)

**Date:** 2026-07-02 (after Phase 1 completion)
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Decision revision only; no source files changed. Updated
`docs/BME_STATUS.md` (§0.2 revision note, §0.7 Q2) and
`docs/BME_HANDOVER.md`.

## What changed and why

Manuel clarified the standing locality assumption: every TSO controller knows
only its *own* area's sensitivities, so coupling `mode="bme"` to the shared
full-network Jacobian (as recorded earlier the same day under Convention B)
contradicted the thesis architecture. Resolution, chosen by Manuel from the
two exact alternatives:

**Convention A (spec §3.5) is now binding for the BME Φ-gradient:**

```
g_i^bme = ∂Φ_i/∂u_i |_{v_b fixed}  +  H_{b,i}ᵀ · Σ_{j ∈ ALL zones} μ_j
```

* `g_i^own` is computed from the zone-internal **port-frozen** Jacobian
  (boundary voltages held fixed) — local by construction, reusing the same
  J_int factorisation as `sensitivity/marginal_computer.py`. This is NOT the
  Ward-PQ reduction (`local_sensitivities_tso`), which leaves the boundary
  floating against a constant-PQ equivalent.
* The **self-marginal μ_i** enters through the price term, undelayed and
  unfiltered (it never crosses a border); neighbour μ_j arrive delayed d and
  filtered β.
* The **only supra-local object remains H_{b,i}** = ∂v_b/∂u_i, served by the
  access-enforcing `RestrictedSensitivityProvider` (simulation stand-in for
  locally identifiable estimation from own moves + boundary measurements).
* Consequently `mode="bme"` does **not** restrict `local_sensitivities_tso`:
  each zone's own MIQP loop (output constraints, prediction) may keep its
  Ward Jacobian — the pre-existing 004 model-quality trade-off, orthogonal to
  the Φ-gradient identity.

The Phase 0 audit *finding* is unchanged as a fact: the existing
private-objective assembly (mode "none") is Convention B in the default
shared-Jacobian mode.

## Consequences for the build plan

* **Phase 2 (`CommonObjective`):** must expose the frozen-boundary
  own-gradient pieces; the port-frozen input response
  ∂x_int/∂u_i|_{v_b fixed} is obtained by extending the `MarginalComputer`
  J_int machinery with zone-internal mismatch-derivative columns
  (∂g_int/∂u_i for Q injections, interior taps, gen vm).
* **Phase 4 identity test 2** uses the Convention-A split
  `dΦ/du_i = ∂Φ_i/∂u_i|_{v_b fixed} + H_{b,i}ᵀ·Σ_{all j} μ_j`.
* Phase 1 deliverables are unaffected (H_{b,i} provider and `MarginalComputer`
  are convention-agnostic building blocks).
