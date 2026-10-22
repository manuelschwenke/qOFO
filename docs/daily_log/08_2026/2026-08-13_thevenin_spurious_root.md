# 2026-08-13 — `tie_boundary="thevenin"` converged onto a spurious low-voltage root

**Author:** Manuel Schwenke / Claude Code
**Timestamp:** 2026-08-13
**Reason:** Reported symptom — with `tie_boundary_equivalent="thevenin"` the
switched-shunt integrator (`controller/shunt_integrator.py`, the separate
dispatcher outside the MIQP) never commits a single step, at any
`shunt_int_g_w`; lowering it 150 → 10 changed nothing. Root cause turned out
not to be a gain problem at all, and it invalidates more than the shunts.

## Symptom → cause chain

1. `add_thevenin_boundary` creates an **auxiliary bus** per tie corridor
   (`sensitivity/network_reduction.py`, step 7). The reduced net inherits the
   plant's cached `res_bus` by deepcopy; the aux bus is created *afterwards*
   and therefore has **no `res_bus` row**.
2. Step 10 converges the reduced net through `runpp_with_stored_jacobian`,
   whose default ladder tries `init="results"` first. With an incomplete
   result table pandapower neither raises nor warns.
3. Newton lands on the **spurious low-voltage root**. A reduced net is
   unusually exposed to it: the stripped tertiary and far-end stubs carry zero
   injection, and `V = 0` satisfies their mismatch equation identically.
   Zone 2 at 2016-01-05 08:00 reported `converged = True` with TN buses at
   0.44–0.66 pu, the 3W tertiaries at ~1e-62 pu, the in-zone slack absorbing
   −1887 Mvar and ±600 Mvar on the WARD tie branches. Zone 3 likewise; zone 1
   happened to land on the correct root.
4. `compute_dV_dQ_shunt` (`sensitivity/jacobian.py:1939`) scales the column by
   `V_pu**2` at the shunt bus — `(1e-62)² ≈ 1e-124`. Measured
   `∂V/∂Q_eq` at the MSC banks, same operating point:

   | zone / tertiary bus | `pq` | `thevenin` (before fix) |
   |---|---|---|
   | 2 / 53 | 5.14e-4 | 7.9e-128 |
   | 2 / 73 | 5.16e-4 | 3.6e-127 |
   | 2 / 93 | 3.62e-4 | 1.6e-133 |
   | 3 / 113 | 2.99e-4 | 1.3e-163 |

5. Hence `grad_g = h_Hᵀ ∇_y f ≡ 0` in `ShuntBank.step`; the relaxation state
   `q_eq_aux` never leaves zero, never reaches the commit band
   (`0.5·q_step + delta = 22.5` Mvar), and no `ShuntCommit` is ever emitted.
   `Δ = g_H / (2 g_w)` with `g_H = 0` is zero for every `g_w` — which is why
   the 15× gain increase did nothing. The `h_qpcc` interface term was dead for
   the same reason.

## Changed — `sensitivity/network_reduction.py`

### 1. `add_thevenin_boundary`: seed the aux bus in `res_bus`
The function already back-solves the EMF `e_pu`, which **is** the aux-bus
voltage phasor by construction, so the warm start can be made exact rather
than merely complete:

```python
seed["vm_pu"]     = abs(e_pu)
seed["va_degree"] = rad2deg(angle(e_pu))
net.res_bus.loc[aux] = ...
```

### 2. New `_assert_reproduces_cached_state(...)` + step 10b guard
Fail-fast check that the converged reduced net actually reproduces the cached
operating point, run for **every** boundary variant. The premise of the whole
reduction is that all variants match the cached state by construction and
differ only in the *derivative*; a converged net that does not match is on a
different root, and every H entry extracted from it is a linearisation about a
fictitious state. Auxiliary boundary buses are excluded (they have no plant
counterpart). New keyword `op_point_tol_pu`, default `_OP_POINT_TOL_PU = 0.1`.

Tolerance rationale — measured `max |vm_reduced − vm_cached|` over kept buses,
healthy cases:

| variant | zone 1 | zone 2 | zone 3 |
|---|---|---|---|
| pq | 1.73e-2 | 2.7e-10 | 1.67e-2 |
| pv | 1.37e-2 | 2.0e-10 | 1.30e-2 |
| z  | 1.72e-2 | 4.6e-10 | 1.67e-2 |

Zone 2 holds the system slack, so it matches to solver tolerance. Zones 1 and
3 must promote a machine to slack and the reduced net solves with
`distributed_slack=False`, so the promoted machine absorbs a mismatch the
plant spread over all machines — a legitimate ~1.7e-2 pu offset. 0.1 pu sits
~6× above that and ~4× below the failure it exists to catch.

## Verification

* All four variants build; min `vm_pu` over all reduced nets: pq/pv/z 1.0132,
  thevenin 0.9941 (the latter is an aux-bus EMF, not a TN bus).
* Shunt columns are live again. `∂V/∂Q_eq` ratio thevenin / pq:
  zone 2 bus 53 **0.366**, bus 73 **0.664**, bus 93 **0.375**, zone 3 bus 113
  **0.536** — i.e. the Thevenin boundary genuinely gives a *smaller* shunt
  column (a stiffer neighbour supports the boundary voltage), landing near the
  `pv` limit (0.334 for bus 53), which is the expected ordering
  `pq > z > thevenin ≳ pv`.
* Guard fires when the seed is suppressed: `ValueError: ... 32/39 bus(es)
  deviate by more than 0.1 pu (bus 2: 0.8697 vs cached 1.0310, ...)`.
* `pytest tests/test_shunt_integrator*.py tests/test_tso_tertiary_shunt.py
  tests/test_jacobian.py tests/tuning/test_io.py` → 17 failed, 78 passed,
  5 skipped — **identical set on HEAD** (all `Unknown scenario: 'base'`,
  pre-existing and unrelated).

## Consequences for prior results — ACTION REQUIRED

Everything computed under `tie_boundary="thevenin"` for **TSO zones 2 and 3**
used an H linearised at a ~0.5 pu fictitious operating point. This is not
confined to the shunts — it is the full zone H (voltage rows, Q_PCC rows,
every actuator column). Affected, at least:

* the BO baseline pinned in `5266850` (*"pin baseline for the Thevenin BO
  study"*) and the setup notes `2026-08-13_bo_thevenin_study_setup.md` /
  `2026-08-13_BO_STUDY_SETUP_PROMPT.md`;
* the `007d/g/h/m/n/o/p/q` Thevenin arms;
* the conclusion recorded in `experiments/run_multi_system_ofo.py` that *"the
  Thevenin H is smaller, so at the PQ gain the loop is under-driven"*, and the
  uniform 0.3 re-gain now folded into the `g_w_*` block. The direction of that
  statement survives (the measured shunt-column ratio is 0.37–0.66), but the
  **magnitude** was derived from the bad root and must be re-derived. Whether
  a per-zone re-gain is still needed is now an open question — the asymmetry
  (zone 1 correct, zones 2/3 broken) is exactly the shape that would masquerade
  as "zones need different scales" in 007m/007p.

Note also that `shunt_int_g_w` was lowered 150 → 10 while chasing this. With
the columns restored at ~0.4–0.6× the `pq` value, the gain-equivalent setting
is nearer 150 × 0.5 ≈ 75; 10 is likely far too aggressive and should be
re-picked before any Thevenin run is trusted.

## Open / not addressed

* **Aux-bus indices break the module's stated invariant.** The docstring
  claims reduced-net bus indices match the plant. `pp.create_bus` assigns
  `max(index) + 1` of the *reduced* net, so aux buses reuse indices belonging
  to plant buses this zone dropped (zone 2 aux = 96–99, which are HV-net buses
  elsewhere). Harmless today — the new index always exceeds every *kept* bus,
  so lookups by plant index cannot collide — but it is luck, not design.
* `build_dso_local_net` has its own `boundary="thevenin"` path. It starts
  flat rather than from results, so the same hole probably does not open
  there, but it was not tested and has no operating-point guard.
* Whether the shunt integrator now engages *sensibly* is untested — with
  `delta = 10` against `q_step = 25` the commit band is 90 % of a full step,
  leaving a thin anti-windup margin.
* Single operating point (2016-01-05 08:00), `local_sensitivities_tso=True`.
