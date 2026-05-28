# 2026-05-27 — Per-controller local-network sensitivity option

## What changed

Added a second mode where each TSO and DSO controller obtains its Jacobian
from a *reduced* pandapower network containing only its own area, with
boundaries replaced by equivalent PQ injections from the cached operating
point.  Existing behaviour (every controller uses the full-net
`shared_jac`) is preserved when the new flags are off.

## Method / structure

* **New module** `sensitivity/network_reduction.py`:
    * `build_tso_local_net(net, zone_bus_indices, gen_indices_in_zone,
      machine_trafo_indices_in_zone, tie_line_indices,
      tie_line_endpoint_buses, hv_networks_in_zone,
      tso_shunt_buses_in_zone, tso_shunt_q_steps_mvar_in_zone, …)` —
      deep-copies the plant net and reduces it via selective deletion:
        * Keep: zone TN buses, gen-trafo LV/HV terminal buses, tie-line
          far-end stubs.  For every in-zone DSO's 3W coupler, keep the
          coupler trafo (in-service) + its primary bus + MV bus + LV
          (tertiary) bus.
        * Drop: every HV sub-network bus *downstream* of the coupler's
          MV bus (i.e. the 10×110 kV sub-network buses), plus every
          line / sgen / load / gen attached to them via `pp.drop_buses`
          (cascade).  Drop every other zone's TN buses and tie-line
          interiors.
        * Replace: at each tie-line far-end bus, strip everything except
          the tie line and add a PQ load equal to the cached
          `(p_xxx_mw, q_xxx_mvar)` at that endpoint (sign-flipped to
          represent the rest-of-system net injection).  At each 3W
          **MV bus**, add a PQ load equal to `(-p_mv_mw, -q_mv_mvar)`
          from cached `res_trafo3w` so that the live coupler trafo
          still carries its cached HV-side flow — the Ward equivalent
          for the dropped HV sub-network.  See the **Two-injection
          model** section below for the rationale.
        * Synthetic shunts: each TSO-owned tertiary shunt is represented
          in the TSO reduced Jacobian by a synthetic `pp.create_shunt`
          at the 3W primary bus with the same `q_mvar` per step and the
          same cached `step` value.  The `{tertiary_bus → primary_bus}`
          map is returned so the controller's
          `shunt_sensitivity_bus_indices` can be set to the synthetic
          location while `shunt_bus_indices` (the apply path) stays at
          the plant tertiary.
        * Slack: keep the existing IEEE 39 slack-gen if it lives in the
          zone; otherwise promote the largest in-zone gen to slack and
          return its machine trafo index so the runner can flag that
          OLTC out-of-service on the controller (a trafo touching the
          slack-reference bus has no `dV/ds` column in pandapower's
          reduced Jacobian, which would otherwise produce a column-count
          mismatch with the controller's H-matrix layout).
        * `pp.runpp(init='results')` (with a 1 e-8 V kick on the first
          bus so NR runs at least one iteration and stores
          `net._ppc['internal']['J']`); fall back to `init='flat'` on
          divergence.
    * `build_dso_local_net(net, hv_info)` — keeps HV sub-network buses,
      3W trafos, primary/MV/LV buses, the TSO-owned tertiary shunt; adds
      a virtual slack-gen at the 3W primary bus pinned to `V_cached`
      with very wide P/Q limits (no PQ load there — the slack
      auto-dispatches the cached HV flow).  This way the DSO controller
      still sees the shunt on its real tertiary bus, the existing
      `receive_disturbance_message` SMW path works unchanged, and the
      boundary upstream is represented by the slack-gen's auto-dispatch.

* **Config flags** in `configs/multi_tso_config.py`:
    * `local_sensitivities_tso: bool = False`
    * `local_sensitivities_dso: bool = False`

* **Controller config** in `controller/tso_controller.py`:
    * New optional `TSOControllerConfig.shunt_sensitivity_bus_indices`
      that overrides `shunt_bus_indices` *only* for the
      `JacobianSensitivities` sub-method calls inside
      `_build_sensitivity_matrix` (the apply path keeps using
      `shunt_bus_indices` → plant tertiary).

* **Coordinator** in `controller/multi_tso_coordinator.py`:
    * `compute_cross_sensitivities(zero_offdiag: bool = False)` — when
      True, `H_ij` for `i ≠ j` is left as the zero matrix of the right
      shape; only diagonal blocks are populated.  Honours the same flag
      on subsequent refreshes inside `step()`.

* **Runner** in `experiments/runners/multi_tso_dso.py`:
    * Post-Phase-2 block branches on the two flags.  Per-controller
      reduced nets are built and converted into per-controller
      `JacobianSensitivities`.  Promoted-slack machine OLTCs are marked
      OOS via `ctrl._oos_oltc_mask` so the controller's H-matrix
      assembly stays in sync with the reduced Jacobian's column count.
    * Cross-sensitivity call: `coordinator.compute_cross_sensitivities(
      jac=shared_jac, zero_offdiag=config.local_sensitivities_tso)`.
    * Shunt-switch handler: under `local_sensitivities_tso=True` it
      rebuilds the reduced Jacobian from scratch (the SMW path's lookup
      uses the plant tertiary bus, which no longer exists in the
      reduced net).

* **Smoke test** in `tests/smoke_local_sensitivities.py`: runs the
  4-minute IEEE 39 multi-TSO/DSO simulation in four configurations
  (baseline, TSO-only local, DSO-only local, both local) and reports
  per-case OK/FAIL.

## Two-injection model in the reduced TSO net

The TSO local net carries **two distinct Q injections** at the
3W-coupler interface, and confusing them — as I did in an earlier
write-up — causes real misunderstanding.  Spelling them out:

1. **Controllable Q_PCC,set actuator** — at the **primary (HV / TS)
   bus**.  This represents the DSO's dispatch capability: the OFO
   commands "deliver X Mvar at the HV side of the coupler" and the DSO
   modulates its internal DERs + OLTC to make that happen.  In the
   controller's H-matrix builder ([tso_controller.py:1574](controller/tso_controller.py)),
   the Q_PCC,set column uses
   `pcc_hv_buses = net.trafo3w.at[t, "hv_bus"]` (the primary bus) as
   the injection bus and calls `compute_dV_dQ_der` at the primary.
   The matching Q_PCC **output** the OFO tracks is the HV-side reading
   `net.res_trafo3w.q_hv_mvar` — also at the primary.

2. **Ward equivalent for the dropped HV sub-network** — at the **MV
   bus**.  This is a *static, uncontrollable* load representing
   "everything the DSO was already consuming at the cached operating
   point".  It is **not** the actuator; it is the baseline against
   which Δ(Q_PCC,set) is measured.

The Ward equivalent has to sit at the MV bus (not the primary) so the
live coupler trafo still carries the cached `q_hv_mvar` flow.  If we
placed it at the primary, the trafo would carry zero static Q in the
reduced net (nothing pulling power through it), while the plant
actually carries the cached HV flow — creating a structural offset on
the Q_PCC output measurement from t = 0.  Anchoring the Ward at MV
keeps `res_trafo3w.q_hv_mvar` in the reduced model identical to the
plant at the cached state, so Δ(Q_PCC,set) commands are interpreted
relative to the correct baseline.

User-facing picture:

```
    TN backbone ─── primary bus ─── 3W trafo ─── MV bus ─── (dropped HV sub-network)
                       ▲                            ▲
                       │                            │
        Controllable Q_PCC,set                 Ward load
        actuator (modelled as              = (-p_mv, -q_mv) cached
        an injection at primary,             (baseline DSO consumption
        Q ≡ HV-side trafo flow)              the trafo carries to it)
```

Both injections respect the user's original framing — the DSO controls
the *primary-side* Q.  Only the *static* baseline lives at the MV side
because that's where the model bookkeeping balances.

## Why

Until now, every TSO and DSO controller in the multi-zone OFO setup
operated on a single `shared_jac = JacobianSensitivities(net)` built
from the full plant net — so each controller's "local" H matrix was a
sub-block of one global Jacobian that implicitly carried full
inter-area coupling information.  The new option lets us study the
genuinely decentralised case: each controller sees only its own area
through equivalent boundary representations, matching the assumption
in the hierarchical control literature that a TSO/DSO controller
"knows" the rest of the system only through cached interface flows.

## How verified

`tests/smoke_local_sensitivities.py` runs 4 cases (4-minute IEEE 39
"wind_replace" scenario) → all OK:

```
========================================================================
Summary
========================================================================
  OK   baseline (shared_jac)
  OK   local TSO only
  OK   local DSO only
  OK   local TSO + DSO
========================================================================
```

DSO Q-tracking quality with both flags on (4-min smoke):
* DSO_1 mean |err| = 4.45 Mvar, max |err| = 6.08 Mvar
* DSO_2 mean |err| = 4.30 Mvar, max |err| = 7.11 Mvar
* DSO_3 mean |err| = 5.18 Mvar, max |err| = 8.13 Mvar
* DSO_4 mean |err| = 8.08 Mvar, max |err| = 10.71 Mvar

(For comparison, longer runs in the canonical scenarios are still TODO
— these 4-minute numbers serve as a sanity check that the controllers
do not diverge or violate hard limits under the new mode.)

## Assumptions & known limitations

* **Synthetic shunt approximation.** A TSO-owned tertiary shunt at the
  20 kV side of a 3W coupler is represented in the TSO local net by a
  shunt at the 3W primary (HV/TS) bus with the same `q_mvar` per step.
  The 3W coupler's series impedance is low enough that the susceptance
  effect on TN voltages has the right sign and order of magnitude, but
  the magnitude is not exact (no `V_tertiary²/V_primary²` rescaling
  applied).  Acceptable for first-cut MIQP gradient direction; will
  warrant a sensitivity study if shunt switching becomes a dominant
  actuator under this mode.
* **No SMW under local TSO mode.** The TSO's
  `apply_shunt_step_change_smw` rank-1 update path is bypassed under
  `local_sensitivities_tso=True` because the shunt's plant tertiary
  bus is not in the reduced net.  Instead the full reduced Jacobian is
  rebuilt on every shunt switch (a deepcopy + flat-start runpp +
  Schur-complement inversion).  Expected cost: a few hundred ms per
  switch event; switch events are rare so the amortised overhead is
  small.
* **Cross-zone contraction guarantee lost.** With
  `local_sensitivities_tso=True`, the coordinator's `H_ij` blocks
  (`i ≠ j`) are forced to zero.  The contraction LHS reported in the
  per-step printouts now reflects only `H_ii`, so the standing
  global-stability argument from the multi-zone OFO theory is no
  longer guaranteed — this is the price of strict decentralisation.

## Files touched

* `sensitivity/network_reduction.py`   (new)
* `sensitivity/__init__.py`             (re-export)
* `configs/multi_tso_config.py`         (+two flags)
* `controller/tso_controller.py`        (+`shunt_sensitivity_bus_indices`,
                                          rerouted shunt calls)
* `controller/multi_tso_coordinator.py` (+`zero_offdiag` parameter,
                                          honoured by `step()` refreshes)
* `experiments/runners/multi_tso_dso.py`(per-controller local-net wiring,
                                          shunt-switch rebuild branch,
                                          promoted-OLTC OOS handling)
* `tests/smoke_local_sensitivities.py`  (new)

## Bug discovered and fixed (later same day)

**Symptom**: under `local_sensitivities_tso=True` + `local_sensitivities_dso=True`,
the TSO commanded *constant* Q_PCC,set values forever — flat dashed lines
in the TSO-DSO interface Q plot — even with `g_w_pcc` reduced to a very
small number.  Switching to `False`/`False` brought back the expected
volatile setpoints.

**Trace** (verified by inspecting H matrix norms via
`tests/diag_local_pcc_h.py`):

* The TSO controller's `_build_sensitivity_matrix` populates the
  `Q_PCC,set` column block via a `pcc_in_trafo3w` precondition:
  `all(t in net.trafo3w.index for t in self.config.pcc_trafo_indices)`.
* The original `build_tso_local_net` dropped the in-zone 3W coupler
  trafos (step 3) and cascade-dropped their MV/LV stub buses (steps
  4-5).  In the local TSO net's `JacobianSensitivities.net`, the
  trafo3w rows were gone.
* `pcc_in_trafo3w` then evaluated `False`, so `pcc_hv_buses = []` and
  the controller skipped the entire `compute_dV_dQ_der` /
  `compute_dI_dQ_der_matrix` / `compute_dQgen_dQder_matrix` /
  `compute_dQ_line_dQ_der_matrix` chain that fills the V_bus,
  I_line, Q_gen, and Q_tie rows of the Q_PCC,set column block.
* Result: in the LOCAL H matrix the Q_PCC,set columns had only the
  diagonal +1.0 on the Q_PCC row (closed-loop identity).  The OFO had
  *no* V-tracking leverage from PCC dispatch, no I/Q_gen/Q_tie
  cross-terms either.  Gradient on the Q_PCC,set columns was
  essentially zero in every output row.  The regulariser
  `g_w_pcc · sigma²` then has its minimum at sigma = 0, regardless
  of how small `g_w_pcc` is.  Hence constant setpoints.

**Confirmed via `tests/diag_local_pcc_h.py`** (zone 2, n_pcc = 9):

| H block                  | FULL `||·||_F` | LOCAL `||·||_F` (before fix) |
| ------------------------ | -------------- | ---------------------------- |
| `H_V_bus  × PCC,set`     | 1.207 e-1      | **0.000**                    |
| `H_Q_PCC  × PCC,set`     | 3.000 (= √9)   | 3.000 (= √9, identity only)  |
| `H_I_line × PCC,set`     | 1.873 e-3      | **0.000**                    |
| `H_Q_gen  × PCC,set`     | 1.635          | **0.000**                    |
| `H_Q_tie  × PCC,set`     | 9.170 e-1      | **0.000**                    |

The Q_PCC identity column is the *only* thing the OFO sees from PCC
under the broken local model.

**Fix**: keep the in-zone 3W coupler trafos live in the reduced TSO
net.  Specifically (`sensitivity/network_reduction.py`):

1. Don't drop or deactivate the PCC trafo3w rows.  Add primary, MV,
   and LV buses for those couplers to `keep_buses`.
2. Drop only the HV sub-network downstream of the MV bus.  Strip the
   original load at the MV bus that used to absorb the downstream
   draw, then add a fresh PQ load equal to `(-p_mv, -q_mv)` from
   cached `res_trafo3w` — the Ward equivalent for the dropped
   sub-network.  Same at the LV bus (strip downstream loads but keep
   any TSO-owned tertiary shunt).
3. Run `pp.runpp(init='results')` with a 1 e-8 V kick to force NR to
   iterate (so `net._ppc['internal']['J']` is populated); fall back
   to `init='flat'` on divergence.

**Post-fix H matrix norms** (same operating point):

| H block                  | FULL          | LOCAL (after fix) |
| ------------------------ | ------------- | ------------------ |
| `H_V_bus  × PCC,set`     | 1.207 e-1     | **3.465 e-1**      |
| `H_Q_PCC  × PCC,set`     | 3.000         | 3.000              |
| `H_I_line × PCC,set`     | 1.873 e-3     | **2.973 e-3**      |
| `H_Q_gen  × PCC,set`     | 1.635         | **4.066**          |
| `H_Q_tie  × PCC,set`     | 9.170 e-1     | **3.255 e-1**      |

The LOCAL Q_PCC,set columns are now non-zero in every row block.  The
V-bus block is actually *larger* in LOCAL than in FULL — the Ward
boundary at MV makes the local model more responsive to PCC dispatch
than the full-net Jacobian's wider-area smearing.

**Runtime confirmation** (15-min LOCAL run, verbose=1):

```
[pcc-set z2 t=0] u_old=+16.37 -> u_new=-28.49  Δ=-44.86  bound=[-49.4,+73.5]  [FREE]
[pcc-set z2 t=1] u_old=+13.96 -> u_new=-31.23  Δ=-45.18  bound=[-31.8,+56.2]  [FREE]
[pcc-set z2 t=2] u_old=+32.29 -> u_new=-11.33  Δ=-43.63  bound=[-33.2,+90.0]  [FREE]
...
```

TSO is now actively dispatching Q_PCC,set across ~30-45 Mvar per TSO
tick, all entries `[FREE]` (not at any bound).  Bug fixed.

## Open follow-ups

* Compare V-tracking + Q-PCC-tracking error between baseline and local
  modes over a 24-hour profile run — quantify the cost of the
  Ward-equivalent approximation vs. the shared-net Jacobian.  Use the
  post-fix LOCAL configuration with retuned `g_w_pcc` (the original
  tuning assumed zero gradient on PCC columns; now that the gradient
  is properly populated, `g_w_pcc ≈ 50-200` is more appropriate).
* Optional: rescale synthetic-shunt `q_mvar` by
  `(V_tertiary/V_primary)²` from the cached operating point to make the
  per-step susceptance effect on the primary bus quantitatively match
  the original tertiary placement.
* Add `tests/diag_local_pcc_h.py` to the regression suite — it
  catches "missing PCC column block" bugs cheaply.
