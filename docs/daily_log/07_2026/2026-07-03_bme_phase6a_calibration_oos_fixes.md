# BME Phase 6a — w_Φ calibration: OOS robustness fixes + sweep

**Date:** 2026-07-03
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Phase 6 work item (1) — `bme_gradient_scale` (w_Φ) calibration of
the bme ladder rung. Continues an interrupted session (session limit) whose
state was reconstructed from its scratchpad and the uncommitted working tree.

## Why the calibration sweep kept failing (three-layer diagnosis)

The 60-min CIGRE calibration scenario **trips gen 2 at minute 60**
(`experiments/005_CIGRE_MULTI.py` l. 204). The bme rung — unlike `none` —
runs `refresh_shared_jac_on_tso=True`, so the shared Jacobian is rebuilt at
the trip tick and the tripped machine's terminal bus is PRUNED from the
internal (ppci) bus set. Three code layers then failed in sequence, each
exposing the next:

1. **Loss gradient ppc/ppci branch misalignment** (previous session's fix,
   in `controller/common_objective.py::_build_loss_state_gradient`): Yf/Yt
   rows cover only in-service branches (`branch_is`); the ppc branch table
   keeps the OOS row → shapes (123,) vs (122,). Fixed by masking with
   `branch_is` + explicit alignment fail-fasts.
2. **BME H_{b,i}/g_own columns for disconnected actuators** (previous
   session's fix, finished and verified here):
   `sensitivity/boundary_sensitivity.py::actuator_active` — a
   tripped/isolated actuator keeps its u-column (alignment) but contributes
   an exactly-zero column/gradient entry, mirroring the controller's OOS
   masking. Applied in `_assemble_stacked` and
   `controller/bme_gradient.py::g_own`.
3. **Pre-existing latent bug in the controller's own H (fixed this
   session):** the Q_gen row block of
   `TSOController._build_sensitivity_matrix` passes the UNFILTERED
   `gen_terminal_buses` to the `compute_dQgen_*matrix` primitives (unlike
   the V_gen columns, which are OOS-filtered), and those primitives indexed
   the pruned bus straight into the internal arrays →
   `IndexError: index 99 out of bounds for size 99` in
   `_compute_dQgen_dx`. Latent because it needs BOTH a machine outage AND a
   refreshed Jacobian — `mode="none"` keeps its stale time-0 Jacobian and
   never sees the pruning.

## Fix (this session) — `sensitivity/jacobian.py`

New helper `JacobianSensitivities._ppc_bus_is_internal(ppc_idx)`: a ppc bus
participates in the solved system iff its row exists in the internal Ybus
(pandapower appends pruned buses after the internal set — the repo-wide
"dropped buses sit at the table end" layout). Guarded call sites, each
substituting the physically exact zero row/column for a disconnected
machine instead of indexing past the internal arrays:

* `compute_dQgen_dQder_matrix` (meas cache) — also covers
  `compute_dQgen_dQ_shunt_matrix` via delegation;
* `compute_dQgen_dVgen_matrix` (meas cache AND chg axis);
* `compute_dQgen_ds_2w_matrix`, `compute_dQgen_ds_3w_matrix` (meas caches);
* `compute_dQgen_dQ_shunt_matrix`: de-energised shunt bus (NaN vm) → zero
  column instead of NaN poisoning.

**Bitwise-safety argument for `mode="none"`/`vref`:** the guards change
behaviour only when a requested bus is pruned in the *sensitivity* net.
Those modes run on the frozen time-0 Jacobian in which every bus is
internal (empirically: the pre-fix `none` reference completed the identical
scenario), so the guards never fire there — no trajectory change. Verified
by the regression suite (see below).

## Verification

* All six files left modified by the interrupted session compile; the BME
  family (topology, boundary sensitivity, marginal computer, common
  objective, gradient-identity hard gate, coordination bus, discrete
  hygiene) passed 100/100 before the jacobian fix.
* After the jacobian fix: `test_jacobian_qgen.py` + hard gate
  `test_bme_gradient_identity.py` — 23 passed.
* `none` reference reproduces the interrupted session's numbers exactly
  (losses 29.581 first / 52.611 last / 32.920 mean-last-10 MW, 180 steps).

## Calibration sweep result

Sweep: `bme_gradient_scale ∈ {1e4, 1e5, 1e6, 1e7}`, 60-min CIGRE scenario
including the gen-2 trip (losses-only Φ, w_band = 0, d = 1, slotting on)
vs the `none` reference; metric = sustained losses (mean of last 10
steps) + run V extremes as the stability proxy:

| w_Φ | losses first/last/mean₁₀ [MW] | V range [pu] |
|---|---|---|
| none | 29.58 / 52.61 / 32.92 | [0.978, 1.049] |
| 1e4 | 29.65 / 54.19 / 33.49 | [0.968, 1.044] |
| 1e5 | 29.62 / 50.03 / 31.00 | [0.986, 1.059] |
| 1e6 | 29.29 / 46.93 / 30.16 | [0.991, 1.140] |
| 1e7 | 28.71 / 46.75 / 30.36 | [1.002, 1.179] |

**Chosen: w_Φ = 1e5** (−5.8 % sustained losses, voltage envelope
contained without the band hinge) — filled into
`experiments/011_BME_LADDER.py::BME_GRADIENT_SCALE`; full reasoning and
the two recorded findings (losses-only voltage escape confirms D2's band
rationale + inert `g_z_voltage` caveat for the ablation rung; over-drive
edge located at ~100× the chosen scale) in the Phase 6 §6a section of
`docs/BME_STATUS.md`.
