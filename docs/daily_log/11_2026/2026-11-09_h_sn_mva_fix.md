# 2026-11-09 — sn_mva fix of the Q-row sensitivities (fixes A + B) and γ in the STS screen

Session "STS capabilities tuning analysis". Branch **`res/h-sn-mva-fix-2026-11-09`** (from
`res/qgen-row-check-2026-11-08` @ 90aaea0, itself from `res/tuning-v5-2026-11-08`), worktree
`F:\qOFO_qgen`. Diagnosis: `2026-11-08d_qgen_row_check.md`. Not merged; procedure v5 (run 0002)
keeps running on the as-built code.

## Author decision (2026-11-09, ~02:10)
- "Fix in parallel": v5 keeps running; fixes A+B on a separate branch, then decide with numbers.
- Author's comment during the work: "we set the influence of a tap change on reactive power flow
  zero anyway (dso_gamma_oltc_q is the factor), so tuning v5 should be fine". Checked; see below.

## Code changes (what, how, why)
`sensitivity/jacobian.py` — per-unit results on the system base are now returned in Mvar, as the
docstrings state and as the controllers read them (`× float(self.net.sn_mva)`):

| function | quantity | change |
|---|---|---|
| `compute_dQgen_dVgen_matrix` | Q_gen × V_set (A) | × sn_mva |
| `compute_dQgen_ds_2w_matrix` | Q_gen × 2W tap (A) | × sn_mva |
| `compute_dQgen_ds_3w_matrix` | Q_gen × 3W tap (A) | × sn_mva |
| `compute_dQtrafo3w_hv_ds` | Q_HV(3W) × 3W tap (B, STS interface Q) | × sn_mva |
| `compute_dQtrafo_2w_ds` | Q_trafo × 2W tap (B) | × sn_mva (its comment already said "scaled to Mvar") |
| `compute_dQtrafo3w_hv_dVgen` | Q_PCC × V_set (B) | × sn_mva **and** full chain ∂Q_HV/∂x · ∂x/∂V_gen (θ and V terms, via `_compute_dQhv_dx_3w` and `_compute_dg_dVk`); the V-only chain had the wrong sign on these small entries |

Q rows per *power* input (∂Q/∂Q_IBG, ∂Q/∂Q_shunt) are dimensionless and unchanged.

`tuning/stage1_screen.py::_dso_curvature` and `controller/dso_controller.py::objective_curvature_inputs`:
the OLTC columns of the interface-Q rows are multiplied by `gamma_oltc_q`, the factor with which
the DSO step sees them (`_compute_objective_gradient`; γ² ‖a_q‖² in the engage formula). With the
as-built entries (100× too small) ignoring γ = 0 was harmless; with fix B it is not (below).

## Tests
- Corrected the three tests that encoded the slip:
  - `test_jacobian_qgen.py`: no `× sn_mva`, ratio band 0.95–1.05 (was 0.3–3);
  - `test_jacobian_tap.py`: no `× sn_mva`;
  - `test_jacobian.py::test_dQhv_ds_vs_finite_difference`: no `× delta_tau` on the FD side, rtol
    0.02 (was 0.20; the 1.222 %/step fixture made the two errors nearly cancel).
- New `tests/test_jacobian_sn_mva.py`: base invariance (sn_mva 100 vs 1) of all six functions, and
  FD at both bases for Q_trafo2w × tap, Q_gen × 3W tap and Q_PCC × V_set (sign included).
- New `tests/test_stage1_screen_gamma.py`: γ = 0 removes the Q part of the tap column, γ = 1
  keeps the Gram, γ scales quadratically.
- Targeted run: 153 (Jacobian, controller, sensitivity updater) + 43 (new, procedure v4/v5,
  preconditioner) passed. Full suite: see below.

## Does the fix change procedure v5? (`2026-11-09_h_sn_mva_fix_screen.py/.out`)
Run 0002's λ bank (6 design + 186 grid points), banks excluded; worst λ per layer type:

| weights | as built | fixed, γ = 0 | fixed, γ = 1 (no γ companion) |
|---|---|---|---|
| v4 | TS 1.460, STS 1.4389 | TS 1.460, STS 1.4390 | STS 13.60 |
| v5 Stage I | TS 0.984, STS 1.4092 | STS 1.4093 | STS 12.99 |
| v5 II-a leader | TS 1.374, STS 1.4092 | STS 1.4093 | STS 12.99 |

- Established: with γ = 0 the screen changes by +7·10⁻⁵, so v5's Stage I and the II-a/II-b
  screens hold under the fix. The author's comment is correct for the DSO.
- Established: the DSO MIQP sees the Q × tap entries only through the γ-scaled gradient, and the
  interface Q has no output bounds (±1e6), so fix B does not change a DSO step.
- Retracted: my 2026-11-08 statement "once fix B is applied, v5's STS selection is obsolete"; it
  ignored γ = 0. The subagent's λ up to 16.4 was the γ = 1 case.
- What fix A still changes (independent of γ): the TSO soft capability constraint on machine Q
  (`g_z_q_gen = 100`). It acts only when a machine is predicted past its capability bound; it was
  inactive at the two FD points. Whether W1–W6 or the MC episodes reach a bound has not been checked.

## Open
- Full test suite on this branch (running at the time of writing).
- Q_PCC × IBG entries (`compute_dQtrafo3w_hv_dQ_der`) use the decoupled ∂V/∂Q chain without θ
  terms; FD/H −0.2…−1.6 on the 4 TS entries (small). Not changed: it is the same construction the
  DSO uses for its own interface-Q × IBG rows, which needs its own check.
- G3 capability bound in the controller 668 Mvar vs plant table limit 400 Mvar (subagent, not
  verified here).
- Merge into the v5 line only after v5 finishes (its e5power sync copies differing Python files).
