# 2026-06-16 — Gate the QV closed-loop H-transform T' off by default

## Reason
The dissertation (Ch. 4, §4.6.3) was rewritten to present the DER actuator as the
reactive-power setpoint `q_DER` (offset `q_0`) commanded **directly**, with the
reference-anchoring (CIGRE 2026, eq. qv/anchor) centring the deadband at every
dispatch — so at the dispatch point `dQ_DER/dq_set = 1` and the OFO uses the
**bare** sensitivity `H = dy/dQ_DER`, no closed-loop transform. The code, however,
still post-multiplied the DER columns of H by `T' = (I + diag(K)·S_VQ)^{-1}`
(a sloping-segment correction). Decision (with M. Schwenke): make the code match
the thesis by defaulting to bare H, behind a flag so the legacy behaviour is
recoverable.

## What changed
New config flag **`apply_qv_h_transform: bool = False`** on both controller configs:
- `controller/dso_controller.py` (`DSOControllerConfig`)
- `controller/tso_controller.py` (`TSOControllerConfig`)

`False` (default) => bare `H = dy/dQ_DER` and physical Q_DER input bounds.
`True` => legacy `T'` transform on the DER columns of H (+ `T'_bb` input-bound scaling).

Gated the three T' application sites by the flag:
1. `dso_controller.py` (~L1290): `T_prime` is computed only if the flag is set;
   otherwise `T_prime=None`, the existing cache-clear branch runs, and
   `_compute_input_bounds` reverts to physical Q_DER bounds (no `T'_bb` scaling).
2. `tso_controller.py` (~L1980): `H[:, :n_der] @ T_prime` applied only if the flag is set.
3. `sensitivity/numerical_h.py` (`compute_numerical_h_tso` ~L384,
   `compute_numerical_h_dso` ~L542): open-loop FD T' application gated by the flag.
   (Closed-loop numerical H, `numerical_h_closed_loop=True`, is unchanged — it bakes
   the droop response into the perturbation directly and never applied T'.)

The free function `compute_w_shift_h_transform` and `_compute_w_shift_transform_T_prime`
are left in place (used when the flag is True; pure-math tests unaffected).
Dead method `_apply_qv_closed_loop_transform` (dso) untouched (still dead).
The non-DER OLTC/shunt term (`_apply_w_shift_closedloop_to_non_der`) remains
env-gated off (`DSO_CLOSED_LOOP_OLTC`).

## Verification
- `py_compile` on dso_controller.py, tso_controller.py, numerical_h.py — OK.
- `pytest tests/test_h_matrix_qcor_transform.py tests/test_qv_deadband.py
  tests/test_qv_local_loop.py tests/test_tag_der_q_modes.py
  tests/test_cosphi_const_loop.py -q` => 51 passed.

## Impact / TODO
- **Behaviour change:** all experiment configs (002/003/005/006, multi_tso_config)
  now run with bare H by default. Results generated before this commit used T'.
  **Re-run the case studies** to regenerate the dissertation numbers under bare H.
  Near the anchored operating point `T' ≈ I`, so the change is expected to be small,
  but it must be confirmed empirically.
- To reproduce pre-2026-06-16 results, set `apply_qv_h_transform=True` on the
  TSO/DSO configs.
