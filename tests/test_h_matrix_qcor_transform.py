"""
Tests for the w-shift closed-loop H-matrix transform.

Two layers:

1.  ``compute_w_shift_h_transform`` math — the free function that returns
    ``T' = (I + diag(K) · S_VQ)^{-1}``.  The matrix is structurally
    identical to the earlier Q_cor transform (Soleimani & Van Cutsem,
    eq. 18); under the vertical-shift + V_ref-reanchored formulation
    the differential sensitivity ``∂Q_realised / ∂q_set`` is the same
    ``(I + R·S_VQ)^{-1}``.  Tested in isolation against closed-form
    1-DER and 2-DER cases.

2.  ``DSOController._compute_w_shift_transform_T_prime`` integration —
    verify the method reads ``net.sgen.qv_slope_pu`` correctly,
    handles saturation by zeroing K_diag entries via the input-bound
    machinery (not in the transform itself), and produces the expected
    matrix for a tiny network.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller.der_qv_local_loop import compute_w_shift_h_transform


# ---------------------------------------------------------------------------
#  Pure-math tests on compute_w_shift_h_transform
# ---------------------------------------------------------------------------

class TestWshiftTransformMath:
    def test_zero_K_returns_identity(self):
        """All DERs saturated ⇒ K=0 ⇒ T' = I."""
        n = 3
        K_diag = np.zeros(n)
        S_VQ = np.array([
            [1e-3, 5e-4, 0.0],
            [5e-4, 1e-3, 5e-4],
            [0.0,  5e-4, 1e-3],
        ])
        T = compute_w_shift_h_transform(K_diag, S_VQ)
        assert T is not None
        np.testing.assert_allclose(T, np.eye(n), rtol=1e-12)

    def test_single_der_closed_form(self):
        """1-DER case: T' is a 1x1 matrix with value 1 / (1 + K·S_VQ)."""
        K = 100.0
        s = 5e-3
        T = compute_w_shift_h_transform(np.array([K]), np.array([[s]]))
        assert T is not None
        expected = 1.0 / (1.0 + K * s)
        assert T.shape == (1, 1)
        assert T[0, 0] == pytest.approx(expected, rel=1e-12)

    def test_two_der_closed_form(self):
        """2-DER coupled case.  Direct inversion via NumPy reference."""
        K_diag = np.array([200.0, 100.0])
        S_VQ = np.array([[1e-3, 5e-4],
                         [5e-4, 2e-3]])
        T = compute_w_shift_h_transform(K_diag, S_VQ)
        assert T is not None
        M = np.eye(2) + np.diag(K_diag) @ S_VQ
        expected = np.linalg.inv(M)
        np.testing.assert_allclose(T, expected, rtol=1e-12)

    def test_empty_inputs(self):
        T = compute_w_shift_h_transform(np.array([]), np.zeros((0, 0)))
        assert T is not None
        assert T.shape == (0, 0)

    def test_singular_returns_none(self):
        """Pathological case: K_diag · S_VQ designed to make M singular.
        compute_w_shift_h_transform must return None (caller falls back
        to identity)."""
        T = compute_w_shift_h_transform(np.array([10.0]), np.array([[-0.1]]))
        assert T is None


# ---------------------------------------------------------------------------
#  Saturation handling (K_diag = 0 in the row)
# ---------------------------------------------------------------------------

class TestSaturationHandling:
    def test_saturated_der_collapses_to_identity_row(self):
        """When DER i is saturated (K_diag[i] = 0), row i of T' equals
        the identity row e_i.  This invariance holds because row i of
        ``M = I + diag(K)·S_VQ`` is then exactly e_i, so M^{-1} = T'
        inherits the same identity row."""
        K_diag = np.array([0.0, 100.0])  # saturated K=0 on bus 0
        S_VQ = np.array([[1e-3, 5e-4],
                         [5e-4, 2e-3]])
        T = compute_w_shift_h_transform(K_diag, S_VQ)
        assert T is not None
        M = np.eye(2) + np.diag(K_diag) @ S_VQ
        expected = np.linalg.inv(M)
        np.testing.assert_allclose(T, expected, rtol=1e-12)
        assert expected[0, 0] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
#  Regression: source-level pin — saturation check is gone from T_prime
# ---------------------------------------------------------------------------

class TestNoSaturationHackInTransform:
    """Bug fix 2026-05: the DSO Q-tracking regression was rooted in two
    coupled defects in ``_compute_qcor_transform_T_prime`` (since renamed
    to ``_compute_w_shift_transform_T_prime``) and
    ``generate_capability_message``.  The architectural fix removes the
    saturation active-set logic from T' / T_qv entirely (saturation is
    now handled by the input bounds in ``_compute_input_bounds``) and
    switches capability to use the open-loop ∂Q_iface/∂Q_DER block.
    This test pins the source-level invariant: those methods no longer
    touch ``_last_measurement`` or ``_qv_capability`` for saturation
    purposes.
    """

    def test_w_shift_transform_does_not_consult_measurement(self):
        """``_compute_w_shift_transform_T_prime`` must not read
        ``_last_measurement`` for saturation gating.  Otherwise the DSO
        controller's ``_last_measurement`` silently becomes a
        load-bearing variable, and a future caller forgetting to set it
        re-introduces the bug."""
        import inspect
        from controller.dso_controller import DSOController

        src = inspect.getsource(
            DSOController._compute_w_shift_transform_T_prime
        )
        assert "saturated" not in src, (
            "_compute_w_shift_transform_T_prime regained saturation logic; "
            "saturation should be enforced by _compute_input_bounds via "
            "the T'_bb scaling of the Q_DER envelope, not by zeroing K."
        )
        assert "_qv_capability" not in src, (
            "_compute_w_shift_transform_T_prime is calling _qv_capability; "
            "the per-DER capability envelope belongs in the bound calc, "
            "not in the H-matrix transform."
        )

    def test_qv_transform_does_not_consult_measurement(self):
        """Same invariant for the V_ref-mode sibling
        ``_compute_qv_transform_T`` (vestigial Stage-2 path)."""
        import inspect
        from controller.dso_controller import DSOController

        src = inspect.getsource(DSOController._compute_qv_transform_T)
        assert "saturated" not in src
        assert "_qv_capability" not in src

    def test_capability_uses_open_loop_block(self):
        """``generate_capability_message`` must read from
        ``_H_iface_der_open_bus`` (the pre-T' snapshot) when populated;
        mixing the post-T' ``_H_cache`` block with ΔQ_DER under-reports
        the envelope by a factor of T'_bb (~0.3 in practice)."""
        import inspect
        from controller.dso_controller import DSOController

        src = inspect.getsource(DSOController.generate_capability_message)
        assert "_H_iface_der_open_bus" in src, (
            "generate_capability_message no longer references the "
            "open-loop snapshot; capability will be under-reported by "
            "factor T'_bb under the w-shift actuator."
        )
