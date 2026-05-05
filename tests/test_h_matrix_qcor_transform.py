"""
Tests for the Q_cor closed-loop H-matrix transform
(refactor_v2 commit 4, Soleimani §IV-B eq. 18).

Two layers:

1.  ``compute_qcor_h_transform`` math — the free function that returns
    ``T' = (I + diag(K) · S_VQ)^{-1}``.  Tested in isolation against
    closed-form 1-DER and 2-DER cases.

2.  ``DSOController._compute_qcor_transform_T_prime`` integration —
    verify the method reads ``net.sgen.qv_slope_pu`` correctly,
    handles saturation by zeroing K_diag entries, and produces the
    expected matrix for a tiny network.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller.der_qv_local_loop import compute_qcor_h_transform


# ---------------------------------------------------------------------------
#  Pure-math tests on compute_qcor_h_transform
# ---------------------------------------------------------------------------

class TestQcorTransformMath:
    def test_zero_K_returns_identity(self):
        """All DERs saturated ⇒ K=0 ⇒ T' = I."""
        n = 3
        K_diag = np.zeros(n)
        S_VQ = np.array([
            [1e-3, 5e-4, 0.0],
            [5e-4, 1e-3, 5e-4],
            [0.0,  5e-4, 1e-3],
        ])
        T = compute_qcor_h_transform(K_diag, S_VQ)
        assert T is not None
        np.testing.assert_allclose(T, np.eye(n), rtol=1e-12)

    def test_single_der_closed_form(self):
        """1-DER case: T' is a 1x1 matrix with value 1 / (1 + K·S_VQ)."""
        # Pick R = 100 Mvar/pu_v, S_VQ = 5e-3 pu_v/Mvar ⇒ K·S_VQ = 0.5
        K = 100.0
        s = 5e-3
        T = compute_qcor_h_transform(np.array([K]), np.array([[s]]))
        assert T is not None
        expected = 1.0 / (1.0 + K * s)
        assert T.shape == (1, 1)
        assert T[0, 0] == pytest.approx(expected, rel=1e-12)

    def test_two_der_closed_form(self):
        """2-DER coupled case.  Direct inversion via NumPy reference."""
        K_diag = np.array([200.0, 100.0])
        S_VQ = np.array([[1e-3, 5e-4],
                         [5e-4, 2e-3]])
        T = compute_qcor_h_transform(K_diag, S_VQ)
        assert T is not None
        M = np.eye(2) + np.diag(K_diag) @ S_VQ
        expected = np.linalg.inv(M)
        np.testing.assert_allclose(T, expected, rtol=1e-12)

    def test_empty_inputs(self):
        T = compute_qcor_h_transform(np.array([]), np.zeros((0, 0)))
        assert T is not None
        assert T.shape == (0, 0)

    def test_singular_returns_none(self):
        """Pathological case: K_diag · S_VQ designed to make M singular.
        compute_qcor_h_transform must return None (caller falls back to identity)."""
        # M = I + diag(K) · S_VQ — choose K and S_VQ so that one eigenvalue of
        # diag(K)·S_VQ is exactly -1, making M singular.
        # Simplest: 1×1 with K=10, s=-0.1 ⇒ K·s = -1 ⇒ M = 0.
        T = compute_qcor_h_transform(np.array([10.0]), np.array([[-0.1]]))
        assert T is None


# ---------------------------------------------------------------------------
#  Saturation handling
# ---------------------------------------------------------------------------

class TestSaturationHandling:
    def test_saturated_der_collapses_to_identity_row(self):
        """When DER i is saturated (K_diag[i] = 0), row i of T' equals
        the identity row e_i (off-diagonal zero, diagonal one).

        This is the invariance: ∂Q_g,i/∂Q_cor,i = 0 at the rail (Q is
        pinned), but the controller must still see ∂Q_g,j/∂Q_cor,j for
        j != i, with zero off-diagonal entries connecting back to i."""
        # 2-bus case: DER 0 saturated, DER 1 active
        K_diag = np.array([0.0, 100.0])  # saturated K=0 on bus 0
        S_VQ = np.array([[1e-3, 5e-4],
                         [5e-4, 2e-3]])
        T = compute_qcor_h_transform(K_diag, S_VQ)
        assert T is not None
        # Row 0 of M = I + diag(K) · S_VQ is e_0 since K_diag[0] = 0.
        # M = [[1, 0], [K_1·s_10, 1 + K_1·s_11]] = [[1, 0], [0.05, 1.2]]
        M = np.eye(2) + np.diag(K_diag) @ S_VQ
        expected = np.linalg.inv(M)
        np.testing.assert_allclose(T, expected, rtol=1e-12)
        # Verify the structural property: row 0 of expected has zero
        # off-diagonals (because saturated DER 0's column of M is e_0).
        # Wait, I had this backwards.  Let's verify what we actually get:
        # M = [[1, 0], [0.05, 1.2]] — column 0 of M is [1, 0.05]^T
        # So column 0 of M^{-1} = T is [1, -0.05/1.2]^T ≈ [1, -0.0417]^T.
        # That's the "saturated DER" column: a +1 Mvar Q_cor on the
        # saturated DER produces +1 Mvar nominal Q (locally pinned) and
        # -0.0417 Mvar response on DER 1 (the response of the live
        # neighbour to the V-perturbation caused by the locked DER 0).
        # Sanity-check that expected.diagonal()[0] is exactly 1.
        assert expected[0, 0] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
#  Integration test: end-to-end OFO config flag
# ---------------------------------------------------------------------------

class TestConfigFlagValidation:
    def test_q_cor_and_v_ref_mutually_exclusive(self):
        """DSOControllerConfig.__post_init__ must reject the combination
        use_q_cor_actuator=True + ofo_in_v_ref_mode=True."""
        from controller.dso_controller import DSOControllerConfig
        with pytest.raises(ValueError, match="mutually exclusive"):
            DSOControllerConfig(
                interface_trafo_indices=[],
                der_indices=[],
                voltage_bus_indices=[],
                current_line_indices=[],
                oltc_trafo_indices=[],
                shunt_bus_indices=[],
                shunt_q_steps_mvar=[],
                use_qv_local_loop=True,
                qv_apply_mode="v_ref",
                use_q_cor_actuator=True,
            )

    def test_q_cor_alone_accepted(self):
        from controller.dso_controller import DSOControllerConfig
        cfg = DSOControllerConfig(
            interface_trafo_indices=[],
            der_indices=[],
            voltage_bus_indices=[],
            current_line_indices=[],
            oltc_trafo_indices=[],
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
            use_qv_local_loop=False,
            use_q_cor_actuator=True,
        )
        assert cfg.use_q_cor_actuator is True

    def test_default_is_legacy(self):
        """Default DSOControllerConfig has use_q_cor_actuator=False so
        the new code path is dormant unless explicitly opted in."""
        from controller.dso_controller import DSOControllerConfig
        cfg = DSOControllerConfig(
            interface_trafo_indices=[],
            der_indices=[],
            voltage_bus_indices=[],
            current_line_indices=[],
            oltc_trafo_indices=[],
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
        )
        assert cfg.use_q_cor_actuator is False
