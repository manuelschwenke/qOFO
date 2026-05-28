"""
Tests for the DSO Stage-2 closed-loop K(I+SK)^{-1} sensitivity transform.

We verify that the transformed sensitivities match what's observed
empirically: perturb V_ref by a small epsilon, run the plant with
``run_control=True`` (Q(V) loop active), measure the resulting Δoutput,
and compare against ``T @ ΔV_ref`` from the K-transform.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import install_qv_local_loops


# ---------------------------------------------------------------------------
#  Algebraic K-transform
# ---------------------------------------------------------------------------


def _k_transform_T(
    K_diag: np.ndarray,
    S_VQ: np.ndarray,
) -> np.ndarray:
    """Reference K (I + S_VQ K)^{-1} for verification."""
    n = len(K_diag)
    K = np.diag(K_diag)
    M = np.eye(n) + S_VQ @ K
    return np.linalg.solve(M.T, K.T).T


class TestKTransformAlgebra:
    def test_zero_K_gives_zero_T(self):
        """If every droop slope is infinite (k=0), V_ref shifts have no
        Q effect, so T must be zero."""
        S_VQ = np.array([[0.01, 0.001], [0.001, 0.02]])
        K_diag = np.array([0.0, 0.0])
        T = _k_transform_T(K_diag, S_VQ)
        np.testing.assert_array_almost_equal(T, np.zeros((2, 2)))

    def test_zero_S_VQ_gives_T_equal_K(self):
        """Without network coupling, V_ref shifts are perfectly local:
        ΔQ_i = k_i ΔV_ref,i, so T = K."""
        S_VQ = np.zeros((2, 2))
        K_diag = np.array([100.0, 50.0])
        T = _k_transform_T(K_diag, S_VQ)
        np.testing.assert_array_almost_equal(T, np.diag(K_diag))

    def test_diagonal_self_consistency(self):
        """For a single DER, T_11 = k / (1 + S_11 k)."""
        S_VQ = np.array([[0.005]])
        k = 200.0
        T = _k_transform_T(np.array([k]), S_VQ)
        expected = k / (1.0 + S_VQ[0, 0] * k)
        assert T[0, 0] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
#  End-to-end: K-transform vs. empirical perturbation in pandapower
# ---------------------------------------------------------------------------


def _build_two_der_net() -> tuple[pp.pandapowerNet, list[int], list[int]]:
    """Two DERs at distinct buses on a 110 kV branch."""
    net = pp.create_empty_network()
    b_slack = pp.create_bus(net, vn_kv=110.0, name="slack_bus")
    b_a = pp.create_bus(net, vn_kv=110.0, name="der_a_bus")
    b_b = pp.create_bus(net, vn_kv=110.0, name="der_b_bus")
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.00)
    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_a,
        length_km=15.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_line_from_parameters(
        net, from_bus=b_a, to_bus=b_b,
        length_km=15.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_load(net, bus=b_a, p_mw=20.0, q_mvar=10.0)
    pp.create_load(net, bus=b_b, p_mw=15.0, q_mvar=8.0)
    s_a = pp.create_sgen(
        net, bus=b_a, p_mw=10.0, q_mvar=0.0, sn_mva=100.0,
        type="WP", name="DER_A",
    )
    s_b = pp.create_sgen(
        net, bus=b_b, p_mw=10.0, q_mvar=0.0, sn_mva=100.0,
        type="WP", name="DER_B",
    )
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["vm_pu_ref"] = 1.03
    return net, [b_a, b_b], [int(s_a), int(s_b)]


class TestKTransformEmpirical:
    def test_dQ_matches_T_dV_ref(self):
        """Numerical check that the K(I+SK)^{-1} transform predicts
        observed Δ(realized Q) from a small ΔV_ref perturbation."""
        net, der_buses, sgens = _build_two_der_net()
        slope = 0.07
        install_qv_local_loops(
            net, sgens, slope_pu=slope,
            damping=0.5, max_step_frac=None, tol_mvar=0.1,
        )

        # Settle baseline.
        pp.runpp(net, run_control=True, max_iteration=200)
        q0 = np.array(
            [float(net.res_sgen.at[s, "q_mvar"]) for s in sgens]
        )
        # Compute S_VQ at the DER buses with a Jacobian-based primitive.
        from sensitivity.jacobian import JacobianSensitivities
        jac = JacobianSensitivities(net)
        S_VQ_full, obs_map, der_map = jac.compute_dV_dQ_der(
            der_bus_indices=der_buses,
            observation_bus_indices=der_buses,
        )
        obs_perm = [obs_map.index(b) for b in der_buses]
        der_perm = [der_map.index(b) for b in der_buses]
        S_VQ = S_VQ_full[np.ix_(obs_perm, der_perm)]
        sn = np.array([float(net.sgen.at[s, "sn_mva"]) for s in sgens])
        K_diag = sn / slope
        T = _k_transform_T(K_diag, S_VQ)

        # Perturb V_ref of DER A by a small amount and re-solve.
        eps = 0.005  # 0.5% pu
        net.sgen.at[sgens[0], "vm_pu_ref"] += eps
        pp.runpp(net, run_control=True, max_iteration=200)
        q1 = np.array(
            [float(net.res_sgen.at[s, "q_mvar"]) for s in sgens]
        )
        dq_observed = q1 - q0
        dV_ref = np.array([eps, 0.0])
        dq_predicted = T @ dV_ref

        # Linearization error around a non-zero operating point — allow
        # a few percent tolerance (system is nonlinear).
        rel_err = np.abs(dq_observed - dq_predicted) / max(
            float(np.linalg.norm(dq_observed)), 1e-3
        )
        assert np.all(rel_err < 0.10), (
            f"K-transform ΔQ prediction off by >10%: "
            f"observed={dq_observed}, predicted={dq_predicted}"
        )
