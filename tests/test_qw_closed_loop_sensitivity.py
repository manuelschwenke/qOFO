"""
Finite-difference check on the w-shift closed-loop sensitivity matrix
``T' = (I + R·S_VQ)^{-1}``.

Under the vertical-shift + V_ref-reanchored formulation, the realised
Q at the DER buses responds to a per-bus ``q_set`` perturbation with

  ΔQ_DER ≈ T' · Δq_set     (single-step, at the reanchored equilibrium).

This test:

1. Runs a base PF on a tiny single-DER network with reanchored V_ref.
2. Computes the analytical T' from the per-DER ``K = S_n / qv_slope_pu``
   and the bus-self ``S_VQ_ii`` from the network's reduced Jacobian.
3. Perturbs ``q_set`` by ε, runs PF, and confirms ``ΔQ ≈ T'_ii · ε``
   to within a few percent.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import (
    QVLocalLoop,
    compute_w_shift_h_transform,
    install_der_q_loops,
)
from sensitivity.jacobian import JacobianSensitivities


def _build_net() -> tuple[pp.pandapowerNet, int, int]:
    net = pp.create_empty_network()
    b_slack = pp.create_bus(net, vn_kv=110.0)
    b_load = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_load,
        length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_load(net, bus=b_load, p_mw=20.0, q_mvar=10.0)
    sgen = pp.create_sgen(net, bus=b_load, p_mw=10.0, q_mvar=0.0,
                          sn_mva=100.0, type="WP", name="der")
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["q_mode"] = "qv"
    net.sgen["qv_slope_pu"] = 0.07
    net.sgen["qv_vref_pu"] = 1.00
    net.sgen["qv_deadband_pu"] = 0.0  # disable deadband so the linear segment dominates
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_set_mvar"] = 0.0
    net.sgen["qv_vref_anchor_pu"] = float("nan")
    return net, int(b_load), int(sgen)


def test_finite_difference_matches_t_prime():
    """Perturb ``q_set`` by ε and confirm ΔQ ≈ T'·ε to within 5 %.

    The test is intentionally simple: one DER on a stiff feeder so the
    sensitivities are smooth.  T'_ii is computed analytically from
    ``K = S_n / qv_slope_pu`` and the bus-self S_VQ entry.
    """
    net, bus, sgen = _build_net()
    install_der_q_loops(net, [sgen], qv_damping=0.5,
                        qv_max_step_frac=None, qv_tol_mvar=0.001)

    # Base PF — establish the reanchored operating point.
    pp.runpp(net, run_control=True, max_iteration=200)
    net.sgen.at[sgen, "qv_vref_anchor_pu"] = float(
        net.res_bus.at[bus, "vm_pu"]
    )
    pp.runpp(net, run_control=True, max_iteration=200)
    q_base = float(net.res_sgen.at[sgen, "q_mvar"])

    # Compute T'_ii analytically.
    sn = float(net.sgen.at[sgen, "sn_mva"])
    slope = float(net.sgen.at[sgen, "qv_slope_pu"])
    K = sn / slope
    sens = JacobianSensitivities(net)
    S_VQ_full, obs_map, der_map = sens.compute_dV_dQ_der(
        der_bus_indices=[bus], observation_bus_indices=[bus],
    )
    s_base = float(getattr(net, "sn_mva", 1.0))
    if s_base <= 0:
        s_base = 1.0
    s_vq = float(S_VQ_full[0, 0]) / s_base
    T = compute_w_shift_h_transform(np.array([K]), np.array([[s_vq]]))
    assert T is not None and T.shape == (1, 1)
    T_prime_ii = float(T[0, 0])

    # Perturb q_set by ε.  Pick ε small but well above floating-point noise.
    eps = 2.0  # Mvar
    net.sgen.at[sgen, "q_set_mvar"] = eps
    # Reanchor stays — the perturbation simulates a single OFO step that
    # commands a Δq_set increment.
    pp.runpp(net, run_control=True, max_iteration=200)
    q_pert = float(net.res_sgen.at[sgen, "q_mvar"])

    delta_q = q_pert - q_base
    expected = T_prime_ii * eps
    rel_err = abs(delta_q - expected) / max(abs(expected), 1e-9)
    assert rel_err < 0.05, (
        f"FD sensitivity mismatch: ΔQ={delta_q:.4f} Mvar, "
        f"expected T'·ε={expected:.4f} Mvar "
        f"(T'_ii={T_prime_ii:.4f}, ε={eps}); rel_err={rel_err:.3f}."
    )
