"""
Tests for :class:`controller.der_qv_local_loop.CosPhiConstLoop`
(refactor_v2 commit 3).

Q follows the configured cos(φ) and the active-power measurement; it is
independent of voltage.  The loop should:

* Compute ``Q = sign · |P| · tan(acos(cosφ))`` and clip to capability.
* Converge in a single damped step (no V feedback, so single-iteration
  convergence is the expected behaviour at damping=1.0).
"""

from __future__ import annotations

import math

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import (
    CosPhiConstLoop,
    QVLocalLoop,
    _qv_capability,
    install_der_q_loops,
)


def _build_tiny_cosphi_net(
    *,
    cosphi: float = 1.0,
    cosphi_sign: int = -1,
    sn_mva: float = 100.0,
    p_mw: float = 10.0,
) -> tuple[pp.pandapowerNet, int, int]:
    """Two-bus network with one slack and one cosphi-mode sgen."""
    net = pp.create_empty_network()
    b_slack = pp.create_bus(net, vn_kv=110.0, name="slack_bus")
    b_load = pp.create_bus(net, vn_kv=110.0, name="load_bus")
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b_slack, to_bus=b_load,
        length_km=10.0,
        r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_load(net, bus=b_load, p_mw=20.0, q_mvar=10.0)
    sgen_idx = pp.create_sgen(
        net, bus=b_load, p_mw=p_mw, q_mvar=0.0, sn_mva=sn_mva,
        type="WP", name="COSPHI_TEST",
    )
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["q_mode"] = "cosphi"
    net.sgen["qv_slope_pu"] = 0.07
    net.sgen["qv_vref_pu"] = 1.00
    net.sgen["qv_deadband_pu"] = 0.0
    net.sgen["cosphi"] = cosphi
    net.sgen["cosphi_sign"] = cosphi_sign
    net.sgen["q_cor_mvar"] = 0.0
    return net, b_load, int(sgen_idx)


# ---------------------------------------------------------------------------
#  Q = sign * |P| * tan(acos(cosphi))
# ---------------------------------------------------------------------------

class TestQTarget:
    def test_unity_pf_yields_zero_q(self):
        net, _, sgen = _build_tiny_cosphi_net(cosphi=1.0, p_mw=10.0)
        loop = CosPhiConstLoop(net, sgen_idx=sgen)
        pp.runpp(net, run_control=False)
        target = loop._compute_target(net)
        assert target == pytest.approx(0.0, abs=1e-9)

    def test_under_excited_negative_q(self):
        # cosphi=0.95 ⇒ tan(phi) = sqrt(1/0.9025 - 1) = sqrt(0.108...) ≈ 0.3287
        net, _, sgen = _build_tiny_cosphi_net(
            cosphi=0.95, cosphi_sign=-1, p_mw=10.0,
        )
        loop = CosPhiConstLoop(net, sgen_idx=sgen)
        pp.runpp(net, run_control=False)
        target = loop._compute_target(net)
        tan_phi = math.sqrt(1.0 / 0.95**2 - 1.0)
        expected = -1.0 * 10.0 * tan_phi
        assert target == pytest.approx(expected, rel=1e-6)

    def test_over_excited_positive_q(self):
        net, _, sgen = _build_tiny_cosphi_net(
            cosphi=0.90, cosphi_sign=+1, p_mw=10.0,
        )
        loop = CosPhiConstLoop(net, sgen_idx=sgen)
        pp.runpp(net, run_control=False)
        target = loop._compute_target(net)
        tan_phi = math.sqrt(1.0 / 0.90**2 - 1.0)
        expected = +1.0 * 10.0 * tan_phi
        assert target == pytest.approx(expected, rel=1e-6)

    def test_q_independent_of_voltage(self):
        """Cos-phi mode: Q must NOT depend on V_meas — only on |P| and cosφ."""
        net, bus, sgen = _build_tiny_cosphi_net(
            cosphi=0.92, cosphi_sign=-1, p_mw=10.0,
        )
        loop = CosPhiConstLoop(net, sgen_idx=sgen)
        pp.runpp(net, run_control=False)
        # Test at three different bus voltages
        for v in (0.95, 1.00, 1.05):
            net.res_bus.at[bus, "vm_pu"] = v
            target = loop._compute_target(net)
            tan_phi = math.sqrt(1.0 / 0.92**2 - 1.0)
            expected = -10.0 * tan_phi
            assert target == pytest.approx(expected, rel=1e-6), \
                f"Q changed with V at V={v}: {target} vs {expected}"

    def test_q_clipped_to_capability_when_p_is_high(self):
        """If P is at full Sn, STATCOM has Q-capability = 0, so Q is clipped to 0."""
        net, _, sgen = _build_tiny_cosphi_net(
            cosphi=0.80, cosphi_sign=-1, p_mw=100.0, sn_mva=100.0,
        )
        loop = CosPhiConstLoop(net, sgen_idx=sgen)
        pp.runpp(net, run_control=False)
        target = loop._compute_target(net)
        assert target == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
#  PF convergence under run_control=True
# ---------------------------------------------------------------------------

class TestPFConvergence:
    def test_q_settles_at_target_after_pf(self):
        """With damping=1.0 and a converged PF, the realised Q on the
        sgen should match the cos-phi target within tolerance."""
        net, _, sgen = _build_tiny_cosphi_net(
            cosphi=0.95, cosphi_sign=-1, p_mw=10.0,
        )
        install_der_q_loops(net, [sgen],
                            cosphi_damping=1.0, cosphi_tol_mvar=0.01)
        pp.runpp(net, run_control=True, max_iteration=50)
        q = float(net.res_sgen.at[sgen, "q_mvar"])
        tan_phi = math.sqrt(1.0 / 0.95**2 - 1.0)
        expected = -10.0 * tan_phi
        assert q == pytest.approx(expected, abs=0.05)


# ---------------------------------------------------------------------------
#  Installer dispatches CosPhiConstLoop for q_mode == "cosphi"
# ---------------------------------------------------------------------------

class TestInstallDispatch:
    def test_install_creates_cosphi_loop(self):
        net, _, sgen = _build_tiny_cosphi_net()
        idx_list = install_der_q_loops(net, [sgen])
        assert len(idx_list) == 1
        ctrl_obj = net.controller.at[idx_list[0], "object"]
        assert isinstance(ctrl_obj, CosPhiConstLoop)

    def test_mixed_modes_dispatch_correctly(self):
        """A network with one qv-mode and one cosphi-mode sgen gets one of
        each plant-side controller, on the right rows."""
        net, _, sgen0 = _build_tiny_cosphi_net()
        # Add a second sgen as qv-mode at the same bus
        sgen1 = pp.create_sgen(
            net, bus=net.sgen.at[sgen0, "bus"], p_mw=5.0, q_mvar=0.0,
            sn_mva=50.0, type="WP", name="QV_SECOND",
        )
        # Re-extend column defaults for the new row
        net.sgen.at[sgen1, "op_diagram"] = "STATCOM"
        net.sgen.at[sgen1, "q_mode"] = "qv"
        net.sgen.at[sgen1, "qv_slope_pu"] = 0.07
        net.sgen.at[sgen1, "qv_vref_pu"] = 1.00
        net.sgen.at[sgen1, "qv_deadband_pu"] = 0.0
        net.sgen.at[sgen1, "cosphi"] = 1.0
        net.sgen.at[sgen1, "cosphi_sign"] = -1
        net.sgen.at[sgen1, "q_cor_mvar"] = 0.0

        idx_list = install_der_q_loops(net, [sgen0, sgen1])
        assert len(idx_list) == 2

        # Map sgen → controller class
        installed: dict[int, type] = {}
        for _, row in net.controller.iterrows():
            obj = row["object"]
            if isinstance(obj, (QVLocalLoop, CosPhiConstLoop)):
                installed[obj.sgen_idx] = type(obj)

        assert installed[sgen0] is CosPhiConstLoop
        assert installed[sgen1] is QVLocalLoop


# ---------------------------------------------------------------------------
#  Q_cor is ignored for cos-phi DERs
# ---------------------------------------------------------------------------

class TestQCorIgnored:
    """cos-phi DERs are excluded from the OFO action vector — the
    controller's Q_cor write into ``net.sgen.q_cor_mvar`` for these
    DERs has no effect on Q.  This test confirms the loop ignores
    q_cor_mvar."""

    def test_q_cor_does_not_affect_target(self):
        net, _, sgen = _build_tiny_cosphi_net(
            cosphi=0.95, cosphi_sign=-1, p_mw=10.0,
        )
        loop = CosPhiConstLoop(net, sgen_idx=sgen)
        pp.runpp(net, run_control=False)
        net.sgen.at[sgen, "q_cor_mvar"] = 999.0
        target = loop._compute_target(net)
        tan_phi = math.sqrt(1.0 / 0.95**2 - 1.0)
        expected = -10.0 * tan_phi
        assert target == pytest.approx(expected, rel=1e-6)
