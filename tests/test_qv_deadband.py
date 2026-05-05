"""
Tests for the piecewise-linear V_Q characteristic with deadband and Q_cor
offset (refactor_v2 commit 3, Soleimani §III-A eq. (2) + §III-B eq. (1)).

The test network is a tiny 2-bus PF where the slack bus voltage and a
single sgen at the load bus give us full control over V at the DER bus.
"""

from __future__ import annotations

import math

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import (
    QVLocalLoop,
    _qv_capability,
    install_der_q_loops,
)


def _build_tiny_qv_net(
    *,
    slope_pu: float = 0.07,
    vref_pu: float = 1.00,
    deadband_pu: float = 0.0,
    q_cor_mvar: float = 0.0,
    sn_mva: float = 100.0,
    p_mw: float = 10.0,
) -> tuple[pp.pandapowerNet, int, int]:
    """Two-bus network with one slack and one sgen.  q_mode columns
    are populated so install_der_q_loops dispatches QVLocalLoop."""
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
        type="WP", name="QV_TEST",
    )
    net.sgen["op_diagram"] = "STATCOM"
    # New q_mode columns
    net.sgen["q_mode"] = "qv"
    net.sgen["qv_slope_pu"] = slope_pu
    net.sgen["qv_vref_pu"] = vref_pu
    net.sgen["qv_deadband_pu"] = deadband_pu
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_cor_mvar"] = q_cor_mvar
    return net, b_load, int(sgen_idx)


# ---------------------------------------------------------------------------
#  Symmetric deadband around V_ref (Q_cor = 0)
# ---------------------------------------------------------------------------

class TestDeadbandStaticTarget:
    """Test ``_compute_target`` directly without running PF.  We pin
    V_ref / V_meas to specific values via the post-PF state and let the
    QVLocalLoop compute the target.  This isolates the math from the
    PF solver."""

    def test_inside_deadband_q_target_zero(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.03, q_cor_mvar=0.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        # Stage a PF result manually so _compute_target sees what we want.
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.005   # within ±0.03 of V_ref=1.00
        net.res_sgen.at[sgen, "p_mw"] = 10.0
        target = loop._compute_target(net)
        assert target == pytest.approx(0.0, abs=1e-9)

    def test_just_outside_deadband_upper(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.03, q_cor_mvar=0.0,
            sn_mva=100.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        # V_eff = 1.05 - 1.00 - 0 = +0.05; segment: V_eff > db ⇒ Q = -R*(0.05-0.03)
        net.res_bus.at[bus, "vm_pu"] = 1.05
        net.res_sgen.at[sgen, "p_mw"] = 10.0
        R = 100.0 / 0.07  # ≈ 1428.57
        # Q_d = -R*0.02 ≈ -28.57; STATCOM cap at P=10 is ±99.499 ⇒ no clip.
        expected = -R * 0.02
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)

    def test_just_outside_deadband_lower(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.40, vref_pu=1.00, deadband_pu=0.03, q_cor_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.40)
        pp.runpp(net, run_control=False)
        # V_eff = 0.95 - 1.00 - 0 = -0.05; segment: V_eff < -db ⇒ Q = -R*(V_eff+db) = -R*(-0.05+0.03) = +R*0.02
        net.res_bus.at[bus, "vm_pu"] = 0.95
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        # Use slope=0.40 and Sn=10 so demanded Q = (10/0.40)*0.02 = 0.5 Mvar — well inside cap.
        R = 10.0 / 0.40
        expected = R * 0.02  # = 0.5
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)

    def test_zero_deadband_recovers_linear_droop(self):
        """db = 0 ⇒ piecewise function collapses to Q = -R*(V-V_ref-V_cor).
        Verify the no-deadband case matches the original linear law exactly
        at a non-trivial V."""
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.0, q_cor_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.01
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        R = 10.0 / 0.07
        # V_eff = +0.01, db = 0 ⇒ Q = -R*(0.01) ≈ -1.4286 (within cap).
        target = loop._compute_target(net)
        assert target == pytest.approx(-R * 0.01, rel=1e-6)


# ---------------------------------------------------------------------------
#  Q_cor shifts the droop / deadband
# ---------------------------------------------------------------------------

class TestQCorOffset:
    """Q_cor shifts the V_Q characteristic horizontally by V_cor = Q_cor / R
    (Soleimani eq. 1).  At V_meas = V_ref + V_cor, the DER sees V_eff = 0
    and is centred in its (shifted) deadband, so Q_target = 0.  This is
    the operational test that anchors the Q_cor semantics."""

    def test_q_cor_shifts_zero_q_point(self):
        sn = 10.0
        slope = 0.07
        R = sn / slope  # ≈ 142.857
        # Pick q_cor = 1.0 Mvar ⇒ V_cor = 1.0 / 142.857 = 0.007 pu
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.00, deadband_pu=0.0,
            q_cor_mvar=1.0, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        # Set V_meas exactly at V_ref + V_cor.
        v_cor = 1.0 / R
        net.res_bus.at[bus, "vm_pu"] = 1.00 + v_cor
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        assert target == pytest.approx(0.0, abs=1e-6)

    def test_q_cor_with_deadband_reaches_capacitive_segment(self):
        """Plan §6a sub-case (iii): V_meas=1.005 pu, V_ref=1.00 pu,
        deadband=0.03 pu, q_cor = +R·0.04 ⇒ V_cor = +0.04 pu.
        ⇒ V_eff = 1.005 - 1.00 - 0.04 = -0.035 ⇒ V_eff < -db ⇒
        Q_target = -R·(V_eff + db) = -R·(-0.035 + 0.03) = +R·0.005."""
        sn = 10.0
        slope = 0.10
        R = sn / slope  # = 100
        q_cor = R * 0.04  # = 4.0 Mvar
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.00, deadband_pu=0.03,
            q_cor_mvar=q_cor, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.005
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        expected = R * 0.005  # = 0.5 Mvar (capacitive)
        assert target == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
#  Saturation
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_high_overvoltage_clips_to_q_min(self):
        """V_meas pushed far above V_ref ⇒ droop demands Q below
        capability ⇒ output clips at Q_min = -|Q_min(P)| (STATCOM)."""
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.03, q_cor_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.20  # well past saturation
        net.res_sgen.at[sgen, "p_mw"] = 0.0
        target = loop._compute_target(net)
        # STATCOM at P=0, Sn=10: q_min = -10
        assert target == pytest.approx(-10.0, abs=1e-3)


# ---------------------------------------------------------------------------
#  Installer dispatches QVLocalLoop for q_mode == "qv"
# ---------------------------------------------------------------------------

class TestInstallDispatch:
    def test_install_der_q_loops_creates_qv_loop_for_qv_mode(self):
        net, _, sgen = _build_tiny_qv_net()
        idx_list = install_der_q_loops(net, [sgen])
        assert len(idx_list) == 1
        ctrl_obj = net.controller.at[idx_list[0], "object"]
        assert isinstance(ctrl_obj, QVLocalLoop)

    def test_install_der_q_loops_idempotent(self):
        net, _, sgen = _build_tiny_qv_net()
        first = install_der_q_loops(net, [sgen])
        second = install_der_q_loops(net, [sgen])
        assert len(first) == 1
        assert second == []
