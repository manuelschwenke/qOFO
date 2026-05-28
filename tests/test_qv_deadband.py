"""
Tests for the piecewise-linear V_Q characteristic with deadband under the
w-shift / V_ref-reanchored DER actuator.

The plant-side ``QVLocalLoop._compute_target`` implements

    V_eff = V - V_anchor
    Q     = q_set - R · piecewise_deadband(V_eff, db)
          (clipped to [Q_min(P), Q_max(P)])

where ``V_anchor = qv_vref_anchor_pu`` (set by the apply step from the
most recent measured bus voltage), ``q_set = q_set_mvar`` (the OFO
command at the reanchored V_ref), and the piecewise function returns 0
inside the deadband so the law collapses to ``Q = q_set`` there.

The tests below pin V_meas and V_anchor via the post-PF state and let
``_compute_target`` run on a controlled state vector, isolating the
math from the PF solver.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import (
    QVLocalLoop,
    install_der_q_loops,
)


def _build_tiny_qv_net(
    *,
    slope_pu: float = 0.07,
    vref_pu: float = 1.00,
    vref_anchor_pu: float | None = None,
    deadband_pu: float = 0.005,
    q_set_mvar: float = 0.0,
    sn_mva: float = 100.0,
    p_mw: float = 10.0,
) -> tuple[pp.pandapowerNet, int, int]:
    """Two-bus network with one slack and one sgen.  q_mode columns are
    populated so :func:`install_der_q_loops` dispatches ``QVLocalLoop``.

    When ``vref_anchor_pu`` is ``None`` the anchor column is set to the
    same value as ``vref_pu`` (mirrors the post-first-apply state).
    """
    if vref_anchor_pu is None:
        vref_anchor_pu = vref_pu

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
    # q_mode columns
    net.sgen["q_mode"] = "qv"
    net.sgen["qv_slope_pu"] = slope_pu
    net.sgen["qv_vref_pu"] = vref_pu
    net.sgen["qv_deadband_pu"] = deadband_pu
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    # w-shift actuator state
    net.sgen["q_set_mvar"] = q_set_mvar
    net.sgen["qv_vref_anchor_pu"] = vref_anchor_pu
    return net, b_load, int(sgen_idx)


# ---------------------------------------------------------------------------
#  Symmetric deadband around V_anchor (q_set = 0)
# ---------------------------------------------------------------------------

class TestDeadbandStaticTarget:
    """Test ``_compute_target`` directly without running PF.  We pin
    V_anchor / V_meas to specific values via the post-PF state and let
    the QVLocalLoop compute the target.  This isolates the math from
    the PF solver."""

    def test_inside_deadband_q_target_equals_q_set(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.03, q_set_mvar=0.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        # Stage a PF result manually so _compute_target sees what we want.
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.005   # within ±0.03 of V_anchor=1.00
        net.res_sgen.at[sgen, "p_mw"] = 10.0
        target = loop._compute_target(net)
        assert target == pytest.approx(0.0, abs=1e-9)

    def test_just_outside_deadband_upper(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.03, q_set_mvar=0.0,
            sn_mva=100.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        # V_eff = 1.05 - 1.00 = +0.05; segment: V_eff > db ⇒ Q = q_set - R·(V_eff-db)
        net.res_bus.at[bus, "vm_pu"] = 1.05
        net.res_sgen.at[sgen, "p_mw"] = 10.0
        R = 100.0 / 0.07  # ≈ 1428.57
        expected = 0.0 - R * 0.02
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)

    def test_just_outside_deadband_lower(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.40, vref_pu=1.00, deadband_pu=0.03, q_set_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.40)
        pp.runpp(net, run_control=False)
        # V_eff = 0.95 - 1.00 = -0.05; segment: V_eff < -db ⇒ Q = q_set - R·(V_eff+db) = +R·0.02
        net.res_bus.at[bus, "vm_pu"] = 0.95
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        R = 10.0 / 0.40
        expected = R * 0.02  # = 0.5
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)

    def test_zero_deadband_recovers_linear_droop(self):
        """db = 0 ⇒ piecewise function collapses to Q = q_set - R·(V-V_anchor)."""
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, deadband_pu=0.0, q_set_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.01
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        R = 10.0 / 0.07
        # V_eff = +0.01, db = 0 ⇒ Q = 0 - R·0.01 ≈ -1.4286 (within cap).
        target = loop._compute_target(net)
        assert target == pytest.approx(-R * 0.01, rel=1e-6)


# ---------------------------------------------------------------------------
#  Vertical shift via q_set
# ---------------------------------------------------------------------------

class TestVerticalShift:
    """The OFO-commanded ``q_set`` is the Q output at the reanchored
    V_ref.  When V_meas = V_anchor the local loop is centred in the
    deadband (V_eff = 0) and the realised Q equals ``q_set``."""

    def test_q_set_realized_at_anchor(self):
        sn = 10.0
        slope = 0.07
        q_set = 1.0  # Mvar
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.00, vref_anchor_pu=1.00,
            deadband_pu=0.005, q_set_mvar=q_set, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        # V_meas exactly at V_anchor — inside the deadband.
        net.res_bus.at[bus, "vm_pu"] = 1.00
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        assert target == pytest.approx(q_set, abs=1e-9)

    def test_q_set_with_deadband_outside_band(self):
        """V_meas=1.005 pu, V_anchor=1.00 pu, db=0.03 pu, q_set=+2 Mvar.
        V_eff = 0.005 ⇒ inside deadband ⇒ Q = q_set = 2 Mvar."""
        sn = 10.0
        slope = 0.10
        R = sn / slope  # = 100
        q_set = 2.0
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.00, vref_anchor_pu=1.00,
            deadband_pu=0.03, q_set_mvar=q_set, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.005
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        assert target == pytest.approx(q_set, abs=1e-9)

    def test_q_set_outside_deadband_drops_by_R_times_excess(self):
        """V_meas=1.06 pu, V_anchor=1.00 pu, db=0.03, q_set=+1 Mvar.
        V_eff = +0.06 > db ⇒ Q = q_set - R·(V_eff-db) = q_set - R·0.03."""
        sn = 10.0
        slope = 0.10
        R = sn / slope  # = 100
        q_set = 1.0
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.00, vref_anchor_pu=1.00,
            deadband_pu=0.03, q_set_mvar=q_set, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.06
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        expected = q_set - R * 0.03
        assert target == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
#  V_ref reanchoring fallback (cold start) — anchor column is NaN
# ---------------------------------------------------------------------------

class TestAnchorFallback:
    """Until the first apply step writes ``qv_vref_anchor_pu``, the
    local loop falls back to the nominal ``qv_vref_pu`` so the cold-start
    PF still produces a sensible droop."""

    def test_nan_anchor_falls_back_to_nominal_vref(self):
        sn = 10.0
        slope = 0.07
        # vref_anchor_pu NaN ⇒ fallback to qv_vref_pu = 1.00.
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.00, deadband_pu=0.0,
            q_set_mvar=0.0, sn_mva=sn,
        )
        net.sgen.at[sgen, "qv_vref_anchor_pu"] = float("nan")
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.01
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        # With anchor falling back to vref=1.00 the math reduces to
        # Q = 0 - R · 0.01.
        R = sn / slope
        expected = -R * 0.01
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
#  Saturation
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_high_overvoltage_clips_to_q_min(self):
        """V_meas pushed far above V_anchor ⇒ droop demands Q below
        capability ⇒ output clips at Q_min = -|Q_min(P)| (STATCOM)."""
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.00, vref_anchor_pu=1.00,
            deadband_pu=0.03, q_set_mvar=0.0, sn_mva=10.0,
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
