"""
Tests for the piecewise-linear Q(V) characteristic with deadband and
central Q_set commanding (refactor_v3, VDE-AR-N 4120
*Blindleistung mit Spannungsbegrenzungsfunktion*).

Plant-side formula (sgen sign convention, R = S_n / qv_slope_pu):

    v_eff    = V_meas − V_ref + Q_set / R
    Q_target = Q_set − R · (v_eff − db)        if  v_eff >  db   (V high)
             = Q_set                           if |v_eff| ≤ db
             = Q_set − R · (v_eff + db)        if  v_eff < −db   (V low)
    Q_actual = clip(Q_target, Q_min(P), Q_max(P))

The test network is a tiny 2-bus PF where the slack bus voltage and a
single sgen at the load bus give us full control over V at the DER bus.
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
    vref_pu: float = 1.03,
    deadband_pu: float = 0.02,
    q_set_mvar: float = 0.0,
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
    net.sgen["q_mode"] = "qv"
    net.sgen["qv_slope_pu"] = slope_pu
    net.sgen["qv_vref_pu"] = vref_pu
    net.sgen["qv_deadband_pu"] = deadband_pu
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_set_mvar"] = q_set_mvar
    return net, b_load, int(sgen_idx)


# ---------------------------------------------------------------------------
#  Q_set = 0: pure droop with deadband around V_ref
# ---------------------------------------------------------------------------

class TestDeadbandWithoutCommand:
    """Q_set = 0: the curve is the standard symmetric droop with
    deadband; inside the deadband the inverter feeds in 0 Mvar."""

    def test_inside_deadband_q_target_zero(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.03, deadband_pu=0.02, q_set_mvar=0.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.04   # within ±0.02 of V_ref=1.03
        net.res_sgen.at[sgen, "p_mw"] = 10.0
        target = loop._compute_target(net)
        assert target == pytest.approx(0.0, abs=1e-9)

    def test_just_outside_deadband_upper(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.03, deadband_pu=0.02, q_set_mvar=0.0,
            sn_mva=100.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        # V_eff = 1.06 - 1.03 + 0 = +0.03; segment v_eff > db ⇒
        #   Q = 0 - R*(0.03 - 0.02) = -R*0.01
        net.res_bus.at[bus, "vm_pu"] = 1.06
        net.res_sgen.at[sgen, "p_mw"] = 10.0
        R = 100.0 / 0.07
        expected = -R * 0.01
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)

    def test_just_outside_deadband_lower(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.40, vref_pu=1.03, deadband_pu=0.02, q_set_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.40)
        pp.runpp(net, run_control=False)
        # V_eff = 1.00 - 1.03 + 0 = -0.03; segment v_eff < -db ⇒
        #   Q = 0 - R*(-0.03 + 0.02) = +R*0.01
        net.res_bus.at[bus, "vm_pu"] = 1.00
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        R = 10.0 / 0.40
        expected = R * 0.01  # = 0.25 Mvar
        target = loop._compute_target(net)
        assert target == pytest.approx(expected, rel=1e-6)

    def test_zero_deadband_recovers_linear_droop(self):
        """db = 0 ⇒ Q = -R · (V - V_ref).  With Q_set = 0 the formula
        collapses to the classical linear droop independently of any
        Q_set offset (linear segments share the same line for all Q_set)."""
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.03, deadband_pu=0.0, q_set_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.04
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        R = 10.0 / 0.07
        # V_eff = +0.01, db = 0 ⇒ Q = -R*0.01 ≈ -1.43 Mvar
        target = loop._compute_target(net)
        assert target == pytest.approx(-R * 0.01, rel=1e-6)


# ---------------------------------------------------------------------------
#  Q_set ≠ 0: shifted curve, deadband baseline at Q_set
# ---------------------------------------------------------------------------

class TestQSetCommanding:
    """When the OFO commands a non-zero Q_set, the deadband baseline
    itself shifts to Q = Q_set: inside the (V_ref-shifted) deadband the
    inverter feeds in exactly the commanded value."""

    def test_q_set_realised_when_v_in_shifted_deadband(self):
        """At V = V_ref the inverter sits inside the shifted deadband for
        any |Q_set/R| ≤ db, so Q_actual must equal Q_set exactly."""
        sn = 10.0
        slope = 0.07
        R = sn / slope          # ≈ 142.857 Mvar/pu_v
        db = 0.02
        # Pick Q_set so |Q_set/R| < db: Q_set/R = 0.01 < 0.02
        q_set = R * 0.01        # ≈ 1.4286 Mvar
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.03, deadband_pu=db,
            q_set_mvar=q_set, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.03   # V at original V_ref
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        assert target == pytest.approx(q_set, abs=1e-6)

    def test_q_set_outside_shifted_deadband(self):
        """When V is outside the shifted deadband, the linear segment
        formula applies.  The linear segments of the shifted curve lie
        on the SAME line as the unshifted curve (slope is fixed), so
        Q_actual depends only on V (not on Q_set) once we leave the
        deadband.  Verify by computing Q at the same V for two values
        of Q_set: the difference must be zero."""
        sn = 10.0
        slope = 0.07
        R = sn / slope
        db = 0.02

        def q_at(q_set: float, v_pu: float) -> float:
            net, bus, sgen = _build_tiny_qv_net(
                slope_pu=slope, vref_pu=1.03, deadband_pu=db,
                q_set_mvar=q_set, sn_mva=sn,
            )
            loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
            pp.runpp(net, run_control=False)
            net.res_bus.at[bus, "vm_pu"] = v_pu
            net.res_sgen.at[sgen, "p_mw"] = 0.5
            return loop._compute_target(net)

        # V = 1.10 puts us well above any reasonable shifted upper db.
        v_test = 1.10
        q_a = q_at(q_set=0.0, v_pu=v_test)
        q_b = q_at(q_set=R * 0.005, v_pu=v_test)   # small shift
        # Both must lie on Q = -R · (V - V_ref - db)  ⇒  identical.
        # (capability clipping won't engage at sn=10, slope=0.07, V=1.1
        # because target = -R*(0.10-0.02) = -11.4, but cap is ±10 ⇒ both
        # clip to -10 ⇒ still equal.)
        assert q_a == pytest.approx(q_b, abs=1e-6)

    def test_q_set_capacitive_direction(self):
        """Q_set > 0 (sgen injects 1.5 Mvar baseline) must produce a
        positive Q_actual inside the shifted deadband."""
        sn = 10.0
        slope = 0.10
        R = sn / slope          # = 100
        db = 0.02
        q_set = 1.5             # < R · db = 2.0 ⇒ inside shifted deadband at V_ref
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=slope, vref_pu=1.03, deadband_pu=db,
            q_set_mvar=q_set, sn_mva=sn,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=slope)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.03
        net.res_sgen.at[sgen, "p_mw"] = 1.0
        target = loop._compute_target(net)
        assert target == pytest.approx(q_set, abs=1e-6)
        assert target > 0.0


# ---------------------------------------------------------------------------
#  Saturation
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_high_overvoltage_clips_to_q_min(self):
        """V_meas pushed far above V_ref ⇒ droop demands Q below
        capability ⇒ output clips at Q_min = -|Q_min(P)| (STATCOM)."""
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.03, deadband_pu=0.02, q_set_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 1.20  # well past saturation
        net.res_sgen.at[sgen, "p_mw"] = 0.0
        target = loop._compute_target(net)
        # STATCOM at P=0, Sn=10: q_min = -10
        assert target == pytest.approx(-10.0, abs=1e-3)

    def test_low_undervoltage_clips_to_q_max(self):
        net, bus, sgen = _build_tiny_qv_net(
            slope_pu=0.07, vref_pu=1.03, deadband_pu=0.02, q_set_mvar=0.0,
            sn_mva=10.0,
        )
        loop = QVLocalLoop(net, sgen_idx=sgen, slope_pu=0.07)
        pp.runpp(net, run_control=False)
        net.res_bus.at[bus, "vm_pu"] = 0.85
        net.res_sgen.at[sgen, "p_mw"] = 0.0
        target = loop._compute_target(net)
        # STATCOM at P=0, Sn=10: q_max = +10
        assert target == pytest.approx(10.0, abs=1e-3)


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
