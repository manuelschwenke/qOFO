"""
Tests for the DSO Q(V) local loop controller (Stage 2 plant model).
"""

from __future__ import annotations

import math

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import (
    QVLocalLoop,
    _qv_capability,
    install_qv_local_loops,
    remove_qv_local_loops,
)


# ---------------------------------------------------------------------------
#  Capability helper
# ---------------------------------------------------------------------------


class TestCapability:
    def test_statcom_full_circle_at_zero_p(self):
        q_min, q_max = _qv_capability(sn=100.0, op_diagram="STATCOM", p_mw=0.0)
        assert q_max == pytest.approx(100.0)
        assert q_min == pytest.approx(-100.0)

    def test_statcom_at_full_p_zero_q(self):
        q_min, q_max = _qv_capability(sn=100.0, op_diagram="STATCOM", p_mw=100.0)
        assert q_max == pytest.approx(0.0, abs=1e-6)
        assert q_min == pytest.approx(0.0, abs=1e-6)

    def test_statcom_half_p(self):
        q_min, q_max = _qv_capability(sn=100.0, op_diagram="STATCOM", p_mw=50.0)
        # sqrt(1 - 0.25) ~ 0.866
        assert q_max == pytest.approx(86.6025, rel=1e-4)
        assert q_min == pytest.approx(-86.6025, rel=1e-4)

    def test_vde_below_deadband(self):
        # P/S_n = 0.05 < 0.1 ⇒ Q = 0
        q_min, q_max = _qv_capability(sn=100.0, op_diagram="VDE-AR-N-4120-v2", p_mw=5.0)
        assert q_min == pytest.approx(0.0)
        assert q_max == pytest.approx(0.0)

    def test_vde_above_full_capability(self):
        # P/S_n = 0.5 ⇒ full capability [-0.33, +0.41] * Sn
        q_min, q_max = _qv_capability(sn=100.0, op_diagram="VDE-AR-N-4120-v2", p_mw=50.0)
        assert q_min == pytest.approx(-33.0)
        assert q_max == pytest.approx(41.0)

    def test_zero_sn(self):
        q_min, q_max = _qv_capability(sn=0.0, op_diagram="STATCOM", p_mw=0.0)
        assert q_min == pytest.approx(0.0)
        assert q_max == pytest.approx(0.0)


# ---------------------------------------------------------------------------
#  Single-sgen Q(V) loop on a tiny test network
# ---------------------------------------------------------------------------


def _build_tiny_network() -> tuple[pp.pandapowerNet, int, int]:
    """Two-bus network with one slack and one sgen.  Returns (net, bus, sgen)."""
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
        net, bus=b_load, p_mw=10.0, q_mvar=0.0, sn_mva=100.0,
        type="WP", name="QV_TEST",
    )
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["vm_pu_ref"] = 1.03
    return net, b_load, int(sgen_idx)


class TestQVLocalLoopConvergence:
    def test_converges_to_droop_law_unsaturated(self):
        net, bus, sgen = _build_tiny_network()
        install_qv_local_loops(net, [sgen], slope_pu=0.07,
                               damping=0.5, max_step_frac=None, tol_mvar=0.5)
        # Drive the load high enough that V drops below V_ref but not so far
        # that the converter saturates.
        net.load.at[0, "p_mw"] = 50.0
        net.load.at[0, "q_mvar"] = 30.0
        pp.runpp(net, run_control=True, max_iteration=200)
        v = float(net.res_bus.at[bus, "vm_pu"])
        v_ref = float(net.sgen.at[sgen, "vm_pu_ref"])
        q = float(net.res_sgen.at[sgen, "q_mvar"])
        sn = float(net.sgen.at[sgen, "sn_mva"])
        slope = 0.07
        k = sn / slope
        q_target = -k * (v - v_ref)
        # Should match the droop within tolerance.
        assert abs(q - q_target) < 1.0, (
            f"Q(V) loop not at droop equilibrium: q={q:.3f}, "
            f"q_target={q_target:.3f}, V={v:.4f}, V_ref={v_ref:.4f}"
        )

    def test_q_clipped_when_droop_demands_more_than_capability(self):
        net, bus, sgen = _build_tiny_network()
        install_qv_local_loops(net, [sgen], slope_pu=0.07,
                               damping=0.5, max_step_frac=None, tol_mvar=0.5)
        # Push V_ref far above achievable to drive the demanded Q past Q_max.
        net.sgen.at[sgen, "vm_pu_ref"] = 1.20
        pp.runpp(net, run_control=True, max_iteration=200)
        q = float(net.res_sgen.at[sgen, "q_mvar"])
        # Capability from STATCOM at P=10, Sn=100: q_max = sqrt(100^2 - 10^2) ~ 99.5
        assert q == pytest.approx(99.499, abs=0.5), (
            f"Q should clip to q_max ~99.5; got {q:.3f}"
        )

    def test_zero_q_when_v_equals_vref(self):
        net, bus, sgen = _build_tiny_network()
        install_qv_local_loops(net, [sgen], slope_pu=0.07,
                               damping=0.5, max_step_frac=None, tol_mvar=0.5)
        # Tune V_ref to the actual operating-point voltage so droop produces 0.
        pp.runpp(net, run_control=False)
        net.sgen.at[sgen, "vm_pu_ref"] = float(net.res_bus.at[bus, "vm_pu"])
        pp.runpp(net, run_control=True, max_iteration=200)
        q = float(net.res_sgen.at[sgen, "q_mvar"])
        assert abs(q) < 1.0, f"Expected Q ~ 0 at V == V_ref, got {q:.3f}"


# ---------------------------------------------------------------------------
#  Install / remove plumbing
# ---------------------------------------------------------------------------


class TestInstallRemove:
    def test_install_creates_one_controller_per_sgen(self):
        net, _, sgen = _build_tiny_network()
        idx_list = install_qv_local_loops(net, [sgen])
        assert len(idx_list) == 1
        assert any(
            isinstance(row["object"], QVLocalLoop)
            for _, row in net.controller.iterrows()
        )
        # qv_local_loop column is set
        assert bool(net.sgen.at[sgen, "qv_local_loop"]) is True

    def test_install_idempotent(self):
        net, _, sgen = _build_tiny_network()
        install_qv_local_loops(net, [sgen])
        # Second call should be a no-op
        idx_list2 = install_qv_local_loops(net, [sgen])
        assert idx_list2 == []
        n_qv_ctrls = sum(
            1 for _, row in net.controller.iterrows()
            if isinstance(row["object"], QVLocalLoop)
        )
        assert n_qv_ctrls == 1

    def test_remove_clears_all(self):
        net, _, sgen = _build_tiny_network()
        install_qv_local_loops(net, [sgen])
        n_removed = remove_qv_local_loops(net)
        assert n_removed == 1
        n_qv_ctrls = sum(
            1 for _, row in net.controller.iterrows()
            if isinstance(row["object"], QVLocalLoop)
        )
        assert n_qv_ctrls == 0
