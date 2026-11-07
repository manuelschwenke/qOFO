"""SVR calibration of a DER that has a plant-side Q(V) loop (fix 2026-11-07).

Every TS-DER runs a ``QVLocalLoop`` and is commanded through ``q_set_mvar`` at a
V_ref anchor reset to the measured voltage (``core.plant.write_der_q_set``).
``SVRZoneController.calibrate`` used to perturb ``sgen.q_mvar`` only; the loop
rewrote it inside the run_control solve, so the DER column of the pilot
sensitivity came out near zero and the RVR integral gain far too high (step
campaign 2026-11-06, S1 at OP-R area 2: ~38 instead of 0.6 per update).

The calibrated DER column must match the plant's command path.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import QVLocalLoop
from controller.svr_controller import SVRZoneConfig, SVRZoneController
from core.plant import write_der_q_set

SN_MVA = 100.0
DQ = 5.0


class _Bounds:
    def compute_gen_q_bounds(self, p, v):
        return np.zeros(0), np.zeros(0)

    def compute_der_q_bounds(self, p):
        n = len(np.atleast_1d(p))
        return -SN_MVA * np.ones(n), SN_MVA * np.ones(n)


def _net(*, with_loop: bool):
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1, length_km=10.0,
                                   r_ohm_per_km=0.1, x_ohm_per_km=0.3,
                                   c_nf_per_km=10.0, max_i_ka=1.0)
    pp.create_load(net, bus=b1, p_mw=20.0, q_mvar=10.0)
    s = int(pp.create_sgen(net, bus=b1, p_mw=10.0, q_mvar=0.0, sn_mva=SN_MVA))
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["q_mode"] = "qv" if with_loop else "fixed"
    net.sgen["qv_slope_pu"] = 0.07
    net.sgen["qv_vref_pu"] = 1.0
    net.sgen["qv_deadband_pu"] = 0.005
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_set_mvar"] = 0.0
    net.sgen["qv_vref_anchor_pu"] = 1.0
    if with_loop:
        QVLocalLoop(net, sgen_idx=s, slope_pu=0.07)
    pp.runpp(net, run_control=True)
    return net, b1, s


def _controller(pilot: int, sgen: int) -> SVRZoneController:
    zone = SimpleNamespace(zone_id=1, gen_indices=[], tso_der_indices=[sgen],
                           pcc_trafo_indices=[], oltc_trafo_indices=[],
                           shunt_bus_indices=[])
    return SVRZoneController(zone, _Bounds(), SVRZoneConfig(pilot_bus=pilot, deadband_pu=0.0))


def _plant_path_sensitivity(net, pilot: int, sgen: int) -> float:
    """dV_pilot / dQ through the plant's write (fresh copy of the base per sign)."""
    v = {}
    for sign in (+1.0, -1.0):
        plant = copy.deepcopy(net)
        write_der_q_set(plant, sgen, float(plant.sgen.at[sgen, "q_mvar"]) + sign * DQ)
        pp.runpp(plant, run_control=True)
        v[sign] = float(plant.res_bus.at[pilot, "vm_pu"])
    return (v[+1.0] - v[-1.0]) / (2 * DQ)


def _q_mvar_sensitivity(net, pilot: int, sgen: int) -> float:
    """dV_pilot / dQ for a DER without a loop (direct ``q_mvar`` write)."""
    v = {}
    for sign in (+1.0, -1.0):
        plant = copy.deepcopy(net)
        plant.sgen.at[sgen, "q_mvar"] += sign * DQ
        pp.runpp(plant, run_control=False)
        v[sign] = float(plant.res_bus.at[pilot, "vm_pu"])
    return (v[+1.0] - v[-1.0]) / (2 * DQ)


def test_qv_der_column_matches_plant_command_path():
    net, pilot, s = _net(with_loop=True)
    ctrl = _controller(pilot, s)
    ctrl.calibrate(net, dict(run_control=True))
    ref = _plant_path_sensitivity(net, pilot, s)
    assert ref > 1e-5  # the command moves the pilot (inside the droop deadband)
    # 1 %: the damped loop converges to its own tolerance in both solves.
    assert ctrl.dv_pilot_dq[0] == pytest.approx(ref, rel=1e-2)
    assert ctrl.pilot_sensitivity == pytest.approx(ref * SN_MVA, rel=1e-2)


def test_qv_der_column_is_not_undone_by_the_loop():
    """Before the fix the loop restored Q and the column was ~0."""
    net, pilot, s = _net(with_loop=True)
    ctrl = _controller(pilot, s)
    ctrl.calibrate(net, dict(run_control=True))
    assert ctrl.dv_pilot_dq[0] == pytest.approx(_q_mvar_sensitivity(net, pilot, s), rel=0.05)


def test_der_without_loop_unchanged():
    net, pilot, s = _net(with_loop=False)
    ctrl = _controller(pilot, s)
    ctrl.calibrate(net, dict(run_control=True))
    assert ctrl.dv_pilot_dq[0] == pytest.approx(_q_mvar_sensitivity(net, pilot, s), rel=1e-3)


def test_calibration_leaves_the_live_net_untouched():
    net, pilot, s = _net(with_loop=True)
    before = net.sgen[["q_mvar", "q_set_mvar", "qv_vref_anchor_pu"]].copy()
    v_before = net.res_bus.vm_pu.copy()
    _controller(pilot, s).calibrate(net, dict(run_control=True))
    np.testing.assert_array_equal(net.sgen[["q_mvar", "q_set_mvar", "qv_vref_anchor_pu"]].to_numpy(),
                                  before.to_numpy())
    np.testing.assert_array_equal(net.res_bus.vm_pu.to_numpy(), v_before.to_numpy())
