"""
Re-solve after the OLTC rate limiter: relays held, DER Q(V) loops active
========================================================================

Fault fixed 2026-11-07.  In local STS mode each plant power flow of a step
runs with the coupler relays (``DiscreteTapControl``) free; the relays bring
the voltage into their band (1.03 +/- 0.01 pu), which is also the droop's dead
band, so the DER Q(V) loops settle near Q = 0.  The rate limiter then moves
the taps back to the step's snapshot +/- 1 and the runner re-solved with
``run_control=False``: Q stayed at its free-relay value while the voltage fell
back below the dead band.  The recorded state was not a fixed point of the
droop law (no-access variants L2, S1, O2; E2 load step, run 0025).

The runner now re-solves with :func:`core.plant.runpp_relays_held`.  The tests
drive one relay and one Q(V) DER through the runner's sequence (snapshot,
disturbance, free solve, clamp, re-solve).

Author: Manuel Schwenke / Claude Code (2026-11-07)
"""

from __future__ import annotations

import pandapower as pp
import pytest
from pandapower.control import DiscreteTapControl

from controller.der_qv_local_loop import QVLocalLoop
from core.plant import runpp_relays_held
from experiments.runners._multi_tso_helpers import _OLTCRateLimiter

V_SET, SLOPE, DB = 1.03, 0.06, 0.01
SN = 30.0
KW = dict(calculate_voltage_angles=True, max_iteration=100, max_iter=300)


def _net():
    net = pp.create_empty_network(sn_mva=100.0)
    hv = pp.create_bus(net, vn_kv=220.0)
    mv = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=hv, vm_pu=1.0)
    t = pp.create_transformer_from_parameters(
        net, hv_bus=hv, lv_bus=mv, sn_mva=300.0, vn_hv_kv=220.0,
        vn_lv_kv=110.0, vk_percent=12.0, vkr_percent=0.3, pfe_kw=0.0,
        i0_percent=0.0, tap_side="hv", tap_neutral=0, tap_min=-10,
        tap_max=10, tap_step_percent=1.5, tap_pos=0,
        tap_changer_type="Ratio")
    pp.create_load(net, bus=mv, p_mw=100.0, q_mvar=30.0)
    s = int(pp.create_sgen(net, bus=mv, p_mw=0.0, q_mvar=0.0, sn_mva=SN))
    for col, val in (("op_diagram", "STATCOM"), ("q_mode", "qv"),
                     ("qv_slope_pu", SLOPE), ("qv_vref_pu", V_SET),
                     ("qv_deadband_pu", DB), ("q_set_mvar", 0.0),
                     ("qv_vref_anchor_pu", float("nan"))):
        net.sgen[col] = val
    loop = QVLocalLoop(net, sgen_idx=s, slope_pu=SLOPE, tol_mvar=0.01)
    relay = DiscreteTapControl(net, element_index=t, vm_lower_pu=V_SET - DB,
                               vm_upper_pu=V_SET + DB, side="lv")
    pp.runpp(net, run_control=True, **KW)
    return net, t, s, mv, loop, relay


def _clamped_after_step(net, t):
    """Snapshot, 120 Mvar load step, free solve, limiter clamp (runner order)."""
    lim = _OLTCRateLimiter(max_step=1, cooldown_s=30.0,
                           cooldown_by_table={"trafo": 120.0})
    lim.snapshot(net)
    tap0 = int(net.trafo.at[t, "tap_pos"])
    pp.create_load(net, bus=1, p_mw=0.0, q_mvar=120.0)
    pp.runpp(net, run_control=True, **KW)          # plant.advance: relays free
    tap_free = int(net.trafo.at[t, "tap_pos"])
    moved = lim.clamp(net, 1000.0)
    return tap0, tap_free, moved


def test_scenario_needs_the_clamp():
    """The free relay moves more than one tap, so the limiter acts."""
    net, t, *_ = _net()
    tap0, tap_free, moved = _clamped_after_step(net, t)
    assert abs(tap_free - tap0) >= 2
    assert moved
    assert abs(int(net.trafo.at[t, "tap_pos"]) - tap0) == 1


def test_run_control_false_leaves_the_droop_off_its_law():
    """The old re-solve: Q stays at its free-relay value below the dead band."""
    net, t, s, mv, loop, _ = _net()
    _clamped_after_step(net, t)
    q_free = float(net.sgen.at[s, "q_mvar"])
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    assert float(net.res_bus.at[mv, "vm_pu"]) < V_SET - DB
    assert float(net.sgen.at[s, "q_mvar"]) == q_free
    assert loop._compute_target(net) - q_free > 1.0       # Mvar


def test_relays_held_resolve_settles_the_droop_at_the_clamped_tap():
    net, t, s, mv, loop, relay = _net()
    _clamped_after_step(net, t)
    tap_clamped = int(net.trafo.at[t, "tap_pos"])
    q_free = float(net.sgen.at[s, "q_mvar"])
    runpp_relays_held(net, **KW)
    q = float(net.sgen.at[s, "q_mvar"])
    # taps stay where the limiter put them; the relay is back in service
    assert int(net.trafo.at[t, "tap_pos"]) == tap_clamped
    assert bool(net.controller.loc[net.controller.object == relay,
                                   "in_service"].iloc[0])
    # the recorded state is a fixed point of the droop law, and it injects
    assert loop._compute_target(net) == pytest.approx(q, abs=0.05)
    assert q - q_free > 1.0
    assert float(net.res_sgen.at[s, "q_mvar"]) == pytest.approx(q)


def test_relay_out_of_service_stays_out():
    net, t, s, mv, loop, relay = _net()
    sel = net.controller.object == relay
    net.controller.loc[sel, "in_service"] = False
    runpp_relays_held(net, **KW)
    assert not bool(net.controller.loc[sel, "in_service"].iloc[0])


def test_without_controllers():
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b)
    pp.create_load(net, bus=b, p_mw=1.0)
    runpp_relays_held(net)
    assert net.converged
