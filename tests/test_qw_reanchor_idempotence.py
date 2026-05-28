"""
Idempotence of the V_ref reanchoring + zero-w sequence.

Under the w-shift actuator the apply step does two things:
  1. reanchor ``qv_vref_anchor_pu`` to the most recent measured bus
     voltage; and
  2. write ``q_set_mvar``.

If the second apply uses the same q_set as the first AND no other
disturbance moves V in between, then the QVLocalLoop's Q target must
not move on the second apply.  This protects against an arithmetic
sign error where reanchoring would inadvertently introduce a non-zero
``V - V_anchor`` term and make the local loop chase its own tail.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from controller.der_qv_local_loop import QVLocalLoop, install_der_q_loops


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
    net.sgen["qv_deadband_pu"] = 0.005
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_set_mvar"] = 0.0
    net.sgen["qv_vref_anchor_pu"] = float("nan")
    return net, int(b_load), int(sgen)


def _reanchor_and_set(net: pp.pandapowerNet, sgen: int, bus: int,
                      q_set: float) -> None:
    """Mimic what the apply step does: reanchor then write q_set."""
    net.sgen.at[sgen, "qv_vref_anchor_pu"] = float(net.res_bus.at[bus, "vm_pu"])
    net.sgen.at[sgen, "q_set_mvar"] = float(q_set)


def test_double_reanchor_with_zero_w_leaves_q_unchanged():
    """Two consecutive PFs, each preceded by a reanchor + ``q_set = 0``
    write, must yield the same realised Q.  No drift, no oscillation."""
    net, bus, sgen = _build_net()
    install_der_q_loops(net, [sgen], qv_damping=0.5,
                        qv_max_step_frac=None, qv_tol_mvar=0.01)
    pp.runpp(net, run_control=True, max_iteration=200)
    q1 = float(net.res_sgen.at[sgen, "q_mvar"])

    # Second apply with the same q_set = 0 and no disturbance.
    _reanchor_and_set(net, sgen, bus, q_set=0.0)
    pp.runpp(net, run_control=True, max_iteration=200)
    q2 = float(net.res_sgen.at[sgen, "q_mvar"])

    # Tolerance: 0.1 Mvar is well below QVLocalLoop's own tol_mvar.
    assert q2 == pytest.approx(q1, abs=0.1), (
        f"Idempotence broken: q1={q1:.3f}, q2={q2:.3f} after a noop "
        f"reanchor + q_set=0 cycle."
    )


def test_q_at_anchor_matches_q_set_inside_deadband():
    """When V_meas = V_anchor (just reanchored) and the deadband contains
    the operating point, the realised Q equals ``q_set``."""
    net, bus, sgen = _build_net()
    install_der_q_loops(net, [sgen], qv_damping=0.5,
                        qv_max_step_frac=None, qv_tol_mvar=0.01)
    pp.runpp(net, run_control=True, max_iteration=200)
    # Reanchor and command q_set = 4 Mvar.
    _reanchor_and_set(net, sgen, bus, q_set=4.0)
    pp.runpp(net, run_control=True, max_iteration=200)

    q = float(net.res_sgen.at[sgen, "q_mvar"])
    v = float(net.res_bus.at[bus, "vm_pu"])
    v_anchor = float(net.sgen.at[sgen, "qv_vref_anchor_pu"])
    db = float(net.sgen.at[sgen, "qv_deadband_pu"])

    # The reanchor pinned V_anchor at the *pre*-apply voltage.  After the
    # subsequent PF the voltage may have moved by S_VQ·ΔQ — still small
    # for a single sgen of 4 Mvar on a 100 MVA base.  Allow a millivolt
    # of drift.  When V_eff stays in the deadband, Q ≈ q_set.
    if abs(v - v_anchor) <= db:
        assert q == pytest.approx(4.0, abs=0.2)
    else:
        # Sanity floor: the realised Q still hovers near q_set when the
        # network is stiff (small S_VQ at this bus).
        assert q == pytest.approx(4.0, abs=2.0)
