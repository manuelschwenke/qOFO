"""Q-row sensitivities w.r.t. voltage set points and taps carry Mvar, not pu.

Regression for the sn_mva slip found on 2026-11-08 (daily log
``2026-11-08d_qgen_row_check``): six functions returned ∂Q[pu]/∂V[pu] or
∂Q[pu]/∂step on the system base while their docstrings and the controllers
treat them as Mvar.  On the thesis networks (sn_mva = 100) the entries were
100x too small; on the development net (sn_mva = 1) the slip is invisible.

Two checks per function:

* **base invariance**: the same physical network solved with sn_mva = 100 and
  sn_mva = 1 must give the same value (a physical sensitivity cannot depend on
  the per-unit base);
* **finite differences** against ``pp.runpp`` in Mvar, for the functions that no
  other test compares with a power flow.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from sensitivity.jacobian import JacobianSensitivities


def _machine_trafo_net(sn_mva: float = 100.0) -> pp.pandapowerNet:
    """EHV slack - line - PQ bus - machine trafo (OLTC) - PV gen."""
    net = pp.create_empty_network(sn_mva=sn_mva)
    b_slack = pp.create_bus(net, vn_kv=345.0)
    b_ehv = pp.create_bus(net, vn_kv=345.0)
    b_gen = pp.create_bus(net, vn_kv=10.5)
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.00)
    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_ehv, length_km=30.0,
        r_ohm_per_km=0.05, x_ohm_per_km=0.3, c_nf_per_km=10.0, max_i_ka=1.5)
    pp.create_load(net, bus=b_ehv, p_mw=60.0, q_mvar=20.0)
    pp.create_transformer_from_parameters(
        net, hv_bus=b_ehv, lv_bus=b_gen, sn_mva=200.0, vn_hv_kv=345.0,
        vn_lv_kv=10.5, vk_percent=12.0, vkr_percent=0.3, pfe_kw=0.0,
        i0_percent=0.0, tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
        tap_pos=0, tap_step_percent=1.25, shift_degree=0.0,
        tap_changer_type="Ratio")
    pp.create_gen(net, bus=b_gen, p_mw=100.0, vm_pu=1.00)
    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged
    return net


def _trafo3w_gen_net(sn_mva: float = 100.0) -> pp.pandapowerNet:
    """380 kV slack - 3W coupler (OLTC, HV side) - 110 kV - line - PV gen bus."""
    net = pp.create_empty_network(sn_mva=sn_mva)
    b_ehv = pp.create_bus(net, vn_kv=380.0)
    b_hv = pp.create_bus(net, vn_kv=110.0)
    b_lv = pp.create_bus(net, vn_kv=20.0)
    b_hv2 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b_ehv, vm_pu=1.05)
    pp.create_transformer3w_from_parameters(
        net, hv_bus=b_ehv, mv_bus=b_hv, lv_bus=b_lv,
        sn_hv_mva=300.0, sn_mv_mva=300.0, sn_lv_mva=75.0,
        vn_hv_kv=380.0, vn_mv_kv=110.0, vn_lv_kv=20.0,
        vk_hv_percent=9.0, vk_mv_percent=5.0, vk_lv_percent=7.5,
        vkr_hv_percent=0.26, vkr_mv_percent=0.06, vkr_lv_percent=0.06,
        pfe_kw=90.0, i0_percent=0.04, shift_mv_degree=0.0,
        shift_lv_degree=150.0, tap_side="hv", tap_neutral=0, tap_min=-9,
        tap_max=9, tap_pos=0, tap_step_percent=1.25, tap_changer_type="Ratio")
    pp.create_line_from_parameters(
        net, from_bus=b_hv, to_bus=b_hv2, length_km=20.0, r_ohm_per_km=0.059,
        x_ohm_per_km=0.35, c_nf_per_km=11.0, max_i_ka=0.6)
    pp.create_load(net, bus=b_hv2, p_mw=80.0, q_mvar=20.0)
    pp.create_gen(net, bus=b_hv2, p_mw=30.0, vm_pu=1.02)
    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged
    return net


def _q_gen(net, gen_idx=0):
    return float(net.res_gen.at[gen_idx, "q_mvar"])


def _fd(net, mutate, read, delta):
    """Central difference of ``read(net)`` for ``mutate(net, ±delta)``."""
    out = []
    for sgn in (+1.0, -1.0):
        n = copy.deepcopy(net)
        mutate(n, sgn * delta)
        pp.runpp(n, calculate_voltage_angles=True)
        out.append(read(n))
    return (out[0] - out[1]) / (2.0 * delta)


def _gen_vm(n, d):
    n.gen.at[0, "vm_pu"] += d


def _tap2w(n, d):
    n.trafo.at[0, "tap_pos"] += int(round(d))


def _tap3w(n, d):
    n.trafo3w.at[0, "tap_pos"] += int(round(d))


# Each case: (label, net builder, analytical value from a JacobianSensitivities)
def _qgen_vgen(s, net):
    b = int(net.gen.at[0, "bus"])
    return float(s.compute_dQgen_dVgen_matrix([b], [b])[0][0, 0])


def _qgen_ds2w(s, net):
    return float(s.compute_dQgen_ds_2w_matrix([int(net.gen.at[0, "bus"])], [0])[0][0, 0])


def _qtr2w_ds(s, net):
    return float(s.compute_dQtrafo_2w_ds(0, 0))


def _qgen_ds3w(s, net):
    return float(s.compute_dQgen_ds_3w_matrix([int(net.gen.at[0, "bus"])], [0])[0][0, 0])


def _qhv_ds3w(s, net):
    return float(s.compute_dQtrafo3w_hv_ds(0, 0))


def _qhv_vgen(s, net):
    return float(s.compute_dQtrafo3w_hv_dVgen_matrix([0], [int(net.gen.at[0, "bus"])])[0][0, 0])


CASES = [
    ("Q_gen x V_gen", _machine_trafo_net, _qgen_vgen),
    ("Q_gen x 2W tap", _machine_trafo_net, _qgen_ds2w),
    ("Q_trafo2w x 2W tap", _machine_trafo_net, _qtr2w_ds),
    ("Q_gen x 3W tap", _trafo3w_gen_net, _qgen_ds3w),
    ("Q_HV(3W) x 3W tap", _trafo3w_gen_net, _qhv_ds3w),
    ("Q_HV(3W) x V_gen", _trafo3w_gen_net, _qhv_vgen),
]


@pytest.mark.parametrize("label,build,value", CASES, ids=[c[0] for c in CASES])
def test_independent_of_the_system_base(label, build, value):
    """A physical sensitivity in Mvar must not depend on sn_mva."""
    v100 = value(JacobianSensitivities(build(100.0)), build(100.0))
    v1 = value(JacobianSensitivities(build(1.0)), build(1.0))
    assert abs(v100) > 1e-3, f"{label}: degenerate fixture ({v100})"
    # 1e-4: the two power flows converge to the same point only within the
    # solver tolerance (measured 1.5e-6); the slip under test is a factor 100.
    np.testing.assert_allclose(v100, v1, rtol=1e-4, err_msg=label)


@pytest.mark.parametrize("sn_mva", [100.0, 1.0])
def test_dQtrafo2w_ds_vs_finite_difference(sn_mva):
    net = _machine_trafo_net(sn_mva)
    ana = _qtr2w_ds(JacobianSensitivities(net), net)
    num = _fd(net, _tap2w, lambda n: float(n.res_trafo.at[0, "q_hv_mvar"]), 1.0)
    np.testing.assert_allclose(ana, num, rtol=0.03)


@pytest.mark.parametrize("sn_mva", [100.0, 1.0])
def test_dQgen_ds_3w_vs_finite_difference(sn_mva):
    net = _trafo3w_gen_net(sn_mva)
    ana = _qgen_ds3w(JacobianSensitivities(net), net)
    num = _fd(net, _tap3w, _q_gen, 1.0)
    np.testing.assert_allclose(ana, num, rtol=0.03)


@pytest.mark.parametrize("sn_mva", [100.0, 1.0])
def test_dQtrafo3w_hv_dVgen_vs_finite_difference(sn_mva):
    """Full chain (θ and V terms) since 2026-11-09; the V-only chain had the
    wrong sign on small cross terms."""
    net = _trafo3w_gen_net(sn_mva)
    ana = _qhv_vgen(JacobianSensitivities(net), net)
    num = _fd(net, _gen_vm, lambda n: float(n.res_trafo3w.at[0, "q_hv_mvar"]), 1e-3)
    assert np.sign(ana) == np.sign(num), (ana, num)
    np.testing.assert_allclose(ana, num, rtol=0.03)
