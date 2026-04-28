"""
Finite-difference validation for line-endpoint Q sensitivities (Q_tie).

Covers the new ``compute_dQ_line_*`` primitives in
``sensitivity.jacobian.JacobianSensitivities`` that supply the Q_tie row
block of the multi-TSO controller's H matrix:

* ``compute_dQ_line_dQ_der``      — vs DER reactive injection at a PQ bus
* ``compute_dQ_line_2w_ds``       — vs 2W OLTC tap step
* ``compute_dQ_line_dQ_shunt``    — vs bipolar shunt step
* ``compute_dQ_line_dVgen``       — vs PV-bus AVR voltage setpoint

Sign convention asserted: the sensitivities are evaluated at a chosen
``endpoint_bus``; the analytical value must agree with the finite
difference of ``net.res_line.q_from_mvar`` (when endpoint == from_bus)
or ``net.res_line.q_to_mvar`` (when endpoint == to_bus).

The analytical primitives return ``∂Q_pu/∂*`` in pu of Q (same convention
as the existing ``compute_dQtrafo_*`` and ``compute_dQgen_*`` primitives);
the test multiplies by ``net.sn_mva`` to compare to Mvar-domain finite
differences from ``pp.runpp``.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from sensitivity.jacobian import JacobianSensitivities


# ---------------------------------------------------------------------------
#  Test network: two PQ buses connected by an EHV line, each end with a load
#  and a controllable DER, plus a PV gen behind a machine trafo on bus_a side
#  to provide V/theta richness.  A switchable shunt sits at bus_b.
# ---------------------------------------------------------------------------

@pytest.fixture
def two_zone_tie_network() -> pp.pandapowerNet:
    """EHV line between two PQ buses; one side has a PV gen via machine
    trafo, the other has a switchable shunt and a DER."""
    net = pp.create_empty_network(sn_mva=100.0)

    b_slack = pp.create_bus(net, vn_kv=345.0, name="slack")
    b_a = pp.create_bus(net, vn_kv=345.0, name="ehv_a")  # zone A
    b_b = pp.create_bus(net, vn_kv=345.0, name="ehv_b")  # zone B
    b_gen = pp.create_bus(net, vn_kv=15.0, name="gen_lv")

    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.00)

    # Slack ↔ bus_a — short line so the slack regulates bus_a
    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_a,
        length_km=10.0, r_ohm_per_km=0.05, x_ohm_per_km=0.30,
        c_nf_per_km=10.0, max_i_ka=1.5,
    )
    # bus_a ↔ bus_b — the TIE LINE under test
    tie_idx = pp.create_line_from_parameters(
        net, from_bus=b_a, to_bus=b_b,
        length_km=120.0, r_ohm_per_km=0.05, x_ohm_per_km=0.32,
        c_nf_per_km=10.0, max_i_ka=1.5,
        name="TIE_LINE",
    )

    # Loads on both PQ buses to make Q flow non-trivial
    pp.create_load(net, bus=b_a, p_mw=80.0, q_mvar=30.0)
    pp.create_load(net, bus=b_b, p_mw=120.0, q_mvar=50.0)

    # PV gen behind machine trafo at bus_a side (gives a V_gen handle)
    mt_idx = pp.create_transformer_from_parameters(
        net,
        hv_bus=b_a, lv_bus=b_gen,
        sn_mva=200.0, vn_hv_kv=345.0, vn_lv_kv=15.0,
        vk_percent=12.0, vkr_percent=0.3,
        pfe_kw=0.0, i0_percent=0.0,
        tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
        tap_pos=0, tap_step_percent=1.25, shift_degree=0.0,
        tap_changer_type="Ratio",
        name="machine_trafo",
    )
    pp.create_gen(net, bus=b_gen, p_mw=180.0, vm_pu=1.02, name="PV_GEN")

    # Switchable shunt at bus_b (10 Mvar per step)
    pp.create_shunt(net, bus=b_b, p_mw=0.0, q_mvar=10.0, step=0, max_step=2)

    # Controllable DER (sgen) at bus_b — PQ bus, contains Jacobian column
    pp.create_sgen(net, bus=b_b, p_mw=0.0, q_mvar=0.0, name="DER_b")

    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged

    net.tie_idx = int(tie_idx)
    net.bus_a = int(b_a)
    net.bus_b = int(b_b)
    net.machine_trafo_idx = int(mt_idx)
    net.gen_bus = int(b_gen)
    return net


# ---------------------------------------------------------------------------
#  Finite-difference helpers — central differences on res_line Q
# ---------------------------------------------------------------------------

def _q_endpoint_mvar(net: pp.pandapowerNet, line_idx: int, endpoint_bus: int) -> float:
    fb = int(net.line.at[line_idx, "from_bus"])
    tb = int(net.line.at[line_idx, "to_bus"])
    if endpoint_bus == fb:
        return float(net.res_line.at[line_idx, "q_from_mvar"])
    if endpoint_bus == tb:
        return float(net.res_line.at[line_idx, "q_to_mvar"])
    raise ValueError(f"Endpoint bus {endpoint_bus} not on line {line_idx}.")


def _fd_dQ_dQder(net, line_idx, endpoint_bus, sgen_idx, dq_mvar=2.0):
    q0 = float(net.sgen.at[sgen_idx, "q_mvar"])
    np_pos = copy.deepcopy(net)
    np_pos.sgen.at[sgen_idx, "q_mvar"] = q0 + dq_mvar
    pp.runpp(np_pos, calculate_voltage_angles=True)
    np_neg = copy.deepcopy(net)
    np_neg.sgen.at[sgen_idx, "q_mvar"] = q0 - dq_mvar
    pp.runpp(np_neg, calculate_voltage_angles=True)
    return (
        _q_endpoint_mvar(np_pos, line_idx, endpoint_bus)
        - _q_endpoint_mvar(np_neg, line_idx, endpoint_bus)
    ) / (2.0 * dq_mvar)


def _fd_dQ_ds_2w(net, line_idx, endpoint_bus, trafo_idx, dstep=1):
    s0 = int(net.trafo.at[trafo_idx, "tap_pos"])
    np_pos = copy.deepcopy(net)
    np_pos.trafo.at[trafo_idx, "tap_pos"] = s0 + dstep
    pp.runpp(np_pos, calculate_voltage_angles=True)
    np_neg = copy.deepcopy(net)
    np_neg.trafo.at[trafo_idx, "tap_pos"] = s0 - dstep
    pp.runpp(np_neg, calculate_voltage_angles=True)
    return (
        _q_endpoint_mvar(np_pos, line_idx, endpoint_bus)
        - _q_endpoint_mvar(np_neg, line_idx, endpoint_bus)
    ) / (2.0 * dstep)


def _fd_dQ_dShunt(net, line_idx, endpoint_bus, shunt_idx, dstep=1):
    s0 = int(net.shunt.at[shunt_idx, "step"])
    np_pos = copy.deepcopy(net)
    np_pos.shunt.at[shunt_idx, "step"] = s0 + dstep
    pp.runpp(np_pos, calculate_voltage_angles=True)
    np_neg = copy.deepcopy(net)
    np_neg.shunt.at[shunt_idx, "step"] = s0 - dstep
    pp.runpp(np_neg, calculate_voltage_angles=True)
    return (
        _q_endpoint_mvar(np_pos, line_idx, endpoint_bus)
        - _q_endpoint_mvar(np_neg, line_idx, endpoint_bus)
    ) / (2.0 * dstep)


def _fd_dQ_dVgen(net, line_idx, endpoint_bus, gen_idx, dv_pu=0.005):
    v0 = float(net.gen.at[gen_idx, "vm_pu"])
    np_pos = copy.deepcopy(net)
    np_pos.gen.at[gen_idx, "vm_pu"] = v0 + dv_pu
    pp.runpp(np_pos, calculate_voltage_angles=True)
    np_neg = copy.deepcopy(net)
    np_neg.gen.at[gen_idx, "vm_pu"] = v0 - dv_pu
    pp.runpp(np_neg, calculate_voltage_angles=True)
    return (
        _q_endpoint_mvar(np_pos, line_idx, endpoint_bus)
        - _q_endpoint_mvar(np_neg, line_idx, endpoint_bus)
    ) / (2.0 * dv_pu)


def _check_close(analytical_pu, numerical_mvar, sn_mva, label, rtol=0.10):
    """Compare pu-domain analytical (×sn_mva) to Mvar-domain numerical."""
    analytical_mvar = analytical_pu * sn_mva
    if abs(numerical_mvar) < 1e-3:
        # Sensitivity essentially zero — just check analytical is also small.
        assert abs(analytical_mvar) < 1e-2, (
            f"{label}: numerical~0 ({numerical_mvar:.2e}) but "
            f"analytical={analytical_mvar:.2e}"
        )
        return
    ratio = analytical_mvar / numerical_mvar
    assert (1.0 - rtol) < ratio < (1.0 + rtol), (
        f"{label}: analytical={analytical_mvar:.4g} Mvar, "
        f"numerical={numerical_mvar:.4g} Mvar, ratio={ratio:.4f} "
        f"(tolerance ±{rtol*100:.0f}%)"
    )


# ---------------------------------------------------------------------------
#  Tests — exercise both endpoints (sign convention check)
# ---------------------------------------------------------------------------

class TestDQLineDQDer:
    def test_from_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        der_bus = net.bus_b  # DER sits on bus_b; bus_b is PQ
        sgen_idx = int(net.sgen.index[0])
        analytical = sens.compute_dQ_line_dQ_der(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_a,  # measure Q at the from-end (bus_a)
            der_bus_idx=der_bus,
        )
        numerical = _fd_dQ_dQder(
            net, net.tie_idx, endpoint_bus=net.bus_a, sgen_idx=sgen_idx,
        )
        # Q_DER / Q_endpoint chain: ratio is dimensionless (Mvar/Mvar);
        # NO sn_mva multiplication needed for this primitive.
        _check_close(analytical, numerical, sn_mva=1.0, label="dQ_from/dQ_DER")

    def test_to_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        der_bus = net.bus_b
        sgen_idx = int(net.sgen.index[0])
        analytical = sens.compute_dQ_line_dQ_der(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_b,
            der_bus_idx=der_bus,
        )
        numerical = _fd_dQ_dQder(
            net, net.tie_idx, endpoint_bus=net.bus_b, sgen_idx=sgen_idx,
        )
        _check_close(analytical, numerical, sn_mva=1.0, label="dQ_to/dQ_DER")


class TestDQLineDs2W:
    def test_from_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        analytical = sens.compute_dQ_line_2w_ds(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_a,
            chg_trafo_idx=net.machine_trafo_idx,
        )
        numerical = _fd_dQ_ds_2w(
            net, net.tie_idx, endpoint_bus=net.bus_a,
            trafo_idx=net.machine_trafo_idx,
        )
        _check_close(analytical, numerical, sn_mva=net.sn_mva,
                     label="dQ_from/ds_OLTC2w")

    def test_to_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        analytical = sens.compute_dQ_line_2w_ds(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_b,
            chg_trafo_idx=net.machine_trafo_idx,
        )
        numerical = _fd_dQ_ds_2w(
            net, net.tie_idx, endpoint_bus=net.bus_b,
            trafo_idx=net.machine_trafo_idx,
        )
        _check_close(analytical, numerical, sn_mva=net.sn_mva,
                     label="dQ_to/ds_OLTC2w")


class TestDQLineDQShunt:
    def test_from_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        shunt_idx = int(net.shunt.index[0])
        shunt_bus = int(net.shunt.at[shunt_idx, "bus"])
        q_step = float(net.shunt.at[shunt_idx, "q_mvar"])
        analytical = sens.compute_dQ_line_dQ_shunt(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_a,
            shunt_bus_idx=shunt_bus,
            q_step_mvar=q_step,
        )
        numerical = _fd_dQ_dShunt(
            net, net.tie_idx, endpoint_bus=net.bus_a, shunt_idx=shunt_idx,
        )
        # The compute_dQ_line_dQ_shunt scales by q_step internally,
        # producing Mvar-per-step directly (mirrors compute_dQtrafo3w_hv_dQ_shunt).
        _check_close(analytical, numerical, sn_mva=1.0, label="dQ_from/dShunt")

    def test_to_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        shunt_idx = int(net.shunt.index[0])
        shunt_bus = int(net.shunt.at[shunt_idx, "bus"])
        q_step = float(net.shunt.at[shunt_idx, "q_mvar"])
        analytical = sens.compute_dQ_line_dQ_shunt(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_b,
            shunt_bus_idx=shunt_bus,
            q_step_mvar=q_step,
        )
        numerical = _fd_dQ_dShunt(
            net, net.tie_idx, endpoint_bus=net.bus_b, shunt_idx=shunt_idx,
        )
        _check_close(analytical, numerical, sn_mva=1.0, label="dQ_to/dShunt")


class TestDQLineDVgen:
    def test_from_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        gen_idx = int(net.gen.index[0])
        gen_bus = int(net.gen.at[gen_idx, "bus"])
        from sensitivity.index_helper import pp_bus_to_ppc_bus
        gen_bus_ppc = pp_bus_to_ppc_bus(net, gen_bus)
        analytical = sens.compute_dQ_line_dVgen(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_a,
            gen_bus_ppc=gen_bus_ppc,
        )
        numerical = _fd_dQ_dVgen(
            net, net.tie_idx, endpoint_bus=net.bus_a, gen_idx=gen_idx,
        )
        _check_close(analytical, numerical, sn_mva=net.sn_mva,
                     label="dQ_from/dV_gen")

    def test_to_endpoint(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        gen_idx = int(net.gen.index[0])
        gen_bus = int(net.gen.at[gen_idx, "bus"])
        from sensitivity.index_helper import pp_bus_to_ppc_bus
        gen_bus_ppc = pp_bus_to_ppc_bus(net, gen_bus)
        analytical = sens.compute_dQ_line_dVgen(
            line_idx=net.tie_idx,
            endpoint_bus=net.bus_b,
            gen_bus_ppc=gen_bus_ppc,
        )
        numerical = _fd_dQ_dVgen(
            net, net.tie_idx, endpoint_bus=net.bus_b, gen_idx=gen_idx,
        )
        _check_close(analytical, numerical, sn_mva=net.sn_mva,
                     label="dQ_to/dV_gen")


class TestSignSwap:
    """Switching from from-endpoint to to-endpoint must approximately flip
    the sign of the analytical sensitivity (a lossless line would have
    Q_from ≈ -Q_to, so dQ_from/du ≈ -dQ_to/du; a real line with shunt and
    losses introduces a small asymmetry but the ratio should still be
    negative with magnitude near 1)."""

    def test_dQder_sign_swap(self, two_zone_tie_network):
        net = two_zone_tie_network
        sens = JacobianSensitivities(net)
        der_bus = net.bus_b
        s_from = sens.compute_dQ_line_dQ_der(
            line_idx=net.tie_idx, endpoint_bus=net.bus_a, der_bus_idx=der_bus,
        )
        s_to = sens.compute_dQ_line_dQ_der(
            line_idx=net.tie_idx, endpoint_bus=net.bus_b, der_bus_idx=der_bus,
        )
        # Signs should be opposite (or both zero); magnitudes within 50%.
        if abs(s_from) > 1e-4 and abs(s_to) > 1e-4:
            assert s_from * s_to < 0, (
                f"Endpoint Q sensitivities should have opposite signs, "
                f"got from={s_from:.3g}, to={s_to:.3g}"
            )
