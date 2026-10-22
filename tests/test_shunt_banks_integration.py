"""
Integration test: the switched-shunt bank model and its sensitivity columns
against the REAL pandapower sensitivities on the IEEE 39 + HV benchmark.

This bridges the pure bank model (``test_shunt_banks.py``) and the plant. It
checks that the bank columns the MIQP is given carry the device direction (an
MSC step must raise voltage, an MSR step must lower it) without the caller
having to sign them, that the interface feedforward is finite and
device-signed, that the predicted-voltage guard blocks an overshoot computed
from a real column, and that a bus hosting two banks is refused rather than
silently resolved to one device.

Author: Manuel Schwenke
Date: 2026-10-22
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandapower as pp
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controller.shunt_banks import MSC, ShuntBankConfig  # noqa: E402


def _build(**bank_kwargs):
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="rural_700", verbose=False)
    kw = dict(
        install_tso_tertiary_shunts=True,
        tso_shunt_kind="msc_msr",
        msc_n_levels=4, msr_n_levels=4,
        msc_q_step_mvar=50.0, msr_q_step_mvar=50.0,
        verbose=False,
    )
    kw.update(bank_kwargs)
    meta = add_hv_networks(net, meta, **kw)
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             distributed_slack=True)
    return net, meta


def _first_bank(meta, kind):
    for sh_idx, bus, t_zone, k in zip(
        meta.tso_tertiary_shunt_indices,
        meta.tso_tertiary_shunt_buses,
        meta.tso_tertiary_shunt_zones,
        meta.tso_tertiary_shunt_kinds,
    ):
        if k == kind:
            return int(sh_idx), int(bus)
    raise AssertionError(f"no {kind} bank found")


def _host_trafo(hv, bus):
    """The coupling 3W transformer whose tertiary hosts ``bus`` (the bank's
    interface).  Since 2026-10-15 the MSC sits on the second coupler, so the
    first coupler is no longer the right interface for it."""
    lv = [int(b) for b in hv.coupling_lv_bus_indices]
    return int(hv.coupling_trafo_indices[lv.index(int(bus))])


def _voltage_column(sens, bus, obs):
    """``∂V/∂(bank level)`` over ``obs``, as the MIQP's column builder gets it:
    the nameplate magnitude goes in, the device supplies the direction."""
    values, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=bus, observation_bus_indices=obs, q_step_mvar=1.0,
    )
    return np.asarray(values, dtype=np.float64), obs_map


def test_bank_columns_carry_the_device_direction():
    """The caller passes a magnitude; the column's sign is the device's.

    ``net.shunt.q_mvar`` is in load convention, so its sign is the device
    class.  Until 2026-10-22 the config-driven columns ignored it and predicted
    absorption for every bank, which inverted every MSC column.
    """
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    obs = list(hv0.bus_indices)
    _, msc_bus = _first_bank(meta, "MSC")
    _, msr_bus = _first_bank(meta, "MSR")

    h_msc, obs_map = _voltage_column(sens, msc_bus, obs)
    assert len(obs_map) > 0
    assert np.all(h_msc > 0.0), f"MSC ∂V/∂level must be > 0, got {h_msc}"

    h_msr, _ = _voltage_column(sens, msr_bus, obs)
    assert np.all(h_msr < 0.0), f"MSR ∂V/∂level must be < 0, got {h_msr}"


def test_interface_feedforward_finite_and_signed():
    """dQ_itf per engaging step is finite and opposite-signed for MSC / MSR."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    _, msc_bus = _first_bank(meta, "MSC")
    _, msr_bus = _first_bank(meta, "MSR")

    dq_msc = sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=_host_trafo(hv0, msc_bus), shunt_bus_idx=msc_bus,
        q_step_mvar=50.0,
    )
    dq_msr = sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=_host_trafo(hv0, msr_bus), shunt_bus_idx=msr_bus,
        q_step_mvar=50.0,
    )
    assert np.isfinite(dq_msc) and np.isfinite(dq_msr)
    assert abs(dq_msc) > 0.0
    # Opposite susceptance signs ⇒ opposite interface-Q contributions.
    assert np.sign(dq_msc) == -np.sign(dq_msr)


def test_guard_blocks_overshoot_computed_from_a_real_column():
    """With the band tight around the present voltage, the guard refuses the
    engaging direction while the releasing one stays admissible."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    msc_idx, msc_bus = _first_bank(meta, "MSC")
    obs = list(hv0.bus_indices)
    h_col, obs_map = _voltage_column(sens, msc_bus, obs)
    v_meas = np.array([float(net.res_bus.at[b, "vm_pu"]) for b in obs_map],
                      dtype=np.float64)

    # Upper rail just above the present maximum, so a capacitive step would
    # overshoot it.  ``h_col`` is per Mvar, which is what ``_feasible`` expects.
    cfg = ShuntBankConfig(
        shunt_idx=msc_idx, bus_idx=msc_bus,
        interface_trafo3w_idx=_host_trafo(hv0, msc_bus),
        dso_id=hv0.net_id, kind="MSC", q_step_mvar=50.0, n_levels=4,
        t_dwell_s=0.0, y_h_min=0.80, y_h_max=float(v_meas.max()) + 1e-4,
    )
    bank = MSC(cfg)
    bank.level = 1
    assert not bank._feasible(+1, v_meas, h_col)
    assert bank._feasible(-1, v_meas, h_col)
    # A band wide enough for the linear prediction admits the step again.
    wide = MSC(ShuntBankConfig(**{**cfg.__dict__, "y_h_max": 10.0,
                                  "y_h_min": 0.0}))
    assert wide._feasible(+1, v_meas, h_col)


def test_two_banks_on_one_tertiary_are_refused():
    """The MIQP and the sensitivity queries address a bank by its bus, so a
    shared tertiary must fail fast rather than resolve to one of the two."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build(msc_coupling_pos=0, msr_coupling_pos=0)
    sens = JacobianSensitivities(net)
    _, msc_bus = _first_bank(meta, "MSC")
    _, msr_bus = _first_bank(meta, "MSR")
    assert msc_bus == msr_bus, "this layout is what the guard is about"
    with pytest.raises(ValueError, match="in-service shunts"):
        sens.shunt_step_sign(msc_bus)


def test_sign_is_one_on_a_bus_without_a_shunt():
    """The queries stay usable on an arbitrary bus."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    bus = int(meta.hv_networks[0].bus_indices[0])
    assert bus not in set(int(b) for b in net.shunt["bus"])
    assert sens.shunt_step_sign(bus) == 1.0
