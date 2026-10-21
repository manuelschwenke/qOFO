"""
Tests for the integrator-mode MSC / MSR tertiary shunt banks installed by
``add_hv_networks(tso_shunt_kind="msc_msr")``.

Verifies the device count / metadata, the pandapower sign convention
(MSC ``q_mvar < 0``, MSR ``q_mvar > 0``), and — crucially — the physical
direction: engaging an MSC step raises the tertiary voltage, engaging an MSR
step lowers it.  A sign error here would invert the integrator's commit
direction.

Author: Manuel Schwenke
Date: 2026-06-22
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


def _build_msc_msr(n_c: int = 4, n_r: int = 4):
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="rural_700", verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=True,
        tso_shunt_kind="msc_msr",
        msc_n_levels=n_c,
        msr_n_levels=n_r,
        msc_q_step_mvar=50.0,
        msr_q_step_mvar=50.0,
        verbose=False,
    )
    return net, meta


def test_two_banks_per_dso_with_metadata():
    net, meta = _build_msc_msr()
    n_dso = len(meta.hv_networks)
    assert len(meta.tso_tertiary_shunt_indices) == 2 * n_dso

    kinds = list(meta.tso_tertiary_shunt_kinds)
    assert kinds.count("MSC") == n_dso
    assert kinds.count("MSR") == n_dso
    assert len(meta.tso_tertiary_shunt_n_levels) == 2 * n_dso
    assert all(int(n) == 4 for n in meta.tso_tertiary_shunt_n_levels)

    # max_step matches the requested lattice depth; pandapower q_mvar sign
    # follows the device class.
    for idx, kind in zip(
        meta.tso_tertiary_shunt_indices, meta.tso_tertiary_shunt_kinds
    ):
        q = float(net.shunt.at[idx, "q_mvar"])
        assert int(net.shunt.at[idx, "max_step"]) == 4
        if kind == "MSC":
            assert q < 0.0, f"MSC shunt {idx} must have q_mvar < 0, got {q}"
        else:
            assert q > 0.0, f"MSR shunt {idx} must have q_mvar > 0, got {q}"


def test_msc_raises_v_msr_lowers_v():
    net, meta = _build_msc_msr()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True, distributed_slack=True)

    # First DSO's MSC and MSR.  Since 2026-10-15 they sit on different
    # tertiaries, so each is checked against the voltage of its own bus.
    msc_idx = next(
        i for i, k in zip(
            meta.tso_tertiary_shunt_indices, meta.tso_tertiary_shunt_kinds
        ) if k == "MSC"
    )
    msr_idx = next(
        i for i, k in zip(
            meta.tso_tertiary_shunt_indices, meta.tso_tertiary_shunt_kinds
        ) if k == "MSR"
    )
    bus_c = int(net.shunt.at[msc_idx, "bus"])
    bus_r = int(net.shunt.at[msr_idx, "bus"])
    assert float(net.bus.at[bus_c, "vn_kv"]) == 20.0
    assert float(net.bus.at[bus_r, "vn_kv"]) == 20.0

    v0_c = float(net.res_bus.at[bus_c, "vm_pu"])
    v0_r = float(net.res_bus.at[bus_r, "vm_pu"])

    # Engage one MSC step → its tertiary voltage rises.
    net.shunt.at[msc_idx, "step"] = 1
    pp.runpp(net, run_control=False, calculate_voltage_angles=True, distributed_slack=True)
    v_msc = float(net.res_bus.at[bus_c, "vm_pu"])
    net.shunt.at[msc_idx, "step"] = 0

    # Engage one MSR step → its tertiary voltage drops.
    net.shunt.at[msr_idx, "step"] = 1
    pp.runpp(net, run_control=False, calculate_voltage_angles=True, distributed_slack=True)
    v_msr = float(net.res_bus.at[bus_r, "vm_pu"])
    net.shunt.at[msr_idx, "step"] = 0

    assert v_msc > v0_c + 1e-4, (
        f"MSC step should raise V: v0={v0_c:.5f}, v_msc={v_msc:.5f}"
    )
    assert v_msr < v0_r - 1e-4, (
        f"MSR step should lower V: v0={v0_r:.5f}, v_msr={v_msr:.5f}"
    )


def test_one_device_per_tertiary_by_default():
    """Default placement (2026-10-15): MSC on the second coupler's tertiary,
    MSR on the first; no tertiary carries two banks."""
    net, meta = _build_msc_msr()
    for hv in meta.hv_networks:
        lv = [int(b) for b in hv.coupling_lv_bus_indices]
        banks = [
            (k, int(b)) for b, k in zip(
                meta.tso_tertiary_shunt_buses, meta.tso_tertiary_shunt_kinds
            ) if int(b) in lv
        ]
        kinds = dict(banks)
        assert kinds["MSC"] == lv[1], f"{hv.net_id}: MSC not on 2nd tertiary"
        assert kinds["MSR"] == lv[0], f"{hv.net_id}: MSR not on 1st tertiary"
        buses = [b for _, b in banks]
        assert len(buses) == len(set(buses)) == 2


def test_shared_tertiary_placement_reproducible():
    """``msc_coupling_pos=0`` restores the pre-2026-10-15 build (both banks on
    the first tertiary); an out-of-range position fails fast."""
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="rural_700", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, tso_shunt_kind="msc_msr",
        msc_n_levels=2, msr_n_levels=2, msc_coupling_pos=0, msr_coupling_pos=0,
        verbose=False,
    )
    for hv in meta.hv_networks:
        first = int(hv.coupling_lv_bus_indices[0])
        on_first = [
            k for b, k in zip(
                meta.tso_tertiary_shunt_buses, meta.tso_tertiary_shunt_kinds
            ) if int(b) == first
        ]
        assert sorted(on_first) == ["MSC", "MSR"]

    net2, meta2 = build_ieee39_net(scenario="rural_700", verbose=False)
    with pytest.raises(ValueError, match="coupling position"):
        add_hv_networks(
            net2, meta2, install_tso_tertiary_shunts=True,
            tso_shunt_kind="msc_msr", msc_coupling_pos=9, verbose=False,
        )


def test_bipolar_still_default():
    """The legacy bipolar build is unchanged when tso_shunt_kind defaults."""
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="rural_700", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, verbose=False,
    )
    n_dso = len(meta.hv_networks)
    assert len(meta.tso_tertiary_shunt_indices) == n_dso  # one bank per DSO
    assert all(k == "BIPOLAR" for k in meta.tso_tertiary_shunt_kinds)
    assert all(int(n) == 1 for n in meta.tso_tertiary_shunt_n_levels)
