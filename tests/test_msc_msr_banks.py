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

    net, meta = build_ieee39_net(scenario="base", verbose=False)
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
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # First DSO's MSC and MSR (they share the tertiary bus).
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
    bus = int(net.shunt.at[msc_idx, "bus"])
    assert int(net.shunt.at[msr_idx, "bus"]) == bus  # same tertiary
    assert float(net.bus.at[bus, "vn_kv"]) == 20.0

    v0 = float(net.res_bus.at[bus, "vm_pu"])

    # Engage one MSC step → tertiary voltage rises.
    net.shunt.at[msc_idx, "step"] = 1
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    v_msc = float(net.res_bus.at[bus, "vm_pu"])
    net.shunt.at[msc_idx, "step"] = 0

    # Engage one MSR step → tertiary voltage drops.
    net.shunt.at[msr_idx, "step"] = 1
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    v_msr = float(net.res_bus.at[bus, "vm_pu"])
    net.shunt.at[msr_idx, "step"] = 0

    assert v_msc > v0 + 1e-4, (
        f"MSC step should raise V: v0={v0:.5f}, v_msc={v_msc:.5f}"
    )
    assert v_msr < v0 - 1e-4, (
        f"MSR step should lower V: v0={v0:.5f}, v_msr={v_msr:.5f}"
    )


def test_bipolar_still_default():
    """The legacy bipolar build is unchanged when tso_shunt_kind defaults."""
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, verbose=False,
    )
    n_dso = len(meta.hv_networks)
    assert len(meta.tso_tertiary_shunt_indices) == n_dso  # one bank per DSO
    assert all(k == "BIPOLAR" for k in meta.tso_tertiary_shunt_kinds)
    assert all(int(n) == 1 for n in meta.tso_tertiary_shunt_n_levels)
