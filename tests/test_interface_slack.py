#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the DSO interface slack-decoupling helper
===================================================

Covers ``experiments.helpers.plant_io.decouple_trafo3w_hv_with_slack``:
replacing the TN feed at each DSO_2 coupling 3W transformer's HV side with a
voltage-holding slack pinned to the operating point, so DSO_2 becomes an
island fed by stiff (distributed) slacks.
"""

from __future__ import annotations

import pandapower as pp
import pytest

from network.ieee39.build import build_ieee39_net
from network.ieee39.hv_networks import add_hv_networks
from experiments.helpers.plant_io import decouple_trafo3w_hv_with_slack


@pytest.fixture(scope="module")
def ieee39_with_dso2():
    """Combined IEEE39 net with HV sub-networks, solved once."""
    net, meta = build_ieee39_net(scenario="wind_replace", verbose=False)
    meta = add_hv_networks(net, meta, verbose=False)
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True)
    assert net.converged
    dso2 = next(h for h in meta.hv_networks if h.net_id == "DSO_2")
    return net, meta, dso2


def test_decouple_repoints_hv_and_adds_slacks(ieee39_with_dso2):
    net, _meta, dso2 = ieee39_with_dso2
    couplers = list(dso2.coupling_trafo_indices)
    assert len(couplers) == 3

    orig_hv_bus = {t: int(net.trafo3w.at[t, "hv_bus"]) for t in couplers}
    vm_pre = {t: float(net.res_bus.at[orig_hv_bus[t], "vm_pu"]) for t in couplers}
    n_ext_before = len(net.ext_grid)
    n_bus_before = len(net.bus)

    created = decouple_trafo3w_hv_with_slack(net, couplers)

    # One new slack ext_grid + one new isolated bus per coupler.
    assert len(created) == len(couplers)
    assert len(net.ext_grid) == n_ext_before + len(couplers)
    assert len(net.bus) == n_bus_before + len(couplers)

    for t, eg in zip(couplers, created):
        new_hv = int(net.trafo3w.at[t, "hv_bus"])
        # HV side now points at a brand-new bus, not the original TN bus.
        assert new_hv != orig_hv_bus[t]
        assert new_hv >= n_bus_before
        # The slack sits on that new bus and is pinned to the op-point voltage.
        assert int(net.ext_grid.at[eg, "bus"]) == new_hv
        assert net.ext_grid.at[eg, "vm_pu"] == pytest.approx(vm_pre[t])

    # The islanded DSO_2 (fed by the 3 boundary slacks) still solves.  pandapower
    # cannot run distributed_slack across several islands, so the decoupled net
    # must be solved with distributed_slack=False (each island keeps its own
    # reference bus(es)).
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=False)
    assert net.converged


def test_requires_solved_net():
    empty = pp.create_empty_network()
    with pytest.raises(RuntimeError):
        decouple_trafo3w_hv_with_slack(empty, [0])
