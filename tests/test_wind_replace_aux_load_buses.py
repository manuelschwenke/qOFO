"""Regression tests for the wind/ZIP auxiliary-load-bus workaround."""

from __future__ import annotations

from copy import deepcopy

import pandapower as pp
import pytest
from pandapower.pypower.idx_bus import PD, QD

from network.ieee39 import build_ieee39_net
from network.ieee39.load_model import apply_zip_load_model
from network.ieee39.scenarios.wind_replace import (
    AUX_LOAD_LINK_LENGTH_KM,
    AUX_LOAD_LINK_R_OHM,
    AUX_LOAD_LINK_X_OHM,
)
from network.zone_partition import fixed_zone_partition_ieee39


@pytest.fixture(scope="module")
def wind_case():
    return build_ieee39_net(scenario="base_410", verbose=False)


def test_auxiliary_topology_is_explicit_and_internal(wind_case):
    net, meta = wind_case
    triples = list(zip(
        meta.internal_aux_bus_indices,
        meta.internal_aux_parent_buses,
        meta.internal_aux_line_indices,
    ))

    assert {parent for _aux, parent, _line in triples} == {18, 24}
    assert len(triples) == 2
    assert set(meta.internal_aux_bus_indices).isdisjoint(meta.tn_bus_indices)
    assert set(meta.internal_aux_line_indices).isdisjoint(meta.tn_line_indices)

    for aux, parent, line in triples:
        assert net.bus.at[aux, "subnet"] == "TN_AUX"
        assert net.line.at[line, "subnet"] == "TN_AUX"
        assert int(net.line.at[line, "from_bus"]) == parent
        assert int(net.line.at[line, "to_bus"]) == aux
        assert float(net.line.at[line, "length_km"]) == pytest.approx(
            AUX_LOAD_LINK_LENGTH_KM
        )
        assert (
            float(net.line.at[line, "length_km"])
            * float(net.line.at[line, "r_ohm_per_km"])
        ) == pytest.approx(AUX_LOAD_LINK_R_OHM)
        assert (
            float(net.line.at[line, "length_km"])
            * float(net.line.at[line, "x_ohm_per_km"])
        ) == pytest.approx(AUX_LOAD_LINK_X_OHM)
        assert float(net.line.at[line, "c_nf_per_km"]) == 0.0
        assert float(net.line.at[line, "g_us_per_km"]) == 0.0

        moved_loads = net.load.index[net.load["bus"].astype(int) == aux]
        assert len(moved_loads) == 2  # constant and profile-driven halves
        assert not (net.load["bus"].astype(int) == parent).any()
        assert (
            net.load.loc[moved_loads, "subnet"].astype(str) == "TN"
        ).all()
        assert (
            net.sgen.loc[list(meta.tso_der_indices), "bus"].astype(int)
            == parent
        ).any()

    # The fixed map is the controller's physical TN monitoring partition.
    zone_map, _ = fixed_zone_partition_ieee39(net)
    monitored = {int(bus) for buses in zone_map.values() for bus in buses}
    assert monitored.isdisjoint(meta.internal_aux_bus_indices)
    assert set(meta.internal_aux_bus_indices).isdisjoint(meta.tso_der_buses)


def test_zip_aggregation_no_longer_scales_wind_injection(wind_case):
    net, meta = wind_case
    net = deepcopy(net)
    apply_zip_load_model(net, anchor_vm_pu=1.03, verbose=False)
    pp.runpp(
        net,
        calculate_voltage_angles=True,
        distributed_slack=True,
        max_iteration=50,
        voltage_depend_loads=True,
    )

    for parent in meta.internal_aux_parent_buses:
        ppc_bus = int(net._pd2ppc_lookups["bus"][parent])
        sgens = net.sgen.loc[
            (net.sgen["bus"].astype(int) == parent)
            & net.sgen["in_service"].astype(bool)
        ]
        # With no load on the same ppc node, the internal demand is exactly
        # the negative constant-PQ wind injection.  The former defect yielded
        # approximately -P_wind * V and -Q_wind * V^2 here.
        assert float(net._ppc["bus"][ppc_bus, PD]) == pytest.approx(
            -float(sgens["p_mw"].sum()), abs=1e-9
        )
        assert float(net._ppc["bus"][ppc_bus, QD]) == pytest.approx(
            -float(sgens["q_mvar"].sum()), abs=1e-9
        )

