"""Regression tests for the full-model DSO ZIP/injection separation layer."""

from __future__ import annotations

import pytest

from export.make_snapshots import DEFAULT_T0, build_snapshot_state
from network.ieee39.aux_load_buses import (
    AUX_LOAD_LINK_LENGTH_KM,
    AUX_LOAD_LINK_R_OHM,
    AUX_LOAD_LINK_X_OHM,
)


@pytest.fixture(scope="module")
def full_state():
    return build_snapshot_state("full", DEFAULT_T0, verbose=0)


def test_dso_zip_loads_and_fixed_sgens_use_distinct_ppc_nodes(full_state):
    net = full_state.net
    meta = full_state.meta

    active_load_buses = {
        int(net.load.at[i, "bus"])
        for i in net.load.index
        if bool(net.load.at[i, "in_service"])
    }
    active_sgen_buses = {
        int(net.sgen.at[i, "bus"])
        for i in net.sgen.index
        if bool(net.sgen.at[i, "in_service"])
    }
    assert active_load_buses.isdisjoint(active_sgen_buses)

    dso_aux = {
        int(b) for hv in meta.hv_networks
        for b in hv.internal_aux_bus_indices
    }
    assert dso_aux
    assert dso_aux.issubset(set(meta.internal_aux_bus_indices))
    assert dso_aux.isdisjoint(meta.dn_bus_indices)

    for hv in meta.hv_networks:
        assert len(hv.internal_aux_bus_indices) == 7
        assert len(hv.internal_aux_parent_buses) == 7
        assert len(hv.internal_aux_line_indices) == 7
        assert set(hv.internal_aux_bus_indices).isdisjoint(hv.bus_indices)
        assert set(hv.internal_aux_line_indices).isdisjoint(hv.line_indices)

        for aux, parent, line in zip(
            hv.internal_aux_bus_indices,
            hv.internal_aux_parent_buses,
            hv.internal_aux_line_indices,
        ):
            assert net.bus.at[aux, "subnet"] == "DN_AUX"
            assert net.line.at[line, "subnet"] == "DN_AUX"
            assert int(net.line.at[line, "from_bus"]) == parent
            assert int(net.line.at[line, "to_bus"]) == aux
            assert float(net.line.at[line, "length_km"]) == pytest.approx(
                AUX_LOAD_LINK_LENGTH_KM
            )
            assert (
                float(net.line.at[line, "r_ohm_per_km"])
                * AUX_LOAD_LINK_LENGTH_KM
            ) == pytest.approx(AUX_LOAD_LINK_R_OHM)
            assert (
                float(net.line.at[line, "x_ohm_per_km"])
                * AUX_LOAD_LINK_LENGTH_KM
            ) == pytest.approx(AUX_LOAD_LINK_X_OHM)

            hv_loads_at_aux = [
                i for i in hv.load_indices
                if int(net.load.at[i, "bus"]) == aux
            ]
            hv_sgens_at_parent = [
                i for i in hv.sgen_indices
                if int(net.sgen.at[i, "bus"]) == parent
            ]
            assert len(hv_loads_at_aux) == 2  # constant + profile row
            assert hv_sgens_at_parent


def test_public_full_solution_has_closed_active_power_balance(full_state):
    """Protect against pandapower voltage-scaling fixed DSO injections."""
    net = full_state.net
    p_generation = float(net.res_gen.p_mw.sum() + net.res_sgen.p_mw.sum())
    if len(net.ext_grid):
        p_generation += float(net.res_ext_grid.p_mw.sum())
    p_load = float(net.res_load.p_mw.sum())
    p_losses = (
        float((net.res_line.p_from_mw + net.res_line.p_to_mw).sum())
        + float((net.res_trafo.p_hv_mw + net.res_trafo.p_lv_mw).sum())
        + float((
            net.res_trafo3w.p_hv_mw
            + net.res_trafo3w.p_mv_mw
            + net.res_trafo3w.p_lv_mw
        ).sum())
    )
    assert p_generation - p_load - p_losses == pytest.approx(0.0, abs=1e-5)
