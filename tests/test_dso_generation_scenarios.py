"""Regression tests for the public IEEE39 + synthetic-DS capacity scenarios."""

from __future__ import annotations

import pandas as pd
import pytest

from network.ieee39 import add_hv_networks, build_ieee39_net
from network.ieee39.constants import DSO_DER_CAPACITY_SCENARIOS
from network.ieee39.scenarios import SCENARIO_REGISTRY


EXPECTED = {
    "base_410": {
        "wind_internal": [40, 60, 50],
        "wind_coupling": [40, 40, 40],
        "pv": [50, 40, 30, 20],
        "wind_total": 270,
        "pv_total": 140,
        "total": 410,
    },
    "rural_700": {
        "wind_internal": [70, 100, 80],
        "wind_coupling": [70, 70, 70],
        "pv": [80, 70, 50, 40],
        "wind_total": 460,
        "pv_total": 240,
        "total": 700,
    },
}


def test_public_registry_contains_only_capacity_named_scenarios():
    assert set(SCENARIO_REGISTRY) == {"base_410", "rural_700"}
    assert set(DSO_DER_CAPACITY_SCENARIOS) == set(SCENARIO_REGISTRY)


@pytest.mark.parametrize("scenario", ["base_410", "rural_700"])
def test_capacity_scenario_builds_exact_integer_ratings(scenario):
    net, meta = build_ieee39_net(scenario=scenario, verbose=False)
    meta = add_hv_networks(
        net,
        meta,
        install_tso_tertiary_shunts=False,
        verbose=False,
    )

    assert net["ieee39_scenario"] == scenario
    assert net["dso_generation_scenario"] == scenario
    assert net.converged

    expected = EXPECTED[scenario]
    for hv in meta.hv_networks:
        dso_sgen = net.sgen.loc[list(hv.sgen_indices)].copy()
        names = dso_sgen["name"].astype(str)

        wind_internal = sorted(
            pd.to_numeric(
                dso_sgen.loc[names.str.contains(r"\|Wind_"), "p_mw"]
            ).astype(int)
        )
        wind_coupling = sorted(
            pd.to_numeric(
                dso_sgen.loc[names.str.contains(r"\|WP_STATCOM_"), "p_mw"]
            ).astype(int)
        )
        pv = sorted(
            pd.to_numeric(
                dso_sgen.loc[names.str.contains(r"\|PV_"), "p_mw"]
            ).astype(int)
        )

        assert wind_internal == sorted(expected["wind_internal"])
        assert wind_coupling == sorted(expected["wind_coupling"])
        assert pv == sorted(expected["pv"])
        assert sum(wind_internal) + sum(wind_coupling) == expected["wind_total"]
        assert sum(pv) == expected["pv_total"]
        assert int(dso_sgen["p_mw"].sum()) == expected["total"]
        assert all(float(value).is_integer() for value in dso_sgen["p_mw"])
        assert all(float(value).is_integer() for value in dso_sgen["sn_mva"])


def test_legacy_wind_replace_alias_resolves_to_base_410():
    with pytest.warns(DeprecationWarning):
        net, _ = build_ieee39_net(scenario="wind_replace", verbose=False)
    assert net["ieee39_scenario"] == "base_410"


@pytest.mark.parametrize("removed", ["base", "reduced_gen_z2"])
def test_removed_scenarios_are_rejected(removed):
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_ieee39_net(scenario=removed, verbose=False)
