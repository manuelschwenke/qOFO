from __future__ import annotations

import copy

import numpy as np

from analysis.annual_dso_pq_characterization import (
    DER_COSPHI_SIGN,
    ProfileApplicationMap,
    _build_study_network,
    _fill_profile_gaps,
    _run_power_flow,
)
from core.profiles import DEFAULT_PROFILES_CSV, load_profiles


def test_isolated_dso_cases_have_stiff_equal_weight_primaries() -> None:
    study_cases, _meta = _build_study_network(
        "rural_700",
        primary_vm_pu=1.03,
        oltc_vref_pu=1.03,
    )

    assert len(study_cases) == 4
    for net, hv in study_cases:
        active_ext_grids = net.ext_grid.loc[net.ext_grid.in_service]
        assert len(active_ext_grids) == 3
        assert set(active_ext_grids.bus.astype(int)) == set(
            hv.coupling_ieee_buses
        )
        np.testing.assert_allclose(active_ext_grids.vm_pu, 1.03)
        np.testing.assert_allclose(active_ext_grids.va_degree, 0.0)
        np.testing.assert_allclose(active_ext_grids.slack_weight, 1.0 / 3.0)

        assert set(net.trafo3w.index[net.trafo3w.in_service]) == set(
            hv.coupling_trafo_indices
        )
        assert set(net.load.index[net.load.in_service]) == set(hv.load_indices)
        assert set(net.sgen.index[net.sgen.in_service]) == set(
            hv.sgen_indices
        )
        assert not net.gen.in_service.any()
        assert len(net["annual_probe_oltc_controller_indices"]) == 3


def test_first_rural_profile_converges_at_unity_power_factor() -> None:
    study_cases, _meta = _build_study_network(
        "rural_700",
        primary_vm_pu=1.03,
        oltc_vref_pu=1.03,
    )
    profiles, _audit = _fill_profile_gaps(
        load_profiles(DEFAULT_PROFILES_CSV, timestep_min=15)
    )
    first_profile = profiles.to_numpy(dtype=float)[0]

    for net, hv in study_cases:
        profile_map = ProfileApplicationMap.from_net(
            net,
            profiles,
            tuple(int(index) for index in hv.sgen_indices),
        )
        profile_map.apply(net, first_profile, der_cosphi=1.0)
        converged, used_retry = _run_power_flow(
            net,
            warm_start=False,
            recycle=True,
        )

        assert converged
        assert not used_retry
        assert np.allclose(net.sgen.loc[list(hv.sgen_indices), "q_mvar"], 0.0)
        assert np.allclose(
            net.res_bus.loc[list(hv.coupling_ieee_buses), "vm_pu"],
            1.03,
        )


def test_dso_specific_stress_overrides_change_only_dso_3() -> None:
    study_cases, _meta = _build_study_network(
        "rural_700",
        primary_vm_pu=1.03,
        oltc_vref_pu=1.03,
        dso_der_scale={"DSO_3": 2.0},
        dso_load_p_scale={"DSO_3": 2.0},
    )

    p_profile_base: dict[str, float] = {}
    q_profile_base: dict[str, float] = {}
    for net, hv in study_cases:
        dso_id = str(hv.net_id)
        expected_der_mw = 1400.0 if dso_id == "DSO_3" else 700.0

        assert np.isclose(
            net.sgen.loc[list(hv.sgen_indices), "base_p_mw"].sum(),
            expected_der_mw,
        )
        p_profile_base[dso_id] = float(
            net.load.loc[list(hv.load_indices), "base_p_mw"].sum(),
        )
        q_profile_base[dso_id] = float(
            net.load.loc[list(hv.load_indices), "base_q_mvar"].sum(),
        )
        expected_p_reference_mw = (
            523.6075 if dso_id == "DSO_3" else 261.80375
        )
        assert np.isclose(hv.total_ref_p_mw, expected_p_reference_mw)
        assert np.isclose(hv.total_ref_q_mvar, 500.0)
        line_indices = list(hv.line_indices)
        line_rows = net.line.loc[line_indices]
        line_types = set(line_rows["std_type"])
        if dso_id == "DSO_3":
            assert line_types == {"490-AL1/64-ST1A 110.0"}
            assert np.allclose(
                net.line.loc[line_indices, "max_i_ka"],
                0.960,
            )
            reinforced = line_rows.loc[
                line_rows["name"] == "DSO_3|Line_(5-6)"
            ]
            assert len(reinforced) == 1
            assert int(reinforced.iloc[0]["parallel"]) == 2
            other_parallel = line_rows.loc[
                line_rows["name"] != "DSO_3|Line_(5-6)",
                "parallel",
            ].to_numpy(dtype=int)
            assert np.all(other_parallel == 1)
        else:
            assert line_types == {"305-AL1/39-ST1A 110.0"}
            assert np.allclose(
                net.line.loc[line_indices, "max_i_ka"],
                0.740,
            )
            assert np.all(line_rows["parallel"].to_numpy(dtype=int) == 1)

    for dso_id in ("DSO_2", "DSO_4"):
        assert np.isclose(p_profile_base[dso_id], p_profile_base["DSO_1"])
        assert np.isclose(q_profile_base[dso_id], q_profile_base["DSO_1"])
    assert np.isclose(
        p_profile_base["DSO_3"],
        2.0 * p_profile_base["DSO_1"],
    )
    for dso_id in ("DSO_2", "DSO_3", "DSO_4"):
        assert np.isclose(q_profile_base[dso_id], q_profile_base["DSO_1"])


def test_all_dsos_use_profile_only_500_mvar_reactive_load() -> None:
    study_cases, _meta = _build_study_network(
        "rural_700",
        primary_vm_pu=1.03,
        oltc_vref_pu=1.03,
    )

    for net, hv in study_cases:
        loads = net.load.loc[list(hv.load_indices)]
        constant = loads.loc[loads.profile_q.isna()]
        profiled = loads.loc[loads.profile_q.notna()]
        assert np.allclose(constant.base_q_mvar, 0.0)
        assert np.allclose(constant.q_mvar, 0.0)
        assert np.isclose(profiled.base_q_mvar.sum(), 500.0)
        assert np.isclose(hv.total_ref_q_mvar, 500.0)


def test_nonunity_dso_power_factor_is_inductive() -> None:
    assert DER_COSPHI_SIGN == -1

    study_cases, _meta = _build_study_network(
        "rural_700",
        primary_vm_pu=1.03,
        oltc_vref_pu=1.03,
    )
    profiles, _audit = _fill_profile_gaps(
        load_profiles(DEFAULT_PROFILES_CSV, timestep_min=15)
    )
    net, hv = study_cases[0]
    profile_map = ProfileApplicationMap.from_net(
        net,
        profiles,
        tuple(int(index) for index in hv.sgen_indices),
    )
    profile_map.apply(
        net,
        profiles.to_numpy(dtype=float)[0],
        der_cosphi=0.98,
    )

    dso_q = net.sgen.loc[list(hv.sgen_indices), "q_mvar"].to_numpy(float)
    assert np.all(dso_q <= 0.0)
    assert np.any(dso_q < 0.0)


def test_oltc_recycle_matches_full_rebuild_after_profile_change() -> None:
    study_cases, _meta = _build_study_network(
        "rural_700",
        primary_vm_pu=1.03,
        oltc_vref_pu=1.03,
        dso_der_scale={"DSO_3": 2.0},
        dso_load_p_scale={"DSO_3": 2.0},
    )
    profiles, _audit = _fill_profile_gaps(
        load_profiles(DEFAULT_PROFILES_CSV, timestep_min=15)
    )
    net_recycle, hv = next(
        (net, hv)
        for net, hv in study_cases
        if str(hv.net_id) == "DSO_3"
    )
    profile_map = ProfileApplicationMap.from_net(
        net_recycle,
        profiles,
        tuple(int(index) for index in hv.sgen_indices),
    )

    low_q = profiles["mv_rural_qload"].idxmin()
    high_q = profiles["mv_rural_qload"].idxmax()
    profile_map.apply(
        net_recycle,
        profiles.loc[low_q].to_numpy(dtype=float),
        der_cosphi=1.0,
    )
    converged, _retried = _run_power_flow(
        net_recycle,
        warm_start=False,
        recycle=True,
    )
    assert converged

    net_full = copy.deepcopy(net_recycle)
    for net in (net_recycle, net_full):
        profile_map.apply(
            net,
            profiles.loc[high_q].to_numpy(dtype=float),
            der_cosphi=1.0,
        )

    assert _run_power_flow(net_recycle, warm_start=True, recycle=True)[0]
    assert _run_power_flow(net_full, warm_start=True, recycle=False)[0]

    line_indices = list(hv.line_indices)
    coupler_indices = list(hv.coupling_trafo_indices)
    np.testing.assert_allclose(
        net_recycle.res_line.loc[line_indices, "loading_percent"],
        net_full.res_line.loc[line_indices, "loading_percent"],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        net_recycle.res_trafo3w.loc[coupler_indices, "loading_percent"],
        net_full.res_trafo3w.loc[coupler_indices, "loading_percent"],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        net_recycle.trafo3w.loc[coupler_indices, "tap_pos"],
        net_full.trafo3w.loc[coupler_indices, "tap_pos"],
    )
