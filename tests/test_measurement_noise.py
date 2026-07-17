"""Tests for the component-wise controller-facing metering model."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from configs.config import MeasurementNoiseConfig
from core.measurement import Measurement
from core.measurement_noise import MeasurementNoiseModel


def _net():
    return SimpleNamespace(
        bus=pd.DataFrame({"vn_kv": [345.0, 110.0]}, index=[0, 1]),
        line=pd.DataFrame(
            {
                "from_bus": [0],
                "to_bus": [1],
                "max_i_ka": [1.0],
                "ct_primary_i_ka": [2.0],
            },
            index=[10],
        ),
        res_line=pd.DataFrame(
            {
                "p_from_mw": [50.0],
                "p_to_mw": [-49.5],
            },
            index=[10],
        ),
        trafo=pd.DataFrame(columns=["sn_mva", "hv_bus"]),
        res_trafo=pd.DataFrame(columns=["p_hv_mw"]),
        trafo3w=pd.DataFrame(
            {"sn_hv_mva": [100.0], "hv_bus": [0]},
            index=[20],
        ),
        res_trafo3w=pd.DataFrame({"p_hv_mw": [50.0]}, index=[20]),
        sgen=pd.DataFrame({"sn_mva": [50.0], "bus": [1]}, index=[30]),
        gen=pd.DataFrame({"sn_mva": [200.0], "bus": [0]}, index=[40]),
    )


def _measurement() -> Measurement:
    return Measurement(
        iteration=7,
        bus_indices=np.array([0, 1], dtype=np.int64),
        voltage_magnitudes_pu=np.array([1.02, 0.99]),
        voltage_angles_deg=np.array([1.5, -0.5]),
        branch_indices=np.array([10], dtype=np.int64),
        current_magnitudes_ka=np.array([0.1]),
        interface_transformer_indices=np.array([20], dtype=np.int64),
        interface_q_hv_side_mvar=np.array([0.0]),
        der_indices=np.array([30], dtype=np.int64),
        der_q_mvar=np.array([5.0]),
        der_p_mw=np.array([10.0]),
        der_vm_pu_ref=np.array([1.03]),
        oltc_indices=np.array([5], dtype=np.int64),
        oltc_tap_positions=np.array([2], dtype=np.int64),
        shunt_indices=np.array([6], dtype=np.int64),
        shunt_states=np.array([-1], dtype=np.int64),
        gen_indices=np.array([40], dtype=np.int64),
        gen_vm_pu=np.array([1.04]),
        gen_p_mw=np.array([100.0]),
        gen_q_mvar=np.array([40.0]),
        tie_line_indices=np.array([10], dtype=np.int64),
        tie_line_endpoint_buses=np.array([0], dtype=np.int64),
        tie_line_p_mw=np.array([50.0]),
        tie_line_q_mvar=np.array([10.0]),
    )


def _assert_complex_channel_within_component_bounds(
    p_true,
    q_true,
    p_measured,
    q_measured,
    *,
    vt_bound,
    ct_bound,
    meter_bound,
    phase_bound_deg,
):
    ratio = complex(p_measured, q_measured) / complex(p_true, q_true)
    gain_min = (1.0 - vt_bound) * (1.0 - ct_bound) * (
        1.0 - meter_bound
    )
    gain_max = (1.0 + vt_bound) * (1.0 + ct_bound) * (
        1.0 + meter_bound
    )
    assert gain_min - 1e-12 <= abs(ratio) <= gain_max + 1e-12
    assert abs(np.angle(ratio, deg=True)) <= phase_bound_deg + 1e-12


def test_profiles_are_component_wise_and_equivalent_bounds_match():
    minimum = MeasurementNoiseConfig(profile="minimum")
    conservative = MeasurementNoiseConfig(profile="conservative")

    assert minimum.profile_components() == {
        "ehv_voltage_transformer": 0.001,
        "hv_voltage_transformer": 0.002,
        "voltage_meter": 0.001,
        "current_transformer": 0.002,
        "current_meter": 0.001,
        "power_meter_gain": 0.002,
        "power_phase_angle_deg": 0.1,
    }
    c = conservative.profile_components()
    assert c["ehv_voltage_transformer"] == 0.001
    assert c["hv_voltage_transformer"] == 0.002
    assert c["voltage_meter"] == 0.002
    assert c["current_transformer"] == 0.005
    assert c["current_meter"] == 0.002
    assert c["power_meter_gain"] == 0.005
    assert np.tan(np.deg2rad(c["power_phase_angle_deg"])) == pytest.approx(
        0.02
    )

    b_min = minimum.equivalent_bounds()
    assert b_min["voltage_ehv"] == pytest.approx(np.hypot(0.001, 0.001))
    assert b_min["voltage_hv"] == pytest.approx(np.hypot(0.002, 0.001))
    assert b_min["current"] == pytest.approx(np.hypot(0.002, 0.001))
    assert b_min["active_power_ehv"] == pytest.approx(0.003)

    b_con = conservative.equivalent_bounds()
    assert b_con["voltage_ehv"] == pytest.approx(np.hypot(0.001, 0.002))
    assert b_con["voltage_hv"] == pytest.approx(np.hypot(0.002, 0.002))
    assert b_con["current"] == pytest.approx(np.hypot(0.005, 0.002))


def test_invalid_profile_and_component_override_are_rejected():
    with pytest.raises(ValueError, match="minimum, conservative|conservative, minimum"):
        MeasurementNoiseConfig(profile="unknown").validate()
    with pytest.raises(ValueError, match="unknown measurement-noise component"):
        MeasurementNoiseConfig(
            component_half_width_overrides={"imaginary_sensor": 0.1}
        ).validate()


def test_disabled_model_preserves_packet_exactly():
    original = _measurement()
    packet = deepcopy(original)
    model = MeasurementNoiseModel(
        MeasurementNoiseConfig(enabled=False, profile="conservative")
    )

    result = model.apply(packet, _net(), sample_id=("control", 7))

    assert result is packet
    for name, value in vars(original).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(packet, name), value)


def test_conservative_profile_respects_component_chain_bounds():
    original = _measurement()
    packet = deepcopy(original)
    cfg = MeasurementNoiseConfig(
        enabled=True,
        profile="conservative",
        seed=17,
    )
    model = MeasurementNoiseModel(cfg)

    model.apply(packet, _net(), sample_id=("control", 7))
    c = cfg.profile_components()

    # Multiplicative EHV and lower-voltage VT/PMD chains.
    ehv_hard = (
        (1.0 + c["ehv_voltage_transformer"])
        * (1.0 + c["voltage_meter"])
        - 1.0
    )
    hv_hard = (
        (1.0 + c["hv_voltage_transformer"])
        * (1.0 + c["voltage_meter"])
        - 1.0
    )
    assert (
        abs(packet.voltage_magnitudes_pu[0] / original.voltage_magnitudes_pu[0] - 1.0)
        <= ehv_hard + 1e-12
    )
    assert (
        abs(packet.voltage_magnitudes_pu[1] / original.voltage_magnitudes_pu[1] - 1.0)
        <= hv_hard + 1e-12
    )

    # Explicit 2 kA CT rating, rather than the 1 kA thermal line limit, sets
    # the 20 %-rating scale floor: max(0.1, 0.2*2.0) = 0.4 kA.
    current_hard = (
        (1.0 + c["current_transformer"])
        * (1.0 + c["current_meter"])
        - 1.0
    )
    assert abs(packet.current_magnitudes_ka[0] - 0.1) <= (
        current_hard * 0.4 + 1e-12
    )
    assert model._ct_rating_ka(_net(), 10) == 2.0

    _assert_complex_channel_within_component_bounds(
        10.0,
        5.0,
        packet.der_p_mw[0],
        packet.der_q_mvar[0],
        vt_bound=c["hv_voltage_transformer"],
        ct_bound=c["current_transformer"],
        meter_bound=c["power_meter_gain"],
        phase_bound_deg=c["power_phase_angle_deg"],
    )
    _assert_complex_channel_within_component_bounds(
        100.0,
        40.0,
        packet.gen_p_mw[0],
        packet.gen_q_mvar[0],
        vt_bound=c["ehv_voltage_transformer"],
        ct_bound=c["current_transformer"],
        meter_bound=c["power_meter_gain"],
        phase_bound_deg=c["power_phase_angle_deg"],
    )
    _assert_complex_channel_within_component_bounds(
        50.0,
        10.0,
        packet.tie_line_p_mw[0],
        packet.tie_line_q_mvar[0],
        vt_bound=c["ehv_voltage_transformer"],
        ct_bound=c["current_transformer"],
        meter_bound=c["power_meter_gain"],
        phase_bound_deg=c["power_phase_angle_deg"],
    )

    # Q=0 is still affected through P*sin(dphi), but remains within the full
    # complex-power component envelope.
    gain_max = (
        (1.0 + c["ehv_voltage_transformer"])
        * (1.0 + c["current_transformer"])
        * (1.0 + c["power_meter_gain"])
    )
    phase_max = np.deg2rad(c["power_phase_angle_deg"])
    assert abs(packet.interface_q_hv_side_mvar[0]) <= (
        50.0 * gain_max * np.sin(phase_max) + 1e-12
    )

    # Digital states, commanded references, and voltage angles remain exact.
    np.testing.assert_array_equal(
        packet.oltc_tap_positions, original.oltc_tap_positions
    )
    np.testing.assert_array_equal(packet.shunt_states, original.shunt_states)
    np.testing.assert_array_equal(packet.gen_vm_pu, original.gen_vm_pu)
    np.testing.assert_array_equal(packet.der_vm_pu_ref, original.der_vm_pu_ref)
    np.testing.assert_array_equal(
        packet.voltage_angles_deg, original.voltage_angles_deg
    )


def test_power_phase_error_creates_q_error_at_unity_power_factor():
    overrides = {
        "ehv_voltage_transformer": 0.0,
        "hv_voltage_transformer": 0.0,
        "voltage_meter": 0.0,
        "current_transformer": 0.0,
        "current_meter": 0.0,
        "power_meter_gain": 0.0,
        "power_phase_angle_deg": 1.0,
    }
    model = MeasurementNoiseModel(
        MeasurementNoiseConfig(
            enabled=True,
            sample_noise_fraction=0.0,
            component_half_width_overrides=overrides,
        )
    )
    packet = _measurement()
    packet.der_p_mw[0] = 10.0
    packet.der_q_mvar[0] = 0.0

    model.apply(packet, _net(), sample_id=("control", 1))

    assert packet.der_q_mvar[0] != 0.0
    assert abs(complex(packet.der_p_mw[0], packet.der_q_mvar[0])) == pytest.approx(
        10.0
    )
    assert abs(np.angle(
        complex(packet.der_p_mw[0], packet.der_q_mvar[0]), deg=True
    )) <= 1.0


def test_seed_sample_cache_and_persistent_bias_behaviour():
    cfg = MeasurementNoiseConfig(
        enabled=True,
        profile="minimum",
        seed=1234,
    )
    model_a = MeasurementNoiseModel(cfg)
    model_b = MeasurementNoiseModel(deepcopy(cfg))

    packet_a1 = model_a.apply(
        _measurement(), _net(), sample_id=("control", 7)
    )
    packet_a2 = model_a.apply(
        _measurement(), _net(), sample_id=("control", 7)
    )
    packet_b = model_b.apply(
        _measurement(), _net(), sample_id=("control", 7)
    )

    fields = (
        "voltage_magnitudes_pu",
        "current_magnitudes_ka",
        "interface_q_hv_side_mvar",
        "der_p_mw",
        "der_q_mvar",
        "gen_p_mw",
        "gen_q_mvar",
        "tie_line_p_mw",
        "tie_line_q_mvar",
    )
    for field in fields:
        np.testing.assert_array_equal(
            getattr(packet_a1, field), getattr(packet_a2, field)
        )
        np.testing.assert_array_equal(
            getattr(packet_a1, field), getattr(packet_b, field)
        )

    next_packet = model_a.apply(
        _measurement(), _net(), sample_id=("control", 8)
    )
    assert not np.array_equal(
        packet_a1.voltage_magnitudes_pu,
        next_packet.voltage_magnitudes_pu,
    )

    # With the per-sample share disabled, the same persistent calibration
    # errors remain unchanged across controller instants.
    bias_only = MeasurementNoiseModel(
        MeasurementNoiseConfig(
            enabled=True,
            profile="minimum",
            seed=1234,
            sample_noise_fraction=0.0,
        )
    )
    first = bias_only.apply(
        _measurement(), _net(), sample_id=("control", 7)
    )
    second = bias_only.apply(
        _measurement(), _net(), sample_id=("control", 8)
    )
    for field in fields:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_ct_rating_fallback_is_explicitly_configurable():
    net = _net()
    net.line = net.line.drop(columns=["ct_primary_i_ka"])

    with_fallback = MeasurementNoiseModel(
        MeasurementNoiseConfig(enabled=True)
    )
    assert with_fallback._ct_rating_ka(net, 10) == 1.0

    without_fallback = MeasurementNoiseModel(
        MeasurementNoiseConfig(
            enabled=True,
            allow_line_rating_as_ct_fallback=False,
        )
    )
    assert np.isnan(without_fallback._ct_rating_ka(net, 10))


def test_initialisation_noise_can_be_disabled_independently():
    packet = _measurement()
    original = deepcopy(packet)
    model = MeasurementNoiseModel(
        MeasurementNoiseConfig(
            enabled=True,
            profile="conservative",
            apply_during_initialisation=False,
        )
    )

    model.apply(
        packet,
        _net(),
        sample_id=("initialisation", 0),
        initialisation=True,
    )

    np.testing.assert_array_equal(
        packet.voltage_magnitudes_pu,
        original.voltage_magnitudes_pu,
    )
    np.testing.assert_array_equal(packet.der_q_mvar, original.der_q_mvar)
