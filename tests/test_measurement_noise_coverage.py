"""Noise coverage beyond the OFO packets: the SVR result view and the droop meter.

Author decision 2026-10-27 (Case 2): the same accuracy class for every
controller, the classical SVR and the dispatch-anchored droop included.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from configs.config import MeasurementNoiseConfig
from core.measurement_noise import (
    DER_VOLTAGE_METER_KEY,
    DerVoltageMeter,
    MeasurementNoiseModel,
    measured_der_voltage,
)
from tests.test_measurement_noise import _measurement, _net


class _DictNet(dict):
    """A pandapower-like net: dict storage with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _result_net():
    base = _net()
    net = _DictNet(vars(base))
    net["res_bus"] = pd.DataFrame({"vm_pu": [1.02, 0.99]}, index=[0, 1])
    net["res_gen"] = pd.DataFrame({"p_mw": [100.0], "q_mvar": [40.0]}, index=[40])
    net["res_sgen"] = pd.DataFrame({"p_mw": [10.0], "q_mvar": [5.0]}, index=[30])
    return net


def test_view_carries_the_packet_errors_on_shared_channels():
    cfg = MeasurementNoiseConfig(profile="conservative", seed=11)
    net = _result_net()
    packet = MeasurementNoiseModel(cfg).apply(
        _measurement(), net, sample_id=("control", 3))
    view = MeasurementNoiseModel(cfg).result_view(net, sample_id=("control", 3))

    np.testing.assert_allclose(view.res_bus["vm_pu"].to_numpy(),
                               packet.voltage_magnitudes_pu, rtol=0, atol=1e-15)
    assert view.res_gen.at[40, "q_mvar"] == pytest.approx(packet.gen_q_mvar[0], abs=1e-12)
    assert view.res_gen.at[40, "p_mw"] == pytest.approx(packet.gen_p_mw[0], abs=1e-12)
    assert view.res_sgen.at[30, "q_mvar"] == pytest.approx(packet.der_q_mvar[0], abs=1e-12)


def test_view_leaves_the_plant_and_everything_else_untouched():
    cfg = MeasurementNoiseConfig(profile="minimum", seed=1)
    net = _result_net()
    before = deepcopy(net["res_bus"])
    view = MeasurementNoiseModel(cfg).result_view(net, sample_id=("control", 1))
    pd.testing.assert_frame_equal(net["res_bus"], before)
    assert view.gen is net["gen"]                    # topology / set-points exact
    assert not np.allclose(view.res_bus["vm_pu"], before["vm_pu"])


def test_disabled_model_returns_the_net_itself():
    net = _result_net()
    cfg = MeasurementNoiseConfig(enabled=False)
    assert MeasurementNoiseModel(cfg).result_view(net, sample_id=0) is net


def test_droop_reads_true_voltage_without_a_meter():
    net = _result_net()
    assert measured_der_voltage(net, 30, 1) == 0.99


def test_droop_meter_bias_is_persistent_and_cancels_against_the_anchor():
    # Pure bias (no per-sample part): the factor is the same at every sample,
    # so V - V_anchor read on the one channel is (1 + e) times the true change.
    cfg = MeasurementNoiseConfig(profile="conservative", seed=5,
                                 sample_noise_fraction=0.0)
    meter = DerVoltageMeter(cfg)
    net = _result_net()
    net[DER_VOLTAGE_METER_KEY] = meter

    meter.set_sample(("plant", 1))
    anchor = measured_der_voltage(net, 30, 1)
    net["res_bus"].at[1, "vm_pu"] = 0.97
    meter.set_sample(("plant", 7))
    running = measured_der_voltage(net, 30, 1)

    factor = anchor / 0.99
    assert factor != pytest.approx(1.0, abs=1e-9)          # a bias exists
    assert running == pytest.approx(0.97 * factor, abs=1e-15)
    assert running - anchor == pytest.approx(factor * (0.97 - 0.99), abs=1e-15)


def test_droop_meter_per_sample_part_changes_with_the_plant_step():
    cfg = MeasurementNoiseConfig(profile="conservative", seed=5,
                                 sample_noise_fraction=0.5)
    meter = DerVoltageMeter(cfg)
    net = _result_net()
    net[DER_VOLTAGE_METER_KEY] = meter
    meter.set_sample(("plant", 1))
    a = measured_der_voltage(net, 30, 1)
    meter.set_sample(("plant", 2))
    b = measured_der_voltage(net, 30, 1)
    meter.set_sample(("plant", 1))
    c = measured_der_voltage(net, 30, 1)
    assert a != b and a == c                     # deterministic per sample id


def test_droop_channel_is_not_the_scada_channel_of_the_same_bus():
    cfg = MeasurementNoiseConfig(profile="conservative", seed=3,
                                 sample_noise_fraction=0.0)
    net = _result_net()
    meter = DerVoltageMeter(cfg)
    meter.set_sample(0)
    droop_v = meter.measure(net, 30, 1)
    scada_v = MeasurementNoiseModel(cfg).result_view(
        net, sample_id=0).res_bus.at[1, "vm_pu"]
    assert droop_v != pytest.approx(scada_v, abs=1e-12)


def test_meter_survives_a_deepcopy_of_the_net():
    cfg = MeasurementNoiseConfig(profile="minimum", seed=9)
    net = _result_net()
    net[DER_VOLTAGE_METER_KEY] = DerVoltageMeter(cfg)
    net[DER_VOLTAGE_METER_KEY].set_sample(4)
    copy = deepcopy(net)
    assert measured_der_voltage(copy, 30, 1) == measured_der_voltage(net, 30, 1)
