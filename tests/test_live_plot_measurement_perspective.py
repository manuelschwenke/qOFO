from types import SimpleNamespace

import numpy as np

from core.measurement import Measurement
from experiments.helpers import MultiTSOIterationRecord
from experiments.runners._multi_tso_helpers import (
    _record_dso_measurement_snapshot,
    _record_tso_measurement_snapshot,
)


class _Bounds:
    def __init__(self, gen_limit=100.0, der_limit=50.0):
        self.gen_limit = float(gen_limit)
        self.der_limit = float(der_limit)

    def compute_gen_q_bounds(self, p_mw, vm_pu):
        shape = np.asarray(p_mw, dtype=float).shape
        return (
            np.full(shape, -self.gen_limit),
            np.full(shape, self.gen_limit),
        )

    def compute_der_q_bounds(self, p_mw):
        shape = np.asarray(p_mw, dtype=float).shape
        return (
            np.full(shape, -self.der_limit),
            np.full(shape, self.der_limit),
        )


def _measurement(**overrides):
    values = dict(
        iteration=1,
        bus_indices=np.array([], dtype=np.int64),
        voltage_magnitudes_pu=np.array([], dtype=float),
        branch_indices=np.array([], dtype=np.int64),
        current_magnitudes_ka=np.array([], dtype=float),
        interface_transformer_indices=np.array([], dtype=np.int64),
        interface_q_hv_side_mvar=np.array([], dtype=float),
        der_indices=np.array([], dtype=np.int64),
        der_q_mvar=np.array([], dtype=float),
        der_p_mw=np.array([], dtype=float),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_positions=np.array([], dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_states=np.array([], dtype=np.int64),
        gen_indices=np.array([], dtype=np.int64),
        gen_vm_pu=np.array([], dtype=float),
    )
    values.update(overrides)
    return Measurement(**values)


def test_tso_snapshot_records_noisy_controller_view_separately_from_truth():
    measurement = _measurement(
        bus_indices=np.array([11, 10]),
        voltage_magnitudes_pu=np.array([1.02, 0.98]),
        branch_indices=np.array([3]),
        current_magnitudes_ka=np.array([1.0]),
        der_indices=np.array([7]),
        der_q_mvar=np.array([10.0]),
        der_p_mw=np.array([20.0]),
        gen_indices=np.array([5]),
        gen_vm_pu=np.array([1.01]),
        gen_p_mw=np.array([40.0]),
        gen_q_mvar=np.array([20.0]),
        tie_line_indices=np.array([99]),
        tie_line_q_mvar=np.array([12.0]),
    )
    cfg = SimpleNamespace(
        voltage_bus_indices=[10, 11],
        v_setpoints_pu=np.array([1.0, 1.0]),
        gen_indices=[5],
        der_indices=[7],
        current_line_indices=[3],
        current_line_max_i_ka=[2.0],
    )
    controller = SimpleNamespace(config=cfg, actuator_bounds=_Bounds())
    rec = MultiTSOIterationRecord(step=1, time_s=60.0)
    rec.zone_v_mean[1] = 9.99
    rec.zone_q_gen[1] = np.array([999.0])

    _record_tso_measurement_snapshot(
        rec,
        {1: measurement},
        {1: SimpleNamespace()},
        {(1, 2): [99]},
        {1: controller},
        default_v_setpoint_pu=1.03,
    )

    assert rec.zone_v_meas_min[1] == 0.98
    assert rec.zone_v_meas_mean[1] == 1.0
    assert rec.zone_v_meas_max[1] == 1.02
    assert np.isclose(rec.zone_v_rms_err_meas_pu[1], 0.02)
    assert np.array_equal(rec.zone_q_gen_meas[1], [20.0])
    assert np.array_equal(rec.zone_q_der_meas[1], [10.0])
    assert np.isclose(rec.zone_line_loading_meas_mean_pct[1], 50.0)
    assert np.isclose(rec.zone_tie_q_meas_mvar[(1, 2)], 12.0)
    assert np.isclose(rec.gen_q_reserve_meas[1][0], 0.4)
    assert np.isclose(rec.tso_der_q_reserve_meas[1][0], 0.4)
    assert rec.zone_v_mean[1] == 9.99
    assert np.array_equal(rec.zone_q_gen[1], [999.0])


def test_dso_snapshot_aggregates_noisy_metering_and_interface_q():
    measurement = _measurement(
        bus_indices=np.array([20, 21]),
        voltage_magnitudes_pu=np.array([0.97, 1.01]),
        branch_indices=np.array([4]),
        current_magnitudes_ka=np.array([0.25]),
        interface_transformer_indices=np.array([30]),
        interface_q_hv_side_mvar=np.array([7.5]),
        der_indices=np.array([8]),
        der_q_mvar=np.array([-4.0]),
        der_p_mw=np.array([10.0]),
    )
    cfg = SimpleNamespace(
        voltage_bus_indices=[20, 21],
        current_line_indices=[4],
        current_line_max_i_ka=[1.0],
        der_indices=[8],
        interface_trafo_indices=[30],
    )
    controller = SimpleNamespace(
        config=cfg,
        actuator_bounds=_Bounds(der_limit=20.0),
    )
    rec = MultiTSOIterationRecord(step=1, time_s=60.0)

    _record_dso_measurement_snapshot(
        rec,
        {"DSO_1": measurement},
        {"DSO_1": controller},
        {"DSO_1": "HV_A"},
    )

    assert np.isclose(rec.dso_group_v_meas_min_pu["HV_A"], 0.97)
    assert np.isclose(rec.dso_group_v_meas_mean_pu["HV_A"], 0.99)
    assert np.isclose(rec.dso_group_v_meas_max_pu["HV_A"], 1.01)
    assert np.isclose(rec.dso_group_i_meas_mean_pct["HV_A"], 25.0)
    assert np.isclose(rec.dso_group_q_der_meas_mvar["HV_A"], -4.0)
    assert np.isclose(rec.dso_group_q_der_meas_min_mvar["HV_A"], -20.0)
    assert np.isclose(rec.dso_group_q_der_meas_max_mvar["HV_A"], 20.0)
    assert np.isclose(rec.dso_trafo_q_meas_mvar["DSO_1|trafo_30"], 7.5)
