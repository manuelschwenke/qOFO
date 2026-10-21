"""
Tests for the per-iteration voltage rescaling of shunt-bank sensitivities.

A constant-susceptance bank injects ``q_step · V²`` per step.  The shunt
sensitivity functions of :class:`JacobianSensitivities` evaluate that injection
at the cached linearisation voltage ``V_0``; the factor ``(V_meas / V_0)²``
(thesis ``eq:shunt_rescaling``) brings them to the measured voltage.  The
factor is shared by :class:`SensitivityUpdater` and, since 2026-10-14, by the
switched-shunt integrator of the multi-zone runner
(``config.shunt_int_v_rescale``).

Verifies that:
1. the ratios are ``(V_meas / V_0)²`` with an exact-identity guard and do not
   mutate their inputs;
2. ``bank_voltage_ratio`` reads ``V_0`` from the cached model network and
   ``V_meas`` from the measurement, and falls back to 1.0 with a warning when
   the bank bus is not measured;
3. on a real pandapower network, a shunt sensitivity column multiplied by the
   ratio equals the column evaluated with the injection at the measured voltage.

Author: Manuel Schwenke
Date: 2026-10-14
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandapower as pp
import pytest

from sensitivity.jacobian import JacobianSensitivities
from sensitivity.sensitivity_updater import (
    bank_voltage_ratio,
    voltage_rescaling_ratios,
)


def _measurement(bus_indices, voltages_pu):
    """Minimal stand-in carrying the two fields ``bank_voltage_ratio`` reads."""
    return SimpleNamespace(
        bus_indices=np.asarray(bus_indices, dtype=np.int64),
        voltage_magnitudes_pu=np.asarray(voltages_pu, dtype=np.float64),
    )


def test_ratios_values_and_identity_guard():
    v_meas_sq = np.array([1.03, 0.97, 1.0 + 1e-13]) ** 2
    v_cached_sq = np.array([1.00, 1.00, 1.0]) ** 2
    r = voltage_rescaling_ratios(v_meas_sq, v_cached_sq)
    np.testing.assert_allclose(r[:2], [1.03 ** 2, 0.97 ** 2], rtol=1e-14)
    # Round-off-level deviation is snapped to exactly one.
    assert r[2] == 1.0


def test_ratios_do_not_mutate_inputs():
    v_meas_sq = np.array([1.0, 1.05 ** 2])
    v_cached_sq = np.array([1.0, 1.0])
    before = (v_meas_sq.copy(), v_cached_sq.copy())
    voltage_rescaling_ratios(v_meas_sq, v_cached_sq)
    np.testing.assert_array_equal(v_meas_sq, before[0])
    np.testing.assert_array_equal(v_cached_sq, before[1])


def test_bank_voltage_ratio_uses_cached_linearisation_voltage():
    cached_net = SimpleNamespace(
        res_bus=pd.DataFrame({"vm_pu": [1.00, 0.98]}, index=[7, 9])
    )
    meas = _measurement([3, 7, 9], [1.01, 1.03, 0.98])
    assert bank_voltage_ratio(meas, cached_net, 7) == pytest.approx(1.03 ** 2)
    # Unchanged voltage -> exactly one (identity guard).
    assert bank_voltage_ratio(meas, cached_net, 9) == 1.0


def test_bank_voltage_ratio_missing_bus_warns_and_keeps_cached():
    cached_net = SimpleNamespace(res_bus=pd.DataFrame({"vm_pu": [1.0]}, index=[7]))
    meas = _measurement([3], [1.01])
    with pytest.warns(RuntimeWarning, match="not in measurement"):
        r = bank_voltage_ratio(meas, cached_net, 7)
    assert r == 1.0


@pytest.fixture
def net_with_bank() -> pp.pandapowerNet:
    """110/20 kV feeder with a 5 Mvar capacitor bank at the end bus."""
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=20.0)
    b2 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02)
    pp.create_transformer_from_parameters(
        net, hv_bus=b0, lv_bus=b1, sn_mva=40.0,
        vn_hv_kv=110.0, vn_lv_kv=20.0,
        vkr_percent=0.5, vk_percent=10.0, pfe_kw=30.0, i0_percent=0.1,
    )
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=8.0,
        r_ohm_per_km=0.1, x_ohm_per_km=0.3, c_nf_per_km=10.0, max_i_ka=0.5,
    )
    pp.create_load(net, bus=b2, p_mw=8.0, q_mvar=3.0)
    pp.create_shunt(net, bus=b2, q_mvar=-5.0, p_mw=0.0, step=1)
    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged
    return net


def test_rescaled_column_equals_column_at_measured_voltage(net_with_bank):
    """h(V_0) · (V_meas/V_0)² = h with the per-step injection at V_meas."""
    net = net_with_bank
    sens = JacobianSensitivities(net)
    bank_bus, obs = 2, [1, 2]

    h0, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=bank_bus, observation_bus_indices=obs, q_step_mvar=-1.0,
    )
    h0 = np.asarray(h0, dtype=np.float64)
    v0 = float(sens.net.res_bus.at[bank_bus, "vm_pu"])

    v_meas = 1.04 * v0
    meas = _measurement(list(net.res_bus.index),
                        [v_meas if b == bank_bus else float(net.res_bus.at[b, "vm_pu"])
                         for b in net.res_bus.index])
    r = bank_voltage_ratio(meas, sens.net, bank_bus)

    # Injection per step at the measured voltage, network sensitivity unchanged.
    expected = h0 / v0 ** 2 * v_meas ** 2
    np.testing.assert_allclose(h0 * r, expected, rtol=1e-12)
    assert r == pytest.approx(1.04 ** 2)
