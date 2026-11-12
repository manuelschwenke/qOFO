"""
Tests for the STS coupler hold after a sudden voltage dip
=========================================================

``sts_coupler_dip_hold_mpu`` / ``sts_coupler_dip_hold_steps`` (2026-11-12):
before each STS step the runner compares the mean measured voltage of the
TS-side buses of an STS's couplers with the previous STS step; a dip larger
than the threshold holds all couplers of that STS through ``hold_oltcs``
(lock itself covered by ``tests/test_sts_coupler_hold.py``).

Author: Manuel Schwenke / Claude Code (2026-11-12)
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from configs.config import MultiTSOConfig
from experiments.runners._multi_tso_helpers import (
    coupler_bus_mean_vm,
    coupler_dip_trigger,
    dip_hold_release_reason,
    measured_vm,
)


# ── configuration ────────────────────────────────────────────────────────────

def test_default_is_off():
    cfg = MultiTSOConfig()
    assert cfg.sts_coupler_dip_hold_mpu == 0.0
    assert cfg.sts_coupler_dip_hold_steps == 0


def test_valid_setting_accepted():
    dataclasses.replace(MultiTSOConfig(), sts_coupler_dip_hold_mpu=20.0,
                        sts_coupler_dip_hold_steps=9).validate_integrator_mode()


@pytest.mark.parametrize("mpu, steps, match", [
    (-1.0, 9, "sts_coupler_dip_hold_mpu"),
    (20.0, -1, "sts_coupler_dip_hold_steps"),
    (20.0, 2.5, "sts_coupler_dip_hold_steps"),
    (20.0, 0, "needs sts_coupler_dip_hold_steps"),
])
def test_invalid_setting_rejected(mpu, steps, match):
    cfg = dataclasses.replace(MultiTSOConfig(), sts_coupler_dip_hold_mpu=mpu,
                              sts_coupler_dip_hold_steps=steps, shunt_dispatch="off")
    with pytest.raises(ValueError, match=match):
        cfg.validate_integrator_mode()


# ── trigger ──────────────────────────────────────────────────────────────────

def test_trigger_fires_on_dip_only():
    assert coupler_dip_trigger(1.030, 0.970, 20.0) == pytest.approx(-60.0)
    assert coupler_dip_trigger(1.030, 1.015, 20.0) is None      # 15 mpu < 20
    assert coupler_dip_trigger(1.030, 1.0105, 20.0) is None     # 19.5 mpu: below
    assert coupler_dip_trigger(1.030, 1.0095, 20.0) is not None  # 20.5 mpu: fires
    assert coupler_dip_trigger(0.970, 1.030, 20.0) is None      # rise never fires


def test_trigger_needs_history_and_threshold():
    assert coupler_dip_trigger(None, 0.9, 20.0) is None         # first STS step
    assert coupler_dip_trigger(1.03, 0.9, 0.0) is None          # off


# ── measurement ──────────────────────────────────────────────────────────────

def test_coupler_bus_mean_reads_the_right_buses():
    meas = SimpleNamespace(bus_indices=np.array([0, 3, 4, 6, 7, 40]),
                           voltage_magnitudes_pu=np.array([1.0, 1.01, 1.02, 1.03, 1.04, 1.05]))
    assert coupler_bus_mean_vm(meas, [6, 7, 4]) == pytest.approx((1.03 + 1.04 + 1.02) / 3)


def test_measured_vm_keeps_the_requested_order():
    meas = SimpleNamespace(bus_indices=np.array([0, 3, 4, 6]),
                           voltage_magnitudes_pu=np.array([1.0, 1.01, 1.02, 1.03]))
    assert measured_vm(meas, [6, 0]) == pytest.approx([1.03, 1.0])


# ── safeguards (early release) ───────────────────────────────────────────────

def _release(v, q_now, margin=0.0, share=0.05):
    q_min, q_max = np.array([-50.0, -50.0]), np.array([50.0, 50.0])   # range 200 Mvar
    return dip_hold_release_reason(np.asarray(v), 0.92, 1.08, margin,
                                   np.asarray(q_now), q_min, q_max, share)


def test_hold_kept_inside_band_with_headroom():
    assert _release([0.93, 1.00], [10.0, 20.0]) is None          # R2-like: 0.925 > 0.92


def test_release_when_band_left_or_margin_reached():
    assert _release([0.919, 1.00], [0.0, 0.0]) == "v_band"
    assert _release([0.95, 1.081], [0.0, 0.0]) == "v_band"
    assert _release([0.925, 1.00], [0.0, 0.0], margin=0.01) == "v_band"
    assert _release([0.935, 1.00], [0.0, 0.0], margin=0.01) is None


def test_release_when_upward_der_headroom_nearly_exhausted():
    assert _release([1.0], [46.0, 45.0]) == "der_headroom"        # headroom 9 < 10 Mvar
    assert _release([1.0], [44.0, 45.0]) is None                  # headroom 11
    assert _release([1.0], [60.0, 30.0]) is None                  # overshoot counts 0, other 20


def test_zero_der_range_skips_headroom_check():
    z = np.zeros(2)
    assert dip_hold_release_reason(np.array([1.0]), 0.92, 1.08, 0.0, z, z, z, 0.05) is None


def test_release_settings_validated():
    for kw in ({"sts_coupler_dip_hold_release_v_margin_pu": -0.01},
               {"sts_coupler_dip_hold_release_headroom": 1.5}):
        cfg = dataclasses.replace(MultiTSOConfig(), shunt_dispatch="off", **kw)
        with pytest.raises(ValueError, match="sts_coupler_dip_hold_release"):
            cfg.validate_integrator_mode()


def test_coupler_bus_mean_missing_bus_raises():
    meas = SimpleNamespace(bus_indices=np.array([0, 3, 4]),
                           voltage_magnitudes_pu=np.array([1.0, 1.01, 1.02]))
    with pytest.raises(KeyError):
        coupler_bus_mean_vm(meas, [4, 5])
    with pytest.raises(KeyError):
        coupler_bus_mean_vm(meas, [9])
