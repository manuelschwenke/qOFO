"""
Per-bus DS voltage deviation (added 2026-10-16, protocol D1).

Checks the recorder (``dso_group_v_rms_dev_pu``), the DS counterpart of the TS
spatial-RMS metric, and that ``_v_quality`` switches to it only for logs that
carry it, so an old pickle re-scored today keeps its old ``v_rms_ds``.

The recorder is exercised with a minimal fake net and controller: no power
flow, only the frames and attributes the helper reads.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.helpers.comparison_metrics import (  # noqa: E402
    has_ds_rms_dev,
    voltage_rms_err_all_ds,
)
from experiments.helpers.records import MultiTSOIterationRecord  # noqa: E402
from experiments.runners._multi_tso_helpers import (  # noqa: E402
    _record_dso_group_and_transformer_data,
)
from tuning.metrics import _v_quality  # noqa: E402

V_SET = 1.03


def _fake(vm, v_ref):
    """Fake net + one DSO controller observing three buses."""
    buses = [10, 11, 12]
    net = SimpleNamespace(
        res_bus=pd.DataFrame({"vm_pu": vm}, index=buses),
        res_sgen=pd.DataFrame({"q_mvar": [1.0, 2.0], "p_mw": [5.0, 5.0]}, index=[0, 1]),
        res_trafo3w=pd.DataFrame(columns=["q_hv_mvar", "p_hv_mw"]),
        trafo3w=pd.DataFrame(columns=["tap_pos"]),
    )
    cfg = SimpleNamespace(
        interface_trafo_indices=[], der_indices=[0, 1],
        voltage_bus_indices=buses, v_setpoints_pu=v_ref,
    )
    ctrl = SimpleNamespace(
        config=cfg,
        actuator_bounds=SimpleNamespace(
            compute_der_q_bounds=lambda p: (np.full(len(p), -3.0), np.full(len(p), 3.0))),
        _last_capability_q_iface_min_mvar=None,
        _last_capability_q_iface_max_mvar=None,
    )
    return net, ctrl


def _record(vm, v_ref):
    net, ctrl = _fake(vm, v_ref)
    rec = MultiTSOIterationRecord(step=0, time_s=0.0, tso_active=True, dso_active=True)
    _record_dso_group_and_transformer_data(
        rec=rec, net=net, dso_ids=["DSO_1"], dsocontrollers={"DSO_1": ctrl},
        dso_group_map={"DSO_1": "DSO_1"}, last_dso_q_set_mvar={"DSO_1": None},
        hv_info_map={},
    )
    return rec


def test_recorder_writes_per_bus_rms_not_mean():
    # Mean exactly on the setpoint, spread +/- 0.03: the mean-based figure is 0,
    # the per-bus RMS is not.
    rec = _record([1.00, 1.03, 1.06], np.full(3, V_SET))
    expected = np.sqrt((0.03 ** 2 + 0.0 + 0.03 ** 2) / 3.0)
    assert rec.dso_group_v_rms_dev_pu["DSO_1"] == pytest.approx(expected, rel=1e-12)
    assert rec.dso_group_v_mean_pu["DSO_1"] == pytest.approx(V_SET, abs=1e-12)


def test_recorder_uses_per_bus_references():
    ref = np.array([1.00, 1.03, 1.06])
    rec = _record([1.00, 1.03, 1.06], ref)
    assert rec.dso_group_v_rms_dev_pu["DSO_1"] == pytest.approx(0.0, abs=1e-15)


def test_recorder_skips_without_references():
    rec = _record([1.00, 1.03, 1.06], None)
    assert "DSO_1" not in rec.dso_group_v_rms_dev_pu
    assert "DSO_1" in rec.dso_group_v_mean_pu   # the old fields are unaffected


def test_recorder_rejects_reference_length_mismatch():
    with pytest.raises(ValueError, match="v_setpoints_pu"):
        _record([1.00, 1.03, 1.06], np.full(2, V_SET))


def _log(with_rms: bool, n: int = 4):
    recs = []
    for i in range(n):
        r = MultiTSOIterationRecord(step=i, time_s=20.0 * i, tso_active=True, dso_active=True)
        r.zone_v_min = {1: 1.02}; r.zone_v_mean = {1: 1.03}; r.zone_v_max = {1: 1.04}
        r.zone_v_rms_err_pu = {1: 0.01}
        r.dso_group_v_min_pu = {"A": 1.00, "B": 1.02}
        r.dso_group_v_mean_pu = {"A": 1.03, "B": 1.04}
        r.dso_group_v_max_pu = {"A": 1.06, "B": 1.06}
        if with_rms:
            r.dso_group_v_rms_dev_pu = {"A": 0.03, "B": 0.04}
        recs.append(r)
    return recs


def test_ds_aggregate_is_quadratic_mean_over_groups():
    rms = voltage_rms_err_all_ds(_log(True))["rms_err_pu"]
    assert np.allclose(rms, np.sqrt((0.03 ** 2 + 0.04 ** 2) / 2.0))


def test_v_quality_uses_per_bus_when_recorded():
    out = _v_quality(_log(True), V_SET, 0.02)
    assert out["v_rms_ds"] == pytest.approx(np.sqrt((0.03 ** 2 + 0.04 ** 2) / 2.0))


def test_v_quality_keeps_old_value_for_old_logs():
    old = _log(False)
    assert not has_ds_rms_dev(old)
    out = _v_quality(old, V_SET, 0.02)
    # |mean over groups of the group means - v_set| = |1.035 - 1.03|
    assert out["v_rms_ds"] == pytest.approx(0.005, abs=1e-12)
