"""Tests for experiments/helpers/profile_ramps.py (STEP campaign of ch:case3)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pandapower as pp
import pytest

from core.profiles import apply_profiles, snapshot_base_values
from experiments.helpers.profile_ramps import (apply_profile_ramps,
                                               hold_profile_frame, ramp_shape)

T0 = datetime(2016, 3, 1, 12, 0)


def _frame(n=40, dt_s=20):
    idx = pd.date_range(T0, periods=n, freq=f"{dt_s}s")
    return pd.DataFrame({"LP": np.linspace(0.5, 1.0, n),
                         "LQ": np.linspace(0.2, 0.4, n)}, index=idx)


def _net():
    net = pp.create_empty_network()
    b0, b1, b2 = (pp.create_bus(net, vn_kv=345.0) for _ in range(3))
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(net, b0, b1, 10, 0.01, 0.1, 10, 1.0)
    pp.create_line_from_parameters(net, b0, b2, 10, 0.01, 0.1, 10, 1.0)
    for b in (b1, b2):            # one TS load, one "STS" load, same columns
        pp.create_load(net, b, p_mw=100.0, q_mvar=40.0)
    net.load["profile_p"] = "LP"
    net.load["profile_q"] = "LQ"
    snapshot_base_values(net)
    return net, b1, b2


def test_hold_freezes_every_row_at_the_start():
    f = hold_profile_frame(_frame(), T0)
    assert np.allclose(f["LP"], 0.5) and np.allclose(f["LQ"], 0.2)


def test_ramp_shape_up_hold_down():
    idx = pd.date_range(T0, periods=10, freq="60s")
    s = ramp_shape(idx, T0, dict(t0_s=60, ramp_s=120, hold_s=120, down=True))
    assert np.allclose(s, [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0])
    s = ramp_shape(idx, T0, dict(t0_s=60, ramp_s=120, down=False))
    assert s[-1] == 1.0 and s[1] == 0.0


def test_bus_ramp_adds_a_load_that_follows_the_shape():
    net, b1, _ = _net()
    f = hold_profile_frame(_frame(), T0)
    f = apply_profile_ramps(net, f, T0, [dict(kind="bus", bus=b1, q_mvar=300.0,
                                              t0_s=100, ramp_s=200, hold_s=0)])
    row = net.load.index[net.load["name"] == "STEP_RAMP_0"][0]
    apply_profiles(net, f, T0 + pd.Timedelta(seconds=200))
    assert net.load.at[row, "q_mvar"] == pytest.approx(150.0)
    assert net.load.at[row, "p_mw"] == 0.0
    apply_profiles(net, f, T0 + pd.Timedelta(seconds=400))
    assert net.load.at[row, "q_mvar"] == pytest.approx(300.0)


def test_ts_ramp_leaves_the_excluded_buses_alone():
    net, b1, b2 = _net()
    f = hold_profile_frame(_frame(), T0)
    f = apply_profile_ramps(net, f, T0, [dict(kind="ts_loads", rel=0.10,
                                              exclude_buses=[b2], t0_s=0,
                                              ramp_s=200)])
    apply_profiles(net, f, T0 + pd.Timedelta(seconds=400))
    ts, sts = (net.load.index[net.load.bus == b][0] for b in (b1, b2))
    assert net.load.at[ts, "p_mw"] == pytest.approx(100.0 * 0.5 * 1.1)
    assert net.load.at[ts, "q_mvar"] == pytest.approx(40.0 * 0.2 * 1.1)
    assert net.load.at[sts, "p_mw"] == pytest.approx(100.0 * 0.5)
