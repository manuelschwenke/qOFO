"""Tests for experiments/ch_11_case2_multisystem/step/metrics_step.py."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ch_11_case2_multisystem.step import metrics_step as ms


def _rec(t, e, vgen=1.03, tap=0):
    return SimpleNamespace(
        time_s=t, zone_v_rms_err_pu={1: e, 2: e, 3: e},
        zone_v_gen={1: [vgen], 2: [1.03], 3: [1.03]},
        zone_oltc_taps={1: [tap], 2: [0], 3: [0]},
        zone_tso_shunt_states={1: [0], 2: [0], 3: [0]},
        dso_trafo_tap_pos={}, dso_group_v_max_pu={"DSO_1": 1.081},
        dso_group_v_min_pu={"DSO_1": 1.0}, dso_group_v_rms_dev_pu={"DSO_1": 0.01},
        dso_trafo_q_actual_mvar={"DSO_1|trafo_0": -10.0},
        dso_trafo_q_set_mvar={"DSO_1|trafo_0": -12.0})


def test_rho_fit_recovers_a_geometric_decay():
    d = 5e-3 * 0.4 ** np.arange(6)
    assert ms.rho_fit(d, 1e-5) == pytest.approx(0.4, rel=1e-6)
    assert np.isnan(ms.rho_fit(np.array([1e-3]), 1e-5))


def test_recovery_counts_dispatches_and_settling():
    # error 10 mpu at the event, halves at each 180 s dispatch, else constant
    log, e = [], 10e-3
    for k in range(1, 201):
        t = 1800.0 + 20.0 * k
        if t % 180.0 == 0.0:
            e = 2e-3 + (e - 2e-3) * 0.5
        log.append(_rec(t, e))
    r = ms.recovery(log, 1820.0, 1820.0 + 200 * 20.0, tol=0.5e-3)["1"]
    assert r["final"] == pytest.approx(2e-3, abs=1e-4)
    assert r["rho_hat"] == pytest.approx(0.5, rel=1e-3)
    assert r["n_rec"] == 3          # deviations 4, 2, 1 mpu, then 0.5 = tol


def test_window_effort_sawtooth_and_sts_excess():
    log = [_rec(20.0 * k, (k % 9) * 1e-4, vgen=1.03 + 0.001 * k, tap=k // 30)
           for k in range(1, 91)]
    w = ms.window(log, 0.0, 1800.0, baseline={1: 0.0, 2: 0.0, 3: 0.0})
    a1 = w["areas"]["1"]
    assert a1["avr_travel_pu"] == pytest.approx(0.001 * (w["n"] - 1), rel=1e-6)
    assert a1["ts_taps"] == 2
    assert a1["saw_pp_max"] == pytest.approx(8e-4, rel=1e-6)
    assert w["sts"]["DSO_1"]["n_above_tol"] == w["n"]      # 1 mpu above 1.08
    assert w["e_sts_rms_mvar"] == pytest.approx(2.0)
