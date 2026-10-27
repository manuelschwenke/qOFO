"""Tests for experiments/ch_11_case2_multisystem/metrics_case2.py on synthetic logs."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ch_11_case2_multisystem import metrics_case2 as mc
from experiments.helpers.records import MultiTSOIterationRecord

DT, TTS = 20.0, 180.0


def _rec(step, **kw):
    r = MultiTSOIterationRecord(step=step, time_s=step * DT)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_u_q_and_reserve_exclude_slack_and_tripped_machines():
    r = _rec(1,
             zone_q_gen={1: np.array([50.0, -20.0, 300.0, 10.0])},
             gen_q_min_mvar={1: np.array([-100.0, -80.0, -900.0, -50.0])},
             gen_q_max_mvar={1: np.array([200.0, 100.0, 900.0, 60.0])},
             zone_gen_in_service={1: np.array([True, True, True, False])},
             zone_gen_is_slack={1: np.array([False, False, True, False])},
             zone_v_rms_err_pu={1: 0.0}, bus_vm_pu={25: 1.03},
             zone_v_min={1: 1.0}, zone_v_max={1: 1.04})
    m = mc.area_metrics([r], 1)
    # machines 0 and 1 only: |50|/200 and |-20|/80
    assert m["u_q"] == pytest.approx((0.25 + 0.25) / 2)
    assert m["r_plus_mean"] == pytest.approx((200 - 50) + (100 + 20))
    assert m["r_minus_mean"] == pytest.approx((50 + 100) + (-20 + 80))


def test_aggregate_is_rms_over_areas():
    per = {1: {"e_v": 0.003}, 2: {"e_v": 0.004}, 3: {"e_v": float("nan")}}
    assert mc.aggregate_e_v(per) == pytest.approx(np.sqrt((9e-6 + 16e-6) / 2))


def test_shunt_switchings_are_counted():
    log = [_rec(k, zone_tso_shunt_states={1: np.array(s)})
           for k, s in enumerate([[0, 0], [1, 0], [1, 0], [0, 1]], start=1)]
    for r in log:
        r.zone_v_rms_err_pu, r.zone_v_min, r.zone_v_max = {1: 0.0}, {1: 1.0}, {1: 1.0}
    assert mc.area_metrics(log, 1)["shunt_ops"] == 3


def test_sts_utilisation_uses_the_capability_in_the_output_direction():
    r = _rec(1, dso_der_q_mvar={"DSO_1": np.array([10.0, -5.0])},
             dso_der_q_min_mvar={"DSO_1": np.array([-20.0, -10.0])},
             dso_der_q_max_mvar={"DSO_1": np.array([40.0, 30.0])},
             dso_group_v_rms_dev_pu={"DSO_1": 0.01},
             dso_group_v_min_pu={"DSO_1": 0.95}, dso_group_v_max_pu={"DSO_1": 1.05})
    m = mc.sts_metrics([r], "DSO_1", 0.92, 1.08)
    assert m["u_q_sts"] == pytest.approx(15.0 / (40.0 + 10.0))
    assert m["margin_min"] == pytest.approx(0.03)
    assert m["f_v_sts"] == pytest.approx(0.01)


def test_interface_end_of_interval_residual_takes_the_step_before_a_dispatch():
    log = []
    for k in range(1, 19):                       # 6 min, dispatches at 180/360 s
        log.append(_rec(k, dso_trafo_q_set_mvar={"DSO_1|trafo_3": 0.0},
                        dso_trafo_q_actual_mvar={"DSO_1|trafo_3": float(k)}))
    out = mc.interface_metrics(log, TTS, DT)["trafo_3"]
    # last steps before a dispatch: t = 160 s (k = 8) and t = 340 s (k = 17)
    assert out["e_sts_end"] == pytest.approx(np.sqrt((8 ** 2 + 17 ** 2) / 2))
    assert out["e_sts_rms"] == pytest.approx(np.sqrt(np.mean(np.arange(1, 19) ** 2)))


def test_recruited_flow_pairs_couplers_across_record_paths():
    a = [_rec(1, dso_trafo_q_actual_mvar={"DSO_1|trafo_3": 10.0}),
         _rec(2, dso_trafo_q_actual_mvar={"DSO_1|trafo_3": 14.0})]
    b = [_rec(1, dso_trafo_q_actual_mvar={"HV1|trafo_3": 4.0}),
         _rec(2, dso_trafo_q_actual_mvar={"HV1|trafo_3": 20.0})]
    assert mc.recruited_flow(a, b) == {"trafo_3": pytest.approx((6.0 + 6.0) / 2)}


def test_epost_uses_the_pre_control_baseline_and_stops_before_the_dispatch():
    # Event at t = 200 s (between dispatches 180 and 360): window 200..340 s.
    e = {9: 0.004, 10: 0.006, 11: 0.007, 12: 0.008, 13: 0.009,
         14: 0.0095, 15: 0.009, 16: 0.0085, 17: 0.008, 18: 0.002}
    log = []
    for k in range(9, 19):
        r = _rec(k, zone_v_rms_err_pu={1: e[k]},
                 dso_trafo_q_actual_mvar={"DSO_1|trafo_3": 0.0})
        log.append(r)
    ev = log[1]                                  # t = 200 s
    ev.pre_control_zone_v_rms_err_pu = {1: 0.0065}
    ev.pre_control_trafo_q_mvar = {"trafo_3": 30.0}   # droop pushed +30
    ev.events_fired = ["gen:trip:2"]
    # the STS pulls the flow back to 0 at every later end-of-step record
    rows = mc.event_metrics(log, TTS, DT)
    assert len(rows) == 1
    row = rows[0]
    assert row["t_d_s"] == 360.0
    assert row["n_window"] == 8                  # 200 .. 340 s
    assert row["epost"] == pytest.approx(0.0095 - 0.0065)  # t = 360 s excluded
    assert row["impact"] == pytest.approx(0.0065 - 0.004)
    # flow back by 30 Mvar over 8 steps of 20 s
    assert row["w_sts_mvar_min"]["trafo_3"] == pytest.approx(30.0 * 8 * 20 / 60)


def test_event_on_a_dispatch_instant_runs_to_the_next_dispatch():
    assert mc._next_dispatch(180.0, TTS) == 360.0
    assert mc._next_dispatch(200.0, TTS) == 360.0
