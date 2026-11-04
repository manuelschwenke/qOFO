"""Procedure v4 (ch_9_14): the lambda without the banks, the hunting dwell, the
screen arguments and the configuration of V3."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V2
from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3
from experiments.ch_09_parameter_selection import ch_9_14_procedure_v4 as V4
from tuning import stage1_screen as s1
from tuning.stage2_search import shunt_summary


@pytest.fixture
def v4():
    prev = V4.configure(["sts_coupler_hold_after_bank=1"])
    run = P.RUN
    P.RUN = V4.ROOT / "results" / "_test_ch9_14"     # P.rel needs a path under ROOT; not created
    try:
        yield
    finally:
        P.RUN = run
        V4.restore(prev)


def _layer():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(5, 4))
    s = h.T @ h
    return s1.LayerCurvature(key="TS_2", layer="TS", area="2", S_V=s, S_Q=np.zeros_like(s),
                             colsum_V=np.abs(h.sum(axis=0)), maxabs_V=np.abs(h).max(axis=0),
                             classes=("der", "pcc", "shunt", "shunt"), n_v=5, n_q=0,
                             exclusive_groups=((2, 3),))


def test_lambda_without_banks_is_the_limit_of_an_infinite_bank_price():
    lc = _layer()
    g_w = np.array([0.5, 0.8, 0.3, 0.4])
    full = s1.lambda_max_B(lc, 1.0, 0.0, g_w)
    excl = s1.lambda_max_B(lc, 1.0, 0.0, g_w, exclude_classes=("shunt",))
    limit = s1.lambda_max_B(lc, 1.0, 0.0, np.array([0.5, 0.8, 1e12, 1e12]))
    assert excl == pytest.approx(limit, rel=1e-9)
    assert excl <= full + 1e-12
    only = s1.lambda_max_B(lc, 1.0, 0.0, g_w, exclude_classes=("der", "pcc", "shunt"))
    assert only == 0.0


def _records(levels):
    return [SimpleNamespace(time_s=float(t), zone_tso_shunt_states={2: np.array([v])})
            for t, v in levels]


def test_bank_hunting_dwell_is_a_parameter():
    # commit at 1 h, release at 2.5 h: a reversal with a 1.5 h dwell and no event
    recs = _records([(0, 0), (3600, 1), (9000, 0), (12000, 0)])
    assert shunt_summary(recs)["n_shunt_hunting"] == 1                  # 2 h default
    assert shunt_summary(recs, dwell_s=3600.0)["n_shunt_hunting"] == 0  # v4


def test_v4_constants_after_configure(v4):
    assert V3.TOLERANCE == 0 and V3.N_PERIODS == 3
    assert V2.VERIFY_P95 == 25.0
    assert V3.RESULT == V4.RESULT and V2.passes_screen is V4.passes_screen
    win = {"key": "k", "aggregate": {"e_v_ts": 0.0099}}
    rep = {"_engagement": {"k": {"sts_all": True, "bank": True}}, "_level": "sts+bank"}
    go = V3.go_criteria(win, rep)
    assert go["go"] and go["limit_e_v_ts"] == pytest.approx(0.01)
    win["aggregate"]["e_v_ts"] = 0.0101
    assert not V3.go_criteria(win, rep)["go"]


def test_v4_screen_arguments(v4):
    out, cj = Path("results/x"), Path("results/x/c.json")
    a = P.screen_argv(out, cj, 4)
    assert a[a.index("--windows") + 1] == "W1,W3,W4,W6"
    assert a[a.index("--abort-tap-reversals") + 1] == "1"
    assert a[a.index("--hunt-dwell-s") + 1] == "3600"
    assert a[a.index("--lambda-exclude") + 1] == "shunt"
    assert a[a.index("--lambda-extra-ops") + 1].endswith("operating_points_lambda")
    assert a[a.index("--config-set") + 1] == "sts_coupler_hold_after_bank=1"
    assert "--startup-exempt-s" in a
    w5 = P.screen_argv(out, cj, 2, windows="W5")
    assert "--hunt-dwell-s" in w5 and "--abort-tap-reversals" not in w5


def test_restore_leaves_v3_as_it_was():
    before = (V3.TOLERANCE, V3.N_PERIODS, V2.VERIFY_P95, V2.passes_screen, P.screen_argv)
    prev = V4.configure()
    V4.restore(prev)
    assert (V3.TOLERANCE, V3.N_PERIODS, V2.VERIFY_P95, V2.passes_screen,
            P.screen_argv) == before
