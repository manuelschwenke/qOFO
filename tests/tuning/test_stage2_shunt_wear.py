"""Switched-shunt wear in the Stage-II selection filter (2026-10-22).

The bank step price is a searched weight since the banks became integer inputs
of the TS-MIQP, so bank switching is filtered like tap switching: a per-bank
daily budget and zero hunting, with the taps' event attribution.  Records
written before that date carry no bank wear and must pass unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tuning.stage2_search import Limits, aggregate, check_limits, shunt_summary


def _records(levels_by_t):
    return [SimpleNamespace(time_s=float(t),
                            zone_tso_shunt_states={2: np.array(lv, dtype=float)})
            for t, lv in levels_by_t]


# Bank 0 engages at 180 s and releases at 3600 s; bank 1 engages at 5400 s.
TRACE = [(0, [0, 0]), (180, [1, 0]), (360, [1, 0]), (3600, [0, 0]),
         (5400, [0, 1])]


def test_per_bank_count_and_unexplained_reversal_is_hunting() -> None:
    s = shunt_summary(_records(TRACE))
    assert s["shunt_commits"] == 3
    assert s["shunt_ops_max_bank"] == 2
    assert s["n_shunt_hunting"] == 1
    assert s["n_shunt_event_explained"] == 0


def test_reversal_after_a_new_event_is_not_hunting() -> None:
    s = shunt_summary(_records(TRACE), events_s=[2000.0])
    assert s["n_shunt_hunting"] == 0
    assert s["n_shunt_event_explained"] == 1


def _row(ops_max_bank: int, hunting: int, hours: float = 12.0) -> dict:
    return {"e_v_ts": 0.005, "e_v_ds": 0.006, "e_q": 1.0,
            "shunt_ops_per_h_bank": ops_max_bank / hours,
            "n_shunt_hunting": hunting, "shunt_commits": ops_max_bank}


def test_aggregate_carries_worst_bank_rate_per_day() -> None:
    agg = aggregate({"W1": _row(2, 0), "W2": _row(5, 0)})
    assert agg["has_shunt_wear"] == 1.0
    assert agg["worst_shunt_ops_per_day_bank"] == pytest.approx(10.0)


@pytest.mark.parametrize("budget, ok", [(8.0, False), (10.0, True)])
def test_bank_budget_is_judged_as_a_count(budget: float, ok: bool) -> None:
    agg = aggregate({"W1": _row(5, 0)})
    feasible, why = check_limits(agg, Limits(shunt_ops_per_day=budget),
                                 window_hours=12.0)
    assert feasible is ok
    assert ok or any("bank" in w for w in why)


def test_bank_hunting_rejects() -> None:
    agg = aggregate({"W1": _row(1, 1)})
    feasible, why = check_limits(agg, Limits(shunt_ops_per_day=48.0),
                                 window_hours=12.0)
    assert not feasible
    assert any("bank hunting" in w for w in why)


def test_records_without_bank_wear_pass_unchanged() -> None:
    agg = aggregate({"W1": {"e_v_ts": 0.005, "e_v_ds": 0.006, "e_q": 1.0}})
    assert agg["has_shunt_wear"] == 0.0
    feasible, why = check_limits(agg, Limits(), window_hours=12.0)
    assert feasible, why
