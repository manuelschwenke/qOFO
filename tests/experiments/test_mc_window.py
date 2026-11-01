"""The comparison window of the Case-2 MC (rules of 2026-11-01, 2026-10-30,
2026-10-07), the outcome criteria of 2026-11-01 and the failure report."""
import json

import pandas as pd

from experiments.ch_11_case2_multisystem.mc import run_mc as M

V = M.ALL_VARIANTS


def _all(seed, outcome):
    return {(seed, v): outcome for v in V}


# -- rule 2026-11-01 (default): any failure replaces the episode -------------

def test_default_rule_is_the_amendment_of_2026_11_01():
    assert M.WINDOW_RULE == "2026-11-01"


def test_an_episode_any_variant_failed_on_is_replaced_by_the_next():
    valid = [1, 2, 3, 4, 5]
    out = {**_all(1, "ok"), **_all(2, "ok"), (2, "S1"): "fail_abort",
           **_all(3, "ok"), **_all(4, "ok")}
    assert M.comparison_window(valid, out, 3) == ([1, 3, 4], [2])


def test_an_episode_all_six_failed_on_is_replaced_too():
    valid = [1, 2, 3]
    out = {**_all(1, "fail_pf"), **_all(2, "ok"), **_all(3, "ok")}
    assert M.comparison_window(valid, out, 2) == ([2, 3], [1])


def test_an_episode_not_yet_run_by_every_variant_counts_as_usable():
    valid = [1, 2, 3]
    out = {**_all(1, "ok"), (1, "S1"): "fail_abort"}
    del out[(1, "M")]                       # one failure, M still missing
    assert M.comparison_window(valid, out, 2) == ([1, 2], [])


def test_the_window_is_short_when_the_valid_seeds_run_out():
    valid = [1, 2]
    out = {**_all(1, "ok"), (1, "L2"): "fail_osc"}
    assert M.comparison_window(valid, out, 2) == ([2], [1])


# -- earlier rules -------------------------------------------------------------

def test_declared_rule_takes_the_first_n_valid_seeds():
    valid = [1, 2, 3, 4, 5]
    out = {**_all(2, "fail_abort")}
    assert M.comparison_window(valid, out, 3, "2026-10-07") == ([1, 2, 3], [])


def test_rule_2026_10_30_replaces_only_an_episode_all_six_failed_on():
    valid = [1, 2, 3, 4, 5, 6]
    out = {**_all(1, "ok"), **_all(2, "fail_abort"), **_all(3, "ok"),
           **_all(4, "fail_pf"), **_all(5, "ok")}
    assert M.comparison_window(valid, out, 3, "2026-10-30") == ([1, 3, 5], [2, 4])


def test_rule_2026_10_30_never_replaces_failures_of_single_variants():
    valid = [1, 2, 3]
    out = {**_all(1, "fail_abort"), (1, "M"): "ok", **_all(2, "ok")}
    assert M.comparison_window(valid, out, 2, "2026-10-30") == ([1, 2], [])


def test_an_unknown_rule_is_refused():
    try:
        M.comparison_window([1], {}, 1, "2026-12-24")
    except ValueError:
        return
    raise AssertionError("unknown rule accepted")


# -- voltage at the horizon end: no failure since 2026-11-01 ------------------

def test_fail_v_end_of_older_runs_counts_as_ok_without_tap_hunting():
    diag = {"tap_reversals_end_ts": 1, "tap_reversals_end_sts": 0}
    assert M.effective_outcome("fail_v_end", diag) == "ok"


def test_fail_v_end_of_older_runs_counts_as_fail_osc_with_tap_hunting():
    diag = {"tap_reversals_end_ts": 0, "tap_reversals_end_sts": M.OSC_REVERSALS}
    assert M.effective_outcome("fail_v_end", diag) == "fail_osc"


def test_other_outcomes_are_unchanged():
    for o in ("ok", "fail_pf", "fail_abort", "fail_osc", "fail_unreachable"):
        assert M.effective_outcome(o, {}) == o


def test_the_rule_reads_older_fail_v_end_records_as_ok(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for v in V:
        rec = {"seed": 1, "variant": v, "outcome": "ok", "diag": {}}
        if v == "L2":
            rec.update(outcome="fail_v_end", diag={"tap_reversals_end_ts": 0})
        (runs / f"1_{v}.json").write_text(json.dumps(rec), encoding="utf-8")
    assert M._outcomes(runs)[(1, "L2")] == "ok"
    assert M._outcomes(runs, "2026-10-30")[(1, "L2")] == "fail_v_end"
    assert M.comparison_window([1], M._outcomes(runs), 1) == ([1], [])


# -- failure report --------------------------------------------------------------

def _pre(seed):
    return {"seed": seed, "valid": True, "start_time": "2016-03-01T08:00:00",
            "class": "single",
            "events": [{"minute": 60, "time_s": 3620.0, "element_type": "load",
                        "element_index": -1, "bus": 7, "q_mvar": 300.0,
                        "action": "connect"}]}


def test_failure_row_times_and_classifies_a_stopped_power_flow():
    rec = {"seed": 5, "variant": "S1", "outcome": "fail_abort", "n_records": 200,
           "dt_s": 20.0, "t_ts_s": 180.0,
           "diag": {"exception": "LoadflowNotConverged: Power Flow nr did not converge"}}
    row = M.failure_row(rec, _pre(5))
    assert row["cause"] == "power flow not converged, run stopped"
    assert abs(row["t_fail_min"] - 67.0) < 1e-9             # step 201 x 20 s
    assert abs(row["after_event_min"] - (4020.0 - 3620.0) / 60.0) < 1e-9
    assert row["at_event"] is False and row["at_dispatch"] is False


def test_failure_report_lists_replaced_episodes_and_end_voltages(tmp_path):
    d = M._dirs(tmp_path)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    for s in (1, 2, 3):
        (d["pre"] / f"{s}.json").write_text(json.dumps(_pre(s)), encoding="utf-8")
        for v in V:
            rec = {"seed": s, "variant": v, "outcome": "ok", "n_records": 540,
                   "class": "single", "start_time": "2016-03-01T08:00:00",
                   "diag": {"v_end_outside_hard_band": (s, v) == (3, "O2")}}
            if (s, v) == (2, "S1"):
                rec.update(outcome="fail_abort", n_records=200,
                           diag={"exception": "LoadflowNotConverged: x"})
            (d["runs"] / f"{s}_{v}.json").write_text(json.dumps(rec), encoding="utf-8")
    window, skipped = M.comparison_window([1, 2, 3], M._outcomes(d["runs"]), 2)
    assert (window, skipped) == ([1, 3], [2])
    M.failure_report(tmp_path, window, skipped, M.WINDOW_RULE)
    fl = pd.read_csv(tmp_path / "failures_per_run.csv")
    assert list(fl[["seed", "variant", "set"]].itertuples(index=False, name=None)) == \
        [(2, "S1", "replaced")]
    text = (tmp_path / "failures.md").read_text(encoding="utf-8")
    assert "| 2 | single |" in text and "PF @67 min" in text
    assert "| O2 | 1 | [3] |" in text
    assert "| S1 | 3 | 1 |" in text
