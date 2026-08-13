"""Offline guards on the Ch. 9.1 timescale battery's post-processing.

No PowerFactory, no licence seat: these exercise the pure functions that turn
measured settling times into thesis Table 9.1 and into the design quantities
of eq. (9.1). They exist because the failure modes here are silent -- a table
row that matches nothing is emitted as ``[not run]`` and looks like a
deliberate omission, and a bound taken over the wrong subset of dispatches is
simply a wrong number with no symptom.
"""

from __future__ import annotations

import math

import pytest

from experiments.ch_9_parameter_selection.ch_9_1_timescale_seperation import (
    TABLE_ROWS,
    T_DS_S,
    _synthetic_results,
    build_table,
    derive,
    write_outputs,
)


@pytest.fixture
def results():
    return _synthetic_results()


def test_every_thesis_table_row_is_filled(results):
    tex, missing = build_table(results)
    assert missing == []
    assert not any("[not run]" in line for line in tex)
    # 8 case rows + the two group headers + the midrule between them.
    assert len(tex) == len(TABLE_ROWS) + 3


def test_tap_names_are_matched_literally_not_as_regex(results):
    """``tap_+1`` read as a regex is ``tap`` followed by one-or-more ``_``."""
    tex, _ = build_table(results)
    assert any("11.90" in line for line in tex), "the +2seq row went missing"
    assert any("7.20" in line for line in tex), "the +1 coupler row went missing"


def test_a_lost_case_is_reported_as_a_missing_row(results):
    partial = [r for r in results if "shunt_" not in r["case"]]
    _tex, missing = build_table(partial)
    assert missing == [r"\gls{MSC} switch-in"]


def test_tap_split_is_the_two_step_minus_one_step_difference(results):
    d = derive(results)
    assert d["t_mech"] == pytest.approx(11.90 - 7.20)
    assert d["t_elec"] == pytest.approx(2 * 7.20 - 11.90)
    # The split must reconstruct the measured single-tap settling exactly.
    assert d["t_mech"] + d["t_elec"] == pytest.approx(d["t_tap"])


def test_single_tap_case_alone_is_flagged_as_unseparable(results):
    one_only = [r for r in results if "tap_+2seq" not in r["case"]]
    d = derive(one_only)
    assert d["t_elec"] == 0.0
    assert d["t_mech"] == pytest.approx(7.20)
    assert "NOT separable" in d["t_tap_source"]


def test_bound_covers_every_realisable_dispatch_not_only_the_continuous_rows(results):
    """The MSC switch-in and the machine-trafo tap are dispatches too.

    Regression guard for the 2026-08-06 fix: the bound used to be
    ``max(T_s^cont, T_s^coupler_tap)``, which excluded both.
    """
    d = derive(results)
    assert d["binding"] == pytest.approx(9.10)          # shunt_+1_MSC_...
    assert d["binding_case"].startswith("shunt_+1")
    assert d["binding_kind"] == "MSC switch-in"
    assert d["margin"] == pytest.approx(T_DS_S - 9.10)
    # T_s^cont is still reported separately; the thesis text names it.
    assert d["t_cont"] == pytest.approx(6.80)


def test_two_step_tap_is_an_instrument_and_never_binds(results):
    """It is not a realisable dispatch: one step per iteration is a hard cap."""
    slow_instrument = [
        dict(r, t_settle_s=999.0) if "tap_+2seq" in r["case"] else r
        for r in results
    ]
    d = derive(slow_instrument)
    assert d["binding"] < 999.0


def test_disturbances_never_enter_the_bound(results):
    d = derive(results)
    assert d["binding"] < min(r["t_settle_s"] for r in results
                              if r["disturbance"])
    assert d["worst_disturbance"]["case"] == "outage_G03"


def test_summary_reports_measured_cases_that_have_no_table_row(tmp_path, results):
    d = derive(results)
    write_outputs(tmp_path, results, d, meta=None)
    body = (tmp_path / "timescale_summary.md").read_text(encoding="utf-8")
    assert "Measured but not in Table 9.1" in body
    assert "tap_+1_MT_g0_t0" in body


def test_summary_does_not_pass_the_configured_ratio_off_as_N_inner(tmp_path, results):
    d = derive(results)
    write_outputs(tmp_path, results, d, meta=None)
    body = (tmp_path / "timescale_summary.md").read_text(encoding="utf-8")
    assert "configured" in body
    assert "does not measure it" in body


def test_censored_settling_times_are_marked_in_the_table_and_the_summary(tmp_path,
                                                                        results):
    censored = [dict(r, censored=True, t_settle_s=r["horizon_s"])
                if "avr_vref" in r["case"] else r for r in results]
    d = derive(censored)
    assert d["censored"] == ["avr_vref_+0.02_G09"]
    tex, _ = build_table(censored)
    assert any(r"$>$" in line for line in tex)
    write_outputs(tmp_path, censored, d, meta=None)
    body = (tmp_path / "timescale_summary.md").read_text(encoding="utf-8")
    assert "Censored settling times" in body


def test_empty_battery_does_not_crash_the_derivation():
    d = derive([])
    assert math.isnan(d["binding"])
    assert d["worst_disturbance"] is None
    assert d["binding_kind"] == "no realisable dispatch measured"
