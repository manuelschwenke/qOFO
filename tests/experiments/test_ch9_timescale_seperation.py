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
    # one line per table row + the two group headers + the midrule.
    assert len(tex) == len(TABLE_ROWS) + 3


def test_tap_names_are_matched_literally_not_as_regex(results):
    """``tap_+1`` read as a regex is ``tap`` followed by one-or-more ``_``.

    Every matcher that carries a ``+`` is checked, because a regex read would
    drop the row silently and the thesis would show a deliberate-looking gap.
    Since 2026-08-19 that is four matchers, not one: the two ``tap_+1_*``
    classes and the two AVR magnitudes, whose ``+0.02`` / ``+0.001`` prefixes
    are what keep the four AVR rows apart.
    """
    tex, _ = build_table(results)
    assert any("7.20" in line for line in tex), "the +1 coupler row went missing"
    assert any("8.30" in line for line in tex), "the +1 MT row went missing"
    assert any("6.80" in line for line in tex), "the +0.02 G09 row went missing"
    assert any("1.90" in line for line in tex), "the +0.001 G09 row went missing"


def test_two_step_instruments_are_measured_but_not_tabulated(results):
    """The thesis dropped the two-step row; the battery still runs the case.

    It is the only way to separate ``T_mech`` from ``T_elec``, so it must
    survive in the results while staying out of the table.
    """
    tex, _ = build_table(results)
    assert not any("11.90" in line or "11.70" in line for line in tex)
    assert {r["case"] for r in results if not r["tabulated"]} == {
        "tap_+2seq_NC3W_DSO_1_t0", "tap_+2seq_MT_g0_t0"}


def test_avr_rows_are_split_by_machine_and_by_magnitude(results):
    """Four AVR rows: {G09, G10} x {0.02, 0.001} pu.

    The 0.001 pu rows are the magnitude the tuned TS-OFO actually issues; a
    single lumped ``avr_vref`` matcher would collapse all four onto whichever
    case the battery emitted first.
    """
    tex, _ = build_table(results)
    avr = [line for line in tex if "voltage-reference" in line]
    assert len(avr) == 4, avr
    assert sum("0.02" in line for line in avr) == 2
    assert sum("0.001" in line for line in avr) == 2


def test_machine_transformer_tap_has_its_own_table_row(results):
    """It is a dispatch, it enters the bound, and it now has a row.

    Regression guard for the divergence found 2026-08-19: the emitter had no
    MT row while the thesis table did, so that row was filled by hand.
    """
    tex, _ = build_table(results)
    assert any("machine transformer" in line and "8.30" in line
               for line in tex)
    assert any("coupling transformer" in line and "7.20" in line
               for line in tex)


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


def test_each_tap_class_is_split_against_its_own_two_step_case(results):
    """A coupler one-step must never be paired with an MT two-step.

    Both classes carry a ``tap_+2seq_*`` case since 2026-08-19. An unqualified
    ``"tap_+2seq" in case`` match would pair whichever the catalogue emitted
    first, producing a ``T_mech`` belonging to neither transformer -- and no
    symptom, because the arithmetic still closes.
    """
    d = derive(results)
    cpl = d["tap_splits"]["coupler"]
    mt = d["tap_splits"]["machine_trafo"]
    assert cpl["two_case"] == "tap_+2seq_NC3W_DSO_1_t0"
    assert mt["two_case"] == "tap_+2seq_MT_g0_t0"
    assert cpl["t_mech"] == pytest.approx(11.90 - 7.20)
    assert mt["t_mech"] == pytest.approx(11.70 - 8.30)
    # The classes genuinely differ, so a leak between them is detectable.
    assert cpl["t_mech"] != pytest.approx(mt["t_mech"])
    for sp in (cpl, mt):
        assert sp["t_mech"] + sp["t_elec"] == pytest.approx(sp["t_tap"])


def test_summary_reports_the_split_for_both_tap_classes(tmp_path, results):
    d = derive(results)
    write_outputs(tmp_path, results, d, meta=None)
    body = (tmp_path / "timescale_summary.md").read_text(encoding="utf-8")
    assert "Tap split by class" in body
    assert "coupler" in body and "machine_trafo" in body


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
    assert "tap_+2seq_MT_g0_t0" in body


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
    assert d["censored"] == ["avr_vref_+0.02_G09", "avr_vref_+0.02_G10",
                             "avr_vref_+0.001_G09", "avr_vref_+0.001_G10"]
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
