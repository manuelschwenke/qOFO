from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from export.dynamic_snapshot import load_snapshot
from experiments.ch_9_parameter_selection import ch_9_1_actuator_location_sweep as sweep


SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "export"
    / "snapshots"
    / "full_t0_20160105-0800.json"
)


def _candidate(
    name: str,
    score: float | None,
    *,
    group: str = "G1",
    direction: int = 1,
    mode: str = "continuous",
) -> sweep.Candidate:
    return sweep.Candidate(
        name=name,
        actuator_class="q",
        domain="TSO",
        group=group,
        mode=mode,
        kind="tap" if mode == "discrete" else "param",
        table="shunt" if mode == "discrete" else "sgen",
        index=0,
        pf_name=name,
        direction=direction,
        variable="qset",
        delta=float(direction),
        physical_delta=float(direction),
        unit="pu",
        qss_score=score,
    )


def test_real_catalogue_is_actuator_only_and_discrete_exhaustive() -> None:
    doc = load_snapshot(SNAPSHOT)
    cases = sweep.build_candidates(doc, oltc_delay_s=5.0)

    discrete = [case for case in cases if case.mode == "discrete"]
    continuous = [case for case in cases if case.mode == "continuous"]
    assert len(discrete) == 46
    assert len(continuous) == 66
    assert {case.kind for case in cases} == {"param", "tap"}
    assert not any(case.disturbance for case in cases)

    assert sum(case.actuator_class == "coupler_oltc" for case in discrete) == 24
    assert sum(case.actuator_class == "tso_oltc" for case in discrete) == 14
    assert sum(case.actuator_class == "shunt" for case in discrete) == 8
    assert all(case.event_offset_s == 5.0 for case in discrete
               if case.actuator_class.endswith("oltc"))
    assert all(case.event_offset_s == 0.0 for case in discrete
               if case.actuator_class == "shunt")
    assert not any("MT_g9_t12" in case.name for case in cases)


def test_output_vector_has_no_currents_and_covers_q_boundaries() -> None:
    specs = sweep.build_output_specs(load_snapshot(SNAPSHOT))
    assert sum(spec.kind == "voltage" for spec in specs) == 70
    assert sum(spec.kind == "dso_interface_q" for spec in specs) == 12
    assert sum(spec.kind == "tso_boundary_q" for spec in specs) == 5
    assert not any("m:I" in spec.variable for spec in specs)
    assert not any("current" in spec.kind for spec in specs)
    assert sweep.PREFLIGHT_V_DRIFT_PU == sweep.BAND_VOLTAGE_PU
    assert sweep.PREFLIGHT_Q_DRIFT_MVAR == sweep.BAND_Q_MVAR


def test_continuous_selection_is_deterministic_and_retains_audit_and_failures() -> None:
    cases = [
        _candidate("disc", None, mode="discrete"),
        _candidate("a", 10.0, group="G1"),
        _candidate("b", 8.0, group="G1"),
        _candidate("c", 7.0, group="G2"),
        _candidate("d", 2.0, group="G2"),
        _candidate("failed", math.inf, group="G3"),
    ]
    selected = sweep.select_candidates(
        cases,
        top_global=1,
        top_per_group=1,
        median_audit=True,
        all_continuous=False,
    )
    names = {case.name for case in selected}
    assert {"disc", "a", "c", "failed"} <= names
    assert any("median QSS audit" in case.selection_reasons for case in selected)

    selected_again = sweep.select_candidates(
        cases,
        top_global=1,
        top_per_group=1,
        median_audit=True,
        all_continuous=False,
    )
    assert [case.name for case in selected] == [case.name for case in selected_again]


def test_fixed_band_metric_settles_and_detects_subband_response() -> None:
    t = np.arange(0.0, 20.0001, 0.05)
    y = np.zeros_like(t)
    after = t >= 5.0
    y[after] = 1.0 + 0.1 * np.exp(-(t[after] - 5.0))
    metric = sweep.fixed_band_metrics(t, y, 5.0, 0.01)
    assert 2.0 < metric["t_settle_s"] < 3.0
    assert metric["censored"] is False
    assert metric["subband"] is False

    small = np.zeros_like(t)
    small[after] = 5.0e-4
    subband = sweep.fixed_band_metrics(t, small, 5.0, 1.0e-3)
    assert subband["subband"] is True
    assert subband["t_settle_s"] == 0.0


def test_fixed_band_metric_marks_unstable_tail_censored() -> None:
    t = np.arange(0.0, 20.0001, 0.05)
    y = np.zeros_like(t)
    after = t >= 5.0
    y[after] = 0.02 * (t[after] - 5.0)
    metric = sweep.fixed_band_metrics(t, y, 5.0, 1.0e-3)
    assert metric["censored"] is True
    assert metric["tail_drift"] > metric["tolerance"]


def test_vde_capability_breakpoints_match_controller_rule() -> None:
    assert sweep.der_q_capability_mvar(9.9, 100.0) == (0.0, 0.0)
    assert sweep.der_q_capability_mvar(10.0, 100.0) == (-10.0, 10.0)
    qmin, qmax = sweep.der_q_capability_mvar(20.0, 100.0)
    assert math.isclose(qmin, -33.0)
    assert math.isclose(qmax, 41.0)


def _metric_row(signal: str, kind: str, t_settle: float, *,
                censored: bool = False, excursion: float = 5.0,
                tolerance: float = 1.0) -> dict:
    return {
        "signal": signal, "output_kind": kind, "t_settle_s": t_settle,
        "censored": censored, "excursion": excursion, "tolerance": tolerance,
        "subband": excursion <= tolerance,
    }


def test_worst_reduction_is_voltage_only_and_ignores_faster_q_flows() -> None:
    """The reported worst may only be picked from WORST_OUTPUT_KINDS.

    The interface-Q band is a fixed 1 Mvar applied to flows whose own steps
    differ by more than an order of magnitude, so a Q row can win the
    reduction on band tightness rather than on corridor speed.  Restricting
    the reduction is the whole point; the unrestricted worst stays available.
    """
    assert sweep.WORST_OUTPUT_KINDS == ("voltage",)
    rows = [
        _metric_row("V_TN_bus1", "voltage", 9.0),
        _metric_row("V_TN_bus2", "voltage", 4.0),
        _metric_row("Q_TSO_Z1_Z2", "tso_boundary_q", 17.0),
        _metric_row("Q_DSO_NC3W_DSO_1_t0", "dso_interface_q", 12.0),
    ]

    assert sweep.reduce_worst(rows, sweep.WORST_OUTPUT_KINDS)["signal"] == "V_TN_bus1"
    assert sweep.reduce_worst(rows, sweep.ALL_OUTPUT_KINDS)["signal"] == "Q_TSO_Z1_Z2"


def test_worst_reduction_ranks_censored_first_then_excursion() -> None:
    rows = [
        _metric_row("V_a", "voltage", 9.0),
        _metric_row("V_b", "voltage", 2.0, censored=True),
    ]
    assert sweep.reduce_worst(rows, ("voltage",))["signal"] == "V_b"

    tied = [
        _metric_row("V_a", "voltage", 0.0, excursion=0.4),
        _metric_row("V_b", "voltage", 0.0, excursion=0.9),
    ]
    assert sweep.reduce_worst(tied, ("voltage",))["signal"] == "V_b"


def test_worst_reduction_refuses_an_empty_scope() -> None:
    """An empty reduction would report 0 s for a case that was never judged."""
    rows = [_metric_row("Q_TSO_Z1_Z2", "tso_boundary_q", 17.0)]
    with pytest.raises(ValueError, match="no measured output"):
        sweep.reduce_worst(rows, sweep.WORST_OUTPUT_KINDS)
