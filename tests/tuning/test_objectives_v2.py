"""Unit tests for ``tuning/objectives_v2.py`` (constrained-scalar objective).

The point of this module is that feasibility is a *constraint*, not a term in a
weighted sum.  These tests pin the properties that were violated by the legacy
scalar cost — most importantly that divergence can never be cheaper than a
converged run.
"""

from __future__ import annotations

import math

import pytest

from configs.config import MultiTSOConfig
from tuning.metrics import INFEASIBLE_SENTINEL, TrajectoryMetrics
from tuning.objectives_v2 import (
    CONSTRAINT_NAMES,
    ConstraintLimits,
    PerfWeights,
    feasibility_constraints,
    performance_scalar,
)
from tuning.runner import RunResult


def _result(name: str = "s", **kw) -> RunResult:
    return RunResult(
        scenario_name=name,
        metrics=TrajectoryMetrics(**kw),
        wall_time_s=1.0,
    )


def _healthy(**over) -> RunResult:
    base = dict(
        n_records=100, rho_emp_p95=0.6, voltage_excess_pu=0.0,
        tap_ops_per_h_tso=4.0, tap_ops_per_h_dso=3.0,
        tap_reversals_per_h_tso=1.0, tap_reversals_per_h_dso=0.5,
        v_rms_ts=0.004, v_rms_ds=0.008, v_worst_ts=0.010,
        itae_q_pcc=9000.0,
    )
    base.update(over)
    return _result(**base)


def test_constraint_vector_has_one_entry_per_name(cfg=MultiTSOConfig()) -> None:
    g = feasibility_constraints([_healthy()], cfg)
    assert len(g) == len(CONSTRAINT_NAMES)


def test_healthy_run_is_feasible() -> None:
    g = feasibility_constraints([_healthy()], MultiTSOConfig())
    assert all(v <= 0.0 for v in g), dict(zip(CONSTRAINT_NAMES, g))


def test_divergence_trips_g1() -> None:
    """A diverged scenario must be inadmissible, not merely expensive.

    Under the legacy scalar a diverged run scored ``w_pf`` = 100, which was
    *cheaper* than 35-43 % of converged runs — so the optimiser was rewarded for
    finding divergent regions.
    """
    g = feasibility_constraints(
        [_result(n_records=0, infeasible_reason="pf_failure")],
        MultiTSOConfig(),
    )
    assert dict(zip(CONSTRAINT_NAMES, g))["g1_diverged"] > 0.0
    assert not all(v <= 0.0 for v in g)


def test_contraction_above_unity_trips_g3() -> None:
    """``rho_emp_p95`` is the only stability evidence the procedure has.

    The search box lies entirely below the LMI stability floor by design, so no
    sampled point carries a certificate.  This constraint was computed but never
    enforced.
    """
    g = dict(zip(CONSTRAINT_NAMES,
                 feasibility_constraints([_healthy(rho_emp_p95=1.4)],
                                         MultiTSOConfig())))
    assert g["g3_contraction"] == pytest.approx(0.4)


def test_switching_wear_and_hunting_are_separate_constraints() -> None:
    """Reversals alone miss monotone over-switching; wear alone misses chatter."""
    limits = ConstraintLimits()
    over_budget = _healthy(tap_ops_per_h_tso=40.0, tap_reversals_per_h_tso=0.0)
    g = dict(zip(CONSTRAINT_NAMES,
                 feasibility_constraints([over_budget], MultiTSOConfig(), limits)))
    assert g["g5a_tap_ops"] > 0.0
    assert g["g5b_tap_reversals"] <= 0.0        # not hunting, just too many

    hunting = _healthy(tap_ops_per_h_tso=5.0, tap_reversals_per_h_tso=9.0)
    g = dict(zip(CONSTRAINT_NAMES,
                 feasibility_constraints([hunting], MultiTSOConfig(), limits)))
    assert g["g5a_tap_ops"] <= 0.0              # within budget
    assert g["g5b_tap_reversals"] > 0.0         # but reversing


def test_constraints_are_worst_case_across_scenarios() -> None:
    """One bad scenario makes the parameter set inadmissible."""
    g = dict(zip(CONSTRAINT_NAMES, feasibility_constraints(
        [_healthy(), _healthy(rho_emp_p95=1.5), _healthy()],
        MultiTSOConfig(),
    )))
    assert g["g3_contraction"] == pytest.approx(0.5)


def test_missing_settling_metric_is_not_a_violation() -> None:
    """An unavailable diagnostic must not make every trial infeasible."""
    g = dict(zip(CONSTRAINT_NAMES,
                 feasibility_constraints([_healthy()], MultiTSOConfig(),
                                         settling_s_by_scenario=None)))
    assert g["g4_settling"] < 0.0


def test_performance_scalar_terms_are_order_one() -> None:
    """No 5-decade weight spread: every term is comparable at tolerance.

    The legacy cost mixed a binary 0/100 term, a 1000-weighted hinge and an
    order-10 tracking term, so the aggregate was dominated by the indicator.
    """
    total, parts = performance_scalar(_healthy().metrics)
    assert math.isfinite(total)
    assert all(0.0 <= v < 10.0 for v in parts.values()), parts


def test_performance_scalar_pins_infeasible_to_sentinel() -> None:
    total, _ = performance_scalar(
        TrajectoryMetrics(infeasible_reason="pf_failure"))
    assert total == pytest.approx(INFEASIBLE_SENTINEL)


def test_performance_scalar_rewards_better_voltage_tracking() -> None:
    good = performance_scalar(_healthy(v_rms_ts=0.002).metrics)[0]
    poor = performance_scalar(_healthy(v_rms_ts=0.012).metrics)[0]
    assert good < poor


def test_worst_bus_term_catches_what_the_mean_hides() -> None:
    """A zone half at 1.00 pu and half at 1.06 has a perfect *mean*.

    The legacy voltage metric used the spatial mean and therefore scored that
    case as ideal; the RMS and worst-bus terms are what make it visible.
    """
    tight = performance_scalar(
        _healthy(v_rms_ts=0.004, v_worst_ts=0.006).metrics)[0]
    spread = performance_scalar(
        _healthy(v_rms_ts=0.004, v_worst_ts=0.030).metrics)[0]
    assert spread > tight


def test_custom_weights_are_honoured() -> None:
    m = _healthy(v_rms_ts=0.010).metrics
    light = performance_scalar(m, PerfWeights(w_v_rms_ts=1.0))[0]
    heavy = performance_scalar(m, PerfWeights(w_v_rms_ts=10.0))[0]
    assert heavy > light


def test_limits_can_be_calibrated_from_a_reference() -> None:
    """Limits must come from a measurement, not from round numbers.

    The defaults on ``ConstraintLimits`` began as values I chose, and the
    hand-tuned reference — the one operating point known to control well —
    failed three of them.  Two of those were the limit's fault: the wear metric
    was extrapolating an event-dense 75-min window to a full day, inflating a
    perfectly ordinary 4 taps/hour into "96 ops/day".  Anchoring on the
    reference is the same non-circular discipline the cost weights need.
    """
    reference = _healthy(tap_ops_per_h_tso=4.0, tap_reversals_per_h_tso=0.8)
    limits = ConstraintLimits.from_reference([reference.metrics], margin=1.5)

    assert limits.tap_ops_per_h == pytest.approx(6.0)
    assert limits.tap_reversals_per_h == pytest.approx(1.2)
    # A stability threshold is theory, not preference: it does not move.
    assert limits.rho_emp_p95 == pytest.approx(1.0)

    # And the reference must then actually pass the limits derived from it.
    g = feasibility_constraints([reference], MultiTSOConfig(), limits)
    assert all(v <= 0.0 for v in g), dict(zip(CONSTRAINT_NAMES, g))


# ---------------------------------------------------------------------------
# perf_exclude: scenarios that constrain but cannot discriminate
# ---------------------------------------------------------------------------

def test_perf_exclude_rejects_unknown_and_exhaustive_sets() -> None:
    """Guard the two ways the option can silently produce nonsense.

    A typo'd scenario name would otherwise exclude nothing and be invisible; an
    exhaustive set would leave the performance aggregate empty, which
    ``cvar_aggregate`` reports as ``inf`` for *every* trial -- a study that looks
    like it ran and ranks nothing.
    """
    from tuning.objectives_v2 import make_constrained_objective
    from tuning.reparam import Gauge

    class _S:
        def __init__(self, name):
            self.name = name

        def event_times_s(self):
            return ()

    cfg = MultiTSOConfig()
    gauge = Gauge.from_config(cfg)
    scenarios = [_S("a"), _S("b")]

    with pytest.raises(ValueError, match="not in the set"):
        make_constrained_objective(cfg, gauge, scenarios,
                                   perf_exclude=frozenset({"typo"}))

    with pytest.raises(ValueError, match="vacuous"):
        make_constrained_objective(cfg, gauge, scenarios,
                                   perf_exclude=frozenset({"a", "b"}))


def test_cvar_over_a_small_set_is_the_maximum() -> None:
    """The defect that made one scenario the entire objective.

    ``cvar_pct=25`` over 4 scenarios keeps ceil(4*0.25) = 1 value, i.e. the
    maximum.  Measured 2026-08-04, ``v2_undervoltage_ramp`` was ~85x the other
    scenarios' scalars, so the objective was that scenario alone -- and it is the
    one where TS-DER reactive capability is zero (winter 18:00), making
    ``tau_der_pcc`` structurally inert.  ``pct=100`` recovers the mean.
    """
    from tuning.objective import cvar_aggregate

    vals = [2.7, 2.8, 231.2, 2.2]
    assert cvar_aggregate(vals, pct=25.0) == pytest.approx(231.2)
    assert cvar_aggregate(vals, pct=100.0) == pytest.approx(
        sum(vals) / len(vals))
    # Dropping the dominating scenario is what restores dynamic range: the
    # retained scalars differ by 27 %, the full set's max by 0.05 %.
    kept = [2.7, 2.8, 2.2]
    assert cvar_aggregate(kept, pct=100.0) == pytest.approx(
        sum(kept) / len(kept))
