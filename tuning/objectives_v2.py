"""
tuning/objectives_v2.py
=======================
Constrained-scalar objective for offline controller tuning.

Why this replaces the single weighted sum
-----------------------------------------
The legacy objective (:func:`tuning.metrics.extract_metrics`) folds three
logically different questions into one number:

* *is this operating point admissible?*  — binary,
* *how good is the response?*            — continuous,
* *how do we trade voltage against interface-Q against wear?* — a preference.

Empirically that scalarisation is what destroyed the tuning signal.  Over 1555
recorded scenario-runs, a random forest predicts the **individual** physical
metrics from the nine log-parameters with out-of-fold ``R^2`` of 0.28-0.53, but
the scalarised CVaR-25 cost with ``R^2 = 0.09``.  The mechanism is that
``contrib_pf`` is an exactly binary 0/100 term and ``contrib_viol`` is a hinge
weighted 1000 against a tracking contribution of order 10 — so the aggregate is
dominated by a feasibility indicator, and the optimiser was effectively solving
a classification problem.

Here feasibility becomes a set of **Optuna constraints** and the scalar keeps
only commensurable tracking / utilisation terms.  Constrained ``TPESampler``
partitions trials by feasibility *before* fitting its densities, so an
infeasible trial's objective value never attracts the sampler — which a penalty
term cannot achieve.

Constraint convention: ``<= 0`` is feasible (Optuna's convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import optuna
from optuna.trial import TrialState

from configs.config import MultiTSOConfig
from tuning.metrics import INFEASIBLE_SENTINEL, MetricScales, TrajectoryMetrics
from tuning.runner import RunResult

__all__ = [
    "CONSTRAINT_NAMES",
    "PERF_WEIGHT_PROFILES",
    "ConstraintLimits",
    "PerfWeights",
    "best_feasible_trial",
    "constraints_func",
    "feasibility_constraints",
    "performance_scalar",
]


#: ``study.user_attrs`` / ``trial.user_attrs`` key holding the constraint vector.
CONSTRAINT_KEY = "constraint"

CONSTRAINT_NAMES: tuple[str, ...] = (
    "g1_diverged",
    "g2_corridor",
    "g3_contraction",
    "g4_settling",
    "g5a_tap_ops",
    "g5b_tap_reversals",
)


@dataclass(frozen=True)
class ConstraintLimits:
    """Admissibility limits.  Each maps to one entry of the constraint vector."""

    #: g2 — tolerance on hard-corridor excess [pu/step].
    #:
    #: I predicted this would be inactive (the corridor is [0.90, 1.10] against
    #: a 1.03 pu setpoint).  **Measured, it is not**: the hand-tuned reference
    #: sits at 0.0022 pu/step, so it does touch the band.  The default below is
    #: therefore calibrated from that measurement with margin rather than set to
    #: a nominal zero — see :meth:`ConstraintLimits.from_reference`.
    #: Calibrated 2026-08-04 via ``from_reference`` (margin 1.5) on the 65-draw
    #: prior sample.  Was 5e-3, at which g2 was violated by 25 % of draws; at
    #: 1e-4 it is 46 % -- tighter, but still informative rather than empty.
    corridor_excess_pu: float = 1e-4

    #: g3 — empirical contraction.  ``rho_emp_p95`` is already recorded but was
    #: never enforced.  Since the search box lies entirely below the LMI
    #: stability floor (deliberately — see ``tuning/ceilings.py``), *no* sampled
    #: point carries a certificate, and this is the only stability evidence the
    #: procedure has.
    rho_emp_p95: float = 1.0

    #: g4 — worst per-event settling time [s].
    #:
    #: Censored windows (signal still outside the 2 % band at the window edge)
    #: contribute the full window width, so with a 1200 s window "never settled"
    #: reads as exactly 1200 s.  The hand-tuned reference hits precisely that,
    #: so either some signal genuinely does not settle within 20 min or the
    #: metric is mis-applied to a channel with a persistent offset.  **Until
    #: that is resolved the default is set above the window width, which makes
    #: g4 inactive** rather than silently rejecting every candidate for a
    #: reason that may be instrumental.  Do not tighten it before checking
    #: which signal is censored.
    settling_s: float = 1500.0

    #: g5a — absolute switching wear, **per transformer per hour of simulated
    #: operation**.  Aggregating over the fleet (as the legacy ``norm_tap`` did)
    #: hides one hunting tap changer behind a quiet fleet, and makes the two
    #: OLTC weights enter only through their sum.
    #:
    #: Per *hour*, not per day: the design scenarios are event-dense 75-minute
    #: windows and a real day is mostly quiet, so extrapolating one to the other
    #: inflates the figure ~19x.  The hand-tuned reference makes 5.0 taps in its
    #: window — an unremarkable 4.0/hour that read as a pathological "96/day"
    #: under the first version of this metric, and tripped a limit it had no
    #: business tripping.
    #: Calibrated 2026-08-04 via ``from_reference`` (margin 1.5): the reference's
    #: worst scenario makes 6.429 ops/h (``v2_undervoltage_ramp``), so the limit
    #: is 9.643.  **The old value of 6.0 sat BELOW the reference's own worst
    #: case**, so g5a was violated by 100 % of draws and the joint feasible set
    #: was empty (0/65).  With the calibrated limits it is 17/65.  Note the unit
    #: trap in the comment above: 6.0 was also the Stage-3 *median* target, but
    #: this constrains the **max** over scenarios and both classes, and the
    #: per-scenario spread is 0.804-6.429 -- so a median calibrated to 6.0 always
    #: violates a max limited to 6.0.
    tap_ops_per_h: float = 9.642857142857142

    #: g5b — hunting.  Reversals alone would miss monotone over-switching and
    #: absolute wear alone would miss chattering within budget, so both are
    #: needed.
    #: Calibrated 2026-08-04 via ``from_reference`` (margin 1.5): the reference is
    #: nearly reversal-free (worst 0.804/h), giving 1.205.  This is *tighter* than
    #: the old hand-set 4.0 -- deliberately, since hunting is the worse wear mode
    #: and the reference demonstrates it is avoidable.  Violated by 45 % of draws
    #: (37 % at 4.0), so it discriminates without emptying the box.
    tap_reversals_per_h: float = 1.2053571428571428

    @classmethod
    def from_reference(
        cls,
        reference: Sequence[TrajectoryMetrics],
        *,
        margin: float = 1.5,
        rho_emp_p95: float = 1.0,
        settling_s: float | None = None,
    ) -> "ConstraintLimits":
        """Derive limits from a measured reference run rather than inventing them.

        The defaults on this class started life as round numbers I chose, and
        the hand-tuned reference — the one operating point known to control
        well — failed three of them.  Two of those were the limit's fault, not
        the controller's.  That is the same circular-calibration failure the
        cost weights suffered (six successive revisions chasing BO output); the
        remedy is the same: anchor on a measurement.

        ``margin`` is the slack above the reference's worst observed value.
        1.5 means "the reference must pass with 50 % headroom", which keeps the
        constraint meaningful without making the reference itself inadmissible.

        ``rho_emp_p95`` stays at 1.0 regardless: it is a *stability* threshold
        derived from theory, not an operational preference, and the reference
        passes it on its own merits (measured 0.929).
        """
        def worst(attr: str) -> float:
            vals = [float(getattr(m, attr)) for m in reference]
            finite = [v for v in vals if math.isfinite(v)]
            return max(finite) if finite else 0.0

        excess = max(
            (m.voltage_excess_pu / max(m.n_records, 1) for m in reference),
            default=0.0,
        )
        return cls(
            corridor_excess_pu=max(excess * margin, 1e-4),
            rho_emp_p95=rho_emp_p95,
            settling_s=(settling_s if settling_s is not None
                        else cls.settling_s),
            tap_ops_per_h=max(
                max(worst("tap_ops_per_h_tso"), worst("tap_ops_per_h_dso"))
                * margin, 1.0),
            tap_reversals_per_h=max(
                max(worst("tap_reversals_per_h_tso"),
                    worst("tap_reversals_per_h_dso")) * margin, 1.0),
        )


@dataclass(frozen=True)
class PerfWeights:
    """Weights for the *performance* scalar.

    Only commensurable tracking / utilisation terms remain; feasibility left
    for the constraint vector.  These are preferences, and they are calibrated
    once from a random-prior sample rather than from optimiser output — the
    circularity of re-tuning them against BO results is what produced six
    successive cost revisions in the study history.
    """

    w_v_rms_ts:     float = 1.0
    w_v_rms_ds:     float = 0.3
    w_v_worst_ts:   float = 0.5
    w_v_band_ts:    float = 1.0
    w_q_pcc:        float = 1.0
    w_pcc_underutil: float = 0.3


#: Named preference profiles.  The *scales* (:class:`MetricScales`) are
#: measurements and are shared; these are the operator's priorities and are
#: therefore explicit, named and stamped into the study rather than edited in
#: place — two studies run under different profiles optimise different
#: functions and their objective values must never be compared.
#:
#: ``ts_voltage_primary`` states the design intent recorded 2026-08-13: TS
#: voltage tracking **is** the objective; interface-Q tracking is the means by
#: which the cascade delivers it, not an end in itself, so it stays in the
#: scalar (it is the coupling between the layers) but at a lower priority; DS
#: voltage matters with a looser tolerance.  OLTC wear is deliberately *not*
#: expressed here — switching count and hunting are the g5a / g5b constraints,
#: and folding them back into the scalar is the failure mode
#: :mod:`tuning.objectives_v2` exists to avoid.
PERF_WEIGHT_PROFILES: dict[str, PerfWeights] = {
    # As calibrated 2026-08-04; keeps the 2026-08 campaign reproducible.
    "calibrated_2026_08": PerfWeights(),
    # Shares of the scalar: TS voltage 66.7 %, DS voltage 16.0 %,
    # interface-Q 13.3 %, PCC utilisation 4.0 % (weights sum to 7.5).
    # Relative to ``calibrated_2026_08`` this is TS x2 and DS-V x4 at unchanged
    # interface-Q, i.e. interface-Q falls from 24 % to 13 % of the objective by
    # being out-weighted rather than by being suppressed -- it must stay a live
    # term, since it is the only place the cascade's inter-layer coupling is
    # scored at all.
    "ts_voltage_primary": PerfWeights(
        w_v_rms_ts=2.0,
        w_v_worst_ts=1.0,
        w_v_band_ts=2.0,
        w_v_rms_ds=1.2,
        w_q_pcc=1.0,
        w_pcc_underutil=0.3,
    ),
}


def _worst_settling_s(
    records: Sequence,
    event_times_s: Sequence[float],
    window_s: float = 1200.0,
) -> float:
    """Worst settling time across event-anchored windows.

    A settling metric is only meaningful relative to a disturbance, so windows
    are anchored on contingencies rather than on a fixed grid.  Returns ``0.0``
    when the scenario has no events (nothing to settle from) and ``nan`` when
    the trajectories cannot be built.
    """
    if not records or not event_times_s:
        return 0.0
    try:
        from experiments.helpers.rms_replay import (
            interval_settling_table,
            static_controlled_trajectories,
        )

        traj = static_controlled_trajectories(records)
        if not traj:
            return 0.0
        t_end = float(max(float(r.time_s) for r in records))
        windows = [
            (float(t), min(float(t) + window_s, t_end))
            for t in event_times_s
            if float(t) < t_end
        ]
        windows = [(a, b) for a, b in windows if b > a]
        if not windows:
            return 0.0
        table = interval_settling_table(traj, total_s=t_end, windows=windows)
        if table.empty:
            return 0.0
        return float(table["settling_time_s"].max())
    except Exception:
        # Diagnostics must never decide feasibility by crashing; an
        # unavailable settling metric is reported as nan and handled by the
        # caller as "no evidence", not as "violated".
        return float("nan")


def feasibility_constraints(
    results: Sequence[RunResult],
    cfg: MultiTSOConfig,
    limits: ConstraintLimits | None = None,
    settling_s_by_scenario: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Constraint vector for one trial; ``<= 0`` means feasible.

    Aggregation is worst-case across scenarios: a parameter set is admissible
    only if it is admissible everywhere in the design set.
    """
    limits = limits or ConstraintLimits()
    if not results:
        return tuple(float("inf") for _ in CONSTRAINT_NAMES)

    ms: list[TrajectoryMetrics] = [r.metrics for r in results]

    g1 = float(sum(1 for m in ms if not m.feasible))

    def _worst(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        return max(finite) if finite else float("inf")

    excess = _worst([
        m.voltage_excess_pu / max(m.n_records, 1) for m in ms
    ])
    g2 = excess - limits.corridor_excess_pu

    g3 = _worst([m.rho_emp_p95 for m in ms]) - limits.rho_emp_p95

    if settling_s_by_scenario is None:
        # No evidence rather than a violation: an absent metric must not make
        # a trial infeasible, or every trial fails for a plumbing reason.
        g4 = -limits.settling_s
    else:
        finite = [s for s in settling_s_by_scenario if math.isfinite(s)]
        g4 = (max(finite) - limits.settling_s) if finite else -limits.settling_s

    g5a = _worst(
        [m.tap_ops_per_h_tso for m in ms] + [m.tap_ops_per_h_dso for m in ms]
    ) - limits.tap_ops_per_h
    g5b = _worst(
        [m.tap_reversals_per_h_tso for m in ms]
        + [m.tap_reversals_per_h_dso for m in ms]
    ) - limits.tap_reversals_per_h

    return (g1, g2, g3, g4, g5a, g5b)


def performance_scalar(
    m: TrajectoryMetrics,
    weights: PerfWeights | None = None,
    scales: MetricScales | None = None,
) -> tuple[float, dict[str, float]]:
    """Tracking + utilisation cost for one trajectory, plus its breakdown.

    Voltage quality is measured against ``v_setpoint_pu`` (spatial RMS and
    worst-bus deviation), *not* against the hard corridor: the corridor is
    +/-7-8 % around the setpoint and therefore carries no information at any
    sane operating point.
    """
    weights = weights or PerfWeights()
    scales = scales or MetricScales()

    def n(value: float, scale: float) -> float:
        if not math.isfinite(value):
            return float("inf")
        return value / scale if scale > 0 else 0.0

    parts = {
        "v_rms_ts":       weights.w_v_rms_ts * n(m.v_rms_ts, scales.v_rms_ts),
        "v_rms_ds":       weights.w_v_rms_ds * n(m.v_rms_ds, scales.v_rms_ds),
        "v_worst_ts":     weights.w_v_worst_ts * n(m.v_worst_ts,
                                                   scales.v_worst_ts),
        "v_band_ts":      weights.w_v_band_ts * n(m.v_band_excess_ts,
                                                  scales.v_band_excess),
        "q_pcc":          weights.w_q_pcc * n(m.itae_q_pcc, scales.q_pcc),
        "pcc_underutil":  weights.w_pcc_underutil * n(m.itae_pcc_underutil,
                                                      scales.pcc_underutil),
    }
    total = sum(parts.values())
    if not math.isfinite(total) or not m.feasible:
        total = INFEASIBLE_SENTINEL
    return float(total), parts


def constraints_func(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
    """``constraints_func`` for ``TPESampler`` / ``NSGAIISampler``.

    Optuna calls this *after* the objective returns, so the objective stores
    the vector on the trial and this reads it back — the documented pattern.
    """
    stored = trial.user_attrs.get(CONSTRAINT_KEY)
    if stored is None:
        return tuple(float("inf") for _ in CONSTRAINT_NAMES)
    return tuple(float(v) for v in stored)


def best_feasible_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Lowest-objective trial **among the feasible ones**.

    ``Study.best_trial`` ignores constraints in a single-objective study, so
    using it would happily report an infeasible point as the answer.
    """
    feasible = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.value is not None
        and all(c <= 0.0 for c in constraints_func(t))
    ]
    if not feasible:
        raise RuntimeError(
            f"No feasible trial in study {study.study_name!r} "
            f"({len(study.trials)} trials). Inspect the per-constraint "
            f"violation counts before widening any limit."
        )
    return min(feasible, key=lambda t: float(t.value))


def sample_reparam_coords(trial: optuna.Trial) -> dict[str, float]:
    """Sample one point from the reparameterised space."""
    from tuning.reparam import BO_DIMS_V2

    return {
        p.name: trial.suggest_float(
            p.name, float(p.low), float(p.high), log=bool(p.log)
        )
        for p in BO_DIMS_V2
    }


def _run_scenario(
    scenario,
    cfg: MultiTSOConfig,
    scales: MetricScales,
) -> tuple[RunResult, list]:
    """Run one scenario and return ``(result, records)``.

    ``tuning.runner.run_one`` expects a BO-param dict to overlay, but the
    reparameterised path has already baked the weights into ``cfg`` — and it
    discards the record list, which the settling metric needs.  Same contract
    otherwise: never raises, suppresses child stdout/stderr.
    """
    import contextlib
    import io
    import time
    import traceback

    from tuning._sim_loader import get_run_multi_tso_dso
    from tuning.metrics import extract_metrics

    cfg_sc = scenario.overlay_on(cfg)
    run_fn = get_run_multi_tso_dso()

    t0 = time.perf_counter()
    failure, log = "", []
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), \
                contextlib.redirect_stderr(buf_err):
            log = run_fn(cfg_sc)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log = []
    wall = time.perf_counter() - t0

    metrics = extract_metrics(log, cfg_sc, scales=scales)
    return RunResult(scenario.name, metrics, wall, failure), log


def make_constrained_objective(
    baseline_cfg: MultiTSOConfig,
    gauge,
    scenarios: Sequence,
    *,
    fixed_overrides: dict | None = None,
    limits: ConstraintLimits | None = None,
    weights: PerfWeights | None = None,
    scales: MetricScales | None = None,
    short_circuit: bool = True,
    cvar_pct: float = 25.0,
    perf_exclude: frozenset[str] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective over the reparameterised, gauge-fixed space.

    One trial = one coordinate vector evaluated across the scenario set.  The
    constraint vector is stored on the trial for :func:`constraints_func`, and
    the returned scalar carries **tracking and utilisation only**.

    ``short_circuit`` abandons a trial after the first scenario that diverges.
    Roughly half of all historical trials landed in divergent regions, so this
    reclaims a large share of the wall clock at no cost in information: one
    divergence already makes the trial inadmissible whatever the remaining
    scenarios would have shown.

    ``perf_exclude`` names scenarios that still **run** and still enter the
    constraint vector -- so a candidate must survive them -- but are left out of
    the performance aggregate.  Measured 2026-08-04, this is required for
    ``v2_undervoltage_ramp``:

    * it starts at ``2016-01-21 18:00``, a winter evening peak, where PV-based
      TS-DER produce no active power and therefore have **zero** reactive
      capability.  ``zone_q_der`` is exactly 0.0 across all 900 (step x DER)
      entries, constant in time, while ``v2_quiet_spring`` on the same network
      spans -133..+64 Mvar;
    * ``tau_der_pcc`` reaches the plant only through
      ``precondition_class_scales = {der: sqrt(tau), pcc: 1/sqrt(tau)}``, so with
      no DER to allocate that coordinate is **structurally inert** there;
    * the scenario's performance scalar is ~85x the others, so under a
      worst-case aggregator (``cvar_pct=25`` over 4 scenarios *is* the maximum)
      it became the entire objective.  Result: 8 of 24 trials shared an
      objective value to 6 significant figures across a 660x range in ``tau``,
      while ``quiet_spring`` and ``gen_trip`` varied by 668 % and 493 %.

    Excluding it costs no stress coverage: ``v2_gen_trip`` is also a stress case
    (generator trip) at spring noon, where DER capability is full.  Pair this
    with ``cvar_pct=100`` (the mean) so the retained scenarios, whose scalars are
    comparable, are not collapsed to their own maximum.
    """
    from tuning.reparam import apply_reparam_to_config
    from tuning.objective import cvar_aggregate

    limits = limits or ConstraintLimits()
    weights = weights or PerfWeights()
    scales = scales or MetricScales()
    excluded = frozenset(perf_exclude or ())
    unknown = excluded - {getattr(s, "name", "") for s in scenarios}
    if unknown:
        raise ValueError(
            f"perf_exclude names scenarios not in the set: {sorted(unknown)}; "
            f"available: {sorted(getattr(s, 'name', '') for s in scenarios)}"
        )
    if excluded and len(excluded) >= len(scenarios):
        raise ValueError(
            "perf_exclude would leave no scenario in the performance "
            "aggregate; the objective would be vacuous."
        )

    def objective(trial: optuna.Trial) -> float:
        coords = sample_reparam_coords(trial)
        cfg = apply_reparam_to_config(
            baseline_cfg, coords, gauge, fixed_overrides=fixed_overrides,
        )

        results: List[RunResult] = []
        settling: list[float] = []
        for scenario in scenarios:
            res, records = _run_scenario(scenario, cfg, scales)
            results.append(res)
            settling.append(
                _worst_settling_s(records, scenario.event_times_s)
            )
            trial.set_user_attr(f"wall_s__{scenario.name}",
                                float(res.wall_time_s))
            if res.failure_reason:
                trial.set_user_attr(f"err__{scenario.name}",
                                    res.failure_reason[:500])
            if short_circuit and not res.metrics.feasible:
                trial.set_user_attr("short_circuited_at", scenario.name)
                break

        g = feasibility_constraints(
            results, cfg, limits,
            settling_s_by_scenario=settling if any(
                math.isfinite(s) and s > 0.0 for s in settling) else None,
        )
        trial.set_user_attr(CONSTRAINT_KEY, [float(v) for v in g])
        for name, value in zip(CONSTRAINT_NAMES, g):
            trial.set_user_attr(f"c__{name}", float(value))

        # Every scenario is scored and recorded -- the excluded ones remain
        # visible as diagnostics -- but only the retained ones are aggregated.
        values: list[float] = []
        excluded_values: list[float] = []
        for res in results:
            total, parts = performance_scalar(res.metrics, weights, scales)
            if res.scenario_name in excluded:
                excluded_values.append(total)
            else:
                values.append(total)
            trial.set_user_attr(f"perf__{res.scenario_name}", float(total))
            for key, part in parts.items():
                trial.set_user_attr(
                    f"perf__{res.scenario_name}__{key}", float(part))
        for name, val in zip(sorted(excluded), excluded_values):
            trial.set_user_attr(f"perf_excluded__{name}", float(val))

        # Aggregate over the retained scenarios.  ``cvar_pct=25`` over a
        # 3-4 scenario set degenerates to the maximum, which handed the whole
        # objective to whichever scenario had the largest absolute scalar --
        # see the ``perf_exclude`` note in the factory docstring.  Use
        # ``cvar_pct=100`` for the mean when the retained scalars are
        # comparable in magnitude.
        agg = cvar_aggregate(values, pct=cvar_pct)
        trial.set_user_attr("cvar_perf", float(agg))
        trial.set_user_attr(
            "mean_perf", float(np.mean(values)) if values else float("inf"))
        trial.set_user_attr("n_perf_scenarios", len(values))
        return float(agg)

    return objective


def constraint_violation_report(study: optuna.Study) -> dict[str, int]:
    """How many completed trials violate each constraint.

    Run this before relaxing a limit: it distinguishes "the box is empty
    because of stability" from "because of switching wear".
    """
    counts = {name: 0 for name in CONSTRAINT_NAMES}
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        for name, value in zip(CONSTRAINT_NAMES, constraints_func(t)):
            if value > 0.0:
                counts[name] += 1
    return counts
