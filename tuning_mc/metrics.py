"""
tuning_mc/metrics.py
====================
Objective, filter criteria and hard constraints for the Stage-1 search.

Three tiers, deliberately separated
-----------------------------------
**Hard constraints** — physical admissibility.  A candidate that violates one
is rejected outright (extreme barrier); there is no trade to be had against
divergence or an unstable contraction estimate.

    g1 divergence, g2 corridor excess, g3 rho_emp_p95,
    g4 settling, g5a taps/h, g5b reversals/h

**Filter criteria** — competing goods with no defensible exchange rate.  A
candidate is accepted when it is *non-dominated*: better on one without being
worse on the other.  This is the Audet-Dennis filter, and it is what the
operator's statement "interface-Q may be violated, but the least violation is
best" actually means.  Two criteria:

    f_ts  TS voltage cost   (RMS + worst-bus + band excess, the stated objective)
    f_q   interface-Q tracking error

Folding ``f_q`` into ``f_ts`` with a weight would require an exchange rate
between pu volts and Mvar that nothing in the plant supplies -- the failure
mode ``tuning/objectives_v2.py`` exists to avoid.  A filter needs none.

**Reported diagnostics** — DS voltage, PCC under-utilisation, per-scenario
breakdown, tap statistics.  Recorded on every candidate, never optimised
directly.

Aggregation
-----------
Per-scenario scalars are combined with CVaR at ``cvar_pct``: the mean of the
worst ``cvar_pct`` fraction.  100 is the plain mean.  **The tail must contain
more than one scenario or CVaR degenerates into the maximum** -- with the
five-scenario tune set, ``cvar_pct <= 20`` selects exactly one scenario and
hands it the entire objective, which is the defect that forced
``perf_exclude`` in the previous campaign.  Default is therefore 100.

Everything reuses ``tuning.objectives_v2`` so the numbers stay comparable with
the existing studies; only the *structure* (filter instead of scalar) is new.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from tuning.metrics import MetricScales, TrajectoryMetrics
from tuning.objectives_v2 import (
    CONSTRAINT_NAMES,
    ConstraintLimits,
    PerfWeights,
    feasibility_constraints,
    performance_scalar,
)

__all__ = [
    "CandidateScore",
    "TS_VOLTAGE_KEYS",
    "dominates",
    "filter_accepts",
    "score_candidate",
]

#: Which parts of ``performance_scalar`` constitute the stated objective.
TS_VOLTAGE_KEYS = ("v_rms_ts", "v_worst_ts", "v_band_ts")


def _cvar(values: Sequence[float], pct: float) -> float:
    """Mean of the worst ``pct`` percent.  ``pct = 100`` is the mean."""
    v = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("inf")
    if pct >= 100.0:
        return float(v.mean())
    k = max(1, int(math.ceil(v.size * pct / 100.0)))
    return float(np.sort(v)[-k:].mean())


@dataclass
class CandidateScore:
    """Everything one candidate produced, on one scenario ensemble."""

    f_ts: float                       # filter criterion 1 (minimise)
    f_q: float                        # filter criterion 2 (minimise)
    hard: tuple[float, ...]           # <= 0 is feasible, per CONSTRAINT_NAMES
    feasible: bool
    f_ds: float = float("nan")        # diagnostic
    f_total: float = float("nan")     # legacy scalar, for comparability
    worst_tap_ops_per_h: float = float("nan")
    worst_reversals_per_h: float = float("nan")
    #: First-class output, not a by-product: this is the quantity the lambda
    #: calibration is measured against, since the linearised design target
    #: (lambda_max over the preconditioned columns) and the realised
    #: contraction differ by a factor that must be measured, not assumed.
    worst_rho_emp_p95: float = float("nan")
    per_scenario: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "f_ts": self.f_ts, "f_q": self.f_q, "f_ds": self.f_ds,
            "f_total": self.f_total, "feasible": self.feasible,
            "hard": {n: float(v) for n, v in zip(CONSTRAINT_NAMES, self.hard)},
            "worst_tap_ops_per_h": self.worst_tap_ops_per_h,
            "worst_reversals_per_h": self.worst_reversals_per_h,
            "worst_rho_emp_p95": self.worst_rho_emp_p95,
            "per_scenario": self.per_scenario,
            "failures": self.failures,
        }


def score_candidate(
    results: Sequence[Any],
    cfg,
    *,
    settling_s_by_scenario: Sequence[float] | None = None,
    limits: ConstraintLimits | None = None,
    weights: PerfWeights | None = None,
    scales: MetricScales | None = None,
    cvar_pct: float = 100.0,
    ds_criterion: str = "v_rms",
) -> CandidateScore:
    """Score one candidate from its per-scenario ``RunResult`` list.

    ``results`` are :class:`tuning.runner.RunResult`; the same objects the
    existing objectives consume, so a candidate scored here and a trial scored
    by ``objectives_v2`` see identical metrics.

    ``ds_criterion`` selects what ``f_ds`` measures:

    ``"v_rms"`` (default, reproduces every study up to 2026-08-18)
        ``v_rms_ds = mean|v_mean_ds - v_set|`` -- the distance of the DSO
        envelope's *centre* from the setpoint.

    ``"guard"``
        ``guard_deficit_ds_pu`` -- mean per-step excess beyond a corridor shrunk
        by ``DS_GUARD_HEADROOM_PU`` at both ends, i.e. a *headroom* measure.

    The default is kept only for reproducibility.  ``"v_rms"`` cannot express
    the failure mode measured on 2026-08-18: with ``dso_gamma_oltc_q = 0`` the
    interface OLTC drives the DSO profile's centre onto ``v_set`` *by
    construction*, so ``v_rms_ds`` is smallest exactly when the tap has spent
    its authority and the network is most stressed.  Measured on the tuned
    reference it ranked DSO_2 -- second-worst of four, already 0.0012 pu outside
    the corridor -- as the *healthiest* area.  New work should pass
    ``ds_criterion="guard"`` together with ``with_ds=True``; see
    ``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md``.
    """
    if ds_criterion not in ("v_rms", "guard"):
        raise ValueError(
            f"ds_criterion must be 'v_rms' or 'guard', got {ds_criterion!r}"
        )
    limits = limits or ConstraintLimits()
    weights = weights or PerfWeights()
    scales = scales or MetricScales()

    hard = feasibility_constraints(
        results, cfg, limits, settling_s_by_scenario=settling_s_by_scenario)

    ts_vals: list[float] = []
    q_vals: list[float] = []
    ds_vals: list[float] = []
    tot_vals: list[float] = []
    per_scenario: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}

    for res in results:
        total, parts = performance_scalar(res.metrics, weights, scales)
        f_ts = float(sum(parts.get(k, 0.0) for k in TS_VOLTAGE_KEYS))
        f_q = float(parts.get("q_pcc", 0.0))
        if ds_criterion == "guard":
            f_ds = float(getattr(res.metrics, "guard_deficit_ds_pu", 0.0))
        else:
            f_ds = float(parts.get("v_rms_ds", 0.0))
        ts_vals.append(f_ts)
        q_vals.append(f_q)
        ds_vals.append(f_ds)
        tot_vals.append(float(total))
        m: TrajectoryMetrics = res.metrics
        per_scenario[res.scenario_name] = {
            "f_ts": f_ts, "f_q": f_q, "f_ds": f_ds, "f_total": float(total),
            "ds_headroom_min_pu": float(
                getattr(m, "ds_headroom_min_pu", float("nan"))),
            "tap_ops_per_h_tso": float(m.tap_ops_per_h_tso),
            "tap_ops_per_h_dso": float(m.tap_ops_per_h_dso),
            "tap_reversals_per_h_tso": float(m.tap_reversals_per_h_tso),
            "tap_reversals_per_h_dso": float(m.tap_reversals_per_h_dso),
            "rho_emp_p95": float(m.rho_emp_p95),
            "feasible": bool(m.feasible),
            # ── Full metric vector, so a criterion change can be re-scored
            # OFFLINE instead of forcing a re-simulation. ────────────────────
            # Learned 2026-08-18 the expensive way: adding a DS-voltage
            # criterion (guard_deficit_ds_pu) could not be applied to any of
            # the 110 archived tier-1 trials, because ``stage_1_search``
            # receives ``(res, records)`` per scenario but persists only this
            # dict, and this dict carried no DS voltage envelope.  Re-scoring
            # would have cost minutes; re-simulating costs ~9 h wall.
            #
            # Everything above is kept verbatim so existing readers
            # (report_0815, select_windows_v2, export_final) are unaffected --
            # this key is purely additive.  TrajectoryMetrics is a flat
            # dataclass of scalars, so the JSON cost is ~40 numbers per
            # scenario.  Non-finite entries are written as JSON ``NaN`` by
            # ``json.dumps`` default ``allow_nan=True``, which the existing
            # payload already relies on and ``json.loads`` reads back.
            "metrics": dataclasses.asdict(m),
        }
        if getattr(res, "failure_reason", ""):
            failures[res.scenario_name] = res.failure_reason[:400]

    def _worst(attr_a: str, attr_b: str) -> float:
        vals = [v[attr_a] for v in per_scenario.values()] + \
               [v[attr_b] for v in per_scenario.values()]
        finite = [v for v in vals if math.isfinite(v)]
        return max(finite) if finite else float("nan")

    return CandidateScore(
        f_ts=_cvar(ts_vals, cvar_pct),
        f_q=_cvar(q_vals, cvar_pct),
        f_ds=_cvar(ds_vals, cvar_pct),
        f_total=_cvar(tot_vals, cvar_pct),
        hard=tuple(float(v) for v in hard),
        feasible=all(v <= 0.0 for v in hard),
        worst_tap_ops_per_h=_worst("tap_ops_per_h_tso", "tap_ops_per_h_dso"),
        worst_reversals_per_h=_worst("tap_reversals_per_h_tso",
                                     "tap_reversals_per_h_dso"),
        worst_rho_emp_p95=max(
            (v["rho_emp_p95"] for v in per_scenario.values()
             if math.isfinite(v["rho_emp_p95"])), default=float("nan")),
        per_scenario=per_scenario,
        failures=failures,
    )


def dominates(a: CandidateScore, b: CandidateScore, *, tol: float = 0.0,
              with_ds: bool = False) -> bool:
    """``a`` dominates ``b``: no worse on any criterion, better on one.

    ``tol`` is a relative slack; a difference below it counts as "no worse",
    which keeps numerical noise from producing spurious filter entries.

    ``with_ds`` adds ``f_ds`` -- the subordinate layer's own voltage cost -- as a
    third criterion.  **Measured 2026-08-15, this is not optional in substance.**
    With the two-criterion filter the search moved ``dso_g_v_ratio`` from 1.0 to
    0.50, improving ``f_q`` by 15 % while degrading ``f_ds`` by 47 % on the
    design bank and 9 % on realistic 12-h windows.  Subordinate nodal voltage is
    a *stated controlled output*, so a criterion outside the filter is one the
    search is free to spend, and it spent it.

    The flag defaults to ``False`` so that re-running the 0814 and early-0815
    studies reproduces what they actually ran.  New work should pass ``True``.
    """
    def le(x: float, y: float) -> bool:
        return x <= y * (1.0 + tol)

    def lt(x: float, y: float) -> bool:
        return x < y * (1.0 - tol)

    pairs = [(a.f_ts, b.f_ts), (a.f_q, b.f_q)]
    if with_ds and math.isfinite(a.f_ds) and math.isfinite(b.f_ds):
        pairs.append((a.f_ds, b.f_ds))
    return (all(le(x, y) for x, y in pairs)
            and any(lt(x, y) for x, y in pairs))


def filter_accepts(
    candidate: CandidateScore,
    incumbents: Sequence[CandidateScore],
    *,
    tol: float = 0.0,
    with_ds: bool = False,
) -> bool:
    """True when ``candidate`` is feasible and dominated by no incumbent.

    Infeasible candidates are rejected outright: the hard tier is a barrier,
    not a trade.
    """
    if not candidate.feasible:
        return False
    return not any(dominates(inc, candidate, tol=tol, with_ds=with_ds)
                   for inc in incumbents)
