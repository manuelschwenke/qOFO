r"""
tuning/stage2_search.py
=======================
Stage II: the parameter search, and the rule that picks a point out of it.

Shape of the search
-------------------
Nine coordinates (``tuning.parameters.SEARCHED_FIELDS``; the bank step price
``g_w_tso_shunt`` is the ninth since 2026-10-22), log-scaled, in a box centred
on the Stage-I design point.  ``g_v`` is pinned: the objective is positively
homogeneous of degree one in the whole weight vector, so a common rescaling
reproduces the trajectory exactly -- including the integer tap and bank
sequence -- and searching all ten would spend budget on a direction that
provably changes nothing.  ``tuning.stage1_screen`` asserts that invariance to
~1e-16 rather than assuming it.

**II-a, screening.**  A scrambled Sobol design (a low-discrepancy quasi-Monte-
Carlo sample, of which Latin hypercube sampling is the more familiar relative)
fills the box far more evenly at a few hundred points than i.i.d. uniform
sampling, and is reproducible from one recorded seed.  Every draw is first put
through the **analytic** Stage-I screen, which costs microseconds; only the
survivors are simulated, on the short bank.

That ordering is the whole reason this is affordable.  Measured on this server
2026-09-10: 5.6 min wall per simulated hour per worker, so one candidate is
~1.7 worker-hours on the 3 h bank and ~5.6 on the 12 h bank.  Screening 512
draws analytically and simulating only the survivors turns a week of compute
into a night.

**II-b, refinement.**  A one-factor-at-a-time sweep of the coordinates II-a
shows actually matter, on the 12 h bank.  The documented failure mode of a pure
compass search -- it cannot find a direction that needs two coordinates to move
together -- is covered by II-a having already sampled joint moves.

**II-c, the tap prices.**  ``g_w_tso_oltc`` and ``g_w_dso_oltc`` are calibrated
by 1-D bisection against a taps/day budget (:mod:`tuning.bisect_switching`),
which proves them monotone in switching count per solve.  That converts a free
tuning parameter into a stated operational requirement.

Selection
---------
Constraints, then a lexicographic order -- no composite cost, and therefore no
scoring weights to justify.

1. **Filter.**  A candidate violating an operational limit is *discarded*, not
   penalised.  Limits are in engineering units (taps/day/transformer,
   reversals/day, pu) and are stated in :class:`Limits`.
2. **Rank** survivors on ``e_v`` in the transmission system alone.
3. **Break ties** within a stated indifference band on ``e_v`` in the
   distribution system.

``e_q`` is reported and bounded above but **never minimised**.  It is a means to
an end -- what matters is that interface-Q tracking buys a smaller ``e_v`` in
the TS -- and minimising it directly is gameable: a prior Bayesian run drove
``g_w_pcc`` to ~270 because freezing the PCC setpoint makes interface-Q error
vanish trivially while suppressing the very support the cascade exists to
provide (``tuning/parameters.py`` history, 2026-05-02).

Author: Manuel Schwenke (with Claude Code), 2026-09-10.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tuning.parameters import (
    GAUGE_FIELD,
    SEARCHED_FIELDS,
    WEIGHT_FIELDS,
    weight_fingerprint,
)

__all__ = [
    "Limits",
    "Box",
    "sobol_candidates",
    "avr_step_metrics",
    "evaluate_candidate",
    "check_limits",
    "rank_lexicographic",
]


# ---------------------------------------------------------------------------
#  Operational limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Limits:
    """Operational bounds a candidate must satisfy to be considered at all.

    These are asset-management statements, not tuning knobs.  They are applied
    as a **filter**: a candidate outside them is discarded rather than given a
    penalty, because a penalty lets the search trade a maintenance budget
    against a tracking error, which is not the control designer decision to
    make.

    Tap limits are per transformer and per day, and may only be evaluated on
    the 12 h bank.  One tap is 0.083/h at 12 h against 0.33/h at 3 h, so a
    daily budget quantised on the short bank is a coin toss rather than a
    constraint -- applying it there does not make the search stricter, it makes
    every candidate infeasible and the search stops without moving.
    """

    tap_ops_per_day: float = 30.0

    #: Hunting reversals tolerated over the whole bank.  Zero: a tap that
    #: reverses inside :attr:`hunt_dwell_h` with no new disturbance in between
    #: is chasing its own tail, and one such cycle is one too many.
    #:
    #: This replaces a per-day reversal budget, which was the wrong quantity.
    #: A raw reversal count punishes a tap for moving on a generator trip and
    #: moving back on the restore -- correct behaviour -- while a rate derived
    #: from it cannot be read off a short window at all.  Measured 2026-09-11
    #: on a 5 h disturbed run: the raw count extrapolated to 9.6-14.4
    #: reversals/day for every candidate including the analytic baseline,
    #: while the hunting count was 0, 0 and 3.
    max_hunting: int = 0
    hunt_dwell_h: float = 2.0

    avr_max_step_pu: float = 0.005
    lambda_max_b: float = 1.6
    #: Records in which a bus left the hard corridor [v_min_pu, v_max_pu].
    #: This is the corridor, NOT the inner quality band: v_band_excess_* is
    #: the mean excess beyond an inner band around the setpoint and is
    #: non-zero for every sane operating point, so constraining it at zero
    #: rejects every candidate rather than discriminating between them.
    #: It is reported as a diagnostic instead.
    max_v_violations: int = 0
    allow_pf_failures: int = 0

    #: Upper bound on interface-Q error, RELATIVE to the Stage-I reference on
    #: the same bank.  A BOUND, not an objective: it stops a candidate
    #: abandoning interface tracking altogether, without giving the search a
    #: reason to freeze the PCC setpoint to drive the error to zero.
    #:
    #: Relative because ``itae_q_pcc`` is a time-weighted INTEGRAL, not a
    #: reactive power: it is not in Mvar and it scales roughly quadratically
    #: with window length, so an absolute threshold would mean a different
    #: thing on the 3 h and 12 h banks.  Comparable within a bank only.
    e_q_max_ratio: float = 3.0

    # -- Switched shunts (bounded since 2026-10-22) ---------------------------
    #
    # From 2026-09-15 to 2026-10-22 bank switching was reported and NOT
    # bounded, by author decision: the banks were dispatched by an integrator
    # outside the MIQP, so their switching was not what the MIQP weights were
    # selected to govern.  Measured then on ``results/THESIS_ch9_3_shunt_live``
    # (12 h per window, ``shunt_int_g_w = 0.03``): 8 / 12 / 17 / 49 / 112 / 15
    # commits in W1-W6.
    #
    # Since the banks became integer inputs of the TS-MIQP, their step price
    # ``g_w_tso_shunt`` is a searched weight.  Left unbounded, the ranking on
    # e_v(TS) could buy voltage with bank switching that nothing charges for,
    # so bank wear is a filter like tap wear.

    #: Commits per bank per day.  No default: the budget is an asset statement
    #: the campaign protocol must make, and ``ch_9_3_select`` refuses to judge
    #: a bank that carries shunt wear records without one.
    shunt_ops_per_day: Optional[float] = None

    #: Bank hunting reversals tolerated over the whole bank, with the tap
    #: definition (:func:`hunting_reversals`, :attr:`hunt_dwell_h`).
    max_shunt_hunting: int = 0


@dataclass(frozen=True)
class Box:
    """Log-space search box: ``centre * 10**(-span) .. centre * 10**(+span)``.

    Bounds are drawn around the Stage-I point and travel with it.  They are not
    stored as absolute limits, which is how the previous search space came to
    exclude its own known-good configuration (``g_v = 1e7`` against a box of
    ``[1e2, 1e5]``, ``g_w_pcc = 80`` against a ceiling of 30).
    """

    centre: Dict[str, float]
    spans: Dict[str, float] = field(default_factory=dict)
    default_span: float = 1.0

    def span_of(self, name: str) -> float:
        return float(self.spans.get(name, self.default_span))

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for name in SEARCHED_FIELDS:
            c = float(self.centre[name])
            s = self.span_of(name)
            out[name] = (c * 10.0 ** -s, c * 10.0 ** s)
        return out


#: Default spans.  Wider on the objective priorities, which are trade-offs with
#: no plant-derived value, and narrower on the integer step prices (taps and,
#: since 2026-10-22, the MSC/MSR banks), which Stage I has already placed from
#: a stated engagement threshold.
DEFAULT_SPANS: Dict[str, float] = {
    "g_q": 1.5,
    "dso_g_v": 1.5,
    "g_w_der": 1.0,
    "g_w_pcc": 1.0,
    "g_w_dso_der": 1.0,
    "g_w_gen": 1.0,
    "g_w_tso_oltc": 0.5,
    "g_w_tso_shunt": 0.5,
    "g_w_dso_oltc": 0.5,
}


def sobol_candidates(box: Box, n: int, seed: int) -> List[Dict[str, float]]:
    """``n`` weight vectors from a scrambled Sobol sequence in log space.

    Owen scrambling makes the design reproducible from ``seed`` alone while
    keeping the low-discrepancy property; ``n`` should be a power of two, which
    is where the Sobol balance properties actually hold.
    """
    from scipy.stats import qmc

    names = list(SEARCHED_FIELDS)
    bounds = box.bounds()
    lo = np.array([math.log10(bounds[k][0]) for k in names])
    hi = np.array([math.log10(bounds[k][1]) for k in names])

    if n & (n - 1):
        raise ValueError(
            f"n = {n} is not a power of two; the Sobol balance properties hold "
            f"on powers of two, and scipy warns rather than silently degrading.")

    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    pts = sampler.random(n)
    scaled = lo + pts * (hi - lo)

    out: List[Dict[str, float]] = []
    for row in scaled:
        w = {GAUGE_FIELD: float(box.centre[GAUGE_FIELD])}
        for k, v in zip(names, row):
            w[k] = float(10.0 ** v)
        out.append(w)
    return out


# ---------------------------------------------------------------------------
#  Metrics the existing extractor does not carry
# ---------------------------------------------------------------------------

def avr_step_metrics(records: Sequence[Any]) -> Dict[str, float]:
    """Smoothness of the generator voltage setpoints.

    ``experiments/ch_10_case_study/ch_10_1_variant_ladder.py::avr_travel``
    reports ``max - min`` over a run, which is a RANGE: a trajectory that
    crosses its band once and a trajectory that oscillates across it every tick
    score identically.  What the requirement is about is per-tick jumpiness, so
    this reports the largest single step and the total variation.
    """
    per_zone: Dict[int, List[np.ndarray]] = {}
    for r in records:
        vg = getattr(r, "zone_v_gen", None) or {}
        for z, v in vg.items():
            v = np.asarray(v, dtype=float)
            if v.size:
                per_zone.setdefault(int(z), []).append(v)

    max_step = 0.0
    total_var = 0.0
    for z, seq in per_zone.items():
        prev = None
        for v in seq:
            if prev is not None and prev.shape == v.shape:
                d = np.abs(v - prev)
                if d.size:
                    max_step = max(max_step, float(d.max()))
                    total_var += float(d.sum())
            prev = v
    return {"avr_max_step_pu": max_step, "avr_total_variation_pu": total_var}


# ---------------------------------------------------------------------------
#  One candidate
# ---------------------------------------------------------------------------

def _per_day(value_per_h: float) -> float:
    return float(value_per_h) * 24.0


def evaluate_candidate(
    weights: Dict[str, float],
    scenarios: Sequence[Any],
    base_cfg: Any,
) -> Dict[str, Any]:
    """Run one weight vector across a bank and summarise it.

    Never raises: a failed window is recorded as such and makes the candidate
    infeasible, which is the correct outcome and not a reason to lose the rest
    of the bank.
    """
    from tuning.metrics import MetricScales
    from tuning.parameters import apply_to_config
    from tuning.runner import run_scenario_with_records

    cfg = apply_to_config(base_cfg, weights)
    scales = MetricScales()

    per_window: Dict[str, Dict[str, float]] = {}
    t0 = time.perf_counter()
    for sc in scenarios:
        result, records = run_scenario_with_records(sc, cfg, scales)
        m = result.metrics
        row: Dict[str, float] = {
            "e_v_ts": float(getattr(m, "rmsd_v_ts", float("nan"))),
            "e_v_ds": float(getattr(m, "rmsd_v_ds", float("nan"))),
            "e_q": float(getattr(m, "itae_q_pcc", float("nan"))),
            "v_band_excess_ts": float(getattr(m, "v_band_excess_ts", 0.0) or 0.0),
            "v_band_excess_ds": float(getattr(m, "v_band_excess_ds", 0.0) or 0.0),
            "n_viol_v_ts": float(getattr(m, "n_viol_v_ts", 0) or 0),
            "n_viol_v_ds": float(getattr(m, "n_viol_v_ds", 0) or 0),
            "v_worst_ts": float(getattr(m, "v_worst_ts", 0.0) or 0.0),
            "v_worst_ds": float(getattr(m, "v_worst_ds", 0.0) or 0.0),
            "tap_ops_per_h_tso": float(getattr(m, "tap_ops_per_h_tso", 0.0) or 0.0),
            "tap_ops_per_h_dso": float(getattr(m, "tap_ops_per_h_dso", 0.0) or 0.0),
            "tap_rev_per_h_tso": float(getattr(m, "tap_reversals_per_h_tso", 0.0) or 0.0),
            "tap_rev_per_h_dso": float(getattr(m, "tap_reversals_per_h_dso", 0.0) or 0.0),
            "pf_failures": float(getattr(m, "pf_failures", 0) or 0),
            "wall_s": float(result.wall_time_s),
        }
        row.update(avr_step_metrics(records))
        row["failed"] = 1.0 if result.failure_reason else 0.0
        row["failure_reason"] = result.failure_reason or ""
        per_window[sc.name] = row

    return {
        "weights": dict(weights),
        "key": weight_fingerprint(weights),
        "per_window": per_window,
        "wall_s": time.perf_counter() - t0,
    }


def aggregate(per_window: Dict[str, Dict[str, Any]],
              exclude: Sequence[str] = ()) -> Dict[str, float]:
    """Bank summary.  Worst-case on wear, quadratic mean on tracking.

    Windows in ``exclude`` -- the zero-capability conditions -- are dropped
    from the ranking aggregate and reported separately.  With no reactive
    capability at either level there is nothing for any weight to allocate, so
    such a window contributes a constant that dilutes every measured effect.
    """
    rows = [v for k, v in per_window.items() if k not in set(exclude)]
    if not rows:
        return {}

    def qmean(key: str) -> float:
        vals = [float(r[key]) for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")

    def worst(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rows]
        return float(max(vals)) if vals else 0.0

    return {
        "e_v_ts": qmean("e_v_ts"),
        "e_v_ds": qmean("e_v_ds"),
        "e_q": qmean("e_q"),
        "worst_tap_ops_per_day_tso": _per_day(worst("tap_ops_per_h_tso")),
        "worst_tap_ops_per_day_dso": _per_day(worst("tap_ops_per_h_dso")),
        "worst_tap_rev_per_day_tso": _per_day(worst("tap_rev_per_h_tso")),
        "worst_tap_rev_per_day_dso": _per_day(worst("tap_rev_per_h_dso")),
        "worst_avr_max_step_pu": worst("avr_max_step_pu"),
        "worst_v_band_excess_ts": worst("v_band_excess_ts"),
        "worst_v_band_excess_ds": worst("v_band_excess_ds"),
        "n_hunting": float(sum(r.get("n_hunting", 0.0) for r in rows)),
        "n_hunting_1h": float(sum(r.get("n_hunting_1h", 0.0) for r in rows)),
        "n_event_explained": float(sum(r.get("n_event_explained", 0.0) for r in rows)),
        "n_viol_v_ts": float(sum(r.get("n_viol_v_ts", 0.0) for r in rows)),
        "n_viol_v_ds": float(sum(r.get("n_viol_v_ds", 0.0) for r in rows)),
        "worst_v_worst_ts": worst("v_worst_ts"),
        "worst_v_worst_ds": worst("v_worst_ds"),
        "pf_failures": worst("pf_failures"),
        "n_failed_windows": sum(1 for r in rows if r.get("failed")),
        # Switched-shunt wear (2026-10-22).  Absent from records written before
        # that date, which is what ``has_shunt_wear`` tells a selection.
        "has_shunt_wear": float(any("shunt_ops_per_h_bank" in r for r in rows)),
        "worst_shunt_ops_per_day_bank": _per_day(worst("shunt_ops_per_h_bank")),
        "n_shunt_hunting": float(sum(r.get("n_shunt_hunting", 0.0) for r in rows)),
        "n_shunt_event_explained": float(
            sum(r.get("n_shunt_event_explained", 0.0) for r in rows)),
        "shunt_commits": float(sum(r.get("shunt_commits", 0.0) for r in rows)),
    }


def check_limits(agg: Dict[str, float], limits: Limits,
                 lambda_max_b: Optional[float] = None,
                 *, wear_is_meaningful: bool = True,
                 e_q_reference: Optional[float] = None,
                 window_hours: Optional[float] = None,
                 ) -> Tuple[bool, List[str]]:
    """``(feasible, reasons)``.  Reasons name the limit, not a score.

    ``wear_is_meaningful`` must be False on the short bank: a per-day tap
    budget cannot be read off a 3 h window, and applying it there rejects every
    candidate rather than discriminating between them.
    """
    bad: List[str] = []
    if agg.get("n_failed_windows", 0) > 0:
        bad.append(f"{int(agg['n_failed_windows'])} window(s) failed to run")
    if agg.get("pf_failures", 0) > limits.allow_pf_failures:
        bad.append(f"power-flow failures {agg['pf_failures']:.0f}")
    if lambda_max_b is not None and lambda_max_b > limits.lambda_max_b:
        bad.append(f"lambda_max(B) {lambda_max_b:.4f} > {limits.lambda_max_b:.3f}")
    n_viol = agg.get("n_viol_v_ts", 0.0) + agg.get("n_viol_v_ds", 0.0)
    if n_viol > limits.max_v_violations:
        bad.append(f"{n_viol:.0f} record(s) outside the hard voltage corridor "
                   f"(TS {agg.get('n_viol_v_ts', 0.0):.0f}, "
                   f"DS {agg.get('n_viol_v_ds', 0.0):.0f})")
    if agg.get("worst_avr_max_step_pu", 0.0) > limits.avr_max_step_pu:
        bad.append(f"AVR step {agg['worst_avr_max_step_pu']:.5f} pu "
                   f"> {limits.avr_max_step_pu:.4f}")
    if e_q_reference is not None and np.isfinite(agg.get("e_q", np.nan)):
        ratio = agg["e_q"] / max(float(e_q_reference), 1e-30)
        if ratio > limits.e_q_max_ratio:
            bad.append(f"interface-Q error {ratio:.2f}x the Stage-I "
                       f"reference (bound {limits.e_q_max_ratio:.1f}x)")
    if wear_is_meaningful:
        # Compared as integer COUNTS over the window, not as per-day rates.
        # Tap operations and reversals are integers; converting them to a rate
        # and back introduces a rounding error the same size as the quantity.
        # Measured 2026-09-11: one reversal in a 12 h window is stored as
        # 0.0834/h, which is 2.0016/day, and a strict comparison against a
        # 2.0/day budget rejected it -- twelve of nineteen candidates were
        # discarded by that artefact alone, which is not a wear finding.
        hours = float(window_hours) if window_hours else 24.0
        for side in ("tso", "dso"):
            ops_n = round(agg.get(f"worst_tap_ops_per_day_{side}", 0.0)
                          / 24.0 * hours)
            ops_max = math.floor(limits.tap_ops_per_day / 24.0 * hours + 1e-9)
            if ops_n > ops_max:
                bad.append(f"{side.upper()} {ops_n} tap ops per {hours:g} h > "
                           f"{ops_max} allowed "
                           f"({limits.tap_ops_per_day:.0f}/day)")
        # Hunting is counted directly, not derived from a per-day rate: what
        # the requirement forbids is a tap reversing within a couple of hours
        # with no new disturbance to explain it.  A reversal that IS explained
        # by an event -- a tap that moves on a trip and moves back on the
        # restore -- is correct behaviour and is not counted here.
        n_hunt = agg.get("n_hunting", 0.0)
        if n_hunt > limits.max_hunting:
            bad.append(f"{n_hunt:.0f} hunting reversal(s) "
                       f"(< {limits.hunt_dwell_h:g} h dwell, unexplained by an "
                       f"event) > {limits.max_hunting} allowed")
        # Switched-shunt wear, by the same two rules as the taps: a per-bank
        # count against a daily budget, and hunting with event attribution.
        if limits.shunt_ops_per_day is not None:
            sh_n = round(agg.get("worst_shunt_ops_per_day_bank", 0.0)
                         / 24.0 * hours)
            sh_max = math.floor(limits.shunt_ops_per_day / 24.0 * hours + 1e-9)
            if sh_n > sh_max:
                bad.append(f"bank {sh_n} commits per {hours:g} h > {sh_max} "
                           f"allowed ({limits.shunt_ops_per_day:g}/day)")
        n_sh_hunt = agg.get("n_shunt_hunting", 0.0)
        if n_sh_hunt > limits.max_shunt_hunting:
            bad.append(f"{n_sh_hunt:.0f} bank hunting reversal(s) "
                       f"(< {limits.hunt_dwell_h:g} h dwell, unexplained by an "
                       f"event) > {limits.max_shunt_hunting} allowed")
    return (not bad), bad


def rank_lexicographic(
    feasible: Sequence[Dict[str, Any]],
    *,
    indifference: float = 0.02,
) -> List[Dict[str, Any]]:
    """Order on ``e_v(TS)``, then on ``e_v(DS)`` within an indifference band.

    The band is what stops the order being decided by a difference too small to
    be a difference: without it a candidate 0.1 % better on the first criterion
    wins outright regardless of the second, which is the standard objection to
    a lexicographic order and is answered by stating the band rather than by
    abandoning the method.
    """
    if not feasible:
        return []
    rows = sorted(feasible, key=lambda r: r["aggregate"]["e_v_ts"])
    best = rows[0]["aggregate"]["e_v_ts"]
    tied = [r for r in rows
            if r["aggregate"]["e_v_ts"] <= best * (1.0 + indifference)]
    rest = [r for r in rows if r not in tied]
    tied.sort(key=lambda r: r["aggregate"]["e_v_ds"])
    return tied + rest


# ---------------------------------------------------------------------------
#  Hunting: short-cycle tap reversals not explained by a disturbance
# ---------------------------------------------------------------------------

#: Dwell below which an opposite-direction tap pair counts as hunting [s].
#: Two hours: far above the ~21 min limit cycles measured 2026-09-11 (seven
#: supervisory periods -- the integer cooldown expiring and the tap immediately
#: reversing), and far below a diurnal load cycle, so it cannot mistake
#: legitimate daily tracking for hunting.
HUNT_DWELL_S: float = 2 * 3600.0


def tap_moves(records: Sequence[Any]) -> List[Tuple[float, str, int]]:
    """Every committed tap move as ``(t_s, transformer_key, direction)``.

    Covers both layers: ``zone_oltc_taps`` (machine transformers, per zone) and
    ``dso_trafo_tap_pos`` (subordinate coupling transformers, per transformer).
    Persisting this list is what lets a wear definition be revised without
    re-simulating -- the first Stage II-b discarded its records after
    extracting metrics and could not be re-scored.
    """
    out: List[Tuple[float, str, int]] = []
    for attr in ("zone_oltc_taps", "dso_trafo_tap_pos"):
        prev: Dict[Any, np.ndarray] = {}
        for r in records:
            d = getattr(r, attr, None) or {}
            for k, v in d.items():
                arr = np.atleast_1d(np.asarray(v, dtype=float))
                if k in prev and prev[k].shape == arr.shape:
                    for i, dv in enumerate(arr - prev[k]):
                        if abs(dv) > 0.5:
                            out.append((float(r.time_s), f"{k}[{i}]",
                                        int(np.sign(dv))))
                prev[k] = arr
    return sorted(out)


#: Event onsets closer together than this are one disturbance, not several.
#: A load ramp is delivered as a sequence of steps ~20 min apart; treating each
#: step as a fresh disturbance would exempt every hunting cycle that happens to
#: straddle one, and a monotonic ramp is precisely the case where a tap has no
#: business reversing.  One hour keeps a generator trip and its restore -- which
#: ARE distinct events -- separate.
EVENT_MERGE_S: float = 30 * 60.0


def merge_events(events_s: Sequence[float],
                 merge_s: float = EVENT_MERGE_S) -> List[float]:
    """Collapse a burst of onsets into the disturbances they represent."""
    out: List[float] = []
    for t in sorted(float(t) for t in events_s):
        if not out or (t - out[-1]) > merge_s:
            out.append(t)
    return out


def hunting_reversals(
    moves: Sequence[Tuple[float, str, int]],
    events_s: Sequence[float] = (),
    *,
    dwell_s: float = HUNT_DWELL_S,
    tol_s: float = 1.0,
) -> Tuple[List[dict], List[dict]]:
    """Split short reversals into ``(hunting, event_explained)``.

    A **reversal** is two consecutive opposite-direction moves on the same tap.
    It counts as **hunting** when the dwell between them is under ``dwell_s``
    *and* no disturbance arrives in between.

    That second clause is the whole point.  A tap that moves on a generator
    trip and moves back when the unit is restored has responded to two distinct
    events and is behaving correctly; a tap that moves and returns with nothing
    having happened in between is chasing its own tail.  Attributing by
    proximity to *any* event instead -- exempting a reversal because its first
    move happened near a disturbance -- hides exactly the case worth catching,
    measured 2026-09-11: it reported zero hunting on a run containing three
    21-minute limit cycles.

    Slow reversals are not counted at any dwell: a tap that moves once in the
    morning and back in the evening is following load, not hunting.
    """
    events_s = merge_events(events_s)

    per: Dict[str, List[Tuple[float, int]]] = {}
    for t, key, s in moves:
        per.setdefault(key, []).append((float(t), int(s)))

    hunt: List[dict] = []
    legit: List[dict] = []
    for key, seq in per.items():
        seq.sort()
        for (t0, s0), (t1, s1) in zip(seq, seq[1:]):
            if s1 == s0 or (t1 - t0) > dwell_s:
                continue
            new_event = any(t0 + tol_s < te <= t1 + tol_s for te in events_s)
            row = {"transformer": key, "t0_s": t0, "t1_s": t1,
                   "dwell_s": t1 - t0}
            (legit if new_event else hunt).append(row)
    return hunt, legit


def hunting_summary(records: Sequence[Any], events_s: Sequence[float] = (),
                    *, dwell_s: float = HUNT_DWELL_S) -> Dict[str, Any]:
    """Hunting counts plus the raw move list, for one window."""
    moves = tap_moves(records)
    hunt, legit = hunting_reversals(moves, events_s, dwell_s=dwell_s)
    hunt_1h, _ = hunting_reversals(moves, events_s, dwell_s=3600.0)
    return {
        "n_hunting": len(hunt),
        "n_hunting_1h": len(hunt_1h),
        "n_event_explained": len(legit),
        "n_moves": len(moves),
        "hunting_detail": hunt,
        # Persisted so a revised wear definition can be applied without
        # re-simulating.  A few hundred bytes per window.
        "moves": [[t, k, s] for t, k, s in moves],
        "events_s": list(events_s),
    }


def shunt_summary(records: Sequence[Any],
                  events_s: Sequence[float] = ()) -> Dict[str, Any]:
    """Switched-shunt commits: counts, short reversals, and bank hunting.

    Since 2026-10-22 a bulk MSC/MSR bank is an integer input of the TS-MIQP,
    priced by ``g_w_tso_shunt`` like a tap by its tap price, so its wear is
    judged like a tap's:

    * ``shunt_ops_max_bank`` -- commits of the busiest single bank in the
      window, the quantity a per-bank daily budget bounds;
    * ``n_shunt_hunting`` -- opposite-direction commit pairs on one bank within
      :data:`HUNT_DWELL_S` with no new disturbance in between, by the same
      event attribution as the taps (:func:`hunting_reversals`).

    ``shunt_reversals`` is the raw count of the earlier definition (2 h dwell,
    NOT event-attributed), from when the banks were dispatched by an integrator
    outside the MIQP that should have averaged every transient away.  Kept as a
    diagnostic; the filter uses ``n_shunt_hunting``.
    """
    per: Dict[Any, List[Tuple[float, np.ndarray]]] = {}
    for r in records:
        d = getattr(r, "zone_tso_shunt_states", None) or {}
        for z, v in d.items():
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            if arr.size:
                per.setdefault(z, []).append((float(r.time_s), arr))

    n_up = n_dn = n_rev = 0
    moves: List[List[Any]] = []
    max_level = 0.0
    for z, seq in per.items():
        prev_t, prev = seq[0]
        last_dir: Dict[int, Tuple[float, int]] = {}
        for t, arr in seq[1:]:
            if arr.shape == prev.shape:
                for i, dv in enumerate(arr - prev):
                    if abs(dv) > 0.5:
                        s = int(np.sign(dv))
                        n_up += int(s > 0)
                        n_dn += int(s < 0)
                        moves.append([t, f"z{z}[{i}]", s])
                        if i in last_dir:
                            t0, s0 = last_dir[i]
                            if s != s0 and (t - t0) <= HUNT_DWELL_S:
                                n_rev += 1
                        last_dir[i] = (t, s)
            max_level = max(max_level, float(np.max(np.abs(arr))))
            prev = arr

    per_bank: Dict[str, int] = {}
    for _t, key, _s in moves:
        per_bank[key] = per_bank.get(key, 0) + 1
    hunt, legit = hunting_reversals(
        [(float(t), str(k), int(s)) for t, k, s in moves], events_s)
    return {"shunt_commits": n_up + n_dn, "shunt_up": n_up, "shunt_down": n_dn,
            "shunt_reversals": n_rev, "shunt_max_level": max_level,
            "shunt_ops_max_bank": max(per_bank.values(), default=0),
            "n_shunt_hunting": len(hunt),
            "n_shunt_event_explained": len(legit),
            "shunt_hunting_detail": hunt,
            "shunt_moves": moves}
