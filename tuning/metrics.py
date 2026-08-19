"""
tuning/metrics.py
=================
Trajectory metric extraction and composite cost J(g_w) for offline
controller tuning.

Re-uses :mod:`experiments.helpers.comparison_metrics` for voltage
envelopes, RMSDs, violation counts, losses, and Q-headroom.  Adds:

* ITAE (integral of time-weighted absolute tracking error)
* Per-actuator-class oscillation counts with noise floors
* Tap-switch counts (TSO + DSO)
* Empirical contraction percentile from
  :attr:`MultiTSOIterationRecord.zone_contraction_lhs`
* Power-flow failure detection
* Composite cost ``J = sum_i w_i * normalised_metric_i``

Cost weights (:class:`CostWeights`) are an *intentionally separate*
concept from controller weights (``g_v``, ``g_q``, ``g_w_*``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from configs.config import MultiTSOConfig
from experiments.helpers.comparison_metrics import (
    loss_series,
    voltage_envelope_ds,
    voltage_envelope_ts,
    voltage_rms_err_all,
    voltage_rmsd_ds,
    voltage_rmsd_ts,
    voltage_violation_counts_ds,
    voltage_violation_counts_ts,
)
from experiments.helpers.records import MultiTSOIterationRecord


# ---------------------------------------------------------------------------
# Cost weights (META-tuning weights, NOT controller g_v / g_q / g_w_*)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostWeights:
    """Composite-cost weights — distinct from
    :attr:`MultiTSOConfig.g_v` / ``g_q`` / ``g_w_*``.

    Defaults are calibrated so that each normalised metric is order one
    under nominal conditions, with priority

        ``pf_failure >> violation > tso_v_track > dso_v_track
         > pcc_underutil > q_track ≈ q_tie_track ≈ oscillation
         > tap_switch``.

    Rationale (revised 2026-05-02; see ``00_daily_log`` for context):
    the previous (2026-04-29) revision promoted ``w_v_track_ts`` above
    ``w_q_track`` to discourage sluggish controllers, but the BO still
    favoured operating points with ``g_w_pcc`` near its ceiling
    (e.g. ``g_w_pcc ≈ 269.7`` on the most recent run).  Root cause:
    ``itae_q_pcc`` measures *internal cascade-coupling fidelity*, not
    a controlled output — the cheapest way to drive it down is to
    freeze the PCC setpoint (high ``g_w_pcc``), which trivially makes
    the DSO catch up but suppresses genuine DSO reactive-power
    support.  Fix:

      (1) demote ``w_q_track`` (6.0 → 1.0) so PCC tracking is a soft
          regulariser, not a primary KPI;
      (2) add ``w_pcc_underutil`` — a *conditional* term that
          penalises idle DSO PCC injection while the TSO voltage is
          stressed.  This gives J an explicit reason to value DSO
          support;
      (3) bump ``w_v_track_ts`` (26.0 → 35.0) to keep its baseline
          contribution rank intact after step (1).

    The earlier 2026-04-29 note (history): a still-earlier
    ``viol > osc > tap > tracking`` ordering yielded operating points
    with very heavy proximal damping (``g_w_pcc``, ``g_w_dso_der`` near
    their ceilings) because tracking errors were cheap and any
    actuator activity was expensive.  That revision moved tracking
    above wear; this revision additionally moves DSO *utilisation*
    above PCC tracking.

    The ``g_w_pcc`` BO upper bound is also capped at 30 in
    ``tuning/parameters.py`` as a safety rail — values above ~30 are
    sluggish without a meaningful end-performance benefit.
    """

    # Tracking / utilisation weights, ranked
    # TSO_V > DSO_V > PCC_underutil > DSO_Q ≈ Q_tie.
    #
    # Calibration philosophy: scales below correspond to **physical
    # engineering tolerances** (5 mpu sustained voltage error, 5 Mvar
    # sustained Q-PCC error, etc.) so ``norm = 1`` means the operating
    # point is at the edge of acceptable.  Weights are sized so that
    # the baseline-trial contribution ranking matches the priority
    # ranking — see the per-line comments below for the target
    # baseline contributions.
    w_v_track_ts:    float = 35.0   # 35 × 0.38 ≈ 13   (primary KPI)
    w_v_track_ds:    float =  4.5   # 4.5 × 0.64 ≈ 3
    w_pcc_underutil: float =  3.0   # explicit DSO-utilisation term
    w_q_track:       float =  1.0   # internal coupling, soft regulariser
    w_q_tie_track:   float =  1.0   #  1 × 0.96 ≈ 1

    # Calibration knobs for the conditional DSO-underutilisation
    # metric (see ``_itae_pcc_underutilization``).  Kept on
    # :class:`CostWeights` so the meta-tuning surface stays in one
    # place.
    pcc_underutil_v_deadband:  float =   0.005  # pu — voltage error
                                                # below this → no
                                                # DSO action expected
    pcc_underutil_q_ref_mvar:  float = 100.0    # Mvar — reference
                                                # "useful DSO PCC
                                                # injection" magnitude

    # Actuator-wear weights, intentionally below the primary tracking
    # weights so that BO does not prefer a sluggish controller over a
    # tracking one.  ``w_tap`` lowered to compensate for the unscaled
    # ``norm_tap = (n_tap_tso + n_tap_dso) / 5`` magnitudes (~8 at
    # baseline).
    w_osc:         float = 1.0
    w_tap:         float = 0.05

    # Operational-safety / catastrophe weights, dominate everything.
    # ``w_viol`` is large because ``norm_viol`` (mean voltage band-edge
    # excess in pu/step) is small in absolute terms (typically 0–0.05);
    # the high weight makes a 1 % pu mean excess give ~10 cost units.
    w_viol:        float = 1000.0
    w_pf:          float = 100.0


# ---------------------------------------------------------------------------
# Oscillation noise floors per actuator class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseFloors:
    """Minimum ``|Δu|`` at which a sign change in ``Δu`` counts as an
    oscillation.

    Below these thresholds, sign changes are treated as numerical
    noise.  Defaults are physical-reasoning based; override per
    experiment if actuator scales differ.
    """

    der_q_mvar: float = 20.0    # ~1 % of typical wind-park rating (500 MW)
    pcc_q_mvar: float = 10.0    # PCC tracking is small-signal
    v_gen_pu:   float = 0.005  # 0.1 % voltage
    oltc_step:  float = 1.0    # one full tap step always counts


# ---------------------------------------------------------------------------
# Normalisation scales — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricScales:
    """Divisors turning raw metrics into ``O(1)`` normalised quantities.

    Previously these constants were duplicated in :func:`cost_components` and
    :func:`extract_metrics` and had to be kept in sync by hand; a divergence
    between the two silently made ``cost_components()`` disagree with
    ``cost_J``.  They now live here and both call sites consume this object.

    Scales are **engineering tolerances**, i.e. ``norm = 1`` means "at the edge
    of acceptable".  They are calibrated once, from a random-prior sample, and
    never from optimiser output -- see ``docs/tuning/`` for the rationale.
    """

    # Calibrated 2026-08-04 from a 65-draw Sobol sample of the *prior*
    # (tuning.scripts.calibrate_metrics --n-draws 64 --n-scenarios 3, all draws
    # 3/3 feasible).  Values are the sample **p90**, not the median.
    #
    # Why p90: a scale is an engineering *tolerance* -- ``norm = 1`` means "at
    # the edge of acceptable" (see the class docstring), which is a high
    # quantile, not a typical value.  Normalising by the median instead makes a
    # heavy-tailed term explode exactly when its tail is realised, and the
    # aggregator deliberately looks at the tail.  Two terms are extremely
    # heavy-tailed in this sample -- ``pcc_underutil`` median 2.35 vs p90 1286.9
    # (547x) and ``v_band_excess`` 0.00042 vs 0.0247 (59x) -- and with median
    # scales they became 71 % and 25 % of the whole objective, i.e. 96 % between
    # them, concentrated in one scenario.  ``calibrate_metrics`` reports both
    # quantiles; take p90.
    #
    # Median values are kept in the trailing comments for reference.
    #
    # Voltage quality, measured against ``v_setpoint_pu`` (not against the
    # hard corridor, which is almost never touched).
    v_rms_ts:      float = 0.020104   # pu, TS spatial RMS   (median 0.0075014)
    v_rms_ds:      float = 0.014107   # pu, DS spatial RMS   (median 0.0047005)
    v_worst_ts:    float = 0.054357   # pu, p95 worst bus TS (median 0.02385)
    v_worst_ds:    float = 0.096455   # pu, p95 worst bus DS (median 0.052996)
    v_quality_band: float = 0.020   # pu, half-width of the inner quality band
    v_band_excess: float = 0.024673   # pu/step beyond band  (median 0.00041973)

    # Interface tracking.
    q_pcc:         float = 33658.0    # min * Mvar  (median 7878.9)
    q_tie:         float = 202390.0   # min * Mvar  (median 69144.0)
    #: min * pu * Mvar.  The original hand-set 1400.0 was close to this p90 and
    #: was therefore *correct*; the median-based 2.3499 briefly replaced it and
    #: made this term 71 % of the objective.
    pcc_underutil: float = 1286.9

    # Actuator activity.
    osc_pct:       float = 1.0      # percent of step-pairs that flipped sign
    tap_ops_tso:   float = 6.0      # tap operations per hour per MT transformer
    tap_ops_dso:   float = 6.0      # tap operations per hour per NC transformer
    tap_reversals: float = 2.0      # reversals per hour per transformer

    # Legacy horizon-ITAE scales, kept so the pre-existing scalar path and its
    # regression tests stay reproducible.  Superseded by the window-normalised
    # metrics; do not use for new terms.
    itae_v_ts:     float = 14.0     # min * pu
    itae_v_ds:     float = 30.0     # min * pu


#: Finite stand-in for "this run is inadmissible".  Deliberately far above any
#: achievable finite cost (the largest observed across 1555 historical runs was
#: ~1.7e3) yet finite, because samplers handle ``inf`` poorly.  Feasibility is
#: reported separately via :attr:`TrajectoryMetrics.infeasible_reason`; this
#: value only keeps the scalar path well-ordered.
INFEASIBLE_SENTINEL: float = 1.0e6


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryMetrics:
    """All metrics extracted from one closed-loop log.

    Every field has a default so that new fields can be appended without
    breaking the explicit-keyword constructions in the test-suite.
    """

    # tracking
    itae_v_ts:           float = 0.0
    itae_v_ds:           float = 0.0
    rmsd_v_ts:           float = 0.0
    rmsd_v_ds:           float = 0.0
    itae_q_pcc:          float = 0.0
    itae_q_tie:          float = 0.0
    itae_pcc_underutil:  float = 0.0

    # constraint health
    n_viol_v_ts:         int = 0
    n_viol_v_ds:         int = 0
    voltage_excess_pu:   float = 0.0   # smooth hard-corridor excess

    # ── Guard band: "stay off the bound", not "do not cross it" ──────────
    guard_deficit_ds_pu: float = 0.0
    """Mean per-step excess of the DSO-group envelope beyond a corridor
    shrunk by :data:`DS_GUARD_HEADROOM_PU` at both ends [pu/step].

    ``voltage_excess_pu`` is a zero-margin barrier: it is exactly ``0`` for a
    controller that rides the bound at 1.0999, which is precisely the
    "feasible but undesired" state a cascaded DSO settles into (the MIQP output
    constraint holds V at the bound and the OLTC then has no gradient left --
    see docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md).  This field
    is the same smooth ramp evaluated against ``[v_lo + h, v_hi - h]``, so it
    starts charging *before* the bound is reached and can express a headroom
    requirement.

    Deliberately an integral over time rather than ``min_t headroom``: a
    worst-case statistic has between-bank variability larger than its in-sample
    margin (the lambda* transfer defect, docs/tuning/METHOD_weight_selection.md).

    The ramp shape matters for the *current* search too, though not for the
    reason ``_voltage_band_excess`` gives -- that docstring predates
    ``tuning_mc`` and argues from TPE's Parzen kernels.  ``tuning_mc`` uses a
    compass search under an Audet-Dennis filter, where a step function is bad
    for a different reason: a poll would flip between "no signal at all" and a
    discontinuous jump, so a direction reads as dead until it suddenly is not,
    and the dominance test inherits the discontinuity.  A ramp gives every poll
    a usable gradient in the criterion."""

    ds_headroom_min_pu:  float = float("nan")
    """``min over time and DSO groups of min(v_hi - V_max, V_min - v_lo)`` [pu].
    Negative means the corridor was left.  Reported diagnostic only -- the
    scored quantity is :attr:`guard_deficit_ds_pu`, for the reason above."""

    # actuator activity
    n_osc_der:           int = 0
    n_osc_pcc:           int = 0
    n_osc_v_gen:         int = 0
    n_tap_switches_tso:  int = 0
    n_tap_switches_dso:  int = 0
    osc_rate:            float = 0.0   # rate in [0, 1] used in cost

    # stability
    rho_emp_p95:         float = 0.0
    pf_failures:         int = 0

    # losses (diagnostic only — not in J by default)
    losses_mean_mw:      float = 0.0

    # composite
    cost_J:              float = 0.0

    # bookkeeping
    n_records:           int = 0
    n_tso_active:        int = 0
    n_dso_active:        int = 0

    # ── Voltage quality against v_setpoint_pu (see ``_v_quality``) ──────────
    v_rms_ts:            float = 0.0   # pu, time-mean spatial RMS deviation
    v_rms_ds:            float = 0.0
    v_worst_ts:          float = 0.0   # pu, p95 worst-bus deviation
    v_worst_ds:          float = 0.0
    v_band_excess_ts:    float = 0.0   # pu, mean excess beyond inner band
    v_band_excess_ds:    float = 0.0

    # ── Per-transformer actuator wear (worst transformer, not fleet sum) ────
    tap_ops_per_h_tso: float = 0.0
    tap_ops_per_h_dso: float = 0.0
    tap_reversals_per_h_tso: float = 0.0
    tap_reversals_per_h_dso: float = 0.0

    # ── Feasibility (consumed by the constrained objective) ────────────────
    infeasible_reason:   str = ""
    """Empty when the run is admissible.  Non-empty values name the first
    failing check (``"pf_failure"``, ``"empty_log"``, ...).  Kept separate from
    :attr:`cost_J` so feasibility can be a *constraint* rather than a term in a
    weighted sum -- mixing them is what let divergence undercut poor-but-stable
    operation."""

    duration_s:          float = 0.0
    """Simulated horizon, needed to express switching as a rate."""

    @property
    def feasible(self) -> bool:
        """True when no admissibility check failed."""
        return not self.infeasible_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _itae(t_min: NDArray[np.float64], abs_err: NDArray[np.float64]) -> float:
    """Integral of time-weighted absolute error: ``∫ t·|e| dt``.

    ``t_min`` is in minutes (matching the
    :mod:`comparison_metrics` convention).  Returns ITAE in
    ``minute · pu`` (or ``minute · Mvar``).  NaN entries are dropped.

    Returns ``nan`` -- **not** ``0.0`` -- when fewer than two finite samples
    survive.  Returning ``0.0`` made a fully diverged (all-NaN) trajectory look
    like *perfect* tracking: combined with ``_normalise(0.0) == 0.0`` it gave a
    diverged run a cost of exactly ``w_pf``, which historically undercut 35-43 %
    of *converged* runs and made divergence a rewarded search direction.
    ``nan`` propagates through :func:`_normalise` to ``inf`` instead.
    """
    if t_min.size < 2:
        return float("nan")
    integrand = t_min * abs_err
    finite = np.isfinite(integrand) & np.isfinite(t_min)
    if int(finite.sum()) < 2:
        return float("nan")
    # ``np.trapezoid`` replaces ``np.trapz`` in NumPy 2.x, which *removes*
    # the old name -- so the fallback must stay unevaluated on 2.x, i.e. no
    # ``getattr(np, "trapezoid", np.trapz)``: that raises AttributeError
    # while computing the default.
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz_fn(integrand[finite], t_min[finite]))


def _stack_dict_arrays(
    d_list: List[dict[Any, Any]],
) -> NDArray[np.float64]:
    """Stack a list of dict-of-arrays into a 2-D array
    ``(steps × actuators)``.

    Missing keys → that column is NaN for that step.  Empty dicts
    produce all-NaN rows.  Returns shape ``(T, N)`` where ``N`` is the
    union of all keys and column widths come from the first non-empty
    occurrence of each key.
    """
    if not d_list:
        return np.zeros((0, 0))
    keys = sorted({k for d in d_list for k in d.keys()})
    if not keys:
        return np.full((len(d_list), 0), np.nan)

    widths: dict[Any, int] = {}
    for k in keys:
        for d in d_list:
            if k in d and d[k] is not None:
                arr = np.atleast_1d(np.asarray(d[k]))
                if arr.size > 0:
                    widths[k] = int(arr.size)
                    break
        else:
            widths[k] = 1

    cols: list[NDArray[np.float64]] = []
    for k in keys:
        col = np.full((len(d_list), widths[k]), np.nan)
        for i, d in enumerate(d_list):
            v = d.get(k)
            if v is None:
                continue
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            n = min(int(arr.size), widths[k])
            col[i, :n] = arr[:n]
        cols.append(col)
    return np.hstack(cols)


def _count_oscillations(
    u_seq: NDArray[np.float64],
    noise_floor: float,
) -> int:
    """Count sign changes in ``Δu`` where ``|Δu| > noise_floor``.

    ``u_seq`` has shape ``(T, N)``.  One count per ``(step, actuator)``
    pair where ``sign(Δu(k)) ≠ sign(Δu(k-1))`` AND both ``|Δu(k)|``,
    ``|Δu(k-1)|`` exceed ``noise_floor``.  NaN values do not flip.
    """
    if u_seq.size == 0 or u_seq.shape[0] < 3:
        return 0
    du = np.diff(u_seq, axis=0)
    sig = np.abs(du) > noise_floor
    sgn = np.sign(du)
    sgn[~sig] = 0          # below-threshold treated as zero (no flip)
    sgn = np.nan_to_num(sgn, nan=0.0)

    flips = (sgn[1:] != sgn[:-1]) & (sgn[1:] != 0) & (sgn[:-1] != 0)
    return int(np.sum(flips))


def _decimate_to_ticks(
    seq: NDArray[np.float64],
    active: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Keep only the rows where the issuing controller actually stepped.

    Commands are logged once per plant step but only *change* on controller
    ticks.  With ``dt_s=20`` and ``tso_period_s=180`` each TSO command is held
    for 9 records, so the raw difference sequence is
    ``[0, ..., 0, Δ, 0, ..., 0, Δ, ...]``.  :func:`_count_oscillations` requires
    two *adjacent* above-floor deltas, so on the held sequence it returns 0 for
    every trajectory -- which is exactly what was observed empirically
    (``norm_osc`` identically zero across 1555 scenario-runs).  Decimating to
    the tick grid first restores the intended semantics.

    Rows whose ``active`` flag is False are dropped.  If the flag is never set
    (older logs), the sequence is returned unchanged so behaviour degrades
    gracefully rather than silently emptying.
    """
    if seq.size == 0 or active.size == 0:
        return seq
    n = min(seq.shape[0], int(active.size))
    mask = active[:n]
    if not bool(mask.any()):
        return seq
    return seq[:n][mask]


def _count_tap_switches(taps_seq: NDArray[np.float64]) -> int:
    """Sum of ``|Δtap|`` across all ``(step, actuator)`` pairs."""
    if taps_seq.size == 0 or taps_seq.shape[0] < 2:
        return 0
    dtaps = np.diff(taps_seq, axis=0)
    return int(np.nansum(np.abs(dtaps)))


def _count_tap_reversals(taps_seq: NDArray[np.float64]) -> int:
    """Count direction changes in the tap sequence, per actuator.

    A *reversal* is a sign change between consecutive non-zero tap movements,
    ignoring the intervening hold periods.  This is the hunting indicator, and
    unlike :func:`_count_oscillations` it is immune to the hold-padding of
    :func:`_decimate_to_ticks` because zero moves are skipped rather than
    treated as sign-zero neighbours.
    """
    if taps_seq.size == 0 or taps_seq.shape[0] < 3:
        return 0
    total = 0
    dtaps = np.diff(taps_seq, axis=0)
    for col in range(dtaps.shape[1]):
        moves = dtaps[:, col]
        moves = moves[np.isfinite(moves) & (moves != 0.0)]
        if moves.size < 2:
            continue
        signs = np.sign(moves)
        total += int(np.sum(signs[1:] != signs[:-1]))
    return total


def _tap_wear(
    taps_seq: NDArray[np.float64],
    duration_s: float,
) -> tuple[float, float]:
    """``(max ops/hour, max reversals/hour)`` over the individual transformers.

    Wear limits are *per transformer*, so aggregating over the fleet (as the
    legacy ``norm_tap = (n_tso + n_dso)/5`` did) hides a single hunting tap
    changer behind a quiet fleet -- and makes the two OLTC weights enter the
    cost only through their sum, which is half the reason they were never
    identifiable.  The worst transformer is what an asset owner constrains.

    **Per hour, deliberately not per day.**  An operational maintenance budget
    is naturally quoted per day, and this metric was first written that way --
    but the design scenarios are event-dense 75-minute windows, and a real day
    is mostly quiet.  Extrapolating the one to the other inflates the figure by
    roughly the ratio of stressed to total time: the hand-tuned reference makes
    5.0 taps in a 75-min disturbance window, an unremarkable 4.0 taps/hour, and
    the same number reads as a pathological "96 ops/day".  Reporting the rate
    over the window actually simulated keeps the quantity honest; convert to a
    daily budget only against a representative daily profile, never against a
    design scenario.
    """
    if taps_seq.size == 0 or taps_seq.shape[0] < 2 or duration_s <= 0.0:
        return 0.0, 0.0
    hours = duration_s / 3600.0
    dtaps = np.diff(taps_seq, axis=0)
    ops = np.nansum(np.abs(dtaps), axis=0)                 # per transformer
    max_ops_per_h = float(np.max(ops) / hours) if ops.size else 0.0
    rev = [
        _count_tap_reversals(taps_seq[:, [c]])
        for c in range(taps_seq.shape[1])
    ]
    max_rev_per_h = float(max(rev) / hours) if rev else 0.0
    return max_ops_per_h, max_rev_per_h


def _detect_pf_failures(records: List[MultiTSOIterationRecord]) -> int:
    """Count records where any zone or DSO group reports non-finite
    voltage (PF divergence).

    An empty log when a scenario was requested counts as one failure.
    """
    if not records:
        return 1
    n = 0
    for r in records:
        for v in (
            *r.zone_v_min.values(),
            *r.zone_v_max.values(),
            *r.zone_v_mean.values(),
            *r.dso_group_v_min_pu.values(),
            *r.dso_group_v_max_pu.values(),
            *r.dso_group_v_mean_pu.values(),
        ):
            if v is not None and not math.isfinite(v):
                n += 1
                break
    return n


DS_GUARD_HEADROOM_PU: float = 0.02
"""Required headroom to each hard bound for :attr:`TrajectoryMetrics.guard_deficit_ds_pu`.

The corridor is [0.90, 1.10]; at ``h = 0.02`` the guard band is [0.92, 1.08], so
an HV network is charged as soon as any of its buses comes within 2 % of a bound.

**Not calibratable from the hand-tuned reference.**  ``ConstraintLimits`` derives
``corridor_excess_pu`` via ``from_reference`` (measure the reference, add margin),
but that procedure is unavailable here: measured 2026-08-18, the reference has
*negative* headroom on two of four DSOs (DSO_2 -0.0012, DSO_4 -0.0010 pu), so
calibrating from it would enshrine the defect.  This value is a design intent --
2 % of nominal is the usual planning margin on a 110 kV network -- and must be
justified as such, not fitted.
"""


def _ds_headroom_min(
    records: List[MultiTSOIterationRecord],
    low: float = 0.9,
    high: float = 1.1,
) -> float:
    """``min`` over time and DSO groups of the distance to the nearer bound."""
    worst = math.inf
    for r in records:
        for group, v_max in r.dso_group_v_max_pu.items():
            v_min = r.dso_group_v_min_pu.get(group)
            if v_max is None or v_min is None:
                continue
            if not (math.isfinite(v_max) and math.isfinite(v_min)):
                continue
            worst = min(worst, high - float(v_max), float(v_min) - low)
    return worst if math.isfinite(worst) else float("nan")


def _voltage_band_excess(
    records: List[MultiTSOIterationRecord],
    low: float = 0.9,
    high: float = 1.1,
    *,
    groups: str = "all",
) -> float:
    """Sum over time and zones/groups of the smooth band-edge excess.

    Per record, contributes ``max(V_max - high, 0) + max(low - V_min, 0)``
    for each TSO zone and DSO group.  Inside the band the contribution is
    exactly zero (no cliff); outside, it grows linearly with how far the
    voltage strays.  Returned in pu·step (sum over records).

    Used as the smooth replacement for the binary violation count in the
    cost — TPE's Parzen kernels handle ramps far better than step
    functions, which the previous ``n_viol / len(records)`` formulation
    introduced at the band boundary.

    ``low``/``high`` should be passed from ``cfg.v_min_pu`` / ``cfg.v_max_pu``
    rather than left at the defaults, which merely happen to match the current
    config.  Note this quantity is near-zero for any sane operating point (the
    corridor is +/-7-8 % around a 1.03 pu setpoint) -- it belongs in the
    feasibility constraints, not in the cost.  The graded quality signal is
    :func:`_v_quality`.
    """
    excess = 0.0
    want_ts = groups in ("all", "ts")
    want_ds = groups in ("all", "ds")
    for r in records:
        if want_ts:
            for v_max in r.zone_v_max.values():
                if v_max is not None and math.isfinite(v_max):
                    excess += max(float(v_max) - high, 0.0)
            for v_min in r.zone_v_min.values():
                if v_min is not None and math.isfinite(v_min):
                    excess += max(low - float(v_min), 0.0)
        if want_ds:
            for v_max in r.dso_group_v_max_pu.values():
                if v_max is not None and math.isfinite(v_max):
                    excess += max(float(v_max) - high, 0.0)
            for v_min in r.dso_group_v_min_pu.values():
                if v_min is not None and math.isfinite(v_min):
                    excess += max(low - float(v_min), 0.0)
    return excess


def _v_quality(
    records: List[MultiTSOIterationRecord],
    v_set: float,
    quality_band: float,
) -> dict[str, float]:
    """Voltage quality **against ``v_set``**, not against the hard corridor.

    The hard corridor is ``[0.90, 1.10]`` around a setpoint of ``1.03``, so band
    excess is essentially never observed and carries no information for tuning.
    It remains valuable as a *constraint* (an always-satisfied constraint is free
    insurance) but it cannot be the quality signal.  These three quantities are
    the quality signal:

    ``v_rms_ts`` / ``v_rms_ds``
        Time-mean of the **spatial RMS** deviation from ``v_set``, read from
        ``zone_v_rms_err_pu`` via
        :func:`~experiments.helpers.comparison_metrics.voltage_rms_err_per_zone`.
        The legacy metric used the spatial *mean* voltage, under which a zone
        half at 1.00 pu and half at 1.06 pu scores as perfect.  Also identical
        to ``rms_v_ts_pu`` in ``cigre_summary_table``, so the tuning objective
        and the reported thesis metric are the same quantity.

    ``v_worst_ts`` / ``v_worst_ds``
        p95 over time of the worst-bus deviation ``max(|v_max - v_set|,
        |v_min - v_set|)``.  Catches an acceptable RMS hiding one far-off bus.

    ``v_band_excess_ts`` / ``v_band_excess_ds``
        Mean per-step excess beyond the **inner** band ``v_set +/- quality_band``.
        Graded and active in the regime the controller actually operates in --
        the role the 0.90/1.10 hinge was meant to play and never did.
    """
    out = {
        "v_rms_ts": float("nan"), "v_rms_ds": float("nan"),
        "v_worst_ts": float("nan"), "v_worst_ds": float("nan"),
        "v_band_excess_ts": 0.0, "v_band_excess_ds": 0.0,
    }
    if not records:
        return out

    rms_ts = voltage_rms_err_all(records, v_set)["rms_err_pu"]
    if rms_ts.size and bool(np.isfinite(rms_ts).any()):
        out["v_rms_ts"] = float(np.nanmean(rms_ts))

    # DSO groups report only min/mean/max, so the spatial RMS is unavailable;
    # |mean - v_set| is the honest fallback and is what the DS envelope gives.
    env_ds = voltage_envelope_ds(records)
    if env_ds["v_mean"].size and bool(np.isfinite(env_ds["v_mean"]).any()):
        out["v_rms_ds"] = float(np.nanmean(np.abs(env_ds["v_mean"] - v_set)))

    def _worst_and_excess(
        v_min: NDArray[np.float64], v_max: NDArray[np.float64],
    ) -> tuple[float, float]:
        dev = np.fmax(np.abs(v_max - v_set), np.abs(v_min - v_set))
        dev = dev[np.isfinite(dev)]
        if dev.size == 0:
            return float("nan"), 0.0
        worst = float(np.percentile(dev, 95.0))
        excess = float(np.mean(np.maximum(dev - quality_band, 0.0)))
        return worst, excess

    env_ts = voltage_envelope_ts(records)
    out["v_worst_ts"], out["v_band_excess_ts"] = _worst_and_excess(
        env_ts["v_min"], env_ts["v_max"],
    )
    out["v_worst_ds"], out["v_band_excess_ds"] = _worst_and_excess(
        env_ds["v_min"], env_ds["v_max"],
    )
    return out


def _itae_q_pcc(records: List[MultiTSOIterationRecord]) -> float:
    """Time-weighted absolute Q-PCC tracking error across all DSOs.

    Returns ``0.0`` when the network simply has no PCC interfaces to track
    (nothing to get wrong), and ``nan`` only when interfaces *are* present but
    their values are non-finite (divergence).  Collapsing those two cases would
    mark a perfectly healthy interface-free run as inadmissible.
    """
    if not records:
        return 0.0
    t_min = np.array([r.time_s / 60.0 for r in records], dtype=float)
    err_per_step = np.full(len(records), np.nan)
    any_keys = False
    for i, r in enumerate(records):
        keys = set(r.dso_trafo_q_set_mvar) & set(r.dso_trafo_q_actual_mvar)
        if not keys:
            continue
        any_keys = True
        e = [abs(r.dso_trafo_q_set_mvar[k] - r.dso_trafo_q_actual_mvar[k])
             for k in keys
             if math.isfinite(r.dso_trafo_q_set_mvar[k])
             and math.isfinite(r.dso_trafo_q_actual_mvar[k])]
        if e:
            err_per_step[i] = float(np.mean(e))
    if not any_keys:
        return 0.0
    return _itae(t_min, err_per_step)


def _itae_q_tie(records: List[MultiTSOIterationRecord]) -> float:
    """Time-weighted absolute Q-tie tracking error across all zone pairs.

    Per step we compute the mean ``|Q_tie_pair − Q_tie_set_pair|`` across
    all zone pairs reported in
    :attr:`MultiTSOIterationRecord.zone_tie_q_mvar`, then ITAE the
    series.  Mean (not sum) keeps the numerical scale comparable to
    :func:`_itae_q_pcc`, so :class:`CostWeights` can use a consistent
    normalisation.

    Setpoint resolution: if records carry a ``zone_tie_q_set_mvar``
    dict (added in a future Phase C runner update), per-pair setpoints
    are used.  Otherwise the metric falls back to the Phase B target
    of 0 Mvar (no inter-zone reactive exchange) — which matches the
    controller's actual setpoint in the current configuration, so the
    fallback is correct, just not future-proof.

    Returns ``0.0`` when no zone pairs are reported at all (a single-zone or
    tie-free network has nothing to track); ``nan`` only when pairs are present
    but non-finite.  See :func:`_itae_q_pcc`.
    """
    if not records:
        return 0.0
    t_min = np.array([r.time_s / 60.0 for r in records], dtype=float)
    err_per_step = np.full(len(records), np.nan)
    any_pairs = False
    for i, r in enumerate(records):
        pair_q = r.zone_tie_q_mvar
        # Forward-compat: read per-pair setpoint when the runner
        # populates it; default to {} which yields 0 setpoint per pair.
        pair_set = getattr(r, "zone_tie_q_set_mvar", {}) or {}
        if not pair_q:
            continue
        any_pairs = True
        e: list[float] = []
        for pair, q in pair_q.items():
            if q is None or not math.isfinite(float(q)):
                continue
            sp = float(pair_set.get(pair, 0.0))
            e.append(abs(float(q) - sp))
        if e:
            err_per_step[i] = float(np.mean(e))
    if not any_pairs:
        return 0.0
    return _itae(t_min, err_per_step)


def _itae_pcc_underutilization(
    records: List[MultiTSOIterationRecord],
    v_mean_ts: NDArray[np.float64],
    t_min: NDArray[np.float64],
    v_set: float,
    deadband_v: float = 0.005,
    q_ref_mvar: float = 100.0,
) -> float:
    """ITAE of the per-step product

        ``max(|v_mean_ts(t) - v_set| - deadband_v, 0)
         × max(q_ref_mvar - mean_DSO |Q_PCC_actual(t)|, 0)``.

    Penalises "TSO voltage stressed AND DSO PCC sitting still."  Zero
    when the TSO mean voltage error is inside the deadband OR the
    DSO-mean ``|Q_PCC_actual|`` is at or above ``q_ref_mvar``.  The
    DSO mean (over the ``r.dso_trafo_q_actual_mvar`` keys) matches
    the convention of :func:`_itae_q_pcc`.

    Units: ``min · pu · Mvar``.  Calibration scale used in
    :func:`extract_metrics` corresponds to a 75-min window with
    sustained 5 mpu voltage error and 100 Mvar of PCC slack — see the
    ``norm_pcc_underutil`` comment there.
    """
    if not records or v_mean_ts.size == 0:
        return 0.0
    n = min(len(records), int(v_mean_ts.size), int(t_min.size))
    if n < 2:
        return 0.0

    stress = np.maximum(np.abs(v_mean_ts[:n] - v_set) - deadband_v, 0.0)

    inactivity = np.full(n, np.nan)
    for i in range(n):
        r = records[i]
        q_vals = [
            abs(float(v))
            for v in r.dso_trafo_q_actual_mvar.values()
            if v is not None and math.isfinite(float(v))
        ]
        if q_vals:
            inactivity[i] = max(q_ref_mvar - float(np.mean(q_vals)), 0.0)

    product = stress * inactivity
    return _itae(t_min[:n], product)


def _rho_emp_percentile(
    records: List[MultiTSOIterationRecord],
    pct: float = 95.0,
) -> float:
    """Percentile of :attr:`zone_contraction_lhs` across all
    ``(record, zone)`` pairs.

    Returns ``0.0`` when no records carry contraction data.
    """
    vals: list[float] = []
    for r in records:
        for v in r.zone_contraction_lhs.values():
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
    if not vals:
        return 0.0
    return float(np.percentile(vals, pct))


def _nanmean_or(arr: NDArray[np.float64], default: float) -> float:
    """``nanmean`` that returns ``default`` for an all-NaN or empty array.

    ``np.nanmean`` emits ``RuntimeWarning: Mean of empty slice`` in that case,
    which is a routine occurrence on the divergence path and would otherwise
    spam every failed trial.
    """
    if arr.size == 0 or not bool(np.isfinite(arr).any()):
        return default
    return float(np.nanmean(arr))


def _normalise(metric: float, scale: float) -> float:
    """Divide by ``scale`` with NaN/inf safety.

    A non-finite metric maps to ``inf`` -- the metric could not be computed, so
    the run must not be scored as merely "nominal-bad".  The previous mapping to
    ``1.0`` understated divergence by roughly two orders of magnitude and was
    the second half of the divergence-discount defect described in
    :func:`_itae`.

    A non-positive ``scale`` yields ``0.0`` (term disabled).
    """
    if not math.isfinite(metric):
        return float("inf")
    if scale <= 0.0:
        return 0.0
    return float(metric / scale)


def cost_components(
    m: TrajectoryMetrics,
    weights: CostWeights | None = None,
    scales: MetricScales | None = None,
) -> dict[str, float]:
    """Per-component weighted contributions to ``J`` for one trajectory.

    Returns a dict with both ``norm_*`` (raw normalised metrics, scale-
    matched but unweighted) and ``contrib_*`` (``weight × norm``) keys.
    Sum of ``contrib_*`` equals ``m.cost_J`` up to floating-point noise.

    Used by the Optuna objective to store a per-scenario breakdown on
    each trial's ``user_attrs`` for offline weight calibration.  Also
    handy for the tuning report to show "which term dominates" per
    trial.
    """
    weights = weights or CostWeights()
    scales = scales or MetricScales()
    n_steps = max(m.n_records, 1)
    # ``norm = 1`` means "at engineering tolerance"; weights then take over to
    # encode priority.  Divisors come from :class:`MetricScales` -- they used to
    # be duplicated here and in ``extract_metrics`` and drift apart silently.
    norm_v_ts          = _normalise(m.itae_v_ts,          scales.itae_v_ts)
    norm_v_ds          = _normalise(m.itae_v_ds,          scales.itae_v_ds)
    norm_q             = _normalise(m.itae_q_pcc,         scales.q_pcc)
    norm_q_tie         = _normalise(m.itae_q_tie,         scales.q_tie)
    norm_pcc_underutil = _normalise(m.itae_pcc_underutil, scales.pcc_underutil)
    norm_osc   = 100.0 * m.osc_rate
    norm_tap   = (m.n_tap_switches_tso + m.n_tap_switches_dso) / 5.0
    norm_viol  = m.voltage_excess_pu / n_steps
    pf_cost    = float(min(m.pf_failures, 1))
    return {
        "norm_v_ts":             norm_v_ts,
        "norm_v_ds":             norm_v_ds,
        "norm_q":                norm_q,
        "norm_q_tie":            norm_q_tie,
        "norm_pcc_underutil":    norm_pcc_underutil,
        "norm_osc":              norm_osc,
        "norm_tap":              norm_tap,
        "norm_viol":             norm_viol,
        "norm_pf":               pf_cost,
        "contrib_v_ts":          weights.w_v_track_ts    * norm_v_ts,
        "contrib_v_ds":          weights.w_v_track_ds    * norm_v_ds,
        "contrib_q":             weights.w_q_track       * norm_q,
        "contrib_q_tie":         weights.w_q_tie_track   * norm_q_tie,
        "contrib_pcc_underutil": weights.w_pcc_underutil * norm_pcc_underutil,
        "contrib_osc":           weights.w_osc           * norm_osc,
        "contrib_tap":           weights.w_tap           * norm_tap,
        "contrib_viol":          weights.w_viol          * norm_viol,
        "contrib_pf":            weights.w_pf            * pf_cost,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metrics(
    records: List[MultiTSOIterationRecord],
    cfg: MultiTSOConfig,
    weights: CostWeights | None = None,
    floors: NoiseFloors | None = None,
    scales: MetricScales | None = None,
) -> TrajectoryMetrics:
    """Extract all metrics from one closed-loop log.

    On total failure (empty log) returns :data:`INFEASIBLE_SENTINEL` with
    ``infeasible_reason`` set, so the run is both maximally costly on the scalar
    path and identifiable as inadmissible by the constraint path.
    """
    weights = weights or CostWeights()
    floors = floors or NoiseFloors()
    scales = scales or MetricScales()
    v_set = float(cfg.v_setpoint_pu)
    v_lo = float(getattr(cfg, "v_min_pu", 0.90))
    v_hi = float(getattr(cfg, "v_max_pu", 1.10))

    pf_fail = _detect_pf_failures(records)

    if not records:
        # Empty log: nothing was simulated.  Scored at the sentinel rather than
        # at ``w_pf`` (=100), which historically sat at only the ~60th
        # percentile of *converged* costs and therefore rewarded divergence.
        return TrajectoryMetrics(
            itae_v_ts=float("nan"), itae_v_ds=float("nan"),
            rmsd_v_ts=float("nan"), rmsd_v_ds=float("nan"),
            itae_q_pcc=float("nan"), itae_q_tie=float("nan"),
            itae_pcc_underutil=float("nan"),
            v_rms_ts=float("nan"), v_rms_ds=float("nan"),
            v_worst_ts=float("nan"), v_worst_ds=float("nan"),
            pf_failures=pf_fail,
            cost_J=INFEASIBLE_SENTINEL,
            infeasible_reason="empty_log",
            n_records=0, n_tso_active=0, n_dso_active=0,
        )

    # voltage envelopes / RMSDs / violations / losses (re-use helpers)
    env_ts  = voltage_envelope_ts(records)
    env_ds  = voltage_envelope_ds(records)
    rmsd_ts = voltage_rmsd_ts(records, v_set)["rmsd_pu"]
    rmsd_ds = voltage_rmsd_ds(records, v_set)["rmsd_pu"]
    vv_ts   = voltage_violation_counts_ts(records, low=v_lo, high=v_hi)
    vv_ds   = voltage_violation_counts_ds(records, low=v_lo, high=v_hi)
    losses  = loss_series(records)

    # Voltage quality measured against v_set (the discriminating signal), as
    # opposed to the hard corridor (which is a constraint and near-always slack).
    vq = _v_quality(records, v_set, scales.v_quality_band)

    # ITAE for voltage tracking (mean spatial error per step → time-weighted)
    abs_err_ts = np.abs(env_ts["v_mean"] - v_set)
    abs_err_ds = np.abs(env_ds["v_mean"] - v_set)
    itae_v_ts = _itae(env_ts["t_min"], abs_err_ts)
    itae_v_ds = _itae(env_ds["t_min"], abs_err_ds)
    itae_q_pcc = _itae_q_pcc(records)
    itae_q_tie = _itae_q_tie(records)
    itae_pcc_underutil = _itae_pcc_underutilization(
        records, env_ts["v_mean"], env_ts["t_min"], v_set,
        deadband_v=weights.pcc_underutil_v_deadband,
        q_ref_mvar=weights.pcc_underutil_q_ref_mvar,
    )

    # oscillations: stack per-actuator commands across steps, then DECIMATE to
    # the issuing controller's tick grid.  Without decimation the held-command
    # padding makes `_count_oscillations` return 0 for every trajectory (see
    # `_decimate_to_ticks`), which is what made this whole term dead.
    tso_tick = np.array([bool(r.tso_active) for r in records], dtype=bool)
    der_seq  = _stack_dict_arrays([r.zone_q_der     for r in records])
    pcc_seq  = _stack_dict_arrays([r.zone_q_pcc_set for r in records])
    vgen_seq = _stack_dict_arrays([r.zone_v_gen     for r in records])
    der_ticks  = _decimate_to_ticks(der_seq,  tso_tick)
    pcc_ticks  = _decimate_to_ticks(pcc_seq,  tso_tick)
    vgen_ticks = _decimate_to_ticks(vgen_seq, tso_tick)
    n_osc_der  = _count_oscillations(der_ticks,  floors.der_q_mvar)
    n_osc_pcc  = _count_oscillations(pcc_ticks,  floors.pcc_q_mvar)
    n_osc_vgen = _count_oscillations(vgen_ticks, floors.v_gen_pu)

    # Horizon length, needed to express switching as a rate.
    times = [float(r.time_s) for r in records if math.isfinite(float(r.time_s))]
    duration_s = (max(times) - min(times)) if len(times) >= 2 else 0.0

    # tap switches (TSO)
    tso_taps_seq = _stack_dict_arrays([r.zone_oltc_taps for r in records])
    n_tap_tso = _count_tap_switches(tso_taps_seq)
    tap_ops_tso, tap_rev_tso = _tap_wear(tso_taps_seq, duration_s)

    # tap switches (DSO): dict[str, int] → reshape into per-step row
    dso_tap_keys = sorted({k for r in records for k in r.dso_trafo_tap_pos})
    if dso_tap_keys:
        dso_taps_seq = np.full((len(records), len(dso_tap_keys)), np.nan)
        for i, r in enumerate(records):
            for j, k in enumerate(dso_tap_keys):
                v_int = r.dso_trafo_tap_pos.get(k)
                if v_int is not None:
                    dso_taps_seq[i, j] = float(v_int)
        n_tap_dso = _count_tap_switches(dso_taps_seq)
        tap_ops_dso, tap_rev_dso = _tap_wear(dso_taps_seq, duration_s)
    else:
        n_tap_dso = 0
        tap_ops_dso, tap_rev_dso = 0.0, 0.0

    rho_p95 = _rho_emp_percentile(records, pct=95.0)

    # ── Soft voltage excess (Issue 2: cliff → ramp) ─────────────────────
    # `voltage_excess_pu` is the sum of per-record band-edge excess in
    # pu·step.  norm_viol divides by step count → mean excess per step.
    # Corridor read from the config rather than hard-coded.
    voltage_excess_pu = _voltage_band_excess(records, low=v_lo, high=v_hi)

    # ── Guard band (headroom): same ramp, corridor shrunk by h at both ends ──
    # DS groups only: this exists to score the subordinate layer's own margin,
    # which is the quantity the filter was blind to.  Normalised per record so
    # it is comparable across horizons, like norm_viol below.
    _h = DS_GUARD_HEADROOM_PU
    guard_deficit_ds_pu = _voltage_band_excess(
        records, low=v_lo + _h, high=v_hi - _h, groups="ds",
    ) / max(len(records), 1)
    ds_headroom_min_pu = _ds_headroom_min(records, low=v_lo, high=v_hi)

    # ── Diagnostic violation counts (kept for the report; not in J) ─────
    n_viol_ts = int(np.nansum(vv_ts["n_low"]) + np.nansum(vv_ts["n_high"]))
    n_viol_ds = int(np.nansum(vv_ds["n_low"]) + np.nansum(vv_ds["n_high"]))

    # ── Oscillation rate (Issue 3: per-actuator-per-step) ───────────────
    # Normalising by total step-pairs × actuator count produces a rate in
    # [0, 1].  Multiplied by 100 below it expresses "% of step-pairs that
    # flipped sign" — comparable across scenarios with different
    # actuator counts and durations.
    n_actuators_total = (
        int(der_seq.shape[1]) + int(pcc_seq.shape[1]) + int(vgen_seq.shape[1])
    )
    n_step_pairs = max(len(records) - 1, 1)
    osc_rate = (n_osc_der + n_osc_pcc + n_osc_vgen) / max(
        n_actuators_total * n_step_pairs, 1
    )

    # ── Composite cost ──────────────────────────────────────────────────
    # Divisors come from ``scales``; the same object drives
    # ``cost_components()``, so the two can no longer disagree.
    norm_v_ts  = _normalise(itae_v_ts,  scales.itae_v_ts)
    norm_v_ds  = _normalise(itae_v_ds,  scales.itae_v_ds)
    norm_q     = _normalise(itae_q_pcc, scales.q_pcc)
    norm_q_tie = _normalise(itae_q_tie, scales.q_tie)
    norm_pcc_underutil = _normalise(itae_pcc_underutil, scales.pcc_underutil)
    norm_osc   = 100.0 * osc_rate                          # percent
    norm_tap   = (n_tap_tso + n_tap_dso) / 5.0
    norm_viol  = voltage_excess_pu / max(len(records), 1)  # mean pu/step

    # Issue 4: cap pf_fail at 1 in the cost so any divergence gives a
    # flat catastrophe penalty.  Diagnostic field below preserves the
    # raw count.
    pf_fail_cost = min(pf_fail, 1)

    J = (
        weights.w_v_track_ts * norm_v_ts
        + weights.w_v_track_ds * norm_v_ds
        + weights.w_q_track * norm_q
        + weights.w_q_tie_track * norm_q_tie
        + weights.w_pcc_underutil * norm_pcc_underutil
        + weights.w_osc * norm_osc
        + weights.w_tap * norm_tap
        + weights.w_viol * norm_viol
        + weights.w_pf * pf_fail_cost
    )

    # Admissibility.  A power-flow failure now makes the run inadmissible and
    # pins the scalar at the sentinel, instead of scoring it at ``w_pf`` (=100)
    # -- a value that undercut 35-43 % of *converged* runs and therefore made
    # divergence a profitable search direction.
    infeasible_reason = "pf_failure" if pf_fail > 0 else ""
    if not math.isfinite(J):
        infeasible_reason = infeasible_reason or "non_finite_metric"
    if infeasible_reason:
        J = INFEASIBLE_SENTINEL

    return TrajectoryMetrics(
        itae_v_ts=itae_v_ts,
        itae_v_ds=itae_v_ds,
        rmsd_v_ts=_nanmean_or(rmsd_ts, 0.0),
        rmsd_v_ds=_nanmean_or(rmsd_ds, 0.0),
        itae_q_pcc=itae_q_pcc,
        itae_q_tie=itae_q_tie,
        itae_pcc_underutil=itae_pcc_underutil,
        n_viol_v_ts=n_viol_ts,
        n_viol_v_ds=n_viol_ds,
        voltage_excess_pu=float(voltage_excess_pu),
        guard_deficit_ds_pu=float(guard_deficit_ds_pu),
        ds_headroom_min_pu=float(ds_headroom_min_pu),
        n_osc_der=n_osc_der,
        n_osc_pcc=n_osc_pcc,
        n_osc_v_gen=n_osc_vgen,
        n_tap_switches_tso=n_tap_tso,
        n_tap_switches_dso=n_tap_dso,
        osc_rate=float(osc_rate),
        rho_emp_p95=rho_p95,
        pf_failures=pf_fail,
        losses_mean_mw=(
            float(np.nanmean(losses["losses_mw"]))
            if losses["losses_mw"].size else 0.0
        ),
        cost_J=float(J),
        n_records=len(records),
        n_tso_active=sum(1 for r in records if r.tso_active),
        n_dso_active=sum(1 for r in records if r.dso_active),
        v_rms_ts=vq["v_rms_ts"],
        v_rms_ds=vq["v_rms_ds"],
        v_worst_ts=vq["v_worst_ts"],
        v_worst_ds=vq["v_worst_ds"],
        v_band_excess_ts=vq["v_band_excess_ts"],
        v_band_excess_ds=vq["v_band_excess_ds"],
        tap_ops_per_h_tso=tap_ops_tso,
        tap_ops_per_h_dso=tap_ops_dso,
        tap_reversals_per_h_tso=tap_rev_tso,
        tap_reversals_per_h_dso=tap_rev_dso,
        infeasible_reason=infeasible_reason,
        duration_s=duration_s,
    )
