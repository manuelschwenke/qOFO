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

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.comparison_metrics import (
    loss_series,
    voltage_envelope_ds,
    voltage_envelope_ts,
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
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryMetrics:
    """All metrics extracted from one closed-loop log."""

    # tracking
    itae_v_ts:           float
    itae_v_ds:           float
    rmsd_v_ts:           float
    rmsd_v_ds:           float
    itae_q_pcc:          float
    itae_q_tie:          float
    itae_pcc_underutil:  float

    # constraint health
    n_viol_v_ts:         int
    n_viol_v_ds:         int
    voltage_excess_pu:   float    # smooth band-edge excess used in cost

    # actuator activity
    n_osc_der:           int
    n_osc_pcc:           int
    n_osc_v_gen:         int
    n_tap_switches_tso:  int
    n_tap_switches_dso:  int
    osc_rate:            float    # rate in [0, 1] used in cost

    # stability
    rho_emp_p95:         float
    pf_failures:         int

    # losses (diagnostic only — not in J by default)
    losses_mean_mw:      float

    # composite
    cost_J:              float

    # bookkeeping
    n_records:           int
    n_tso_active:        int
    n_dso_active:        int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _itae(t_min: NDArray[np.float64], abs_err: NDArray[np.float64]) -> float:
    """Integral of time-weighted absolute error: ``∫ t·|e| dt``.

    ``t_min`` is in minutes (matching the
    :mod:`comparison_metrics` convention).  Returns ITAE in
    ``minute · pu`` (or ``minute · Mvar``).  NaN entries are dropped.
    """
    if t_min.size < 2:
        return 0.0
    integrand = t_min * abs_err
    finite = np.isfinite(integrand) & np.isfinite(t_min)
    if int(finite.sum()) < 2:
        return 0.0
    # ``np.trapezoid`` replaces ``np.trapz`` in NumPy 2.x; the latter
    # raises a DeprecationWarning.
    trapz_fn = getattr(np, "trapezoid", np.trapz)
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


def _count_tap_switches(taps_seq: NDArray[np.float64]) -> int:
    """Sum of ``|Δtap|`` across all ``(step, actuator)`` pairs."""
    if taps_seq.size == 0 or taps_seq.shape[0] < 2:
        return 0
    dtaps = np.diff(taps_seq, axis=0)
    return int(np.nansum(np.abs(dtaps)))


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


def _voltage_band_excess(
    records: List[MultiTSOIterationRecord],
    low: float = 0.9,
    high: float = 1.1,
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
    """
    excess = 0.0
    for r in records:
        for v_max in r.zone_v_max.values():
            if v_max is not None and math.isfinite(v_max):
                excess += max(float(v_max) - high, 0.0)
        for v_min in r.zone_v_min.values():
            if v_min is not None and math.isfinite(v_min):
                excess += max(low - float(v_min), 0.0)
        for v_max in r.dso_group_v_max_pu.values():
            if v_max is not None and math.isfinite(v_max):
                excess += max(float(v_max) - high, 0.0)
        for v_min in r.dso_group_v_min_pu.values():
            if v_min is not None and math.isfinite(v_min):
                excess += max(low - float(v_min), 0.0)
    return excess


def _itae_q_pcc(records: List[MultiTSOIterationRecord]) -> float:
    """Time-weighted absolute Q-PCC tracking error across all DSOs."""
    if not records:
        return 0.0
    t_min = np.array([r.time_s / 60.0 for r in records], dtype=float)
    err_per_step = np.full(len(records), np.nan)
    for i, r in enumerate(records):
        keys = set(r.dso_trafo_q_set_mvar) & set(r.dso_trafo_q_actual_mvar)
        if not keys:
            continue
        e = [abs(r.dso_trafo_q_set_mvar[k] - r.dso_trafo_q_actual_mvar[k])
             for k in keys
             if math.isfinite(r.dso_trafo_q_set_mvar[k])
             and math.isfinite(r.dso_trafo_q_actual_mvar[k])]
        if e:
            err_per_step[i] = float(np.mean(e))
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
    """
    if not records:
        return 0.0
    t_min = np.array([r.time_s / 60.0 for r in records], dtype=float)
    err_per_step = np.full(len(records), np.nan)
    for i, r in enumerate(records):
        pair_q = r.zone_tie_q_mvar
        # Forward-compat: read per-pair setpoint when the runner
        # populates it; default to {} which yields 0 setpoint per pair.
        pair_set = getattr(r, "zone_tie_q_set_mvar", {}) or {}
        if not pair_q:
            continue
        e: list[float] = []
        for pair, q in pair_q.items():
            if q is None or not math.isfinite(float(q)):
                continue
            sp = float(pair_set.get(pair, 0.0))
            e.append(abs(float(q) - sp))
        if e:
            err_per_step[i] = float(np.mean(e))
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


def _normalise(metric: float, scale: float) -> float:
    """Divide by ``scale`` with NaN/inf safety.

    Non-finite metrics are mapped to ``1.0`` (treated as nominal-bad);
    a non-positive ``scale`` yields ``0.0``.
    """
    if not math.isfinite(metric):
        return 1.0
    if scale <= 0.0:
        return 0.0
    return float(metric / scale)


def cost_components(
    m: TrajectoryMetrics,
    weights: CostWeights | None = None,
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
    n_steps = max(m.n_records, 1)
    # Scales correspond to **physical engineering tolerances** for a
    # 75-min scenario:
    #   v_ts:    5.0 mpu sustained → ITAE = 0.005 × 75²/2 = 14 min·pu
    #   v_ds:   10.7 mpu sustained → 30 min·pu
    #   q:       5.3 Mvar sustained → 15000 min·Mvar
    #   q_tie:  19.6 Mvar sustained → 55000 min·Mvar
    # ``norm = 1`` means "at engineering tolerance"; weights then take
    # over to encode priority and to size baseline contribution.
    norm_v_ts          = _normalise(m.itae_v_ts,          scale=14.0)     # min · pu
    norm_v_ds          = _normalise(m.itae_v_ds,          scale=30.0)     # min · pu
    norm_q             = _normalise(m.itae_q_pcc,         scale=15000.0)  # min · Mvar
    norm_q_tie         = _normalise(m.itae_q_tie,         scale=55000.0)  # min · Mvar
    norm_pcc_underutil = _normalise(m.itae_pcc_underutil, scale=1400.0)   # min · pu · Mvar
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
) -> TrajectoryMetrics:
    """Extract all metrics from one closed-loop log.

    On total failure (empty log) returns a high-cost sentinel with
    ``pf_failures = 1`` so BO drives away from this region.
    """
    weights = weights or CostWeights()
    floors = floors or NoiseFloors()
    v_set = float(cfg.v_setpoint_pu)

    pf_fail = _detect_pf_failures(records)

    if not records:
        # Empty log: catastrophe sentinel.  pf_fail capped at 1 (binary)
        # for J so an early-divergence is the same flat penalty as a
        # late-divergence — TPE can model "stay out" cleanly.
        return TrajectoryMetrics(
            itae_v_ts=1.0, itae_v_ds=1.0, rmsd_v_ts=1.0, rmsd_v_ds=1.0,
            itae_q_pcc=1.0, itae_q_tie=1.0, itae_pcc_underutil=1.0,
            n_viol_v_ts=0, n_viol_v_ds=0, voltage_excess_pu=0.0,
            n_osc_der=0, n_osc_pcc=0, n_osc_v_gen=0,
            n_tap_switches_tso=0, n_tap_switches_dso=0,
            osc_rate=0.0,
            rho_emp_p95=0.0, pf_failures=pf_fail,
            losses_mean_mw=0.0,
            cost_J=float(weights.w_pf * min(pf_fail, 1)),
            n_records=0, n_tso_active=0, n_dso_active=0,
        )

    # voltage envelopes / RMSDs / violations / losses (re-use helpers)
    env_ts  = voltage_envelope_ts(records)
    env_ds  = voltage_envelope_ds(records)
    rmsd_ts = voltage_rmsd_ts(records, v_set)["rmsd_pu"]
    rmsd_ds = voltage_rmsd_ds(records, v_set)["rmsd_pu"]
    vv_ts   = voltage_violation_counts_ts(records, low=0.9, high=1.1)
    vv_ds   = voltage_violation_counts_ds(records, low=0.9, high=1.1)
    losses  = loss_series(records)

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

    # oscillations: stack per-actuator commands across steps
    der_seq  = _stack_dict_arrays([r.zone_q_der     for r in records])
    pcc_seq  = _stack_dict_arrays([r.zone_q_pcc_set for r in records])
    vgen_seq = _stack_dict_arrays([r.zone_v_gen     for r in records])
    n_osc_der  = _count_oscillations(der_seq,  floors.der_q_mvar)
    n_osc_pcc  = _count_oscillations(pcc_seq,  floors.pcc_q_mvar)
    n_osc_vgen = _count_oscillations(vgen_seq, floors.v_gen_pu)

    # tap switches (TSO)
    tso_taps_seq = _stack_dict_arrays([r.zone_oltc_taps for r in records])
    n_tap_tso = _count_tap_switches(tso_taps_seq)

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
    else:
        n_tap_dso = 0

    rho_p95 = _rho_emp_percentile(records, pct=95.0)

    # ── Soft voltage excess (Issue 2: cliff → ramp) ─────────────────────
    # `voltage_excess_pu` is the sum of per-record band-edge excess in
    # pu·step.  norm_viol divides by step count → mean excess per step.
    voltage_excess_pu = _voltage_band_excess(records, low=0.9, high=1.1)

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
    # Scales = physical engineering tolerances (see ``cost_components``
    # docstring).  MUST stay in sync with ``cost_components()`` above.
    norm_v_ts  = _normalise(itae_v_ts,  scale=14.0)        # min · pu
    norm_v_ds  = _normalise(itae_v_ds,  scale=30.0)        # min · pu
    norm_q     = _normalise(itae_q_pcc, scale=15000.0)     # min · Mvar
    norm_q_tie = _normalise(itae_q_tie, scale=55000.0)     # min · Mvar
    # Conditional DSO-underutilisation term.  Scale = 1400 ≈
    # 0.005 pu × 100 Mvar × 75²/2 min² — i.e. ``norm = 1`` corresponds
    # to "5 mpu sustained voltage error AND 100 Mvar of DSO PCC slack
    # for the entire 75-min scenario," matching the engineering-
    # tolerance philosophy of ``norm_v_ts`` (scale 14 = 5 mpu × 75²/2).
    norm_pcc_underutil = _normalise(itae_pcc_underutil, scale=1400.0)
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

    return TrajectoryMetrics(
        itae_v_ts=itae_v_ts,
        itae_v_ds=itae_v_ds,
        rmsd_v_ts=float(np.nanmean(rmsd_ts)) if rmsd_ts.size else 0.0,
        rmsd_v_ds=float(np.nanmean(rmsd_ds)) if rmsd_ds.size else 0.0,
        itae_q_pcc=itae_q_pcc,
        itae_q_tie=itae_q_tie,
        itae_pcc_underutil=itae_pcc_underutil,
        n_viol_v_ts=n_viol_ts,
        n_viol_v_ds=n_viol_ds,
        voltage_excess_pu=float(voltage_excess_pu),
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
    )
