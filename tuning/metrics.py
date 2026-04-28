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

        ``pf_failure >> violation > oscillation > tap_switch > tracking_error``.
    """

    w_v_track: float = 1.0
    w_q_track: float = 1.0
    w_osc:     float = 5.0
    w_tap:     float = 0.5
    w_viol:    float = 10.0
    w_pf:      float = 1000.0


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

    der_q_mvar: float = 5.0    # ~1 % of typical wind-park rating (500 MW)
    pcc_q_mvar: float = 1.0    # PCC tracking is small-signal
    v_gen_pu:   float = 0.001  # 0.1 % voltage
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

    # constraint health
    n_viol_v_ts:         int
    n_viol_v_ds:         int

    # actuator activity
    n_osc_der:           int
    n_osc_pcc:           int
    n_osc_v_gen:         int
    n_tap_switches_tso:  int
    n_tap_switches_dso:  int

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
        return TrajectoryMetrics(
            itae_v_ts=1.0, itae_v_ds=1.0, rmsd_v_ts=1.0, rmsd_v_ds=1.0,
            itae_q_pcc=1.0,
            n_viol_v_ts=0, n_viol_v_ds=0,
            n_osc_der=0, n_osc_pcc=0, n_osc_v_gen=0,
            n_tap_switches_tso=0, n_tap_switches_dso=0,
            rho_emp_p95=0.0, pf_failures=pf_fail,
            losses_mean_mw=0.0,
            cost_J=float(weights.w_pf * pf_fail),
            n_records=0, n_tso_active=0, n_dso_active=0,
        )

    # voltage envelopes / RMSDs / violations / losses (re-use helpers)
    env_ts  = voltage_envelope_ts(records)
    env_ds  = voltage_envelope_ds(records)
    rmsd_ts = voltage_rmsd_ts(records, v_set)["rmsd_pu"]
    rmsd_ds = voltage_rmsd_ds(records, v_set)["rmsd_pu"]
    vv_ts   = voltage_violation_counts_ts(records, low=0.95, high=1.05)
    vv_ds   = voltage_violation_counts_ds(records, low=0.95, high=1.05)
    losses  = loss_series(records)

    # ITAE for voltage tracking (mean spatial error per step → time-weighted)
    abs_err_ts = np.abs(env_ts["v_mean"] - v_set)
    abs_err_ds = np.abs(env_ds["v_mean"] - v_set)
    itae_v_ts = _itae(env_ts["t_min"], abs_err_ts)
    itae_v_ds = _itae(env_ds["t_min"], abs_err_ds)
    itae_q_pcc = _itae_q_pcc(records)

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

    # composite cost
    norm_v_ts = _normalise(itae_v_ts,  scale=10.0)        # min · pu
    norm_v_ds = _normalise(itae_v_ds,  scale=10.0)
    norm_q    = _normalise(itae_q_pcc, scale=1000.0)      # min · Mvar
    norm_osc  = (n_osc_der + n_osc_pcc + n_osc_vgen) / 10.0
    norm_tap  = (n_tap_tso + n_tap_dso) / 5.0
    n_viol_ts = int(np.nansum(vv_ts["n_low"]) + np.nansum(vv_ts["n_high"]))
    n_viol_ds = int(np.nansum(vv_ds["n_low"]) + np.nansum(vv_ds["n_high"]))
    norm_viol = (n_viol_ts + n_viol_ds) / max(1, len(records))

    J = (
        weights.w_v_track * (norm_v_ts + norm_v_ds)
        + weights.w_q_track * norm_q
        + weights.w_osc * norm_osc
        + weights.w_tap * norm_tap
        + weights.w_viol * norm_viol
        + weights.w_pf * pf_fail
    )

    return TrajectoryMetrics(
        itae_v_ts=itae_v_ts,
        itae_v_ds=itae_v_ds,
        rmsd_v_ts=float(np.nanmean(rmsd_ts)) if rmsd_ts.size else 0.0,
        rmsd_v_ds=float(np.nanmean(rmsd_ds)) if rmsd_ds.size else 0.0,
        itae_q_pcc=itae_q_pcc,
        n_viol_v_ts=n_viol_ts,
        n_viol_v_ds=n_viol_ds,
        n_osc_der=n_osc_der,
        n_osc_pcc=n_osc_pcc,
        n_osc_v_gen=n_osc_vgen,
        n_tap_switches_tso=n_tap_tso,
        n_tap_switches_dso=n_tap_dso,
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
