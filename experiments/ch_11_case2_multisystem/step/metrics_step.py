"""
experiments/ch_11_case2_multisystem/step/metrics_step.py
========================================================

Window metrics of the Case-2 step tests (``ch:case3:step``; plan
``docs/architecture/2026-10-28_case2_step_mc_plan.md`` §2-4), from one run's
record log (one record per 20 s step, plant truth).  Everything is computed
per run in the worker; paired quantities (recruited flow) come later from the
logs.

* :func:`window` -- per area and aggregated (RMS over the areas) over a time
  window ``[a, b)``: time-RMS, mean and peak of e_v,a, the same of the
  deviation from a baseline, the in-period sawtooth (peak-to-peak within each
  TS period: mean and max), the effort (AVR travel, TS taps, banks, coupler
  taps), the STS band (smallest margin, samples above the band by more than
  0.5 mpu) and the interface tracking error where set-points exist.
* :func:`recovery` -- after an event at ``t_e`` up to ``t_end``: per area the
  peak, the final value (last sample before ``t_end``), the settling time and
  the number of TS dispatches until e_v,a stays within ``tol`` of the final
  value, and the contraction per dispatch rho_hat (log-linear least-squares fit
  of |e(t_d,k) - e_final| over the dispatch instants until it falls below
  ``fit_floor``).

Author: Manuel Schwenke / Claude Code (2026-10-28)
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

AREAS = (1, 2, 3)
#: STS operating band (``MultiTSOConfig.dso_v_min_pu`` / ``dso_v_max_pu``).
V_LO_STS, V_HI_STS = 0.92, 1.08
EXCESS_TOL = 0.5e-3


def _t(log) -> np.ndarray:
    return np.array([float(r.time_s) for r in log])


def _err(log, z) -> np.ndarray:
    return np.array([float(r.zone_v_rms_err_pu.get(z, np.nan)) for r in log])


def _travel(rows: Sequence) -> float:
    a = [np.asarray(x, dtype=float).ravel() for x in rows]
    n = max((x.size for x in a), default=0)
    if n == 0 or len(a) < 2:
        return 0.0
    m = np.array([np.pad(x, (0, n - x.size), constant_values=np.nan) for x in a])
    return float(np.nansum(np.abs(np.diff(m, axis=0))))


def _switchings(rows: Sequence) -> int:
    a = [np.asarray(x, dtype=float).ravel() for x in rows]
    n = max((x.size for x in a), default=0)
    if n == 0 or len(a) < 2:
        return 0
    m = np.array([np.pad(x, (0, n - x.size), constant_values=np.nan) for x in a])
    return int(np.nansum(np.abs(np.diff(m, axis=0)) > 0.5))


def sawtooth(t: np.ndarray, e: np.ndarray, t_ts: float) -> Dict[str, float]:
    """Peak-to-peak of ``e`` within each complete TS period of the window."""
    k = np.floor((t - 1e-6) / t_ts).astype(int)
    pp = [float(np.ptp(e[k == kk])) for kk in np.unique(k)
          if np.sum(k == kk) >= max(2, int(round(t_ts / np.median(np.diff(t)))) - 1)]
    if not pp:
        return {"pp_mean": float("nan"), "pp_max": float("nan")}
    return {"pp_mean": float(np.mean(pp)), "pp_max": float(np.max(pp))}


def window(log: list, a_s: float, b_s: float, t_ts: float = 180.0,
           baseline: Optional[Dict[int, float]] = None) -> Dict[str, object]:
    t = _t(log)
    sel = (t >= a_s - 1e-6) & (t < b_s - 1e-6)
    recs = [r for r, s in zip(log, sel) if s]
    out: Dict[str, object] = {"a_s": a_s, "b_s": b_s, "n": int(sel.sum())}
    if not recs:
        return out
    tw = t[sel]
    areas: Dict[str, Dict[str, float]] = {}
    for z in AREAS:
        e = _err(log, z)[sel]
        d = {"rms": float(np.sqrt(np.nanmean(e ** 2))), "mean": float(np.nanmean(e)),
             "max": float(np.nanmax(e))}
        if baseline is not None and z in baseline:
            de = e - float(baseline[z])
            d.update(rms_dev=float(np.sqrt(np.nanmean(de ** 2))),
                     mean_dev=float(np.nanmean(de)), max_dev=float(np.nanmax(de)))
        d.update({f"saw_{k}": v for k, v in sawtooth(tw, e, t_ts).items()})
        d["avr_travel_pu"] = _travel([r.zone_v_gen.get(z, []) for r in recs])
        d["ts_taps"] = _switchings([r.zone_oltc_taps.get(z, []) for r in recs])
        d["banks"] = _switchings([r.zone_tso_shunt_states.get(z, []) for r in recs])
        areas[str(z)] = d
    out["areas"] = areas
    for key in ("rms", "rms_dev", "max", "max_dev"):
        vals = [areas[str(z)].get(key) for z in AREAS]
        if all(v is not None for v in vals):
            out[f"agg_{key}"] = float(np.sqrt(np.mean(np.square(vals))))
    keys = sorted({k for r in recs for k in r.dso_trafo_tap_pos})
    out["coupler_taps"] = _switchings([[r.dso_trafo_tap_pos.get(k, np.nan) for k in keys]
                                       for r in recs])
    sts: Dict[str, Dict[str, float]] = {}
    for g in sorted({g for r in recs for g in r.dso_group_v_max_pu}):
        hi = np.array([r.dso_group_v_max_pu.get(g, np.nan) for r in recs])
        lo = np.array([r.dso_group_v_min_pu.get(g, np.nan) for r in recs])
        dev = np.array([r.dso_group_v_rms_dev_pu.get(g, np.nan) for r in recs])
        exc = np.maximum(hi - V_HI_STS, V_LO_STS - lo)
        sts[g] = {"margin_min": float(np.nanmin(-exc)),
                  "n_above_tol": int(np.nansum(exc > EXCESS_TOL)),
                  "f_v_rms": float(np.sqrt(np.nanmean(dev ** 2)))}
    out["sts"] = sts
    gaps = []
    for r in recs:
        for k, q in r.dso_trafo_q_actual_mvar.items():
            qs = r.dso_trafo_q_set_mvar.get(k)
            if q is not None and qs is not None and np.isfinite(q) and np.isfinite(qs):
                gaps.append(float(q) - float(qs))
    out["e_sts_rms_mvar"] = float(np.sqrt(np.mean(np.square(gaps)))) if gaps else float("nan")
    return out


def baseline(log: list, a_s: float, b_s: float) -> Dict[int, float]:
    """Mean e_v,a over ``[a, b)`` -- the settled state before a disturbance."""
    t = _t(log)
    sel = (t >= a_s - 1e-6) & (t < b_s - 1e-6)
    return {z: float(np.nanmean(_err(log, z)[sel])) for z in AREAS}


def rho_fit(d: np.ndarray, fit_floor: float) -> float:
    """Contraction per dispatch from |e_k - e_final| (k = 1, 2, ...)."""
    d = np.asarray(d, dtype=float)
    keep = []
    for x in d:
        if not np.isfinite(x) or x < fit_floor:
            break
        keep.append(x)
    if len(keep) < 2:
        return float("nan")
    k = np.arange(len(keep), dtype=float)
    slope = np.polyfit(k, np.log(keep), 1)[0]
    return float(math.exp(slope))


def recovery(log: list, t_e: float, t_end: float, t_ts: float = 180.0,
             tol: float = 1e-3, fit_floor: float = 0.2e-3) -> Dict[str, Dict[str, float]]:
    t = _t(log)
    sel = (t >= t_e - 1e-6) & (t < t_end - 1e-6)
    out: Dict[str, Dict[str, float]] = {}
    if sel.sum() < 2:
        return out
    tw = t[sel]
    disp = np.isclose(np.mod(tw, t_ts), 0.0) | np.isclose(np.mod(tw, t_ts), t_ts)
    for z in AREAS:
        e = _err(log, z)[sel]
        e_fin = float(e[-1])
        outside = np.abs(e - e_fin) > tol
        if outside.any():
            last = int(np.max(np.nonzero(outside)[0]))
            settle_s = float(tw[min(last + 1, len(tw) - 1)] - t_e)
            n_disp = int(np.sum(disp[: last + 1]))
        else:
            settle_s, n_disp = 0.0, 0
        dk = np.abs(e[disp] - e_fin)
        out[str(z)] = {"peak": float(np.nanmax(e)), "final": e_fin,
                       "settle_s": settle_s, "n_rec": n_disp,
                       "rho_hat": rho_fit(dk, fit_floor)}
    return out
