"""
experiments/ch_11_case2_multisystem/metrics_case2.py
====================================================

Metrics of Case Study 2 (thesis ``ch:case3``) from a run's record log
(``List[MultiTSOIterationRecord]``, one record per 20 s step, plant truth).

Per TS area ``z`` (:func:`area_metrics`)
---------------------------------------
* ``e_v, e_vp, worst, t_out, avr_trav, q_move, taps`` -- exactly Case 1's
  :func:`experiments.ch_10_case1_svr.report_case1.area_metrics`
  (``eq:case3:ev``, ``eq:case3:evp``, worst bus, time outside [0.95, 1.05] pu).
* ``u_q`` -- ``eq:case3:uq``: time mean of the fleet mean of |Q_g| / Q_g^lim,
  Q_g^lim the capability in the direction of the present output, over the
  area's in-service machines EXCLUDING the network equivalent (slack).
* ``r_plus_mean/min``, ``r_minus_mean/min`` -- ``eq:setup:metrics:reserve``,
  directional synchronous reserve [Mvar], same fleet.
* ``shunt_ops`` -- MSC/MSR switchings of the area (step changes of a bank).

Aggregate: :func:`aggregate_e_v`, root-mean-square over the areas -- the form
Case 1 reports (declared 2026-10-27; the Ch 10 text must say so).

Per sub-transmission system ``m`` (:func:`sts_metrics`)
------------------------------------------------------
* ``u_q_sts`` -- ``eq:setup:metrics:uqsts``: time mean of sum|Q_i| / sum Q_i^lim,
  Q_i^lim from the plant's own capability law in the direction of the output.
* ``f_v_sts`` -- ``eq:setup:metrics:vcost``: time-RMS of the spatial RMS
  deviation from the STS voltage set-point.
* ``margin_min`` -- smallest distance of any STS bus to the band [lo, hi].
* ``coupler_taps`` -- coupler tap operations.

Per coupling transformer (:func:`interface_metrics`, set-point variants)
-----------------------------------------------------------------------
* ``e_sts_rms`` -- time-RMS of Q_act - Q_set over the horizon.
* ``e_sts_end`` -- RMS of the same at the last step before every TS dispatch,
  the residual the parent sees (settling after a new set-point excluded).

Paired (:func:`recruited_flow`) -- ``eq:setup:metrics:recruit``:
``T[|Q^A - Q^B|]`` per coupler, A with access, B the same law without.

Per event (:func:`event_metrics`) -- ``eq:case3:epost`` as defined for Case 2:
* baseline ``e(t_e+)`` = the area error just after the event, BEFORE any
  controller of that step acts (``pre_control_zone_v_rms_err_pu``);
* window = the states held between the event and the next TS dispatch t_d
  (end-of-step records at t_e <= t < t_d);
* ``epost = max(window) - e(t_e+)`` (signed; > 0 means the error grew while the
  parent had not yet re-dispatched);
* ``impact = e(t_e+) - e(t_e - dt)``;
* ``w_sts`` per coupler -- the withdrawn support: the time integral over the
  window of the interface flow moving back against the direction in which the
  event moved it, in Mvar*min.

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from experiments.ch_10_case1_svr import report_case1 as _rc1

V_LO, V_HI = 0.95, 1.05


# ---------------------------------------------------------------------------
#  Small pieces
# ---------------------------------------------------------------------------

def _coupler(key: str) -> str:
    """``"<group>|trafo_<t>"`` (either path) -> ``"trafo_<t>"``."""
    return key.split("|")[-1]


def _fleet(r, z):
    """Q, q_min, q_max of the area's in-service non-slack machines at ``r``."""
    if z not in getattr(r, "gen_q_min_mvar", {}) or z not in r.zone_q_gen:
        return None
    q = np.asarray(r.zone_q_gen[z], dtype=float)
    lo = np.asarray(r.gen_q_min_mvar[z], dtype=float)
    hi = np.asarray(r.gen_q_max_mvar[z], dtype=float)
    keep = np.asarray(r.zone_gen_in_service.get(z, np.ones(q.shape, bool)), bool)
    keep &= ~np.asarray(r.zone_gen_is_slack.get(z, np.zeros(q.shape, bool)), bool)
    return q[keep], lo[keep], hi[keep]


def _switchings(series: Sequence[np.ndarray]) -> int:
    n, prev = 0, None
    for a in series:
        a = np.asarray(a, dtype=float)
        if prev is not None and prev.shape == a.shape:
            n += int(np.sum(np.abs(a - prev) > 0.5))
        prev = a
    return n


# ---------------------------------------------------------------------------
#  Areas
# ---------------------------------------------------------------------------

def area_metrics(log: list, z: int, ref: float = 1.03, v_lo: float = V_LO,
                 v_hi: float = V_HI) -> Dict[str, float]:
    m = dict(_rc1.area_metrics(log, z, ref, v_lo, v_hi))
    util, rp, rm = [], [], []
    for r in log:
        f = _fleet(r, z)
        if f is None or f[0].size == 0:
            continue
        q, lo, hi = f
        lim = np.where(q >= 0.0, hi, -lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(lim > 1e-9, np.abs(q) / lim, np.nan)
        util.append(float(np.nanmean(share)))
        rp.append(float(np.sum(hi - q)))
        rm.append(float(np.sum(q - lo)))
    m["u_q"] = float(np.nanmean(util)) if util else float("nan")
    m["r_plus_mean"] = float(np.mean(rp)) if rp else float("nan")
    m["r_plus_min"] = float(np.min(rp)) if rp else float("nan")
    m["r_minus_mean"] = float(np.mean(rm)) if rm else float("nan")
    m["r_minus_min"] = float(np.min(rm)) if rm else float("nan")
    m["shunt_ops"] = _switchings(
        [r.zone_tso_shunt_states[z] for r in log
         if z in getattr(r, "zone_tso_shunt_states", {})])
    return m


def aggregate_e_v(per_area: Dict[int, Dict[str, float]],
                  key: str = "e_v") -> float:
    """Root-mean-square over the areas (Case 1's form)."""
    vals = [d[key] for d in per_area.values() if np.isfinite(d.get(key, np.nan))]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


# ---------------------------------------------------------------------------
#  Sub-transmission systems and interfaces
# ---------------------------------------------------------------------------

def sts_metrics(log: list, net_id: str, v_lo: float, v_hi: float
                ) -> Dict[str, float]:
    util, dev, margin = [], [], []
    for r in log:
        if net_id in getattr(r, "dso_der_q_mvar", {}):
            q = np.asarray(r.dso_der_q_mvar[net_id], dtype=float)
            lim = np.where(q >= 0.0,
                           np.asarray(r.dso_der_q_max_mvar[net_id], dtype=float),
                           -np.asarray(r.dso_der_q_min_mvar[net_id], dtype=float))
            tot = float(np.sum(lim))
            util.append(float(np.sum(np.abs(q))) / tot if tot > 1e-9 else np.nan)
        if net_id in r.dso_group_v_rms_dev_pu:
            dev.append(float(r.dso_group_v_rms_dev_pu[net_id]))
        if net_id in r.dso_group_v_min_pu and net_id in r.dso_group_v_max_pu:
            margin.append(min(float(r.dso_group_v_min_pu[net_id]) - v_lo,
                              v_hi - float(r.dso_group_v_max_pu[net_id])))
    taps: Dict[str, List[float]] = {}
    for r in log:
        for k, t in r.dso_trafo_tap_pos.items():
            if r.dso_trafo_group.get(k, k.split("|")[0]) == net_id:
                taps.setdefault(_coupler(k), []).append(float(t))
    n_taps = sum(int(np.sum(np.abs(np.diff(v)) > 0.5)) for v in taps.values())
    return dict(
        u_q_sts=float(np.nanmean(util)) if util else float("nan"),
        f_v_sts=float(np.sqrt(np.mean(np.square(dev)))) if dev else float("nan"),
        margin_min=float(np.min(margin)) if margin else float("nan"),
        coupler_taps=n_taps,
    )


def interface_metrics(log: list, t_ts: float, dt: float) -> Dict[str, Dict[str, float]]:
    """Interface tracking error per coupler, where set-points are tracked."""
    err: Dict[str, List[float]] = {}
    end: Dict[str, List[float]] = {}
    for r in log:
        last_before_dispatch = math.isclose(
            math.fmod(r.time_s + dt, t_ts), 0.0, abs_tol=1e-6) or math.isclose(
            math.fmod(r.time_s + dt, t_ts), t_ts, abs_tol=1e-6)
        for k, q_set in r.dso_trafo_q_set_mvar.items():
            q_act = r.dso_trafo_q_actual_mvar.get(k)
            if q_set is None or q_act is None:
                continue
            e = float(q_act) - float(q_set)
            if not np.isfinite(e):
                continue
            c = _coupler(k)
            err.setdefault(c, []).append(e)
            if last_before_dispatch:
                end.setdefault(c, []).append(e)
    out = {}
    for c, e in err.items():
        out[c] = dict(
            e_sts_rms=float(np.sqrt(np.mean(np.square(e)))),
            e_sts_end=float(np.sqrt(np.mean(np.square(end[c])))) if end.get(c)
            else float("nan"),
        )
    return out


def _coupler_q_series(log: list) -> Dict[str, Dict[float, float]]:
    s: Dict[str, Dict[float, float]] = {}
    for r in log:
        for k, q in r.dso_trafo_q_actual_mvar.items():
            if q is not None and np.isfinite(q):
                s.setdefault(_coupler(k), {})[round(r.time_s, 6)] = float(q)
    return s


def recruited_flow(log_access: list, log_without: list) -> Dict[str, float]:
    """``T[|Q^A - Q^B|]`` per coupler over the common time instants [Mvar]."""
    a, b = _coupler_q_series(log_access), _coupler_q_series(log_without)
    out = {}
    for c in sorted(set(a) & set(b)):
        ts = sorted(set(a[c]) & set(b[c]))
        if ts:
            out[c] = float(np.mean([abs(a[c][t] - b[c][t]) for t in ts]))
    return out


# ---------------------------------------------------------------------------
#  Events
# ---------------------------------------------------------------------------

def _next_dispatch(t_e: float, t_ts: float) -> float:
    """First TS dispatch strictly after the event's own step."""
    return (math.floor(t_e / t_ts + 1e-9) + 1) * t_ts


def event_metrics(log: list, t_ts: float, dt: float) -> List[Dict[str, object]]:
    """One row per (event step, area) and the per-coupler withdrawal."""
    t = np.array([r.time_s for r in log], dtype=float)
    rows: List[Dict[str, object]] = []
    for i, r in enumerate(log):
        pre = getattr(r, "pre_control_zone_v_rms_err_pu", {})
        if not pre:
            continue
        t_e = float(r.time_s)
        t_d = _next_dispatch(t_e, t_ts)
        win = np.where((t >= t_e - 1e-6) & (t < t_d - 1e-6))[0]
        q_pre = getattr(r, "pre_control_trafo_q_mvar", {})
        q_before = ({_coupler(k): v for k, v in log[i - 1].dso_trafo_q_actual_mvar.items()}
                    if i > 0 else {})
        w_sts = {}
        for c, q_plus in q_pre.items():
            if c not in q_before:
                continue
            s_e = np.sign(q_plus - q_before[c])
            if s_e == 0:
                w_sts[c] = 0.0
                continue
            back = 0.0
            for k in win:
                qk = {_coupler(kk): vv for kk, vv
                      in log[k].dso_trafo_q_actual_mvar.items()}.get(c)
                if qk is not None:
                    back += max(0.0, -s_e * (qk - q_plus)) * dt / 60.0
            w_sts[c] = back
        for z, e_plus in pre.items():
            e_win = [log[k].zone_v_rms_err_pu.get(z, np.nan) for k in win]
            e_prev = log[i - 1].zone_v_rms_err_pu.get(z, np.nan) if i > 0 else np.nan
            rows.append(dict(
                t_e_s=t_e, t_d_s=t_d, area=int(z),
                events=";".join(getattr(r, "events_fired", [])),
                e_plus=float(e_plus),
                epost=float(np.nanmax(e_win) - e_plus) if e_win else float("nan"),
                impact=float(e_plus - e_prev),
                n_window=len(win),
                w_sts_mvar_min=w_sts,
            ))
    return rows
