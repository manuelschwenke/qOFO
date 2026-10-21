"""
experiments/case1_svr/report_case1.py
=====================================

Metrics of the Case 1 campaign (thesis ``ch:case_svr``).  Reads
``<run_dir>/<variant>/log.pkl`` as written by ``run_case1.py``.

Per variant and TSO area
------------------------
* ``e_v``      time-RMS of the spatial RMS error over the area's monitored
               buses (``zone_v_rms_err_pu``) -- thesis ``eq:case3:ev``
* ``e_vp``     time-RMS of the pilot-bus error -- ``eq:case3:evp``
* ``worst``    largest |V - V_ref| of any area bus over the horizon
* ``t_out``    share of samples with any area bus outside the normal
               operating range [v_lo, v_hi] (absolute, default 0.95-1.05 pu;
               declared 2026-10-07 before any ensemble run)
* ``avr_trav`` summed |change| of the area's AVR set-points [pu]
* ``q_move``   summed |change| of the area's machine reactive power [Mvar]
* ``taps``     transmission tap operations of the area
* ``o0_act``   (O0 only) share of regional steps with an active constraint

Post-event growth (``eq:case3:epost``)
--------------------------------------
For each contingency at t_e: max of the area error over the following
supervisory interval (t_e, t_e + T_TS] minus its value at t_e.

Equivalence S1m vs O0
---------------------
Largest deviation of every recorded bus voltage and AVR set-point, over all
steps and over the steps in which O0 had no active constraint.

Writes ``metrics.csv``, ``epost.csv``, ``pareto.csv``, ``equivalence.json``.

Run
---
    python -m experiments.case1_svr.report_case1 --run-dir <dir> [--v-lo 0.95 --v-hi 1.05]

Author: Manuel Schwenke / Claude Code (2026-10-06)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.case1_svr.run_case1 import (
    K1_EXTRA, K1_PAR, K1_REF, PILOT_BUSES, base_config, resolved,
)


def _load(run_dir: Path, name: str) -> Optional[list]:
    p = run_dir / name / "log.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        log = pickle.load(f)
    return log or None


def _rms(x) -> float:
    a = np.asarray([v for v in x if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def _travel(series: List[np.ndarray]) -> float:
    tot, prev = 0.0, None
    for a in series:
        a = np.asarray(a, dtype=float)
        if prev is not None and prev.shape == a.shape:
            tot += float(np.nansum(np.abs(a - prev)))
        prev = a
    return tot


def _taps(series: List[np.ndarray]) -> int:
    n, prev = 0, None
    for a in series:
        a = np.asarray(a, dtype=float)
        if prev is not None and prev.shape == a.shape:
            n += int(np.sum(np.abs(a - prev) > 0.5))
        prev = a
    return n


def area_metrics(log: list, z: int, ref: float, v_lo: float,
                 v_hi: float) -> Dict[str, float]:
    pilot = PILOT_BUSES[z]
    e = [r.zone_v_rms_err_pu.get(z) for r in log if r.zone_v_rms_err_pu]
    ep = [r.bus_vm_pu[pilot] - ref for r in log
          if r.bus_vm_pu and pilot in r.bus_vm_pu]
    vmin = np.array([r.zone_v_min.get(z, np.nan) for r in log], dtype=float)
    vmax = np.array([r.zone_v_max.get(z, np.nan) for r in log], dtype=float)
    worst = float(np.nanmax(np.maximum(np.abs(vmin - ref), np.abs(vmax - ref))))
    out = (vmin < v_lo) | (vmax > v_hi)
    vg = [r.zone_v_gen[z] for r in log if z in r.zone_v_gen and len(r.zone_v_gen[z])]
    qg = [np.asarray(r.zone_q_gen[z], dtype=float) for r in log
          if z in r.zone_q_gen and r.zone_q_gen[z] is not None]
    taps = [r.zone_oltc_taps[z] for r in log if z in r.zone_oltc_taps]
    m = dict(e_v=_rms(e), e_vp=_rms(ep), worst=worst,
             t_out=float(np.mean(out)) if out.size else float("nan"),
             avr_trav=_travel(vg), q_move=_travel(qg), taps=_taps(taps))
    # O0 only.  "act" = an OUTPUT constraint binds (monitored-bus voltage at the
    # regional step, machine capability at the plant step) -- the only thing
    # that can make O0 depart from S1m.  "box" = an input bound binds, which is
    # the saturation the integral law applies as well.
    rvr = [r for r in log if z in getattr(r, "zone_svr_ofo_rvr_n_active", {})
           and r.tso_active]
    if rvr:
        m["o0_act"] = float(np.mean([r.zone_svr_ofo_rvr_n_active[z] > 0 for r in rvr]))
        m["o0_box"] = float(np.mean(
            [bool(r.zone_svr_ofo_rvr_box_active.get(z, False)) for r in rvr]))
    rpr = [r for r in log if z in getattr(r, "zone_svr_ofo_rpr_active", {})]
    if rpr:
        m["o0_rpr_act"] = float(np.mean(
            [np.any(r.zone_svr_ofo_rpr_active[z]) for r in rpr]))
        m["o0_rpr_box"] = float(np.mean(
            [np.any(getattr(r, "zone_svr_ofo_rpr_box_active", {}).get(z, False))
             for r in rpr]))
    return m


def epost(log: list, z: int, events_min: List[float], t_ts: float) -> Dict[float, float]:
    t = np.array([r.time_s for r in log], dtype=float)
    e = np.array([r.zone_v_rms_err_pu.get(z, np.nan) for r in log], dtype=float)
    res = {}
    for m in events_min:
        te = 60.0 * m
        idx = np.where(np.isclose(t, te))[0]
        if not idx.size:
            continue
        i = int(idx[0])
        win = (t > te) & (t <= te + t_ts)
        if not np.any(win):
            continue
        res[m] = float(np.nanmax(e[win]) - e[i])
    return res


def equivalence(a: list, b: list) -> Dict[str, float]:
    """Largest deviation S1m vs O0, overall and on steps where no OUTPUT
    constraint of O0 was active (input bounds are shared saturation)."""
    n = min(len(a), len(b))
    dv_all, dv_free, dg_all, dg_free, first = 0.0, 0.0, 0.0, 0.0, None
    first_out = None
    n_free = 0
    for k in range(n):
        ra, rb = a[k], b[k]
        keys = set(ra.bus_vm_pu) & set(rb.bus_vm_pu)
        dv = max((abs(ra.bus_vm_pu[i] - rb.bus_vm_pu[i]) for i in keys), default=0.0)
        dg = 0.0
        for z in set(ra.zone_v_gen) & set(rb.zone_v_gen):
            x, y = np.asarray(ra.zone_v_gen[z]), np.asarray(rb.zone_v_gen[z])
            if x.shape == y.shape and x.size:
                dg = max(dg, float(np.max(np.abs(x - y))))
        free = (not any(v > 0 for v in rb.zone_svr_ofo_rvr_n_active.values())
                and not any(np.any(v) for v in rb.zone_svr_ofo_rpr_active.values()))
        dv_all, dg_all = max(dv_all, dv), max(dg_all, dg)
        if free:
            n_free += 1
            dv_free, dg_free = max(dv_free, dv), max(dg_free, dg)
        elif first_out is None:
            first_out = float(rb.time_s)
        if first is None and max(dv, dg) > 1e-6:
            first = float(ra.time_s)
    return {"n_steps": n, "n_steps_no_output_constraint": n_free,
            "max_dv_bus_all_pu": dv_all, "max_dv_bus_no_output_pu": dv_free,
            "max_dv_avr_all_pu": dg_all, "max_dv_avr_no_output_pu": dg_free,
            "first_time_s_output_constraint_active": first_out,
            "first_time_s_deviation_above_1e-6": first}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--v-lo", type=float, default=0.95,
                    help="lower edge of the normal operating range [pu]")
    ap.add_argument("--v-hi", type=float, default=1.05,
                    help="upper edge of the normal operating range [pu]")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    cfg = base_config()
    zones = sorted(PILOT_BUSES)
    refs = {z: float((cfg.zone_v_setpoints_pu or {}).get(z, cfg.v_setpoint_pu))
            for z in zones}
    t_ts = float(cfg.tso_period_s)

    rows, erows, logs = [], [], {}
    for name in K1_REF + K1_EXTRA + K1_PAR:
        log = _load(run_dir, name)
        if log is None:
            continue
        logs[name] = log
        c = resolved(name)
        horizon = len(log) * float(c.dt_s) / 60.0
        events = sorted(e.minute for e in c.contingencies if e.minute < horizon)
        for z in zones:
            m = area_metrics(log, z, refs[z], args.v_lo, args.v_hi)
            rows.append(dict(variant=name, area=z, **m))
            for ev, g in epost(log, z, events, t_ts).items():
                erows.append(dict(variant=name, area=z, event_min=ev, e_post=g))

    if not rows:
        raise SystemExit(f"no logs under {run_dir}")
    met = pd.DataFrame(rows)
    met.to_csv(run_dir / "metrics.csv", index=False)
    ep = pd.DataFrame(erows)
    ep.to_csv(run_dir / "epost.csv", index=False)
    met[["variant", "area", "avr_trav", "q_move", "e_vp", "e_v"]].to_csv(
        run_dir / "pareto.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.5g}")
    print("\n== metrics per variant and area " + "=" * 40)
    print(met.to_string(index=False))
    if not ep.empty:
        print("\n== post-event error growth [pu] " + "=" * 40)
        print(ep.pivot_table(index=["variant", "area"], columns="event_min",
                             values="e_post").to_string())

    if "S1m" in logs and "O0" in logs:
        eq = equivalence(logs["S1m"], logs["O0"])
        (run_dir / "equivalence.json").write_text(json.dumps(eq, indent=2),
                                                  encoding="utf-8")
        print("\n== equivalence S1m vs O0 " + "=" * 47)
        for k, v in eq.items():
            print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
