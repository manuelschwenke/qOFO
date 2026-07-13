#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_v5_util.py
===========================
Throwaway diagnostic (2026-06-05): decompose *why* the centralized V5 shows
higher generator-Q utilisation than the cascaded V4 in 006_CIGRE_MONTECARLO,
even though both apply identical generator penalties (g_w_gen, g_z_q_gen).

Reproduces a handful of the *already-accepted* 006 scenarios (same start_time +
contingency schedule, by seed) and runs four variants on each:

* ``V4``         -- cascade (reference).
* ``V5``         -- central, period 10 s (reference).
* ``V5_slowgen`` -- central with ``central_period_s=180`` (the whole central
                    controller, generators included, now fires every 3 min like
                    V4's TS-OFO).  Isolates the *cadence* hypothesis.  (Caveat:
                    slows ALL central actuators, not only the AVRs.)
* ``V4_notie``   -- cascade with ``tso_g_q_tie=0`` (inter-zone tie-Q objective
                    removed).  Tests the *flow-term* hypothesis: does removing
                    the implicit flow penalty raise V4's gen utilisation toward
                    V5?

Per (variant, seed) it records the Table-3 metrics (res_util, m_bar_pu, ...)
plus a **breadth** measurement computed from the full in-memory log:

* ``frac_engaged``     -- mean over steps of (#machines with per-machine
                          utilisation > 0.05) / (#machines).  Higher = more of
                          the fleet recruited into reactive duty.
* ``per_machine_util`` -- time-mean per-machine utilisation, keyed ``z<zone>k<k>``
                          (the ``zone_q_gen`` index), each annotated with S_n.

Run one variant per process (parallelise by launching several):
    python experiments/diag_v5_util.py --variant V4   --seeds 20260602,20260603
    ...
    python experiments/diag_v5_util.py --aggregate

Outputs: ``experiments/results/diag_v5/<variant>.json`` and, on --aggregate,
``summary.txt`` printed to stdout.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import glob
import importlib.util
import json
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "experiments", "results", "diag_v5")
ENGAGE_THRESH = 0.05  # per-machine utilisation above which a machine is "engaged"


def _load_mc006():
    """Import the digit-named 006 driver by path (not importable normally)."""
    path = os.path.join(ROOT, "experiments", "006_CIGRE_MONTECARLO.py")
    spec = importlib.util.spec_from_file_location("mc006", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANT_OVERRIDES = None  # filled lazily from mc006.VARIANTS


def _variant_overrides(mc006) -> Dict[str, Dict[str, Any]]:
    return {
        "V4": dict(mc006.VARIANTS["V4"]),
        "V5": dict(mc006.VARIANTS["V5"]),
        "V5_slowgen": {**mc006.VARIANTS["V5"], "central_period_s": 180},
        "V4_notie": {**mc006.VARIANTS["V4"], "tso_g_q_tie": 0.0},
        # Central WITHOUT HV/distribution voltage tracking (only TN voltage):
        # tests whether holding the HV buses at setpoint is what loads the
        # load-zone TN machines.  HV stays within bounds via the g_z_voltage
        # slack regardless, so this isolates the *tracking* objective.
        "V5_noHVv": {**mc006.VARIANTS["V5"], "central_dso_g_v": 0.0},
    }


def _breadth(log, total_machines: int):
    """Per-machine utilisation + fleet engagement breadth from a full log."""
    sum_u: Dict[tuple, float] = {}
    cnt_u: Dict[tuple, int] = {}
    sum_q: Dict[tuple, float] = {}
    engaged_frac: List[float] = []
    for r in log:
        eng = 0
        mach = 0
        for z, q_arr in r.zone_q_gen.items():
            if q_arr is None:
                continue
            h_arr = r.gen_q_headroom_mvar.get(z)
            if h_arr is None:
                continue
            qa = np.abs(np.asarray(q_arr, dtype=np.float64))
            ha = np.asarray(h_arr, dtype=np.float64)
            n = min(qa.size, ha.size)
            for k in range(n):
                denom = qa[k] + ha[k]
                if denom <= 1e-9:
                    continue
                u = qa[k] / denom
                if not np.isfinite(u):
                    continue
                key = (int(z), int(k))
                sum_u[key] = sum_u.get(key, 0.0) + u
                cnt_u[key] = cnt_u.get(key, 0) + 1
                sum_q[key] = sum_q.get(key, 0.0) + float(qa[k])
                mach += 1
                if u > ENGAGE_THRESH:
                    eng += 1
        if mach > 0:
            engaged_frac.append(eng / total_machines)
    frac_engaged = float(np.mean(engaged_frac)) if engaged_frac else float("nan")
    per_machine_util = {f"z{z}k{k}": sum_u[(z, k)] / cnt_u[(z, k)]
                        for (z, k) in sum_u}
    per_machine_q = {f"z{z}k{k}": sum_q[(z, k)] / cnt_u[(z, k)]
                     for (z, k) in sum_q}
    return frac_engaged, per_machine_util, per_machine_q


def _build_group_zone(cfg) -> Dict[str, int]:
    """Map each DSO group id (HV-network net_id, the key in ``dso_group_*``) to
    its TSO zone, rebuilding the net exactly as the runner does."""
    from network.ieee39 import build_ieee39_net, add_hv_networks
    from network.zone_partition import fixed_zone_partition_ieee39

    net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                                 verbose=False)
    fixed_zone_partition_ieee39(net, verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
        verbose=False,
    )
    return {str(hv.net_id): int(hv.zone) for hv in meta.hv_networks}


def _der_util(log, group_zone: Dict[str, int]):
    """Per-zone DSO-DER reactive utilisation (capability-weighted, time-mean).

    Uses the recorded group totals ``dso_group_q_der_mvar`` and the directional
    capability ``dso_group_q_der_max_mvar`` (Q>=0) / ``|dso_group_q_der_min_mvar|``
    (Q<0).  Returns (util_by_zone, q_mvar_by_zone, cap_mvar_by_zone) with int
    zone keys.
    """
    zones = sorted(set(group_zone.values()))
    util_acc: Dict[int, List[float]] = {z: [] for z in zones}
    q_acc: Dict[int, List[float]] = {z: [] for z in zones}
    cap_acc: Dict[int, List[float]] = {z: [] for z in zones}
    for r in log:
        qd = getattr(r, "dso_group_q_der_mvar", None)
        if not qd:
            continue
        qmax = r.dso_group_q_der_max_mvar
        qmin = r.dso_group_q_der_min_mvar
        sumQ = {z: 0.0 for z in zones}
        sumC = {z: 0.0 for z in zones}
        seen = {z: False for z in zones}
        for g, q in qd.items():
            z = group_zone.get(str(g))
            if z is None:
                continue
            q = float(q)
            cap = (float(qmax.get(g, 0.0)) if q >= 0
                   else abs(float(qmin.get(g, 0.0))))
            sumQ[z] += abs(q)
            sumC[z] += cap
            seen[z] = True
        for z in zones:
            if not seen[z]:
                continue
            q_acc[z].append(sumQ[z])
            cap_acc[z].append(sumC[z])
            if sumC[z] > 1e-9:
                util_acc[z].append(sumQ[z] / sumC[z])
    f = lambda d, z: (float(np.mean(d[z])) if d[z] else float("nan"))
    return ({z: f(util_acc, z) for z in zones},
            {z: f(q_acc, z) for z in zones},
            {z: f(cap_acc, z) for z in zones})


def _build_ts_der_caps(cfg) -> Dict[int, List[tuple]]:
    """{zone: [(q_min,q_max), ...]} for the transmission-level (TSO) DER, in the
    same order as ``record.zone_q_der`` (= zone_der_indices, assigned by bus)."""
    from network.ieee39 import build_ieee39_net, add_hv_networks
    from network.zone_partition import fixed_zone_partition_ieee39
    from experiments.helpers.plant_io import _sgen_q_capability

    net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                                 verbose=False)
    zone_map, _bz = fixed_zone_partition_ieee39(net, verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
        verbose=False,
    )
    caps: Dict[int, List[tuple]] = {int(z): [] for z in zone_map}
    for s_idx, s_bus in zip(meta.tso_der_indices, meta.tso_der_buses):
        for z, buses in zone_map.items():
            if s_bus in set(buses):
                caps[int(z)].append(_sgen_q_capability(net, int(s_idx)))
                break
    return {z: v for z, v in caps.items() if v}


def _ts_der_util(log, ts_caps: Dict[int, List[tuple]]):
    """Per-zone transmission-DER utilisation (capability-weighted, time-mean)
    from ``record.zone_q_der``.  Returns (util, q_mvar, cap_mvar) by zone."""
    zones = sorted(ts_caps)
    util_acc = {z: [] for z in zones}
    q_acc = {z: [] for z in zones}
    cap_acc = {z: [] for z in zones}
    for r in log:
        zq = getattr(r, "zone_q_der", None)
        if not zq:
            continue
        for z in zones:
            arr = zq.get(z)
            caps = ts_caps[z]
            if arr is None or not caps:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            n = min(arr.size, len(caps))
            sQ = 0.0
            sC = 0.0
            seen = False
            for k in range(n):
                q = float(arr[k])
                qmin, qmax = caps[k]
                sQ += abs(q)
                sC += qmax if q >= 0 else abs(qmin)
                seen = True
            if seen:
                q_acc[z].append(sQ)
                cap_acc[z].append(sC)
                if sC > 1e-9:
                    util_acc[z].append(sQ / sC)
    f = lambda d, z: (float(np.mean(d[z])) if d[z] else float("nan"))
    return ({z: f(util_acc, z) for z in zones},
            {z: f(q_acc, z) for z in zones},
            {z: f(cap_acc, z) for z in zones})


def run_variant(variant: str, seeds: List[int]) -> None:
    mc006 = _load_mc006()
    from experiments.runners import run_multi_tso_dso
    from experiments.helpers.comparison_metrics import (
        cigre_summary_table, gen_s_rated_by_zone,
    )

    overrides = _variant_overrides(mc006)[variant]
    base = mc006.make_cigre_config()
    elements = mc006.enumerate_elements(base)
    n_total_min = int(round(base.n_total_s / 60.0))
    ratings = gen_s_rated_by_zone(base.scenario)
    total_machines = sum(len(v) for v in ratings.values())
    srated_flat = {f"z{z}k{k}": float(ratings[z][k])
                   for z in ratings for k in range(len(ratings[z]))}
    group_zone = _build_group_zone(base)
    ts_caps = _build_ts_der_caps(base)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        start_time = mc006.random_start_time(rng)
        schedule = mc006.build_random_schedule(rng, elements, n_total_min)

        cfg = mc006.make_cigre_config()
        cfg.start_time = start_time
        cfg.contingencies = copy.deepcopy(schedule)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.verbose = 0
        cfg.result_dir = os.path.join(OUT_DIR, "_scratch", f"{variant}_{seed}")
        os.makedirs(cfg.result_dir, exist_ok=True)
        n_steps = int(round(cfg.n_total_s / cfg.dt_s))

        print(f"[{variant}] seed={seed} start={start_time:%Y-%m-%d %H:%M} "
              f"{len(schedule)} events ...", flush=True)
        try:
            log = run_multi_tso_dso(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[{variant}] seed={seed} DIVERGED: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append({"variant": variant, "seed": int(seed),
                         "converged": False})
            continue
        if len(log) != n_steps:
            print(f"[{variant}] seed={seed} short log {len(log)}/{n_steps}",
                  flush=True)
            rows.append({"variant": variant, "seed": int(seed),
                         "converged": False})
            continue

        s = cigre_summary_table({variant: log}, v_set=mc006.V_SET,
                                gen_s_rated_mva=ratings).iloc[0]
        frac_engaged, pmu, pmq = _breadth(log, total_machines)
        der_util, der_q, der_cap = _der_util(log, group_zone)
        ts_util, ts_q, ts_cap = _ts_der_util(log, ts_caps)
        rows.append({
            "variant": variant, "seed": int(seed), "converged": True,
            "res_util": float(s["res_util"]),
            "m_bar_pu": float(s["m_bar_pu"]),
            "m_bar_mvar": float(s["m_bar_mvar"]),
            "rms_v_ts_pu": float(s["rms_v_ts_pu"]),
            "rms_q_tie_mvar": float(s["rms_q_tie_mvar"]),
            "n_sw": float(s["n_sw"]),
            "frac_engaged": frac_engaged,
            "per_machine_util": pmu,
            "per_machine_q_mvar": pmq,
            "der_util_by_zone": {str(z): v for z, v in der_util.items()},
            "der_q_mvar_by_zone": {str(z): v for z, v in der_q.items()},
            "der_cap_mvar_by_zone": {str(z): v for z, v in der_cap.items()},
            "ts_der_util_by_zone": {str(z): v for z, v in ts_util.items()},
            "ts_der_q_mvar_by_zone": {str(z): v for z, v in ts_q.items()},
            "ts_der_cap_mvar_by_zone": {str(z): v for z, v in ts_cap.items()},
        })
        print(f"[{variant}] seed={seed} res_util={s['res_util']:.3f} "
              f"frac_engaged={frac_engaged:.3f} "
              f"tieQ={s['rms_q_tie_mvar']:.1f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"variant": variant, "s_rated_mva": srated_flat,
           "total_machines": total_machines, "rows": rows}
    with open(os.path.join(OUT_DIR, f"{variant}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[{variant}] wrote {os.path.join(OUT_DIR, variant + '.json')}",
          flush=True)


def aggregate() -> None:
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.json")))
    files = [f for f in files if os.path.basename(f) != "summary.json"]
    if not files:
        print("no per-variant JSONs in", OUT_DIR)
        return
    data = {}
    srated = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        data[j["variant"]] = [r for r in j["rows"] if r.get("converged")]
        srated.update(j.get("s_rated_mva", {}))

    order = [v for v in ("V4", "V4_notie", "V5", "V5_slowgen", "V5_noHVv")
             if v in data]

    def m(v, key):
        vals = [r[key] for r in data[v] if key in r]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals \
            else (float("nan"), float("nan"), 0)

    print("\n" + "=" * 78)
    print("  V5-utilisation decomposition (mean over converged seeds)")
    print("=" * 78)
    hdr = f"{'variant':<11}{'n':>3}{'res_util':>10}{'m_bar_pu':>10}" \
          f"{'frac_eng':>10}{'tieQ':>8}{'n_sw':>7}{'rms_v':>9}"
    print(hdr)
    for v in order:
        ru = m(v, "res_util"); mb = m(v, "m_bar_pu"); fe = m(v, "frac_engaged")
        tq = m(v, "rms_q_tie_mvar"); sw = m(v, "n_sw"); rv = m(v, "rms_v_ts_pu")
        print(f"{v:<11}{ru[2]:>3}{ru[0]:>10.3f}{mb[0]:>10.3f}{fe[0]:>10.3f}"
              f"{tq[0]:>8.1f}{sw[0]:>7.1f}{rv[0]:>9.5f}")

    # Per-machine mean utilisation table (averaged over seeds).
    keys = sorted(srated, key=lambda kk: -srated[kk])  # largest S_n first
    print("\n  Per-machine time-mean utilisation (machines sorted by S_n):")
    print(f"{'machine':<9}{'S_n[MVA]':>10}  " +
          "".join(f"{v:>11}" for v in order))
    for kk in keys:
        line = f"{kk:<9}{srated[kk]:>10.0f}  "
        for v in order:
            vals = [r["per_machine_util"].get(kk, np.nan)
                    for r in data[v] if "per_machine_util" in r]
            vals = [x for x in vals if np.isfinite(x)]
            line += f"{(np.mean(vals) if vals else float('nan')):>11.3f}"
        print(line)

    # Per-zone DSO-DER utilisation + absolute Q (only variants that recorded it).
    der_zones = sorted({z for v in order for r in data[v]
                        for z in r.get("der_util_by_zone", {})},
                       key=lambda s: int(s))

    def _zmean(v, field, z):
        vals = [r[field].get(z) for r in data[v] if field in r]
        vals = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    if der_zones:
        print("\n  Per-zone DSO-DER utilisation (capability-weighted, time-mean):")
        print(f"{'zone':<6}" + "".join(f"{v:>12}" for v in order))
        for z in der_zones:
            print(f"{z:<6}" +
                  "".join(f"{_zmean(v, 'der_util_by_zone', z):>12.3f}"
                          for v in order))
        print("\n  Per-zone DSO-DER reactive output |Q| [Mvar], time-mean:")
        print(f"{'zone':<6}" + "".join(f"{v:>12}" for v in order))
        for z in der_zones:
            print(f"{z:<6}" +
                  "".join(f"{_zmean(v, 'der_q_mvar_by_zone', z):>12.1f}"
                          for v in order))

    ts_zones = sorted({z for v in order for r in data[v]
                       for z in r.get("ts_der_util_by_zone", {})},
                      key=lambda s: int(s))
    if ts_zones:
        print("\n  Per-zone TS-DER (transmission-level) utilisation "
              "(capability-weighted, time-mean):")
        print(f"{'zone':<6}" + "".join(f"{v:>12}" for v in order))
        for z in ts_zones:
            print(f"{z:<6}" +
                  "".join(f"{_zmean(v, 'ts_der_util_by_zone', z):>12.3f}"
                          for v in order))
        print("\n  Per-zone TS-DER reactive output |Q| [Mvar], time-mean:")
        print(f"{'zone':<6}" + "".join(f"{v:>12}" for v in order))
        for z in ts_zones:
            print(f"{z:<6}" +
                  "".join(f"{_zmean(v, 'ts_der_q_mvar_by_zone', z):>12.1f}"
                          for v in order))
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, default=None,
                    choices=["V4", "V5", "V5_slowgen", "V4_notie", "V5_noHVv"])
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    if not args.variant:
        ap.error("need --variant NAME or --aggregate")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        ap.error("need --seeds s1,s2,...")
    run_variant(args.variant, seeds)


if __name__ == "__main__":
    main()
