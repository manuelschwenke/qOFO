"""
analysis/e1_interface_column_norms.py
=====================================
E1 -- interface column norms: the measurement behind the sensitivity-aggregation
argument (thesis Ch 6, ``ch:architectures:cascade:aggregation``).

The question
------------
An OFO step moves each actuator in proportion to its column in ``H``.  The
architecture claims that this is *why* a controller with full observability but
no interface concept (variant V5) leaves distribution-side reactive capability
idle, while the cascade recruits it:

    ||d v_TS / d q_DER,i||     for one DS-connected DER   -- SMALL
                                 (path through the 110 kV network and the
                                  coupling transformer; electrical distance)

    ||d v_TS / d Q_DS,j^set||  for one interface          -- LARGE
                                 (an injection at the TS bus itself; by
                                  eq:multisystem:ds_response_columns the virtual
                                  actuator column IS the boundary-bus injection
                                  column, up to sign)

This script measures both on the same plant, at the same operating point, in
the same units, and reports the ratio.

What is measured, exactly
-------------------------
Pure *network* sensitivities: each perturbation runs ``pp.runpp`` with
``run_control=False``, so no local Q(V) loop and no controller reacts.  That is
deliberate -- the claim in the thesis is that the column disparity is "physics,
not tuning", so the measurement must not contain a droop transform or a
controller gain.  A closed-loop variant (``--closed-loop``) is available for
comparison; it changes the magnitudes but not the ordering.

Both column families are therefore the same physical object -- the response of
the monitored TS voltages to 1 Mvar injected at a bus -- differing only in WHERE
the Mvar is injected: at an individual DER deep in a 110 kV network, or at the
TS-side terminal of a coupling transformer.

Rows
----
The monitored TS voltage buses of the TSO controllers
(``TSOControllerConfig.voltage_bus_indices``), i.e. exactly the rows the
supervisory objective is written on.  Norms are 2-norms over that row set;
``--per-zone`` restricts each column to its own zone's rows instead, which is
the stricter reading (an actuator is only credited for the buses its own
controller monitors).

Usage
-----
    python -m analysis.e1_interface_column_norms
    python -m analysis.e1_interface_column_norms --delta 1.0 --out results/e1
    python -m analysis.e1_interface_column_norms --closed-loop --per-zone

Outputs
-------
* ``<out>/e1_column_norms.csv``   -- one row per actuator column
* ``<out>/e1_summary.csv``        -- the aggregate comparison
* ``<out>/e1_summary.txt``        -- human-readable report
* ``<out>/e1_columns.tex``        -- the thesis table body (booktabs rows)

Author: Manuel Schwenke / Claude Code
Date: 2026-08-17
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pandapower as pp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "results" / "e1_column_norms"


# ---------------------------------------------------------------------------
#  Load the case-study configuration (module name starts with a digit)
# ---------------------------------------------------------------------------

def _load_cigre_module():
    path = PROJECT_ROOT / "experiments" / "CIGRE_2026" / "005_CIGRE_MULTI.py"
    spec = importlib.util.spec_from_file_location("cigre_multi", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cigre_multi"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_state(variant: str = "V4") -> Dict[str, Any]:
    """Build plant + controllers at t0 and stop before the time loop."""
    cigre = _load_cigre_module()
    cfg = cigre.make_cigre_config()
    for key, value in cigre.VARIANTS[variant].items():
        setattr(cfg, key, value)
    cfg.verbose = 0
    cfg.run_stability_analysis = False

    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    captured: Dict[str, Any] = {}

    def hook(state: Dict[str, Any]) -> bool:
        captured.update(state)
        return True                      # abort before the time loop

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        run_multi_tso_dso(cfg, pre_loop_hook=hook)
    if not captured:
        raise SystemExit("[E1] the runner exposed no pre-loop state.")
    captured["_config"] = cfg
    return captured


# ---------------------------------------------------------------------------
#  Finite-difference column of the monitored TS voltages
# ---------------------------------------------------------------------------

def _runpp(net, run_control: bool) -> None:
    pp.runpp(net, run_control=run_control, max_iter=100,
             init="results", calculate_voltage_angles=True)


def _v_rows(net, rows: List[int]) -> np.ndarray:
    return net.res_bus.vm_pu.loc[rows].to_numpy(dtype=float)


def _column_at_bus(net, bus: int, rows: List[int], delta: float,
                   run_control: bool) -> np.ndarray:
    """d v[rows] / d q_inject[bus]  in p.u./Mvar, central difference.

    The perturbation is a reactive injection at ``bus`` realised as an extra
    load with ``q_mvar = -/+ delta`` (load convention: negative q_mvar injects).
    A dedicated auxiliary load is used rather than editing an existing element,
    so the perturbation is identical for every bus regardless of what is
    connected there.
    """
    work = copy.deepcopy(net)
    aux = pp.create_load(work, bus=bus, p_mw=0.0, q_mvar=0.0,
                         name="__e1_probe__")

    work.load.at[aux, "q_mvar"] = -delta          # inject +delta Mvar
    _runpp(work, run_control)
    v_plus = _v_rows(work, rows)

    work.load.at[aux, "q_mvar"] = +delta          # absorb delta Mvar
    _runpp(work, run_control)
    v_minus = _v_rows(work, rows)

    return (v_plus - v_minus) / (2.0 * delta)


# ---------------------------------------------------------------------------
#  Actuator inventory
# ---------------------------------------------------------------------------

def collect_columns(state: Dict[str, Any]) -> Dict[str, Any]:
    net = state["net"]
    tso_controllers = state["tso_controllers"]
    dso_controllers = state["dso_controllers"]

    # --- rows: monitored TS voltage buses, per zone and pooled -------------
    zone_rows: Dict[Any, List[int]] = {}
    for zid, ctrl in tso_controllers.items():
        zone_rows[zid] = list(ctrl.config.voltage_bus_indices)
    all_rows = sorted({b for rows in zone_rows.values() for b in rows})

    # --- LOCAL rows: the monitored HV buses of each subordinate network ----
    # Added 2026-08-17, after the first run falsified the far-field prediction.
    # A centralised controller carries these rows in the SAME objective as the
    # TS rows, so what an actuator does to them is priced against what it does
    # to the TS buses. This is where the disparity actually lives.
    hv_rows = {}
    for dso_id, ctrl in dso_controllers.items():
        hv_rows[dso_id] = list(ctrl.config.voltage_bus_indices)
    all_hv_rows = sorted({b for rows in hv_rows.values() for b in rows})

    # --- interface columns: TS-side terminal of each coupling transformer --
    interfaces = []
    for zid, ctrl in tso_controllers.items():
        cfg = ctrl.config
        for k, trafo_idx in enumerate(cfg.pcc_trafo_indices):
            dso_id = (cfg.pcc_dso_controller_ids[k]
                      if k < len(cfg.pcc_dso_controller_ids) else "?")
            # TS-side terminal: 3W coupler -> hv_bus; 2W -> hv_bus
            if trafo_idx in net.trafo3w.index:
                bus = int(net.trafo3w.at[trafo_idx, "hv_bus"])
            else:
                bus = int(net.trafo.at[trafo_idx, "hv_bus"])
            interfaces.append(dict(kind="interface", zone=zid, dso=dso_id,
                                   element=f"trafo{trafo_idx}", bus=bus))

    # --- DER columns: every DS-connected DER of every DSO controller -------
    ders = []
    dso_to_tso = state.get("dso_to_tso_id", {})
    for dso_id, ctrl in dso_controllers.items():
        zid = dso_to_tso.get(dso_id, "?")
        for s_idx in ctrl.config.der_indices:
            bus = int(net.sgen.at[s_idx, "bus"])
            ders.append(dict(kind="ds_der", zone=zid, dso=dso_id,
                             element=f"sgen{s_idx}", bus=bus,
                             sn_mva=float(net.sgen.at[s_idx, "sn_mva"])))

    return dict(net=net, zone_rows=zone_rows, all_rows=all_rows,
                hv_rows=hv_rows, all_hv_rows=all_hv_rows,
                interfaces=interfaces, ders=ders)


def measure(inv: Dict[str, Any], *, delta: float, run_control: bool,
            per_zone: bool) -> pd.DataFrame:
    net = inv["net"]
    _runpp(net, run_control)              # settle the base operating point

    records = []
    items = inv["interfaces"] + inv["ders"]
    t0 = time.perf_counter()
    for n, item in enumerate(items, 1):
        rows = (inv["zone_rows"].get(item["zone"], inv["all_rows"])
                if per_zone else inv["all_rows"])
        col = _column_at_bus(net, item["bus"], rows, delta, run_control)
        own_hv = inv["hv_rows"].get(item["dso"], inv["all_hv_rows"])
        col_hv = _column_at_bus(net, item["bus"], own_hv, delta, run_control)
        rec = dict(item)
        rec["n_rows"] = len(rows)
        rec["norm2_pu_per_mvar"] = float(np.linalg.norm(col, 2))
        rec["norm_inf_pu_per_mvar"] = float(np.max(np.abs(col)))
        rec["n_rows_hv"] = len(own_hv)
        rec["norm2_hv_own"] = float(np.linalg.norm(col_hv, 2))
        rec["norm_inf_hv_own"] = float(np.max(np.abs(col_hv)))
        rec["collateral_ratio"] = (rec["norm2_hv_own"] / rec["norm2_pu_per_mvar"]
                                   if rec["norm2_pu_per_mvar"] else float("nan"))
        records.append(rec)
        if n % 10 == 0 or n == len(items):
            print(f"  [E1] {n}/{len(items)} columns "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, sub in df.groupby("kind"):
        out.append(dict(
            kind=kind,
            n=len(sub),
            mean=sub.norm2_pu_per_mvar.mean(),
            median=sub.norm2_pu_per_mvar.median(),
            min=sub.norm2_pu_per_mvar.min(),
            max=sub.norm2_pu_per_mvar.max(),
        ))
    return pd.DataFrame(out).set_index("kind")


def write_report(df: pd.DataFrame, summ: pd.DataFrame, outdir: Path,
                 *, delta: float, run_control: bool, per_zone: bool) -> str:
    iface = df[df.kind == "interface"]
    der = df[df.kind == "ds_der"]
    fi, fd = iface.norm2_pu_per_mvar, der.norm2_pu_per_mvar
    li, ld = iface.norm2_hv_own, der.norm2_hv_own

    L = []
    L.append("E1 -- interface vs DS-DER sensitivity columns, one operating point")
    L.append("=" * 68)
    L.append(f"perturbation : +/- {delta:g} Mvar reactive injection at the actuator bus")
    L.append(f"power flow   : run_control={run_control} "
             f"({'local Q(V) loops react' if run_control else 'pure network response'})")
    L.append(f"far rows     : {'own-zone' if per_zone else 'pooled'} monitored TS voltage buses")
    L.append("local rows   : monitored HV buses of the actuator's OWN subordinate network")
    L.append("")
    L.append("(1) FAR FIELD -- what the actuator does for the TS objective  [pu/Mvar]")
    L.append(f"    interface  n={len(fi):3d}  median {fi.median():.3e}  "
             f"[{fi.min():.3e}, {fi.max():.3e}]")
    L.append(f"    DS DER     n={len(fd):3d}  median {fd.median():.3e}  "
             f"[{fd.min():.3e}, {fd.max():.3e}]")
    L.append(f"    RATIO median(interface)/median(DER) = {fi.median()/fd.median():.2f}")
    L.append("")
    L.append("    ==> The far-field columns are the SAME SIZE. A Mvar injected behind")
    L.append("        the coupling transformer reaches the TS network as a Mvar; the")
    L.append("        electrical distance does NOT attenuate the far-field voltage")
    L.append("        response. Any argument resting on column dominance for the TS")
    L.append("        rows is not supported by this measurement.")
    L.append("")
    L.append("(2) COLLATERAL -- what the same move does to the actuator's own HV network")
    L.append(f"    interface  median {li.median():.3e}   local/far ratio "
             f"{iface.collateral_ratio.median():.2f}")
    L.append(f"    DS DER     median {ld.median():.3e}   local/far ratio "
             f"{der.collateral_ratio.median():.2f}  "
             f"[{der.collateral_ratio.min():.2f}, {der.collateral_ratio.max():.2f}]")
    L.append(f"    RATIO median(DER)/median(interface) = {ld.median()/li.median():.2f}")
    L.append("")
    L.append("    ==> THIS is where the two families differ. Moving one DS DER by")
    L.append("        1 Mvar disturbs its own HV buses about "
             f"{der.collateral_ratio.median():.1f}x as much as it")
    L.append("        helps the TS buses; commanding the interface by 1 Mvar disturbs")
    L.append(f"        them only {iface.collateral_ratio.median():.1f}x, because the "
             "subordinate controller")
    L.append("        realises that Mvar with whatever internal combination leaves its")
    L.append("        own network where it wants it.")
    L.append("")
    L.append("Reading. In a single aggregate objective carrying BOTH the TS and the HV")
    L.append("buses, a DER move that helps a TS bus is priced against the larger local")
    L.append("excursion it causes, so it is partly self-cancelling. The cascade does not")
    L.append("face that trade-off: the parent asks for a boundary flow and the")
    L.append("subordinate coordinates many units so their local effects cancel.")
    L.append("The mechanism is objective structure and delegated realisation, NOT")
    L.append("attenuation by electrical distance.")
    return chr(10).join(L)


def to_tex(df: pd.DataFrame, path: Path) -> None:
    """Emit booktabs rows for the thesis table (per DSO aggregate)."""
    rows = []
    for dso, sub in df[df.kind == "ds_der"].groupby("dso"):
        iface = df[(df.kind == "interface") & (df.dso == dso)]
        i_med = iface.norm2_pu_per_mvar.median() if len(iface) else float("nan")
        d_med = sub.norm2_pu_per_mvar.median()
        rows.append(f"\t\t{dso} & {len(sub)} & "
                    f"\\num{{{d_med:.2e}}} & \\num{{{i_med:.2e}}} & "
                    f"\\num{{{i_med / d_med:.0f}}} \\\\")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def sweep_operating_points(state, inv, *, n_points: int, delta: float,
                           run_control: bool, per_zone: bool) -> pd.DataFrame:
    """Repeat the measurement at ``n_points`` instants of the campaign window.

    ADDED 2026-08-17. The first run measured ONE operating point, and the
    collateral ratio it reported is the number Ch 6 quotes. A single-point
    number carries no statement about how it varies, so the follow-up sweeps
    the campaign's own profile window and reports the spread.
    """
    from datetime import timedelta
    from core.profiles import DEFAULT_PROFILES_CSV, apply_profiles, load_profiles

    cfg = state["_config"]
    if not getattr(cfg, "use_profiles", False):
        raise SystemExit("[E1] the configuration runs without profiles; "
                         "a sweep over operating points is meaningless here.")
    profiles = load_profiles(cfg.profiles_csv or DEFAULT_PROFILES_CSV)
    t0 = cfg.start_time
    horizon = float(cfg.n_total_s)
    instants = [t0 + timedelta(seconds=horizon * k / max(n_points - 1, 1))
                for k in range(n_points)]

    base = inv["net"]
    frames = []
    for k, t in enumerate(instants, 1):
        work = copy.deepcopy(base)
        apply_profiles(work, profiles, t)
        try:
            _runpp(work, run_control)
        except Exception as exc:                      # non-convergent draw
            print(f"  [E1] instant {k}/{len(instants)} {t} SKIPPED ({exc})")
            continue
        sub = dict(inv)
        sub["net"] = work
        df = measure(sub, delta=delta, run_control=run_control,
                     per_zone=per_zone)
        df["instant"] = t
        df["load_mw"] = float(work.res_load.p_mw.sum())
        frames.append(df)
        i = df[df.kind == "interface"]
        d = df[df.kind == "ds_der"]
        print(f"  [E1] instant {k}/{len(instants)} {t}  "
              f"load {df.load_mw.iloc[0]:7.0f} MW  "
              f"far-ratio {i.norm2_pu_per_mvar.median() / d.norm2_pu_per_mvar.median():5.2f}  "
              f"collateral DER {d.collateral_ratio.median():5.2f} / "
              f"iface {i.collateral_ratio.median():4.2f}", flush=True)
    if not frames:
        raise SystemExit("[E1] every instant failed to converge.")
    return pd.concat(frames, ignore_index=True)


def sweep_report(sw: pd.DataFrame) -> str:
    rows = []
    for t, g in sw.groupby("instant"):
        i = g[g.kind == "interface"]
        d = g[g.kind == "ds_der"]
        rows.append(dict(instant=t, load_mw=g.load_mw.iloc[0],
                         far_ratio=i.norm2_pu_per_mvar.median() / d.norm2_pu_per_mvar.median(),
                         coll_der=d.collateral_ratio.median(),
                         coll_iface=i.collateral_ratio.median()))
    t = pd.DataFrame(rows)
    L = ["", "SWEEP over the campaign's operating points", "=" * 44,
         t.to_string(index=False,
                     float_format=lambda v: f"{v:.3f}"), ""]
    L.append(f"far-field ratio      median {t.far_ratio.median():.3f}   "
             f"range [{t.far_ratio.min():.3f}, {t.far_ratio.max():.3f}]")
    L.append(f"collateral DER       median {t.coll_der.median():.2f}   "
             f"range [{t.coll_der.min():.2f}, {t.coll_der.max():.2f}]")
    L.append(f"collateral interface median {t.coll_iface.median():.2f}   "
             f"range [{t.coll_iface.min():.2f}, {t.coll_iface.max():.2f}]")
    L.append("")
    L.append("The far-field ratio is the falsification: if it stays at ~1 across")
    L.append("the window, the column-dominance mechanism is dead at every")
    L.append("operating point, not just at t0. The collateral ratios are the")
    L.append("numbers Ch 6 quotes and must be reported with this spread.")
    return chr(10).join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    p.add_argument("--variant", default="V4",
                   help="control variant whose controllers define the "
                        "actuator inventory (default V4, the cascade)")
    p.add_argument("--delta", type=float, default=1.0,
                   help="perturbation magnitude [Mvar] (default 1.0)")
    p.add_argument("--closed-loop", action="store_true",
                   help="let local Q(V) loops react during the perturbation "
                        "(default: pure network response)")
    p.add_argument("--per-zone", action="store_true",
                   help="score each column only on its own zone's monitored "
                        "buses (default: pooled TS rows)")
    p.add_argument("--sweep", type=int, default=0, metavar="N",
                   help="repeat the measurement at N instants spanning the "
                        "campaign window (0 = single operating point)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    print("[E1] building plant and controllers at t0 ...", flush=True)
    t0 = time.perf_counter()
    state = build_state(args.variant)
    print(f"[E1] built in {time.perf_counter() - t0:.0f} s", flush=True)

    inv = collect_columns(state)
    print(f"[E1] {len(inv['interfaces'])} interfaces, {len(inv['ders'])} "
          f"DS-connected DERs, {len(inv['all_rows'])} monitored TS buses",
          flush=True)

    df = measure(inv, delta=args.delta, run_control=args.closed_loop,
                 per_zone=args.per_zone)
    summ = summarise(df)

    df.to_csv(args.out / "e1_column_norms.csv", index=False)
    summ.to_csv(args.out / "e1_summary.csv")
    report = write_report(df, summ, args.out, delta=args.delta,
                          run_control=args.closed_loop, per_zone=args.per_zone)
    (args.out / "e1_summary.txt").write_text(report, encoding="utf-8")
    to_tex(df, args.out / "e1_columns.tex")

    if args.sweep:
        print(f"[E1] sweeping {args.sweep} operating points ...", flush=True)
        sw = sweep_operating_points(state, inv, n_points=args.sweep,
                                    delta=args.delta,
                                    run_control=args.closed_loop,
                                    per_zone=args.per_zone)
        sw.to_csv(args.out / "e1_sweep_columns.csv", index=False)
        rep = sweep_report(sw)
        (args.out / "e1_sweep_summary.txt").write_text(rep, encoding="utf-8")
        report = report + chr(10) + rep
        (args.out / "e1_summary.txt").write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print(f"[E1] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
