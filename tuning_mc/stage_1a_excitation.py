"""
tuning_mc/stage_1a_excitation.py
================================
Stage 1a — choose the tuning windows from *measurement* instead of assertion.

The problem this solves
-----------------------
A scenario set justified by design intent ("this one stresses the taps") is a
claim.  The previous campaign shows what that costs: three of five coordinates
turned out to carry no signal, the MSC/MSR banks never committed on the set
they were tuned against, and ``tau`` was structurally inert in the one scenario
that dominated the objective -- all discovered *after* 12 h of search.

So the set is selected here, before any tuning, from two screens.

Screen 1 — reactive capability, exact and free
----------------------------------------------
DER reactive capability under ``VDE-AR-N-4120-v2`` is **not** a smooth function
of active power; it has a hard dead zone:

    P/Sn < 0.1   ->  Q capability is exactly ZERO
    P/Sn = 0.1   ->  +/-0.10 Sn
    P/Sn >= 0.2  ->  -0.33 .. +0.41 Sn   (saturated)

So a window's DER reactive capability is decided by how many machines clear
P/Sn = 0.1, and it can be computed from the profiles alone -- no simulation.
This matters because every knob that allocates reactive effort (``tau``,
``lambda_dso``, and the DSO objective trade-off) is **structurally inert** in a
window where capability is zero: there is nothing to allocate.  Screen 1 is run
over the whole profile year and costs seconds per hundred windows.

The capability curve is imported from the plant's own implementation
(:func:`controller.der_qv_local_loop._qv_capability`) rather than re-typed, so
this screen cannot drift away from what the simulation actually enforces.

Screen 2 — measured actuator activity
-------------------------------------
For the windows that survive Screen 1, one quiescent 90-min run at the Stage-0
design point records what the *operating point itself* excites: DER Q range
used, PCC Q movement, tap operations, tap sign changes, interface-Q excursion,
voltage band excursion.  Disturbances are **not** included here on purpose --
see below.

What is selected and what is designed
-------------------------------------
Operating points are *selected* from data.  Disturbances are *injected by
design*, because a gen trip or a line outage does not occur in a profile year
at all; their justification is the N-1 requirement, not the measurement.  The
final set is the factorial of the two, and the two halves are argued
differently -- conflating them is what made the first hand-picked set
indefensible.

A worked example of the second kind, from the operator (2026-08-14): run the
generator trip at ``2016-01-05 08:00`` because *a system must carry enough
dispatched reactive reserve to be N-1 secure at that hour*, so the window is
representative of a condition the controller is required to handle.  Screen 1
tests the premise (is capability really available there?) rather than assuming
it.

Selection criterion, fixed in advance
-------------------------------------
Normalise each excitation column, then greedy **maximin**: repeatedly add the
window that most raises the weakest column.  This selects for breadth -- five
windows that all stress the same loop score badly -- and it is reproducible
from the recorded matrix rather than from taste.

Usage::

    python -m tuning_mc.stage_1a_excitation --screen1 --stride-h 2
    python -m tuning_mc.stage_1a_excitation --screen1 --windows 2016-01-05T08:00
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_BASELINE = (_REPO_ROOT / "tuning" / "scripts" / "configs"
                    / "baseline_ieee39_thevenin.yaml")
DEFAULT_OUT = _REPO_ROOT / "results" / "tuning_mc" / "stage1a"

#: Below this P/Sn a DER has exactly zero reactive capability (VDE dead zone).
DEAD_ZONE_P_RATIO = 0.1


def _build_plant(baseline: Path, network: str, start: datetime):
    """Build net + controllers once, aborting before the time loop."""
    from tuning._io import load_config_yaml
    from tuning._sim_loader import get_run_multi_tso_dso

    cfg = load_config_yaml(baseline)
    cfg = dataclasses.replace(
        cfg, scenario=network, start_time=start, n_total_s=60.0,
        contingencies=[], verbose=0, live_plot_controller=False,
        live_plot_cascade=False, live_plot_system=False,
        run_stability_analysis=False, precondition_g_w=False,
    )
    cap: dict[str, Any] = {}

    def hook(state):
        cap.update(state)
        return True

    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)
    return cfg, cap


def _der_groups(cap: dict[str, Any]) -> dict[str, list[int]]:
    """``{loop label: sgen indices}`` taken from the controllers themselves."""
    groups: dict[str, list[int]] = {}
    for z, ctrl in sorted(cap.get("tso_controllers", {}).items()):
        idx = list(getattr(ctrl.config, "der_indices", []) or [])
        if idx:
            groups[f"TSO-z{z}"] = [int(i) for i in idx]
    for d, ctrl in sorted(cap.get("dso_controllers", {}).items()):
        idx = list(getattr(ctrl.config, "der_indices", []) or [])
        if idx:
            groups[f"DSO-{d}"] = [int(i) for i in idx]
    return groups


def _capability(net, sgen_idx: list[int]) -> dict[str, float]:
    """Aggregate DER reactive capability over a set of sgens, at net's state.

    Uses the plant's own capability curve so the screen cannot disagree with
    what the simulation enforces.
    """
    from controller.der_qv_local_loop import _qv_capability

    q_min_tot = q_max_tot = 0.0
    n_live = 0
    p_ratios: list[float] = []
    for i in sgen_idx:
        if i not in net.sgen.index:
            continue
        row = net.sgen.loc[i]
        sn = float(row.get("sn_mva", 0.0) or 0.0)
        p = float(row.get("p_mw", 0.0) or 0.0)
        if sn <= 0.0:
            continue
        op = str(row.get("op_diagram", "VDE-AR-N-4120-v2")
                 or "VDE-AR-N-4120-v2")
        q_min, q_max = _qv_capability(sn, op, p)
        q_min_tot += float(q_min)
        q_max_tot += float(q_max)
        r = abs(p) / sn
        p_ratios.append(r)
        if r >= DEAD_ZONE_P_RATIO:
            n_live += 1
    return {
        "q_min_mvar": q_min_tot, "q_max_mvar": q_max_tot,
        "q_range_mvar": q_max_tot - q_min_tot,
        "n_der": len(p_ratios), "n_above_deadzone": n_live,
        "frac_above_deadzone": (n_live / len(p_ratios)) if p_ratios else 0.0,
        "p_ratio_median": float(np.median(p_ratios)) if p_ratios else 0.0,
        "p_ratio_max": float(max(p_ratios)) if p_ratios else 0.0,
    }


def screen1(args) -> int:
    """Sweep candidate start times; report DER capability per loop."""
    from core.profiles import DEFAULT_PROFILES_CSV, apply_profiles, load_profiles

    start0 = datetime.fromisoformat(args.year_start)
    cfg, cap = _build_plant(Path(args.baseline), args.network, start0)
    groups = _der_groups(cap)
    net = cap["net"]
    if not groups:
        raise SystemExit("[1a] no DER index sets found on the controllers")
    print(f"[1a] network={args.network}  DER groups: "
          f"{ {k: len(v) for k, v in groups.items()} }", flush=True)

    profiles = load_profiles(cfg.profiles_csv or DEFAULT_PROFILES_CSV,
                             timestep_s=cfg.dt_s)

    if args.windows:
        stamps = [datetime.fromisoformat(w) for w in args.windows.split(",")]
    else:
        stamps = []
        t = start0
        end = start0 + timedelta(days=args.days)
        while t < end:
            stamps.append(t)
            t += timedelta(hours=args.stride_h)

    rows: list[dict[str, Any]] = []
    for t in stamps:
        apply_profiles(net, profiles, t)
        entry: dict[str, Any] = {"timestamp": t.isoformat(),
                                 "iso_week": t.isocalendar()[1],
                                 "hour": t.hour,
                                 "load_p_mw": float(net.load.p_mw.sum()),
                                 "sgen_p_mw": float(net.sgen.p_mw.sum())}
        for label, idx in groups.items():
            c = _capability(net, idx)
            for k, v in c.items():
                entry[f"{label}.{k}"] = v
        entry["q_range_total_mvar"] = sum(
            entry[f"{label}.q_range_mvar"] for label in groups)
        entry["frac_above_deadzone_min"] = min(
            entry[f"{label}.frac_above_deadzone"] for label in groups)
        rows.append(entry)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen1.json").write_text(json.dumps(rows, indent=1),
                                      encoding="utf-8")

    print(f"\n{'timestamp':<20}{'wk':>4}{'load MW':>10}{'DER MW':>9}"
          f"{'Qrange Mvar':>13}{'min frac>dz':>13}   per-loop frac above dead zone")
    show = rows if args.windows else sorted(
        rows, key=lambda r: -r["q_range_total_mvar"])[:args.top]
    for r in show:
        per = "  ".join(
            f"{lbl.split('-')[-1]}={100 * r[f'{lbl}.frac_above_deadzone']:.0f}%"
            for lbl in groups)
        print(f"{r['timestamp']:<20}{r['iso_week']:>4}{r['load_p_mw']:>10.0f}"
              f"{r['sgen_p_mw']:>9.0f}{r['q_range_total_mvar']:>13.1f}"
              f"{100 * r['frac_above_deadzone_min']:>12.0f}%   {per}")
    if not args.windows:
        worst = min(rows, key=lambda r: r["q_range_total_mvar"])
        print(f"\n[1a] worst window: {worst['timestamp']} "
              f"Qrange={worst['q_range_total_mvar']:.1f} Mvar")
        zero = [r for r in rows if r["q_range_total_mvar"] <= 1e-9]
        print(f"[1a] windows with EXACTLY ZERO DER capability: "
              f"{len(zero)} / {len(rows)} "
              f"({100 * len(zero) / max(len(rows), 1):.0f} %) -- every "
              f"reactive-allocation knob is inert in these")
    print(f"[1a] wrote {out / 'screen1.json'} ({len(rows)} windows)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.stage_1a_excitation")
    p.add_argument("--screen1", action="store_true")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--network", default="rural_700")
    p.add_argument("--year-start", default="2016-01-01T00:00:00")
    p.add_argument("--days", type=int, default=366)
    p.add_argument("--stride-h", type=float, default=2.0)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--windows", default=None,
                   help="Comma-separated ISO timestamps to evaluate instead of "
                        "sweeping, e.g. '2016-01-05T08:00,2016-07-10T03:00'.")
    args = p.parse_args(argv)
    if args.screen1:
        return screen1(args)
    p.error("nothing to do: pass --screen1")
    return 2


if __name__ == "__main__":
    sys.exit(main())
