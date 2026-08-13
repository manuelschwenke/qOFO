#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007f_ZTH_PER_CORRIDOR.py
===============================================
Is the fitted boundary stiffness k ~ 1.5 the PHYSICAL Thevenin impedance, or
just a number that happened to work?

``007d`` swept the boundary impedance as a multiple ``k`` of each tie line's
own series impedance and found the corridor-row H error minimised at
k in [1, 2] in all three zones.  That is a fit.  This script computes the
impedance the neighbouring system ACTUALLY presents at each corridor
far-end bus and asks whether it agrees.

If it does, ``k ~ 1.5`` stops being a fitted constant: it becomes a measured
consequence, the per-corridor route (one number from the neighbour, folded
into the BRC-H agreement) becomes the refinement rather than a competing
option, and the whole boundary model is derived rather than tuned.

Method
------
For each tie line of each zone, with far-end bus ``b`` in the neighbour:

1.  Delete the zone entirely from a copy of the plant -- its buses, its
    subordinate DSOs, and (by cascade) the tie lines themselves.  What
    remains is exactly the external system the zone is condensing, still
    containing ``b``.
2.  Restore a slack if the zone happened to own the system one.
3.  Perturb ``b``: inject +/- dQ, re-solve, read d|V_b|.  Repeat with +/- dP.

For a source ``E`` behind ``Z = R + jX`` feeding an injection at ``b``,

    X ~ V_b * d|V_b|/dQ ,    R ~ V_b * d|V_b|/dP     (all per unit)

so the perturbation directly yields the Thevenin impedance.  Measuring it by
perturbation rather than from Ybus is deliberate: it is the same "perturb and
measure" definition the H comparison uses, and it automatically embeds the
AVR behaviour of the neighbour's machines, which hold their terminal voltage
and therefore act as sources.  That is the right stiffness for a STEADY-STATE
QV sensitivity -- and it is NOT the fault-study Sk'', which embeds the
subtransient reactance Xd'' and would give a boundary that is too stiff.

Linearity is checked by evaluating at two perturbation sizes.

Output
------
Per corridor: |Z_th| in ohms, the tie line's own |Z_line|, and their ratio
k_phys = |Z_th| / |Z_line| -- directly comparable with the fitted optimum of
``007d``.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-11
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandapower as pp

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.network_reduction import line_series_impedance_ohm  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

OUT_DIR = _ROOT / "results" / "007_tie_boundary"
S_BASE_MVA = 100.0

#: Fitted per-zone optima from 007d (corridor-row H error), for reference.
FITTED_K = {1: 2.0, 2: 1.0, 3: 2.0}


def _capture() -> Dict[str, Any]:
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    got: Dict[str, Any] = {}

    def _hook(state):
        got.update(state)
        return True

    run_multi_tso_dso(cfg, pre_loop_hook=_hook)
    return got


def _runpp(net) -> bool:
    for init in ("results", "dc", "flat"):
        try:
            pp.runpp(net, run_control=False, distributed_slack=False,
                     calculate_voltage_angles=True, enforce_q_lims=False,
                     init=init, max_iteration=200)
            return True
        except Exception:
            continue
    return False


def _external_net(plant, zone_bus_indices: List[int]) -> Optional[Any]:
    """The plant with one zone deleted: exactly what that zone condenses.

    ``pp.drop_buses`` cascades, so the tie lines go with the zone's buses --
    which is what we want, since the Thevenin impedance is defined with the
    corridor open.
    """
    ext = copy.deepcopy(plant)
    if hasattr(ext, "controller") and len(ext.controller) > 0:
        ext.controller.drop(index=ext.controller.index, inplace=True)

    drop = [int(b) for b in zone_bus_indices if int(b) in ext.bus.index]
    if drop:
        pp.drop_buses(ext, drop)

    # The deleted zone may have owned the system slack.
    has_slack = (
        ("slack" in ext.gen.columns and bool(ext.gen["slack"].any()))
        or not ext.ext_grid.empty
    )
    if not has_slack:
        if ext.gen.empty:
            return None
        in_svc = ext.gen.index[ext.gen["in_service"].astype(bool)]
        if not len(in_svc):
            return None
        big = int(ext.gen.loc[in_svc, "sn_mva"].idxmax()
                  if "sn_mva" in ext.gen.columns else in_svc[0])
        if "slack" not in ext.gen.columns:
            ext.gen["slack"] = False
        ext.gen["slack"] = False
        ext.gen.at[big, "slack"] = True
    return ext if _runpp(ext) else None


def _z_th_ohm(ext, bus: int, d_mva: float) -> Optional[complex]:
    """Thevenin impedance at *bus* by central-difference perturbation.

    X = V * dV/dQ and R = V * dV/dP, evaluated in per unit on
    (S_BASE_MVA, vn_kv) and returned in ohms.
    """
    if bus not in ext.bus.index:
        return None
    vn = float(ext.bus.at[bus, "vn_kv"])
    z_base = vn ** 2 / S_BASE_MVA

    def _v_with(p_mw: float, q_mvar: float) -> Optional[float]:
        w = copy.deepcopy(ext)
        # Negative load = injection.
        pp.create_load(w, bus=int(bus), p_mw=-p_mw, q_mvar=-q_mvar,
                       name="_ZTH_PROBE")
        if not _runpp(w):
            return None
        return float(w.res_bus.at[bus, "vm_pu"])

    v0 = float(ext.res_bus.at[bus, "vm_pu"])
    vq_p, vq_m = _v_with(0.0, +d_mva), _v_with(0.0, -d_mva)
    vp_p, vp_m = _v_with(+d_mva, 0.0), _v_with(-d_mva, 0.0)
    if None in (vq_p, vq_m, vp_p, vp_m):
        return None

    dq_pu = d_mva / S_BASE_MVA
    dvdq = (vq_p - vq_m) / (2.0 * dq_pu)
    dvdp = (vp_p - vp_m) / (2.0 * dq_pu)
    x_pu, r_pu = v0 * dvdq, v0 * dvdp
    return complex(r_pu * z_base, x_pu * z_base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmva", type=float, default=20.0,
                    help="perturbation size [MW / Mvar]")
    ap.add_argument("--dmva-check", type=float, default=40.0,
                    help="second size, for the linearity check")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st = _capture()
    plant, zone_defs = st["net"], st["zone_defs"]

    print("=" * 78)
    print("PHYSICAL THEVENIN IMPEDANCE PER CORRIDOR")
    print("=" * 78)
    print("k_phys = |Z_th| / |Z_line|  -- compare with the fitted optimum of 007d")

    rows: List[Dict[str, Any]] = []
    for z in sorted(zone_defs.keys()):
        zd = zone_defs[z]
        print("\n" + "-" * 78)
        print(f"ZONE {z}   (deleting it leaves the external system)")
        print("-" * 78)

        ext = _external_net(plant, list(zd.bus_indices))
        if ext is None:
            print("  external system did not converge -- skipped")
            continue
        print(f"  external system: {len(ext.bus)} buses, {len(ext.gen)} gens")

        for li, b_in in zip(zd.tie_line_indices, zd.tie_line_endpoint_buses):
            li, b_in = int(li), int(b_in)
            if li not in plant.line.index:
                continue
            fb = int(plant.line.at[li, "from_bus"])
            tb = int(plant.line.at[li, "to_bus"])
            far = tb if b_in == fb else fb

            z_line = line_series_impedance_ohm(plant, li)
            z_th = _z_th_ohm(ext, far, args.dmva)
            z_th2 = _z_th_ohm(ext, far, args.dmva_check)
            if z_th is None:
                print(f"    line {li:>3} far bus {far:>3}: probe failed")
                continue

            k_phys = abs(z_th) / abs(z_line) if abs(z_line) > 0 else np.nan
            lin = (abs(abs(z_th2) - abs(z_th)) / abs(z_th)
                   if z_th2 is not None and abs(z_th) > 0 else np.nan)
            rows.append(dict(zone=z, line=li, far_bus=far,
                             z_th_ohm=abs(z_th), x_th_ohm=z_th.imag,
                             r_th_ohm=z_th.real, z_line_ohm=abs(z_line),
                             k_phys=k_phys, linearity=lin))
            lin_txt = (f"   [lin dev {lin * 100:4.1f} %]"
                       if np.isfinite(lin) else "   [lin dev n/a]")
            print(f"    line {li:>3} far bus {far:>3}:  "
                  f"|Z_th| = {abs(z_th):7.2f} ohm "
                  f"(R {z_th.real:6.2f}, X {z_th.imag:6.2f})   "
                  f"|Z_line| = {abs(z_line):6.2f} ohm   "
                  f"k_phys = {k_phys:5.2f}{lin_txt}")

        sub = [r for r in rows if r["zone"] == z and np.isfinite(r["k_phys"])]
        if sub:
            ks = np.array([r["k_phys"] for r in sub])
            print(f"  zone {z}: k_phys mean {ks.mean():.2f}  "
                  f"median {np.median(ks):.2f}  range [{ks.min():.2f}, {ks.max():.2f}]"
                  f"   vs fitted k* = {FITTED_K.get(z, float('nan'))}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hdr = (f"{'zone':>5} {'line':>5} {'far':>5} {'|Z_th|':>9} {'|Z_line|':>9} "
           f"{'k_phys':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['zone']:>5} {r['line']:>5} {r['far_bus']:>5} "
              f"{r['z_th_ohm']:>9.2f} {r['z_line_ohm']:>9.2f} "
              f"{r['k_phys']:>8.2f}")

    if rows:
        allk = np.array([r["k_phys"] for r in rows if np.isfinite(r["k_phys"])])
        print(f"\nall corridors: k_phys mean {allk.mean():.2f}  "
              f"median {np.median(allk):.2f}  "
              f"range [{allk.min():.2f}, {allk.max():.2f}]")
        print(f"fitted optima from 007d: "
              f"{', '.join(f'zone {z}: {k}' for z, k in FITTED_K.items())}")
        print("\nper zone, physical vs fitted:")
        for z in sorted({r['zone'] for r in rows}):
            ks = np.array([r["k_phys"] for r in rows if r["zone"] == z])
            print(f"  zone {z}: k_phys mean {ks.mean():5.2f}   "
                  f"fitted k* {FITTED_K.get(z, float('nan')):4.1f}")

    out = OUT_DIR / "zth_per_corridor.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("zone,line,far_bus,r_th_ohm,x_th_ohm,z_th_ohm,z_line_ohm,"
                 "k_phys,linearity\n")
        for r in rows:
            fh.write(f"{r['zone']},{r['line']},{r['far_bus']},{r['r_th_ohm']},"
                     f"{r['x_th_ohm']},{r['z_th_ohm']},{r['z_line_ohm']},"
                     f"{r['k_phys']},{r['linearity']}\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
