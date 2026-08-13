#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007d_THEVENIN_SWEEP.py
=============================================
Is there ONE boundary stiffness that works, and where does it sit between
the PQ and PV extremes 007 measured?

007 showed that condensing a neighbouring TSO area as a constant PQ load
over-states an area's corridor authority (gain 1.4-4.5) while a PV bus
under-states it (0.48-0.65).  Both are limits of the same object -- a
Thevenin source behind an impedance ``Z_th``, with PQ at ``Z_th -> inf`` and
PV at ``Z_th -> 0``.  This script sweeps ``Z_th`` as a multiple ``k`` of the
tie line's own series impedance and finds, per zone, the ``k`` that
minimises the corridor-row H error against the full interconnected plant.

Two questions:

1.  Does the error curve have a clear interior minimum?  If it does, the
    finite-impedance model is genuinely better than either extreme rather
    than just interpolating between them.
2.  Is the optimal ``k`` similar across zones?  If yes, a single default is
    a usable engineering rule and no per-corridor data exchange is needed.
    If no, the boundary needs a per-corridor short-circuit figure.

Cost control
------------
A sweep through the runner would need one full setup per ``k``.  Instead the
production setup runs ONCE and the ``build_tso_local_net`` arguments are
reconstructed from the captured state.  The reconstruction is not trusted:
it is first used to rebuild the ``pq`` net and that net is compared against
the one the runner itself produced.  The sweep only proceeds if they match
element for element.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-11
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.jacobian import JacobianSensitivities  # noqa: E402
from sensitivity.network_reduction import build_tso_local_net  # noqa: E402
from sensitivity.numerical_h import compute_numerical_h_tso  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

OUT_DIR = _ROOT / "results" / "007_tie_boundary"


def _freeze(net):
    """Deep copy with the controller table emptied (see 007, ``_freeze``)."""
    w = copy.deepcopy(net)
    if hasattr(w, "controller") and len(w.controller) > 0:
        w.controller.drop(index=w.controller.index, inplace=True)
    return w


def _capture() -> Dict[str, Any]:
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.tie_boundary_equivalent = "pq"
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    got: Dict[str, Any] = {}

    def _hook(state):
        got.update(state)
        return True

    run_multi_tso_dso(cfg, pre_loop_hook=_hook)
    return got


def _zone_args(st: Dict[str, Any], z: int) -> Dict[str, Any]:
    """Reconstruct the runner's ``build_tso_local_net`` call for zone *z*.

    Mirrors ``_build_tso_local_jac`` and the ``_zone_all_machine_trafos``
    assembly in ``experiments/runners/multi_tso_dso.py``.  Validated against
    the runner's own output before use.
    """
    net, meta, zone_defs = st["net"], st["meta"], st["zone_defs"]
    zd = zone_defs[z]

    # tn_zone_map: the zone map BEFORE the HV/gen-terminal extension, i.e.
    # the TN-level buses only.
    tn_buses = [
        int(b) for b in zd.bus_indices
        if b in net.bus.index and str(net.bus.at[b, "subnet"]) == "TN"
    ]

    # _zone_all_machine_trafos: machine trafos of in-zone gens, plus every
    # OLTC the zone's MIQP can act on (covers gen_idx = -1 interconnectors).
    zone_gen_sets = {
        zz: {int(g) for g in zone_defs[zz].gen_indices} for zz in zone_defs
    }
    machine_trafos: List[int] = []
    for t_idx, g_idx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        if int(g_idx) < 0:
            continue
        for zz in zone_defs:
            if int(g_idx) in zone_gen_sets[zz]:
                if zz == z:
                    machine_trafos.append(int(t_idx))
                break
    for t_idx in zd.oltc_trafo_indices:
        if int(t_idx) not in machine_trafos:
            machine_trafos.append(int(t_idx))

    return dict(
        net=net,
        zone_bus_indices=tn_buses,
        gen_indices_in_zone=zd.gen_indices,
        machine_trafo_indices_in_zone=machine_trafos,
        tie_line_indices=zd.tie_line_indices,
        tie_line_endpoint_buses=zd.tie_line_endpoint_buses,
        hv_networks_in_zone=[hv for hv in meta.hv_networks if int(hv.zone) == z],
        tso_shunt_buses_in_zone=zd.shunt_bus_indices,
        tso_shunt_q_steps_mvar_in_zone=zd.shunt_q_steps_mvar,
    )


def _same_net(a, b) -> bool:
    """Structural equality of two reduced nets on the tables that matter."""
    for tbl, cols in (
        ("bus", ["vn_kv"]),
        ("line", ["from_bus", "to_bus"]),
        ("trafo", ["hv_bus", "lv_bus"]),
        ("trafo3w", ["hv_bus", "mv_bus", "lv_bus"]),
        ("load", ["bus", "p_mw", "q_mvar"]),
        ("sgen", ["bus"]),
        ("gen", ["bus", "vm_pu"]),
        ("shunt", ["bus", "q_mvar"]),
    ):
        da, db = getattr(a, tbl), getattr(b, tbl)
        if len(da) != len(db):
            print(f"    mismatch: {tbl} has {len(da)} vs {len(db)} rows")
            return False
        if len(da) == 0:
            continue
        if sorted(da.index.tolist()) != sorted(db.index.tolist()):
            print(f"    mismatch: {tbl} index differs")
            return False
        for c in cols:
            if c not in da.columns or c not in db.columns:
                continue
            va = da.loc[sorted(da.index), c].to_numpy(dtype=float, na_value=np.nan)
            vb = db.loc[sorted(db.index), c].to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(va, vb, rtol=1e-9, atol=1e-9, equal_nan=True):
                print(f"    mismatch: {tbl}.{c}")
                return False
    return True


def _corridor_rows(ctrl, zd) -> List[int]:
    vb = list(ctrl.config.voltage_bus_indices)
    return sorted({vb.index(int(b)) for b in (zd.tie_line_endpoint_buses or [])
                   if int(b) in vb})


def _corr_err(h_var, h_truth, rows) -> float:
    if not rows:
        return float("nan")
    a, b = h_var[rows, :], h_truth[rows, :]
    d = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / d) if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="0.1,0.25,0.5,1,2,4,8,16",
                    help="comma-separated Z_th / Z_line multipliers")
    args = ap.parse_args()
    ks = [float(x) for x in args.ks.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("THEVENIN BOUNDARY SWEEP -- corridor-row H error vs Z_th / Z_line")
    print("=" * 78)

    st = _capture()
    plant = st["net"]
    zones = sorted(st["tso_controllers"].keys())

    # ── Validate the reconstruction before trusting it ────────────────────
    print("\n[validate] rebuilding the 'pq' nets from reconstructed arguments")
    for z in zones:
        mine = build_tso_local_net(**_zone_args(st, z), tie_boundary="pq").net
        theirs = st["tso_controllers"][z].sensitivities.net
        if not _same_net(mine, theirs):
            print(f"  zone {z}: RECONSTRUCTION MISMATCH -- aborting")
            return 1
        print(f"  zone {z}: matches the runner's own net "
              f"({len(mine.bus)} buses) OK")

    rows: List[Dict[str, Any]] = []
    for z in zones:
        ctrl = st["tso_controllers"][z]
        zd = st["zone_defs"][z]
        cr = _corridor_rows(ctrl, zd)
        base = _zone_args(st, z)

        print("\n" + "-" * 78)
        print(f"ZONE {z}  ({len(cr)} corridor-terminal voltage rows)")
        print("-" * 78)
        print("  truth H on the full plant ...", flush=True)
        h_truth = compute_numerical_h_tso(_freeze(plant), ctrl, closed_loop=False)

        for label, kwargs in (
            [("pq", dict(tie_boundary="pq")), ("pv", dict(tie_boundary="pv"))]
            + [(f"th k={k:g}", dict(tie_boundary="thevenin", tie_thevenin_k=k))
               for k in ks]
        ):
            try:
                red = build_tso_local_net(**base, **kwargs).net
                h = compute_numerical_h_tso(_freeze(red), ctrl, closed_loop=False)
                e = _corr_err(h, h_truth, cr)
            except Exception as exc:
                print(f"    {label:>10}: FAILED ({type(exc).__name__}: {exc})")
                continue
            k_val = kwargs.get("tie_thevenin_k", np.nan)
            rows.append(dict(zone=z, label=label, k=k_val, relF_corr=e))
            print(f"    {label:>10}: relF_corr = {e:.4f}")

        best = min((r for r in rows if r["zone"] == z and np.isfinite(r["relF_corr"])),
                   key=lambda r: r["relF_corr"], default=None)
        if best:
            print(f"  -> best: {best['label']} at relF_corr {best['relF_corr']:.4f}")

    print("\n" + "=" * 78)
    print("SUMMARY -- corridor-row relative H error")
    print("=" * 78)
    labels = sorted({r["label"] for r in rows},
                    key=lambda s: (s.startswith("th"), s))
    hdr = f"{'label':>10}" + "".join(f"{'z' + str(z):>10}" for z in zones)
    print(hdr)
    print("-" * len(hdr))
    for lb in labels:
        line = f"{lb:>10}"
        for z in zones:
            v = [r["relF_corr"] for r in rows if r["zone"] == z and r["label"] == lb]
            line += f"{v[0]:>10.4f}" if v else f"{'--':>10}"
        print(line)

    print("\nbest per zone:")
    for z in zones:
        sub = [r for r in rows if r["zone"] == z and np.isfinite(r["relF_corr"])]
        if sub:
            b = min(sub, key=lambda r: r["relF_corr"])
            print(f"  zone {z}: {b['label']:>10}  relF_corr {b['relF_corr']:.4f}")

    out = OUT_DIR / "thevenin_sweep.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("zone,label,k,relF_corr\n")
        for r in rows:
            fh.write(f"{r['zone']},{r['label']},{r['k']},{r['relF_corr']}\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
