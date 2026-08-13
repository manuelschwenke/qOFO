#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007_TIE_BOUNDARY_COMPARE.py
==================================================
Which boundary condition should represent a neighbouring TSO area in an
area's own reduced model?

The reduced zone net of ``sensitivity/network_reduction.py`` keeps every
incident tie line together with its far-end terminal bus and condenses the
neighbouring area behind that bus.  Historically that condensation is a
constant PQ load at the cached corridor flow.  This script measures whether
that is the right choice, by comparing three boundary conventions against
the truth:

* ``pq`` -- constant PQ load (infinite Thevenin impedance behind the bus)
* ``pv`` -- PV gen at the cached far-end voltage and active in-feed, Q free
            (zero Thevenin impedance)
* ``z``  -- constant admittance matched to the cached flow (finite, but only
            well posed where the equivalent absorbs)

``pq`` and ``pv`` bracket the true finite-impedance equivalent, so the
spread between them is the modelling uncertainty the choice introduces.

Method
------
All three variants reproduce the SAME operating point at the far-end bus by
construction.  They differ only in the derivative, which is the entire
content of the controller's H matrix.  So the experiment compares H, not a
closed-loop trajectory:

1.  Run ``run_multi_tso_dso`` once per variant, stopping at
    ``pre_loop_hook`` -- this gives the production plant net and the
    production reduced net per zone, with no setup replicated here.
2.  ``H_truth[z]`` = finite-difference H of zone ``z``'s controller
    evaluated on the FULL interconnected plant net, neighbour actuators
    held fixed.  This is the quantity every reduced model is trying to
    approximate.
3.  ``H_var[z]`` = the same finite-difference estimator on the reduced net
    of each variant.

Using one estimator throughout removes the analytical-formula bias from the
comparison, and comparing H rather than a closed-loop run removes the
tuning confound (``G_w`` was BO-tuned with the ``pq`` model in place, so a
closed-loop swap would measure tuning mismatch, not the equivalent).

Both the plant copy and the reduced copies have their pandapower controller
table emptied before perturbation, so the DER Q(V) droop is frozen at its
converged setpoints in every case.  Synchronous-machine AVR action is
retained everywhere -- it is inherent in the PV-bus model, not in the
controller loop -- and it is the dominant boundary-stiffening effect.

Reported per zone and per variant
---------------------------------
* ``relF``      -- ||H_var - H_truth||_F / ||H_truth||_F, whole matrix
* ``relF_corr`` -- the same restricted to the corridor-terminal voltage rows
                   (the quantity BRC-H actually tracks)
* ``gain``      -- median over columns of ||H_var[:,k]|| / ||H_truth[:,k]||
                   on the corridor rows.  > 1 = the model over-states the
                   area's authority at the boundary and the MIQP over-steps;
                   < 1 = under-states it and the loop is sluggish.

Usage
-----
    python experiments/CIGRE_2026/007_TIE_BOUNDARY_COMPARE.py
    python experiments/CIGRE_2026/007_TIE_BOUNDARY_COMPARE.py --variants pq,pv

Writes ``results/007_tie_boundary/h_fidelity.csv``.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-11
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandapower as pp  # noqa: E402

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.numerical_h import compute_numerical_h_tso  # noqa: E402

# Reuse the case-study configuration verbatim (module name starts with a
# digit, so it needs importlib rather than a plain import).
import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

make_cigre_config = _CIGRE.make_cigre_config
VARIANTS = _CIGRE.VARIANTS

OUT_DIR = _ROOT / "results" / "007_tie_boundary"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _freeze(net) -> Any:
    """Deep-copy *net* and empty its controller table.

    The reduced nets are deep copies of the plant and therefore inherit
    pandapower controllers, some of which reference elements the reduction
    dropped.  ``compute_numerical_h_tso`` calls ``runpp(run_control=True)``
    for its baseline regardless of ``closed_loop``, which would trip over
    those stale controllers.  Emptying the table on BOTH the plant copy and
    the reduced copies keeps the comparison symmetric: every net is a
    frozen-droop algebraic plant with AVR still active through the PV-bus
    model.
    """
    work = copy.deepcopy(net)
    if hasattr(work, "controller") and len(work.controller) > 0:
        work.controller.drop(index=work.controller.index, inplace=True)
    return work


def _capture_state(cfg) -> Dict[str, Any]:
    """Run the production setup and stop at the pre-loop hook."""
    captured: Dict[str, Any] = {}

    def _hook(state: Dict[str, Any]) -> bool:
        captured.update(state)
        return True  # skip the main loop

    run_multi_tso_dso(cfg, pre_loop_hook=_hook)
    if not captured:
        raise RuntimeError("pre_loop_hook never fired -- setup aborted early?")
    return captured


def _build_config(variant: str):
    """``make_cigre_config()`` + the V4 (cascaded, proposed) overrides."""
    cfg = make_cigre_config()
    for k, v in VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.tie_boundary_equivalent = variant
    cfg.verbose = 0
    # Nothing below the hook runs, but keep the setup quiet and deterministic.
    cfg.run_stability_analysis = False
    return cfg


def _corridor_rows(ctrl, zd) -> List[int]:
    """Row indices of the corridor-terminal voltages in the H layout.

    Rows are ``[V_bus | Q_PCC | I_line | Q_gen]``, so the corridor-terminal
    voltages sit in the leading V block at the position of their bus in
    ``cfg.voltage_bus_indices``.
    """
    vbuses = list(ctrl.config.voltage_bus_indices)
    rows: List[int] = []
    for b in getattr(zd, "tie_line_endpoint_buses", []) or []:
        b = int(b)
        if b in vbuses:
            r = vbuses.index(b)
            if r not in rows:
                rows.append(r)
    return sorted(rows)


def _metrics(
    h_var: np.ndarray, h_truth: np.ndarray, corr_rows: List[int]
) -> Dict[str, float]:
    """Fidelity of one variant's H against the truth."""
    out: Dict[str, float] = {}

    den = float(np.linalg.norm(h_truth))
    out["relF"] = float(np.linalg.norm(h_var - h_truth) / den) if den > 0 else np.nan

    if corr_rows:
        a = h_var[corr_rows, :]
        b = h_truth[corr_rows, :]
        den_c = float(np.linalg.norm(b))
        out["relF_corr"] = (
            float(np.linalg.norm(a - b) / den_c) if den_c > 0 else np.nan
        )
        # Column-wise gain ratio on the corridor rows.  Columns whose true
        # influence on the corridor is numerically nil carry no information
        # about over- or under-statement, so they are dropped rather than
        # allowed to blow the ratio up.
        num = np.linalg.norm(a, axis=0)
        dnm = np.linalg.norm(b, axis=0)
        thresh = 1e-3 * float(np.max(dnm)) if dnm.size and np.max(dnm) > 0 else 0.0
        keep = dnm > max(thresh, 1e-12)
        out["gain"] = float(np.median(num[keep] / dnm[keep])) if keep.any() else np.nan
        out["n_cols_gain"] = int(keep.sum())
    else:
        out["relF_corr"] = np.nan
        out["gain"] = np.nan
        out["n_cols_gain"] = 0
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pq,pv,z",
                    help="comma-separated subset of pq,pv,z")
    ap.add_argument("--delta-q", type=float, default=1.0,
                    help="Mvar perturbation for DER / PCC_set columns")
    ap.add_argument("--delta-v", type=float, default=0.001,
                    help="p.u. perturbation for V_gen columns")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TIE-LINE BOUNDARY EQUIVALENT -- H FIDELITY AGAINST THE FULL PLANT")
    print("=" * 78)

    # ── Setup per variant (the plant net is identical; only the reduced
    #    nets differ, so the truth is built once from the first run) ──────
    states: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        print(f"\n[setup] variant '{v}': running production setup to hook ...")
        states[v] = _capture_state(_build_config(v))
        print(f"[setup] variant '{v}': done "
              f"({len(states[v]['tso_controllers'])} TSO zones)")

    ref = states[variants[0]]
    plant = ref["net"]
    zone_defs = ref["zone_defs"]
    zones = sorted(ref["tso_controllers"].keys())

    # Sanity: the plant net must not depend on the boundary flag.
    for v in variants[1:]:
        d = float(np.max(np.abs(
            states[v]["net"].res_bus["vm_pu"].values
            - plant.res_bus["vm_pu"].values
        )))
        if d > 1e-9:
            print(f"  !! plant net differs between 'pq' and '{v}' "
                  f"(max |dV| = {d:.2e}) -- truth is not shared; aborting")
            return 1
    print("\n[check] plant net identical across variants (max |dV| < 1e-9) OK")

    rows: List[Dict[str, Any]] = []

    for z in zones:
        ctrl = ref["tso_controllers"][z]
        zd = zone_defs[z]
        corr_rows = _corridor_rows(ctrl, zd)
        n_ties = len(getattr(zd, "tie_line_indices", []) or [])

        print("\n" + "-" * 78)
        print(f"ZONE {z}:  {n_ties} tie line(s), "
              f"{len(corr_rows)} corridor-terminal voltage row(s), "
              f"{len(ctrl.config.voltage_bus_indices)} monitored buses")
        print("-" * 78)
        if not corr_rows:
            print("  (no monitored corridor terminal -- corridor metrics are NaN)")

        print("  building truth H on the full plant net ...", flush=True)
        h_truth = compute_numerical_h_tso(
            _freeze(plant), ctrl,
            delta_q_mvar=args.delta_q, delta_v_pu=args.delta_v,
            closed_loop=False,
        )
        print(f"    H_truth shape {h_truth.shape}, "
              f"||H||_F = {np.linalg.norm(h_truth):.4g}")

        for v in variants:
            red = states[v]["tso_controllers"][z].sensitivities.net
            print(f"  building H on reduced net '{v}' "
                  f"({len(red.bus)} buses) ...", flush=True)
            h_var = compute_numerical_h_tso(
                _freeze(red), ctrl,
                delta_q_mvar=args.delta_q, delta_v_pu=args.delta_v,
                closed_loop=False,
            )
            m = _metrics(h_var, h_truth, corr_rows)
            m.update(zone=z, variant=v, n_ties=n_ties,
                     n_buses_reduced=int(len(red.bus)),
                     fro_truth=float(np.linalg.norm(h_truth)),
                     fro_var=float(np.linalg.norm(h_var)))
            rows.append(m)
            print(f"    relF = {m['relF']:.4f}   "
                  f"relF_corr = {m['relF_corr']:.4f}   "
                  f"gain = {m['gain']:.4f}  (n={m['n_cols_gain']} cols)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hdr = f"{'zone':>5} {'variant':>8} {'relF':>9} {'relF_corr':>10} {'gain':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['zone']:>5} {r['variant']:>8} {r['relF']:>9.4f} "
              f"{r['relF_corr']:>10.4f} {r['gain']:>8.4f}")

    print("\nper-variant mean over zones:")
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        print(f"  {v:>3}:  relF {np.nanmean([r['relF'] for r in sub]):.4f}   "
              f"relF_corr {np.nanmean([r['relF_corr'] for r in sub]):.4f}   "
              f"gain {np.nanmean([r['gain'] for r in sub]):.4f}")

    out_csv = OUT_DIR / "h_fidelity.csv"
    keys = ["zone", "variant", "n_ties", "n_buses_reduced", "relF",
            "relF_corr", "gain", "n_cols_gain", "fro_truth", "fro_var"]
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\nwritten: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
