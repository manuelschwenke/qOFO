#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007e_DS_THEVENIN.py
==========================================
Does moving the DS-side boundary source one bus back restore the primary-bus
sensitivities the current model structurally cannot have?

``007c`` established that under the present ``boundary="slack"`` convention
every coupling-transformer primary bus is a slack or PV bus of the DSO's
reduced net, so ``dV/dQ`` there does not merely evaluate to zero -- it does
not exist, and the query raises.  The DSO therefore cannot monitor or
constrain its own transmission-side terminals.  ``007c`` also found a second
artefact: the coupler that happens to be indexed first becomes the angle
reference, and its OLTC column comes out systematically weaker than those of
its otherwise-identical siblings.

This script checks whether ``boundary="thevenin"`` fixes both, and what it
costs in H fidelity against the full interconnected plant.

Reported per DSO:

* whether ``dV/dQ`` at each primary bus is available, and its value;
* OLTC column norms, to see whether the first-coupler asymmetry survives;
* ``relF`` of the DSO H against the numerical truth on the full plant.

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
from sensitivity.network_reduction import build_dso_local_net  # noqa: E402
from sensitivity.numerical_h import compute_numerical_h_dso  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

OUT_DIR = _ROOT / "results" / "007_tie_boundary"


def _freeze(net):
    w = copy.deepcopy(net)
    if hasattr(w, "controller") and len(w.controller) > 0:
        w.controller.drop(index=w.controller.index, inplace=True)
    return w


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


def _probe_primary_dvdq(jac, primaries: List[int]) -> List[str]:
    """One report line per primary bus: value, structural zero, or absent."""
    out: List[str] = []
    for b in primaries:
        try:
            s = jac.compute_dV_dQ_der(
                der_bus_indices=[b], observation_bus_indices=[b],
            )
            v = np.asarray(s[0] if isinstance(s, tuple) else s, dtype=float).ravel()
            if v.size == 0:
                out.append(f"bus {b}: EMPTY")
            elif abs(float(v[0])) < 1e-12:
                out.append(f"bus {b}: {float(v[0]):+.3e} (structurally zero)")
            else:
                out.append(f"bus {b}: {float(v[0]):+.3e} pu/Mvar")
        except Exception as exc:
            out.append(f"bus {b}: UNAVAILABLE ({type(exc).__name__})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="0.5,1,2", help="thevenin_k values")
    args = ap.parse_args()
    ks = [float(x) for x in args.ks.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st = _capture()
    plant = st["net"]
    hv_by_id = {hv.net_id: hv for hv in st["meta"].hv_networks}

    print("=" * 78)
    print("DS BOUNDARY: slack-on-primary vs Thevenin-behind-primary")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    for dso_id, ctrl in sorted(st["dso_controllers"].items()):
        hv = hv_by_id[dso_id]
        primaries = [int(b) for b in hv.coupling_ieee_buses]
        cfg = ctrl.config
        n_der_cols = len({int(plant.sgen.at[s, "bus"]) for s in cfg.der_indices})

        print("\n" + "-" * 78)
        print(f"DSO {dso_id}   primary buses {primaries}")
        print("-" * 78)

        print("  truth H on the full plant ...", flush=True)
        h_truth = compute_numerical_h_dso(_freeze(plant), ctrl, closed_loop=False)

        variants = [("slack", dict(boundary="slack"))]
        variants += [(f"th k={k:g}", dict(boundary="thevenin", thevenin_k=k))
                     for k in ks]

        for label, kw in variants:
            try:
                red = build_dso_local_net(plant, hv, **kw).net
                jac = JacobianSensitivities(red)
            except Exception as exc:
                print(f"  {label:>9}: BUILD FAILED "
                      f"({type(exc).__name__}: {exc})")
                continue

            print(f"  {label:>9}:  ({len(red.bus)} buses)")
            for ln in _probe_primary_dvdq(jac, primaries):
                print(f"             dV/dQ  {ln}")

            # OLTC column norms from a controller whose sensitivities are
            # temporarily swapped to this variant's Jacobian.  The H build
            # is diagnostic only -- a failure must not skip the fidelity
            # number below, which does not depend on it.
            spread = float("nan")
            saved = ctrl.sensitivities
            try:
                ctrl.sensitivities = jac
                ctrl.invalidate_sensitivity_cache()
                H = np.asarray(ctrl._build_sensitivity_matrix(), dtype=float)
                n_oltc = len(getattr(cfg, "oltc_trafo_indices", []) or [])
                if n_oltc and H.shape[1] >= n_der_cols + n_oltc:
                    nrm = np.linalg.norm(
                        H[:, n_der_cols:n_der_cols + n_oltc], axis=0)
                    spread = float(nrm.max() / max(nrm.min(), 1e-30))
                    print(f"             OLTC |col| = "
                          f"[{', '.join(f'{v:.4f}' for v in nrm)}]  "
                          f"max/min = {spread:.3f}")
            except Exception as exc:
                print(f"             OLTC probe unavailable "
                      f"({type(exc).__name__}: {exc})")
            finally:
                ctrl.sensitivities = saved
                ctrl.invalidate_sensitivity_cache()

            # Fidelity, split out on the Q_iface rows.  Those are the DSO's
            # tracked output and a BRANCH flow of the retained coupler, so
            # they are the rows that decide this boundary choice -- the
            # whole-matrix figure is diluted by the V and I blocks.
            # DSO row layout is [Q_iface | V_bus | I_line].
            n_if = len(getattr(cfg, "interface_trafo_indices", []) or [])
            try:
                h_num = compute_numerical_h_dso(
                    _freeze(red), ctrl, closed_loop=False)
                den = float(np.linalg.norm(h_truth))
                relF = (float(np.linalg.norm(h_num - h_truth) / den)
                        if den > 0 else float("nan"))
                if n_if:
                    a, b = h_num[:n_if, :], h_truth[:n_if, :]
                    d_if = float(np.linalg.norm(b))
                    relF_if = (float(np.linalg.norm(a - b) / d_if)
                               if d_if > 0 else float("nan"))
                    gain_if = (float(np.linalg.norm(a) / d_if)
                               if d_if > 0 else float("nan"))
                else:
                    relF_if = gain_if = float("nan")
                print(f"             relF (all rows) = {relF:.4f}   "
                      f"relF (Q_iface) = {relF_if:.4f}   "
                      f"gain (Q_iface) = {gain_if:.4f}")
            except Exception as exc:
                relF = relF_if = gain_if = float("nan")
                print(f"             relF unavailable "
                      f"({type(exc).__name__}: {exc})")
            rows.append(dict(dso=dso_id, label=label,
                             k=kw.get("thevenin_k", np.nan),
                             relF=relF, relF_if=relF_if, gain_if=gain_if,
                             oltc_spread=spread))

    print("\n" + "=" * 78)
    print("SUMMARY  (Q_iface = the DSO's tracked output; the deciding rows)")
    print("=" * 78)
    hdr = (f"{'dso':>8} {'variant':>9} {'relF_all':>9} "
           f"{'relF_iface':>11} {'gain_iface':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dso']:>8} {r['label']:>9} {r['relF']:>9.4f} "
              f"{r['relF_if']:>11.4f} {r['gain_if']:>11.4f}")

    labels = sorted({r["label"] for r in rows}, key=lambda s: (s != "slack", s))
    print("\nmean over DSOs:")
    for lb in labels:
        sub = [r for r in rows if r["label"] == lb]
        print(f"  {lb:>9}: relF_all {np.nanmean([r['relF'] for r in sub]):.4f}   "
              f"relF_iface {np.nanmean([r['relF_if'] for r in sub]):.4f}   "
              f"gain_iface {np.nanmean([r['gain_if'] for r in sub]):.4f}")

    out = OUT_DIR / "ds_thevenin.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("dso,label,k,relF_all,relF_iface,gain_iface,oltc_spread\n")
        for r in rows:
            fh.write(f"{r['dso']},{r['label']},{r['k']},{r['relF']},"
                     f"{r['relF_if']},{r['gain_if']},{r['oltc_spread']}\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
