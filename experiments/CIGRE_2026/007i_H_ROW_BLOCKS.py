#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007i_H_ROW_BLOCKS.py
===========================================
Why does the Thevenin boundary control WORSE despite a 12x better H?

``007d`` measured H fidelity on the corridor-VOLTAGE rows only, and ``007``
otherwise reported a whole-matrix Frobenius norm that is dominated by the
Q_gen block (Mvar/pu units, ~1e3 larger than the voltage rows).  The Q_PCC
rows -- the TSO's model of its own TS-DS interface flows -- were never looked
at.  ``007h`` then found, closed loop, that Thevenin incurs ~40 % more
interface tracking error than PQ at matched setpoint drive, which is a
boundary-specific penalty the corridor-row metric could not have seen.

Candidate mechanism: the boundary condition also enters the Q_PCC output rows
through the network, and Thevenin may degrade those while improving the
corridor rows.  If so the closed-loop result follows directly -- the TSO would
be dispatching interface setpoints from a worse interface model.

This splits the H comparison by row block and settles it:

    rows = [ V_bus | Q_PCC | I_line | Q_gen ]

reporting, per block, the relative Frobenius error against the numerical truth
on the full interconnected plant, and the gain ratio ||H_var|| / ||H_truth||
(>1 = the model over-states that block's response).

Only zones with subordinate DSOs have Q_PCC rows; zone 1 parents none, so its
Q_PCC block is empty and reported as n/a.

Caveat kept in view: ``compute_numerical_h_tso`` forces the Q_PCC diagonal to
1.0 for the identity convention on the Q_PCC_set columns.  That constant is
common to truth and to every variant, so it damps the apparent relative
difference on this block -- differences reported here are if anything an
UNDER-estimate.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sensitivity.network_reduction import build_tso_local_net  # noqa: E402
from sensitivity.numerical_h import compute_numerical_h_tso  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).with_name(fname)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Reuse the validated setup capture and argument reconstruction from 007d
# rather than re-deriving them (007d proves the reconstruction reproduces the
# runner's own reduced net element for element before using it).
_D = _load("_cigre007d", "007d_THEVENIN_SWEEP.py")

OUT_DIR = _ROOT / "results" / "007_tie_boundary"
K_TH = 1.5


def _blocks(ctrl, zd) -> List[Tuple[str, np.ndarray]]:
    """Row-index sets of the H layout ``[V | Q_PCC | I | Q_gen]``."""
    cfg = ctrl.config
    n_v = len(cfg.voltage_bus_indices)
    n_pcc = len(cfg.pcc_trafo_indices)
    n_i = len(cfg.current_line_indices)
    n_gen = len(cfg.gen_indices)

    vb = list(cfg.voltage_bus_indices)
    corridor = sorted({vb.index(int(b))
                       for b in (zd.tie_line_endpoint_buses or [])
                       if int(b) in vb})

    out: List[Tuple[str, np.ndarray]] = [
        ("V_all", np.arange(0, n_v)),
        ("V_corridor", np.asarray(corridor, dtype=int)),
        ("Q_PCC", np.arange(n_v, n_v + n_pcc)),
        ("I_line", np.arange(n_v + n_pcc, n_v + n_pcc + n_i)),
        ("Q_gen", np.arange(n_v + n_pcc + n_i, n_v + n_pcc + n_i + n_gen)),
    ]
    return [(nm, ix) for nm, ix in out if ix.size]


def _block_stats(h_var, h_truth, rows) -> Tuple[float, float]:
    a, b = h_var[rows, :], h_truth[rows, :]
    d = float(np.linalg.norm(b))
    if d <= 0:
        return float("nan"), float("nan")
    return (float(np.linalg.norm(a - b) / d),
            float(np.linalg.norm(a) / d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=K_TH)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("H FIDELITY BY ROW BLOCK -- does Thevenin degrade the Q_PCC "
          "(interface) rows?")
    print("=" * 88)

    st = _D._capture()
    plant = st["net"]
    zones = sorted(st["tso_controllers"].keys())

    variants = [("pq", dict(tie_boundary="pq")),
                ("th", dict(tie_boundary="thevenin",
                            tie_thevenin_k=float(args.k)))]

    rows_out: List[Dict[str, Any]] = []
    for z in zones:
        ctrl = st["tso_controllers"][z]
        zd = st["zone_defs"][z]
        blocks = _blocks(ctrl, zd)
        base = _D._zone_args(st, z)

        print("\n" + "-" * 88)
        print(f"ZONE {z}   blocks: "
              + ", ".join(f"{nm}({ix.size})" for nm, ix in blocks))
        print("-" * 88)
        print("  truth H on the full plant ...", flush=True)
        h_truth = compute_numerical_h_tso(_D._freeze(plant), ctrl,
                                          closed_loop=False)

        hs: Dict[str, np.ndarray] = {}
        for label, kw in variants:
            red = build_tso_local_net(**base, **kw).net
            hs[label] = compute_numerical_h_tso(_D._freeze(red), ctrl,
                                                closed_loop=False)

        hdr = (f"{'block':>12} {'n':>4} " +
               " ".join(f"{lb + ' relF':>10} {lb + ' gain':>10}"
                        for lb, _ in variants) +
               f" {'winner':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for nm, ix in blocks:
            cells, errs = [], {}
            for lb, _ in variants:
                e, g = _block_stats(hs[lb], h_truth, ix)
                errs[lb] = e
                cells.append(f"{e:>10.4f} {g:>10.4f}")
            win = min(errs, key=lambda k: (np.inf if not np.isfinite(errs[k])
                                           else errs[k]))
            print(f"{nm:>12} {ix.size:>4} " + " ".join(cells) + f" {win:>8}")
            rows_out.append(dict(zone=z, block=nm, n=int(ix.size),
                                 **{f"{lb}_relF": errs[lb] for lb, _ in variants},
                                 winner=win))

    # ── Verdict on the candidate mechanism ────────────────────────────────
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    for nm in ("V_corridor", "Q_PCC"):
        sub = [r for r in rows_out if r["block"] == nm]
        if not sub:
            continue
        pq = np.array([r["pq_relF"] for r in sub], dtype=float)
        th = np.array([r["th_relF"] for r in sub], dtype=float)
        m = np.isfinite(pq) & np.isfinite(th)
        if not m.any():
            continue
        print(f"  {nm:>12}: mean relF  pq {pq[m].mean():.4f}   "
              f"th {th[m].mean():.4f}   "
              f"-> {'th better' if th[m].mean() < pq[m].mean() else 'PQ better'}"
              f"   (zones {[r['zone'] for r, k in zip(sub, m) if k]})")
    print("\n  If Thevenin wins V_corridor but LOSES Q_PCC, the closed-loop")
    print("  interface penalty of 007h is explained: the corridor rows improve")
    print("  while the rows the TSO dispatches its interface setpoints from")
    print("  get worse.")

    out = OUT_DIR / "h_row_blocks.csv"
    keys = ["zone", "block", "n", "pq_relF", "th_relF", "winner"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows_out:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
