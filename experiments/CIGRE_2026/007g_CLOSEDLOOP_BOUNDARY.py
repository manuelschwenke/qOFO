#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007g_CLOSEDLOOP_BOUNDARY.py
==================================================
Does the better boundary model actually control better?

007-007f established, open loop, that condensing a neighbouring TSO area as a
constant PQ load over-states an area's corridor authority by a mean factor of
2.8, and that a Thevenin equivalent at k ~ 1.5 -- which is the impedance the
neighbour physically presents -- cuts the corridor-row H error 12-fold.  None
of that is a control result.  This closes the loop: 2 h of QSS on the tuned V4
cascade, everything identical except the boundary the TSO controllers
linearised on.

The tuning confound, and how it is handled
------------------------------------------
``G_w`` was BO-tuned with the PQ model in place.  If PQ inflates the corridor
rows of H by ~2.8x, that tuning has already absorbed the inflation, and simply
swapping in a truer H with the SAME ``G_w`` makes the corridor loop ~2.8x
slower than intended.  A naive two-arm comparison would therefore measure
tuning mismatch, not model quality, and could easily show the better model
performing worse.

Three arms instead:

* ``pq``        -- state of the art: PQ boundary, nominal G_w.
* ``th``        -- Thevenin k=1.5, nominal G_w.  The honest "what happens if
                   you just switch the flag" result, gain mismatch included.
* ``th_gw``     -- Thevenin k=1.5 with the TSO G_w block scaled by KAPPA so the
                   loop gain is restored.  Isolates model quality from loop
                   gain.  Only the TSO weights are scaled: the DSO model is
                   unchanged by this experiment, so its tuning must not move.

``th`` and ``th_gw`` bracket the effect: if BOTH beat ``pq`` the conclusion is
robust to the confound; if only ``th_gw`` does, the gain matters more than the
model and that is worth knowing too.

Disturbance
-----------
The inherited contingency schedule trips a generator at minute 60.  Over a
2 h horizon it never restores, so the second hour runs post-contingency on a
model frozen at t = 0 -- which is exactly the regime where a truer boundary
should pay off, and where a merely-luckier one should not.  Identical across
arms.

Metric
------
``rms_v_ts_pu`` from ``cigre_summary_table``: the unweighted across-zone RMS of
(V - v_set) on TS buses, time-averaged.  Deliberately not the controllers' own
objective, so it does not flatter whichever arm happens to share its weighting.
Switching counts and actuator movement are reported alongside, because an arm
that merely moves less will look calmer without controlling better.

Usage
-----
    python experiments/CIGRE_2026/007g_CLOSEDLOOP_BOUNDARY.py
    python experiments/CIGRE_2026/007g_CLOSEDLOOP_BOUNDARY.py --hours 2 --arms pq,th
    python experiments/CIGRE_2026/007g_CLOSEDLOOP_BOUNDARY.py --replot

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.helpers.comparison_metrics import (  # noqa: E402
    cigre_summary_table, gen_s_rated_by_zone, voltage_rms_err_all,
    voltage_rms_err_per_zone,
)
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

OUT_DIR = _ROOT / "results" / "007_tie_boundary" / "closedloop"

#: Thevenin stiffness: the population-mean physical value measured in 007f
#: (k_phys mean 1.48 over ten corridors) and the best single default from the
#: 007d H-error sweep.  The two agree, which is why one number is used here.
K_TH = 1.5

#: G_w scale for the gain-restored arm.  PQ over-states the mean corridor gain
#: by 2.84 (007), so the truer H shrinks the corridor contribution to the MIQP
#: gradient by about that factor; dividing the TSO G_w block by it restores a
#: comparable step magnitude.  Crude by construction -- the gradient is not
#: purely corridor rows -- which is why this arm BRACKETS rather than replaces
#: the nominal-G_w arm.
KAPPA = 1.0 / 2.84

#: TSO-side step weights only.  The DSO's model is untouched by the boundary
#: change, so scaling its weights would confound the comparison.
_TSO_GW_KEYS = ("g_w_der", "g_w_gen", "g_w_pcc", "g_w_tso_oltc")

#: PER-ZONE gain restoration.  The over-statement measured open loop is not
#: uniform -- 1.41 / 4.47 / 2.62 in zones 1 / 2 / 3 -- so the single global
#: KAPPA above OVER-compensates zone 1 and UNDER-compensates zone 2.  That is a
#: confound in the ``th_gw`` arm itself: any per-zone pattern it produces could
#: be the uneven compensation rather than the boundary model.  This arm scales
#: each zone's own ``g_w`` vector by the inverse of ITS measured over-statement,
#: which removes the gain effect zone by zone and leaves the model effect alone.
KAPPA_PER_ZONE: Dict[int, float] = {1: 1.0 / 1.4138, 2: 1.0 / 4.4745,
                                    3: 1.0 / 2.6194}

ARMS: Dict[str, Dict[str, Any]] = {
    "pq":     dict(tie_boundary_equivalent="pq"),
    "th":     dict(tie_boundary_equivalent="thevenin", tie_thevenin_k=K_TH),
    "th_gw":  dict(tie_boundary_equivalent="thevenin", tie_thevenin_k=K_TH,
                   _scale_tso_gw=KAPPA),
    "th_gwz": dict(tie_boundary_equivalent="thevenin", tie_thevenin_k=K_TH,
                   _scale_gw_per_zone=True),
    # Control for th_gwz: the SAME per-zone gain change on the OLD boundary.
    # If this reproduces th_gwz, the effect is retuning; if th_gwz beats it,
    # the boundary model is contributing something the tuning cannot.
    "pq_gwz": dict(tie_boundary_equivalent="pq", _scale_gw_per_zone=True),
}


def _build_cfg(arm: str, hours: float):
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.n_total_s = 3600.0 * hours
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    for k, v in ARMS[arm].items():
        if k == "_scale_tso_gw":
            for gk in _TSO_GW_KEYS:
                setattr(cfg, gk, getattr(cfg, gk) * float(v))
            continue
        if k == "_scale_gw_per_zone":
            continue  # applied in the pre-loop hook, which sees the zones
        setattr(cfg, k, v)
    return cfg


def _per_zone_gw_hook(state: Dict[str, Any]) -> bool:
    """Scale each TSO controller's own ``g_w`` vector, then let the loop run.

    ``params.g_w`` is the per-actuator step-weight vector the MIQP divides by,
    so scaling it is the direct per-zone analogue of scaling the config
    weights.  Returning falsy continues into the main loop.
    """
    for z, ctrl in state.get("tso_controllers", {}).items():
        kap = KAPPA_PER_ZONE.get(int(z))
        if kap is None:
            continue
        gw = getattr(getattr(ctrl, "params", None), "g_w", None)
        if gw is None:
            print(f"  [gwz] zone {z}: params.g_w absent -- NOT scaled")
            continue
        # OFOParameters is a frozen dataclass, so rebuild it rather than
        # assigning into it.
        scaled = (np.asarray(gw, dtype=float) * float(kap)
                  if np.ndim(gw) else float(gw) * float(kap))
        ctrl.params = dataclasses.replace(ctrl.params, g_w=scaled)
        print(f"  [gwz] zone {z}: g_w scaled by {kap:.4f} "
              f"({np.size(scaled)} entries)")
    return False


def _run(arm: str, hours: float) -> List[Any]:
    cfg = _build_cfg(arm, hours)
    gw = {k: getattr(cfg, k) for k in _TSO_GW_KEYS}
    per_zone = bool(ARMS[arm].get("_scale_gw_per_zone"))
    print(f"\n[{arm}] boundary={cfg.tie_boundary_equivalent} "
          f"k={getattr(cfg, 'tie_thevenin_k', '-')}  "
          f"n_total_s={cfg.n_total_s:.0f}  TSO g_w={gw}"
          f"{'  + per-zone g_w scaling' if per_zone else ''}")
    t0 = time.time()
    recs = run_multi_tso_dso(
        cfg, pre_loop_hook=_per_zone_gw_hook if per_zone else None)
    print(f"[{arm}] {len(recs)} records in {time.time() - t0:.0f} s")
    return recs


def _split_metric(recs, v_set: float, hours: float) -> Dict[str, float]:
    """RMS TS voltage error over the whole run and either side of minute 60.

    The generator trip at minute 60 is the only disturbance; splitting there
    separates steady tracking from post-contingency behaviour on a frozen
    model, which is where the boundary choice should matter most.
    """
    va = voltage_rms_err_all(recs, v_set)
    err = np.asarray(va["rms_err_pu"], dtype=float)
    t = np.asarray(va.get("t_min", np.arange(len(err))), dtype=float)
    pre, post = t < 60.0, t >= 60.0
    f = lambda m: (float(np.nanmean(err[m]))
                   if m.any() and np.any(np.isfinite(err[m])) else np.nan)
    return {"all": f(np.ones_like(pre, dtype=bool)), "pre": f(pre),
            "post": f(post)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--arms", default="pq,th,th_gw")
    ap.add_argument("--replot", action="store_true",
                    help="reuse pickled logs, recompute metrics only")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"CLOSED-LOOP BOUNDARY COMPARISON -- {args.hours:g} h QSS, V4 cascade")
    print("=" * 78)

    logs: Dict[str, List[Any]] = {}
    for arm in arms:
        pkl = OUT_DIR / f"{arm}_{args.hours:g}h.pkl"
        if args.replot and pkl.exists():
            with open(pkl, "rb") as fh:
                logs[arm] = pickle.load(fh)
            print(f"[{arm}] loaded {len(logs[arm])} records from disk")
            continue
        try:
            recs = _run(arm, args.hours)
        except Exception as exc:
            print(f"[{arm}] RUN FAILED ({type(exc).__name__}: {exc})")
            continue
        if not recs:
            print(f"[{arm}] produced no records")
            continue
        logs[arm] = recs
        with open(pkl, "wb") as fh:
            pickle.dump(recs, fh)

    if not logs:
        print("\nno arm produced records -- nothing to compare")
        return 1

    v_set = 1.03
    try:
        rated = gen_s_rated_by_zone("base_410")
    except Exception:
        rated = None

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    try:
        df = cigre_summary_table(logs, v_set=v_set, gen_s_rated_mva=rated)
        print(df.to_string(index=False))
        df.to_csv(OUT_DIR / f"summary_{args.hours:g}h.csv", index=False)
    except Exception as exc:
        print(f"summary table unavailable ({type(exc).__name__}: {exc})")

    print("\nTS voltage RMS error [p.u.], split at the minute-60 trip:")
    hdr = f"{'arm':>7} {'all':>10} {'pre-trip':>10} {'post-trip':>10} {'vs pq':>9}"
    print(hdr)
    print("-" * len(hdr))
    base = None
    for arm in arms:
        if arm not in logs:
            continue
        m = _split_metric(logs[arm], v_set, args.hours)
        if base is None:
            base = m["all"]
        rel = ((m["all"] - base) / base * 100.0
               if base and np.isfinite(base) and base > 0 else np.nan)
        print(f"{arm:>7} {m['all']:>10.5f} {m['pre']:>10.5f} "
              f"{m['post']:>10.5f} {rel:>8.1f}%")

    # ── Per-zone breakdown: the mechanism test ────────────────────────────
    # Open loop, the PQ boundary over-stated the corridor gain by 1.41 / 4.47 /
    # 2.62 in zones 1 / 2 / 3 -- the error scaling inversely with in-zone
    # machine count.  If the closed-loop benefit is real and not incidental it
    # should follow the same ordering: largest in zone 2, smallest in zone 1.
    # An aggregate that improves while the per-zone pattern does NOT follow
    # would mean the gain came from somewhere other than the boundary.
    print("\nPer-zone TS voltage RMS error [p.u.]  "
          "(open-loop PQ gain over-statement: z1 1.41, z2 4.47, z3 2.62)")
    per_arm: Dict[str, Dict[int, float]] = {}
    zones_seen: List[int] = []
    for arm in arms:
        if arm not in logs:
            continue
        pz = voltage_rms_err_per_zone(logs[arm], v_set)
        zs = list(pz.get("zones", []))
        zones_seen = zs or zones_seen
        per_arm[arm] = {}
        for z in zs:
            e = np.asarray(pz["rms_err_pu"][z], dtype=float)
            per_arm[arm][z] = (float(np.nanmean(e))
                               if np.any(np.isfinite(e)) else np.nan)
    if per_arm and zones_seen:
        hdr = f"{'arm':>7}" + "".join(f"{'zone ' + str(z):>12}" for z in zones_seen)
        print(hdr)
        print("-" * len(hdr))
        for arm in arms:
            if arm not in per_arm:
                continue
            print(f"{arm:>7}" + "".join(
                f"{per_arm[arm].get(z, np.nan):>12.5f}" for z in zones_seen))
        ref = arms[0] if arms[0] in per_arm else next(iter(per_arm))
        for arm in arms:
            if arm not in per_arm or arm == ref:
                continue
            deltas = []
            for z in zones_seen:
                a, b = per_arm[arm].get(z, np.nan), per_arm[ref].get(z, np.nan)
                deltas.append((a - b) / b * 100.0
                              if np.isfinite(a) and np.isfinite(b) and b > 0
                              else np.nan)
            print(f"{arm + ' vs ' + ref:>7}" + "".join(
                f"{d:>11.1f}%" for d in deltas))

    print(f"\nlogs + csv in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
