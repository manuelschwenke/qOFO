"""
tuning_mc/stage_0_lambda_curve.py
=================================
The **analytic** half of the lambda calibration: what the design rule itself
says the loop gain will be, as a function of its own coordinate, before any
simulation is run.

Why this exists as a separate step
----------------------------------
``stage_1_search --phase scan --scan-knob lambda_tso`` measures the *realised*
contraction ``rho_emp_p95`` and fits it against the design target.  That fit has
an intercept -- the part of the contraction no continuous weight can scale away
-- and reading it off a 20-minute-per-point simulation sweep is both expensive
and ambiguous, because three different effects land in the same number:

1. the **integer columns**, which the curvature rule does not price at all
   (``precondition_g_w`` scales only the continuous classes);
2. the **inter-zone coupling** term, which the coordinator adds to the local
   ``lambda_max(M_ii)`` before testing ``0 < rho < 2``
   (:meth:`controller.multi_tso_coordinator.MultiTSOCoordinator.check_contraction`);
3. the **model gap** between the cached ``H`` the rule designs on and the live
   sensitivities the loop actually runs against.

Only (1) is analytic, and it is available for free: ``precondition_g_w``
already returns ``lambda_floor``, the ``lambda_max(M)`` contributed by the
*fixed* columns alone.  This module sweeps the design coordinate through Stage 0
and tabulates, per control area,

    lambda_target  ->  lambda_floor,  lambda_max(M) over the FULL column set

which is affine in the target by construction (the preconditioned columns
contribute proportionally, the fixed ones a constant).  Subtracting this
analytic relation from the measured one separates "the rule knew about it" from
"the plant added it", which is the difference the thesis reports rather than
absorbs.

It also supplies the one contraction statement available for the **subordinate**
layer.  ``rho_emp_p95`` is written per TSO zone by the coordinator and has no DSO
equivalent, so a measured DSO contraction does not exist; the analytic
``lambda_floor`` of each DSO loop does, and it is what bounds ``lambda_dso`` from
below.

Cost: one Stage-0 controller build per grid point, ~20 s.  Nothing is simulated.

Usage::

    python -m tuning_mc.stage_0_lambda_curve
    python -m tuning_mc.stage_0_lambda_curve --lam-values 0.1,0.2,0.4,0.9 \
        --out results/tuning_mc/campaign/lambda_curve.json

Author: Manuel Schwenke (with Claude Code), 2026-08-14
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_BASELINE = (_REPO_ROOT / "tuning" / "scripts" / "configs"
                    / "baseline_ieee39_thevenin.yaml")
DEFAULT_OUT = _REPO_ROOT / "results" / "tuning_mc" / "campaign_0814" / "lambda_curve.json"


def run_stage0(lam: float, *, baseline: Path, scenario: str, tau: float,
               workdir: Path) -> dict[str, Any]:
    """One Stage-0 design at ``lambda_tso = lambda_dso = lam``.

    Both targets are moved together on purpose: Stage 0 applies each to its own
    layer, so one invocation yields both curves and the layers cannot contaminate
    each other -- there is no simulation here for them to interact through.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"stage0_lam_{lam:g}.json"
    if not out.exists():
        cmd = [sys.executable, "-m", "tuning_mc.stage_0_preconditioning",
               "--baseline", str(baseline), "--scenario", scenario,
               "--lambda-tso", repr(float(lam)), "--lambda-dso", repr(float(lam)),
               "--tau", repr(float(tau)), "--out", str(out)]
        env = dict(os.environ)
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "PYTHONPATH": str(_REPO_ROOT)})
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"stage_0 failed at lambda={lam} "
                               f"(rc={proc.returncode}):\n{proc.stdout[-1500:]}\n"
                               f"{proc.stderr[-1500:]}")
    return json.loads(out.read_text(encoding="utf-8-sig"))


def _fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Least-squares ``y = floor + slope * x``; returns (floor, slope, resid)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([xa, np.ones_like(xa)]).T
    (slope, floor), *_ = np.linalg.lstsq(A, ya, rcond=None)
    return float(floor), float(slope), float(np.abs(ya - (slope * xa + floor)).max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.stage_0_lambda_curve")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--scenario", default="none",
                   help="'none' keeps the baseline's own operating point, which "
                        "is what the Stage-1 campaign designs at.")
    p.add_argument("--lam-values", default="0.10,0.15,0.20,0.25,0.40,0.60,0.90")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    lams = [float(v) for v in args.lam_values.split(",")]
    workdir = Path(args.out).parent / "lambda_curve_designs"
    print(f"[lam-curve] {len(lams)} Stage-0 designs at lambda = {lams}",
          flush=True)

    # loop label -> {"kind", "area", "lambda_floor", points: [(target, full)]}
    loops: dict[str, dict[str, Any]] = {}
    for lam in lams:
        t0 = time.perf_counter()
        data = run_stage0(lam, baseline=Path(args.baseline),
                          scenario=args.scenario, tau=args.tau, workdir=workdir)
        for rec in data.get("loops", []):
            meta = rec.get("continuous_meta") or {}
            if "lambda_full_after" not in meta:
                continue
            label = rec["label"]
            e = loops.setdefault(label, {"kind": rec.get("kind"),
                                         "area": rec.get("area"),
                                         "lambda_floor": [], "points": []})
            e["points"].append((float(meta["lambda_target"]),
                                float(meta["lambda_full_after"])))
            if "lambda_floor" in meta:
                e["lambda_floor"].append(float(meta["lambda_floor"]))
        print(f"[lam-curve] lambda={lam:<5g} done ({time.perf_counter() - t0:.0f} s)",
              flush=True)

    # The floor is reported DIRECTLY -- lambda_max(M) with the preconditioned
    # columns' weights sent to infinity -- and the affine fit is kept only as a
    # cross-check.  lambda_max of a sum of PSD terms is subadditive, not
    # additive, so "intercept of a straight line through the curve" is an
    # approximation of the floor and a poor one wherever the fixed columns
    # dominate: the DSO loops flatten completely below lambda ~ 0.3, where a
    # least-squares line through the whole range under-reads the floor by ~35 %.
    print(f"\n{'loop':<14}{'kind':<6}{'FLOOR':>9}{'fit int.':>10}{'slope':>9}"
          f"{'resid':>9}   lambda_full at target")
    summary: dict[str, Any] = {}
    for label, e in loops.items():
        pts = sorted(e["points"])
        floor, slope, resid = _fit([a for a, _ in pts], [b for _, b in pts])
        direct = (float(np.mean(e["lambda_floor"]))
                  if e["lambda_floor"] else float("nan"))
        vals = "  ".join(f"{a:g}->{b:.3f}" for a, b in pts)
        print(f"{label:<14}{str(e['kind']):<6}{direct:>9.4f}{floor:>10.4f}"
              f"{slope:>9.4f}{resid:>9.5f}   {vals}")
        summary[label] = {"kind": e["kind"], "area": e["area"],
                          "floor": direct, "floor_fit_intercept": floor,
                          "slope": slope, "max_residual": resid,
                          "points": [{"lambda_target": a, "lambda_full": b}
                                     for a, b in pts]}

    tso = {k: v for k, v in summary.items() if v["kind"] == "tso"}
    dso = {k: v for k, v in summary.items() if v["kind"] == "dso"}
    if tso:
        worst = max(tso.values(), key=lambda v: v["floor"])
        print(f"\n[lam-curve] TSO layer: worst analytic floor "
              f"{worst['floor']:.4f} (area {worst['area']}). This is the "
              f"cached-model, single-zone counterpart of the measured "
              f"rho_emp_p95 intercept; the coordinator adds the inter-zone "
              f"coupling sum on top of it, and the live H adds the rest.")
    if dso:
        worst = max(dso.values(), key=lambda v: v["floor"])
        print(f"[lam-curve] DSO layer: worst analytic floor "
              f"{worst['floor']:.4f} (area {worst['area']}). No measured "
              f"DSO contraction exists (zone_contraction_lhs is written per TSO "
              f"zone only), so this is the ONLY contraction statement available "
              f"for the subordinate layer.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"lam_values": lams, "scenario": args.scenario, "tau": args.tau,
         "baseline": str(args.baseline), "loops": summary}, indent=1),
        encoding="utf-8")
    print(f"[lam-curve] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
