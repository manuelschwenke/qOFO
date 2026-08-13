"""tuning/scripts/calibrate_metrics.py — Phase-A calibration and dead-term audit.

Runbook step 3.  Draws a Sobol sample from the *prior* over the reparameterised
space, runs it, and reports three things:

1. **Normalisation scales** for :class:`tuning.metrics.MetricScales`, taken as
   medians over the sample.  Drawing from the prior rather than from optimiser
   output is what breaks the circularity: the reference sample depends on the
   search space, not on the objective it is calibrating.  The study history shows
   what the circular version costs — six successive cost-function revisions, each
   chasing the previous run's output, none of them comparable to the others.

2. **Constraint limits** via :meth:`ConstraintLimits.from_reference`, anchored on
   the reference (hand-tuned) point.  This exists because the limits were first
   invented as round numbers and the reference — the one operating point known to
   control well — failed three of six.

3. **A dead-term audit.**  For every performance term and every constraint:
   the fraction of runs in which it is exactly zero, and its coefficient of
   variation.  **A cost term that is zero in > 90 % of runs contributes nothing
   and must be removed or rescaled before spending a tuning budget.**  Two terms
   in the legacy cost failed exactly this way and nothing caught it: the
   oscillation term was zero in *100 %* of 1555 scenario-runs, and the tap term
   contributed 0.2 % of the mean cost.

Usage::

    python -m tuning.scripts.calibrate_metrics --n-draws 40 --n-scenarios 3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
DEFAULT_BASELINE = _SCRIPT_DIR / "configs" / "baseline_ieee39.yaml"

#: Terms of the performance scalar, as returned by ``performance_scalar``.
_PERF_TERMS = ("v_rms_ts", "v_rms_ds", "v_worst_ts", "v_band_ts",
               "q_pcc", "pcc_underutil")

#: Raw metric attributes whose medians become the normalisation scales.
_SCALE_SOURCES = {
    "v_rms_ts":       "v_rms_ts",
    "v_rms_ds":       "v_rms_ds",
    "v_worst_ts":     "v_worst_ts",
    "v_worst_ds":     "v_worst_ds",
    "v_band_excess":  "v_band_excess_ts",
    "q_pcc":          "itae_q_pcc",
    "q_tie":          "itae_q_tie",
    "pcc_underutil":  "itae_pcc_underutil",
}


def _sobol_draws(n: int, seed: int) -> list[dict[str, float]]:
    """Sobol sample over the reparameterised space.

    Sobol rather than uniform random: at n = 40 in 4 dimensions a uniform draw
    leaves large holes, and the point of this sample is *coverage* of the prior,
    not randomness.  Sobol balance is strictly best at powers of two, so 32 or 64
    are marginally preferable to 40 if exact balance matters.
    """
    from scipy.stats import qmc

    from tuning.reparam import BO_DIMS_V2

    dims = list(BO_DIMS_V2)
    sampler = qmc.Sobol(d=len(dims), scramble=True, seed=seed)
    unit = sampler.random(n)
    out: list[dict[str, float]] = []
    for row in unit:
        coords: dict[str, float] = {}
        for u, p in zip(row, dims):
            lo, hi = float(p.low), float(p.high)
            if p.log:
                coords[p.name] = float(
                    10.0 ** (math.log10(lo) + u * (math.log10(hi)
                                                   - math.log10(lo))))
            else:
                coords[p.name] = float(lo + u * (hi - lo))
        out.append(coords)
    return out


def _summarise(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "zero_frac": 1.0, "median": 0.0, "cov": 0.0}
    mean = float(arr.mean())
    return {
        "n": int(arr.size),
        "zero_frac": float(np.mean(arr == 0.0)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "cov": float(arr.std() / mean) if mean > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.calibrate_metrics")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--n-draws", type=int, default=40)
    p.add_argument("--n-scenarios", type=int, default=3,
                   help="Scenarios per draw. 3 of 5 keeps the cost near 6 h; "
                        "the scales only need to make terms O(1).")
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "results" / "tuning"
                   / "metric_calibration.json")
    args = p.parse_args(argv)

    from tuning._io import load_config_yaml
    from tuning.metrics import MetricScales
    from tuning.objectives_v2 import (
        CONSTRAINT_NAMES,
        ConstraintLimits,
        _run_scenario,
        feasibility_constraints,
        performance_scalar,
    )
    from tuning.parameters import FIXED_OVERRIDES
    from tuning.reparam import Gauge, apply_reparam_to_config, coords_from_config
    from tuning.scenarios import tune_set_v2

    cfg0 = load_config_yaml(args.baseline)
    cfg0 = dataclasses.replace(cfg0, **FIXED_OVERRIDES)
    gauge = Gauge.from_config(cfg0)
    scenarios = tune_set_v2()[:args.n_scenarios]
    scales = MetricScales()

    draws = [coords_from_config(cfg0, gauge)] + _sobol_draws(
        args.n_draws, args.seed)
    est_h = len(draws) * len(scenarios) * 175.0 / 3600.0
    print(f"[calib] {len(draws)} draws (draw 0 = reference) x "
          f"{len(scenarios)} scenarios ~ {est_h:.1f} h")

    per_draw: list[dict] = []
    reference_metrics = None
    for i, coords in enumerate(draws):
        cfg = apply_reparam_to_config(cfg0, coords, gauge,
                                      fixed_overrides=FIXED_OVERRIDES)
        results, mets = [], []
        for sc in scenarios:
            res, _records = _run_scenario(sc, cfg, scales)
            results.append(res)
            mets.append(res.metrics)
        if i == 0:
            reference_metrics = mets

        terms: dict[str, list[float]] = {k: [] for k in _PERF_TERMS}
        for m in mets:
            _total, parts = performance_scalar(m, scales=scales)
            for k, v in parts.items():
                terms[k].append(v)
        g = feasibility_constraints(results, cfg)

        per_draw.append({
            "coords": coords,
            "n_feasible": sum(1 for m in mets if m.feasible),
            "raw": {name: [float(getattr(m, attr)) for m in mets]
                    for name, attr in _SCALE_SOURCES.items()},
            "terms": terms,
            "constraints": dict(zip(CONSTRAINT_NAMES,
                                    [float(v) for v in g])),
        })
        feas = per_draw[-1]["n_feasible"]
        print(f"  draw {i:3d}: {feas}/{len(scenarios)} feasible", flush=True)

    # ── Scales: medians over FEASIBLE runs only ─────────────────────────────
    print(f"\n{'scale':16s}{'median':>12s}{'p90':>12s}{'current':>12s}"
          f"{'ratio':>9s}")
    print("-" * 61)
    suggested: dict[str, float] = {}
    for name in _SCALE_SOURCES:
        pool = [v for d in per_draw if d["n_feasible"] for v in d["raw"][name]]
        s = _summarise(pool)
        cur = float(getattr(scales, name, float("nan")))
        suggested[name] = s["median"]
        ratio = s["median"] / cur if cur > 0 else float("nan")
        print(f"{name:16s}{s['median']:12.5g}{s.get('p90', 0):12.5g}"
              f"{cur:12.5g}{ratio:9.2f}")

    # ── Dead-term audit ────────────────────────────────────────────────────
    print(f"\nDEAD-TERM AUDIT  (a term zero in >90 % of runs contributes "
          f"nothing)")
    print(f"{'term':18s}{'zero_frac':>11s}{'CoV':>9s}{'median':>12s}  verdict")
    print("-" * 62)
    dead: list[str] = []
    for k in _PERF_TERMS:
        pool = [v for d in per_draw for v in d["terms"][k]]
        s = _summarise(pool)
        bad = s["zero_frac"] > 0.9
        if bad:
            dead.append(k)
        print(f"{k:18s}{s['zero_frac']:11.2f}{s['cov']:9.2f}"
              f"{s['median']:12.4g}  {'DEAD' if bad else 'ok'}")

    print(f"\n{'constraint':20s}{'viol_frac':>11s}{'median':>12s}  verdict")
    print("-" * 55)
    for name in CONSTRAINT_NAMES:
        pool = [d["constraints"][name] for d in per_draw]
        arr = np.asarray([v for v in pool if math.isfinite(v)])
        viol = float(np.mean(arr > 0)) if arr.size else float("nan")
        note = ("never binds — no information"
                if viol == 0.0 else
                "always binds — box is empty" if viol == 1.0 else "ok")
        print(f"{name:20s}{viol:11.2f}{float(np.median(arr)):12.4g}  {note}")

    limits = (ConstraintLimits.from_reference(reference_metrics)
              if reference_metrics else ConstraintLimits())
    print(f"\nSuggested ConstraintLimits (from the reference, margin 1.5):")
    for f in dataclasses.fields(limits):
        print(f"   {f.name:22s} {getattr(limits, f.name):.5g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_draws": len(draws),
        "scenarios": [s.name for s in scenarios],
        "suggested_scales": suggested,
        "suggested_limits": dataclasses.asdict(limits),
        "dead_terms": dead,
        "per_draw": per_draw,
    }, indent=2), encoding="utf-8")
    print(f"\n[calib] wrote {args.out}")

    if dead:
        print(f"[calib] REMOVE OR RESCALE before tuning: {dead}")
        return 1
    print("[calib] no dead terms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
