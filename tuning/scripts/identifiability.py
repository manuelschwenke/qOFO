"""
tuning/scripts/identifiability.py
=================================
Stage 6 — is the objective identifiable from the decision coordinates?

This re-runs the three diagnostics that condemned the original setup
(``docs/daily_log/07_2026/2026-07-31_bo_tuning_audit.md`` §1.1), so the new study
is judged by the same instrument rather than a friendlier one:

===================================================  ==================  ============
diagnostic                                           original result     success
===================================================  ==================  ============
random-forest out-of-fold R2, log-coords -> scalar   **0.09**            materially above
parameter spread across the 10 best trials           **1.1-3.8 decades** well under 1 decade
marginal Spearman \\|rho\\| vs the scalar               <= 0.27             larger, and signed
===================================================  ==================  ============

Why it gates the holdout
------------------------
The handover is explicit: if these fail, the parameterisation is still wrong and
no further budget should be spent.  The holdout can be evaluated only once, so
spending it on an unidentifiable optimum destroys the campaign's only
independent evidence.  Run this first.

Pass ``--study-name`` twice to compare two studies.  That is the strongest form
of the evidence: the same diagnostic on the degenerate and corrected objectives
shows the objective *design* was the obstacle, not the search space.

Usage::

    python -m tuning.scripts.identifiability \\
        --storage sqlite:///<path>.db --study-name v5_reparam_v2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

R2_BASELINE = 0.09          # the value that condemned the original setup
SPREAD_DECADES_MAX = 1.0    # "well under one decade"


def _feasible(trial) -> bool:
    c = trial.system_attrs.get("constraints")
    return c is not None and all(v <= 0 for v in c)


def _oof_r2(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Random-forest out-of-fold R2. ``nan`` when there is too little data."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score

    n = X.shape[0]
    if n < 10:
        return float("nan")
    k = min(5, n)
    pred = cross_val_predict(
        RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=1),
        X, y, cv=KFold(n_splits=k, shuffle=True, random_state=seed),
    )
    return float(r2_score(y, pred))


def _spread_decades(vals: np.ndarray) -> float:
    v = vals[np.isfinite(vals) & (vals > 0)]
    if v.size < 2:
        return float("nan")
    return float(np.log10(v.max()) - np.log10(v.min()))


def _analyse(study, name: str, top_n: int) -> dict:
    import optuna
    from scipy.stats import spearmanr

    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and math.isfinite(t.value)]
    if not done:
        print(f"\n=== {name}: no completed trials ===")
        return {"study": name, "n_complete": 0}

    coord_names = sorted(done[0].params.keys())
    X_all = np.array([[t.params[k] for k in coord_names] for t in done], float)
    y_all = np.array([t.value for t in done], float)
    feas_mask = np.array([_feasible(t) for t in done], bool)

    print(f"\n{'='*72}\n=== {name} ===\n{'='*72}")
    print(f"completed trials: {len(done)}   feasible: {int(feas_mask.sum())} "
          f"({feas_mask.mean():.0%})")
    print(f"coordinates: {coord_names}")
    print(f"objective range: {y_all.min():.5g} .. {y_all.max():.5g}   "
          f"distinct: {len(set(np.round(y_all, 6)))}/{len(y_all)}")

    out: dict = {
        "study": name,
        "n_complete": len(done),
        "n_feasible": int(feas_mask.sum()),
        "coords": coord_names,
        "objective_distinct": len(set(np.round(y_all, 6))),
        "objective_min": float(y_all.min()),
        "objective_max": float(y_all.max()),
    }

    # ── 1. RF out-of-fold R2 on log-coordinates ─────────────────────────────
    print(f"\n1. Random-forest out-of-fold R2  (log-coords -> objective)")
    print(f"   original setup scored {R2_BASELINE}; success is materially above")
    for label, mask in (("all completed", np.ones_like(feas_mask)),
                        ("feasible only", feas_mask)):
        if mask.sum() < 10:
            print(f"   {label:15s} n={int(mask.sum()):3d}  (too few for CV)")
            continue
        r2 = _oof_r2(np.log10(X_all[mask]), y_all[mask])
        verdict = ("PASS" if r2 > 2 * R2_BASELINE else
                   "marginal" if r2 > R2_BASELINE else "FAIL")
        print(f"   {label:15s} n={int(mask.sum()):3d}  R2 = {r2:+.3f}   {verdict}")
        out[f"r2_{label.split()[0]}"] = r2

    # ── 2. Spread of the top-N trials, per coordinate ───────────────────────
    print(f"\n2. Log-spread across the best {top_n} FEASIBLE trials, per coordinate")
    print(f"   original setup: 1.1-3.8 decades in every coordinate; "
          f"success is well under 1")
    idx = np.argsort(y_all)
    feas_idx = [i for i in idx if feas_mask[i]][:top_n]
    if len(feas_idx) < 2:
        print("   too few feasible trials to measure")
    else:
        spreads = {}
        for j, k in enumerate(coord_names):
            d = _spread_decades(X_all[feas_idx, j])
            spreads[k] = d
            flag = ("ok" if d < SPREAD_DECADES_MAX
                    else "WIDE -- coordinate not pinned")
            print(f"   {k:18s} {d:5.2f} decades   {flag}")
        out["top_spread_decades"] = spreads
        out["top_spread_worst"] = float(max(spreads.values()))

    # ── 3. Marginal Spearman correlation ────────────────────────────────────
    print(f"\n3. Marginal Spearman |rho| vs the objective (feasible trials)")
    print(f"   original setup: all |rho| <= 0.27, none significant")
    if feas_mask.sum() >= 8:
        rhos = {}
        for j, k in enumerate(coord_names):
            r, p = spearmanr(X_all[feas_mask, j], y_all[feas_mask])
            rhos[k] = {"rho": float(r), "p": float(p)}
            print(f"   {k:18s} rho = {r:+.3f}   p = {p:.4f}"
                  f"{'   *' if p < 0.05 else ''}")
        out["spearman"] = rhos
    else:
        print("   too few feasible trials")

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.identifiability")
    p.add_argument("--storage", type=str, required=True)
    p.add_argument("--study-name", action="append", required=True,
                   help="Repeatable: pass twice to compare two studies.")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "results" / "tuning" / "identifiability.json")
    args = p.parse_args(argv)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    reports = []
    for name in args.study_name:
        try:
            st = optuna.load_study(study_name=name, storage=args.storage)
        except Exception as exc:
            print(f"[ident] cannot load {name!r}: {exc}", file=sys.stderr)
            continue
        reports.append(_analyse(st, name, args.top_n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2, default=str))
    print(f"\n[ident] wrote {args.out}")

    # ── Verdict on the last study analysed ──────────────────────────────────
    if not reports:
        return 2
    last = reports[-1]
    r2 = last.get("r2_feasible", last.get("r2_all", float("nan")))
    spread = last.get("top_spread_worst", float("nan"))
    print(f"\n{'='*72}\nVERDICT for {last['study']!r}")
    ok_r2 = math.isfinite(r2) and r2 > 2 * R2_BASELINE
    ok_sp = math.isfinite(spread) and spread < SPREAD_DECADES_MAX
    print(f"  R2 = {r2:+.3f}  (> {2*R2_BASELINE:.2f} required)      "
          f"{'PASS' if ok_r2 else 'FAIL'}")
    print(f"  worst top-{args.top_n} spread = {spread:.2f} decades "
          f"(< {SPREAD_DECADES_MAX} required)   {'PASS' if ok_sp else 'FAIL'}")
    if ok_r2 and ok_sp:
        print("  => identifiable.  Spending the holdout is justified.")
        return 0
    print("  => NOT identifiable on at least one test.  The handover is explicit "
          "that no further\n     budget should be spent -- in particular do NOT "
          "evaluate the holdout, which\n     can only be read once.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
