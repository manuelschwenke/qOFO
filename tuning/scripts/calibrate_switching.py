"""tuning/scripts/calibrate_switching.py — set the OLTC weights from a budget.

Runbook step 2.  Calibrates ``g_w_tso_oltc`` (machine transformers) and
``g_w_dso_oltc`` (network coupling transformers) against an operational
tap-operations-per-day budget, by log-space bisection.

Why these two are not BO coordinates is argued in
:mod:`tuning.bisect_switching`: they price integer switching, the switch count
is provably monotone non-increasing in the weight, and the response has two
*exactly flat* tails that a density-ratio sampler cannot represent.  Across 1555
recorded runs they scored ``|rho| <= 0.27`` against the objective.

The output is a *specified* value — "``g_w_dso_oltc = X`` gives a median N tap
operations per day per transformer across the design envelope, within the ±20 %
band" — rather than a number an optimiser happened to land on.  Since EHV
maintenance budgets are a real operational constraint, that is the stronger
position for the thesis.

Usage::

    python -m tuning.scripts.calibrate_switching --target-tso 10 --target-dso 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
DEFAULT_BASELINE = _SCRIPT_DIR / "configs" / "baseline_ieee39.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.calibrate_switching")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--target-tso", type=float, default=10.0,
                   help="Tap operations per HOUR, worst machine transformer. "
                        "Not per day: the metric consumed is "
                        "tap_ops_per_h_tso (metrics.py normalises by "
                        "duration_s/3600). The design scenarios are "
                        "event-dense 75-min windows, so extrapolating to a "
                        "day inflates the figure ~19x.")
    p.add_argument("--target-dso", type=float, default=10.0,
                   help="Tap operations per HOUR, worst coupling transformer. "
                        "Not per day -- see --target-tso.")
    p.add_argument("--tol-rel", type=float, default=0.2)
    p.add_argument("--max-iter", type=int, default=8)
    p.add_argument("--n-scenarios", type=int, default=0,
                   help="Limit the scenario set (0 = all).")
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "results" / "tuning"
                   / "switching_calibration.json")
    args = p.parse_args(argv)

    from tuning._io import load_config_yaml
    from tuning.bisect_switching import calibrate_switching_price
    from tuning.parameters import FIXED_OVERRIDES
    from tuning.scenarios import tune_set_v2

    cfg = load_config_yaml(args.baseline)
    cfg = dataclasses.replace(cfg, **FIXED_OVERRIDES)
    scenarios = tune_set_v2()
    if args.n_scenarios:
        scenarios = scenarios[:args.n_scenarios]

    print(f"[calib] baseline={args.baseline}")
    print(f"[calib] {len(scenarios)} scenarios; "
          f"~{(2 + args.max_iter) * len(scenarios)} simulations per class")
    print(f"[calib] current: g_w_tso_oltc={cfg.g_w_tso_oltc:g} "
          f"g_w_dso_oltc={cfg.g_w_dso_oltc:g}")

    results = {}
    for field, target in (("g_w_tso_oltc", args.target_tso),
                          ("g_w_dso_oltc", args.target_dso)):
        print(f"\n[calib] --- {field}: target {target:g} ops/day/trafo ---")
        res = calibrate_switching_price(
            field, target, cfg, scenarios,
            tol_rel=args.tol_rel, max_iter=args.max_iter,
        )
        results[field] = res
        print(f"[calib] {field}: status={res.status} "
              f"g_w={res.g_w:.4g} achieved={res.achieved_ops_per_day:.3f} "
              f"ops/day  ({res.n_evaluations} evaluations)")
        if res.status == "plateau_high":
            print("        The budget is SLACK -- this actuator is already "
                  "quieter than the limit, so the weight is not what "
                  "constrains switching. Do not read g_w as 'tuned'.")
        elif res.status == "plateau_low":
            print("        The budget is UNREACHABLE within the bracket -- "
                  "the binding constraint is elsewhere (cooldowns, scenario "
                  "severity, or the loop gain), not this weight.")
        elif not res.within_tolerance:
            print("        Converged outside the tolerance band; the "
                  "per-trajectory response is rough. Inspect the ladder.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        field: {
            "g_w": res.g_w,
            "status": res.status,
            "target_ops_per_day": res.target_ops_per_day,
            "achieved_ops_per_day": res.achieved_ops_per_day,
            "within_tolerance": res.within_tolerance,
            "n_evaluations": res.n_evaluations,
            # Publish the ladder: it is the evidence that the response really
            # is monotone over the bracket, and a thesis figure on its own.
            "ladder": [dataclasses.asdict(p) for p in res.ladder],
        }
        for field, res in results.items()
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[calib] wrote {args.out}")

    ok = all(r.status == "bracketed" for r in results.values())
    print("[calib] both classes bracketed" if ok else
          "[calib] at least one class did NOT bracket -- read the notes above "
          "before using these values")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
