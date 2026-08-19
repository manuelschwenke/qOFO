"""
tuning_mc/report_0815.py
========================
Read the campaign's result JSONs and answer the three questions the 0814
campaign could not answer from its own output.

1. **Does any single window own the objective?**  On the 0814 bank
   ``mc_undervolt_ramp_winter`` contributed ~60 % of the aggregate ``f_ts``, so
   the "mean over five windows" was in effect one window -- and nothing in the
   output said so.  ``--windows`` prints each window's share.
2. **What does each capability stratum contribute?**  ``tau``, ``lambda_dso``
   and ``dso_g_v_ratio`` are *structurally* inert where DER reactive capability
   is zero, so an ``f_q`` averaged across strata mixes a signal with a constant.
   ``--strata`` splits it.
3. **How is wear distributed over the fleet?**  The constraint is on the worst
   transformer, but a fleet with one transformer at the budget and eleven near
   zero is a different picture from twelve at a third of it.  ``--wear`` prints
   per transformer, and converts to taps/day -- **only** for Tier-2 windows,
   where that conversion is legitimate.

Usage::

    python -m tuning_mc.report_0815 --dir results/tuning_mc/campaign_0815 \\
        --set tier1 --windows --strata
    python -m tuning_mc.report_0815 --result <one.json> --wear
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:                       # noqa: BLE001
            print(f"[report] skipping {p.name}: {exc}")
    return out


def _knob_txt(r: dict) -> str:
    k = r.get("knobs", {})
    return " ".join(f"{n}={k[n]:g}" for n in sorted(k))


def show_windows(r: dict) -> None:
    per = r.get("per_scenario", {})
    meta = r.get("window_meta", {})
    if not per:
        return
    tot_ts = sum(v["f_ts"] for v in per.values())
    tot_q = sum(v["f_q"] for v in per.values())
    print(f"\n  per-window contribution  (aggregate is the plain mean, "
          f"cvar_pct=100)")
    print(f"  {'window':<24}{'stratum':<9}{'f_ts':>10}{'share':>8}"
          f"{'f_q':>10}{'share':>8}{'taps/h':>9}{'rev/h':>8}")
    for name, v in sorted(per.items(), key=lambda kv: -kv[1]["f_ts"]):
        st = meta.get(name, {}).get("stratum", "?")
        taps = max(v["tap_ops_per_h_tso"], v["tap_ops_per_h_dso"])
        rev = max(v["tap_reversals_per_h_tso"], v["tap_reversals_per_h_dso"])
        print(f"  {name:<24}{st:<9}{v['f_ts']:>10.4f}"
              f"{100 * v['f_ts'] / tot_ts:>7.1f}%{v['f_q']:>10.4f}"
              f"{(100 * v['f_q'] / tot_q if tot_q else float('nan')):>7.1f}%"
              f"{taps:>9.3f}{rev:>8.3f}")
    top = max(per.values(), key=lambda v: v["f_ts"])
    share = 100 * top["f_ts"] / tot_ts
    n = len(per)
    even = 100.0 / n
    verdict = ("OK" if share < 2 * even else
               "CONCENTRATED -- the aggregate is being driven by one window")
    print(f"  worst window is {share:.1f} % of f_ts "
          f"({n} windows, even share {even:.1f} %)  -> {verdict}")


def show_strata(rows: list[dict]) -> None:
    print(f"\n  per-stratum means  (f_ts / f_q, n windows)")
    strata = sorted({s for r in rows for s in r.get("by_stratum", {})})
    if not strata:
        print("    (no by_stratum data -- result predates the field)")
        return
    hdr = "".join(f"{s:>22}" for s in strata)
    print(f"  {'candidate':<34}{hdr}")
    for r in rows:
        cells = ""
        for s in strata:
            b = r.get("by_stratum", {}).get(s)
            cells += (f"{b['f_ts']:>10.4f}/{b['f_q']:<11.4f}" if b
                      else f"{'-':>22}")
        print(f"  {_knob_txt(r)[:33]:<34}{cells}")


def show_wear(r: dict, *, budget_per_day: float = 30.0) -> None:
    wear = r.get("per_transformer_wear", {})
    if not wear:
        print("    (no per_transformer_wear -- result predates the field)")
        return
    is_audit = r.get("scenario_set") == "audit"
    print(f"\n  per-transformer wear   scenario_set={r.get('scenario_set')}")
    if is_audit:
        print(f"  12-h profile windows: ops/day is a legitimate conversion "
              f"here (budget {budget_per_day:g}/day).")
    else:
        print(f"  *** NOT a Tier-2 window: these are event-dense 90-min rates. "
              f"Do NOT convert to taps/day. ***")
    names = sorted({t for sc in wear.values() for t in sc})
    print(f"  {'transformer':<18}" + "".join(f"{sc[:14]:>16}" for sc in wear)
          + f"{'worst ops/h':>13}" + (f"{'-> ops/day':>12}" if is_audit else ""))
    for t in names:
        cells, worst = "", 0.0
        for sc, per_t in wear.items():
            v = per_t.get(t, {}).get("ops_per_h", float("nan"))
            rv = per_t.get(t, {}).get("reversals_per_h", float("nan"))
            cells += f"{v:>9.3f}/{rv:<6.3f}"
            if math.isfinite(v):
                worst = max(worst, v)
        day = f"{24 * worst:>12.1f}" if is_audit else ""
        flag = ""
        if is_audit and 24 * worst > budget_per_day:
            flag = "  OVER"
        print(f"  {t:<18}{cells}{worst:>13.3f}{day}{flag}")
    print(f"  (cells are ops/h / reversals/h)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.report_0815")
    p.add_argument("--dir", type=Path,
                   default=_REPO_ROOT / "results" / "tuning_mc" / "campaign_0815")
    p.add_argument("--set", dest="sset", default="tier1",
                   help="scenario-set prefix of the eval files to read")
    p.add_argument("--result", type=Path, default=None,
                   help="one specific result JSON instead of a directory")
    p.add_argument("--windows", action="store_true")
    p.add_argument("--strata", action="store_true")
    p.add_argument("--wear", action="store_true")
    p.add_argument("--sort", default="f_ts", choices=("f_ts", "f_q"))
    args = p.parse_args(argv)

    if args.result:
        rows = load([args.result])
    else:
        rows = load(sorted((args.dir / "evals").glob(f"{args.sset}_*.json")))
    if not rows:
        raise SystemExit(f"[report] no results found")
    rows.sort(key=lambda r: r.get(args.sort, float("inf")))

    print(f"[report] {len(rows)} candidate(s), scenario_set="
          f"{rows[0].get('scenario_set')}, limits={rows[0].get('limits_source')}")
    print(f"\n  {'candidate':<52}{'f_ts':>10}{'f_q':>10}{'rho':>9}"
          f"{'taps/h':>9}{'rev/h':>8}  feas")
    for r in rows:
        print(f"  {_knob_txt(r)[:51]:<52}{r['f_ts']:>10.5f}{r['f_q']:>10.5f}"
              f"{r.get('worst_rho_emp_p95', float('nan')):>9.4f}"
              f"{r.get('worst_tap_ops_per_h', float('nan')):>9.3f}"
              f"{r.get('worst_reversals_per_h', float('nan')):>8.3f}"
              f"  {r['feasible']}")

    if args.strata:
        show_strata(rows)
    if args.windows:
        for r in rows[: 1 if len(rows) > 3 else len(rows)]:
            print(f"\n=== {_knob_txt(r)} ===")
            show_windows(r)
    if args.wear:
        for r in rows:
            print(f"\n=== {_knob_txt(r)} ===")
            show_wear(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
