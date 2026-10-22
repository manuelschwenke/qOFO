#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_9_parameter_selection/ch_9_3_select.py
=====================================================
Apply the selection rule to a Stage-II bank and report the result.

The rule, in full
-----------------
1. **Filter.** A candidate violating an operational limit is discarded, not
   penalised. A penalty would let the search trade a maintenance budget against
   a tracking error, which is the asset owner decision, not the control
   designer one.
2. **Rank** the survivors on ``e_v`` in the transmission system alone.
3. **Break ties** on ``e_v`` in the distribution system, within a stated
   indifference band.

``e_q`` is reported and bounded above, never minimised: it is a means to an end
-- what matters is that interface-Q tracking buys a smaller ``e_v`` in the TS --
and minimising it directly is gameable by freezing the PCC setpoint.

Everything the rule needs is stated: the limits (:class:`tuning.stage2_search.Limits`),
the band (``--indifference``), and the baseline every improvement is quoted
against. There are no scoring weights.

Wear is only judged when ``--hours >= 6``. One tap is 0.33/h on a 3 h window
against 0.083/h at 12 h, so a per-day budget read off a short window is a coin
toss rather than a constraint.

Usage::

    python experiments/ch_9_parameter_selection/ch_9_3_select.py \
        --candidates results/THESIS_ch9_3_stage2/refine/candidates \
        --hours 12 --write-paramset ch9_selected
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _pareto(recs: List[Dict[str, Any]], x: str, y: str) -> List[Dict[str, Any]]:
    """Non-dominated set on two minimised criteria."""
    front: List[Dict[str, Any]] = []
    best = float("inf")
    for r in sorted(recs, key=lambda r: r["aggregate"][x]):
        v = r["aggregate"][y]
        if v < best - 1e-12:
            front.append(r)
            best = v
    return front


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ch_9_3_select")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--hours", type=float, required=True,
                   help="Bank length the records came from. Wear limits are "
                        "applied only at >= 6 h.")
    p.add_argument("--indifference", type=float, default=0.02,
                   help="Tie band on e_v(TS) before the DS tie-break applies.")
    p.add_argument("--taps-per-day", type=float, default=30.0)
    p.add_argument("--max-hunting", type=int, default=0,
                   help="Hunting reversals tolerated over the whole bank. A "
                        "hunting reversal is an opposite-direction tap pair "
                        "within --hunt-dwell-h with NO new disturbance in "
                        "between; a tap that moves on a trip and back on the "
                        "restore is not one.")
    p.add_argument("--hunt-dwell-h", type=float, default=2.0)
    p.add_argument("--shunt-ops-per-day", type=float, default=None,
                   help="Commits per MSC/MSR bank per day. No default: a bank "
                        "whose records carry shunt wear (written since "
                        "2026-10-22) is not judged without a stated budget.")
    p.add_argument("--max-shunt-hunting", type=int, default=0,
                   help="Bank hunting reversals tolerated over the whole bank, "
                        "with the tap definition (event-attributed).")
    p.add_argument("--write-paramset", default=None,
                   help="Write the winner to configs/paramsets/<name>.json.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the full report as JSON.")
    args = p.parse_args(argv)

    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    from tuning.stage2_search import Limits, check_limits, rank_lexicographic

    recs = [json.loads(p_.read_text(encoding="utf-8"))
            for p_ in sorted(Path(args.candidates).glob("cand_*.json"))]
    recs = [r for r in recs if r.get("aggregate")]
    if not recs:
        raise SystemExit(f"no usable candidate records in {args.candidates}")

    base = next((r for r in recs if r.get("is_baseline")), None)
    if base is None:
        raise SystemExit(
            "no baseline record in the bank. Every improvement has to be quoted "
            "against a named baseline; re-run the sweep so the Stage-I point is "
            "evaluated alongside the candidates.")
    b = base["aggregate"]

    wear = args.hours >= 6.0
    has_shunt_wear = any(r["aggregate"].get("has_shunt_wear") for r in recs)
    if wear and has_shunt_wear and args.shunt_ops_per_day is None:
        raise SystemExit(
            "the records carry switched-shunt wear (banks in the TS-MIQP), so "
            "the bank budget must be stated: pass --shunt-ops-per-day. It is "
            "an asset statement of the campaign protocol, not a default.")
    limits = Limits(tap_ops_per_day=args.taps_per_day,
                    max_hunting=args.max_hunting,
                    hunt_dwell_h=args.hunt_dwell_h,
                    shunt_ops_per_day=args.shunt_ops_per_day,
                    max_shunt_hunting=args.max_shunt_hunting)

    feasible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for r in recs:
        ok, why = check_limits(r["aggregate"], limits, r.get("lambda_max_B"),
                               wear_is_meaningful=wear, e_q_reference=b["e_q"],
                               window_hours=args.hours)
        (feasible if ok else rejected).append({**r, "_why": why})

    print(f"bank: {len(recs)} candidates at {args.hours:g} h  "
          f"(wear {'JUDGED' if wear else 'NOT judged'})")
    print(f"baseline {base['key']}: e_v_ts={b['e_v_ts']:.5f} "
          f"e_v_ds={b['e_v_ds']:.5f} e_q={b['e_q']:.0f} "
          f"lambda={base['lambda_max_B']:.3f}")
    ok_base = not any(r["key"] == base["key"] for r in rejected)
    if not ok_base:
        why = next(r["_why"] for r in rejected if r["key"] == base["key"])
        print(f"  !! the BASELINE itself is infeasible: {'; '.join(why)}")
    print(f"feasible: {len(feasible)} / {len(recs)}")

    if rejected:
        print("\nrejected:")
        for r in sorted(rejected, key=lambda r: r["aggregate"]["e_v_ts"]):
            print(f"  {r['key']}  e_v_ts={r['aggregate']['e_v_ts']:.5f}  "
                  f"{'; '.join(r['_why'])}")

    if not feasible:
        print("\nNO FEASIBLE CANDIDATE. The limits and the plant disagree; that "
              "is a result, not an error. Report the binding limit rather than "
              "relaxing it silently.")
        return 2

    ranked = rank_lexicographic(feasible, indifference=args.indifference)
    print(f"\nRANKED (e_v_ts, tie-break e_v_ds within {100*args.indifference:.0f} %)")
    hdr = (f"{'#':>3} {'key':>13} {'lam':>6} {'e_v_ts':>9} {'e_v_ds':>9} "
           f"{'e_q':>9} {'taps/d TS':>10} {'taps/d DS':>10} {'hunt':>6}"
           f" {'bank/d':>7} {'b.hunt':>6}")
    print(hdr)
    for i, r in enumerate(ranked, 1):
        a = r["aggregate"]
        tag = "  <- BASELINE" if r.get("is_baseline") else ""
        print(f"{i:>3} {r['key']:>13} {r['lambda_max_B']:>6.3f} "
              f"{a['e_v_ts']:>9.5f} {a['e_v_ds']:>9.5f} {a['e_q']:>9.0f} "
              f"{a['worst_tap_ops_per_day_tso']:>10.1f} "
              f"{a['worst_tap_ops_per_day_dso']:>10.1f} "
              f"{a.get('n_hunting', 0.0):>7.0f}"
              f" {a.get('worst_shunt_ops_per_day_bank', 0.0):>7.1f}"
              f" {a.get('n_shunt_hunting', 0.0):>6.0f}"
              f"{tag}")

    front = _pareto(feasible, "e_v_ts", "e_v_ds")
    print(f"\nPARETO FRONTIER e_v_ts vs e_v_ds ({len(front)} points)")
    for r in front:
        a = r["aggregate"]
        print(f"  {r['key']}  e_v_ts={a['e_v_ts']:.5f} ({100*(a['e_v_ts']/b['e_v_ts']-1):+.1f} %)"
              f"  e_v_ds={a['e_v_ds']:.5f} ({100*(a['e_v_ds']/b['e_v_ds']-1):+.1f} %)")

    win = ranked[0]
    a = win["aggregate"]
    print(f"\nSELECTED: {win['key']}")
    print(f"  e_v(TS) {a['e_v_ts']:.5f}  ({100*(a['e_v_ts']/b['e_v_ts']-1):+.1f} % vs baseline)")
    print(f"  e_v(DS) {a['e_v_ds']:.5f}  ({100*(a['e_v_ds']/b['e_v_ds']-1):+.1f} %)")
    print(f"  e_q     {a['e_q']:.0f}     ({a['e_q']/b['e_q']:.2f}x, reported not minimised)")
    print(f"  lambda_max(B) {win['lambda_max_B']:.4f}")
    if win.get("is_baseline"):
        print("  NOTE: the winner IS the baseline -- the search found no "
              "feasible improvement, which is a result worth stating plainly.")

    dom = [r for r in feasible
           if r["aggregate"]["e_v_ts"] < b["e_v_ts"]
           and r["aggregate"]["e_v_ds"] < b["e_v_ds"]]
    print(f"\n{len(dom)} candidate(s) dominate the baseline on BOTH voltage criteria.")

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "candidates_dir": str(args.candidates),
        "hours": args.hours, "wear_judged": wear,
        "indifference": args.indifference,
        "limits": {k: getattr(limits, k) for k in
                   ("tap_ops_per_day", "max_hunting", "hunt_dwell_h",
                    "shunt_ops_per_day", "max_shunt_hunting",
                    "avr_max_step_pu", "lambda_max_b", "max_v_violations",
                    "e_q_max_ratio")},
        "baseline": {"key": base["key"], "aggregate": b},
        "n_total": len(recs), "n_feasible": len(feasible),
        "rejected": [{"key": r["key"], "why": r["_why"]} for r in rejected],
        "ranked": [{"key": r["key"], "aggregate": r["aggregate"],
                    "lambda_max_B": r["lambda_max_B"],
                    "weights": r["weights"]} for r in ranked],
        "pareto": [r["key"] for r in front],
        "selected": {"key": win["key"], "weights": win["weights"],
                     "aggregate": a, "lambda_max_B": win["lambda_max_B"]},
        "dominating_baseline": [r["key"] for r in dom],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=float),
                            encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.write_paramset:
        fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
        for t in ("tso_period_s", "dso_period_s", "dt_s"):
            if t in win["weights"]:
                fields[t] = float(win["weights"][t])
        doc = {
            "name": args.write_paramset,
            "title": "Chapter-9.3 selected controller weights",
            "description": [
                f"Selected {datetime.now():%Y-%m-%d} from the Stage-II bank at "
                f"{args.hours:g} h by constraints plus a lexicographic order on "
                f"e_v(TS) with a {100*args.indifference:.0f} % indifference band "
                f"on e_v(DS).",
                "e_q is bounded above and reported, never minimised.",
                "No scoring weights enter the selection.",
            ],
            "provenance": {
                "producer": "experiments/ch_9_parameter_selection/ch_9_3_select.py",
                "candidates_dir": str(args.candidates),
                "hours": args.hours,
                "baseline_key": base["key"],
                "selected_key": win["key"],
                "lambda_max_B": win["lambda_max_B"],
                "aggregate": a,
                "limits": report["limits"],
                "indifference": args.indifference,
                "weight_fingerprint": weight_fingerprint(fields),
            },
            "fields": fields,
        }
        out = REPO_ROOT / "configs" / "paramsets" / f"{args.write_paramset}.json"
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
