#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_3_shortlist.py
========================================================
Stage II-a -> II-b: build the 12 h shortlist from the screening bank by a
stated rule, so the step that was assembled by hand on 2026-09-11 is a
reproducible part of the procedure.

The rule (2026-09-11 re-run, protocol docs/tuning/2026-10-16_protocol_
experiment_B_rerun.md §3.4)
--------------------------------------------------------------------------
Among the II-a candidates that pass the limits judgeable on the short bank
(:func:`tuning.stage2_search.check_limits` with ``wear_is_meaningful=False``,
e_q against the baseline of the same bank):

* the whole Pareto front on (e_v(TS), e_v(DS)),
* the best ``--top-ts`` on e_v(TS),
* the best ``--top-ds`` on e_v(DS),
* the baseline (the Stage-I point), whatever its rank,

deduplicated by weight fingerprint.  Two-sided on purpose: a list ranked on
the TS criterion alone would hand II-b only candidates that buy TS voltage
with DS voltage.

Writes a JSON list of weight dicts for ``ch_9_3_stage2_screen.py
--candidates-json`` (which re-screens each and always adds the baseline).

Author: Manuel Schwenke / Claude Code, 2026-10-16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ch_9_3_shortlist")
    p.add_argument("--candidates", type=Path, required=True,
                   help="II-a candidate records (cand_*.json).")
    p.add_argument("--hours", type=float, required=True,
                   help="Window length of the II-a bank (wear is not judged "
                        "below 6 h).")
    p.add_argument("--top-ts", type=int, default=14)
    p.add_argument("--top-ds", type=int, default=8)
    p.add_argument("--out", type=Path, required=True,
                   help="Shortlist JSON (list of weight dicts).")
    args = p.parse_args(argv)

    from experiments.ch_09_parameter_selection.ch_9_3_select import _pareto
    from tuning.parameters import weight_fingerprint
    from tuning.stage2_search import Limits, check_limits

    recs = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(args.candidates.glob("cand_*.json"))]
    recs = [r for r in recs if r.get("aggregate")]
    base = next((r for r in recs if r.get("is_baseline")), None)
    if base is None:
        raise SystemExit("no baseline record in the II-a bank")
    wear = args.hours >= 6.0
    feasible: List[Dict[str, Any]] = []
    for r in recs:
        ok, _why = check_limits(r["aggregate"], Limits(), r.get("lambda_max_B"),
                                wear_is_meaningful=wear,
                                e_q_reference=base["aggregate"]["e_q"],
                                window_hours=args.hours)
        if ok:
            feasible.append(r)

    front = _pareto(feasible, "e_v_ts", "e_v_ds")
    by_ts = sorted(feasible, key=lambda r: r["aggregate"]["e_v_ts"])[: args.top_ts]
    by_ds = sorted(feasible, key=lambda r: r["aggregate"]["e_v_ds"])[: args.top_ds]

    chosen: Dict[str, Dict[str, Any]] = {}
    roles: Dict[str, List[str]] = {}
    for role, group in (("baseline", [base]), ("pareto", front),
                        ("top_ts", by_ts), ("top_ds", by_ds)):
        for r in group:
            k = weight_fingerprint(r["weights"])
            chosen.setdefault(k, r)
            roles.setdefault(k, []).append(role)

    shortlist = [chosen[k]["weights"] for k in chosen]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(shortlist, indent=1, default=float),
                        encoding="utf-8")
    meta = {
        "rule": "baseline + Pareto(e_v_ts, e_v_ds) + top-ts on e_v_ts + "
                "top-ds on e_v_ds among II-a candidates passing check_limits "
                "(wear judged only for hours >= 6), deduplicated",
        "candidates_dir": str(args.candidates), "hours": args.hours,
        "n_records": len(recs), "n_feasible": len(feasible),
        "n_pareto": len(front), "top_ts": args.top_ts, "top_ds": args.top_ds,
        "n_shortlist": len(shortlist),
        "members": {k: roles[k] for k in chosen},
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[shortlist] {len(recs)} records, {len(feasible)} feasible, "
          f"Pareto {len(front)}, shortlist {len(shortlist)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
