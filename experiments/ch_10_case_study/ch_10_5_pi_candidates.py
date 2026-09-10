"""
ch_10_5_pi_candidates.py
========================

The two surviving S1 PI candidates, on the full 360-minute horizon.

Both carry the gains inherited from the 42-minute screen, ``k_p = 0.2`` on
both loops.  They differ in ONE field -- ``svr_rpr_voltage_priority`` -- so the
pair isolates the pilot-voltage safeguard and nothing else.  The gains are
deliberately NOT re-tuned here: S1 is the reference scheme, not the
contribution, and it needs to be a fair opponent rather than an optimal one.

Both are run regardless of how ``ch_10_4_pi_screen`` comes out.  Whichever is
chosen is needed at 360 minutes anyway, and the screen's short horizon reaches
only the generator trip at 30 min -- the line trip at 210 min, which is the
event the safeguard was built for, fires only here.

Config construction follows ``ch_10_4_pi_screen.main()`` exactly: the
authoritative factory via ``ch_10_1_variant_ladder.make_thesis_config``, the
``VARIANTS["S1"]`` overrides on top, then the two swept fields.  No weight is
restated -- see the ladder's module docstring for why that matters.

Run:
    python -m experiments.ch_10_case_study.ch_10_5_pi_candidates
    python -m experiments.ch_10_case_study.ch_10_5_pi_candidates --minutes 360
"""

from __future__ import annotations

import argparse
import json
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from experiments.paths import RESULTS_ROOT

OUT_DIR = Path(RESULTS_ROOT) / "ch10_s1_candidates"

#: (k_p on both loops, safeguard).  One field apart, by construction.
CANDIDATES = [
    dict(svr_k_p_rvr=0.2, svr_k_p_rpr=0.2, svr_rpr_voltage_priority=True),
    dict(svr_k_p_rvr=0.2, svr_k_p_rpr=0.2, svr_rpr_voltage_priority=False),
]


def _tag(c: Dict[str, Any]) -> str:
    return (f"kp{c['svr_k_p_rvr']:g}_"
            f"{'guard' if c['svr_rpr_voltage_priority'] else 'noguard'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=360)
    ap.add_argument("--only", default=None,
                    help="comma-separated tags to run, e.g. kp0.2_guard; "
                         "default runs both")
    args = ap.parse_args()

    wanted = ([t.strip() for t in args.only.split(",")] if args.only
              else [_tag(c) for c in CANDIDATES])
    unknown = set(wanted) - {_tag(c) for c in CANDIDATES}
    if unknown:
        raise SystemExit(f"unknown tag(s): {sorted(unknown)}")

    import importlib.util as u
    here = Path(__file__).parent
    spec = u.spec_from_file_location("lad", here / "ch_10_1_variant_ladder.py")
    lad = u.module_from_spec(spec)
    spec.loader.exec_module(lad)
    lad.HORIZON_MIN = args.minutes

    from experiments.runners.multi_tso_dso import run_multi_tso_dso
    from experiments.ch_10_case_study.ch_10_4_pi_screen import evaluate

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    failed = False

    for cand in CANDIDATES:
        tag = _tag(cand)
        if tag not in wanted:
            continue
        d = OUT_DIR / tag
        d.mkdir(parents=True, exist_ok=True)
        cfg = lad.make_thesis_config()
        for k, v in lad.VARIANTS["S1"].items():
            setattr(cfg, k, v)
        for k, v in cand.items():
            setattr(cfg, k, v)
        cfg.verbose = 1
        cfg.result_dir = str(d)
        fired, skipped = lad.fired_contingencies(cfg)
        print("\n" + "=" * 70)
        print(f"  S1  k_p={cfg.svr_k_p_rvr:g}  "
              f"safeguard={'on' if cfg.svr_rpr_voltage_priority else 'off'}  "
              f"horizon={args.minutes} min")
        print(f"  contingencies fired: {len(fired)}  {fired}")
        if skipped:
            print(f"  SKIPPED (beyond horizon): {skipped}")
        print("=" * 70, flush=True)

        # A failure is reported in full and ends the sweep.  Continuing would
        # produce a half-populated directory that looks complete on disk.
        try:
            log = run_multi_tso_dso(cfg)
        except Exception:
            print(f"  [{tag}] FAILED -- traceback follows", flush=True)
            traceback.print_exc()
            results[tag] = {"completed": False, "records": 0,
                            "traceback": traceback.format_exc()}
            failed = True
            break

        with open(d / "log.pkl", "wb") as f:
            pickle.dump(log, f)
        results[tag] = {"completed": True, "records": len(log),
                        "metrics": evaluate(log)}
        print(f"  [{tag}] wrote {len(log)} records -> {d/'log.pkl'}", flush=True)
        print(f"  [{tag}] {results[tag]['metrics']}", flush=True)

    # candidates.json MERGES: the sweep can be split across invocations
    # (one tag now, the other later), and a plain overwrite would drop the
    # entry written by the earlier call.
    summary_path = OUT_DIR / "candidates.json"
    merged: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            merged = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(results)
    summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'case':>16}{'records':>10}{'e_v':>10}{'e_v,p':>10}"
          f"{'peak gen':>11}{'peak line':>11}")
    print("-" * 78)
    for tag, r in merged.items():
        if not r or not r.get("completed"):
            print(f"{tag:>16}{'FAILED':>10}")
            continue
        m = r["metrics"]
        print(f"{tag:>16}{r['records']:>10d}{m['e_v']:>10.5f}{m['e_p']:>10.5f}"
              f"{m['peak_gen_trip']:>11.5f}{m['peak_line_trip']:>11.5f}")
    print(f"\nwritten: {OUT_DIR/'candidates.json'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
