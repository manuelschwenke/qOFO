#!/usr/bin/env python3
"""Pre-flight for the Stage-1 re-run: check the expensive things before paying.

A tier-1 sweep is ~9 h wall.  Every check below corresponds to a way the run can
finish and be worthless, and each has actually happened or was found while
preparing this re-run (2026-08-18):

  * the objective changed but the cache replayed old scores  -> scoring stamp
  * the bank changed but the cache replayed old scores       -> bank stamp
  * the relief was written absolute while dso_g_v is searched -> loop-gain drift
  * the archive dropped the field the new criterion needs     -> forced re-run
  * the guard threshold left at a value the reference cannot meet

Usage::

    python -m tuning_mc.preflight_rerun --ds-criterion guard --filter-ds \\
        --relief DSO_2=20,DSO_4=20 --scenario-set tier1

Exit code 0 = safe to launch; 1 = at least one blocking problem.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "


def _parse_relief(spec: str) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-criterion", choices=("v_rms", "guard"), default="guard")
    ap.add_argument("--filter-ds", action="store_true")
    ap.add_argument("--relief", default="",
                    help="e.g. DSO_2=20,DSO_4=20 (empty = none)")
    ap.add_argument("--scenario-set", default="tier1")
    ap.add_argument("--limits", type=Path, default=None,
                    help="the --limits file the run will use (omit = DEFAULTS)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from configs.config import apply_dso_v_relief  # noqa: F401
    from tuning._io import load_config_yaml
    from tuning.metrics import DS_GUARD_HEADROOM_PU, TrajectoryMetrics
    from tuning_mc.stage_1_search import (
        DEFAULT_BASELINE, DEFAULT_OUT, bank_fingerprint, build_config,
        load_limits, scoring_fingerprint,
    )
    import dataclasses

    out_dir = args.out or DEFAULT_OUT
    relief = _parse_relief(args.relief)
    problems = 0

    def say(status: str, msg: str) -> None:
        nonlocal problems
        if status is BAD:
            problems += 1
        print(f"[{status}] {msg}")

    print("=" * 74)
    print("  Stage-1 re-run pre-flight")
    print("=" * 74)

    # 1 -- objective selection is actually plumbed
    print("\n1. Objective")
    say(OK if args.ds_criterion == "guard" else WARN,
        f"ds_criterion = {args.ds_criterion}"
        + ("" if args.ds_criterion == "guard" else
           "  (v_rms measures the DSO profile's CENTRE, which the OLTC pins on "
           "v_set by construction -- it ranked the 2nd-worst area healthiest)"))
    say(OK if args.filter_ds else BAD,
        f"--filter-ds = {args.filter_ds}"
        + ("" if args.filter_ds else
           "  BLOCKING: f_ds is computed but NOT in the dominance test, so the "
           "search is free to spend DS voltage (measured 2026-08-15: -47 % f_ds "
           "for +15 % f_q)"))
    say(OK, f"guard band = [{0.90 + DS_GUARD_HEADROOM_PU:.3f}, "
            f"{1.10 - DS_GUARD_HEADROOM_PU:.3f}] pu  "
            f"(DS_GUARD_HEADROOM_PU = {DS_GUARD_HEADROOM_PU})")

    # 1b -- constraint limits.  Added 2026-08-18 after a tier-1 run was launched
    # without --limits: ConstraintLimits() defaults put rho_emp_p95 at 1.0
    # against the tier-1 file's 1.5, every candidate came back feasible=False on
    # g3 at a measured rho of 1.4357, and filter_accepts rejects infeasible
    # candidates outright -- the filter would have stayed empty for the whole
    # campaign, at ~9 h of compute.
    print("\n1b. Constraint limits")
    limits = load_limits(args.limits)
    if args.limits is None:
        say(BAD, "no --limits given: falls back to ConstraintLimits() DEFAULTS "
                 f"(rho_emp_p95={limits.rho_emp_p95}, "
                 f"tap_ops_per_h={limits.tap_ops_per_h:.3g}).  BLOCKING: the "
                 "defaults are not the calibrated set for any bank -- pass e.g. "
                 "tuning_mc/configs/limits_mc_v2_tier1.json")
    else:
        say(OK, f"limits = {args.limits}  (rho_emp_p95={limits.rho_emp_p95}, "
                f"tap_ops_per_h={limits.tap_ops_per_h:.3g}, "
                f"tap_reversals_per_h={limits.tap_reversals_per_h:.3g})")
        if limits.rho_emp_p95 < 1.44:
            say(WARN, f"rho_emp_p95 limit {limits.rho_emp_p95} is tight -- the "
                      "design point measured 1.4357 on this bank")
        if limits == load_limits(None):
            say(WARN, "the given file is identical to the package defaults")

    # 2 -- cache invalidation
    print("\n2. Evaluation cache")
    fp = scoring_fingerprint(args.ds_criterion, limits)
    say(OK, f"scoring fingerprint = {fp}   bank fingerprint = {bank_fingerprint()}")
    # stage_1_search writes to ``Path(args.out) / "evals"`` -- glob there, not
    # at the top level, or this check silently reports a clean run.
    evals = glob.glob(str(out_dir / "evals" / f"{args.scenario_set}_*.json"))
    if not evals:
        say(OK, f"no cached {args.scenario_set}_*.json in {out_dir} -- clean run")
    else:
        reusable = 0
        for f in evals:
            try:
                d = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                continue
            if d.get("_scoring_fingerprint") == fp:
                reusable += 1
        stale = len(evals) - reusable
        say(OK, f"{len(evals)} cached rows: {reusable} reusable, {stale} will be "
                f"re-evaluated (moved to *.json.scoring_changed)")
        if reusable and args.ds_criterion == "guard":
            say(WARN, f"{reusable} rows already carry this objective's stamp -- "
                      f"they are genuine cache hits, not a problem, but confirm "
                      f"they came from the run you think they did")

    # 3 -- relief pairing across the search range
    print("\n3. Per-DSO voltage relief")
    if not relief:
        say(WARN, "no relief configured; DSO_2 and DSO_4 will ride their bound "
                  "as in the 2026-08-18 baseline (headroom -0.0012 / -0.0010 pu)")
    else:
        base = load_config_yaml(Path(DEFAULT_BASELINE))
        say(OK, f"relief = {relief}")
        worst = 0.0
        for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
            cfg = build_config({"dso_g_v_ratio": ratio}, {}, base,
                               dso_v_relief=relief)
            for d, f in relief.items():
                r_gv = cfg.dso_g_v_per_area[d] / cfg.dso_g_v
                r_gw = cfg.dso_g_w_class[d]["dso_oltc"] / cfg.g_w_dso_oltc
                worst = max(worst, abs(r_gv / r_gw - 1.0))
        say(OK if worst < 1e-9 else BAD,
            f"OLTC loop gain invariant across dso_g_v_ratio in [0.25, 4]: "
            f"max drift {worst:.2e}"
            + ("" if worst < 1e-9 else "  BLOCKING: the tap will limit-cycle"))

    # 4 -- archive completeness
    print("\n4. Archive re-scorability")
    n_fields = len(dataclasses.fields(TrajectoryMetrics))
    has_guard = any(f.name == "guard_deficit_ds_pu"
                    for f in dataclasses.fields(TrajectoryMetrics))
    say(OK if has_guard else BAD,
        f"TrajectoryMetrics carries guard_deficit_ds_pu ({n_fields} fields total)")
    from tuning_mc.metrics import score_candidate

    class _R:
        def __init__(s):
            s.scenario_name, s.failure_reason = "probe", ""
            s.metrics = TrajectoryMetrics(guard_deficit_ds_pu=0.01)

    class _C:
        v_min_pu, v_max_pu, v_setpoint_pu = 0.90, 1.10, 1.03

    stored = score_candidate([_R()], _C()).per_scenario["probe"]
    say(OK if "metrics" in stored else BAD,
        "per-scenario archive carries the full metric vector "
        "(a later criterion change is then a re-score, not a re-run)")

    print("\n" + "=" * 74)
    if problems:
        print(f"  {problems} BLOCKING problem(s) -- do not launch")
    else:
        print("  clear to launch")
    print("=" * 74)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
