#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_14_procedure_v4.py
==============================================================
Weight selection by procedure v4 on the unit-commitment base
(``docs/tuning/2026-11-04_weight_selection_procedure_v4.md``; author decisions of
2026-11-04 in ``docs/tuning/2026-11-04_procedure_v4_requirements_draft.md``).

v4 is v3 (``ch_9_11_procedure_v3``, imported as ``V3``, with its amendments 1 and
2) with these changes, and otherwise the same code path:

* **Hunting.**  Tolerance 0 for taps and banks; the early stop at the first
  counted reversal.  A reversal counts only with a dwell below 1 h (v3: 2 h):
  slower reversals follow the load profile (author: "does not count as hunting").
* **Settling.**  Budget 3 TS periods: acceptance p95 <= 27, verification margin 25.
* **Go criterion** e_v_ts <= 0.01 (absolute; v3: 1.10 x 0.00827).
* **Screen.**  ``lambda_max(B) <= 1.546`` without the bank columns (B1b: banks are
  discrete and dwell-locked and are judged by the hunting filter), on W1-W6 plus
  the fresh operating points of run 0007 with G3 committed.  The design envelope
  is "G3 committed" (author's scope decision); theta_TS and theta_STS are read on
  W1-W6 as before.
* ``--config-set`` (repeatable) is applied to every simulated window, for the tap
  hold at bank switching if the author adopts it.
* Hosts: 10 workers here, pinned to ``--affinity-mask`` (inherited by every child),
  30 on e5power.

The changes are applied by :func:`configure`, not at import, so that importing
this module leaves ``V3`` unchanged (the v3 tests import both).

Usage (from the repository root, on e5wimi)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_14_procedure_v4.py --dry-run
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_14_procedure_v4.py \
        --reuse-stage1 results/ch_09_parameter_selection/0005_2026-11-03_164755_ch_9_11_procedure_v3 \
        --note "..."

Author: Manuel Schwenke / Claude Code (2026-11-04)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V2  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3  # noqa: E402

PROCEDURE = "docs/tuning/2026-11-04_weight_selection_procedure_v4.md"
RESULT = "ch9_selected_uc_2026_11_v4"
TOLERANCE = 0
N_PERIODS = 3
HUNT_DWELL_S = 3600.0
GO_E_V_TS = 0.01
LAMBDA_EXCLUDE = ("shunt",)
#: Fresh operating points of the eps_op re-measurement (thesis run 0007).
RUN_0007 = (ROOT / "results" / "ch_09_parameter_selection"
            / "0007_2026-11-04_111251_ch_9_13_eps_op_remeasure")
AFFINITY_MASK = 0x55555                        # e5wimi: 10 logical processors (author 2026-11-04)
CONFIG_SET: List[str] = []
LAM_OPS: List[Any] = []

_V3_SCREEN_ARGV = V3.screen_argv
_V3_VERIFY = V3.verify
_V3_WRITE_RESULT = V3.write_result
_P_PUSH_INPUTS = P.push_inputs


# -- the lambda bank ----------------------------------------------------------------------------

def g3_committed(op) -> bool:
    """G3 is the only machine of zone 2: its column has voltage curvature iff committed."""
    lc = next(l for l in op.layers if l.key == "TS_2")
    j = lc.classes.index("gen")
    return float(lc.S_V[j, j]) > 0.0


def lambda_bank_dir() -> Path:
    return P.RUN / "operating_points_lambda"


def prepare_lambda_bank(design_ops: Sequence[Any]) -> List[Any]:
    """W1-W6 plus run 0007's fresh points with G3 committed, copied into this run."""
    if not all(g3_committed(op) for op in design_ops):
        raise RuntimeError("a design window has G3 off; the v4 envelope is 'G3 committed'")
    dst = lambda_bank_dir()
    dst.mkdir(parents=True, exist_ok=True)
    extra = []
    for f in sorted((RUN_0007 / "operating_points_fresh").glob("op_*.npz")):
        with f.open("rb") as fh:
            op = pickle.load(fh)
        if g3_committed(op):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
            extra.append(op)
    LAM_OPS[:] = list(design_ops) + extra
    P.log(f"lambda bank: {len(design_ops)} design + {len(extra)} fresh G3-on points "
          f"({P.rel(dst)}), without {', '.join(LAMBDA_EXCLUDE)}")
    return LAM_OPS


def lam_v4(w: Dict[str, float]) -> float:
    from tuning import stage1_screen as s1
    if not LAM_OPS:
        raise RuntimeError("lambda bank not prepared")
    return s1.worst(s1.screen_weights(LAM_OPS, w, exclude_classes=LAMBDA_EXCLUDE)).lambda_max


def passes_screen(weights: Dict[str, float], ops) -> bool:
    """v4 screen: lambda on the lambda bank without the banks; theta_TS on ``ops``."""
    from tuning import stage1_screen as s1
    return (lam_v4(weights) <= s1.select_ceiling()
            and V2.theta_ts_min(weights, ops) <= V2.THETA_TS_MAX)


def stage2a(centre: Dict[str, float], ops, w_local: int, w_remote: int,
            split_local: Optional[float] = None) -> dict:
    """``V2.stage2a`` with the v4 lambda."""
    from tuning import stage1_screen as s1
    from tuning.stage2_search import Box, sobol_candidates
    out = P.RUN / "stage2a"
    spans = V2.spans_v2()
    draws = sobol_candidates(Box(centre=centre, spans=spans), V2.N_2A, seed=P.SEED)
    lam = [lam_v4(w) for w in draws]
    tts = [V2.theta_ts_min(w, ops) for w in draws]
    keep = [w for w, l, t in zip(draws, lam, tts)
            if l <= s1.select_ceiling() and t <= V2.THETA_TS_MAX]
    out.mkdir(parents=True, exist_ok=True)
    (out / "draws.json").write_text(json.dumps(
        {"seed": P.SEED, "n_draws": V2.N_2A, "centre": centre, "spans": spans,
         "select_ceiling": s1.select_ceiling(), "theta_ts_max_pu": V2.THETA_TS_MAX,
         "lambda_bank_points": len(LAM_OPS), "lambda_exclude": list(LAMBDA_EXCLUDE),
         "lambda_max_all": lam, "theta_ts_min_all": tts, "n_survivors": len(keep)},
        indent=1), "utf-8")
    P.log(f"stage II-a: {len(keep)} / {V2.N_2A} draws pass the v4 screen "
          f"(lambda <= {s1.select_ceiling():.4f} on {len(LAM_OPS)} points without banks: "
          f"{sum(1 for l in lam if l <= s1.select_ceiling())}, theta_TS <= "
          f"{V2.THETA_TS_MAX:g}: {sum(1 for t in tts if t <= V2.THETA_TS_MAX)})")
    P.simulate(out, keep, w_local, w_remote, split_local)
    return V3.stage_select([out], "2a", ops)


# -- arguments, verification, result -------------------------------------------------------------

def screen_argv(out: Path, cj: Path, workers: int, windows: Optional[str] = None,
                skip_baseline: bool = False) -> List[str]:
    """v3's units (windows, early stop at TOLERANCE + 1, start-up exemption) plus the
    v4 hunting dwell, the v4 lambda and ``--config-set``."""
    a = _V3_SCREEN_ARGV(out, cj, workers, windows=windows, skip_baseline=skip_baseline)
    a += ["--hunt-dwell-s", f"{HUNT_DWELL_S:g}", "--lambda-exclude", ",".join(LAMBDA_EXCLUDE),
          "--lambda-extra-ops", P.rel(lambda_bank_dir())]
    for c in CONFIG_SET:
        a += ["--config-set", c]
    return a


def verify(theta: float, name: str, w: Dict[str, float], workers: int) -> Dict[str, Any]:
    """v3's verification, judged at the v4 margin (a reused record keeps its numbers)."""
    o = _V3_VERIFY(theta, name, w, workers)
    ok = (o.get("p95", 1e9) <= V2.VERIFY_P95
          and o.get("median_residual_mvar", 1e9) <= V2.VERIFY_RESIDUAL)
    o["verified"] = ok
    o["margin_v4"] = {"p95": V2.VERIFY_P95, "median_residual_mvar": V2.VERIFY_RESIDUAL}
    path = P.RUN / "stage1" / "verification" / "verification_v3.json"
    stored = json.loads(path.read_text("utf-8")) if path.exists() else {}
    stored[f"{theta:g}"] = o
    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    P.log(f"  v4 margin: theta_STS {theta:g} {'PASS' if ok else 'FAIL'} (p95 {o.get('p95')} "
          f"<= {V2.VERIFY_P95:g}, median residual {o.get('median_residual_mvar')} <= "
          f"{V2.VERIFY_RESIDUAL:g})")
    return o


def push_inputs() -> None:
    _P_PUSH_INPUTS()
    files = sorted(lambda_bank_dir().glob("op_*.npz"))
    for f in files:
        P.push(f, P.remote_path(f))
    P.log(f"pushed {len(files)} lambda-bank points to {P.REMOTE}")


def write_result(win: dict, rep: dict, c_res: dict, w5: dict, ver: Dict[float, Any],
                 go: Dict[str, Any], ops) -> Path:
    path = _V3_WRITE_RESULT(win, rep, c_res, w5, ver, go, ops)
    doc = json.loads(path.read_text("utf-8"))
    doc["title"] = "Controller weights selected by procedure v4 on the unit-commitment base"
    doc["description"] = [
        f"Selected by {PROCEDURE}: v3 (amendments 1 and 2) with hunting tolerance 0 and "
        f"dwell < {HUNT_DWELL_S / 3600:g} h, settling budget {N_PERIODS} periods, go "
        f"e_v_ts <= {GO_E_V_TS:g}, lambda_max(B) <= 1.546 without the banks on W1-W6 plus "
        f"{len(LAM_OPS) - len(ops)} G3-on points of run 0007 (envelope: G3 committed)"
        + (f"; config {', '.join(CONFIG_SET)}" if CONFIG_SET else "") + "."]
    doc["provenance"]["procedure"] = PROCEDURE
    doc["provenance"]["producer"] = P.rel(Path(__file__))
    doc["provenance"]["v4"] = {
        "hunting_tolerance": TOLERANCE, "hunt_dwell_s": HUNT_DWELL_S,
        "settling_periods": N_PERIODS, "go_e_v_ts": GO_E_V_TS,
        "lambda_exclude": list(LAMBDA_EXCLUDE), "lambda_bank_points": len(LAM_OPS),
        "lambda_max_B_v4": lam_v4(win["weights"]), "config_set": list(CONFIG_SET)}
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    return path


# -- configuration ------------------------------------------------------------------------------

_PATCHES = (
    (V3, "PROCEDURE", PROCEDURE), (V3, "RESULT", RESULT),
    (V2, "PROCEDURE", PROCEDURE), (V2, "RESULT", RESULT),
    (V3, "TOLERANCE", TOLERANCE), (V3, "N_PERIODS", N_PERIODS),
    (V2, "VERIFY_P95", N_PERIODS * 9 - 2.0),
    (V3, "V2_E_V_TS", GO_E_V_TS), (V3, "GO_FACTOR", 1.0),
    (V2, "passes_screen", passes_screen), (V2, "stage2a", stage2a),
    (V3, "screen_argv", screen_argv), (P, "screen_argv", screen_argv),
    (V3, "verify", verify), (V3, "write_result", write_result),
    (P, "push_inputs", push_inputs),
)


def configure(config_set: Sequence[str] = ()) -> Dict[Any, Any]:
    """Apply the v4 changes to V3, V2 and P; returns what :func:`restore` needs."""
    from experiments.ch_09_parameter_selection import ch_9_3_amendment1 as _A1
    if abs(_A1.N_INNER - 9) > 0:
        raise RuntimeError(f"N_inner is {_A1.N_INNER}, the v4 margin assumes 9")
    prev = {(m, k): getattr(m, k) for m, k, _ in _PATCHES}
    for m, k, v in _PATCHES:
        setattr(m, k, v)
    CONFIG_SET[:] = list(config_set)
    return prev


def restore(prev: Dict[Any, Any]) -> None:
    for (m, k), v in prev.items():
        setattr(m, k, v)
    CONFIG_SET.clear()


def _pin(mask: Optional[int]) -> None:
    if mask and sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            raise OSError(f"SetProcessAffinityMask({mask:#x}) failed")


# -- main ---------------------------------------------------------------------------------------

def procedure(args) -> int:
    from tuning import stage1_screen as s1
    P.log(f"procedure v4: hunting tolerance {TOLERANCE}, dwell < {HUNT_DWELL_S:g} s; "
          f"settling {N_PERIODS} periods (p95 <= {N_PERIODS * 9}, verification <= "
          f"{V2.VERIFY_P95:g}); go e_v_ts <= {GO_E_V_TS:g}; lambda without "
          f"{', '.join(LAMBDA_EXCLUDE)} on the G3-on bank; config_set {CONFIG_SET or 'none'}")
    ops = V2.load_ops()
    prepare_lambda_bank(ops)
    if args.dry_run:
        from tuning.stage2_search import Box, sobol_candidates
        w, _ = V3.design(ops, V3.THETA_LADDER[0])
        draws = sobol_candidates(Box(centre=w, spans=V2.spans_v2()), V2.N_2A, seed=P.SEED)
        n4 = sum(1 for x in draws if passes_screen(x, ops))
        n3 = sum(1 for x in draws
                 if s1.worst(s1.screen_weights(ops, x)).lambda_max <= s1.select_ceiling()
                 and V2.theta_ts_min(x, ops) <= V2.THETA_TS_MAX)
        P.log(f"dry run: Stage-I point lambda v4 {lam_v4(w):.4f} (v3 rule "
              f"{s1.worst(s1.screen_weights(ops, w)).lambda_max:.4f}); II-a draws passing: "
              f"v4 {n4} / {V2.N_2A}, v3 rule {n3} / {V2.N_2A}")
        P.log("dry run: screen arguments " + " ".join(
            screen_argv(P.RUN / "x", P.RUN / "x" / "c.json", args.local_workers)))
        return 0
    return V3.procedure(args)


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_14_procedure_v4")
    p.add_argument("--local-workers", type=int, default=10)
    p.add_argument("--remote-workers", type=int, default=30)
    p.add_argument("--verify-workers", type=int, default=30)
    p.add_argument("--confirm-workers", type=int, default=30)
    p.add_argument("--local-speed", type=float, default=0.7)
    p.add_argument("--affinity-mask", type=lambda x: int(x, 0), default=AFFINITY_MASK)
    p.add_argument("--config-set", action="append", default=[], metavar="FIELD=VALUE",
                   help="Passed to every simulated window (ch_9_3_stage2_screen.py).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--reuse-stage1", type=Path, default=None,
                   help="Run folder whose stage1/verification is taken over.")
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    _pin(args.affinity_mask)
    configure(args.config_set)
    P.START = V2.V1_START                  # operating points only, until Stage I is written
    if args.dry_run:
        P.RUN = ROOT / "results" / "_dry_ch9_14" / f"{datetime.now():%Y%m%d-%H%M%S}"
        P.RUN.mkdir(parents=True)
        return procedure(args)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            parameters={"procedure": PROCEDURE, "base": "procedure v3 with amendments 1, 2",
                        "hunting_tolerance": TOLERANCE, "hunt_dwell_s": HUNT_DWELL_S,
                        "settling_periods": N_PERIODS, "go_e_v_ts": GO_E_V_TS,
                        "lambda_exclude": list(LAMBDA_EXCLUDE),
                        "lambda_bank": "W1-W6 + G3-on points of run 0007",
                        "search_windows": list(V3.SEARCH), "n_2a": V2.N_2A,
                        "rounds_max": P.MAX_ROUNDS, "result_paramset": RESULT,
                        "config_set": list(args.config_set),
                        "reuse_stage1": str(args.reuse_stage1),
                        "workers": f"{args.local_workers} here (mask {args.affinity_mask:#x}) "
                                   f"+ {args.remote_workers} on {P.REMOTE}"},
            time_series=("Stage-I verification and settling confirmation: 120 isolated "
                         "interface steps in W1-W4, W6; Stage II: 12-h simulations of W1, "
                         "W3, W4, W6 (early stop); band completed on W2; W5 report only"),
            note=args.note or "Ch 9 weight selection, procedure v4",
            tag=args.tag, resume=args.resume) as trun:
        P.RUN = trun.root
        P.log(f"run folder {P.RUN}")
        t0 = time.time()
        try:
            rc = procedure(args)
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            P.log(f"driver error: {type(exc).__name__}: {exc}")
            (P.RUN / "FAILED.txt").write_text(traceback.format_exc(), "utf-8")
            rc = 1
        trun.describe_output("driver.log", "every stage with times")
        trun.describe_output("operating_points_lambda/", "the G3-on points of the v4 lambda bank")
        trun.describe_output("selection/", "theta, engagement, unions, ch_9_3_select reports")
        trun.describe_output("stage2c/", "bank price on its own")
        trun.describe_output("complete/", "the band completed on W2")
        trun.describe_output("expC/", "settling confirmation in rank order")
        P.log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
