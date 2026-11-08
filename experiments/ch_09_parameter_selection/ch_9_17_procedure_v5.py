#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_17_procedure_v5.py
==============================================================
Weight selection by procedure v5 on the study network ``rural_700_north_res``
(``docs/tuning/2026-11-08_weight_selection_procedure_v5.md``; author decisions of
2026-11-08: "Bound g_w_pcc", G3-off states in the lambda bank plus a report).

v5 is v4 (``ch_9_14_procedure_v4``, imported as ``V4``) with these changes, and
otherwise the same code path:

* **Network.**  The config default ``rural_700_north_res``; W1-W6 captured anew
  into this run (v4 copied run 0001's ``rural_700`` captures).
* **lambda bank.**  W1-W6 plus the ``ch_9_13`` grid (96 starts) captured on this
  network with a 12-h and a 3-h horizon.  Both G3 states enter (v4: G3 committed
  only); the banks stay out of B.
* **STS reactive engagement.**  On the G3-off points of the bank,
  ``kappa_p = lambda_max(g_v H_V,pcc^T H_V,pcc)`` of TS zone 2.  A candidate must
  satisfy ``min_p kappa_p / g_w_pcc >= LAMBDA_STS_MIN`` (responsiveness) besides
  the v4 screen (which now contains the G3-off points).  The II-a box of g_w_pcc
  is ``[max_p kappa_p / ceiling, min_p kappa_p / LAMBDA_STS_MIN]``, log-uniform;
  the Stage-I g_w_pcc is its geometric centre.
* Result ``ch9_selected_nr_2026_11_v5``; Stage-I sets ``stage1_v5_nr_2026_11_*``;
  e5power worktree ``F:\\qOFO_runs\\qOFO_v5``.

The changes are applied by :func:`configure`, not at import, so that importing
this module leaves ``V4``, ``V3`` and ``V2`` unchanged.

Usage (from the repository root, on e5wimi)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_17_procedure_v5.py --dry-run
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_17_procedure_v5.py \
        --reuse-ops results/_dry_ch9_17/<stamp> --config-set sts_coupler_hold_after_bank=1 \
        --note "..."

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V2  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_14_procedure_v4 as V4  # noqa: E402

PROCEDURE = "docs/tuning/2026-11-08_weight_selection_procedure_v5.md"
RESULT = "ch9_selected_nr_2026_11_v5"
NETWORK = "rural_700_north_res"
VARIANT_NAME = "stage1_v5_nr_2026_11_pq{:g}"
#: Responsiveness of the STS in the scarce state: the PCC-only loop gain of TS
#: zone 2 at every G3-off point of the bank (half the error per TS period).
LAMBDA_STS_MIN = 0.5
SCARCE_ZONE = "TS_2"
#: The lambda-bank grid of ch_9_13, captured with both horizons.
BANK_HOURS = (12.0, 3.0)
REMOTE_ROOT = r"F:\qOFO_runs\qOFO_v5"
AFFINITY_MASK = 0x5555555555                   # e5wimi: 20 physical cores (author 2026-11-08)
BOUNDS: Dict[str, Any] = {}

_ORIG: Dict[str, Any] = {}


# -- operating points ---------------------------------------------------------------------------

def bank_conditions() -> List[Tuple[str, datetime, float]]:
    from experiments.ch_09_parameter_selection.ch_9_13_eps_op_remeasure import fresh_conditions
    return [(f"{name}_{h:g}h", start, h) for h in BANK_HOURS for name, start in fresh_conditions()]


def _capture_config():
    from configs.paramsets import load_paramset
    from tuning.parameters import apply_to_config
    start = {k: float(v) for k, v in load_paramset(V2.V1_START)["fields"].items()}
    return apply_to_config(P.base_config(), start)


def _capture_one(job: Tuple[str, str, float, str]) -> Tuple[str, float, Optional[str]]:
    """Capture one condition into ``out_dir``; returns (name, seconds, error)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    name, iso, hours, out_dir = job
    from tuning import stage1_screen as s1
    t0 = time.time()
    try:
        s1.load_or_build_operating_points(_capture_config(), [(name, datetime.fromisoformat(iso))],
                                          Path(out_dir), verbose=False,
                                          horizon_s=3600.0 * hours)
        return name, time.time() - t0, None
    except Exception as exc:  # a condition whose capture fails is reported, not dropped silently
        return name, time.time() - t0, f"{type(exc).__name__}: {exc}"


def load_ops():
    """W1-W6 on this network, captured into this run (12 h, as v4)."""
    cfg = P.base_config()
    if cfg.scenario != NETWORK:
        raise RuntimeError(f"config scenario {cfg.scenario!r}; v5 runs on {NETWORK!r}")
    ops = P.load_ops()
    if len(ops) != 6:
        raise RuntimeError(f"{len(ops)} design operating points, expected W1-W6")
    return ops


def capture_bank(workers: int) -> List[Any]:
    """The 192 bank points, captured in parallel (cached per file)."""
    import pickle
    dst = V4.lambda_bank_dir()
    dst.mkdir(parents=True, exist_ok=True)
    # A cached point returns at once: load_or_build_operating_points keys its file
    # on condition, scenario and network (incl. the horizon).
    jobs = [(n, s.isoformat(), h, str(dst)) for n, s, h in bank_conditions()]
    n0 = len(list(dst.glob("op_*.npz")))
    P.log(f"lambda bank: {len(jobs)} points ({n0} files present), {workers} workers")
    failed = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for name, dt, err in ex.map(_capture_one, jobs):
            if err:
                failed.append((name, err))
                P.log(f"  capture {name} failed after {dt:.0f} s: {err}")
    (dst.parent / "lambda_bank_failures.json").write_text(json.dumps(failed, indent=1), "utf-8")
    ops = []
    for f in sorted(dst.glob("op_*.npz")):
        with f.open("rb") as fh:
            ops.append(pickle.load(fh))
    return ops


def prepare_lambda_bank(design_ops: Sequence[Any], workers: int = 20) -> List[Any]:
    """W1-W6 plus the grid at 12 h and 3 h, both G3 states (v5 change 2)."""
    if not all(V4.g3_committed(op) for op in design_ops):
        raise RuntimeError("a design window has G3 off; v5 expects W1-W6 to commit G3")
    extra = capture_bank(workers)
    V4.LAM_OPS[:] = list(design_ops) + extra
    n_off = sum(1 for op in V4.LAM_OPS if not V4.g3_committed(op))
    P.log(f"lambda bank: {len(design_ops)} design + {len(extra)} grid points "
          f"({P.rel(V4.lambda_bank_dir())}); G3 off at {n_off}, without "
          f"{', '.join(V4.LAMBDA_EXCLUDE)}")
    return V4.LAM_OPS


# -- the g_w_pcc bounds -------------------------------------------------------------------------

def kappa(op, g_v: float) -> float:
    """lambda_max(g_v H_V,pcc^T H_V,pcc) of TS zone 2: its PCC-only loop gain times g_w_pcc."""
    import numpy as np
    lc = next(l for l in op.layers if l.key == SCARCE_ZONE)
    idx = [i for i, c in enumerate(lc.classes) if c == "pcc"]
    if not idx:
        raise RuntimeError(f"{op.condition}: {SCARCE_ZONE} has no PCC column")
    return float(np.linalg.eigvalsh(g_v * lc.S_V[np.ix_(idx, idx)])[-1])


def scarce_points() -> List[Any]:
    return [op for op in V4.LAM_OPS if not V4.g3_committed(op)]


def compute_bounds(g_v: float) -> Dict[str, Any]:
    from tuning import stage1_screen as s1
    pts = scarce_points()
    if not pts:
        raise RuntimeError("the lambda bank has no G3-off point; the v5 bounds are undefined")
    k = {op.condition: kappa(op, g_v) for op in pts}
    lo = max(k.values()) / s1.select_ceiling()
    hi = min(k.values()) / LAMBDA_STS_MIN
    BOUNDS.clear()
    BOUNDS.update({"g_v": g_v, "lambda_sts_min": LAMBDA_STS_MIN,
                   "ceiling": s1.select_ceiling(), "n_scarce_points": len(pts),
                   "kappa": k, "kappa_min": min(k.values()), "kappa_max": max(k.values()),
                   "g_w_pcc_lo": lo, "g_w_pcc_hi": hi,
                   "g_w_pcc_centre": math.sqrt(lo * hi) if hi > lo else float("nan"),
                   "span_decades": 0.5 * math.log10(hi / lo) if hi > lo else 0.0})
    return BOUNDS


def lambda_sts(w: Dict[str, float]) -> float:
    """The smallest PCC-only loop gain of TS zone 2 over the G3-off points."""
    return BOUNDS["kappa_min"] * float(w["g_v"]) / BOUNDS["g_v"] / float(w["g_w_pcc"])


def responsive(w: Dict[str, float]) -> bool:
    return lambda_sts(w) >= LAMBDA_STS_MIN * (1.0 - 1e-9)


def passes_screen(weights: Dict[str, float], ops) -> bool:
    """v4 screen on the v5 bank, plus the responsiveness bound."""
    return responsive(weights) and V4.passes_screen(weights, ops)


def spans_v2() -> Dict[str, float]:
    """v2/v4 spans with the g_w_pcc span set to the half-width of its bound interval."""
    s = dict(_ORIG["V2.spans_v2"]())
    s["g_w_pcc"] = float(BOUNDS["span_decades"])
    return s


def stage2a(centre: Dict[str, float], ops, w_local: int, w_remote: int,
            split_local: Optional[float] = None) -> dict:
    """``V4.stage2a`` with the v5 bounds."""
    from tuning import stage1_screen as s1
    from tuning.stage2_search import Box, sobol_candidates
    out = P.RUN / "stage2a"
    spans = V2.spans_v2()
    draws = sobol_candidates(Box(centre=centre, spans=spans), V2.N_2A, seed=P.SEED)
    lam = [V4.lam_v4(w) for w in draws]
    tts = [V2.theta_ts_min(w, ops) for w in draws]
    lst = [lambda_sts(w) for w in draws]
    keep = [w for w, l, t, r in zip(draws, lam, tts, lst)
            if l <= s1.select_ceiling() and t <= V2.THETA_TS_MAX
            and r >= LAMBDA_STS_MIN * (1.0 - 1e-9)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "draws.json").write_text(json.dumps(
        {"seed": P.SEED, "n_draws": V2.N_2A, "centre": centre, "spans": spans,
         "select_ceiling": s1.select_ceiling(), "theta_ts_max_pu": V2.THETA_TS_MAX,
         "lambda_sts_min": LAMBDA_STS_MIN, "bounds": BOUNDS,
         "lambda_bank_points": len(V4.LAM_OPS), "lambda_exclude": list(V4.LAMBDA_EXCLUDE),
         "lambda_max_all": lam, "theta_ts_min_all": tts, "lambda_sts_all": lst,
         "n_survivors": len(keep)}, indent=1, default=float), "utf-8")
    P.log(f"stage II-a: {len(keep)} / {V2.N_2A} draws pass the v5 screen (lambda <= "
          f"{s1.select_ceiling():.4f} on {len(V4.LAM_OPS)} points without banks: "
          f"{sum(1 for l in lam if l <= s1.select_ceiling())}, theta_TS <= "
          f"{V2.THETA_TS_MAX:g}: {sum(1 for t in tts if t <= V2.THETA_TS_MAX)}, lambda_STS >= "
          f"{LAMBDA_STS_MIN:g}: {sum(1 for r in lst if r >= LAMBDA_STS_MIN * (1 - 1e-9))})")
    P.simulate(out, keep, w_local, w_remote, split_local)
    return V3.stage_select([out], "2a", ops)


# -- Stage I and the result ---------------------------------------------------------------------

def design(ops, theta: float = V3.THETA_LADDER[0]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """v3's Stage I (STS weights and bank price designed here, g_v, g_w_der and
    g_w_tso_oltc from v2's Stage I, g_w_gen 5e6) with g_w_pcc at the centre of its
    v5 interval."""
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    wd, rep = s1.design_stage1(
        ops, pi_q=V3.PI_Q, pi_v_ds=V2.PI_V_DS, tau=V2.TAU, g_w_gen=V2.G_W_GEN,
        d_engage_ts=V3.D_ENGAGE["ts"], d_engage_sts=theta,
        d_engage_shunt=V3.D_ENGAGE["shunt"], lambda_target_ts=V2.LAMBDA_TARGET["ts"],
        lambda_target_sts=V2.LAMBDA_TARGET["sts"])
    w = {k: float(v) for k, v in load_paramset(V3.V2_STAGE1)["fields"].items()}
    for k in (*V2.STS_FIELDS, "g_w_tso_shunt"):
        w[k] = float(wd[k])
    w["g_w_gen"] = V3.G_W_GEN
    w["g_w_pcc"] = float(BOUNDS["g_w_pcc_centre"])
    if not passes_screen(w, ops):
        raise RuntimeError(f"the v5 Stage-I point fails the analytic screen: {w} "
                           f"(lambda v4 {V4.lam_v4(w):.4f}, theta_TS "
                           f"{V2.theta_ts_min(w, ops):.4f}, lambda_STS {lambda_sts(w):.4f})")
    rep["op_horizon_s"] = 3600.0 * P.HOURS
    rep["ts_continuous_from"] = V3.V2_STAGE1
    rep["g_w_pcc_from"] = "v5 interval centre"
    return w, rep


def write_result(win: dict, rep: dict, c_res: dict, w5: dict, ver: Dict[float, Any],
                 go: Dict[str, Any], ops) -> Path:
    path = V4.write_result(win, rep, c_res, w5, ver, go, ops)
    doc = json.loads(path.read_text("utf-8"))
    doc["title"] = "Controller weights selected by procedure v5 on rural_700_north_res"
    doc["description"] = [
        f"Selected by {PROCEDURE}: v4 on the study network {NETWORK}, lambda bank W1-W6 plus "
        f"the ch_9_13 grid at 12 h and 3 h with both G3 states, and g_w_pcc bounded by the "
        f"STS responsiveness lambda_pcc >= {LAMBDA_STS_MIN:g} at every G3-off point and the "
        f"lambda ceiling {BOUNDS['ceiling']:.4g}; config "
        f"{', '.join(V4.CONFIG_SET) or 'none'}."]
    doc["provenance"]["procedure"] = PROCEDURE
    doc["provenance"]["producer"] = P.rel(Path(__file__))
    doc["provenance"]["network"] = NETWORK
    doc["provenance"]["v5"] = {
        "lambda_sts_min": LAMBDA_STS_MIN, "lambda_sts_selected": lambda_sts(win["weights"]),
        "g_w_pcc_interval": [BOUNDS["g_w_pcc_lo"], BOUNDS["g_w_pcc_hi"]],
        "n_scarce_points": BOUNDS["n_scarce_points"], "kappa_min": BOUNDS["kappa_min"],
        "kappa_max": BOUNDS["kappa_max"], "lambda_bank_points": len(V4.LAM_OPS),
        "bank_hours": list(BANK_HOURS)}
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    return path


# -- configuration ------------------------------------------------------------------------------

def configure(config_set: Sequence[str] = ()) -> Dict[Any, Any]:
    """v4's changes, then the v5 ones; returns what :func:`restore` needs."""
    prev4 = V4.configure(config_set)
    patches = (
        (V4, "RESULT", RESULT), (V4, "PROCEDURE", PROCEDURE),
        (V3, "RESULT", RESULT), (V3, "PROCEDURE", PROCEDURE),
        (V2, "RESULT", RESULT), (V2, "PROCEDURE", PROCEDURE),
        (V3, "VARIANT_NAME", VARIANT_NAME), (V2, "OPS_SOURCE", V2.OPS_SOURCE),
        (V2, "load_ops", load_ops), (V2, "passes_screen", passes_screen),
        (V2, "stage2a", stage2a), (V2, "spans_v2", spans_v2),
        (V3, "design", design), (V3, "write_result", write_result),
        (P, "REMOTE_ROOT", REMOTE_ROOT),
    )
    prev = {(m, k): getattr(m, k) for m, k, _ in patches}
    _ORIG["V2.spans_v2"] = V2.spans_v2
    for m, k, v in patches:
        setattr(m, k, v)
    return {"v4": prev4, "v5": prev}


def restore(prev: Dict[Any, Any]) -> None:
    for (m, k), v in prev["v5"].items():
        setattr(m, k, v)
    V4.restore(prev["v4"])
    BOUNDS.clear()
    V4.LAM_OPS.clear()


# -- main ---------------------------------------------------------------------------------------

def reuse_ops(src: Path) -> None:
    """Copy weight-independent captures (W1-W6 and the bank) of an earlier run or dry run."""
    n = 0
    for sub in ("operating_points", "operating_points_lambda"):
        dst = P.RUN / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / sub).glob("op_*.npz")):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
                n += 1
    P.log(f"reused {n} captured operating point(s) from {P.rel(src)}")


def procedure(args) -> int:
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    from tuning.stage2_search import Box, sobol_candidates
    P.log(f"procedure v5 on {NETWORK}: v4 rules, lambda bank with both G3 states, "
          f"lambda_STS >= {LAMBDA_STS_MIN:g} on {SCARCE_ZONE}; config_set "
          f"{V4.CONFIG_SET or 'none'}; e5power root {P.REMOTE_ROOT}")
    if args.reuse_ops is not None:
        reuse_ops((ROOT / args.reuse_ops).resolve())
    V2.OPS_SOURCE = P.ops_dir()            # Stage-I provenance: this run's own captures
    ops = load_ops()
    prepare_lambda_bank(ops, args.local_workers)
    g_v = float(load_paramset(V3.V2_STAGE1)["fields"]["g_v"])
    b = compute_bounds(g_v)
    (P.RUN / "v5_bounds.json").write_text(json.dumps(b, indent=1, default=float), "utf-8")
    P.log(f"v5 bounds: {b['n_scarce_points']} G3-off points, kappa {b['kappa_min']:.4g}.."
          f"{b['kappa_max']:.4g}; g_w_pcc in [{b['g_w_pcc_lo']:.4g}, {b['g_w_pcc_hi']:.4g}] "
          f"(centre {b['g_w_pcc_centre']:.4g}, +-{b['span_decades']:.3f} decade)")
    if not b["g_w_pcc_hi"] > b["g_w_pcc_lo"]:
        (P.RUN / "FAILED.txt").write_text("v5: the g_w_pcc interval is empty\n", "utf-8")
        P.log("the g_w_pcc interval is empty; the campaign stops before Stage I")
        return 2
    if args.dry_run:
        for theta in V3.THETA_LADDER:
            w, _ = design(ops, theta)
            d = V2.describe(w, ops)
            draws = sobol_candidates(Box(centre=w, spans=V2.spans_v2()), V2.N_2A, seed=P.SEED)
            n = sum(1 for x in draws if passes_screen(x, ops))
            P.log(f"dry run theta {theta:g}: {', '.join(f'{k} {v:.5g}' for k, v in w.items())}")
            P.log(f"  lambda v5 bank {V4.lam_v4(w):.4f} (W1-W6 only {d['lambda_max_B']:.4f}), "
                  f"lambda_STS {lambda_sts(w):.4f}, theta_STS {d['theta_sts_min']:.4f}, "
                  f"theta_TS {d['theta_ts_min']:.4f}; II-a draws passing: {n} / {V2.N_2A}")
        P.log("dry run: screen arguments " + " ".join(
            V4.screen_argv(P.RUN / "x", P.RUN / "x" / "c.json", args.local_workers)))
        return 0
    return V3.procedure(args)


def _set_threads() -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_17_procedure_v5")
    p.add_argument("--local-workers", type=int, default=20)
    p.add_argument("--remote-workers", type=int, default=30)
    p.add_argument("--verify-workers", type=int, default=30)
    p.add_argument("--confirm-workers", type=int, default=30)
    p.add_argument("--local-speed", type=float, default=0.75)
    p.add_argument("--affinity-mask", type=lambda x: int(x, 0), default=AFFINITY_MASK)
    p.add_argument("--config-set", action="append", default=[], metavar="FIELD=VALUE",
                   help="Passed to every simulated window (ch_9_3_stage2_screen.py).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--reuse-stage1", type=Path, default=None,
                   help="Run folder whose stage1/verification is taken over.")
    p.add_argument("--reuse-ops", type=Path, default=None,
                   help="Run or dry-run folder whose captures (weight-independent) are copied.")
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    _set_threads()
    V4._pin(args.affinity_mask)
    configure(args.config_set)
    P.START = V2.V1_START                  # operating points only, until Stage I is written
    if args.dry_run:
        P.RUN = ROOT / "results" / "_dry_ch9_17" / f"{datetime.now():%Y%m%d-%H%M%S}"
        P.RUN.mkdir(parents=True)
        return procedure(args)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            parameters={"procedure": PROCEDURE, "network": NETWORK,
                        "base": "procedure v4 (v3 with amendments 1, 2)",
                        "hunting_tolerance": V4.TOLERANCE, "hunt_dwell_s": V4.HUNT_DWELL_S,
                        "settling_periods": V4.N_PERIODS, "go_e_v_ts": V4.GO_E_V_TS,
                        "lambda_exclude": list(V4.LAMBDA_EXCLUDE),
                        "lambda_bank": "W1-W6 + ch_9_13 grid at 12 h and 3 h, both G3 states",
                        "lambda_sts_min": LAMBDA_STS_MIN, "scarce_zone": SCARCE_ZONE,
                        "search_windows": list(V3.SEARCH), "n_2a": V2.N_2A,
                        "rounds_max": P.MAX_ROUNDS, "result_paramset": RESULT,
                        "config_set": list(args.config_set),
                        "reuse_stage1": str(args.reuse_stage1),
                        "reuse_ops": str(args.reuse_ops),
                        "workers": f"{args.local_workers} here (mask {args.affinity_mask:#x}) "
                                   f"+ {args.remote_workers} on {P.REMOTE} ({P.REMOTE_ROOT})"},
            time_series=("Stage-I verification and settling confirmation: 120 isolated "
                         "interface steps in W1-W4, W6; Stage II: 12-h simulations of W1, "
                         "W3, W4, W6 (early stop); band completed on W2; W5 report only"),
            note=args.note or "Ch 9 weight selection, procedure v5",
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
        trun.describe_output("v5_bounds.json", "kappa per G3-off point and the g_w_pcc interval")
        trun.describe_output("operating_points/", "W1-W6 on rural_700_north_res (12 h)")
        trun.describe_output("operating_points_lambda/", "the v5 lambda bank (grid, 12 h and 3 h)")
        trun.describe_output("selection/", "theta, engagement, unions, ch_9_3_select reports")
        trun.describe_output("stage2c/", "bank price on its own")
        trun.describe_output("complete/", "the band completed on W2")
        trun.describe_output("expC/", "settling confirmation in rank order")
        P.log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
