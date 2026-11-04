"""Re-measurement of the finite-bank allowance eps_op on the commitment base (2026-11-04).

The weight selection screens at ``lambda_max(B) <= 1.6`` and selects at
``1.6 / (1 + eps_op) = 1.546`` (``tuning.stage1_screen.select_ceiling``).  eps_op = 0.035
was measured in August 2026 on twelve-window banks of the August network; the screen bank
of the commitment base has six operating points (W1-W6 at their start times).  This script
measures how far the maximum over that bank underestimates the maximum over a larger bank
of the same capture convention.

Rule, declared in ``docs/tuning/2026-11-04_procedure_v4_requirements_draft.md`` before any
result:

* fresh bank F: 96 conditions of 2016 (12 months x days {8, 22} x hours {02, 08, 14, 20}),
  captured exactly as the design bank (window-start state, 12-h unit commitment, frozen
  network);
* design bank D: W1-W6 of run 0001;
* candidates C: every candidate evaluated in run 0006 and the Stage-I point;
* s(w) = lambda_max(D u F) / lambda_max(D) - 1 per candidate;
* eps_op,new = max s(w) over the candidates with lambda_max(D) >= 0.9 * 1.546.

Reported besides, not decisive: median and p95 of s, the conditions and layers that set
lambda_max(D u F), and the August-style resampling (max over random six-point banks drawn
from D u F against the max over D u F).

Pure linear algebra on captured Gram matrices after the capture: no time-domain simulation.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import pickle
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DESIGN_OPS = (ROOT / "results" / "ch_09_parameter_selection"
              / "0001_2026-10-31_135000_ch_9_6_full_procedure" / "operating_points")
RUN_0006 = (ROOT / "results" / "ch_09_parameter_selection"
            / "0006_2026-11-03_211954_ch_9_11_procedure_v3")
#: Weights only shape the capture config; the Gram matrices do not depend on them.
CAPTURE_PARAMSET = "stage1_start_uc_2026_10"
HOURS = 12.0
MONTHS = tuple(range(1, 13))
DAYS = (8, 22)
CLOCK = (2, 8, 14, 20)
NEAR = 0.9
EPS_AUGUST = 0.035
N_RESAMPLE = 2000
SEED = 20261104


def fresh_conditions() -> List[Tuple[str, datetime]]:
    return [(f"F{m:02d}{d:02d}_{h:02d}", datetime(2016, m, d, h, 0))
            for m in MONTHS for d in DAYS for h in CLOCK]


def _capture_config():
    from configs.paramsets import load_paramset
    from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
    from tuning.parameters import apply_to_config
    start = {k: float(v) for k, v in load_paramset(CAPTURE_PARAMSET)["fields"].items()}
    return apply_to_config(P.base_config(), start)


def _capture_one(job: Tuple[str, str, str]) -> Tuple[str, float, Optional[str]]:
    """Capture one condition into ``out_dir``; returns (name, seconds, error)."""
    name, iso, out_dir = job
    from tuning import stage1_screen as s1
    t0 = time.time()
    try:
        s1.load_or_build_operating_points(_capture_config(), [(name, datetime.fromisoformat(iso))],
                                          Path(out_dir), verbose=False,
                                          horizon_s=3600.0 * HOURS)
        return name, time.time() - t0, None
    except Exception as exc:  # a condition whose capture fails is reported, not dropped silently
        return name, time.time() - t0, f"{type(exc).__name__}: {exc}"


def _load(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def candidates() -> Dict[str, Dict[str, float]]:
    """Every candidate evaluated in run 0006, by key (the Stage-I point included)."""
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(RUN_0006.glob("**/candidates/cand_*.json")):
        c = json.loads(f.read_text("utf-8"))
        out.setdefault(c["key"], {k: float(v) for k, v in c["weights"].items()})
    return out


def lambda_matrix(ops, cands: Dict[str, Dict[str, float]]):
    """L[i, j] = lambda_max over the layers of operating point j for candidate i, and the
    layer that sets it."""
    from tuning import stage1_screen as s1
    names = [op.name if hasattr(op, "name") else None for op in ops]
    keys = sorted(cands)
    L = np.zeros((len(keys), len(ops)))
    by_layer = {"TS": np.zeros_like(L), "STS": np.zeros_like(L)}
    arg = np.empty((len(keys), len(ops)), dtype=object)
    for i, k in enumerate(keys):
        for j, op in enumerate(ops):
            rows = s1.screen_weights([op], cands[k])
            r = s1.worst(rows)
            L[i, j] = r.lambda_max
            arg[i, j] = f"{r.layer} {r.area}"
            names[j] = r.condition
            for layer, M in by_layer.items():
                M[i, j] = max((x.lambda_max for x in rows if x.layer == layer), default=0.0)
    return keys, names, L, arg, by_layer


def layer_shift(M: np.ndarray, n_design: int) -> dict:
    """The shift of one layer class alone (reported, not decisive)."""
    d, a = M[:, :n_design].max(axis=1), M.max(axis=1)
    s = a / d - 1.0
    return {"lambda_design_max": float(d.max()), "lambda_all_max": float(a.max()),
            "s_p50": float(np.percentile(s, 50)), "s_p95": float(np.percentile(s, 95)),
            "s_max": float(s.max())}


def summarise(keys, names, L, arg, n_design: int) -> dict:
    from tuning import stage1_screen as s1
    ceil = s1.select_ceiling()
    lam_d = L[:, :n_design].max(axis=1)
    lam_all = L.max(axis=1)
    s = lam_all / lam_d - 1.0
    near = lam_d >= NEAR * ceil
    rng = np.random.default_rng(SEED)
    resample = []
    for i in np.flatnonzero(near):
        draws = np.array([L[i, rng.choice(L.shape[1], n_design, replace=False)].max()
                          for _ in range(N_RESAMPLE)])
        resample.append(lam_all[i] / draws - 1.0)
    resample = np.concatenate(resample) if resample else np.array([])
    j_max = L.argmax(axis=1)
    setters: Dict[str, int] = {}
    for i in np.flatnonzero(near):
        tag = f"{names[j_max[i]]} ({arg[i, j_max[i]]})"
        setters[tag] = setters.get(tag, 0) + 1
    i_worst = int(np.flatnonzero(near)[np.argmax(s[near])]) if near.any() else None
    pct = (lambda a, q: float(np.percentile(a, q)) if a.size else None)
    return {
        "select_ceiling": ceil,
        "near_threshold": NEAR * ceil,
        "n_candidates": len(keys),
        "n_near": int(near.sum()),
        "n_conditions": len(names),
        "n_design": n_design,
        "eps_op_new": float(s[near].max()) if near.any() else None,
        "eps_op_august": EPS_AUGUST,
        "worst_candidate": keys[i_worst] if i_worst is not None else None,
        "worst_candidate_lambda_design": float(lam_d[i_worst]) if i_worst is not None else None,
        "worst_candidate_lambda_all": float(lam_all[i_worst]) if i_worst is not None else None,
        "s_near": {"p50": pct(s[near], 50), "p95": pct(s[near], 95),
                   "max": float(s[near].max()) if near.any() else None},
        "s_all": {"p50": pct(s, 50), "p95": pct(s, 95), "max": float(s.max())},
        "share_near_with_s_gt_august": float(np.mean(s[near] > EPS_AUGUST)) if near.any() else None,
        "setters_of_lambda_all_near": dict(sorted(setters.items(), key=lambda kv: -kv[1])),
        "resample_6_of_all_near": {"n": int(resample.size), "p50": pct(resample, 50),
                                   "p95": pct(resample, 95),
                                   "max": float(resample.max()) if resample.size else None},
    }


def _pin(mask: Optional[int]) -> None:
    if mask and sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(mask))


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import start_thesis_run
    from tuning import stage1_screen as s1
    ap = argparse.ArgumentParser(prog="ch_9_13_eps_op_remeasure")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--affinity-mask", type=lambda x: int(x, 0), default=None,
                    help="Windows CPU affinity mask, inherited by the capture workers.")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    _pin(args.affinity_mask)
    conds = fresh_conditions()
    params = {"fresh_conditions": f"{len(conds)}: 2016, months 1-12 x days {DAYS} x hours "
                                  f"{CLOCK}", "design_bank": "W1-W6 of run 0001",
              "candidates": "all of run 0006", "near_ceiling": f">= {NEAR} x select_ceiling",
              "n_resample": N_RESAMPLE, "seed": SEED, "capture_horizon_h": HOURS}
    with start_thesis_run(
            __file__, target=["ch:param:experiment:d"], paramset=CAPTURE_PARAMSET,
            parameters=params,
            time_series=(f"{len(conds)} capture instants of 2016 and the W1-W6 start states; "
                         "no time-domain simulation"),
            note=args.note or "Re-measurement of eps_op (finite-bank allowance) on the "
                              "commitment base") as run:
        fresh_dir = run.path("operating_points_fresh")
        fresh_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(n, t.isoformat(), str(fresh_dir)) for n, t in conds]
        t0 = time.time()
        with Pool(processes=max(1, args.workers)) as pool:
            log = pool.map(_capture_one, jobs, chunksize=1)
        failed = {n: e for n, _, e in log if e}
        print(f"captures: {len(conds) - len(failed)} / {len(conds)} in {time.time() - t0:.0f} s"
              f"; failed: {sorted(failed)}", flush=True)
        (run.path("captures.json")).write_text(json.dumps(
            [{"condition": n, "seconds": round(sec, 1), "error": e} for n, sec, e in log],
            indent=1), "utf-8")
        design = [_load(f) for f in sorted(DESIGN_OPS.glob("op_*.npz"))]
        fresh = [_load(f) for f in sorted(fresh_dir.glob("op_*.npz"))]
        cands = candidates()
        keys, names, L, arg, by_layer = lambda_matrix(design + fresh, cands)
        np.savez(run.path("lambda_matrix.npz"), L=L, L_ts=by_layer["TS"], L_sts=by_layer["STS"],
                 keys=np.array(keys), conditions=np.array(names), setter=arg.astype(str),
                 n_design=len(design))
        summ = summarise(keys, names, L, arg, len(design))
        summ["by_layer_class"] = {k: layer_shift(M, len(design)) for k, M in by_layer.items()}
        summ["captures_failed"] = failed
        run.path("summary.json").write_text(json.dumps(summ, indent=1), "utf-8")
        print(json.dumps(summ, indent=1), flush=True)
        run.describe_output("operating_points_fresh/", "the 96 fresh captures (pickled "
                            "OperatingPoint, cache key as the design bank)")
        run.describe_output("captures.json", "capture time and error per condition")
        run.describe_output("lambda_matrix.npz", "lambda_max per candidate and operating point "
                            "(design bank first), with the setting layer")
        run.describe_output("summary.json", "eps_op,new by the declared rule, and the reported "
                            "statistics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
