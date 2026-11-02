#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py
=============================================================
Weight selection by procedure v2 on the unit-commitment base
(``docs/tuning/2026-11-02_weight_selection_procedure_v2.md``, frozen 2026-11-02;
diagnosis and options in ``docs/tuning/2026-11-02_procedure_v2_proposal.md``).

The subordinate (STS) layer is designed first and verified once; Stage II then
searches only the supervisory (TS) weights.  The settling test does not see the
TS weights (the TS controller is frozen in it), so the verified STS design
cannot be undone by the search.

Stages, in one invocation:

1. **Stage I** -- the TS part as in v1 (gauge, lambda_TS 1, Delta v_eng 0.0075 /
   0.010 pu); three STS variants pi_q 10, 20, 30 (pi_v,DS 1, lambda_STS 1.4,
   Delta v_eng,STS 0.03 pu), each written as a parameter set.
2. **Verification** -- the settling test (``ch_9_1_ninner_isolated_sts.py``, as
   in v1) on the three variants in parallel on e5power.  The variant with the
   largest rho = dso_g_v / g_q whose measured p95 <= 25 and median residual
   <= 0.8 Mvar is the Stage-I point; if none qualifies the campaign stops.
3. **Stage II-a** -- 512 Sobol draws over the five TS weights around the
   Stage-I point; analytic screen lambda_max(B) <= 1.546 and theta_TS,min <=
   0.009 pu; the survivors simulated in W1-W4, W6 on both hosts.
4. **Stage II-b** -- the v1 rounds with the same fixed weights and screen.
5. **Selection** -- theta filters, the UNCHANGED ``ch_9_3_select.py`` with hunting
   tolerance 0 (1 only if nothing is feasible at 0), then the settling
   confirmation in rank order on e5power.
6. **Result** -- W5 for the report, then
   ``configs/paramsets/ch9_selected_uc_2026_11_v2.json``.  SELECTED_PARAMSET is
   NOT switched here (author decision).

Transport, simulation units and the selection script are those of
``ch_9_6_full_procedure`` (imported as ``P``).

Usage (from the repository root, on the master host)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py --note "..."
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py --resume <run dir>
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py --dry-run

Author: Manuel Schwenke / Claude Code (2026-11-02)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402

PROCEDURE = "docs/tuning/2026-11-02_weight_selection_procedure_v2.md"
#: The 12-h captures of run 0001 (weight-independent, so reused; key netf8c419bf).
OPS_SOURCE = (ROOT / "results" / "ch_09_parameter_selection"
              / "0001_2026-10-31_135000_ch_9_6_full_procedure" / "operating_points")
V1_START = "stage1_start_uc_2026_10"          # v1 Stage I: the TS part must equal it
RESULT = "ch9_selected_uc_2026_11_v2"
VARIANT_NAME = "stage1_v2_uc_2026_11_pq{:g}"

PI_Q_VARIANTS = (10.0, 20.0, 30.0)
PI_V_DS, TAU, G_W_GEN = 1.0, 1.0, 1.0e8
D_ENGAGE = {"ts": 0.0075, "sts": 0.03, "shunt": 0.010}       # pu
LAMBDA_TARGET = {"ts": 1.0, "sts": 1.4}
VERIFY_P95, VERIFY_RESIDUAL = 25.0, 0.8        # margins to 27 and 1 Mvar
THETA_STS = (0.02, 0.04)                       # pu, amendment 3; 0.03 by construction
THETA_TS_MAX = 0.009                           # pu, "TS taps not inert"
STS_FIELDS = ("g_q", "dso_g_v", "g_w_dso_der", "g_w_dso_oltc")
N_2A = 512
#: Paths a change in which does not touch what an e5power shard runs.
#: (This run's own parameter sets are pushed explicitly before a shard needs them.)
SHARD_NEUTRAL = ("docs/", "tests/", "experiments/ch_11_case2_multisystem/",
                 "experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py",
                 "experiments/ch_09_parameter_selection/ch_9_7_amendment3.py",
                 "experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py",
                 "configs/paramsets/stage1_v2_uc_2026_11_",
                 f"configs/paramsets/{RESULT}")


def spans_v2() -> Dict[str, float]:
    """The v1 spans with the STS weights held (g_v is never in the box)."""
    from tuning.stage2_search import DEFAULT_SPANS
    return {k: (0.0 if k in STS_FIELDS else float(v)) for k, v in DEFAULT_SPANS.items()}


def theta_ts_min(weights: Dict[str, float], ops) -> float:
    """Strongest TS tap's systematic commit threshold over all conditions (the TS
    counterpart of ``ch_9_3_amendment1.theta_sts_min``)."""
    import numpy as np

    from tuning import stage1_screen as s1
    best = float("inf")
    for op in ops:
        for lc in op.layers:
            if lc.layer != "TS":
                continue
            g_v, g_q = s1._layer_weights(lc, weights)
            dev = s1.engage_deviation(lc, g_v, g_q, lc.g_w_vector(weights))
            for i, c in enumerate(lc.classes):
                if c == "oltc" and np.isfinite(dev[i]):
                    best = min(best, float(dev[i]))
    return best


def admitted(theta_sts: float, theta_ts: float) -> bool:
    return THETA_STS[0] <= theta_sts <= THETA_STS[1] and theta_ts <= THETA_TS_MAX


def passes_screen(weights: Dict[str, float], ops) -> bool:
    from tuning import stage1_screen as s1
    return (s1.worst(s1.screen_weights(ops, weights)).lambda_max <= s1.select_ceiling()
            and theta_ts_min(weights, ops) <= THETA_TS_MAX)


def choose_variant(results: Dict[float, Dict[str, Any]]) -> Optional[float]:
    """pi_q of the variant with the largest rho (smallest pi_q) whose measured
    check keeps the margins; ``None`` when no variant does."""
    for pi_q in sorted(results):
        o = results[pi_q]
        if ("p95" in o and float(o["p95"]) <= VERIFY_P95
                and float(o["median_residual_mvar"]) <= VERIFY_RESIDUAL):
            return pi_q
    return None


def settle_argv(label: str, wf: Path, paramset: str, out: Path, workers: int) -> List[str]:
    """``P.exp_c_argv`` with the parameter set and folder stated."""
    return ["experiments/ch_09_parameter_selection/ch_9_1_ninner_isolated_sts.py",
            "--paramset", paramset, "--weights-json", P.rel(wf), "--label", label,
            "--out", P.rel(out), "--workers", str(workers),
            "--max-step-mvar", "10", "--save-trajectories", *P.EXPC_EXTRA]


# -- 1. Stage I --------------------------------------------------------------------------

def load_ops():
    """Run 0001's operating points, copied into this run (never rebuilt here)."""
    dst = P.ops_dir()
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(OPS_SOURCE.glob("op_*.npz"))
    if len(files) != 6:
        raise RuntimeError(f"{OPS_SOURCE}: {len(files)} operating points, expected W1-W6")
    for f in files:
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
    return P.load_ops()


def design_variants(ops) -> Dict[float, Tuple[Dict[str, float], Dict[str, Any]]]:
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    v1 = {k: float(v) for k, v in load_paramset(V1_START)["fields"].items()}
    out: Dict[float, Tuple[Dict[str, float], Dict[str, Any]]] = {}
    for pi_q in PI_Q_VARIANTS:
        w, rep = s1.design_stage1(
            ops, pi_q=pi_q, pi_v_ds=PI_V_DS, tau=TAU, g_w_gen=G_W_GEN,
            d_engage_ts=D_ENGAGE["ts"], d_engage_sts=D_ENGAGE["sts"],
            d_engage_shunt=D_ENGAGE["shunt"], lambda_target_ts=LAMBDA_TARGET["ts"],
            lambda_target_sts=LAMBDA_TARGET["sts"])
        w = {k: float(v) for k, v in w.items()}
        bad = [k for k, v in v1.items() if k not in STS_FIELDS
               and abs(w[k] - v) > 1e-6 * max(1.0, abs(v))]
        if bad:
            raise RuntimeError(f"Stage I-TS differs from {V1_START} in {bad}")
        rep["op_horizon_s"] = 3600.0 * P.HOURS
        out[pi_q] = (w, rep)
    return out


def describe(w: Dict[str, float], ops) -> Dict[str, float]:
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import theta_sts_min
    from tuning import stage1_screen as s1
    rows = s1.screen_weights(ops, w)
    return {"rho": w["dso_g_v"] / w["g_q"], "kappa": w["g_q"] / w["g_w_dso_der"],
            "theta_sts_min": theta_sts_min(w, ops), "theta_ts_min": theta_ts_min(w, ops),
            "lambda_max_B": s1.worst(rows).lambda_max,
            "lambda_max_B_sts": max(r.lambda_max for r in rows if r.layer == "STS")}


def write_variant(pi_q: float, w: Dict[str, float], rep: Dict[str, Any], ops) -> str:
    from configs.paramsets import PARAMSET_DIR
    from tuning import stage1_screen as s1
    from tuning.parameters import WEIGHT_FIELDS
    name = VARIANT_NAME.format(pi_q)
    path = PARAMSET_DIR / f"{name}.json"
    fields = {k: float(w[k]) for k in WEIGHT_FIELDS}
    if path.exists():
        old = json.loads(path.read_text("utf-8"))["fields"]
        if any(abs(float(old[k]) - v) > 1e-9 * max(1.0, abs(v)) for k, v in fields.items()):
            raise SystemExit(f"{path} exists with other weights; a parameter set is "
                             f"never overwritten")
        return name
    cfg = P.base_config()
    doc = {
        "name": name,
        "title": f"Procedure v2 Stage-I variant pi_q = {pi_q:g} (subordinate layer first)",
        "description": [
            f"Designed by {P.rel(Path(__file__))} under {PROCEDURE}.",
            "NOT a tuned point: one of three Stage-I variants, of which the settling "
            "verification chooses the starting point of the Stage-II search.",
            "TS part identical to stage1_start_uc_2026_10 (procedure v1 Stage I).",
        ],
        "provenance": {
            "producer": P.rel(Path(__file__)),
            "procedure": PROCEDURE,
            "date": datetime.now().date().isoformat(),
            "network": str(cfg.scenario),
            "dso_der_scale": dict(getattr(cfg, "dso_der_scale", None) or {}),
            "dso_load_p_scale": dict(getattr(cfg, "dso_load_p_scale", None) or {}),
            "unit_commitment": True,
            "operating_points": P.rel(OPS_SOURCE),
            "operating_point_horizon_s": rep.get("op_horizon_s"),
            "priorities": rep.get("priorities"),
            "tau": rep.get("tau"),
            "d_engage_pu": rep.get("d_engage"),
            "lambda_targets": rep.get("targets"),
            "kappa": rep.get("kappa"),
            "integer_floors": rep.get("floors"),
            "select_ceiling": s1.select_ceiling(),
            **describe(fields, ops),
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {P.rel(path)}")
    return name


# -- 2. verification ---------------------------------------------------------------------

def verify(variants: Dict[float, str], weights: Dict[float, Dict[str, float]],
           workers: int) -> Dict[float, Dict[str, Any]]:
    from configs.paramsets import PARAMSET_DIR
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import _criterion
    from tuning.parameters import WEIGHT_FIELDS
    vdir = P.RUN / "stage1" / "verification"
    vdir.mkdir(parents=True, exist_ok=True)
    path = vdir / "verification.json"
    stored = json.loads(path.read_text("utf-8")) if path.exists() else {}
    results = {float(k): v for k, v in stored.items()}
    todo = [pi_q for pi_q in variants if pi_q not in results]
    P.log(f"verification of pi_q {[f'{q:g}' for q in todo]} on {P.REMOTE} "
          f"({workers} workers each); stored {[f'{q:g}' for q in sorted(results)]}")

    def one(pi_q: float) -> Dict[str, Any]:
        name, label = variants[pi_q], f"v2_pq{pi_q:g}"
        wf = vdir / f"{label}_weights.json"
        wf.write_text(json.dumps({k: float(weights[pi_q][k]) for k in WEIGHT_FIELDS},
                                 indent=1), "utf-8")
        P.push(PARAMSET_DIR / f"{name}.json",
               P.REMOTE_ROOT + rf"\configs\paramsets\{name}.json")
        P.push(wf, P.remote_path(wf))
        P.run_remote(label, settle_argv(label, wf, name, vdir, workers), vdir,
                     [(vdir / label, True)])
        return _criterion(vdir, label)

    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            outs = list(ex.map(one, todo))
        for pi_q, o in zip(todo, outs):
            results[pi_q] = o
        path.write_text(json.dumps({f"{k:g}": v for k, v in sorted(results.items())},
                                   indent=1), encoding="utf-8")
    for pi_q in sorted(results):
        o = results[pi_q]
        P.log(f"  pi_q {pi_q:g}: {'PASS' if o.get('pass') else 'FAIL'} p95 {o.get('p95')} "
              f"median residual {o.get('median_residual_mvar')} {o.get('error', '')}")
    return results


# -- 3./4. Stage II ----------------------------------------------------------------------

def stage_select(dirs: List[Path], tag: str, ops) -> dict:
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import theta_sts_min
    sel = P.RUN / "selection"
    union = sel / f"union_{tag}"
    union.mkdir(parents=True, exist_ok=True)
    th_sts: Dict[str, float] = {}
    th_ts: Dict[str, float] = {}
    base_key = None
    for d in dirs:
        for f in sorted((d / "candidates").glob("cand_*.json")):
            c = json.loads(f.read_text("utf-8"))
            k = c["key"]
            if k not in th_sts:
                th_sts[k] = theta_sts_min(c["weights"], ops)
                th_ts[k] = theta_ts_min(c["weights"], ops)
            if c.get("is_baseline"):
                base_key = k
            if admitted(th_sts[k], th_ts[k]):
                shutil.copy2(f, union / f.name)
    if base_key is None or not admitted(th_sts[base_key], th_ts[base_key]):
        raise RuntimeError(f"Stage-I point missing or outside the engagement bounds "
                           f"({base_key}: theta_STS {th_sts.get(base_key)}, "
                           f"theta_TS {th_ts.get(base_key)})")
    (sel / f"theta_{tag}.json").write_text(json.dumps(
        {"theta_sts_band_pu": list(THETA_STS), "theta_ts_max_pu": THETA_TS_MAX,
         "theta_sts_min_pu": th_sts, "theta_ts_min_pu": th_ts,
         "removed": sorted(k for k in th_sts if not admitted(th_sts[k], th_ts[k]))},
        indent=1), "utf-8")
    rep = None
    for tol in (0, 1):
        out = sel / f"selection_{tag}_h{tol}.json"
        P.run(["experiments/ch_09_parameter_selection/ch_9_3_select.py",
               "--candidates", P.rel(union), "--hours", f"{P.HOURS:g}",
               "--shunt-ops-per-day", f"{P.BANK_BUDGET:g}", "--max-hunting", str(tol),
               "--out", P.rel(out)], sel / f"select_{tag}_h{tol}.log")
        rep = json.loads(out.read_text("utf-8")) if out.exists() else None
        if rep and rep.get("ranked"):
            rep["_max_hunting"] = tol
            break
        rep = None
    if rep is None:
        raise RuntimeError(f"selection {tag}: no feasible candidate, with one hunting "
                           f"reversal tolerated either")
    rep["_theta_sts"], rep["_theta_ts"] = th_sts, th_ts
    n_adm = sum(1 for k in th_sts if admitted(th_sts[k], th_ts[k]))
    P.log(f"selection {tag}: {n_adm} / {len(th_sts)} inside the engagement bounds, "
          f"{len(rep['ranked'])} feasible with hunting tolerance {rep['_max_hunting']}, "
          f"ranked first {[r['key'] for r in rep['ranked'][:4]]}")
    return rep


def stage2a(centre: Dict[str, float], ops, w_local: int, w_remote: int,
            split_local: Optional[float] = None) -> dict:
    from tuning import stage1_screen as s1
    from tuning.stage2_search import Box, sobol_candidates
    out = P.RUN / "stage2a"
    spans = spans_v2()
    draws = sobol_candidates(Box(centre=centre, spans=spans), N_2A, seed=P.SEED)
    lam = [s1.worst(s1.screen_weights(ops, w)).lambda_max for w in draws]
    tts = [theta_ts_min(w, ops) for w in draws]
    keep = [w for w, l, t in zip(draws, lam, tts)
            if l <= s1.select_ceiling() and t <= THETA_TS_MAX]
    out.mkdir(parents=True, exist_ok=True)
    (out / "draws.json").write_text(json.dumps(
        {"seed": P.SEED, "n_draws": N_2A, "centre": centre, "spans": spans,
         "select_ceiling": s1.select_ceiling(), "theta_ts_max_pu": THETA_TS_MAX,
         "lambda_max_all": lam, "theta_ts_min_all": tts, "n_survivors": len(keep)},
        indent=1), "utf-8")
    P.log(f"stage II-a: {len(keep)} / {N_2A} draws pass the screen "
          f"(lambda <= {s1.select_ceiling():.4f}: "
          f"{sum(1 for l in lam if l <= s1.select_ceiling())}, theta_TS <= "
          f"{THETA_TS_MAX:g}: {sum(1 for t in tts if t <= THETA_TS_MAX)})")
    P.simulate(out, keep, w_local, w_remote, split_local)
    return stage_select([out], "2a", ops)


def stage2b(rep: dict, ops, w_local: int, w_remote: int,
            split_local: Optional[float] = None) -> Tuple[dict, List[Path]]:
    """``P.stage2b`` with the v2 spans, screen and selection."""
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    from tuning.stage2_search import Box, sobol_candidates
    s2a = P.RUN / "stage2a"
    dirs = [s2a]
    inc = rep["ranked"][0]
    evaluated = {p.stem[5:] for p in (s2a / "candidates").glob("cand_*.json")}
    base = next(json.loads(p.read_text("utf-8"))
                for p in (s2a / "candidates").glob("cand_*.json")
                if json.loads(p.read_text("utf-8")).get("is_baseline"))
    f, r = P.F0, 0
    while r < P.MAX_ROUNDS:
        r += 1
        rd = P.RUN / "stage2b" / f"round{r}"
        centre = {k: float(inc["weights"][k]) for k in WEIGHT_FIELDS}
        spans = {k: f * v for k, v in spans_v2().items()}
        draws = sobol_candidates(Box(centre=centre, spans=spans), P.N_LOCAL, seed=P.SEED + r)
        keep = [w for w in draws
                if passes_screen(w, ops) and weight_fingerprint(w) not in evaluated]
        (rd / "windows").mkdir(parents=True, exist_ok=True)
        (rd / "draws.json").write_text(json.dumps(
            {"round": r, "seed": P.SEED + r, "f": f, "centre": centre, "spans": spans,
             "n_draws": P.N_LOCAL, "n_survivors": len(keep)}, indent=1), "utf-8")
        P.log(f"round {r}: f {f:g}, centre {weight_fingerprint(centre)}, "
              f"{len(keep)} / {P.N_LOCAL} draws pass the screen")
        # The Stage-I point's windows are the stored ones (same key, same flags).
        for f_ in (s2a / "windows").glob(f"{base['key']}__*.json"):
            shutil.copy2(f_, rd / "windows" / f_.name)
        P.simulate(rd, keep, w_local, w_remote, split_local)
        dirs.append(rd)
        evaluated |= {p.stem[5:] for p in (rd / "candidates").glob("cand_*.json")}
        rep = stage_select(dirs, f"2b_round{r}", ops)
        new = rep["ranked"][0]
        if new["key"] != inc["key"]:
            P.log(f"round {r}: new best {new['key']} (e_v_ts "
                  f"{new['aggregate']['e_v_ts']:.5f}, e_v_ds "
                  f"{new['aggregate']['e_v_ds']:.5f}); recentre, f stays {f:g}")
            inc = new
        else:
            f /= 2
            P.log(f"round {r}: best unchanged ({inc['key']}); f -> {f:g}")
            if f < P.F_MIN:
                break
    P.log(f"stage II-b ends after {r} round(s); best {inc['key']}")
    return rep, dirs


# -- 5. settling confirmation ------------------------------------------------------------

def confirm(rep: dict, workers: int) -> Tuple[Optional[dict], dict]:
    """The v1 settling loop, one check at a time on e5power."""
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import _criterion
    from tuning.parameters import WEIGHT_FIELDS
    from tuning.stage2_search import rank_lexicographic
    expc = P.RUN / "expC"
    expc.mkdir(parents=True, exist_ok=True)
    path = expc / "criterion_results.json"
    results = json.loads(path.read_text("utf-8")) if path.exists() else {}
    feasible = {r["key"]: r for r in rep["ranked"]}
    while True:
        remaining = [r for k, r in feasible.items()
                     if not (k in results and not results[k]["pass"])]
        if not remaining:
            return None, results
        order = rank_lexicographic(remaining, indifference=0.02)
        if order[0]["key"] in results:
            return order[0], results
        r = order[0]
        label = f"cand_{r['key']}"
        wf = expc / f"{label}_weights.json"
        wf.write_text(json.dumps({k: float(r["weights"][k]) for k in WEIGHT_FIELDS},
                                 indent=1), "utf-8")
        P.log(f"settling confirmation of {r['key']} on {P.REMOTE} ({workers} workers)")
        P.push(wf, P.remote_path(wf))
        P.run_remote(label, settle_argv(label, wf, P.START, expc, workers), expc,
                     [(expc / label, True)])
        o = _criterion(expc, label)
        results[r["key"]] = o
        P.log(f"  {r['key']}: {'PASS' if o['pass'] else 'FAIL'} p95 {o.get('p95')} "
              f"median residual {o.get('median_residual_mvar')} {o.get('error', '')}")
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")


# -- 6. result ---------------------------------------------------------------------------

def write_result(win: dict, rep: dict, c_res: dict, w5: dict, chosen: float,
                 ver: Dict[float, Dict[str, Any]], ops) -> Path:
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    path = PARAMSET_DIR / f"{RESULT}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a parameter set is never overwritten")
    fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
    doc = {
        "name": RESULT,
        "title": "Controller weights selected by procedure v2 (subordinate layer first) "
                 "on the unit-commitment base",
        "description": [
            f"Selected by {PROCEDURE} (frozen 2026-11-02): STS weights designed in "
            f"Stage I and verified by the settling test (variant pi_q = {chosen:g}); "
            "Stage II-a/II-b over the five TS weights; filter with theta_STS,min in "
            f"[{THETA_STS[0]:g}, {THETA_STS[1]:g}] pu and theta_TS,min <= "
            f"{THETA_TS_MAX:g} pu; ranking on the per-bus TS voltage deviation with a "
            "2 % band on the STS deviation; settling confirmation in rank order.",
        ],
        "provenance": {
            "producer": P.rel(Path(__file__)),
            "run_dir": P.rel(P.RUN),
            "procedure": PROCEDURE,
            "starting_paramset": P.START,
            "stage1_variant_pi_q": chosen,
            "stage1_verification": {f"{k:g}": v for k, v in sorted(ver.items())},
            "dso_voltage_band_pu": list(P.BAND),
            "simulated_windows": P.WINDOWS,
            "w5_report": w5,
            "selected_key": win["key"],
            "max_hunting_applied": rep.get("_max_hunting"),
            "n_feasible": len(rep.get("ranked", [])),
            "theta_sts_min_pu": rep["_theta_sts"].get(win["key"]),
            "theta_ts_min_pu": rep["_theta_ts"].get(win["key"]),
            "failed_settling": sorted(k for k, v in c_res.items() if not v["pass"]),
            "lambda_max_B": win.get("lambda_max_B"),
            "aggregate_ranked_windows_per_bus": win["aggregate"],
            "baseline_aggregate": rep.get("baseline", {}).get("aggregate"),
            "settling": c_res.get(win["key"]),
            "weight_fingerprint": weight_fingerprint(fields),
            **{f"design_{k}": v for k, v in describe(fields, ops).items()},
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {path}")
    return path


# -- main -------------------------------------------------------------------------------

def check_remote_code() -> None:
    """The shards run e5power's checkout: everything they execute must equal ours."""
    rh = P.remote_head()
    changed = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "diff", "--name-only", rh],
        text=True).split()
    changed += subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "ls-files", "--others",
         "--exclude-standard"], text=True).split()
    bad = [f for f in changed if not f.startswith(SHARD_NEUTRAL)]
    if bad:
        raise RuntimeError(f"code differs from {P.REMOTE} ({rh[:7]}) in {bad[:8]}")
    P.log(f"{P.REMOTE} at {rh[:7]}; differences here are shard-neutral "
          f"({len(changed)} file(s))")


def procedure(args) -> int:
    from tuning import stage1_screen as s1
    P.check_config()
    if not args.dry_run:
        check_remote_code()
    P.log("== 1. Stage I: TS part as v1, three STS variants")
    ops = load_ops()
    designs = design_variants(ops)
    for pi_q, (w, _) in designs.items():
        d = describe(w, ops)
        P.log(f"  pi_q {pi_q:g}: rho {d['rho']:.4g}, kappa {d['kappa']:.4g}, "
              f"theta_STS {d['theta_sts_min']:.4f}, theta_TS {d['theta_ts_min']:.4f}, "
              f"lambda_max(B) {d['lambda_max_B']:.4f} (STS {d['lambda_max_B_sts']:.4f})")
    if args.dry_run:
        from tuning.stage2_search import Box, sobol_candidates
        w0 = designs[PI_Q_VARIANTS[1]][0]
        draws = sobol_candidates(Box(centre=w0, spans=spans_v2()), N_2A, seed=P.SEED)
        lam = [s1.worst(s1.screen_weights(ops, w)).lambda_max <= s1.select_ceiling()
               for w in draws]
        tts = [theta_ts_min(w, ops) <= THETA_TS_MAX for w in draws]
        P.log(f"dry run: II-a around pi_q {PI_Q_VARIANTS[1]:g}: {sum(lam)} / {N_2A} pass "
              f"lambda, {sum(tts)} pass theta_TS, {sum(a and b for a, b in zip(lam, tts))} "
              f"both; nothing simulated, no parameter set written")
        return 0
    variants = {pi_q: write_variant(pi_q, w, rep, ops) for pi_q, (w, rep) in designs.items()}
    P.log("== 2. settling verification of the STS variants")
    ver = verify(variants, {q: w for q, (w, _) in designs.items()}, args.verify_workers)
    chosen = choose_variant(ver)
    if chosen is None:
        P.log(f"no STS variant has p95 <= {VERIFY_P95:g} and median residual <= "
              f"{VERIFY_RESIDUAL:g} Mvar; the campaign stops")
        (P.RUN / "FAILED.txt").write_text("no STS variant passes the verification\n", "utf-8")
        return 2
    P.START = variants[chosen]
    P.log(f"Stage-I point: {P.START} (pi_q {chosen:g})")
    P.push_inputs()
    # The split weighs each host by its throughput, not only by its workers: a run
    # keeps the split of a stage that has begun (P.kept_split), so a new factor acts
    # from the next stage or round on (author 2026-11-02: rebalance from round 2).
    split_local = args.local_workers * args.local_speed
    P.log("== 3. Stage II-a")
    rep = stage2a(designs[chosen][0], ops, args.local_workers, args.remote_workers,
                  split_local)
    P.log("== 4. Stage II-b")
    rep, _ = stage2b(rep, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 5. selection over all evaluated candidates, settling confirmation")
    win, c_res = confirm(rep, args.confirm_workers)
    if win is None:
        P.log("no feasible candidate passes the settling confirmation")
        (P.RUN / "FAILED.txt").write_text("no candidate passes\n", "utf-8")
        return 2
    P.log(f"SELECTED {win['key']} (theta_TS {rep['_theta_ts'][win['key']]:.4f} pu)")
    P.log("== 6. W5 for the report, result")
    w5 = P.report_w5(win)
    P.log(f"  W5: e_v_ts {w5.get('e_v_ts')}, e_v_ds {w5.get('e_v_ds')}")
    write_result(win, rep, c_res, w5, chosen, ver, ops)
    (P.RUN / "DONE.txt").write_text(f"selected {win['key']}\n", "utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_8_procedure_v2")
    p.add_argument("--local-workers", type=int, default=8,
                   help="simulation workers here (author: 8 on e5wimi)")
    p.add_argument("--remote-workers", type=int, default=36,
                   help=f"simulation workers on {P.REMOTE}")
    p.add_argument("--verify-workers", type=int, default=12,
                   help=f"workers per verification check (three at a time on {P.REMOTE})")
    p.add_argument("--confirm-workers", type=int, default=36,
                   help=f"workers for a settling confirmation on {P.REMOTE}")
    p.add_argument("--local-speed", type=float, default=1.0,
                   help=f"throughput of a worker here relative to one on {P.REMOTE}; "
                        "weighs the host split (results do not depend on it)")
    p.add_argument("--dry-run", action="store_true",
                   help="Stage I and the II-a screen only, in results/_dry_ch9_8; "
                        "nothing simulated, no parameter set written")
    p.add_argument("--resume", type=Path, default=None)
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")

    if args.dry_run:
        P.RUN = ROOT / "results" / "_dry_ch9_8" / f"{datetime.now():%Y%m%d-%H%M%S}"
        P.RUN.mkdir(parents=True)
        P.START = V1_START
        return procedure(args)
    P.START = V1_START                 # until the verification has chosen a variant
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            parameters={"procedure": PROCEDURE,
                        "stage1_sts_variants_pi_q": list(PI_Q_VARIANTS),
                        "lambda_target_sts": LAMBDA_TARGET["sts"],
                        "d_engage_pu": D_ENGAGE,
                        "verification_margins": {"p95": VERIFY_P95,
                                                 "median_residual_mvar": VERIFY_RESIDUAL},
                        "theta_sts_band_pu": list(THETA_STS),
                        "theta_ts_max_pu": THETA_TS_MAX,
                        "n_2a": N_2A, "result_paramset": RESULT,
                        "workers": f"{args.local_workers} here + {args.remote_workers} on "
                                   f"{P.REMOTE}; verification 3 x {args.verify_workers}, "
                                   f"confirmation {args.confirm_workers} on {P.REMOTE}"},
            time_series=("Stage I-STS verification and settling confirmation: 120 "
                         "isolated interface steps in W1-W4, W6; Stage II: 12-h "
                         "simulations of W1-W4, W6; W5 report only"),
            note=args.note or "Ch 9 weight selection, procedure v2 (subordinate layer first)",
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
        trun.describe_output("stage1/", "STS variant verification (settling test)")
        trun.describe_output("stage2a/", "II-a draws, screen, simulated candidates")
        trun.describe_output("stage2b/", "II-b rounds")
        trun.describe_output("selection/", "theta per candidate, admitted unions, "
                             "unchanged ch_9_3_select.py reports (hunting 0, else 1)")
        trun.describe_output("expC/", "settling confirmation in rank order")
        P.log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
