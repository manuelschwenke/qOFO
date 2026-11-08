#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py
==============================================================
Weight selection by procedure v3 on the unit-commitment base
(``docs/tuning/2026-11-03_weight_selection_procedure_v3.md``; author decisions of
2026-11-03 in ``docs/daily_log/11_2026/2026-11-01b_ch9_settling_failure_diagnosis.md``).

v3 is v2 (``ch_9_8_procedure_v2``, imported as ``V2``) with these changes, and
otherwise the same code path:

* **Stage I.**  One STS variant, pi_q 10 (v2's verified choice), with the coupler
  engagement target Delta v_eng,STS = 0.01 pu instead of 0.03; bank engagement
  0.005 pu instead of 0.010 (price about halved); g_w_gen 5e6 instead of 1e8, so
  the II-a box spans 5e5..5e7.  The settling verification is that of v2.
* **Search windows** W1, W3, W4, W6.  A candidate's windows stop early at its
  2nd unexplained tap reversal or its 1st bank reversal (hunting tolerance 1,
  bank 0); the Stage-I point is never stopped.
* **Filter** v2's limits at hunting tolerance 1, theta_TS,min <= 0.009 pu, and two
  measured engagement filters: STS couplers move in every search window; banks
  commit within 2 h after an event in at least two search windows.  If no
  candidate passes both, the bank filter is dropped, then the coupler filter
  (each level is logged; the Stage-I point is the reference and never ranked
  against a filter it fails).
* **Completion.** The 2 % band of the final search ranking is completed on W2
  and ranked on all five windows; settling confirmation and W5 as in v2.
* **Go criteria** for the follow-ups (author 2026-11-03), written to GO.json or
  NOGO.json: e_v_ts <= 1.10 x 0.00827; couplers in every search window; banks
  after events in >= 2 windows; all limits met.
* **Amendment 2** (author 2026-11-03, after run 0005's II-a; run 0006):
  (a) a reversal formed by a device's first two moves in a window, the first
  before 15 min, is reported and not counted (filter and early stop);
  (b) Stage II-c tunes the bank price on its own: the II-b leader's other weights
  fixed, g_w_tso_shunt on N_2C log-spaced values from the II-a box edge up to the
  leader's, no early stop; the lowest feasible price within the 2 % e_v_ts band of
  the leader joins the completion band, where the engagement levels decide.
  A level whose candidates all fail the settling test passes to the next level.
  ``--reuse-stage1 <run>`` takes the Stage-I verification of run 0005 (unchanged).

Usage (from the repository root, on e5wimi)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py --note "..."
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py --resume <run dir>
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py --dry-run

Author: Manuel Schwenke / Claude Code (2026-11-03)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V2  # noqa: E402

PROCEDURE = "docs/tuning/2026-11-03_weight_selection_procedure_v3.md"
RESULT = "ch9_selected_uc_2026_11_v3"
VARIANT_NAME = "stage1_v3_uc_2026_11_pq{:g}"
PI_Q = 10.0
G_W_GEN = 5.0e6
D_ENGAGE = {"ts": 0.0075, "sts": 0.01, "shunt": 0.005}       # pu
THETA_STS = (0.005, 0.015)                     # pu; 0.01 by construction, kept as a check
SEARCH = ("W1", "W3", "W4", "W6")
ALL = "W1,W2,W3,W4,W6"
TOLERANCE = 1                                  # unexplained tap reversals tolerated
AFTER_EVENT_S = 7200.0                         # a bank commit "after an event"
BANK_WINDOWS = 2
V2_E_V_TS, GO_FACTOR = 0.00827, 1.10           # go criterion on e_v_ts
STAGE = {"now": "search"}
#: Settling budget in supervisory periods (author 2026-11-03: 4 instead of 3), so
#: acceptance p95 <= 4 N_inner = 36; the verification keeps v2's 2-iteration margin.
N_PERIODS = 4
#: Coupler engagement ladder (author 2026-11-03): the first target whose Stage-I point
#: passes the verification is used.
THETA_LADDER = (0.01, 0.015, 0.02)
#: Amendment 2 (a): start-up exemption of the reversal count [s] (the first TSO
#: dispatch of a window is at 3 min; run 0005 II-a, W6, device 1[0]).
STARTUP_EXEMPT_S = 900.0
#: Amendment 2 (b): bank prices on the II-c grid, the leader's own included.
N_2C = 8
STAGE2C: Dict[str, Any] = {}

# v2's design and verification code, with the v3 constants.
V2.PROCEDURE, V2.RESULT, V2.VARIANT_NAME = PROCEDURE, RESULT, VARIANT_NAME
V2.PI_Q_VARIANTS, V2.D_ENGAGE, V2.THETA_STS = (PI_Q,), D_ENGAGE, THETA_STS
P.WINDOWS = ALL

from experiments.ch_09_parameter_selection import ch_9_3_amendment1 as _A1  # noqa: E402

_criterion_v1 = _A1._criterion


def criterion(expc: Path, label: str) -> Dict[str, Any]:
    """Run 0001's criterion with the budget N_PERIODS x N_inner (v2: 3 x 9 = 27)."""
    o = _criterion_v1(expc, label)
    if "p95" not in o:
        return o
    import pandas as pd
    lim = N_PERIODS * _A1.N_INNER
    rd = expc / label / (expc / label / "_latest.txt").read_text("utf-8").strip()
    n = pd.read_csv(rd / "steps.csv").n_settle_final_1mvar
    o["n_over"] = int((n > lim).sum())
    o["limit_p95"] = lim
    o["pass"] = bool(o["p95"] <= lim and o["median_residual_mvar"] <= 1.0)
    return o


_A1._criterion = criterion                     # V2.verify / V2.confirm import it at call time
V2.VERIFY_P95 = N_PERIODS * _A1.N_INNER - 2.0  # 34: v2's margin of 2 iterations

_screen_argv = P.screen_argv


def screen_argv(out: Path, cj: Path, workers: int, windows: Optional[str] = None,
                skip_baseline: bool = False) -> List[str]:
    """Search units: the four search windows with the early stop.  Completion:
    all five windows (only W2 is missing).  Bank stage (II-c): the search windows
    without the early stop.  An explicit window list (W5 report) is passed
    through unchanged.  Every unit counts reversals with the start-up exemption
    of amendment 2."""
    su = ["--startup-exempt-s", f"{STARTUP_EXEMPT_S:g}"]
    if windows is not None:
        return _screen_argv(out, cj, workers, windows=windows, skip_baseline=skip_baseline) + su
    if STAGE["now"] == "complete":
        return _screen_argv(out, cj, workers, windows=ALL, skip_baseline=skip_baseline) + su
    a = _screen_argv(out, cj, workers, windows=",".join(SEARCH),
                     skip_baseline=skip_baseline) + su
    if STAGE["now"] == "bank":
        return a
    return a + ["--abort-tap-reversals", str(TOLERANCE + 1), "--abort-bank-reversals", "1"]


P.screen_argv = screen_argv


# -- Stage I --------------------------------------------------------------------------------

V2_STAGE1 = "stage1_v2_uc_2026_11_pq10"


def variant_name(theta: float) -> str:
    """The first rung keeps the name it was launched with (run 0005)."""
    base = VARIANT_NAME.format(PI_Q)
    return base if abs(theta - THETA_LADDER[0]) < 1e-12 else f"{base}_th{round(theta * 1000):03d}"


def verify_label(theta: float) -> str:
    """The first rung's label is the one v2's verify gave it (its shard is running)."""
    return "v2_pq10" if abs(theta - THETA_LADDER[0]) < 1e-12 else f"v3_th{round(theta * 1000):03d}"


def verify(theta: float, name: str, w: Dict[str, float], workers: int) -> Dict[str, Any]:
    """v2's verification of one Stage-I point, keyed by the engagement target."""
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS
    vdir = P.RUN / "stage1" / "verification"
    vdir.mkdir(parents=True, exist_ok=True)
    path = vdir / "verification_v3.json"
    stored = json.loads(path.read_text("utf-8")) if path.exists() else {}
    if f"{theta:g}" in stored:
        return stored[f"{theta:g}"]
    label = verify_label(theta)
    wf = vdir / f"{label}_weights.json"
    wf.write_text(json.dumps({k: float(w[k]) for k in WEIGHT_FIELDS}, indent=1), "utf-8")
    P.push(PARAMSET_DIR / f"{name}.json", P.REMOTE_ROOT + rf"\configs\paramsets\{name}.json")
    P.push(wf, P.remote_path(wf))
    P.log(f"verification of theta_STS {theta:g} ({name}) on {P.REMOTE} ({workers} workers)")
    P.run_remote(label, V2.settle_argv(label, wf, name, vdir, workers), vdir,
                 [(vdir / label, True)])
    o = criterion(vdir, label)
    o["label"], o["paramset"] = label, name
    stored[f"{theta:g}"] = o
    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    ok = (o.get("p95", 1e9) <= V2.VERIFY_P95
          and o.get("median_residual_mvar", 1e9) <= V2.VERIFY_RESIDUAL)
    P.log(f"  theta_STS {theta:g}: {'PASS' if ok else 'FAIL'} p95 {o.get('p95')} (margin "
          f"{V2.VERIFY_P95:g}), median residual {o.get('median_residual_mvar')} "
          f"{o.get('error', '')}")
    o["verified"] = ok
    stored[f"{theta:g}"] = o
    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    return o


def design(ops, theta: float = THETA_LADDER[0]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The STS weights and the bank price from the v3 engagement targets; g_v,
    g_w_der, g_w_pcc and g_w_tso_oltc from v2's Stage I; g_w_gen 5e6.

    Re-running the TS contraction design with the cheaper bank raises g_w_der and
    g_w_pcc about fivefold (0.09 -> 0.49), into the region where v2 found nothing
    feasible (0 of 40 candidates with g_w_der >= 0.3), so the TS continuous
    prices stay v2's; the screen bound lambda_max(B) <= 1.546 still applies."""
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    wd, rep = s1.design_stage1(
        ops, pi_q=PI_Q, pi_v_ds=V2.PI_V_DS, tau=V2.TAU, g_w_gen=V2.G_W_GEN,
        d_engage_ts=D_ENGAGE["ts"], d_engage_sts=theta,
        d_engage_shunt=D_ENGAGE["shunt"], lambda_target_ts=V2.LAMBDA_TARGET["ts"],
        lambda_target_sts=V2.LAMBDA_TARGET["sts"])
    w = {k: float(v) for k, v in load_paramset(V2_STAGE1)["fields"].items()}
    for k in (*V2.STS_FIELDS, "g_w_tso_shunt"):
        w[k] = float(wd[k])
    w["g_w_gen"] = G_W_GEN
    if not V2.passes_screen(w, ops):
        raise RuntimeError(f"the v3 Stage-I point fails the analytic screen: {w}")
    rep["op_horizon_s"] = 3600.0 * P.HOURS
    rep["ts_continuous_from"] = V2_STAGE1
    return w, rep


def write_variant(w: Dict[str, float], rep: Dict[str, Any], ops,
                  theta: float = THETA_LADDER[0]) -> str:
    from configs.paramsets import PARAMSET_DIR
    from tuning import stage1_screen as s1
    from tuning.parameters import WEIGHT_FIELDS
    name = variant_name(theta)
    path = PARAMSET_DIR / f"{name}.json"
    fields = {k: float(w[k]) for k in WEIGHT_FIELDS}
    if path.exists():
        old = json.loads(path.read_text("utf-8"))["fields"]
        if any(abs(float(old[k]) - v) > 1e-9 * max(1.0, abs(v)) for k, v in fields.items()):
            raise SystemExit(f"{path} exists with other weights; never overwritten")
        return name
    doc = {
        "name": name,
        "title": f"Procedure v3 Stage-I point (pi_q {PI_Q:g}, theta_STS {theta:g} pu)",
        "description": [f"Designed by {P.rel(Path(__file__))} under {PROCEDURE}.",
                        "NOT a tuned point: the starting point of the v3 Stage-II search."],
        "provenance": {"producer": P.rel(Path(__file__)), "procedure": PROCEDURE,
                       "date": datetime.now().date().isoformat(),
                       "operating_points": P.rel(V2.OPS_SOURCE),
                       "operating_point_horizon_s": rep.get("op_horizon_s"),
                       "d_engage_pu": rep.get("d_engage"), "g_w_gen": G_W_GEN,
                       "lambda_targets": rep.get("targets"),
                       "select_ceiling": s1.select_ceiling(),
                       **V2.describe(fields, ops)},
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {P.rel(path)}")
    return name


# -- selection ------------------------------------------------------------------------------

def engagement(c: Dict[str, Any]) -> Dict[str, Any]:
    """Measured engagement over the search windows a record holds."""
    pw = c["per_window"]
    ws = [w for w in SEARCH if w in pw]
    sts = {w: sum(1 for m in pw[w].get("moves", []) if "DSO" in str(m[1])) for w in ws}
    after = {w: any(0.0 <= float(m[0]) - float(e) <= AFTER_EVENT_S
                    for m in pw[w].get("shunt_moves", []) for e in pw[w].get("events_s", []))
             for w in ws}
    return {"sts_moves": sts, "sts_all": bool(ws) and all(v > 0 for v in sts.values()),
            "bank_after_windows": sum(after.values()),
            "bank": sum(after.values()) >= BANK_WINDOWS}


LEVELS = (("sts+bank", lambda e: e["sts_all"] and e["bank"]),
          ("sts", lambda e: e["sts_all"]),
          ("none", lambda e: True))


def stage_select(dirs: List[Path], tag: str, ops, levels=LEVELS) -> dict:
    """theta bounds, engagement level (strictest of ``levels`` that leaves a ranked
    candidate), then the unchanged ch_9_3_select.py at hunting tolerance 1."""
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import theta_sts_min
    sel = P.RUN / "selection"
    sel.mkdir(parents=True, exist_ok=True)
    recs: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for d in dirs:
        for f in sorted((d / "candidates").glob("cand_*.json")):
            c = json.loads(f.read_text("utf-8"))
            recs.setdefault(c["key"], (f, c))
    th_sts = {k: theta_sts_min(c["weights"], ops) for k, (_, c) in recs.items()}
    th_ts = {k: V2.theta_ts_min(c["weights"], ops) for k, (_, c) in recs.items()}
    eng = {k: engagement(c) for k, (_, c) in recs.items()}
    base_key = next((k for k, (_, c) in recs.items() if c.get("is_baseline")), None)
    if base_key is None:
        raise RuntimeError(f"selection {tag}: no Stage-I point record")
    adm = {k for k in recs if THETA_STS[0] <= th_sts[k] <= THETA_STS[1]
           and th_ts[k] <= V2.THETA_TS_MAX}
    (sel / f"engagement_{tag}.json").write_text(json.dumps(
        {"theta_sts_band_pu": list(THETA_STS), "theta_ts_max_pu": V2.THETA_TS_MAX,
         "theta_sts_min_pu": th_sts, "theta_ts_min_pu": th_ts, "engagement": eng,
         "admitted": sorted(adm)}, indent=1), "utf-8")
    for level, ok in levels:
        union = sel / f"union_{tag}_{level}"
        if union.exists():
            shutil.rmtree(union)
        union.mkdir(parents=True)
        keep = {k for k in adm if ok(eng[k])} | {base_key}
        for k in keep:
            shutil.copy2(recs[k][0], union / recs[k][0].name)
        out = sel / f"selection_{tag}_{level}.json"
        P.run(["experiments/ch_09_parameter_selection/ch_9_3_select.py",
               "--candidates", P.rel(union), "--hours", f"{P.HOURS:g}",
               "--shunt-ops-per-day", f"{P.BANK_BUDGET:g}",
               "--max-hunting", str(TOLERANCE), "--out", P.rel(out)],
              sel / f"select_{tag}_{level}.log")
        rep = json.loads(out.read_text("utf-8")) if out.exists() else None
        if rep and rep.get("ranked"):
            if base_key not in adm or not ok(eng[base_key]):
                rep["ranked"] = [r for r in rep["ranked"] if r["key"] != base_key]
            if rep["ranked"]:
                rep["_level"], rep["_max_hunting"] = level, TOLERANCE
                rep["_theta_sts"], rep["_theta_ts"], rep["_engagement"] = th_sts, th_ts, eng
                P.log(f"selection {tag}: {len(adm)} / {len(recs)} inside the theta bounds; "
                      f"engagement level '{level}' ({len(keep) - 1} candidates); "
                      f"{len(rep['ranked'])} feasible at hunting tolerance {TOLERANCE}; "
                      f"ranked first {[r['key'] for r in rep['ranked'][:4]]}")
                return rep
        P.log(f"selection {tag}: nothing ranked at engagement level '{level}'")
    raise RuntimeError(f"selection {tag}: no feasible candidate at any engagement level")


V2.stage_select = stage_select


def complete_band(rep: dict, dirs: List[Path], ops, w_local: int, w_remote: int,
                  split_local: float, extra: Tuple[dict, ...] = ()) -> dict:
    """The final search band (and the Stage-I point) on W2; ranked on all windows.
    ``extra``: the II-c pick (amendment 2), added to the band."""
    best = min(r["aggregate"]["e_v_ts"] for r in rep["ranked"])
    band = [r for r in rep["ranked"] if r["aggregate"]["e_v_ts"] <= 1.02 * best]
    band += [r for r in extra if r["key"] not in {b["key"] for b in band}]
    keys = {r["key"] for r in band} | {rep["baseline"]["key"]}
    out = P.RUN / "complete"
    (out / "windows").mkdir(parents=True, exist_ok=True)
    for d in dirs:
        for f in (d / "windows").glob("*__W*.json"):
            if f.name.split("__")[0] in keys and not (out / "windows" / f.name).exists():
                shutil.copy2(f, out / "windows" / f.name)
    P.log(f"completion: {len(band)} band candidate(s) and the Stage-I point on W2")
    STAGE["now"] = "complete"
    try:
        P.simulate(out, [r["weights"] for r in band if r["key"] != rep["baseline"]["key"]],
                   w_local, w_remote, split_local)
    finally:
        STAGE["now"] = "search"
    return stage_select([out], "complete", ops)


def bank_grid(lo: float, hi: float) -> List[float]:
    """N_2C log-spaced bank prices from ``lo`` to ``hi``, without ``hi`` itself
    (the leader's own price, already simulated); empty if the leader sits at the
    lower edge."""
    import numpy as np
    return [float(x) for x in np.geomspace(lo, hi, N_2C)[:-1]] if hi > 1.05 * lo else []


def pick_lowest_price(ranked: List[dict], e_ref: float) -> dict:
    """The lowest bank price among the feasible records within 2 % of ``e_ref``."""
    ok = [r for r in ranked if r["aggregate"]["e_v_ts"] <= 1.02 * e_ref]
    return min(ok, key=lambda r: float(r["weights"]["g_w_tso_shunt"]))


def stage2c(rep: dict, dirs: List[Path], ops, w_local: int, w_remote: int,
            split_local: float) -> Tuple[dict, Path]:
    """Amendment 2 (b): the bank price on its own.

    The II-b leader's other weights are kept; g_w_tso_shunt runs on N_2C
    log-spaced values from the lower edge of the II-a box up to the leader's own,
    on the search windows without the early stop, so the response to every event
    is recorded.  The pick is the lowest price that is feasible (theta bounds,
    ch_9_3_select at hunting tolerance 1, no bank reversal) with e_v_ts within
    the 2 % band of the leader on the same records.  It joins the completion
    band, where the engagement levels decide."""
    from configs.paramsets import load_paramset
    from tuning.parameters import WEIGHT_FIELDS
    lead = rep["ranked"][0]
    base = rep["baseline"]
    rc = P.RUN / "stage2c"
    (rc / "windows").mkdir(parents=True, exist_ok=True)
    lo = (float(load_paramset(P.START)["fields"]["g_w_tso_shunt"])
          * 10.0 ** -V2.spans_v2()["g_w_tso_shunt"])
    hi = float(lead["weights"]["g_w_tso_shunt"])
    grid = bank_grid(lo, hi)
    centre = {k: float(lead["weights"][k]) for k in WEIGHT_FIELDS}
    cands = [{**centre, "g_w_tso_shunt": x} for x in grid]
    keep = [w for w in cands if V2.passes_screen(w, ops)]
    P.log(f"stage II-c: leader {lead['key']} (g_w_tso_shunt {hi:.4g}); grid "
          f"{[round(x, 3) for x in grid]}; {len(keep)} / {len(cands)} pass the screen")
    # The Stage-I point's and the leader's windows are the stored ones: neither was
    # stopped early, so their records equal runs without the stop.
    for d in dirs:
        for k in (base["key"], lead["key"]):
            for f in (d / "windows").glob(f"{k}__*.json"):
                if not (rc / "windows" / f.name).exists():
                    shutil.copy2(f, rc / "windows" / f.name)
    STAGE["now"] = "bank"
    try:
        if keep:
            P.simulate(rc, keep, w_local, w_remote, split_local)
        cj = rc / "with_leader.json"
        cj.write_text(json.dumps(keep + [centre], indent=1), "utf-8")
        if P.run(screen_argv(rc, cj, max(w_local, 1)), rc / "assemble_leader.log") != 0:
            raise RuntimeError("stage II-c: assembly with the leader did not complete")
    finally:
        STAGE["now"] = "search"
    rep_c = stage_select([rc], "2c", ops, levels=LEVELS[-1:])
    ranked = {r["key"]: r for r in rep_c["ranked"]}
    if lead["key"] not in ranked:
        raise RuntimeError(f"stage II-c: the leader {lead['key']} is not feasible on its "
                           "own records")
    e_ref = ranked[lead["key"]]["aggregate"]["e_v_ts"]
    pick = pick_lowest_price(list(ranked.values()), e_ref)
    rows = []
    for f in sorted((rc / "candidates").glob("cand_*.json")):
        c = json.loads(f.read_text("utf-8"))
        if c.get("is_baseline"):
            continue
        e = rep_c["_engagement"][c["key"]]
        a = c["aggregate"]
        rows.append({"key": c["key"], "g_w_tso_shunt": float(c["weights"]["g_w_tso_shunt"]),
                     "feasible": c["key"] in ranked, "e_v_ts": a.get("e_v_ts"),
                     "e_v_ds": a.get("e_v_ds"), "n_hunting": a.get("n_hunting"),
                     "n_shunt_hunting": a.get("n_shunt_hunting"),
                     "bank_after_windows": e["bank_after_windows"], "sts_all": e["sts_all"]})
    rows.sort(key=lambda r: r["g_w_tso_shunt"])
    for r in rows:
        P.log(f"  II-c {r['key']} price {r['g_w_tso_shunt']:7.3f}: "
              f"{'feasible' if r['feasible'] else 'rejected'}, e_v_ts {r['e_v_ts']:.5f}, "
              f"hunting {r['n_hunting']:g}, bank hunting {r['n_shunt_hunting']:g}, "
              f"bank after events in {r['bank_after_windows']} window(s)")
    STAGE2C.update({"leader": lead["key"], "grid": grid, "e_v_ts_leader": e_ref,
                    "rows": rows, "pick": pick["key"],
                    "pick_g_w_tso_shunt": float(pick["weights"]["g_w_tso_shunt"])})
    (rc / "stage2c_summary.json").write_text(json.dumps(STAGE2C, indent=1, default=float),
                                             "utf-8")
    same = " (the leader itself)" if pick["key"] == lead["key"] else ""
    P.log(f"stage II-c: pick {pick['key']} (g_w_tso_shunt "
          f"{float(pick['weights']['g_w_tso_shunt']):.4g}){same}")
    return pick, rc


# -- go criteria and result -------------------------------------------------------------------

def go_criteria(win: dict, rep: dict) -> Dict[str, Any]:
    e = rep["_engagement"][win["key"]]
    checks = {"e_v_ts": win["aggregate"]["e_v_ts"] <= GO_FACTOR * V2_E_V_TS,
              "sts_couplers_every_search_window": e["sts_all"],
              "banks_after_events_2_windows": e["bank"],
              "all_limits_met": True}           # it is in rep["ranked"]
    return {"go": all(checks.values()), "checks": checks,
            "e_v_ts": win["aggregate"]["e_v_ts"], "limit_e_v_ts": GO_FACTOR * V2_E_V_TS,
            "engagement": e, "engagement_level": rep.get("_level")}


def write_result(win: dict, rep: dict, c_res: dict, w5: dict, ver: Dict[float, Any],
                 go: Dict[str, Any], ops) -> Path:
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    path = PARAMSET_DIR / f"{RESULT}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a parameter set is never overwritten")
    fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
    doc = {
        "name": RESULT,
        "title": "Controller weights selected by procedure v3 on the unit-commitment base",
        "description": [f"Selected by {PROCEDURE}: v2 with coupler engagement "
                        f"{sum(THETA_STS) / 2:g} pu (ladder {THETA_LADDER}), bank engagement "
                        "0.005 pu, g_w_gen box 5e5..5e7, search on W1, W3, W4, W6 with an "
                        "early stop, hunting tolerance 1, measured engagement filters, the "
                        f"band completed on W2, settling budget {N_PERIODS} periods; "
                        f"amendment 2: start-up exemption {STARTUP_EXEMPT_S:g} s, bank price "
                        "tuned on its own (Stage II-c)."],
        "provenance": {
            "producer": P.rel(Path(__file__)), "run_dir": P.rel(P.RUN),
            "procedure": PROCEDURE, "starting_paramset": P.START,
            "stage1_verification": {f"{k:g}": v for k, v in ver.items()},
            "dso_voltage_band_pu": list(P.BAND), "search_windows": list(SEARCH),
            "ranked_windows": ALL, "w5_report": w5, "selected_key": win["key"],
            "engagement_level": rep.get("_level"), "max_hunting_applied": TOLERANCE,
            "n_feasible": len(rep.get("ranked", [])),
            "theta_sts_min_pu": rep["_theta_sts"].get(win["key"]),
            "theta_ts_min_pu": rep["_theta_ts"].get(win["key"]),
            "engagement": rep["_engagement"].get(win["key"]),
            "failed_settling": sorted(k for k, v in c_res.items() if not v["pass"]),
            "aggregate_all_windows": win["aggregate"],
            "baseline_aggregate": rep.get("baseline", {}).get("aggregate"),
            "settling": c_res.get(win["key"]), "go_criteria": go,
            "amendment2": {"startup_exempt_s": STARTUP_EXEMPT_S, "stage2c": STAGE2C},
            "weight_fingerprint": weight_fingerprint(fields),
            **{f"design_{k}": v for k, v in V2.describe(fields, ops).items()},
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {path}")
    return path


# -- code on e5power ----------------------------------------------------------------------------

def sync_remote_code() -> None:
    """Copy every Python file that differs from e5power's commit, then compare
    hashes: the shards must run exactly this checkout's code."""
    rh = P.remote_head()
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    names = set(subprocess.check_output(git + ["diff", "--name-only", rh], text=True).split())
    names |= set(subprocess.check_output(git + ["ls-files", "--others", "--exclude-standard"],
                                         text=True).split())
    files = sorted(n for n in names if n.endswith(".py") and (ROOT / n).exists()
                   and not n.startswith(("docs/", "tests/", "results/")))
    if not files:
        # e5power already runs this code; an empty list made Get-FileHash fail on an
        # empty path and stop the driver (procedure v5, 2026-11-08).
        P.log(f"{P.REMOTE}: no Python file differs from {rh[:7]}; nothing copied")
        return
    for n in files:
        P.push(ROOT / n, P.REMOTE_ROOT + "\\" + n.replace("/", "\\"))
    local = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest().upper() for n in files}
    paths = "', '".join(P.REMOTE_ROOT + "\\" + n.replace("/", "\\") for n in files)
    out = P.ps(f"Invoke-Command -ComputerName {P.REMOTE} -ScriptBlock {{ "
               f"foreach ($f in @('{paths}')) {{ (Get-FileHash -Algorithm SHA256 $f).Hash }} }}")
    remote = out.split()
    bad = [n for n, h in zip(files, remote) if local[n] != h]
    if len(remote) != len(files) or bad:
        raise RuntimeError(f"code on {P.REMOTE} differs after the copy: {bad[:8]}")
    P.log(f"{P.REMOTE}: {len(files)} Python file(s) copied over {rh[:7]}; hashes equal")


# -- main -----------------------------------------------------------------------------------

def procedure(args) -> int:
    from tuning import stage1_screen as s1
    cfg = P.check_config()
    if not P.base_config().shunt_ff_ledger:
        raise RuntimeError("v3 runs with shunt_ff_ledger = True")
    global THETA_STS
    P.log(f"amendment 2: start-up exemption {STARTUP_EXEMPT_S:g} s; Stage II-c with "
          f"{N_2C} bank prices")
    P.log(f"== 1./2. Stage I (v3) and its verification: theta_STS ladder {THETA_LADDER}, "
          f"banks 0.005, g_w_gen 5e6; settling budget {N_PERIODS} periods "
          f"(acceptance p95 <= {N_PERIODS * _A1.N_INNER}, verification <= {V2.VERIFY_P95:g})")
    ops = V2.load_ops()
    if args.dry_run:
        from tuning.stage2_search import Box, sobol_candidates
        for theta in THETA_LADDER:
            w, _ = design(ops, theta)
            d = V2.describe(w, ops)
            draws = sobol_candidates(Box(centre=w, spans=V2.spans_v2()), V2.N_2A, seed=P.SEED)
            n = sum(1 for x in draws if V2.passes_screen(x, ops))
            P.log(f"dry run theta {theta:g}: g_w_dso_oltc {w['g_w_dso_oltc']:.4g}, "
                  f"theta_STS {d['theta_sts_min']:.4f}, lambda_max(B) {d['lambda_max_B']:.4f}; "
                  f"{n} / {V2.N_2A} II-a draws pass the screen")
        return 0
    sync_remote_code()
    if args.reuse_stage1 is not None:
        src = (ROOT / args.reuse_stage1).resolve() / "stage1" / "verification"
        dst = P.RUN / "stage1" / "verification"
        if not dst.exists():
            shutil.copytree(src, dst)
        P.log(f"Stage-I verification reused from {P.rel(src)} (amendment 2: Stage I unchanged)")
    ver: Dict[float, Any] = {}
    chosen = None
    for theta in THETA_LADDER:
        w, rep1 = design(ops, theta)
        d = V2.describe(w, ops)
        P.log(f"  theta {theta:g}: {', '.join(f'{k} {v:.6g}' for k, v in w.items())}")
        P.log(f"  rho {d['rho']:.4g}, kappa {d['kappa']:.4g}, theta_STS "
              f"{d['theta_sts_min']:.4f}, theta_TS {d['theta_ts_min']:.4f}, "
              f"lambda_max(B) {d['lambda_max_B']:.4f}")
        name = write_variant(w, rep1, ops, theta)
        ver[theta] = verify(theta, name, w, args.verify_workers)
        if ver[theta].get("verified"):
            chosen = (theta, name, w)
            break
    if chosen is None:
        P.log("no rung of the theta ladder passes the verification; the campaign stops")
        (P.RUN / "FAILED.txt").write_text("Stage-I verification failed at every rung\n", "utf-8")
        return 2
    theta, name, w = chosen
    THETA_STS = (theta - 0.005, theta + 0.005)
    (P.RUN / "stage1" / "chosen.json").write_text(json.dumps(
        {"theta_sts": theta, "paramset": name, "label": verify_label(theta),
         "theta_sts_band_pu": list(THETA_STS)}, indent=1), "utf-8")
    P.log(f"Stage-I point: {name} (theta_STS {theta:g})")
    P.START = name
    P.push_inputs()
    split_local = args.local_workers * args.local_speed
    P.log("== 3. Stage II-a (W1, W3, W4, W6; early stop)")
    rep = V2.stage2a(w, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 4. Stage II-b")
    rep, dirs = V2.stage2b(rep, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 4c. Stage II-c (bank price on its own; W1, W3, W4, W6 without the early stop)")
    pick, rc = stage2c(rep, dirs, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 5. completion of the band on W2, ranking on all windows")
    rep = complete_band(rep, dirs + [rc], ops, args.local_workers, args.remote_workers,
                        split_local, extra=(pick,))
    P.log("== 6. settling confirmation")
    win, c_res = V2.confirm(rep, args.confirm_workers)
    names = [lv for lv, _ in LEVELS]
    i = names.index(rep["_level"])
    while win is None and i + 1 < len(LEVELS):
        P.log(f"settling: no candidate at engagement level '{names[i]}' passes; "
              f"next level '{names[i + 1]}'")
        i += 1
        rep = stage_select([P.RUN / "complete"], f"complete_l{i}", ops, levels=LEVELS[i:])
        win, c_res = V2.confirm(rep, args.confirm_workers)
    if win is None:
        (P.RUN / "FAILED.txt").write_text("no candidate passes the settling test\n", "utf-8")
        return 2
    P.log(f"SELECTED {win['key']}")
    P.log("== 7. W5 for the report, result, go criteria")
    w5 = P.report_w5(win)
    go = go_criteria(win, rep)
    write_result(win, rep, c_res, w5, ver, go, ops)
    (P.RUN / ("GO.json" if go["go"] else "NOGO.json")).write_text(
        json.dumps(go, indent=1, default=float), "utf-8")
    P.log(f"go criteria: {'GO' if go['go'] else 'NO GO'} {go['checks']}")
    (P.RUN / "DONE.txt").write_text(f"selected {win['key']}\n", "utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_11_procedure_v3")
    p.add_argument("--local-workers", type=int, default=30)
    p.add_argument("--remote-workers", type=int, default=36)
    p.add_argument("--verify-workers", type=int, default=36)
    p.add_argument("--confirm-workers", type=int, default=36)
    p.add_argument("--local-speed", type=float, default=0.7)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--reuse-stage1", type=Path, default=None,
                   help="Run folder whose stage1/verification is taken over (amendment 2).")
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    P.START = V2.V1_START                  # operating points only, until Stage I is written
    if args.dry_run:
        P.RUN = ROOT / "results" / "_dry_ch9_11" / f"{datetime.now():%Y%m%d-%H%M%S}"
        P.RUN.mkdir(parents=True)
        return procedure(args)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            parameters={"procedure": PROCEDURE, "pi_q": PI_Q, "g_w_gen": G_W_GEN,
                        "d_engage_pu": D_ENGAGE, "theta_sts_band_pu": list(THETA_STS),
                        "theta_ts_max_pu": V2.THETA_TS_MAX, "search_windows": list(SEARCH),
                        "hunting_tolerance": TOLERANCE, "n_2a": V2.N_2A,
                        "theta_ladder": list(THETA_LADDER), "settling_periods": N_PERIODS,
                        "rounds_max": P.MAX_ROUNDS, "result_paramset": RESULT,
                        "amendment2": {"startup_exempt_s": STARTUP_EXEMPT_S, "n_2c": N_2C,
                                       "reuse_stage1": str(args.reuse_stage1)},
                        "workers": f"{args.local_workers} here + {args.remote_workers} on "
                                   f"{P.REMOTE}"},
            time_series=("Stage-I verification and settling confirmation: 120 isolated "
                         "interface steps in W1-W4, W6; Stage II: 12-h simulations of W1, "
                         "W3, W4, W6 (early stop); band completed on W2; W5 report only"),
            note=args.note or "Ch 9 weight selection, procedure v3",
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
        trun.describe_output("selection/", "theta, engagement, unions, ch_9_3_select reports")
        trun.describe_output("stage2c/", "bank price on its own (amendment 2)")
        trun.describe_output("complete/", "the band completed on W2")
        trun.describe_output("expC/", "settling confirmation in rank order")
        P.log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
