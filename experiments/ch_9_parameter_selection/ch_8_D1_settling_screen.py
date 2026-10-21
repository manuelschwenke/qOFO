r"""D1 -- subordinate-settling screen of DSO weight sets (thesis Ch 8, Exp. D).

Handoff ``HANDOFF_D_settling_rerun.md`` (2026-09-28).  For every weight set in
``--sets-json`` this runs the UNCHANGED isolated-STS step protocol of
``ch_9_1_ninner_isolated_sts.py`` (frozen OFO parent, 10 Mvar step per
interface transformer, both directions, W1-W4 + W6, settle 600 s, observe
900 s, T_STS = 20 s, doubled DSO_3) and scores it with
``n_settle_final_1mvar``: first iteration from which the interface flow stays
within 1 Mvar of its FINAL value.

All sets share one process pool (jobs submitted set by set, so sets finish in
order), every case is persisted as JSON the moment it finishes, and a rerun
resumes from disk.

Also written per set, before any simulation: the analytic ``lambda_max(B)``
(worst over layers and W1-W6, ``tuning.stage1_screen``) and the linear
settling prediction (``tuning.settling_predictor``, an OPTIMISTIC ranking
proxy -- it ignores constraints, noise and nonlinearity).

``--sets-json`` is a list of ``{"name": str, "weights": {complete weight
vector}, "cfg_overrides": {optional MultiTSOConfig fields}, "note": str}``.

Usage::

    python experiments/ch_9_parameter_selection/ch_8_D1_settling_screen.py \
        --sets-json <sets.json> --out results/THESIS_ch8_D_settling_<date>/d1 \
        --workers 36
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ch_9_1_ninner_isolated_sts as sts  # noqa: E402

#: Reference paramset: the TSO part and every field not in the set come from
#: here.  In the isolated test the TSO weights only affect the t = 0 solve.
BASE_PARAMSET = "ch9_selected"
MEASURED_WINDOWS = ("W1", "W2", "W3", "W4", "W6")   # W5 dead zone, excluded


def _case_path(out: Path, name: str, job: Dict[str, Any]) -> Path:
    t = job["trafo"].replace("|", "_")
    return out / "cases" / name / f"{job['window']}__{t}__{job['direction']}.json"


def _run(job: Dict[str, Any]) -> Dict[str, Any]:
    """Pool entry: one step, persisted.  Module level, so picklable."""
    path = Path(job["_out"])
    try:
        res = sts._run_case(job)
    except Exception as exc:                      # noqa: BLE001 - recorded
        import traceback
        res = {**job, "failure": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(), "rows": []}
    for row in res.get("rows", []):
        row["case_wall_s"] = res.get("wall_s", float("nan"))
    for pt in res.get("trajectory", []) or []:
        pt["case_wall_s"] = res.get("wall_s", float("nan"))
    rec = {"set": job["_set"], "job": {k: v for k, v in job.items()
                                       if not k.startswith("_")},
           "failure": res.get("failure", ""), "rows": res.get("rows", []),
           "trajectory": res.get("trajectory", []) or [],
           "wall_s": res.get("wall_s", float("nan"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float), encoding="utf-8")
    tmp.replace(path)
    return {"set": job["_set"], "failure": rec["failure"],
            "n": [r.get("n_settle_final_1mvar") for r in rec["rows"]],
            "wall_s": rec["wall_s"], "case": path.name}


def jobs_for_set(s: Dict[str, Any], out: Path, a: argparse.Namespace
                 ) -> List[Dict[str, Any]]:
    jobs = []
    for w in MEASURED_WINDOWS:
        for d_idx, d in enumerate(("DSO_1", "DSO_2", "DSO_3", "DSO_4")):
            for i in sts._trafo_indices(d, 3):
                for dirn in ("up", "down"):
                    j = {"window": w, "dso": d, "direction": dirn,
                         "variant": "frozen_ofo", "band_mvar": 1.0,
                         "band_fraction": sts.DEFAULT_BAND_FRACTION,
                         "max_step_mvar": 10.0, "der_q_override_pu": None,
                         "flat_mvar": sts.DEFAULT_FLAT_MVAR,
                         "save_trajectories": True,
                         "trafo": f"{d}|trafo_{i}",
                         "paramset": BASE_PARAMSET,
                         "weights": s["weights"],
                         "cfg_overrides": s.get("cfg_overrides") or None,
                         "frozen_network": True, "bank": "conditions",
                         "no_measurement_noise": False, "_set": s["name"]}
                    j["_out"] = str(_case_path(out, s["name"], j))
                    jobs.append(j)
    return jobs


def analytic(sets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """lambda_max(B) and the linear settling prediction per set."""
    from _ch9_selected_design import build_selected_config
    from tuning.settling_predictor import load_or_capture, predict, summarise
    from tuning.stage1_screen import (load_or_build_operating_points,
                                      screen_weights)
    from tuning_mc.scenarios_mc_v2 import CONDITIONS

    cfg, _ = build_selected_config(paramset=BASE_PARAMSET)
    conds = [(r[0], r[1]) for r in CONDITIONS]
    cache = REPO_ROOT / "results" / "stage1" / "operating_points"
    ops = load_or_build_operating_points(cfg, conds, cache, verbose=False)
    areas = load_or_capture(cfg, conds, cache, verbose=False)
    out: Dict[str, Dict[str, Any]] = {}
    for s in sets:
        w = dict(s["weights"])
        per_area = (s.get("cfg_overrides") or {}).get("dso_g_q_per_area") or {}
        rows = screen_weights(ops, w)
        if per_area:
            # Re-screen the STS layers of areas with their own g_q.
            rows = [r for r in rows if not (r.layer == "STS" and r.area in per_area)]
            for area, gq in per_area.items():
                rows += [r for r in screen_weights(ops, {**w, "g_q": gq})
                         if r.layer == "STS" and r.area == area]
        worst = max(rows, key=lambda r: r.lambda_max)
        pred_rows = []
        for d in ("DSO_1", "DSO_2", "DSO_3", "DSO_4"):
            wd = {**w, "g_q": per_area.get(d, w["g_q"])}
            pred_rows += [r for r in predict(areas, wd,
                                             conditions=MEASURED_WINDOWS)
                          if r["dso"] == d]
        out[s["name"]] = {
            "lambda_max_B": worst.lambda_max,
            "lambda_binding": f"{worst.key}@{worst.condition}",
            "lambda_max_B_sts": max(r.lambda_max for r in rows
                                    if r.layer == "STS"),
            **summarise(pred_rows)}
    return out


def _q(xs: List[float], q: float) -> float:
    return sts._quantile(xs, q)


def assemble(out: Path, sets: List[Dict[str, Any]],
             ana: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Per-set run folders plus d1_screen.csv from the persisted cases."""
    table: List[Dict[str, Any]] = []
    for s in sets:
        d = out / "cases" / s["name"]
        recs = ([json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(d.glob("*.json"))] if d.exists() else [])
        rows = [r for rec in recs for r in rec["rows"]]
        traj = [p for rec in recs for p in rec["trajectory"]]
        fails = [{**rec["job"], "failure": rec["failure"]}
                 for rec in recs if rec["failure"]]
        row: Dict[str, Any] = {"set": s["name"], "note": s.get("note", ""),
                               **{k: s["weights"][k] for k in sorted(s["weights"])},
                               "cfg_overrides": json.dumps(s.get("cfg_overrides") or {}),
                               **ana.get(s["name"], {}),
                               "n_cases_done": len(recs), "n_failed": len(fails)}
        if rows:
            run_dir = out / "sets" / s["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            m = {**meta, "design": {"paramset": BASE_PARAMSET,
                                    "candidate_key": s["name"],
                                    "campaign": "D1 settling screen",
                                    "frozen_network": True,
                                    "weights": s["weights"],
                                    "cfg_overrides": s.get("cfg_overrides")},
                 "constants": {"settle_s": sts.SETTLE_S,
                               "observe_s": sts.OBSERVE_S,
                               "band_mvar": 1.0,
                               "band_fraction": sts.DEFAULT_BAND_FRACTION,
                               "flat_mvar": sts.DEFAULT_FLAT_MVAR,
                               "max_step_mvar": 10.0,
                               "variant": "frozen_ofo"}}
            sts.write_outputs(run_dir, rows, sts.aggregate(rows), fails, m)
            if traj:
                sts.write_trajectories(run_dir, traj)
            for a in sts.aggregate_settle(rows):
                tag = "pooled" if a["dso"] == "__pooled__" else a["dso"]
                row[f"p50_{tag}"] = a["n_settle_median"]
                row[f"p95_{tag}"] = a["n_settle_p95"]
                row[f"max_{tag}"] = a["n_settle_max"]
                if tag == "pooled":
                    row["p90_pooled"] = a["n_settle_p90"]
                    row["share_le_9"] = a["share_le_9"]
                    row["n_censored"] = a["n_settle_censored"]
                    row["residual_median_mvar"] = a["residual_median_mvar"]
                    row["residual_p95_mvar"] = a["residual_p95_mvar"]
                    row["n_residual_gt_1mvar"] = a["n_residual_gt_1mvar"]
                    row["v_violations"] = a["v_violations"]
                    row["tap_moves_total"] = a["tap_moves_total"]
        table.append(row)
    cols: List[str] = []
    for r in table:
        cols += [k for k in r if k not in cols]
    with (out / "d1_screen.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ch_8_D1_settling_screen")
    ap.add_argument("--sets-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args(argv)

    sets = json.loads(a.sets_json.read_text(encoding="utf-8"))
    names = [s["name"] for s in sets]
    if len(set(names)) != len(names):
        raise SystemExit("set names must be unique")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sets.json").write_text(json.dumps(sets, indent=1),
                                     encoding="utf-8")

    meta = sts.provenance(argparse.Namespace(
        band_fraction=sts.DEFAULT_BAND_FRACTION, band_mvar=1.0,
        variant="frozen_ofo", der_q_override_pu=None,
        flat_mvar=sts.DEFAULT_FLAT_MVAR, max_step_mvar=10.0,
        case_design="full", bank="conditions", **vars(a)), {})
    meta["script_d1"] = "experiments/ch_9_parameter_selection/ch_8_D1_settling_screen.py"
    meta["excluded_windows"] = ["W5"]
    (a.out / "run_meta.json").write_text(json.dumps(meta, indent=1,
                                                    default=str),
                                         encoding="utf-8")

    print(f"[D1] {len(sets)} set(s); analytic screen ...", flush=True)
    ana = analytic(sets)
    for s in sets:
        x = ana[s["name"]]
        print(f"  {s['name']:>24}: lambda_max(B) {x['lambda_max_B']:.3f} "
              f"({x['lambda_binding']}), predicted p95 {x['pred_p95']:.0f} "
              f"(DSO_3 {x['pred_p95_DSO_3']:.0f})", flush=True)
    (a.out / "d1_analytic.json").write_text(json.dumps(ana, indent=1),
                                            encoding="utf-8")
    if a.assemble_only:
        assemble(a.out, sets, ana, meta)
        return 0

    jobs = [j for s in sets for j in jobs_for_set(s, a.out, a)]
    todo = [j for j in jobs if not Path(j["_out"]).exists()]
    print(f"[D1] {len(jobs)} cases, {len(todo)} to run, {a.workers} workers",
          flush=True)
    t0 = time.perf_counter()
    done = 0
    attempt = 0
    per_set_left = {n: sum(1 for j in todo if j["_set"] == n) for n in names}
    while todo and attempt <= a.max_restarts:
        attempt += 1
        broken = False
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(_run, j): j for j in todo}
                for fut in as_completed(futs):
                    j = futs[fut]
                    try:
                        r = fut.result()
                    except BrokenProcessPool:
                        broken = True
                        break
                    done += 1
                    per_set_left[j["_set"]] -= 1
                    tag = (f"FAILED {r['failure'].splitlines()[0]}"
                           if r["failure"] else f"n_settle={r['n']}")
                    print(f"  [{done}/{len(todo)}] {r['set']} {r['case']} "
                          f"{tag} ({r['wall_s']:.0f} s, "
                          f"{(time.perf_counter() - t0) / 3600:.2f} h)",
                          flush=True)
                    if per_set_left[j["_set"]] == 0:
                        assemble(a.out, sets, ana, meta)
                        print(f"[D1] set {j['_set']} complete -> d1_screen.csv",
                              flush=True)
        except BrokenProcessPool:
            broken = True
        todo = [j for j in jobs if not Path(j["_out"]).exists()]
        if not broken:
            break
        print(f"[D1] pool broke; {len(todo)} cases remain", flush=True)

    assemble(a.out, sets, ana, meta)
    print(f"[D1] finished in {(time.perf_counter() - t0) / 3600:.2f} h; "
          f"wrote {a.out / 'd1_screen.csv'}")
    return 0 if not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
