"""
experiments/ch_11_case2_multisystem/ref/run_ref.py
==================================================

Reference run of Case Study 2 (thesis ``ch:case3:nominal``): every variant
once over the reference period, full per-step logs, Case-2 metrics.

What is fixed here, and where it comes from
-------------------------------------------
* Variants and the base configuration: ``../variants.py`` (selected Chapter-9
  paramset, DSO_3 x2, minimum-class metering for every controller and the
  droop, no event inherited from ``make_config``).  The pair guard runs before
  anything is simulated.
* The disturbance schedule: ``scenarios.py`` (candidates A/B/C, events strictly
  between two TS dispatches; the choice is the author's after a screening).
* Provenance: ``experiments/thesis_runs.py`` -- README, manifest and the
  ``latex_header.tex`` RESULT PROVENANCE block to paste under
  ``ch:case3:nominal``.  Variants added later join the run with ``--run-dir``.

Output (``results/ch_11_case2_multisystem/ref/<NNNN>_<stamp>_run_ref[_<tag>]/``)
-------------------------------------------------------------------------------
``<variant>/log.pkl`` (all records; the records up to an abort if it failed),
``<variant>/status.json``, ``<variant>/config_repr.txt``, ``metrics.json``,
``events.csv``, ``report.md``.

Run
---
    python -m experiments.ch_11_case2_multisystem.ref.run_ref --scenario A --jobs 4 --note "..."
    python -m experiments.ch_11_case2_multisystem.ref.run_ref --scenario A --variants L2,O2 --minutes 10 --tag smoke
    python -m experiments.ch_11_case2_multisystem.ref.run_ref --run-dir <run> --variants S2   # add later
    python -m experiments.ch_11_case2_multisystem.ref.run_ref --report <run>                  # tables only

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""
from __future__ import annotations

import os

# Headless + UTF-8 + one BLAS thread per worker; must precede numpy imports.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import dataclasses
import datetime as _dt
import json
import pickle
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.ch_11_case2_multisystem import metrics_case2 as mc
from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.ref import scenarios as sc
from experiments.thesis_runs import add_run_arguments, start_thesis_run

TARGET = "ch:case3:nominal"
FEEDS = ("fig:case3:vtrack", "fig:case3:summary", "tab:case3:ref",
         "fig:case3:qtrack", "fig:case3:qerr", "fig:case3:event", "fig:case3:stsv",
         "fig:case3:tieflow")
HORIZON_MIN = 360
#: One realisation for the reference run (all variants share it).
NOISE_SEED = 20261027


def _run_variant(name: str, out: str, minutes: int, scenario: str,
                 noise_seed: int) -> Dict[str, object]:
    """Worker: one variant, its full log and status (a failure is a result)."""
    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = v.resolved(name, minutes, sc.events(scenario), noise_seed)
    cfg = dataclasses.replace(cfg, result_dir=str(d), verbose=1)
    if sc.start_time(scenario) is not None:
        cfg = dataclasses.replace(cfg, start_time=sc.start_time(scenario))
    if sc.ramps(scenario):
        cfg = dataclasses.replace(cfg, profile_ramps=tuple(sc.ramps(scenario)))
    (d / "config_repr.txt").write_text(repr(cfg), encoding="utf-8")
    t0 = _dt.datetime.now()
    with open(d / "console.log", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        try:
            log = run_multi_tso_dso(cfg)
            status = "ok"
        except Exception as exc:     # keep the trajectory up to the failure
            log = list(getattr(exc, "partial_log", None) or [])
            status = f"failed: {type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()
    with open(d / "log.pkl", "wb") as fh:
        pickle.dump(log, fh)
    st = {"variant": name, "status": status, "n_records": len(log),
          "minutes": minutes, "scenario": scenario, "noise_seed": noise_seed,
          "host": socket.gethostname(),
          "wall_s": (_dt.datetime.now() - t0).total_seconds()}
    (d / "status.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    return st


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def _load(run: Path, name: str) -> Optional[list]:
    p = run / name / "log.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh) or None


def _hv_ids(log: list) -> List[str]:
    for r in log:
        if r.dso_group_v_min_pu:
            return sorted(r.dso_group_v_min_pu)
    return []


def report(run: Path) -> Dict[str, object]:
    """Compute and write metrics.json, events.csv and report.md for ``run``."""
    cfg0 = v.base_config(10)
    t_ts, dt = float(cfg0.tso_period_s), float(cfg0.dt_s)
    v_lo, v_hi = float(cfg0.dso_v_min_pu), float(cfg0.dso_v_max_pu)
    names = sorted(p.name for p in run.iterdir()
                   if (p / "status.json").exists())
    logs = {n: _load(run, n) for n in names}
    res: Dict[str, object] = {}
    ev_rows = []
    for n, log in logs.items():
        st = json.loads((run / n / "status.json").read_text(encoding="utf-8"))
        if not log:
            res[n] = {"status": st["status"], "n_records": 0}
            continue
        areas = {z: mc.area_metrics(log, z) for z in (1, 2, 3)}
        res[n] = {
            "status": st["status"], "n_records": len(log),
            "areas": areas,
            "e_v_rms_areas": mc.aggregate_e_v(areas),
            "e_vp_rms_areas": mc.aggregate_e_v(areas, "e_vp"),
            "sts": {m: mc.sts_metrics(log, m, v_lo, v_hi) for m in _hv_ids(log)},
            "interface": mc.interface_metrics(log, t_ts, dt),
        }
        for row in mc.event_metrics(log, t_ts, dt):
            row = dict(row, variant=n)
            row["w_sts_mvar_min_total"] = float(sum(row["w_sts_mvar_min"].values()))
            row["w_sts_mvar_min"] = json.dumps(row["w_sts_mvar_min"])
            ev_rows.append(row)
    pairs = {"O3-O2": ("O3", "O2"), "S2-S1": ("S2", "S1")}
    res["recruited_flow"] = {
        k: mc.recruited_flow(logs[a], logs[b]) for k, (a, b) in pairs.items()
        if logs.get(a) and logs.get(b)}
    (run / "metrics.json").write_text(json.dumps(res, indent=2, default=float),
                                      encoding="utf-8")
    if ev_rows:
        import pandas as pd
        pd.DataFrame(ev_rows).to_csv(run / "events.csv", index=False)
    (run / "report.md").write_text(_report_md(res, ev_rows), encoding="utf-8")
    return res


def _f(x, scale=1e3, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x * scale:.{nd}f}"


def _report_md(res: Dict[str, object], ev_rows: List[dict]) -> str:
    L = ["# Case 2 reference run -- metrics", "",
         "Voltages in mpu (time-RMS / max / share of samples); reserve in Mvar.", "",
         "| variant | status | e_v (RMS areas) | e_vp (RMS areas) |",
         "|---|---|---|---|"]
    var = {k: d for k, d in res.items() if isinstance(d, dict) and "status" in d}
    for n, d in var.items():
        L.append(f"| {n} | {d['status'][:40]} | {_f(d.get('e_v_rms_areas'))} "
                 f"| {_f(d.get('e_vp_rms_areas'))} |")
    L += ["", "## Per area", "",
          "| variant | area | e_v | e_vp | worst | t_out [%] | u_q | R+ mean | R+ min "
          "| R- mean | R- min | AVR travel [pu] | TS taps | banks |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, d in var.items():
        for z, a in (d.get("areas") or {}).items():
            L.append(
                f"| {n} | {z} | {_f(a['e_v'])} | {_f(a['e_vp'])} | {_f(a['worst'])} "
                f"| {_f(a['t_out'], 100, 1)} | {_f(a['u_q'], 1, 3)} "
                f"| {_f(a['r_plus_mean'], 1, 0)} | {_f(a['r_plus_min'], 1, 0)} "
                f"| {_f(a['r_minus_mean'], 1, 0)} | {_f(a['r_minus_min'], 1, 0)} "
                f"| {_f(a['avr_trav'], 1, 3)} | {a['taps']} | {a['shunt_ops']} |")
    L += ["", "## Sub-transmission systems", "",
          "| variant | STS | u_q,STS | f_V,STS [mpu] | min margin to band [mpu] | coupler taps |",
          "|---|---|---|---|---|---|"]
    for n, d in var.items():
        for m, s in (d.get("sts") or {}).items():
            L.append(f"| {n} | {m} | {_f(s['u_q_sts'], 1, 3)} | {_f(s['f_v_sts'])} "
                     f"| {_f(s['margin_min'])} | {s['coupler_taps']} |")
    if ev_rows:
        L += ["", "## Events (eq:case3:epost; W = withdrawn interface support)", "",
              "| variant | t_e [min] | events | area | e(t_e+) | impact | epost "
              "| W total [Mvar min] |", "|---|---|---|---|---|---|---|---|"]
        for r in ev_rows:
            L.append(f"| {r['variant']} | {r['t_e_s'] / 60:.2f} | {r['events']} "
                     f"| {r['area']} | {_f(r['e_plus'])} | {_f(r['impact'])} "
                     f"| {_f(r['epost'])} | {_f(r['w_sts_mvar_min_total'], 1, 1)} |")
    rf = res.get("recruited_flow") or {}
    if rf:
        L += ["", "## Recruited interface flow T[|Q^A - Q^B|] [Mvar]", ""]
        for k, d in rf.items():
            L.append(f"* {k}: " + ", ".join(f"{c} {q:.1f}" for c, q in d.items()))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default=None,
                    help=f"one of {sorted(sc.SCENARIOS)}")
    ap.add_argument("--variants", default=",".join(v.WAVE_1))
    ap.add_argument("--minutes", type=int, default=HORIZON_MIN)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    ap.add_argument("--run-dir", default=None,
                    help="add the variants to this existing run_ref run")
    ap.add_argument("--allow-not-ready", action="store_true",
                    help="permit variants listed in variants.NOT_READY")
    ap.add_argument("--report", default=None, metavar="RUN",
                    help="only (re)compute the tables of an existing run")
    add_run_arguments(ap)
    a = ap.parse_args()

    if a.report:
        report(Path(a.report))
        print(f"report written to {a.report}")
        return

    sel = [s.strip() for s in a.variants.split(",") if s.strip()]
    for n in sel:
        if n not in v.VARIANTS:
            raise SystemExit(f"unknown variant {n!r}; valid {sorted(v.VARIANTS)}")
        if n in v.NOT_READY and not a.allow_not_ready:
            raise SystemExit(f"{n} is not ready: {v.NOT_READY[n]}")

    if a.run_dir:
        manifest = json.loads((Path(a.run_dir) / "run_manifest.json")
                              .read_text(encoding="utf-8"))
        params = manifest["parameters"]
        scenario, minutes = params["scenario"], int(params["horizon_min"])
        noise_seed = int(params["noise_seed"])
    else:
        if not a.scenario:
            raise SystemExit("--scenario is required for a new run")
        scenario, minutes, noise_seed = a.scenario, a.minutes, a.noise_seed
    evs = sc.events(scenario)
    v.assert_case2_controlled(minutes, evs, noise_seed)
    print("guard: every Case-2 step changes only what it is allowed to")

    base = v.base_config(minutes, evs, noise_seed)
    if sc.start_time(scenario) is not None:
        base = dataclasses.replace(base, start_time=sc.start_time(scenario))
    if sc.ramps(scenario):
        base = dataclasses.replace(base, profile_ramps=tuple(sc.ramps(scenario)))
    with start_thesis_run(
            __file__, target=TARGET, feeds=FEEDS, config=base,
            paramset=v.PARAMSET,
            parameters={
                "variants": sel, "scenario": scenario,
                "scenario_title": sc.SCENARIOS[scenario]["title"],
                "events": sc.describe(scenario), "ramps": sc.ramps(scenario),
                "horizon_min": minutes,
                "start_time": base.start_time.isoformat(),
                "unit_commitment": bool(base.unit_commitment),
                "slack_q_limit_mvar": base.slack_q_limit_mvar,
                "noise": (f"{base.measurement_noise.profile}, seed {noise_seed}, "
                          "incl. SVR and droop"),
                "noise_seed": noise_seed,
                "network": "DSO_3 x2 (load and DER)",
                "host": socket.gethostname()},
            note=a.note, tag=a.tag, resume=a.run_dir) as run:
        out = run.root
        print(f"run dir: {out} | scenario {scenario} | {minutes} min | {sel}",
              flush=True)
        failed: List[str] = []
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
            futs = {pool.submit(_run_variant, n, str(out), minutes, scenario,
                                noise_seed): n for n in sel}
            for fut in as_completed(futs):
                n = futs[fut]
                st = fut.result()
                print(f"  [{n}] {st['status']} ({st['n_records']} records, "
                      f"{st['wall_s'] / 60:.1f} min)", flush=True)
                if st["status"] != "ok":
                    failed.append(n)
                run.describe_output(n, f"variant {n}: {st['status']}, "
                                       f"{st['n_records']} records (log.pkl, "
                                       "status.json, config_repr.txt, console.log)")
                run.write()
        report(out)
        for f, what in (("metrics.json", "Case-2 metrics per variant, area, STS, coupler"),
                        ("events.csv", "per event and area: e(t_e+), impact, epost, W"),
                        ("report.md", "the tables")):
            if (out / f).exists():
                run.describe_output(f, what)
        if failed:
            run.fail(f"variant(s) failed: {', '.join(failed)}; see <variant>/status.json")


if __name__ == "__main__":
    main()
