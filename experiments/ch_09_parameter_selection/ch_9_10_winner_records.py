r"""Record-keeping re-run of selected candidates (run 0004 follow-up, 2026-11-03).

The Stage II screen (``ch_9_3_stage2_screen``) keeps one summary row per window
and drops the record list, so two questions about the v2 result cannot be
answered from what run 0004 stored:

* whether the distribution-system voltages stay inside the DSO operating band
  0.92-1.08 pu (``v_worst_ds`` is a p95 of ``|v - v_set|``, side unknown);
* why the selected candidate never commits a bank and hardly moves an STS tap.

This script re-runs given candidates exactly as ``_run_one_window`` of the
screen does -- same parameter set, weights, frozen network and evented 12 h
windows -- and keeps, per (candidate, window):

* ``<key>__<W>.json``: the screen's row, recomputed with the same helpers, so
  the re-run can be checked against the stored window result, plus a DSO-band
  summary per HV group;
* ``<key>__<W>_env.npz``: the per-step voltage envelopes (DS per HV group, TS);
* ``<key>__<W>_records.pkl.gz``: the full record list, for the actuator analysis.

Usage (from the project root)::

    python experiments\ch_09_parameter_selection\ch_9_10_winner_records.py ^
        --paramset stage1_v2_uc_2026_11_pq10 --frozen-network --evented ^
        --candidates-json <file> --out-dir <dir> --workers 12
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOWS = ("W1", "W2", "W3", "W4", "W5", "W6")
DSO_BAND = (0.92, 1.08)          # pu, the DSO operating band of procedure v2


def _pin_threads() -> None:
    """One BLAS thread per worker, as in the screen."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _longest_run(mask, dt_min: float) -> float:
    """Longest consecutive stretch of ``mask`` in minutes."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best * dt_min


def _band_summary(records) -> Dict[str, Any]:
    """DSO-band summary per HV group and over all groups, and the envelopes."""
    import numpy as np

    from experiments.helpers.comparison_metrics import (voltage_envelope_ds,
                                                        voltage_envelope_ts)

    env_ds = voltage_envelope_ds(records)
    env_ts = voltage_envelope_ts(records)
    t = np.asarray(env_ds["t_min"], dtype=float)
    dt_min = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    lo, hi = DSO_BAND
    groups = sorted({g for r in records for g in r.dso_group_v_max_pu})
    vmax = {g: np.array([r.dso_group_v_max_pu.get(g, np.nan) for r in records], float)
            for g in groups}
    vmin = {g: np.array([r.dso_group_v_min_pu.get(g, np.nan) for r in records], float)
            for g in groups}

    def one(v_hi, v_lo) -> Dict[str, float]:
        over, under = v_hi > hi, v_lo < lo
        return {
            "v_max_pu": float(np.nanmax(v_hi)),
            "v_min_pu": float(np.nanmin(v_lo)),
            "v_max_p95_pu": float(np.nanpercentile(v_hi, 95.0)),
            "share_over_1p08": float(np.mean(over)),
            "share_under_0p92": float(np.mean(under)),
            "max_excess_over_1p08_pu": float(max(np.nanmax(v_hi) - hi, 0.0)),
            "longest_over_1p08_min": _longest_run(over, dt_min),
        }

    summary = {"dt_min": dt_min, "band_pu": list(DSO_BAND),
               "all_groups": one(env_ds["v_max"], env_ds["v_min"]),
               "per_group": {g: one(vmax[g], vmin[g]) for g in groups},
               "ts_v_max_pu": float(np.nanmax(env_ts["v_max"])),
               "ts_v_min_pu": float(np.nanmin(env_ts["v_min"]))}
    arrays = {"t_min": t, "ts_v_max": env_ts["v_max"], "ts_v_min": env_ts["v_min"],
              "ds_v_max": env_ds["v_max"], "ds_v_min": env_ds["v_min"],
              "groups": np.array(groups)}
    for i, g in enumerate(groups):
        arrays[f"ds_v_max_g{i}"] = vmax[g]
        arrays[f"ds_v_min_g{i}"] = vmin[g]
    return {"summary": summary, "arrays": arrays}


def _run_unit(job: Dict[str, Any]) -> Dict[str, Any]:
    """One (candidate, window), set up exactly as the screen's ``_run_one_window``."""
    _pin_threads()
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np

    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config
    from tuning.metrics import MetricScales
    from tuning.parameters import apply_to_config
    from tuning.runner import run_scenario_with_records
    from tuning.scenarios_mc_v2 import condition_set, condition_set_evented
    from tuning.stage2_search import (avr_step_metrics, hunting_summary,
                                      shunt_summary)

    out = Path(job["out"])
    stem = f"{job['key']}__{job['window']}"
    if (out / f"{stem}.json").exists():
        return {"stem": stem, "status": "cached"}

    base = apply_to_config(
        apply_paramset(make_config(), job["paramset"], verbose=False), job["weights"])
    if job["frozen_network"]:
        import dataclasses as _dc

        from experiments.ch_09_parameter_selection._ch9_selected_design import (
            FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE)
        base = _dc.replace(base, dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                           dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
    bank = condition_set_evented if job["evented"] else condition_set
    sc = next(s for s in bank(hours=job["hours"]) if s.name == job["window"])

    t0 = time.perf_counter()
    result, records = run_scenario_with_records(sc, base, MetricScales())
    m = result.metrics
    # The screen's row, field by field (``ch_9_3_stage2_screen._run_one_window``,
    # v_metric 'bus'), so the re-run can be compared with the stored window.
    row = {
        "e_v_ts": float(getattr(m, "v_rms_ts", float("nan"))),
        "e_v_ds": float(getattr(m, "v_rms_ds", float("nan"))),
        "e_q": float(getattr(m, "itae_q_pcc", float("nan"))),
        "v_worst_ts": float(getattr(m, "v_worst_ts", 0.0) or 0.0),
        "v_worst_ds": float(getattr(m, "v_worst_ds", 0.0) or 0.0),
        "n_viol_v_ts": float(getattr(m, "n_viol_v_ts", 0) or 0),
        "n_viol_v_ds": float(getattr(m, "n_viol_v_ds", 0) or 0),
        "tap_ops_per_h_tso": float(getattr(m, "tap_ops_per_h_tso", 0.0) or 0.0),
        "tap_ops_per_h_dso": float(getattr(m, "tap_ops_per_h_dso", 0.0) or 0.0),
        "tap_rev_per_h_tso": float(getattr(m, "tap_reversals_per_h_tso", 0.0) or 0.0),
        "tap_rev_per_h_dso": float(getattr(m, "tap_reversals_per_h_dso", 0.0) or 0.0),
        "pf_failures": float(getattr(m, "pf_failures", 0) or 0),
        "failure_reason": result.failure_reason or "",
    }
    row.update(avr_step_metrics(records))
    row.update(hunting_summary(records, sc.event_times_s))
    row.update(shunt_summary(records, sc.event_times_s))
    band = _band_summary(records)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{stem}_env.npz", **band["arrays"])
    with gzip.open(out / f"{stem}_records.pkl.gz", "wb", compresslevel=3) as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out / f"{stem}.json").write_text(json.dumps(
        {"key": job["key"], "window": job["window"], "weights": job["weights"],
         "wall_s": time.perf_counter() - t0, "row": row, "dso_band": band["summary"]},
        indent=1, default=float), encoding="utf-8")
    return {"stem": stem, "status": "ok",
            "v_max_ds": band["summary"]["all_groups"]["v_max_pu"]}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paramset", required=True)
    ap.add_argument("--candidates-json", type=Path, required=True,
                    help="list of {key, weights}")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--frozen-network", action="store_true")
    ap.add_argument("--evented", action="store_true")
    a = ap.parse_args(argv)

    cands = json.loads(a.candidates_json.read_text("utf-8"))
    windows = [w.strip() for w in a.windows.split(",") if w.strip()]
    # Window-major, in the order given: list the long windows first (W6, W3) and a
    # small pool does not end on a lone W6.  The order does not change a result.
    jobs = [{"key": c["key"], "weights": c["weights"], "window": w,
             "paramset": a.paramset, "hours": a.hours, "out": str(a.out_dir),
             "frozen_network": a.frozen_network, "evented": a.evented}
            for w in windows for c in cands]
    print(f"[records] {len(cands)} candidate(s) x {len(windows)} window(s) = "
          f"{len(jobs)} units, {a.workers} workers -> {a.out_dir}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run_unit, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
                print(f"  [{k}/{len(jobs)}] {r['stem']} {r['status']}"
                      + (f" v_max_ds={r['v_max_ds']:.4f}" if "v_max_ds" in r else ""),
                      flush=True)
            except Exception as exc:          # noqa: BLE001 - reported, run continues
                print(f"  [{k}/{len(jobs)}] {j['key']}__{j['window']} FAILED: {exc}",
                      flush=True)
    print(f"[records] finished in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
