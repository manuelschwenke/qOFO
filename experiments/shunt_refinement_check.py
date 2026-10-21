"""Reproducible, optional shunt-controller comparisons on the AC benchmark.

The trial weights are experimental, not a replacement for a selected thesis
parameter set. The four cases isolate persistence, tracking assistance, and
near-limit reserve support. The configured profiles are retained and horizontal
coordination is disabled to isolate the vertical controllers. The normal MIQPs and
AC power flows are executed. No synthetic measurements are injected.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paramsets import apply_paramset
from experiments.run_multi_system_ofo import make_config
from experiments.runners.multi_tso_dso import run_multi_tso_dso
from experiments.helpers.records import ContingencyEvent


def json_data(value):
    if isinstance(value, dict):
        return {str(k): json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_config(case, seconds, paramset, disturbance=True, frozen_network=False):
    cfg = apply_paramset(make_config(), paramset, verbose=False)
    switches = {f: False for f in cfg.__dataclass_fields__ if f.startswith("live_plot_")}
    cfg = replace(cfg, **switches, n_total_s=float(seconds), verbose=0,
        use_profiles=True, coordination_mode="none", run_stability_analysis=False,
        contingencies=([
            ContingencyEvent(minute=10, element_type="load", bus=11,
                             p_mw=0, q_mvar=150, action="connect"),
            ContingencyEvent(minute=30, element_type="load", bus=11,
                             p_mw=0, q_mvar=150, action="trip"),
        ] if disturbance else []),
        shunt_int_t_persistence_s=0.0 if case == "legacy" else 360.0,
        shunt_int_gradient_deadband=1e-8,
        shunt_sts_support=case in ("assistance", "reserve"),
        shunt_sts_tracking_weight=10.0,
        shunt_sts_reserve_weight=100.0 if case == "reserve" else 0.0,
        shunt_sts_tracking_deadband_mvar=5.0,
        shunt_sts_tracking_scale_mvar=25.0,
        shunt_sts_reserve_start=0.8,
        shunt_sts_response_wait_s=40.0,
    )
    if frozen_network:
        cfg = replace(cfg, dso_der_scale={"DSO_3": 2.0},
                      dso_load_p_scale={"DSO_3": 2.0})
    return cfg


def summarise(records):
    errors, volts_ts, volts_sts, utilisation, statuses = [], [], [], [], Counter()
    commits = []
    for r in records:
        for key, target in r.dso_trafo_q_set_mvar.items():
            measured = r.dso_trafo_q_actual_mvar.get(key)
            if target is not None and measured is not None:
                errors.append(float(target-measured))
        volts_ts.extend(r.zone_v_min.values())
        volts_ts.extend(r.zone_v_max.values())
        volts_sts.extend(r.dso_group_v_min_pu.values())
        volts_sts.extend(r.dso_group_v_max_pu.values())
        statuses.update(v for v in r.dso_status.values() if v is not None)
        for key, q in r.dso_group_q_der_mvar.items():
            lo = r.dso_group_q_der_min_mvar.get(key)
            hi = r.dso_group_q_der_max_mvar.get(key)
            if lo is not None and hi is not None and hi-lo > 1e-6:
                utilisation.append(abs((2*q-lo-hi)/(hi-lo)))
        commits.extend(dict(time_s=r.time_s, **c) for c in r.shunt_commits)
    e = np.asarray(errors)
    return {
        "n_records": len(records), "n_switches": len(commits),
        "switch_reasons": dict(Counter(c["reason"] for c in commits)),
        "interface_error_rmse_mvar": float(np.sqrt(np.mean(e**2))) if e.size else None,
        "interface_error_abs_p95_mvar": float(np.quantile(abs(e), .95)) if e.size else None,
        "ts_voltage_min_max_pu": [min(volts_ts), max(volts_ts)] if volts_ts else None,
        "sts_voltage_min_max_pu": [min(volts_sts), max(volts_sts)] if volts_sts else None,
        "aggregate_der_utilisation_p95": float(np.quantile(utilisation, .95)) if utilisation else None,
        "dso_solver_status_counts": dict(statuses), "commits": commits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["legacy", "confirmed", "assistance", "reserve"],
                        choices=["legacy", "confirmed", "assistance", "reserve"])
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--paramset", default="ch9_selected_pcc27_gqceil_comp_q_off")
    parser.add_argument("--no-disturbance", action="store_true")
    parser.add_argument("--frozen-network", action="store_true",
                        help="Use the October campaign's DSO_3 DER/load multipliers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        case_dir = args.output / case
        case_dir.mkdir(exist_ok=True)
        cfg = build_config(case, args.seconds, args.paramset, not args.no_disturbance,
                           args.frozen_network)
        (case_dir / "config.json").write_text(json.dumps(json_data(asdict(cfg)), indent=2, default=str), encoding="utf-8")
        files = ["controller/shunt_integrator.py", "controller/shunt_support.py",
                 "controller/tso_controller.py", "experiments/runners/multi_tso_dso.py"]
        hashes = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
        (case_dir / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        print(f"Starting {case}: {args.seconds/60:g} simulated minutes", flush=True)
        started = time.perf_counter()
        with (case_dir / "run.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                records = run_multi_tso_dso(cfg)
        summary = summarise(records)
        summary["wall_time_s"] = time.perf_counter()-started
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (case_dir / "records.pkl").open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(json.dumps({k:v for k,v in summary.items() if k != "commits"}), flush=True)


if __name__ == "__main__":
    main()
