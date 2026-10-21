"""AC comparison of the switched-shunt arms: banks held against banks dispatched.

Two cases on the same parameter set, the same profiles and the same disturbance:

``off``
    The MSC / MSR banks are installed in the plant and held at their initial
    level. They are not control variables, so the zone MIQP dispatches only the
    continuous actuators and the OLTCs.
``miqp``
    The bank level is an integer variable of the zone MIQP, bounded by the
    device's own holding time and level limits, restricted to at most one step
    per zone and iteration, and subject to a hard predicted-voltage corridor.

Horizontal coordination is disabled so the vertical cascade is isolated. The
ordinary MIQPs and AC power flows run; no synthetic measurements are injected.
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
from experiments.helpers.records import ContingencyEvent
from experiments.run_multi_system_ofo import make_config
from experiments.runners.multi_tso_dso import run_multi_tso_dso

#: Files whose content decides what the run means; hashed into every case
#: directory so a result can be tied to the code that produced it.
SOURCE_FILES = (
    "controller/tso_controller.py",
    "controller/base_controller.py",
    "controller/shunt_integrator.py",
    "optimisation/miqp_solver.py",
    "sensitivity/jacobian.py",
    "experiments/runners/multi_tso_dso.py",
    "configs/config.py",
    "experiments/shunt_dispatch_check.py",
)


def json_data(value):
    if isinstance(value, dict):
        return {str(k): json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_config(case, seconds, paramset, disturbance=True, frozen_network=False):
    """The parameter set, with the switched-shunt arm selected by ``case``."""
    cfg = apply_paramset(make_config(), paramset, verbose=False)
    switches = {f: False for f in cfg.__dataclass_fields__
                if f.startswith("live_plot_")}
    cfg = replace(
        cfg, **switches, n_total_s=float(seconds), verbose=0,
        use_profiles=True, coordination_mode="none",
        run_stability_analysis=False,
        install_tso_tertiary_shunts=True, tso_shunt_kind="msc_msr",
        shunt_dispatch="miqp" if case == "miqp" else "off",
        contingencies=([
            ContingencyEvent(minute=10, element_type="load", bus=11,
                             p_mw=0, q_mvar=150, action="connect"),
            ContingencyEvent(minute=30, element_type="load", bus=11,
                             p_mw=0, q_mvar=150, action="trip"),
        ] if disturbance else []),
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
                errors.append(float(target - measured))
        volts_ts.extend(r.zone_v_min.values())
        volts_ts.extend(r.zone_v_max.values())
        volts_sts.extend(r.dso_group_v_min_pu.values())
        volts_sts.extend(r.dso_group_v_max_pu.values())
        statuses.update(v for v in r.dso_status.values() if v is not None)
        for key, q in r.dso_group_q_der_mvar.items():
            lo = r.dso_group_q_der_min_mvar.get(key)
            hi = r.dso_group_q_der_max_mvar.get(key)
            if lo is not None and hi is not None and hi - lo > 1e-6:
                utilisation.append(abs((2 * q - lo - hi) / (hi - lo)))
        commits.extend(dict(time_s=r.time_s, **c) for c in r.shunt_commits)
    e = np.asarray(errors)
    return {
        "n_records": len(records), "n_switches": len(commits),
        "switch_kinds": dict(Counter(c["kind"] for c in commits)),
        "interface_error_rmse_mvar":
            float(np.sqrt(np.mean(e ** 2))) if e.size else None,
        "interface_error_abs_p95_mvar":
            float(np.quantile(abs(e), .95)) if e.size else None,
        "ts_voltage_min_max_pu":
            [min(volts_ts), max(volts_ts)] if volts_ts else None,
        "sts_voltage_min_max_pu":
            [min(volts_sts), max(volts_sts)] if volts_sts else None,
        "aggregate_der_utilisation_p95":
            float(np.quantile(utilisation, .95)) if utilisation else None,
        "dso_solver_status_counts": dict(statuses), "commits": commits,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", choices=["off", "miqp"],
                    default=["off", "miqp"])
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--paramset", default="ch9_selected_pcc27_gqceil")
    ap.add_argument("--no-disturbance", action="store_true")
    ap.add_argument("--frozen-network", action="store_true",
                    help="the October campaign's DSO_3 DER/load multipliers")
    # The step price. 10 and 20 gave identical trajectories on the selected
    # set; 30 loses the first commit and most of the benefit (2026-10-21).
    ap.add_argument("--movement-weight", type=float, default=None,
                    help="g_w_tso_shunt; None keeps the parameter set's value")
    # The one parameter the benefit turns on: 1800 s keeps the engagement,
    # 900 s pays +5.6% interface RMSE for an engage-release cycle.
    ap.add_argument("--dwell-s", type=float, default=None,
                    help="minimum holding time per bank; None keeps the "
                         "parameter set's value")
    ap.add_argument("--verbose", type=int, default=0,
                    help="1 prints a per-TS-instant progress line to run.log")
    ap.add_argument("--start-time", default=None,
                    help="profile start, e.g. '2016-11-26 10:00'")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for case in args.cases:
        cfg = build_config(case, args.seconds, args.paramset,
                           not args.no_disturbance, args.frozen_network)
        cfg = replace(cfg, verbose=args.verbose)
        if args.movement_weight is not None:
            cfg = replace(cfg, g_w_tso_shunt=args.movement_weight)
        if args.dwell_s is not None:
            cfg = replace(cfg, shunt_int_t_dwell_s=args.dwell_s)
        if args.start_time is not None:
            import pandas as pd
            cfg = replace(cfg, start_time=pd.Timestamp(args.start_time))

        d = args.output/case
        d.mkdir(parents=True, exist_ok=False)
        (d/"config.json").write_text(
            json.dumps(json_data(asdict(cfg)), indent=2, default=str),
            encoding="utf-8")
        hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                  for p in SOURCE_FILES}
        (d/"source_hashes.json").write_text(json.dumps(hashes, indent=2),
                                            encoding="utf-8")
        print(f"Starting {case}, {args.seconds/60:g} simulated minutes",
              flush=True)
        start = time.perf_counter()
        with (d/"run.log").open("w", encoding="utf-8") as log, \
                redirect_stdout(log), redirect_stderr(log):
            records = run_multi_tso_dso(cfg)
        result = summarise(records)
        result["wall_time_s"] = time.perf_counter() - start
        (d/"summary.json").write_text(json.dumps(result, indent=2),
                                      encoding="utf-8")
        (d/"records.pkl").write_bytes(pickle.dumps(records))
        print(json.dumps({k: v for k, v in result.items() if k != "commits"}),
              flush=True)


if __name__ == "__main__":
    main()
