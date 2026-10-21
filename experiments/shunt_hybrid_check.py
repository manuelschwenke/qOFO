"""Diagnostic AC comparisons for the joint MIQP and slow shunt supervisor."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.shunt_refinement_check import build_config, json_data, summarise
from experiments.helpers.records import ContingencyEvent
from experiments.runners.multi_tso_dso import run_multi_tso_dso


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # "fleet" is "regular" with an honest name: both subordinate support
    # weights zero, so only the regular path and the supervisory TS fleet
    # reserve can act (2026-10-21).
    ap.add_argument("--cases", nargs="+",
                    choices=["legacy", "regular", "hybrid", "slow", "fleet"],
                    default=["legacy", "regular", "hybrid", "slow"])
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--paramset", default="ch9_selected_pcc27_gqceil_comp_q_off")
    ap.add_argument("--frozen-network", action="store_true")
    # Parked by default (2026-10-20): the verification at 20 showed the regular
    # path buys TS voltage with interface-Q and STS reserve, so it must be
    # un-parked explicitly. Admission needs |grad_f| per level above this.
    ap.add_argument("--movement-weight", type=float, default=3600.)
    # None keeps the parameter set's shunt_int_g_w (0.06 in the selected set).
    # It used to default to 0.01 and silently override it (fixed 2026-10-21).
    ap.add_argument("--slow-weight", type=float, default=None)
    ap.add_argument("--voltage-error-allowance-pu", type=float, default=.001)
    # Objective weights of the selected set are g_q = 0 and both TS reserve
    # terms 0, so a bank step is priced on its voltage effect alone
    # (2026-10-21). These override them for the fleet-relief screen.
    ap.add_argument("--g-q", type=float, default=None,
                    help="TS interface-Q tracking weight; None keeps the paramset value")
    ap.add_argument("--g-res-sg", type=float, default=None,
                    help="TS synchronous-generator reactive-reserve weight")
    ap.add_argument("--g-res-der", type=float, default=None,
                    help="TS DER reactive-reserve weight")
    ap.add_argument("--support-gate", default="voltage_allowance",
                    choices=["voltage_allowance", "shared_objective"])
    # Zero admits support only when the shared objective does not worsen.
    ap.add_argument("--support-exchange", type=float, default=0.)
    # Supervisory TS fleet reserve: seen by the bank decision only.
    ap.add_argument("--ts-reserve-weight", type=float, default=0.)
    ap.add_argument("--ts-reserve-start", type=float, default=.8)
    # Fleet-stress scenario: replace the 150 Mvar connect/trip pair with ONE
    # sustained reactive load, so the zone's machines are driven up their Q
    # curve and stay there instead of being relieved at minute 30.
    ap.add_argument("--stress-q-mvar", type=float, default=None)
    ap.add_argument("--stress-bus", type=int, default=11)
    ap.add_argument("--stress-minute", type=float, default=10.)
    # Generator outage: the remaining machines of that zone carry its reactive
    # duty, which shrinks their margin where reactive loading could not —
    # loading sags the voltage and moves the capability box with it.
    ap.add_argument("--trip-gen", type=int, default=None,
                    help="net.gen row index to trip; zone 3 holds 2 (G4) and 5 (G7)")
    # The profile hour sets the TS DER active power and therefore, through the
    # VDE dead zone below P/S_n = 0.1, their reactive capability. At
    # 2016-11-26 10:00 the whole 2198 MVA of TS wind parks can supply nothing.
    ap.add_argument("--start-time", default=None,
                    help="profile start, e.g. '2016-11-26 10:00'")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for case in args.cases:
        cfg = build_config("legacy" if case == "legacy" else "reserve",
            args.seconds, args.paramset, True, args.frozen_network)
        if case != "legacy":
            cfg = replace(cfg, shunt_dispatch="hybrid", shunt_sts_support=True,
                g_w_tso_shunt=1e6 if case == "slow" else args.movement_weight,
                **({} if args.slow_weight is None
                   else {"shunt_int_g_w": args.slow_weight}),
                shunt_hybrid_voltage_error_allowance_pu=args.voltage_error_allowance_pu,
                shunt_hybrid_support_gate=args.support_gate,
                shunt_hybrid_support_exchange=args.support_exchange,
                shunt_ts_reserve_weight=args.ts_reserve_weight,
                shunt_ts_reserve_start=args.ts_reserve_start,
                shunt_sts_tracking_weight=0. if case in ("regular", "fleet") else 10.,
                shunt_sts_reserve_weight=0. if case in ("regular", "fleet") else 100.)
        for name, value in (("g_q", args.g_q), ("tso_g_res_sg", args.g_res_sg),
                            ("tso_g_res_der", args.g_res_der)):
            if value is not None:
                cfg = replace(cfg, **{name: value})
        if args.start_time is not None:
            import pandas as pd
            cfg = replace(cfg, start_time=pd.Timestamp(args.start_time))
        if args.trip_gen is not None:
            cfg = replace(cfg, contingencies=[ContingencyEvent(
                minute=args.stress_minute, element_type="gen",
                element_index=args.trip_gen, action="trip")])
        elif args.stress_q_mvar is not None:
            cfg = replace(cfg, contingencies=[ContingencyEvent(
                minute=args.stress_minute, element_type="load",
                bus=args.stress_bus, p_mw=0, q_mvar=args.stress_q_mvar,
                action="connect")])
        d = args.output/case
        d.mkdir(parents=True, exist_ok=False)
        (d/"config.json").write_text(json.dumps(json_data(asdict(cfg)), indent=2, default=str))
        files = ["controller/shunt_hybrid.py", "controller/base_controller.py",
            "controller/shunt_integrator.py", "controller/shunt_support.py",
            "optimisation/miqp_solver.py", "experiments/helpers/shunt_hybrid.py",
            "experiments/runners/multi_tso_dso.py", "configs/config.py",
            "experiments/shunt_hybrid_check.py"]
        hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
        (d/"source_hashes.json").write_text(json.dumps(hashes, indent=2))
        print(f"Starting {case}, {args.seconds/60:g} simulated minutes", flush=True)
        start = time.perf_counter()
        with (d/"run.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            records = run_multi_tso_dso(cfg)
        result = summarise(records)
        result["wall_time_s"] = time.perf_counter()-start
        result["joint_candidate_solves"] = sum(diag.get("candidate_solves", 0)
            for r in records for diag in r.zone_tso_shunt_hybrid.values())
        (d/"summary.json").write_text(json.dumps(result, indent=2))
        (d/"records.pkl").write_bytes(pickle.dumps(records))
        print(json.dumps({k:v for k,v in result.items() if k!="commits"}), flush=True)


if __name__ == "__main__":
    main()
