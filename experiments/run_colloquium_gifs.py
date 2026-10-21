"""Reproducible four-hour QSS runs for the research-colloquium animations.

The physics and controller implementations remain in the shared runner.
Only the presentation horizon, headless display flags and established S1/O3
architecture overlays differ from the frozen selected configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from configs.paramsets import load_paramset
from experiments.ch_9_parameter_selection._ch9_selected_design import build_selected_config
from experiments.ch_10_case_study.ch_10_1_variant_ladder import VARIANTS
from experiments.results_io import new_run_dir, _jsonable
from experiments.runners import run_multi_tso_dso


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False),
                          encoding="utf-8")


def prepare(trip_gen=None, cases=("pi", "ofo")):
    base, provenance = build_selected_config(paramset="ch9_selected",
                                             apply_frozen_network=True)
    if trip_gen is not None:
        # Move the scheduled generator outage (trip and restoration) to another
        # generator; timing and all other events stay unchanged.
        gen_events = [e for e in base.contingencies if e.element_type == "gen"]
        if len(gen_events) != 2 or len({e.element_index for e in gen_events}) != 1:
            raise ValueError(f"Expected one generator trip/restore pair: {gen_events}")
        base = replace(base, contingencies=[
            replace(e, element_index=trip_gen) if e.element_type == "gen" else e
            for e in base.contingencies
        ])
    configs = {
        name: replace(base, **VARIANTS[rung], n_total_s=14400.0,
                      shunt_int_t_dwell_s=3 * 3600.0,
                      verbose=1, run_stability_analysis=False)
        for name, rung in (("pi", "S1"), ("ofo", "O3")) if name in cases
    }
    fields = load_paramset("ch9_selected")["fields"]
    for name, cfg in configs.items():
        for field, value in fields.items():
            if getattr(cfg, field) != value:
                raise ValueError(f"{name}: selected parameter changed: {field}")
    run = new_run_dir("research_colloquium_gifs", subdirs=())
    for name, cfg in configs.items():
        folder = run.root / name
        folder.mkdir()
        cfg.result_dir = str(folder)
        with (folder / "config.pkl").open("wb") as f:
            pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(folder / "config.json", cfg)
    write_json(run.root / "design.json", {
        "created": datetime.now().astimezone().isoformat(),
        "design": provenance, "parameter_set": load_paramset("ch9_selected"),
        "duration_s": 14400,
        "shunt_dwell_s": 10800,
        "shunt_dwell_reason": "Author corrected the colloquium requirement to 3 h on 2026-09-16; original run inherited 1800 s.",
        "network": "rural_700; DSO_3 DER and P-load x2",
        "pi_definition": "S1: PI RVR/RPR, voltage-priority safeguard; fixed machine taps; local STS Q(V)/OLTC",
        "ofo_definition": "O3: TS+STS OFO cascade, inherited SBX-H, machine OLTCs and integrator MSC/MSR",
        "scope": "Full architecture comparison, not an isolated control-law comparison.",
        "shunt_wear": "Measured and reported; not bounded by the selected parameter set.",
        "cases": list(configs),
        "generator_outage_index": trip_gen,
        "fired_events": [e for e in base.contingencies if e.minute <= 240],
        "outside_horizon": [e for e in base.contingencies if e.minute > 240],
    })
    print(run.root, flush=True)


def run_case(folder):
    folder = Path(folder).resolve()
    with (folder / "config.pkl").open("rb") as f:
        cfg = pickle.load(f)
    if (folder / "records.pkl").exists():
        raise FileExistsError("Refusing to overwrite completed simulation records")
    start = time.perf_counter()
    write_json(folder / "status.json", {"status": "running", "started": datetime.now().astimezone().isoformat()})

    def metadata_hook(state):
        net = state["net"]
        zones = {}
        for z, zd in state["zone_defs"].items():
            zones[z] = {
                "voltage_buses": list(zd.v_bus_indices),
                "generators": list(zd.gen_indices),
                "der": list(zd.tso_der_indices),
                "details": _jsonable(zd),
            }
        dsos = {k: _jsonable(v.config) for k, v in state["dso_controllers"].items()}
        write_json(folder / "network_metadata.json", {
            "zones": zones, "dso_controllers": dsos,
            "dso_to_tso": state["dso_to_tso_id"],
            "gen": net.gen.reset_index().to_dict(orient="records"),
            "sgen": net.sgen.reset_index().to_dict(orient="records"),
            "trafo": net.trafo.reset_index().to_dict(orient="records"),
            "trafo3w": net.trafo3w.reset_index().to_dict(orient="records"),
        })
        return False

    with (folder / "simulation.log").open("w", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                records = run_multi_tso_dso(cfg, pre_loop_hook=metadata_hook)
                expected = int(round(cfg.n_total_s / cfg.dt_s))
                times = np.array([r.time_s for r in records])
                if len(records) != expected or not np.allclose(
                    times, np.arange(1, expected + 1) * cfg.dt_s
                ):
                    raise ValueError(f"Incomplete timeline: {len(records)} records")
                with (folder / "records.pkl").open("wb") as f:
                    pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
                statuses = {
                    "tso": dict(Counter(str(v) for r in records for v in r.zone_tso_status.values())),
                    "dso": dict(Counter(str(v) for r in records for v in r.dso_status.values())),
                }
                write_json(folder / "status.json", {
                    "status": "complete", "records": len(records),
                    "first_time_s": float(times[0]), "last_time_s": float(times[-1]),
                    "wall_s": time.perf_counter() - start, "solver_statuses": statuses,
                    "records_sha256": hashlib.sha256((folder / "records.pkl").read_bytes()).hexdigest(),
                })
            except BaseException as exc:
                traceback.print_exc()
                write_json(folder / "status.json", {
                    "status": "failed", "error": repr(exc),
                    "wall_s": time.perf_counter() - start,
                })
                raise
    print(f"Completed {folder}: {len(records)} records", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--trip-gen", type=int,
                        help="With --prepare: generator index for the 30-180 min outage")
    parser.add_argument("--cases", default="pi,ofo",
                        help="With --prepare: comma-separated subset of pi,ofo")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.trip_gen, tuple(args.cases.split(",")))
    elif args.case_dir:
        run_case(args.case_dir)
    else:
        parser.error("choose --prepare or --case-dir")


if __name__ == "__main__":
    main()
