#!/usr/bin/env python3
"""Run the PowerFactory RMS co-simulation alone (no QSS reference).

Same controller stack, same plant, same artefacts as
``run_comparison_rms_cosim_qss.py`` -- it simply skips the quasi-static leg
and the static-versus-RMS comparison.  Roughly halves the wall time, which is
what you want for exploratory runs where the QSS reference is not needed.

Use ``run_comparison_rms_cosim_qss.py`` when you want the comparison, and
``run_openloop_qss_to_rms.py`` for the open-loop ``u -> y`` plant-equivalence
test (that one IS a replay: it applies the QSS run's recorded actuator
timeline to the RMS plant).

Author: Manuel Schwenke / Claude Code (2026-07-31)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.helpers.rms_cosim_config import (  # noqa: E402
    DT_S,
    CoSimSpecification,
    add_common_arguments,
    apply_cli_overrides,
    make_cosim_config,
    validate_duration,
)
from experiments.helpers.rms_replay import (  # noqa: E402
    rms_controlled_trajectories,
    trajectories_long_frame,
)
from experiments.results_io import new_run_dir  # noqa: E402
from experiments.runners import run_multi_tso_dso  # noqa: E402
from pf.replay import PowerFactoryReplayFactory  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="PowerFactory RMS co-simulation (no QSS comparison)")
    add_common_arguments(parser)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)

    validate_duration(args.duration)
    if args.stride < 1:
        raise ValueError("stride must be >= 1")
    project = args.project or DEFAULT_PROJECT_PATH

    cfg = make_cosim_config(args.duration, verbose=args.verbose)
    apply_cli_overrides(args, (cfg,))

    spec = CoSimSpecification(
        runner_static=None, runner_rms=cfg,
        comparison="RMS co-simulation only; no quasi-static reference")
    run_dir = new_run_dir("rms_cosim", spec,
                          subdirs=("figures", "csv", "snapshot"))
    cfg.result_dir = str(run_dir.root)
    print(f"[cosim] results -> {run_dir.root}")

    factory = PowerFactoryReplayFactory(
        out_dir=run_dir.snapshot,
        project=project,
        on_missing_avr="skip",
        distributed_slack=cfg.distributed_slack,
        enforce_q_lims=cfg.enforce_q_lims_plant,
        event_pool_slots=int(round(args.duration / DT_S)) + 5,
        preallocate_profiles=bool(
            args.profiles and args.profile_delivery == "events"),
        profile_delivery=(args.profile_delivery if args.profiles else "events"),
        show_gui=True,
        start_hidden=not args.show_gui,
        gui_off_flag=(run_dir.root / "DISABLE_GUI"),
        gui_refresh_every=args.gui_refresh_every,
        live_plot=args.live_plot,
    )

    print("\n[cosim] PowerFactory RMS closed loop")
    rms_log = run_multi_tso_dso(cfg, plant_factory=factory)
    with (run_dir.root / "rms_records.pkl").open("wb") as handle:
        pickle.dump(rms_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if factory.plant is None or factory.snapshot_doc is None:
        raise RuntimeError("RMS runner returned without constructing its plant")
    if abs(factory.plant.t - args.duration) > 1e-6:
        raise RuntimeError(
            f"RMS plant ended at t={factory.plant.t}, expected {args.duration}")

    pool = factory.plant.ctx.event_pool_stats()
    print(f"  [events] pool param {pool['param_used']}/{pool['param_total']}, "
          f"tap {pool['tap_used']}/{pool['tap_total']}")

    print("  [results] bulk-exporting ElmRes through ComRes")
    started = perf_counter()
    bulk_raw = factory.plant.harvest_trajectories_bulk(
        run_dir.csv / "rms_comres_full.csv",
        since_s=0.0, stride=args.stride,
        labels=lambda label: label.startswith(
            ("qSTS_", "u_", "qDER_", "uDER_")),
    )
    print(f"  [results] ComRes export completed in "
          f"{perf_counter() - started:.1f} s")

    raw = {k: v for k, v in bulk_raw.items()
           if k.startswith(("qSTS_", "u_"))}
    trajectories_long_frame(raw).to_csv(
        run_dir.csv / "rms_monitors_raw.csv", index=False)
    der_raw = {k: v for k, v in bulk_raw.items()
               if k.startswith(("qDER_", "uDER_"))}
    if der_raw:
        trajectories_long_frame(der_raw).to_csv(
            run_dir.csv / "rms_der_raw.csv", index=False)

    rms_traj = rms_controlled_trajectories(raw, factory.snapshot_doc)
    trajectories_long_frame(rms_traj).to_csv(
        run_dir.csv / "rms_controlled_outputs.csv", index=False)

    summary = {
        "experiment": "rms_cosim",
        "duration_s": float(args.duration),
        "dispatch_windows": int(round(args.duration / DT_S)),
        "rms_records": int(len(rms_log)),
        "rms_final_time_s": float(factory.plant.t),
        "scenario": str(cfg.scenario),
        "der_q_capability_override_pu": (
            None if cfg.der_q_capability_override_pu is None
            else float(cfg.der_q_capability_override_pu)),
        "dso_der_scale": dict(cfg.dso_der_scale),
        "dso_load_p_scale": dict(cfg.dso_load_p_scale),
        "der_qv_local_control_equivalent": bool(
            getattr(factory.plant, "der_qv_local_control_equivalent", False)),
        "rms_event_pool": dict(pool),
        "missing_avr_write_count": int(len(factory.plant.skipped_writes)),
    }
    with (run_dir.root / "cosim_summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2, allow_nan=False)
    print(f"[cosim] done -> {run_dir.root / 'cosim_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
