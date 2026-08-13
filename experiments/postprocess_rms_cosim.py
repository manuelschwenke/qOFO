#!/usr/bin/env python3
"""Complete an interrupted RMS replay from its retained ComRes CSV.

This does not initialise or run the RMS simulation.  PowerFactory is opened
only to recover the registered monitor-object catalogue; all result values are
loaded from ``csv/rms_comres_full.csv`` with pandas/NumPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.gate_e_diagnostics import (  # noqa: E402
    load_run as load_gate_e_run,
    plot_actuators,
    plot_dso_voltages,
    plot_zone_voltages,
    tap_divergence_table as actuator_divergence_table,
)
from experiments.helpers.rms_replay import (  # noqa: E402
    endpoint_comparison,
    endpoint_summary,
    interval_settling_table,
    plot_controlled_output_overlays,
    rms_controlled_trajectories,
    settling_summary,
    static_controlled_trajectories,
    trajectories_long_frame,
)
from pf.result_export import load_comres_trajectories  # noqa: E402
from pf.screening import monitored_outputs  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH, connect  # noqa: E402


DT_S = 20.0
TSO_PERIOD_S = 180.0


def _snapshot(run_dir: Path) -> dict:
    preferred = run_dir / "snapshot" / "gate_e_post_init.json"
    if preferred.is_file():
        path = preferred
    else:
        candidates = [
            path for path in (run_dir / "snapshot").glob("*.json")
            if "roundtrip" not in path.name and "sync" not in path.name
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"cannot identify one replay snapshot in {run_dir / 'snapshot'}"
            )
        path = candidates[0]
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _quantity_endpoint_summary(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for kind, group in frame.groupby("signal_type"):
        error = group["error"].to_numpy(dtype=float)
        rows.append(
            {
                "signal_type": str(kind),
                "n": int(len(error)),
                "rmse": float((error @ error / len(error)) ** 0.5),
                "mae": float(abs(group["error"]).mean()),
                "max_abs": float(group["abs_error"].max()),
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    if args.stride < 1:
        raise ValueError("stride must be >= 1")

    started = perf_counter()
    run_dir = args.run_dir.resolve()
    csv_dir = run_dir / "csv"
    figure_dir = run_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    bulk_csv = csv_dir / "rms_comres_full.csv"
    if not bulk_csv.is_file():
        raise FileNotFoundError(bulk_csv)

    with (run_dir / "static_records.pkl").open("rb") as handle:
        static_log = pickle.load(handle)
    with (run_dir / "rms_records.pkl").open("rb") as handle:
        rms_log = pickle.load(handle)
    if not rms_log:
        raise ValueError("saved RMS record list is empty")
    duration_s = float(rms_log[-1].time_s)
    snapshot_doc = _snapshot(run_dir)

    print("[postprocess] loading PowerFactory monitor catalogue")
    app = connect(args.project, study_case="02_RMS_CoSim")
    monitors = monitored_outputs(app, include_der=True)
    keep = lambda label: label.startswith(
        ("qSTS_", "u_", "qDER_", "uDER_")
    )
    bulk_raw = load_comres_trajectories(
        bulk_csv,
        monitors,
        since_s=0.0,
        stride=args.stride,
        labels=keep,
    )
    final_times = {float(time[-1]) for time, _ in bulk_raw.values()}
    if final_times != {duration_s}:
        raise ValueError(
            f"ComRes final time {sorted(final_times)} != records {duration_s}"
        )

    raw = {
        label: trajectory
        for label, trajectory in bulk_raw.items()
        if label.startswith(("qSTS_", "u_"))
    }
    der_raw = {
        label: trajectory
        for label, trajectory in bulk_raw.items()
        if label.startswith(("qDER_", "uDER_"))
    }
    trajectories_long_frame(raw).to_csv(
        csv_dir / "rms_monitors_raw.csv", index=False
    )
    trajectories_long_frame(der_raw).to_csv(
        csv_dir / "rms_der_raw.csv", index=False
    )

    static_traj = static_controlled_trajectories(static_log)
    rms_traj = rms_controlled_trajectories(raw, snapshot_doc)
    trajectories_long_frame(static_traj).to_csv(
        csv_dir / "static_controlled_outputs.csv", index=False
    )
    trajectories_long_frame(rms_traj).to_csv(
        csv_dir / "rms_controlled_outputs.csv", index=False
    )

    endpoint = endpoint_comparison(static_traj, rms_traj)
    endpoint.to_csv(csv_dir / "endpoint_comparison.csv", index=False)
    endpoint_summary(endpoint).to_csv(
        csv_dir / "endpoint_summary_by_signal.csv", index=False
    )

    settling = interval_settling_table(
        rms_traj,
        interval_s=DT_S,
        total_s=duration_s,
    )
    settling.to_csv(csv_dir / "settling_per_interval.csv", index=False)
    settling_signal, settling_type = settling_summary(settling)
    settling_signal.to_csv(
        csv_dir / "settling_summary_by_signal.csv", index=False
    )
    settling_type.to_csv(
        csv_dir / "settling_summary_by_quantity.csv", index=False
    )

    plot_controlled_output_overlays(
        static_traj,
        rms_traj,
        figure_dir,
        dt_s=DT_S,
        tso_period_s=TSO_PERIOD_S,
        write_pdf=not args.no_pdf,
    )
    diagnostics = load_gate_e_run(run_dir)
    plot_actuators(diagnostics, figure_dir)
    plot_zone_voltages(diagnostics, figure_dir)
    plot_dso_voltages(diagnostics, figure_dir)
    actuator_divergence_table(diagnostics).to_csv(
        csv_dir / "actuator_divergence.csv", index=False
    )

    recovered = {
        "postprocessing": "recovered from retained ComRes CSV; no RMS rerun",
        "source_csv": str(bulk_csv),
        "duration_s": duration_s,
        "stride": int(args.stride),
        "static_records": int(len(static_log)),
        "rms_records": int(len(rms_log)),
        "controlled_monitor_signals": int(len(raw)),
        "der_diagnostic_signals": int(len(der_raw)),
        "settling_all_intervals": bool(
            len(settling) > 0
            and settling["settled_within_interval"].astype(bool).all()
        ),
        "settling_by_quantity": settling_type.to_dict(orient="records"),
        "endpoint_error_by_quantity": _quantity_endpoint_summary(endpoint),
        "elapsed_s": float(perf_counter() - started),
    }
    with (run_dir / "postprocess_recovery.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(recovered, handle, indent=2, allow_nan=False)
    print(
        f"[postprocess] complete in {recovered['elapsed_s']:.1f} s; "
        f"{len(raw)} controlled + {len(der_raw)} DER diagnostic signals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
