#!/usr/bin/env python3
"""Phase 6 Gate E closed-loop static-versus-RMS replay.

The experiment holds every exogenous injection fixed for 900 s, runs the
authoritative ``run_multi_system_ofo`` controller configuration twice (static
and PowerFactory RMS plants), and evaluates the RMS response inside each 20 s
STS dispatch window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter
from typing import Any, Dict

# Force UTF-8 before project imports can wrap the console streams with Colorama.
# This keeps Unicode controller diagnostics from failing on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MultiTSOConfig  # noqa: E402
from pf.screening import RMS_STEP_MS  # noqa: E402
from experiments.helpers.rms_cosim_config import (  # noqa: E402
    DT_S,
    TSO_PERIOD_S,
    CoSimSpecification,
    ReplaySpecification,
    add_common_arguments,
    apply_cli_overrides,
    make_cosim_config,
    make_gate_e_config,
    validate_duration as _validate_duration,
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
from analysis.gate_e_diagnostics import (  # noqa: E402
    load_run as load_gate_e_run,
    plot_actuators,
    plot_dso_voltages,
    plot_zone_voltages,
    tap_divergence_table as actuator_divergence_table,
)
from experiments.results_io import new_run_dir  # noqa: E402
from experiments.run_multi_system_ofo import (  # noqa: E402
    make_config as make_reference_config,
)
from experiments.runners import run_multi_tso_dso  # noqa: E402
from pf.replay import PowerFactoryReplayFactory  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH  # noqa: E402


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' tabulate extra."""

    if frame.empty:
        return "(no rows)"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", r"\|").replace("\n", " ")

    headers = [render(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _type_endpoint_summary(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
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


def _write_summary(
    run_root: Path,
    *,
    duration_s: float,
    n_static: int,
    n_rms: int,
    config: MultiTSOConfig,
    plant,
    settling: pd.DataFrame,
    settling_by_type: pd.DataFrame,
    endpoint: pd.DataFrame,
) -> bool:
    gate_ok = bool(
        len(settling) > 0
        and settling["settled_within_interval"].astype(bool).all()
    )
    qv_equivalent = bool(
        getattr(plant, "der_qv_local_control_equivalent", False)
    )
    validation_ok = gate_ok and qv_equivalent
    validation_verdict = (
        "PASS" if validation_ok else
        ("FAIL_SETTLING" if qv_equivalent else "BLOCKED_DER_QV_MISMATCH")
    )
    skipped_unique = sorted(
        {
            (kind, int(index), float(value))
            for kind, index, value in plant.skipped_writes
        }
    )
    event_pool_stats: Dict[str, int] = {}
    event_folder_count = None
    plant_ctx = getattr(plant, "ctx", None)
    if plant_ctx is not None and hasattr(plant_ctx, "event_pool_stats"):
        event_pool_stats = dict(plant_ctx.event_pool_stats())
        event_folder_count = len(list(plant_ctx.evt_folder.GetContents()))
    summary = {
        "gate_e_validation_verdict": validation_verdict,
        "gate_e_settling_verdict": "PASS" if gate_ok else "FAIL",
        # Recorded so a run made under an artificial DER capability can
        # never be mistaken for one made under the real operating diagrams.
        "der_q_capability_override_pu": (
            None if getattr(config, "der_q_capability_override_pu", None) is None
            else float(config.der_q_capability_override_pu)
        ),
        "der_qv_local_control_equivalent": qv_equivalent,
        "comparison_validity": (
            "VALID" if qv_equivalent else "PROVISIONAL_NON_EQUIVALENT_PLANT_LAW"
        ),
        "duration_s": float(duration_s),
        "dispatch_windows": int(round(duration_s / DT_S)),
        "static_records": int(n_static),
        "rms_records": int(n_rms),
        "rms_final_time_s": float(plant.t),
        "rms_event_pool": event_pool_stats,
        "rms_event_folder_count": event_folder_count,
        "configuration_source": "experiments.run_multi_system_ofo.make_config",
        "coordination_mode": str(config.coordination_mode),
        "tso_tertiary_shunts_installed": bool(
            config.install_tso_tertiary_shunts
        ),
        "shunt_dispatch": str(config.shunt_dispatch),
        "settling_by_quantity": settling_by_type.to_dict(orient="records"),
        "endpoint_error_by_quantity": _type_endpoint_summary(endpoint),
        "missing_avr_write_count": int(len(plant.skipped_writes)),
        "missing_avr_unique_writes": [
            {"kind": kind, "index": index, "value": value}
            for kind, index, value in skipped_unique
        ],
    }
    with (run_root / "gate_e_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    lines = [
        "# Phase 6 Gate E closed-loop replay",
        "",
        f"Gate E validation verdict: **{validation_verdict}**.",
        f"Settling-only verdict: **{summary['gate_e_settling_verdict']}**.",
        "",
        (
            "**Plant law:** both plants apply the re-anchored Q(V) "
            "characteristic -- the RMS parks via a `QVPRE` DSL block in the "
            "WECC composite's Plant Control slot, driven by dispatch-time "
            "`qset`/`Vanchor` parameter events. Endpoint errors are therefore "
            "an equivalent-plant comparison."
            if qv_equivalent else
            "**Validity warning:** the RMS DERs hold `REEC_D.Qext` constant "
            "between dispatches, whereas the static plant applies the "
            "re-anchored Q(V) characteristic. Endpoint errors and overlay "
            "figures are therefore diagnostic, not an equivalent-plant "
            "validation."
        ),
        "",
        "## Assumptions and constraints",
        "",
        # Stated from the config, not asserted: this line claimed "profiles and
        # contingencies are disabled" unconditionally, which has been false for
        # every profiled run since 2026-07-21 and would be false for every N-1
        # run. A report that misdescribes its own disturbance is worse than no
        # report.
        "- Exogenous drive: " + (
            (f"profiles ON over the full {duration_s:g} s"
             if getattr(config, "use_profiles", False) else
             f"load and active-power injections fixed for the full "
             f"{duration_s:g} s")
            + (("; contingencies: " + ", ".join(
                f"{e.action} {e.element_type}[{e.element_index}] "
                f"at t={e.effective_time_s:g}s"
                for e in (getattr(config, "contingencies", None) or [])))
               if getattr(config, "contingencies", None) else
               "; no contingencies")
            + (f"; exogenous load step at t="
               f"{getattr(config, 'load_step_time_s'):g}s"
               if getattr(config, "load_step_time_s", None) is not None
               else "")
        ) + ".",
        "- Measurement noise is disabled, so the comparison isolates the plant "
        "model and closed-loop dynamics.",
        "- Controllers see only their cached sensitivities and their own "
        "plant's measurements; no controller is given the RMS equations.",
        "- The quasi-static voltage-reachability guard is disabled because it "
        "would solve a second plant behind the RMS adapter.",
        "",
        "## Actuators and controlled outputs",
        "",
        "- Actuators: DER reactive-power references, synchronous-machine AVR "
        "references, 2W/3W OLTC taps, and the reference configuration's "
        "tertiary MSC/MSR banks (integrator dispatch).",
        "- Controlled outputs evaluated here: all 12 TS-STS interface "
        "reactive-power flows and mean voltage over each TS zone's TN PQ buses.",
        "",
        "## Settling statistics",
        "",
        _markdown_table(settling_by_type),
        "",
        "The 2% band uses absolute floors of 1 Mvar for interface Q and "
        "0.001 pu for voltage. An interval is failed if its final sample "
        "remains outside the band.",
        "",
        "## Static-equilibrium versus RMS endpoint errors",
        "",
    ]
    endpoint_type = pd.DataFrame.from_records(
        summary["endpoint_error_by_quantity"]
    )
    lines.append(_markdown_table(endpoint_type))
    # Known model limitations are emitted only when they actually apply to this
    # run.  Historically these two bullets were hardcoded and became stale once
    # the QVPRE Q(V) law and the G 01 AVR landed -- a PASS run then still read
    # "Gate E remains blocked" / "G 01 has no AVR block", contradicting its own
    # verdict.  Derive them from run state instead.
    limitations = []
    if not qv_equivalent:
        limitations.append(
            "- DER Q(V) actuator-law equivalence is not active in this run: "
            "the RMS parks hold `REEC_D.Qext` constant between dispatches, so "
            "the endpoint comparison is diagnostic, not an equivalent-plant "
            "validation."
        )
    n_skipped = len(plant.skipped_writes)
    if n_skipped:
        limitations.append(
            f"- {n_skipped} generator-voltage write(s) were skipped: the "
            "target machine has no AVR block in the adopted dynamic template "
            "(e.g. a network equivalent). This is an actuator-set mismatch, "
            "not a settling failure, and is retained in the JSON provenance."
        )
    if limitations:
        lines.extend(["", "## Known model limitation", ""] + limitations + [""])
    (run_root / "gate_e_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return validation_ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RMS co-simulation vs quasi-static (QSS) comparison")
    add_common_arguments(parser)
    parser.add_argument(
        "--rms-step-ms", type=float, default=None,
        help="RMS integration step [ms] (default 10). With --adaptive-step "
             "this is the SMALLEST step PowerFactory may take.")
    parser.add_argument(
        "--rms-step-max-ms", type=float, default=None,
        help="largest step [ms] with --adaptive-step (default 10).")
    parser.add_argument(
        "--adaptive-step", action="store_true",
        help="enable PF automatic step-size adaptation (ComInc iopt_adapt). "
             "The converter models carry dynamics far faster than the 10 ms "
             "default, so a fixed step under-integrates them; adaptation "
             "shortens the step only where the error tolerance demands it. "
             "Off reproduces every run before 2026-08-06.")
    parser.add_argument(
        "--qv-deadband-at-contingency", type=float, default=None,
        help="dead-band x droop study only: Q(V) dead band [pu] installed on "
             "the RMS parks at the instant of the contingency; the run-up "
             "keeps --tso-deadband/--dso-deadband. Use e.g. 0.5 for the "
             "no-droop leg and omit it for the droop leg, holding "
             "--tso-deadband/--dso-deadband IDENTICAL on both -- those also "
             "feed the controllers and the static plant, so separating the "
             "legs through them makes the closed loops diverge from t=0. The "
             "STATIC plant is unaffected, so Gate E cannot certify such a "
             "run.")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)

    _validate_duration(args.duration)
    if args.stride < 1:
        raise ValueError("stride must be >= 1")
    args.project = args.project or DEFAULT_PROJECT_PATH

    # Connect and raise the desktop up front only when it is wanted from the
    # start: App.Show() takes 20-30 s and the desktop cannot paint while the
    # engine is busy, so it is raised here (during the pure-pandapower static
    # leg) rather than next to the RMS build.  Engine mode permits one
    # session, so the handle is handed to the factory.
    gui_app = None
    if args.show_gui:
        from pf.session import connect as _pf_connect
        from pf.replay import show_desktop
        gui_app = _pf_connect(args.project, study_case="02_RMS_CoSim")
        show_desktop(gui_app)

    static_cfg = make_cosim_config(args.duration, verbose=args.verbose)
    rms_cfg = make_cosim_config(args.duration, verbose=args.verbose)
    apply_cli_overrides(args, (static_cfg, rms_cfg))
    if args.no_qv_seed:
        static_cfg.disable_qv_seed = True
        print("  [option2] static plant seed_qv_equilibrium DISABLED")

    spec = ReplaySpecification(
        static_cfg, rms_cfg,
        # Solver + experiment switches that change the numerics or the
        # meaning of the run and cannot be recovered from the trace.
        rms_step_ms=(RMS_STEP_MS if args.rms_step_ms is None
                     else float(args.rms_step_ms)),
        rms_step_max_ms=(None if args.rms_step_max_ms is None
                         else float(args.rms_step_max_ms)),
        adaptive_step=bool(args.adaptive_step),
        qv_deadband_at_contingency=args.qv_deadband_at_contingency,
    )
    run_dir = new_run_dir(
        "rms_phase6_replay",
        spec,
        subdirs=("figures", "csv", "snapshot"),
    )
    static_cfg.result_dir = str(run_dir.root)
    rms_cfg.result_dir = str(run_dir.root)
    print(f"[Gate E] results -> {run_dir.root}")

    print("\n[Gate E] static closed-loop reference")
    static_log = run_multi_tso_dso(static_cfg)
    with (run_dir.root / "static_records.pkl").open("wb") as handle:
        pickle.dump(static_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n[Gate E] PowerFactory RMS closed-loop replay")
    factory = PowerFactoryReplayFactory(
        out_dir=run_dir.snapshot,
        project=args.project,
        on_missing_avr="skip",
        distributed_slack=rms_cfg.distributed_slack,
        enforce_q_lims=rms_cfg.enforce_q_lims_plant,
        # One pre-created (before-ComInc) slot PER FIRING per target: each DSO
        # dispatch fires one qset/Vanchor/load event per interval, so the pool
        # must hold n_intervals (+margin) slots or it grows mid-run -- and PF
        # only admits a couple of mid-run-created events before firing dies
        # (validated 2026-07-23: default of 1 slot froze every actuator at
        # ~t=41 s).  Pre-created slots are known to ComInc and fire reliably
        # across intervals (probe_event_preallocation_volume PASS).
        event_pool_slots=int(round(args.duration / DT_S)) + 5,
        preallocate_profiles=bool(
            args.profiles and args.profile_delivery == "events"
        ),
        profile_delivery=(args.profile_delivery if args.profiles else "events"),
        qv_deadband_at_contingency=args.qv_deadband_at_contingency,
        **({} if args.rms_step_ms is None
           else {"rms_step_ms": float(args.rms_step_ms)}),
        **({} if args.rms_step_max_ms is None
           else {"rms_step_max_ms": float(args.rms_step_max_ms)}),
        adaptive_step=bool(args.adaptive_step),
        # GUI machinery is always installed so the run directory's
        # show_gui.bat / hide_gui.bat work; ``start_hidden`` decides only
        # whether the desktop is up from the beginning.
        show_gui=True,
        start_hidden=not args.show_gui,
        live_plot=args.live_plot,
        gui_off_flag=(run_dir.root / "DISABLE_GUI"),
        gui_refresh_every=args.gui_refresh_every,
        app_handle=gui_app,
    )
    rms_log = run_multi_tso_dso(rms_cfg, plant_factory=factory)
    with (run_dir.root / "rms_records.pkl").open("wb") as handle:
        pickle.dump(rms_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if factory.plant is None or factory.snapshot_doc is None:
        raise RuntimeError("RMS runner returned without constructing its plant")
    if abs(factory.plant.t - args.duration) > 1e-6:
        raise RuntimeError(
            f"RMS plant ended at t={factory.plant.t}, expected {args.duration}"
        )
    _pool = factory.plant.ctx.event_pool_stats()
    _folder_count = len(list(factory.plant.ctx.evt_folder.GetContents()))
    print(
        f"  [events] folder={_folder_count}; "
        f"pool param {_pool['param_used']}/{_pool['param_total']}, "
        f"load {_pool['load_used']}/{_pool['load_total']}"
    )

    print("  [results] bulk-exporting ElmRes through ComRes")
    _bulk_started = perf_counter()
    bulk_raw = factory.plant.harvest_trajectories_bulk(
        run_dir.csv / "rms_comres_full.csv",
        since_s=0.0,
        stride=args.stride,
        labels=lambda label: label.startswith(
            ("qSTS_", "u_", "qDER_", "uDER_")
        ),
    )
    print(
        "  [results] ComRes export + pandas/NumPy sampling "
        f"completed in {perf_counter() - _bulk_started:.1f} s"
    )
    raw = {
        label: trajectory
        for label, trajectory in bulk_raw.items()
        if label.startswith(("qSTS_", "u_"))
    }
    raw_long = trajectories_long_frame(raw)
    raw_long.to_csv(run_dir.csv / "rms_monitors_raw.csv", index=False)

    # Per-park Q/V: the diagnostic that separates a capability clip (park Q
    # pinned at a limit) from droop amplification of a voltage difference.
    der_raw = {
        label: trajectory
        for label, trajectory in bulk_raw.items()
        if label.startswith(("qDER_", "uDER_"))
    }
    if der_raw:
        trajectories_long_frame(der_raw).to_csv(
            run_dir.csv / "rms_der_raw.csv", index=False
        )

    static_traj = static_controlled_trajectories(static_log)
    rms_traj = rms_controlled_trajectories(raw, factory.snapshot_doc)
    trajectories_long_frame(static_traj).to_csv(
        run_dir.csv / "static_controlled_outputs.csv", index=False
    )
    trajectories_long_frame(rms_traj).to_csv(
        run_dir.csv / "rms_controlled_outputs.csv", index=False
    )

    endpoint = endpoint_comparison(static_traj, rms_traj)
    endpoint.to_csv(run_dir.csv / "endpoint_comparison.csv", index=False)
    endpoint_summary(endpoint).to_csv(
        run_dir.csv / "endpoint_summary_by_signal.csv", index=False
    )

    settling = interval_settling_table(
        rms_traj,
        interval_s=DT_S,
        total_s=args.duration,
    )
    settling.to_csv(run_dir.csv / "settling_per_interval.csv", index=False)
    settling_signal, settling_type = settling_summary(settling)
    settling_signal.to_csv(
        run_dir.csv / "settling_summary_by_signal.csv", index=False
    )
    settling_type.to_csv(
        run_dir.csv / "settling_summary_by_quantity.csv", index=False
    )

    plot_controlled_output_overlays(
        static_traj,
        rms_traj,
        run_dir.figures,
        dt_s=DT_S,
        tso_period_s=TSO_PERIOD_S,
        write_pdf=not args.no_pdf,
    )

    # Discrete-actuator and per-bus diagnostics.  These live in
    # analysis/gate_e_diagnostics.py and were originally run by hand, which
    # meant a run could finish with no actuator figures at all -- exactly
    # what happened to run 0018.  Wrapped so a plotting failure cannot
    # discard an otherwise complete run.
    try:
        diag = load_gate_e_run(run_dir.root)
        plot_actuators(diag, run_dir.figures)
        plot_zone_voltages(diag, run_dir.figures)
        plot_dso_voltages(diag, run_dir.figures)
        actuator_divergence_table(diag).to_csv(
            run_dir.csv / "actuator_divergence.csv", index=False
        )
    except Exception as exc:                       # noqa: BLE001
        print(f"  [warn] actuator/voltage diagnostics failed: "
              f"{type(exc).__name__}: {exc}")

    gate_ok = _write_summary(
        run_dir.root,
        duration_s=args.duration,
        n_static=len(static_log),
        n_rms=len(rms_log),
        config=rms_cfg,
        plant=factory.plant,
        settling=settling,
        settling_by_type=settling_type,
        endpoint=endpoint,
    )
    print(
        f"[Gate E] {'PASS' if gate_ok else 'FAIL'} -> "
        f"{run_dir.root / 'gate_e_summary.md'}"
    )
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
