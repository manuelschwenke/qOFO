#!/usr/bin/env python3
"""
300-minute SBX-H v6 capability demonstration.

This is the single thesis-oriented experiment for the retained horizontal
ex-post remuneration mechanism. It runs one three-area TS/DS simulation
with a constant 1.03 pu planning reference and SBX-H enabled from the
converged initial operating point:

- minute 0: SBX-H contracts and the live mechanism plot start;
- minute 60: a 500 Mvar reactive sink is connected in area 3;
- minute 240: the sink is removed;
- minute 300: experiment ends after a recovery interval.

The live figure shows, for every corridor, Q_meas, the measured-P
baseline Q_0, the deadband, paid Q_sup, both terminal-voltage pairs,
scheduled voltage references, hold/violation states, and cumulative payments.

Default run:
    python experiments/archived/014_SBX_SINGLE_DEMO.py

Planning assumption:
    every area uses 1.03 pu for the full experiment; no schedule change
    or real-time support request is issued.

Headless verification with the same final figure:
    python experiments/archived/014_SBX_SINGLE_DEMO.py --no-live

A shorter horizon may be used for smoke tests; events outside that
horizon are omitted automatically.

Outputs:
    results/014_SBX_H_DEMO/constant_1p03/sbx_h_mechanism.png
    results/014_SBX_H_DEMO/constant_1p03/sbx_h_settlement_ledger.csv
    results/014_SBX_H_DEMO/constant_1p03/sbx_h_settlement_summary.md
    results/014_SBX_H_DEMO/constant_1p03/experiment_summary.md

Author: Manuel Schwenke / OpenAI Codex
Date: 2026-07-14
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.helpers.records import ContingencyEvent  # noqa: E402
from sbx_h.config import SBXConfig  # noqa: E402
from sbx_h.fail import rep1  # noqa: E402

RESULT_ROOT = REPO / "results" / "014_SBX_H_DEMO"

DEFAULT_MINUTES = 300.0
SBX_START_MIN = 0.0
STRESS_ON_MIN = 60.0
STRESS_OFF_MIN = 240.0
STRESS_BUS = 15
STRESS_Q_MVAR = 500.0
STRESSED_AREA = 3
STRESSED_AREA_V_MIN_PU = 1.00

PLANNING_V_REF_PU = 1.03
RESULT_CASE = "constant_1p03"


def make_config(
    *,
    minutes: float = DEFAULT_MINUTES,
    stress_on_min: float = STRESS_ON_MIN,
    stress_off_min: Optional[float] = STRESS_OFF_MIN,
    sink_mvar: float = STRESS_Q_MVAR,
    verbose: int = 1,
):
    """Build the constant-1.03-pu ex-post SBX-H demonstration.

    Planning schedules remain an architectural input, but this experiment
    deliberately holds every area reference at 1.03 pu for the full run.
    """
    if minutes <= SBX_START_MIN:
        rep1(
            "experiment horizon must extend beyond the SBX-H start",
            minutes=minutes,
            sbx_start_min=SBX_START_MIN,
        )
    if not (SBX_START_MIN < stress_on_min < minutes):
        rep1(
            "stress onset must lie after SBX-H starts and inside the horizon",
            sbx_start_min=SBX_START_MIN,
            stress_on_min=stress_on_min,
            minutes=minutes,
        )
    if stress_off_min is not None and stress_off_min <= stress_on_min:
        rep1(
            "stress removal must follow stress onset",
            stress_on_min=stress_on_min,
            stress_off_min=stress_off_min,
        )
    if sink_mvar <= 0.0:
        rep1("reactive sink must be positive", sink_mvar=sink_mvar)
    base = importlib.import_module("experiments.run_multi_system_ofo")
    cfg = base.make_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = int(verbose)
    cfg.zone_v_setpoints_pu = {
        area: PLANNING_V_REF_PU for area in (1, 2, 3)
    }

    cfg.coordination_mode = "sbx_h"
    cfg.sbx_config = SBXConfig(
        tso_period_s=float(cfg.tso_period_s),
        k_sched=2,
        q_band_mvar=10.0,
        p_support_eur_per_mvarh=5.0,
        v_hold_tolerance_pu=0.0025,
        v_sag_threshold_pu=0.005,
        n_need=2,
        release_threshold_pu=0.001,
        escalation_cycles=4,
        w_track_factor=1.0,
    )
    cfg.sbx_warmup_s = 60.0 * SBX_START_MIN
    cfg.sbx_support_intervals = None

    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.refresh_shared_jac_on_tso = False
    cfg.run_stability_analysis = False

    cfg.zone_v_min_pu = {STRESSED_AREA: STRESSED_AREA_V_MIN_PU}
    events = [
        ContingencyEvent(
            minute=stress_on_min,
            element_type="load",
            bus=STRESS_BUS,
            p_mw=0.0,
            q_mvar=sink_mvar,
            action="connect",
        )
    ]
    if stress_off_min is not None and stress_off_min < minutes:
        events.append(
            ContingencyEvent(
                minute=stress_off_min,
                element_type="load",
                bus=STRESS_BUS,
                p_mw=0.0,
                q_mvar=sink_mvar,
                action="trip",
            )
        )
    cfg.contingencies = events
    # cfg.contingencies = [
    #         # Example: trip line 0 at t=30 min, restore at t=60 min
    #         # ContingencyEvent(minute=100, element_type="line", element_index=8, action="trip"),
    #         # ContingencyEvent(minute=150, element_type="line", element_index=8, action="restore"),
    #         ContingencyEvent(minute=90, element_type="gen", element_index=5, action="trip"),
    #         ContingencyEvent(minute=180, element_type="gen", element_index=5, action="restore"),
    #         ContingencyEvent(minute=120, element_type="load", bus=5, p_mw=400, q_mvar=200, action="connect"),
    #         ContingencyEvent(minute=300, element_type="load", bus=5, p_mw=400, q_mvar=200, action="trip"),
    #         ContingencyEvent(minute=330, element_type="gen", element_index=4, action="trip"),
    #         ContingencyEvent(minute=420, element_type="gen", element_index=4, action="restore"),
    #         ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
    #         ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
    #         ContingencyEvent(minute=720, element_type="load", bus=7, p_mw=300, q_mvar=100, action="connect"),
    #         ContingencyEvent(minute=900, element_type="load", bus=7, p_mw=300, q_mvar=100, action="trip"),
    #     ]
    # One live figure only. In headless mode the same plotter runs on
    # the Agg backend so the saved thesis figure follows the live path.
    cfg.live_plot_controller = True
    cfg.live_plot_cascade = True
    cfg.live_plot_system = False
    cfg.live_plot_tracking = False
    cfg.live_plot_sbx = True
    return cfg


def _solver_ok(records) -> bool:
    accepted = {"optimal", "optimal_inaccurate"}
    return all(
        status is None or status in accepted
        for record in records
        for status in record.zone_tso_status.values()
    )


def _corridor_metrics(adapter, key: Tuple[int, int]) -> Dict[str, object]:
    settlements = adapter.scheduler.settlements[key]
    return {
        "cycles": len(settlements),
        "a_sags_b_holds": sum(
            item.support_state == "a_sags_b_holds"
            for item in settlements
        ),
        "b_sags_a_holds": sum(
            item.support_state == "b_sags_a_holds"
            for item in settlements
        ),
        "both_sag": sum(
            item.support_state == "both_sag"
            for item in settlements
        ),
        "paid": sum(item.support_eur > 0.0 for item in settlements),
        "support_mvarh": sum(
            item.support_mvarh for item in settlements
        ),
        "support_eur": sum(item.support_eur for item in settlements),
    }


def write_experiment_summary(
    path: Path,
    *,
    adapter,
    records,
    minutes: float,
    stress_on_min: float,
    stress_off_min: Optional[float],
    sink_mvar: float,
    wall_s: float,
) -> None:
    """Write a compact thesis-facing run summary."""
    metrics = {
        key: _corridor_metrics(adapter, key)
        for key in sorted(adapter.scheduler.corridors)
    }
    payments: Dict[int, float] = {
        area: 0.0 for area in adapter.scheduler.area_ids
    }
    for engine in adapter.scheduler.settlement_engines.values():
        for area, amount in engine.ledger.payments_eur.items():
            payments[area] += amount
    diagnostics = adapter.initial_schedule_diagnostics
    initially_holding = sum(
        bool(row["initially_holds"]) for row in diagnostics
    )
    worst_hold_margin_mpu = 1e3 * min(
        (float(row["hold_margin_pu"]) for row in diagnostics),
        default=float("nan"),
    )

    lines = [
        "# SBX-H capability demonstration",
        "",
        "## Scenario",
        "",
        f"- Horizon: {minutes:.1f} min",
        f"- SBX-H contract initialization: {SBX_START_MIN:.1f} min",
        f"- Contract schedule source: {adapter.schedule_source}",
        f"- Initial terminal hold pre-check: {initially_holding}/"
        f"{len(diagnostics)} inside tolerance; worst margin "
        f"{worst_hold_margin_mpu:+.2f} mpu",
        f"- Reactive sink: {sink_mvar:.1f} Mvar at bus {STRESS_BUS}",
        f"- Stress onset: {stress_on_min:.1f} min",
        (
            f"- Stress removal: {stress_off_min:.1f} min"
            if stress_off_min is not None and stress_off_min < minutes
            else "- Stress removal: outside simulated horizon"
        ),
        f"- Planning voltage reference: constant {PLANNING_V_REF_PU:.4f} pu",
        "- Planning-schedule changes: disabled in this experiment",
        "- Real-time support requests/commands: absent",
        f"- Plant records: {len(records)}",
        f"- Wall time: {wall_s:.1f} s",
        f"- All TSO solves accepted: {_solver_ok(records)}",
        "",
        "## SBX-H outcome",
        "",
        "| Corridor | Settled windows | A violates / B holds | "
        "B violates / A holds | Both violate | Paid windows | "
        "Support [Mvar h] | Gross support value [EUR] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in metrics.items():
        lines.append(
            f"| ({key[0]},{key[1]}) | {row['cycles']} | "
            f"{row['a_sags_b_holds']} | "
            f"{row['b_sags_a_holds']} | "
            f"{row['both_sag']} | {row['paid']} | "
            f"{row['support_mvarh']:.3f} | "
            f"{row['support_eur']:.2f} |"
        )
    lines.extend([
        "",
        "Net bilateral payments: "
        + ", ".join(
            f"area {area}: {amount:+.2f} EUR"
            for area, amount in sorted(payments.items())
        ),
        "",
        "Escalation events: "
        + (
            str(adapter.scheduler.escalations)
            if adapter.scheduler.escalations
            else "none"
        ),
        "",
        "## Interpretation",
        "",
        "Payment is issued only when exactly one corridor side violates "
        "its symmetric scheduled-voltage band, the other side holds, and "
        "the beyond-band reactive flow has the relieving sign.",
        "",
        "Q_0 is the counterfactual flow at scheduled terminal voltages "
        "and measured active transfer. B_Q is a no-payment deadband around "
        "Q_0. Signed Q_sup is the relieving residual beyond that deadband; "
        "it is not a sold strength quantity.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the 300-minute SBX-H capability demonstration."
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=DEFAULT_MINUTES,
        help=("simulation horizon in minutes "
              f"(default: {DEFAULT_MINUTES:g})"),
    )
    parser.add_argument(
        "--stress-on",
        type=float,
        default=STRESS_ON_MIN,
        help=("reactive-sink connection minute "
              f"(default: {STRESS_ON_MIN:g})"),
    )
    parser.add_argument(
        "--stress-off",
        type=float,
        default=STRESS_OFF_MIN,
        help=("reactive-sink removal minute "
              f"(default: {STRESS_OFF_MIN:g})"),
    )
    parser.add_argument(
        "--sink-mvar",
        type=float,
        default=STRESS_Q_MVAR,
        help=("reactive sink magnitude "
              f"(default: {STRESS_Q_MVAR:g} Mvar)"),
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="use a headless backend; the final figure is still saved",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("output directory (default: "
              "results/014_SBX_H_DEMO/<variant>)"),
    )
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.no_live:
        os.environ["MPLBACKEND"] = "Agg"

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else RESULT_ROOT / RESULT_CASE
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(
        minutes=args.minutes,
        stress_on_min=args.stress_on,
        stress_off_min=args.stress_off,
        sink_mvar=args.sink_mvar,
        verbose=args.verbose,
    )

    print("=== SBX-H v6 capability demonstration ===")
    print(f"  planning reference={PLANNING_V_REF_PU:.4f} pu, constant")
    print("  coordination=ex-post remuneration only")
    print(
        f"  horizon={args.minutes:.0f} min, "
        f"SBX start={SBX_START_MIN:.0f} min, "
        f"stress={args.stress_on:.0f}-"
        f"{min(args.stress_off, args.minutes):.0f} min"
    )
    print(
        f"  reactive sink={args.sink_mvar:.0f} Mvar at bus "
        f"{STRESS_BUS}; live plot={'OFF' if args.no_live else 'ON'}"
    )

    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    start = time.perf_counter()
    records = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    wall_s = time.perf_counter() - start

    runtime = captured.get("sbx_runtime") or {}
    adapter = runtime.get("adapter")
    plotter = runtime.get("live_plotter")
    if adapter is None:
        rep1(
            "run completed without an SBX-H adapter",
            sbx_start_s=cfg.sbx_warmup_s,
            horizon_s=cfg.n_total_s,
        )
    if plotter is None:
        rep1("SBX-H live plotter was not constructed")

    from sbx_h.settlement import write_settlement_outputs

    generated_csv, generated_md = write_settlement_outputs(
        adapter.scheduler.settlement_engines,
        output_dir,
        "sbx_h",
    )
    csv_path = output_dir / "sbx_h_settlement_ledger.csv"
    settlement_md = output_dir / "sbx_h_settlement_summary.md"
    os.replace(generated_csv, csv_path)
    os.replace(generated_md, settlement_md)
    figure_path = output_dir / "sbx_h_mechanism.png"
    plotter.save(figure_path)
    summary_path = output_dir / "experiment_summary.md"
    write_experiment_summary(
        summary_path,
        adapter=adapter,
        records=records,
        minutes=args.minutes,
        stress_on_min=args.stress_on,
        stress_off_min=args.stress_off,
        sink_mvar=args.sink_mvar,
        wall_s=wall_s,
    )

    paid_windows = sum(
        item.support_eur > 0.0
        for rows in adapter.scheduler.settlements.values()
        for item in rows
    )
    print(f"  completed {len(records)} plant steps in {wall_s:.1f} s")
    print(f"  paid support windows: {paid_windows}")
    print(f"  figure:             {figure_path}")
    print(f"  settlement ledger: {csv_path}")
    print(f"  settlement report: {settlement_md}")
    print(f"  experiment report: {summary_path}")
    return 0


if __name__ == "__main__":
    main()
