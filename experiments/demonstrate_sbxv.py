"""Standalone SBX-V mechanism demonstration on the three-zone CIGRE case.

Controlled outputs are EHV-zone voltages and EHV-HV interface Q.
Actuators remain the configured TSO/DSO DER, AVR, OLTC and shunt devices;
SBX-V changes only the vertical band/request pricing layer.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.helpers.records import ContingencyEvent
from experiments.results_io import new_run_dir
from experiments.runners.multi_tso_dso import run_multi_tso_dso
from sbx_v.config import SBXVConfig
from sbx_v.settlement import write_settlement_csv

_CIGRE = importlib.import_module("experiments.CIGRE_2026.005_CIGRE_MULTI")


def make_config(minutes: float, *, live: bool = True):
    """Build an explicit stressed SBX-V configuration."""
    cfg = _CIGRE.make_cigre_config()
    cfg.n_total_s = float(minutes) * 60.0
    cfg.verbose = 1
    cfg.coordination_mode = "sbx_v"
    cfg.live_plot_sbxv = bool(live)
    cfg.sbxv_config = SBXVConfig(
        tso_period_s=float(cfg.tso_period_s),
        band_preset="fixed",
        band_q_raise_mvar=25.0,
        band_q_lower_mvar=25.0,
    )
    stress_off = max(35.0, min(80.0, float(minutes) - 15.0))
    cfg.contingencies = [
        ContingencyEvent(
            minute=20.0, element_type="load", bus=9,
            p_mw=0.0, q_mvar=400.0, action="connect",
        ),
        ContingencyEvent(
            minute=stress_off, element_type="load", bus=9,
            p_mw=0.0, q_mvar=400.0, action="trip",
        ),
    ]
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SBX-V live demonstrator.")
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--no-live", action="store_true")
    args = parser.parse_args()
    if args.minutes < 45.0:
        parser.error("--minutes must be at least 45 to include complete stress windows")

    cfg = make_config(args.minutes, live=not args.no_live)
    run_dir = new_run_dir("demonstrate_sbxv", cfg)
    captured = {}

    def hook(state):
        captured.update(state)
        return None

    records = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    with (run_dir.root / "records.pkl").open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    runtime = captured.get("sbxv_runtime") or {}
    adapter = runtime.get("adapter")
    final = adapter.finalise() if adapter is not None else None
    summary = {
        "records": len(records),
        "areas": sorted(adapter.areas) if adapter is not None else [],
        "bands_mvar": final.get("bands") if final else {},
        "requests": sum(
            1 for events in (final.get("pipeline_logs", {}) if final else {}).values()
            for event in events if event[0] == "request"
        ),
        "grants": len(final.get("grant_records", ())) if final else 0,
        "dropped_grants": len(final.get("dropped_grants", ())) if final else 0,
    }
    settlement = final.get("settlement") if final else None
    if settlement is not None:
        write_settlement_csv(settlement, str(run_dir.csv / "settlement"))
        summary["remuneration_eur"] = sum(
            total.pay_total_eur for total in settlement.totals
        )
    with (run_dir.root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plotter = runtime.get("live_plotter")
    if plotter is not None:
        plotter.save(run_dir.figures / "sbxv_mechanism.png")

    print(f"SBX-V demonstration complete: {run_dir.root}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
