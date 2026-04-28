#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/002_M_TSO_M_DSO_COMPARE.py
======================================
Compare the Multi-TSO / Multi-DSO OFO controller against four local-control
and partial-OFO baselines on the IEEE 39-bus ``wind_replace`` scenario.

Scenarios
---------
* **L0**     -- All TSO-connected windparks AND all DSO-connected SGENs at
                cos phi = 1 (Q = 0).  Worst case; may diverge.
* **L1**     -- Q(V) droop at every TSO-connected windpark
                (``pp.control.CharacteristicControl``); DSO SGENs at cos phi = 1.
* **L2**     -- Q(V) droop at TSO windparks AND DSO SGENs.
                V_set = 1.03 pu, slope = 0.07 pu (full Q at delta-V = +/- 0.07).
* **T-OFO**  -- TSO-only OFO: zone-level OFO MIQP at the TSO layer; DSO SGENs
                follow Q(V) droop (V_set = 1.03 pu, slope = 0.07 pu) without
                interface-Q tracking.  Ablation between L2 and C-OFO.
* **C-OFO**  -- Cascade OFO: multi-zone OFO MIQP at TSO and DSO layers
                (default behaviour).

All five scenarios share the OFO base configuration from
``experiments/000_M_TSO_M_DSO.main()`` (16-hour profile, 10-event
contingency timeline, identical TSO/DSO weights and timing) and only
the control mode is varied; this guarantees a fair comparison.  Each
scenario runs ``run_multi_tso_dso(cfg)`` with the scenario-specific
overrides; the five record lists are persisted as pickle and
aggregated into:

* ``results/002_compare/summary.csv``           -- one row per scenario.
* ``results/002_compare/compare_voltage_envelope.{png,pdf}``
* ``results/002_compare/compare_voltage_violations.{png,pdf}``
* ``results/002_compare/compare_losses.{png,pdf}``
* ``results/002_compare/compare_gen_q_headroom.{png,pdf}``

Non-converged scenarios (most likely L0) are caught at the per-scenario
boundary, persisted as empty logs, and annotated in the comparison plots.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

import os
import pickle
import sys
import importlib
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent, MultiTSOIterationRecord
from visualisation.plot_compare_scenarios import plot_scenario_comparison

# 000_M_TSO_M_DSO.py starts with a digit, so the import has to go through
# importlib rather than a normal `from ... import ...`.
_runner = importlib.import_module("experiments.000_M_TSO_M_DSO")
run_multi_tso_dso = _runner.run_multi_tso_dso


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


def make_base_config() -> MultiTSOConfig:
    """Shared simulation conditions for the four-scenario comparison.

    Mirrors the OFO configuration used in ``main()`` of
    ``experiments/000_M_TSO_M_DSO.py`` so that all four scenarios run on
    the same grid, profile, contingency timeline, and TSO/DSO weights —
    only the control mode is varied per scenario.

    Notes
    -----
    * The OFO weights (g_v, g_q, g_w_*) are no-ops in the L0/L1/L2
      scenarios because the OFO TSO and DSO controllers are not stepped.
      Keeping them in the shared base config simplifies maintenance and
      guarantees that the OFO scenario uses the user's tuned values
      without duplication.
    * Live plots are forced OFF: a four-fold sequential sweep with live
      plotting either crashes headless or clutters the screen.  Use
      ``visualisation/plot_compare_scenarios.plot_scenario_comparison``
      for post-hoc comparison plots instead.
    """
    cfg = MultiTSOConfig(
        n_total_s    = 60.0 * 60 * 6,    # 4-hour full simulation
        tso_period_s = 60.0 * 3,          # TSO every 3 minutes
        dso_period_s = 10.0,              # DSO at fastest viable cadence
        g_v          = 120000.0,
        g_q          = 200,
        tso_g_q_tie  = 5,
        # ── DSO objective tuning ──────────────────────────────────────
        dso_g_v                  = 20000.0,
        dso_g_qi                 = 0,
        dso_lambda_qi            = 0.9,
        dso_q_integral_max_mvar  = 50.0,
        dso_gamma_oltc_q         = 0.0,
        # ── TSO weights ──────────────────────────────────────────────
        install_tso_tertiary_shunts=False,
        g_w_der      = 10,
        g_w_gen      = 4e7,
        g_w_pcc      = 100,
        g_w_tso_oltc = 100,
        g_w_tso_shunt=50000,
        # ── DSO weights ──────────────────────────────────────────────
        g_w_dso_der  = 800,
        g_w_dso_oltc = 30,
        use_fixed_zones              = True,
        run_stability_analysis       = False,
        sensitivity_update_interval  = int(1e6),
        verbose                      = 1,
        # Live plots forced OFF for sequential comparison sweep.
        live_plot_controller = True,
        live_plot_cascade    = True,
        live_plot_system     = False,
        # ── Profile & contingency settings ───────────────────────────
        start_time              = datetime(2016, 4, 15, 8, 0),
        use_profiles            = True,
        use_zonal_gen_dispatch  = True,
        contingencies           = [
            ContingencyEvent(minute=90,  element_type="gen",  element_index=5, action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=5, action="restore"),
            ContingencyEvent(minute=120, element_type="load", bus=5,  p_mw=300, q_mvar=100, action="connect"),
            ContingencyEvent(minute=300, element_type="load", bus=5,  p_mw=300, q_mvar=100, action="trip"),
            # ContingencyEvent(minute=330, element_type="gen",  element_index=2, action="trip"),
            # ContingencyEvent(minute=420, element_type="gen",  element_index=2, action="restore"),
            ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
            # ContingencyEvent(minute=720, element_type="load", bus=7,  p_mw=300, q_mvar=100, action="connect"),
            # ContingencyEvent(minute=900, element_type="load", bus=7,  p_mw=300, q_mvar=100, action="trip"),
        ],
    )
    cfg.scenario = "wind_replace"
    cfg.warmup_s = 0.0
    return cfg


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "L0":  dict(
        tso_mode="local", tso_local_mode="cos_phi_1",
        dso_mode="local", local_der_mode="cos_phi_1",
    ),
    "L1":  dict(
        tso_mode="local", tso_local_mode="qv",
        tso_qv_setpoint_pu=1.03, tso_qv_slope_pu=0.07,
        dso_mode="local", local_der_mode="cos_phi_1",
    ),
    "L2":  dict(
        tso_mode="local", tso_local_mode="qv",
        tso_qv_setpoint_pu=1.03, tso_qv_slope_pu=0.07,
        dso_mode="local", local_der_mode="qv",
        qv_setpoint_pu=1.03, qv_slope_pu=0.07,
    ),
    "T-OFO": dict(
        tso_mode="ofo",
        dso_mode="local", local_der_mode="qv",
        qv_setpoint_pu=1.03, qv_slope_pu=0.07,
    ),
    "C-OFO": dict(
        tso_mode="ofo",
        dso_mode="ofo",
    ),
}


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------


def run_one_scenario(name: str, overrides: Dict[str, Any], out_root: str
                     ) -> List[MultiTSOIterationRecord]:
    """Run a single scenario and pickle its log.  Returns the log."""
    cfg = make_base_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    scen_dir = os.path.join(out_root, name)
    cfg.result_dir = scen_dir
    os.makedirs(scen_dir, exist_ok=True)

    print()
    print("=" * 72)
    print(f"  RUNNING SCENARIO {name}")
    print("=" * 72)
    print(f"  tso_mode={cfg.tso_mode}, tso_local_mode={cfg.tso_local_mode}")
    print(f"  dso_mode={cfg.dso_mode}, local_der_mode={cfg.local_der_mode}")
    print(f"  result_dir={scen_dir}")

    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
        log = []

    pkl_path = os.path.join(scen_dir, "log.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [{name}] wrote {len(log)} records -> {pkl_path}")
    return log


def load_logs(out_root: str = os.path.join("results", "002_compare"),
              names: Optional[List[str]] = None,
              ) -> Dict[str, List[MultiTSOIterationRecord]]:
    """Reload pickled scenario logs from a previous run.

    Parameters
    ----------
    out_root : str
        Directory containing one ``<name>/log.pkl`` per scenario.
    names : list of str, optional
        Subset of scenarios to load.  Defaults to all keys in
        :data:`SCENARIOS` (L0, L1, L2, T-OFO, C-OFO).  Missing logs are returned
        as empty lists (treated as "scenario diverged" by the plotter).

    Returns
    -------
    dict ``{scenario_name: List[MultiTSOIterationRecord]}``
    """
    if names is None:
        names = list(SCENARIOS.keys())
    out: Dict[str, List[MultiTSOIterationRecord]] = {}
    for name in names:
        pkl_path = os.path.join(out_root, name, "log.pkl")
        if not os.path.isfile(pkl_path):
            print(f"  [load_logs] missing {pkl_path} -- treating {name} as diverged")
            out[name] = []
            continue
        with open(pkl_path, "rb") as f:
            out[name] = pickle.load(f)
        print(f"  [load_logs] {name}: {len(out[name])} records from {pkl_path}")
    return out


def replot(out_root: str = os.path.join("results", "002_compare")) -> None:
    """Regenerate all comparison figures from the persisted ``log.pkl``
    files in ``out_root`` -- useful when a previous run wrote the pickles
    but the plot pass failed (e.g. PDF locked open in a viewer).
    """
    logs = load_logs(out_root)
    print()
    print("=" * 72)
    print(f"  REPLOTTING from {out_root}/")
    print("=" * 72)
    plot_scenario_comparison(logs, out_dir=out_root)
    print(f"  Re-wrote summary.csv and comparison figures to {out_root}/")


def main() -> None:
    out_root = os.path.join("results", "002_compare")
    os.makedirs(out_root, exist_ok=True)

    logs: Dict[str, List[MultiTSOIterationRecord]] = {}
    for name, overrides in SCENARIOS.items():
        logs[name] = run_one_scenario(name, overrides, out_root)

    print()
    print("=" * 72)
    print("  AGGREGATING COMPARISON METRICS")
    print("=" * 72)
    plot_scenario_comparison(logs, out_dir=out_root)
    print(f"  Wrote summary.csv and comparison figures to {out_root}/")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot()
    else:
        main()
