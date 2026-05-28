#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/002_M_TSO_M_DSO_COMPARE.py
======================================
Compare the Multi-TSO / Multi-DSO OFO controller against local-control
baselines on the IEEE 39-bus ``wind_replace`` scenario.

The five-scenario design factors out two axes:

* **TSO control**: local Q(V) droop (L) vs zone-level OFO MIQP (T, C).
* **DSO control**: cos phi=1 (suffix ``0``), local Q(V) droop
  (suffix ``1``), or DSO OFO MIQP (cascade ``C``).

Plant-side Q is driven by the q_mode loops
(``QVLocalLoop`` / ``CosPhiConstLoop``) installed at [3c-deferred]
in ``experiments/000_M_TSO_M_DSO.py``; the OFO scenarios additionally
command ``q_set_mvar`` per DER (and reanchor ``qv_vref_anchor_pu``)
from the MIQP layer above.

Scenarios
---------
* **L0**  -- TSO local Q(V) (V_set=1.03, deadband ±0.005,
             slope=q_max/0.05); DSO SGENs at cos phi=1.
* **L1**  -- TSO and DSO both under local Q(V) droop with the same
             parameters as L0.
* **T0**  -- TSO OFO MIQP; DSO SGENs at cos phi=1.  Mitigation H
             pin ``g_w_pcc=1e8`` keeps Q_PCC near its current value
             (no DSO MIQP to track Q_PCC,set).
* **T1**  -- TSO OFO MIQP; DSO SGENs under local Q(V) droop with the
             same parameters as L0.  Mitigation H pin retained.
* **C**   -- Cascade OFO: OFO MIQP at both TSO and DSO layers.

All five scenarios share the OFO base configuration from
``make_base_config()`` (160-min profile, 8-event contingency timeline,
identical TSO/DSO weights and timing) and only the control mode is
varied; this guarantees a fair comparison.  Each scenario runs
``run_multi_tso_dso(cfg)`` with the scenario-specific overrides; the
five record lists are persisted as pickle and aggregated into:

* ``results/002_compare/summary.csv``           -- one row per scenario.
* ``results/002_compare/compare_voltage_envelope.{png,pdf}``
* ``results/002_compare/compare_voltage_violations.{png,pdf}``
* ``results/002_compare/compare_losses.{png,pdf}``
* ``results/002_compare/compare_gen_q_headroom.{png,pdf}``
* ``results/002_compare/compare_gen_capability_pq.{png,pdf}``

Non-converged scenarios are caught at the per-scenario boundary,
persisted as empty logs, and annotated in the comparison plots.

CLI flags
---------
* ``--replot``         -- skip simulation, regenerate plots from
                          the existing ``log.pkl`` files in
                          ``results/002_compare/``.
* ``--only L0,L1,T1``  -- run only the named scenarios.  Existing
                          pickles for the other scenarios are still
                          picked up by the comparison plots.
* ``--skip L0,T0``     -- inverse of ``--only``.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent, MultiTSOIterationRecord
from experiments.runners import run_multi_tso_dso
from visualisation.plot_compare_scenarios import plot_scenario_comparison


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


def make_base_config() -> MultiTSOConfig:
    """Shared simulation conditions for the six-scenario comparison.

    Mirrors the OFO configuration used in ``main()`` of
    ``experiments/000_M_TSO_M_DSO.py`` so that all six scenarios run on
    the same grid, profile, contingency timeline, and TSO/DSO weights —
    only the control mode and the TSO statcom partition are varied per
    scenario.

    Notes
    -----
    * The OFO weights (g_v, g_q, g_w_*) are no-ops in the L0/L1
      scenarios because the OFO TSO and DSO controllers are not stepped.
      Keeping them in the shared base config simplifies maintenance and
      guarantees that the OFO scenarios use the user's tuned values
      without duplication.
    * Live plots are forced OFF: a six-fold sequential sweep with live
      plotting either crashes headless or clutters the screen.  Use
      ``visualisation/plot_compare_scenarios.plot_scenario_comparison``
      for post-hoc comparison plots instead.
    """
    cfg = MultiTSOConfig(
        n_total_s=60*60*3, #60.0 * 60 * 24,      # 720-min full simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=20.0,    # DSO every 5 seconds (more inner iterations)
        g_v=3E5,  # TSO voltage tracking; drives PCC Q dispatch
        g_q=250,  # DSO Q-tracking
        tso_g_q_tie=1,
        # ── DSO objective tuning ──
        # DER actuator: w-shift (q_set + V_ref reanchoring).
        dso_g_v=20000.0,  # reduced to avoid competing with Q tracking
        dso_g_qi=0,  # integral Q-tracking (0 = off)
        dso_lambda_qi=0.95,  # leaky integrator decay
        dso_q_integral_max_mvar=200.0,  # anti-windup clamp
        dso_gamma_oltc_q=0.0,  # OLTC Q-tracking attenuation: DER-primary, OLTC-backup
        # ── TSO weights — re-tuned for Q_cor closed-loop curvature ──
        # Under Q_cor the H matrix is post-multiplied by T'=(I+R*S_VQ)^-1,
        # which reduces curvature by ~3x on DSO and ~80x on TSO.  Start
        # at ~1/3 (DSO) to ~1/80 (TSO STATCOMs) of the legacy direct-Q
        # values and sweep from there.
        g_w_der=10,    # was 20 (direct-Q); T-STATCOM curvature ~80x lower under T'
        g_w_gen=5e7,
        g_w_pcc=10,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        # ── DSO weights ──
        g_w_dso_der=1000,  # was 1000 (direct-Q); ~3x lower curvature under T'
        g_w_dso_oltc=40,
        use_fixed_zones=True,      # literature 3-area partition (not spectral)
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,  # refresh H_ij every N TSO steps
        verbose=1,
        live_plot_controller=True,
        live_plot_cascade=True,
        live_plot_system=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # ── Profile & contingency settings ───────────────────────────────
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles            = True,
        use_zonal_gen_dispatch  = True,
        contingencies           = [
            ContingencyEvent(minute=30,  element_type="gen",  element_index=2, action="trip"),
            ContingencyEvent(minute=90, element_type="gen", element_index=2, action="restore"),
            ContingencyEvent(minute=50, element_type="load", bus=27,  p_mw=200, q_mvar=100, action="connect"),
            ContingencyEvent(minute=100, element_type="load", bus=27,  p_mw=200, q_mvar=100, action="trip"),
            ContingencyEvent(minute=120, element_type="load",
                             element_index=4, action="trip"),
            ContingencyEvent(minute=160, element_type="load",
                             element_index=4, action="restore"),
            ContingencyEvent(minute=130, element_type="line", element_index=18, action="trip"),
            ContingencyEvent(minute=150, element_type="line", element_index=18, action="restore"),
            #ContingencyEvent(minute=180, element_type="gen",  element_index=5, action="trip"),
            #ContingencyEvent(minute=280, element_type="gen",  element_index=5, action="restore"),
            # ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            # ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
            # ContingencyEvent(minute=720, element_type="load", bus=7,  p_mw=300, q_mvar=100, action="connect"),
            # ContingencyEvent(minute=900, element_type="load", bus=7,  p_mw=300, q_mvar=100, action="trip"),
        ],
    )
    cfg.scenario = "wind_replace"
    cfg.warmup_s = 0.0
    return cfg


SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ── Local-control baselines (L0, L1) ─────────────────────────────────
    # No OFO at any layer.  Plant-side Q control is provided by the
    # refactor_v2 q_mode loops (QVLocalLoop / CosPhiConstLoop installed
    # at [3c-deferred] in experiments/000_M_TSO_M_DSO.py).
    # Q(V) parameters: V_set=1.03 pu, deadband ±0.005 pu,
    # slope=q_max/0.05 (i.e. ``qv_slope_pu=0.05``).
    "L0": dict(
        tso_mode="local", tso_local_mode="qv",
        dso_mode="local", local_der_mode="cos_phi_1",
        tso_q_mode="qv",  dso_q_mode="cosphi",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
    ),
    "L1": dict(
        tso_mode="local", tso_local_mode="qv",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv",  dso_q_mode="qv",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
        dso_qv_vref_pu=1.03, dso_qv_slope_pu=0.06, dso_qv_deadband_pu=0.01,
    ),
    # ── TSO-only OFO (T0, T1) ───────────────────────────────────────────
    # OFO MIQP active at the transmission layer; DSO uses the local
    # plant-side controllers (no DSO MIQP).  Mitigation H from
    # ``experiments/results/002_compare/diagnostics_T-OFO/summary.md``:
    # the DSO has no MIQP tracking Q_PCC,set, so the OFO regulariser
    # pin (``g_w_pcc=1e8``) keeps each Q_PCC setpoint near its current
    # value while the TSO OFO drives V via V_gen / OLTC / shunt.
    "T0": dict(
        tso_mode="ofo",
        dso_mode="local", local_der_mode="cos_phi_1",
        tso_q_mode="qv",  dso_q_mode="cosphi",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
        g_w_pcc=1.0e8,
    ),
    "T1": dict(
        tso_mode="ofo",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv",  dso_q_mode="qv",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
        dso_qv_vref_pu=1.03, dso_qv_slope_pu=0.06, dso_qv_deadband_pu=0.01,
        g_w_pcc=1.0e8,
    ),
    # ── Cascade OFO (C) ─────────────────────────────────────────────────
    # OFO MIQP active at both TSO and DSO layers (default behaviour).
    "C": dict(
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
        :data:`SCENARIOS` (L0, L1, T0, T1, C).  Missing logs
        are returned as empty lists (treated as "scenario diverged" by
        the plotter).

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


def main(only: Optional[List[str]] = None,
         skip: Optional[List[str]] = None) -> None:
    """Run the scenario comparison.

    Parameters
    ----------
    only : list of str, optional
        If given, run only these scenarios (others are skipped).
    skip : list of str, optional
        If given, skip these scenarios.  Applied after ``only``.

    The comparison plot is always built from every available
    ``log.pkl`` on disk via :func:`load_logs`, so partial runs still
    produce full-suite comparison figures (existing pickles for
    skipped scenarios are picked up).
    """
    out_root = os.path.join("results", "002_compare")
    os.makedirs(out_root, exist_ok=True)

    selected = list(SCENARIOS.keys())
    if only is not None:
        unknown = [s for s in only if s not in SCENARIOS]
        if unknown:
            raise ValueError(
                f"--only contains unknown scenarios: {unknown}; "
                f"valid names are {list(SCENARIOS.keys())}"
            )
        selected = [s for s in selected if s in only]
    if skip is not None:
        unknown = [s for s in skip if s not in SCENARIOS]
        if unknown:
            raise ValueError(
                f"--skip contains unknown scenarios: {unknown}; "
                f"valid names are {list(SCENARIOS.keys())}"
            )
        selected = [s for s in selected if s not in skip]

    if not selected:
        print("[main] No scenarios selected; nothing to run.")
    else:
        print(f"[main] Running scenarios: {selected}")
        for name in selected:
            run_one_scenario(name, SCENARIOS[name], out_root)

    # Build the comparison from every available pickle, including ones
    # the user skipped this run.
    logs = load_logs(out_root)

    print()
    print("=" * 72)
    print("  AGGREGATING COMPARISON METRICS")
    print("=" * 72)
    plot_scenario_comparison(logs, out_dir=out_root)
    print(f"  Wrote summary.csv and comparison figures to {out_root}/")


def _parse_csv_arg(argv: List[str], flag: str) -> Optional[List[str]]:
    """Return the comma-separated list following ``flag`` in *argv*, or
    ``None`` if the flag is absent.  Whitespace around each item is
    stripped; empty items are dropped.

    Raises ``ValueError`` if the flag is present but no value follows.
    """
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(
            f"{flag} requires a comma-separated value, e.g. {flag} L0,L2"
        )
    return [s.strip() for s in argv[idx + 1].split(",") if s.strip()]


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot()
    else:
        main(
            only=_parse_csv_arg(sys.argv, "--only"),
            skip=_parse_csv_arg(sys.argv, "--skip"),
        )
