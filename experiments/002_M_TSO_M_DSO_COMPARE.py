#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/002_M_TSO_M_DSO_COMPARE.py
======================================
Compare the Multi-TSO / Multi-DSO OFO controller against four
local-control baselines on the IEEE 39-bus ``wind_replace`` scenario.

The L family is split along two axes — TSO statcom partition and DSO
local control — so the partition asymmetry between the legacy
``pp.sgen`` model and the promoted ``pp.gen`` model used by the OFO is
explicitly represented in the comparison.

Scenarios
---------
* **L0**     -- Legacy partition: TSO statcoms stay as ``pp.sgen`` with
                local Q(V) droop (V_set=1.03 pu, slope=0.07 pu); DSO
                SGENs at cos phi=1.
* **L1**     -- Legacy partition (TSO sgens with local Q(V) droop) and
                DSO SGENs also under Q(V) droop.
* **L2**     -- Promoted-gen partition: TSO statcoms promoted to
                ``pp.gen`` (PV) with constant AVR ``vm_pu = 1.03``
                (no Q control loop at the TSO layer); DSO SGENs at
                cos phi=1.
* **L3**     -- Promoted-gen partition (constant-AVR TSO gens) and
                DSO SGENs under Q(V) droop.
* **T-OFO**  -- TSO-only OFO: zone-level OFO MIQP at the TSO layer;
                DSO SGENs follow Q(V) droop (V_set=1.03, slope=0.07)
                without interface-Q tracking.  Ablation between L3
                and C-OFO.
* **C-OFO**  -- Cascade OFO: multi-zone OFO MIQP at the TSO and DSO
                layers (default behaviour).

The L0/L1 vs L2/L3/T-OFO/C-OFO grouping aligns the active-power
dispatch across the four "promoted" scenarios so that
``compute_zonal_gen_dispatch`` sees the same residual on each (no
``net.sgen`` ↔ ``net.gen`` partition asymmetry); L0 and L1 retain the
legacy partition as the "before-promotion" reference.

All six scenarios share the OFO base configuration from
``experiments/000_M_TSO_M_DSO.main()`` (16-hour profile, 10-event
contingency timeline, identical TSO/DSO weights and timing) and only
the control mode and partition are varied; this guarantees a fair
comparison.  Each scenario runs ``run_multi_tso_dso(cfg)`` with the
scenario-specific overrides; the six record lists are persisted as
pickle and aggregated into:

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
* ``--replot``           -- skip simulation, regenerate plots from
                            the existing ``log.pkl`` files in
                            ``results/002_compare/``.
* ``--only L1,L3,T-OFO`` -- run only the named scenarios.  Existing
                            pickles for the other scenarios are still
                            picked up by the comparison plots.
* ``--skip L0,L2``       -- inverse of ``--only``.

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
    """Shared simulation conditions for the six-scenario comparison.

    Mirrors the OFO configuration used in ``main()`` of
    ``experiments/000_M_TSO_M_DSO.py`` so that all six scenarios run on
    the same grid, profile, contingency timeline, and TSO/DSO weights —
    only the control mode and the TSO statcom partition are varied per
    scenario.

    Notes
    -----
    * The OFO weights (g_v, g_q, g_w_*) are no-ops in the L0/L1/L2/L3
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
        n_total_s=60.0 * 60 * 2,      # 720-min full simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=10.0,    # DSO every 5 seconds (more inner iterations)
        g_v=3E5,  # TSO voltage tracking; drives PCC Q dispatch
        g_q=200,  # DSO Q-tracking
        # ── DSO objective tuning ──
        use_qv_local_loop=True,
        force_all_der_grid_following=False,
        dso_g_v=20000.0,  # reduced to avoid competing with Q tracking
        dso_g_qi=0,  # integral Q-tracking (0 = off)
        dso_lambda_qi=0.9,  # leaky integrator decay
        dso_q_integral_max_mvar=50.0,  # anti-windup clamp
        dso_gamma_oltc_q=0.0,  # OLTC Q-tracking attenuation: DER-primary, OLTC-backup
        # ── TSO weights (alpha=1, spectral rho(C)/2) ──
        g_w_der=10,   # single-DER zones; rho~C_jj=396 -> min 198
        g_w_gridforming= 5E6,
        g_w_gen=5e7,   # excluded from stability
        g_w_pcc=50,   # 9 correlated PCCs; rho(C)=221 -> min 111
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,   # bipolar tertiary shunts; mirror g_w_tso_oltc
        # ── DSO weights (alpha=1, rho(C_DER)=790 -> min 395) ──
        g_w_dso_der=2000,  # 2026-05-04: 500 caused massive DSO Q chatter; backed off toward 1000 per plan fallback. Per-step decay 0.286 (vs 0.20 at 1000, 0.40 at 500). Still > floor 395.
        g_w_dso_der_vref=1E6,
        g_w_dso_oltc=20,   # rho(C_OLTC)~1.1; higher for switching suppression
        # ── Adaptive g_w (paper Eq.16, sign-only) — all continuous classes ──
        # `g_w_*` values above become the warm-start; the meta below
        # controls the multiplicative rate and clip box.  Discrete
        # actuators (OLTC, shunt) intentionally left static.
        use_fixed_zones              = True,
        run_stability_analysis       = False,
        sensitivity_update_interval  = int(1e6),
        verbose                      = 1,
        # Live plots forced OFF for sequential comparison sweep.
        live_plot_controller = False,
        live_plot_cascade    = False,
        live_plot_system     = False,
        # ── Profile & contingency settings ───────────────────────────
        start_time              = datetime(2016, 4, 15, 8, 0),
        use_profiles            = True,
        use_zonal_gen_dispatch  = True,
        contingencies           = [
            ContingencyEvent(minute=60,  element_type="gen",  element_index=2, action="trip"),
            ContingencyEvent(minute=100, element_type="gen",  element_index=2, action="restore"),
            ContingencyEvent(minute=120, element_type="load", bus=2,  p_mw=300, q_mvar=150, action="connect"),
            ContingencyEvent(minute=240, element_type="load", bus=2,  p_mw=300, q_mvar=150, action="trip"),
            ContingencyEvent(minute=150, element_type="load", bus=14, p_mw=100, q_mvar=50, action="connect"),
            ContingencyEvent(minute=160, element_type="load", bus=14, p_mw=100, q_mvar=50, action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=5, action="trip"),
            ContingencyEvent(minute=280, element_type="gen",  element_index=5, action="restore"),
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
    # ── Legacy partition (L0, L1) ───────────────────────────────────────
    # ``force_all_der_grid_following=True`` keeps every TSO converter as
    # a current-source ``pp.sgen``.  The legacy local-controller install
    # path (``install_qv_characteristic_controllers`` / ``install_cos_phi_one``
    # in ``experiments/000_M_TSO_M_DSO.py``) sets the plant-side Q(V)
    # behavior on those sgens, so disable the new plant-side Q(V) local
    # loop to avoid duplicate Q(V) controllers per sgen.
    "L0":  dict(
        tso_mode="local", tso_local_mode="qv",
        tso_qv_setpoint_pu=1.03, tso_qv_slope_pu=0.07,
        dso_mode="local", local_der_mode="cos_phi_1",
        use_qv_local_loop=False,
        force_all_der_grid_following=True,
    ),
    "L1":  dict(
        tso_mode="local", tso_local_mode="qv",
        tso_qv_setpoint_pu=1.03, tso_qv_slope_pu=0.07,
        dso_mode="local", local_der_mode="qv",
        qv_setpoint_pu=1.03, qv_slope_pu=0.07,
        use_qv_local_loop=False,
        force_all_der_grid_following=True,
    ),
    # ── Promoted-gen partition (L2, L3) ─────────────────────────────────
    # ``force_all_der_grid_following`` defaults to False (base config),
    # so every TSO converter classified GRID_FORMING is promoted into a
    # ``pp.gen`` (PV bus) by ``apply_der_classification``.  ``tso_mode=
    # "local"`` means no OFO step ever writes ``net.gen.vm_pu``, so the
    # promoted-gen AVR setpoint stays at the default 1.03 pu set by
    # ``apply_der_classification`` for the entire run.  The TSO-side
    # ``install_qv_characteristic_controllers`` / ``install_cos_phi_one``
    # paths are gated on a non-empty list of TSO sgens and become no-ops
    # here (``_tso_der_idx_list`` is empty after promotion).
    "L2":  dict(
        tso_mode="local",
        dso_mode="local", local_der_mode="cos_phi_1",
        use_qv_local_loop=False,
    ),
    "L3":  dict(
        tso_mode="local",
        dso_mode="local", local_der_mode="qv",
        qv_setpoint_pu=1.03, qv_slope_pu=0.07,
        use_qv_local_loop=False,
    ),
    # ── OFO scenarios ───────────────────────────────────────────────────
    # T-OFO: TSO uses the new OFO (grid-forming gens stay).  DSO is
    # local with legacy Q(V) install, so disable the new plant loop here
    # too to avoid duplicate Q(V) controllers per sgen.
    #
    # Mitigations against the TSO-OFO / legacy-droop closed-loop
    # sensitivity error documented in
    # ``tests/diag_t_ofo_oscillation.py`` and
    # ``experiments/results/002_compare/diagnostics_T-OFO/summary.md``:
    #
    # H. g_w_pcc=1e8
    #    The DSO has no MIQP that tracks Q_PCC,set, so optimising it
    #    only winds up an unobservable variable (diagnostic 3 showed
    #    Q_PCC swing ~131 GVar with windup index 1.00).  A very large
    #    g_w_pcc pins each Q_PCC,set near its current value via the
    #    OFO regulariser — pure config override, no controller code
    #    changes.  Q_PCC stays effectively immobile while the OFO
    #    uses its physical actuators (V_gen, OLTC, shunt) to track V.
    #
    # D. tso_command_relaxation_alpha=0.3
    #    Under-relax the OFO step on continuous actuators only.
    #    Discrete actuators (OLTC, shunt) keep alpha=1 (cannot move
    #    fractional steps).  Damps the V_gen limit cycle at the AVR
    #    upper box bound (sign-flip fraction was 0.91 in zone 2).
    "T-OFO": dict(
        tso_mode="ofo",
        dso_mode="local", local_der_mode="qv",
        qv_setpoint_pu=1.03, qv_slope_pu=0.07,
        # Must be False: dso_mode="local" + local_der_mode="qv" already
        # runs apply_qv_local_control() at every DSO tick (manual q_mvar
        # update).  Leaving use_qv_local_loop=True would *additionally*
        # install a QVLocalLoop pp.controller per DSO sgen that iterates
        # inside every pp.runpp(run_control=True), and the two paths
        # fight each other -> ControllerNotConverged on step 1.
        use_qv_local_loop=False,
        g_w_pcc=1.0e8,
        tso_command_relaxation_alpha=1.0,
    ),
    # C-OFO: full Stage-1 + Stage-2 cascade architecture; keep the base
    # config's use_qv_local_loop=True / force_all_der_grid_following=False.
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
        :data:`SCENARIOS` (L0, L1, L2, L3, T-OFO, C-OFO).  Missing logs
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
