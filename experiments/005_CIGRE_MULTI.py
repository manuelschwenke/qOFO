#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/005_CIGRE_MULTI.py
==============================
Case-study driver for the CIGRE Energy Forum 2026 paper.  Runs the four
control variants of the cascaded multi-TSO / multi-STS OFO framework on the
IEEE 39-bus ``wind_replace`` system over a single 360-minute SimBench-driven
time series with the paper's contingency schedule, then renders the three
paper figures (PDF, Times New Roman) and the Section-5 summary table.

Variants (Table ``tab:variants``):

* **V1** -- TS local Q(V), STS DER at cos phi = 1            (002 key ``L0``)
* **V2** -- TS local Q(V), STS local Q(V) on DER             (002 key ``L1``)
* **V3** -- TS-OFO, STS local Q(V) on DER (one-sided OFO)    (002 key ``T1``)
* **V4** -- cascaded TS-OFO + STS-OFO (proposed)            (002 key ``C``)
* **V5** -- single centralized OFO over all zones + DSOs (best-case upper
  bound; ``control_scope='central'``)

All variants share ``make_cigre_config()`` (the user's tuned run config) and
differ only in the control layer.  Each variant is run by
``run_multi_tso_dso(cfg)`` and pickled to ``results/005_cigre/<V>/log.pkl``.

V5 is the **upper-bound reference**: one OFO controller owns every actuator
(all gen AVR, all TSO+DSO DER, all 2W machine OLTCs, all 3W coupler OLTCs, all
shunts) and observes every TN+HV bus, removing the information barriers and the
cascade-decomposition optimality gap of V1--V4.  Its sole objective is voltage
tracking, with ``g_v`` on TN buses and ``central_dso_g_v`` on HV buses.

Outputs
-------
* ``results/005_cigre/<V>/log.pkl``          -- per-variant record list.
* ``results/005_cigre/cigre_summary.csv``    -- Table ``tab:summary`` metrics.
* ``results/005_cigre/inventory.txt``        -- Table ``tab:inv`` / ``tab:params`` data.
* ``Fig3a_voltage_tracking.pdf`` / ``Fig3b_iface_tracking.pdf`` /
  ``Fig4_capability.pdf`` (+ ``_all``) / ``Fig5_tieflow.pdf`` -- written to
  BOTH ``results/005_cigre/`` and the paper's ``Figures/`` directory.

Live plotting is forced OFF for the batch sweep (a four-fold sequential run
with live plots is not viable headless); use the post-hoc PDFs instead.

CLI
---
* ``--only V1,V4``  -- run only the named variants (others picked up from disk).
* ``--skip V2``     -- inverse of ``--only``.
* ``--replot``      -- skip simulation, just rebuild figures + tables from the
                       existing ``log.pkl`` files (fast iteration on styling).

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

# Headless rendering for the batch sweep (must precede any matplotlib import,
# which is pulled in transitively via visualisation.style).
import os
os.environ["QT_API"] = "pyqt5"

import matplotlib as mpl
mpl.use("Qt5Agg")

import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Force UTF-8 console so Unicode in controller logs (e.g. the Greek Δ in
# capability messages) does not raise UnicodeEncodeError when stdout is
# redirected to a cp1252 file on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent, MultiTSOIterationRecord
from experiments.paths import results_path
from experiments.runners import run_multi_tso_dso
from analysis.reachability import ReachabilityViolation


# ---------------------------------------------------------------------------
#  User-facing knobs
# ---------------------------------------------------------------------------

#: Output directory inside the repo.
OUT_ROOT = results_path("005_cigre")

#: The CIGRE paper's figure directory (PDFs are written here too).
PAPER_FIG_DIR = r"C:\Users\Manuel Schwenke\Desktop\CIGRE_2026\Figures\dump"

#: Generators to show in the Fig. 4 *subset* (paper) plot.  Match by generator
#: name (str) or net.gen index (int).  ``None`` -> one representative per zone.
#: First inspect ``Fig4_capability_all.pdf`` (every machine), then set the 2-3
#: you want here and re-run with ``--replot``.
GEN_SELECT: Optional[List] = ["G3_bus31","G4_bus32", "G7_bus35"]

#: Voltage setpoint (matches MultiTSOConfig.v_setpoint_pu / paper V_ref).
V_SET = 1.03

#: Show the V5 centralized-reference trace in the Fig. 3b interface-flow plot
#: (measured flow drawn dotted).  Set False to show the proposed V4 alone.
IFACE_SHOW_V5 = False


# ---------------------------------------------------------------------------
#  Configuration (user's tuned run config, verbatim; live plots forced off)
# ---------------------------------------------------------------------------


def make_cigre_config() -> MultiTSOConfig:
    """Shared run configuration for all four CIGRE variants.

    Values are the user's tuned configuration for the 360-minute case study
    (``n_total_s = 60*60*6`` s).  ``dt_s`` is left at its 60 s default, so the
    plant is solved once per minute (360 steps), the STS-OFO fires every step
    and the TS-OFO every 6th step (``tso_period_s = 360`` s).
    """
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 5,      # 36-hour (2160-min) simulation
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=20.0,            # DSO-OFO each plant step (dt_s=60 >= 10)
        dt_s=20,
        g_v=1E7,                      # TSO voltage tracking; drives PCC Q dispatch
        g_q=200,                      # DSO Q-tracking
        tso_g_q_tie=0,
        tso_g_res_sg=0,
        # ── DSO objective tuning ──
        dso_g_v=1E5,              # reduced to avoid competing with Q tracking
        dso_g_qi=0,                   # integral Q-tracking (0 = off)
        dso_lambda_qi=0.95,           # leaky integrator decay
        dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,         # DER-primary, OLTC-backup
        # ── TSO weights (w-shift closed-loop curvature) ──
        g_w_der=100,
        g_w_gen=5e9,
        g_w_pcc=200,
        g_w_tso_oltc=1E4,
        # shunt
        install_tso_tertiary_shunts=False,
        shunt_dispatch="off", #"integrator"
        g_w_tso_shunt=12000,
        tso_shunt_kind="msc_msr",  # one capacitor + one reactor bank per DSO
        tso_shunt_msc_n_levels=2,  # MSC steps 0..N
        tso_shunt_msr_n_levels=2,  # MSR steps 0..N
        tso_shunt_msc_q_step_mvar=25.0,  # Mvar per MSC step
        tso_shunt_msr_q_step_mvar=25.0,  # Mvar per MSR step
        # tie coordination
        enable_tie_coordination=True,
        zone_v_setpoints_pu={1: 1.04, 2: 1.02, 3: 1.00},
        tie_grad_step=0.1, tie_anchor=0.5,
        tie_grad_eps=1E-4,
        # integrator tuning
        shunt_int_g_w=150,  # step = g_H/(2*g_w); SMALLER = bigger step — TUNE THIS
        shunt_int_delta_mvar=10.0,  # hysteresis half-width (must be < q_step/2 = 25)
        shunt_int_t_dwell_s=30*60.0,  # min seconds between commits per bank (anti-chatter)
        shunt_int_v_min_pu=0.90,  # HV feasibility band (overshoot guard)
        shunt_int_v_max_pu=1.10,
        # ── DSO weights ──
        g_w_dso_der=1000,
        g_w_dso_oltc=200,
        # ── Local-mode OLTC tap-rate limits (V1/V2 MT+NC, V3 NC) ──
        # max_step=1 (default) + wall-clock cooldown per OLTC type:
        #   MT (machine 2W gen-trafo) -> 1 tap / 180 s = once per TS interval.
        #   NC (coupler 3W interface) -> 1 tap / 60 s  = once per minute.
        # Cooldowns are wall-clock, hence robust to dt_s / dso_period_s.
        local_oltc_max_step_per_dt=1,
        oltc_cooldown_s_mt=180.0,
        oltc_cooldown_s_nc=60.0,
        use_fixed_zones=True,         # literature 3-area partition
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,
        verbose=1,
        tso_qv_vref_pu=1.03,
        tso_qv_slope_pu=0.06,
        tso_qv_deadband_pu=0.01,
        dso_qv_vref_pu=1.03,
        dso_qv_slope_pu=0.06,
        dso_qv_deadband_pu=0.01,
        # Live plots OFF for the batch sweep (see module docstring).
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        live_plot_tracking=False,
        live_plot_show_reserves=True,
        live_plot_show_tie_flows=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # Preconditioning of g_w
        precondition_g_w=False,  # turn it on
        precondition_lambda_target=0.5,  # target λ_max(M); 0.9 = well-damped, <2 stable
        # optional:
        precondition_granularity="class",  # or "column"
        precondition_exclude_classes=("gen",),  # AVR setpoint left at config
        # ── Profile & contingency settings ──
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            ContingencyEvent(minute=60, element_type="gen",  element_index=2,  action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=2,  action="restore"),
            # ContingencyEvent(minute=90, element_type="load", bus=15, p_mw=0, q_mvar=300, action="connect"),
            # ContingencyEvent(minute=360, element_type="load", bus=15, p_mw=0, q_mvar=300, action="trip"),
            ContingencyEvent(minute=90, element_type="load", bus=11, p_mw=200, q_mvar=100, action="connect"),
            ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=200, q_mvar=100, action="trip"),
            ContingencyEvent(minute=260, element_type="line", element_index=25, action="trip"),
            ContingencyEvent(minute=360, element_type="line", element_index=25, action="restore"),
        ],
    )
    cfg.scenario = "wind_replace"
    cfg.warmup_s = 0.0
    return cfg


#: Per-variant control-mode overrides (mirror 002_M_TSO_M_DSO_COMPARE.SCENARIOS).
VARIANTS: Dict[str, Dict[str, Any]] = {
    # "V1": dict(  # 002 "L0" -- TS Q(V), STS cos phi = 1
    #     tso_mode="local", tso_local_mode="qv",
    #     dso_mode="local", local_der_mode="cos_phi_1",
    #     tso_q_mode="qv", dso_q_mode="cosphi",
    #     tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
    # ),
    # "V2": dict(  # 002 "L1" -- TS Q(V), STS local Q(V)
    #     tso_mode="local", tso_local_mode="qv",
    #     dso_mode="local", local_der_mode="qv",
    #     tso_q_mode="qv", dso_q_mode="qv",
    #     tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
    #     dso_qv_vref_pu=1.03, dso_qv_slope_pu=0.06, dso_qv_deadband_pu=0.01,
    # ),
    # "V3": dict(  # 002 "T1" -- TS-OFO, STS local Q(V); g_w_pcc pin
    #     tso_mode="ofo",
    #     dso_mode="local", local_der_mode="qv",
    #     tso_q_mode="qv", dso_q_mode="qv",
    #     tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
    #     dso_qv_vref_pu=1.03, dso_qv_slope_pu=0.06, dso_qv_deadband_pu=0.01,
    #     g_w_pcc=1.0e10,
    # ),
    "V4": dict(  # 002 "C" -- cascaded TS-OFO + STS-OFO (proposed)
        tso_mode="ofo",
        dso_mode="ofo",
    ),
    "V4_NOCOORD": dict(  # 002 "C" -- cascaded TS-OFO + STS-OFO (proposed)
        tso_mode="ofo",
        dso_mode="ofo",
        enable_tie_coordination=False,
    ),
    # "V5": dict(  # single centralized OFO -- best-case upper-bound reference
    #     control_scope="central",
    #     tso_mode="ofo",
    #     dso_mode="local",
    #     tso_q_mode="qv", dso_q_mode="qv",
    #     local_sensitivities_tso=False,
    #     local_sensitivities_dso=False,
    #     # ── Tuning (plan golden-popping-hartmanis): make V5 a VALID upper
    #     #    bound (>= V4 on rms_v_ts_pu).  Strategy: match V4's per-loop gain
    #     #    ratios g_v/g_w, match V4's control cadence, then cool the whole g_w
    #     #    block by one global KAPPA_V5 (set from the lambda_max probe below).
    #     central_dso_g_v=1E5,        # HV voltage weight = V4 dso_g_v
    #     central_period_s=180,       # = V4 tso_period_s: cadence-matched comparison
    #     # -- V4 gain ratios restored (g_v/g_w per class), then x KAPPA_V5 on g_w --
    #     # g_v=1E7,                    # TN voltage weight = V4 g_v (was 5E7)
    #     # g_w_der=100,                # = V4 (TSO DER);  ratio g_v/g_w_der = 1e5
    #     # g_w_dso_der=1000,           # = V4 (DSO DER);  ratio cdso_g_v/g_w = 1e2
    #     # g_w_gen=5E9,                # = V4 (very cautious AVR)
    #     # g_w_tso_oltc=1E4,           # = V4 (was 2E4)
    #     # g_w_dso_oltc=200,           # = V4 (was 1E3)
    #     # debug_central_curvature=True,  # enable to print lambda_max(M) at t=0
    # ),
}

#: Global g_w cooling factor for V5 (plan Step 2).  V5 is monolithic, so the
#: same g_w that suits V4 makes lambda_max(M) > 1 (over-driven).  After matching
#: V4's per-class gain ratios above, the whole g_w block is scaled by KAPPA_V5
#: (>= 1 cools) so lambda_max(M) lands in the well-damped band (~1).  1.0 = the
#: raw V4-ratio starting point; set from the lambda_max probe (diag_v5_curvature.py).
KAPPA_V5: float = 1.0  # lambda_max(M)=1.237 at kappa=1 -> 1.25 gives ~0.99 (well-damped)
_V5_GW_KEYS = ("g_w_der", "g_w_dso_der", "g_w_gen", "g_w_tso_oltc", "g_w_dso_oltc")
if KAPPA_V5 != 1.0:
    for _k in _V5_GW_KEYS:
        VARIANTS["V5"][_k] = VARIANTS["V5"][_k] * KAPPA_V5

#: V5 at the fast cadence (every step) — text-only sensitivity result.  Identical
#: weights to V5 (the curvature tuning via lambda_max(M) is cadence-independent, so
#: the same g_v/g_w are well-damped at both rates); only ``central_period_s`` differs.
#: Reported in the console summary table for the paper text, NOT drawn in any figure
#: (see ``FIG_EXCLUDE``) so the headline figures keep V5@180s as the sole reference.
#VARIANTS["V5-20s"] = {**VARIANTS["V5"], "central_period_s": 20}

#: Variants kept out of the rendered figures (still appear in the summary table/CSV).
FIG_EXCLUDE = {} #{"V5"} #{"V5_20s"}


# ---------------------------------------------------------------------------
#  Run / load
# ---------------------------------------------------------------------------


def run_one(name: str, overrides: Dict[str, Any]) -> List[MultiTSOIterationRecord]:
    """Run a single variant and pickle its log.  Returns the log (``[]`` on failure)."""
    cfg = make_cigre_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    scen_dir = os.path.join(OUT_ROOT, name)
    cfg.result_dir = scen_dir
    os.makedirs(scen_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"  RUNNING VARIANT {name}  "
          f"(tso_mode={cfg.tso_mode}, dso_mode={cfg.dso_mode})")
    print("=" * 72)

    try:
        log = run_multi_tso_dso(cfg)
    except ReachabilityViolation as rv:
        # Voltage-stability (nose-curve) rejection -- distinct from a power-flow
        # divergence.  Keep the records computed before the violation and dump
        # the margin trajectory so the approach to the nose is inspectable.
        log = list(rv.partial_log) if rv.partial_log else []
        print(f"  [{name}] VOLTAGE-UNSTABLE (nose): {rv}")
        _dump_reach_margins(scen_dir, rv)
    except Exception as exc:  # noqa: BLE001 - persist failure, keep sweep alive
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
        log = []

    with open(os.path.join(scen_dir, "log.pkl"), "wb") as f:
        pickle.dump(log, f)
    print(f"  [{name}] wrote {len(log)} records -> {scen_dir}/log.pkl")
    return log


def _dump_reach_margins(scen_dir: str, rv: ReachabilityViolation) -> None:
    """Persist the per-step voltage-stability margin trajectory carried by a
    :class:`ReachabilityViolation` to ``<scen_dir>/reach_margins.csv``."""
    if not rv.margins:
        return
    import pandas as pd
    try:
        pd.DataFrame(rv.margins).to_csv(
            os.path.join(scen_dir, "reach_margins.csv"), index=False
        )
        print(f"  [reach] wrote {len(rv.margins)} margin rows -> "
              f"{scen_dir}/reach_margins.csv")
    except (PermissionError, OSError) as exc:  # never block the sweep on I/O
        print(f"  [reach] could not write reach_margins.csv: {exc}")


def load_logs(names: Optional[List[str]] = None
              ) -> Dict[str, List[MultiTSOIterationRecord]]:
    """Reload pickled variant logs from ``OUT_ROOT``.  Missing -> empty list."""
    if names is None:
        names = list(VARIANTS.keys())
    out: Dict[str, List[MultiTSOIterationRecord]] = {}
    for name in names:
        pkl = os.path.join(OUT_ROOT, name, "log.pkl")
        if not os.path.isfile(pkl):
            print(f"  [load_logs] missing {pkl} -- {name} treated as empty")
            out[name] = []
            continue
        with open(pkl, "rb") as f:
            out[name] = pickle.load(f)
        print(f"  [load_logs] {name}: {len(out[name])} records")
    return out


# ---------------------------------------------------------------------------
#  Tables (tab:inv, tab:params, tab:summary)
# ---------------------------------------------------------------------------


def _zone_of_bus(bus, bus_zone, net):
    """Zone of a bus, resolving generator LV-terminal buses via their trafo."""
    z = bus_zone.get(bus)
    if z is not None:
        return z
    for ti in net.trafo.index:
        if int(net.trafo.at[ti, "lv_bus"]) == bus:
            return bus_zone.get(int(net.trafo.at[ti, "hv_bus"]))
    return None


def build_inventory(cfg: MultiTSOConfig) -> str:
    """Best-effort Table ``tab:inv`` data: per-zone TS actuators + per-STS counts.

    Rebuilds the combined net exactly as ``run_multi_tso_dso`` does
    (``build_ieee39_net`` -> ``fixed_zone_partition_ieee39`` -> ``add_hv_networks``)
    and counts elements.  Returns a formatted text block; never raises.
    """
    lines: List[str] = ["== Table tab:inv (test system & actuator inventory) =="]
    try:
        from network.ieee39 import build_ieee39_net, add_hv_networks
        from network.zone_partition import fixed_zone_partition_ieee39, get_tie_lines

        net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario)
        zone_map, bus_zone = fixed_zone_partition_ieee39(net, verbose=False)
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
            verbose=False,
        )
        existing = set(net.bus.index)
        zone_map = {z: [b for b in bs if b in existing] for z, bs in zone_map.items()}

        hv_buses = {b for hv in meta.hv_networks for b in hv.bus_indices}
        hv_sgens = {s for hv in meta.hv_networks for s in hv.sgen_indices}

        # tn-only zone bus sets for tie-line counting
        tn_zone = {z: [b for b in bs if b not in hv_buses] for z, bs in zone_map.items()}
        zsorted = sorted(zone_map)
        tie_count = {z: 0 for z in zsorted}
        for i, zi in enumerate(zsorted):
            for zj in zsorted[i + 1:]:
                t = get_tie_lines(net, set(tn_zone[zi]), set(tn_zone[zj]))
                tie_count[zi] += len(t)
                tie_count[zj] += len(t)

        # Per-zone synchronous machines + machine transformers.
        n_avr = {z: 0 for z in zsorted}
        for g in net.gen.index:
            z = _zone_of_bus(int(net.gen.at[g, "bus"]), bus_zone, net)
            if z in n_avr:
                n_avr[z] += 1
        n_mt = {z: 0 for z in zsorted}
        for ti, gi in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
            if gi is None or int(gi) < 0:
                continue
            z = _zone_of_bus(int(net.gen.at[int(gi), "bus"]), bus_zone, net)
            if z in n_mt:
                n_mt[z] += 1
        # Per-zone TS DER (sgen not in any HV sub-network).
        n_der_ts = {z: 0 for z in zsorted}
        for s in net.sgen.index:
            if s in hv_sgens:
                continue
            z = _zone_of_bus(int(net.sgen.at[s, "bus"]), bus_zone, net)
            if z in n_der_ts:
                n_der_ts[z] += 1

        lines.append(f"{'Zone':<6}{'n_AVR':>7}{'n_MT':>6}{'n_DER_TS':>10}{'n_tie':>7}")
        for z in zsorted:
            lines.append(f"T{z:<5}{n_avr[z]:>7}{n_mt[z]:>6}{n_der_ts[z]:>10}{tie_count[z]:>7}")

        lines.append("")
        lines.append(f"{'STS':<8}{'zone':>5}{'n_DER':>7}{'n_NC':>6}{'n_sh':>6}")
        for hv in meta.hv_networks:
            n_sh = sum(1 for sh in net.shunt.index
                       if int(net.shunt.at[sh, "bus"]) in set(hv.bus_indices))
            lines.append(f"{hv.net_id:<8}{hv.zone:>5}{len(hv.sgen_indices):>7}"
                         f"{len(hv.coupling_trafo_indices):>6}{n_sh:>6}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  (inventory build failed: {type(exc).__name__}: {exc})")
    return "\n".join(lines)


def params_block(cfg: MultiTSOConfig) -> str:
    """Table ``tab:params`` controller settings, read from the run config."""
    return "\n".join([
        "== Table tab:params (controller settings) ==",
        f"  TS-OFO update period   : {cfg.tso_period_s:.0f} s",
        f"  STS-OFO update period  : {cfg.dso_period_s:.0f} s "
        f"(dt_s={getattr(cfg, 'dt_s', 60.0):.0f} s -> fires each step)",
        f"  Local Q(V) deadband    : 0.005 p.u. (paper)  / run dv_db="
        f"{getattr(cfg, 'dso_qv_deadband_pu', 0.01)} p.u.",
        f"  g_v (TS V tracking)    : {cfg.g_v:g}",
        f"  g_q (STS Q tracking)   : {cfg.g_q:g}",
        f"  G_w  TS  der/gen/pcc/oltc/shunt : "
        f"{cfg.g_w_der:g}/{cfg.g_w_gen:g}/{cfg.g_w_pcc:g}/"
        f"{cfg.g_w_tso_oltc:g}/{cfg.g_w_tso_shunt:g}",
        f"  G_w  STS der/oltc      : {cfg.g_w_dso_der:g}/{cfg.g_w_dso_oltc:g}",
        f"  Sensitivity variant    : S3 (local per-area Jacobian)",
    ])


def per_variant_params_block(variants: Dict[str, Dict[str, Any]]) -> str:
    """Per-variant tuned-weight table (Table ``tab:params``, per-variant).

    The CIGRE comparison tunes each variant separately to the same
    controller-agnostic metric (``rms_v_ts_pu``), so the single
    :func:`params_block` (which reflects only the shared base config) is
    not enough — each variant's own ``g_v`` / ``g_w`` / cadence must be
    reported.  This applies each variant's overrides on top of a fresh
    :func:`make_cigre_config` and prints one row of the knobs that differ
    in practice across the variants.  V5's row reflects the global
    ``KAPPA_V5`` cooling already folded into ``VARIANTS['V5']``.
    """
    cols = ("scope", "tso", "dso", "T[s]", "g_v", "g_v_HV", "g_q",
            "gw_der", "gw_dder", "gw_gen", "gw_to", "gw_do")
    lines = ["== Table tab:params (per-variant tuned weights) ==",
             f"{'var':<5}" + "".join(f"{c:>9}" for c in cols)]
    for name, ov in variants.items():
        cfg = make_cigre_config()
        for k, v in ov.items():
            setattr(cfg, k, v)
        scope = getattr(cfg, "control_scope", "cascaded")
        period = (cfg.central_period_s if scope == "central"
                  and cfg.central_period_s is not None else cfg.tso_period_s)
        g_v_hv = cfg.central_dso_g_v if scope == "central" else cfg.dso_g_v
        row = [
            f"{scope[:8]:>9}", f"{cfg.tso_mode[:8]:>9}", f"{cfg.dso_mode[:8]:>9}",
            f"{period:>9.0f}", f"{cfg.g_v:>9.2g}", f"{g_v_hv:>9.2g}",
            f"{cfg.g_q:>9.2g}", f"{cfg.g_w_der:>9.2g}", f"{cfg.g_w_dso_der:>9.2g}",
            f"{cfg.g_w_gen:>9.2g}", f"{cfg.g_w_tso_oltc:>9.2g}",
            f"{cfg.g_w_dso_oltc:>9.2g}",
        ]
        lines.append(f"{name:<5}" + "".join(row))
    lines.append("  (g_v_HV = central_dso_g_v for V5, else dso_g_v; "
                 "T = central_period_s for V5, else tso_period_s)")
    return "\n".join(lines)


def write_tables(cfg: MultiTSOConfig,
                 logs: Dict[str, List[MultiTSOIterationRecord]]) -> None:
    """Write cigre_summary.csv and inventory.txt; echo to stdout."""
    from experiments.helpers.comparison_metrics import (
        cigre_summary_table, gen_s_rated_by_zone,
    )

    os.makedirs(OUT_ROOT, exist_ok=True)
    # Pass per-zone machine ratings so the table also reports m_bar_pu (headroom
    # in p.u. of S_n -- size-comparable, not dominated by the largest machine).
    try:
        gen_srated = gen_s_rated_by_zone(cfg.scenario)
    except Exception as exc:  # noqa: BLE001 - m_bar_pu is optional, never block
        print(f"  [write_tables] m_bar_pu ratings unavailable: "
              f"{type(exc).__name__}: {exc}")
        gen_srated = None
    summary = cigre_summary_table(logs, v_set=V_SET, gen_s_rated_mva=gen_srated)
    csv_path = os.path.join(OUT_ROOT, "cigre_summary.csv")
    try:
        summary.to_csv(csv_path)
        print(f"\n  wrote {csv_path}")
    except (PermissionError, OSError) as exc:
        print(f"  WARNING: could not write {csv_path}: {exc}")
    print("\n== Table tab:summary (per-variant metrics) ==")
    print(summary.to_string(float_format=lambda x: f"{x:.4g}"))

    inv = build_inventory(cfg)
    par = params_block(cfg)
    par_pv = per_variant_params_block(VARIANTS)
    print("\n" + inv + "\n\n" + par + "\n\n" + par_pv)
    try:
        with open(os.path.join(OUT_ROOT, "inventory.txt"), "w", encoding="utf-8") as f:
            f.write(inv + "\n\n" + par + "\n\n" + par_pv + "\n")
    except (PermissionError, OSError):
        pass


# ---------------------------------------------------------------------------
#  Plot orchestration
# ---------------------------------------------------------------------------


def build_event_markers(cfg: MultiTSOConfig):
    """Resolve the contingency schedule to ``(t_min, label)`` markers.

    Labels follow the paper convention: generators ``G<k>-T`` (trip) / ``-C``
    (reconnect), loads ``LO@B<bus>-C`` / ``-T`` (bus = IEEE number from
    ``net.bus['name']``), lines ``LI(<from>-<to>) T`` / ``C``.  Only events
    inside the simulated window (``effective_time_s <= n_total_s``) are kept.
    Best-effort: returns ``[]`` and never raises if the net cannot be built.
    """
    markers = []
    try:
        from network.ieee39 import build_ieee39_net
        net, _meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario)

        def _busno(idx):
            try:
                return net.bus.at[int(idx), "name"]
            except Exception:  # noqa: BLE001
                return int(idx) + 1

        for ev in cfg.contingencies:
            if ev.effective_time_s > cfg.n_total_s:
                continue
            t_min = ev.effective_time_s / 60.0
            sfx = "T" if ev.action == "trip" else "C"  # restore/connect -> C
            et = ev.element_type
            if et == "gen":
                gname = str(net.gen.at[int(ev.element_index), "name"]).split("_")[0]
                markers.append((t_min, f"{gname}-{sfx}"))
            elif et == "load":
                bus = ev.bus if ev.bus is not None else net.load.at[int(ev.element_index), "bus"]
                markers.append((t_min, f"LO@B{_busno(bus)}-{sfx}"))
            elif et == "line":
                li = int(ev.element_index)
                frm, to = _busno(net.line.at[li, "from_bus"]), _busno(net.line.at[li, "to_bus"])
                markers.append((t_min, f"LI({frm}-{to}) {sfx}"))
    except Exception as exc:  # noqa: BLE001
        print(f"  [build_event_markers] skipped ({type(exc).__name__}: {exc})")
        return []
    return markers


def make_figures(logs: Dict[str, List[MultiTSOIterationRecord]]) -> None:
    from visualisation.plot_cigre import make_cigre_figures

    out_dirs = [OUT_ROOT, PAPER_FIG_DIR]
    # Keep figure-only exclusions (e.g. V5_20s) out of every rendered figure;
    # they remain in the summary table via the unfiltered logs in write_tables.
    fig_logs = {k: v for k, v in logs.items() if k not in FIG_EXCLUDE}
    make_cigre_figures(
        fig_logs, out_dirs,
        scenario="wind_replace",
        gen_select=GEN_SELECT,
        v_set=V_SET,
        iface_show_v5=IFACE_SHOW_V5,
        events=build_event_markers(make_cigre_config()),
        png_dir=OUT_ROOT,
        tso_steps_only=True
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def _parse_csv_arg(argv: List[str], flag: str) -> Optional[List[str]]:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(f"{flag} requires a comma-separated value, e.g. {flag} V1,V4")
    return [s.strip() for s in argv[idx + 1].split(",") if s.strip()]


def main(only: Optional[List[str]] = None,
         skip: Optional[List[str]] = None) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    selected = list(VARIANTS.keys())
    if only is not None:
        bad = [s for s in only if s not in VARIANTS]
        if bad:
            raise ValueError(f"--only unknown variants {bad}; valid {list(VARIANTS)}")
        selected = [s for s in selected if s in only]
    if skip is not None:
        bad = [s for s in skip if s not in VARIANTS]
        if bad:
            raise ValueError(f"--skip unknown variants {bad}; valid {list(VARIANTS)}")
        selected = [s for s in selected if s not in skip]

    if not selected:
        print("[main] No variants selected; nothing to run.")
    else:
        print(f"[main] Running variants: {selected}")
        for name in selected:
            run_one(name, VARIANTS[name])

    logs = load_logs()  # build from every pickle on disk
    cfg = make_cigre_config()
    write_tables(cfg, logs)
    make_figures(logs)
    print(f"\n[done] Figures + tables written to {OUT_ROOT}/ and {PAPER_FIG_DIR}")


def replot() -> None:
    print("=" * 72)
    print(f"  REPLOT from {OUT_ROOT}/ (no simulation)")
    print("=" * 72)
    logs = load_logs()
    write_tables(make_cigre_config(), logs)
    make_figures(logs)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot()
    else:
        main(only=_parse_csv_arg(sys.argv, "--only"),
             skip=_parse_csv_arg(sys.argv, "--skip"))

