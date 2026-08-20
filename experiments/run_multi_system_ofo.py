#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/000_M_TSO_M_DSO.py
==============================
Multi-TSO / Multi-DSO OFO simulation entry point on the IEEE 39-bus network.

This script is the multi-zone analogue of ``experiments/001_S_TSO_S_DSO.py``.
It uses the same OFO controller infrastructure (TSOController, DSOController)
but orchestrates N=3 independent TSO zones via the MultiTSOCoordinator.  Each
zone has its own TSO controller and underlying DSO controllers, one per HV
sub-network (5 total: DSO_1..DSO_5 from add_hv_networks).

The simulation loop itself lives in :mod:`experiments.runners.multi_tso_dso`
so it can be shared with 002, 003, and the tuning pipeline.  This script
only defines the experiment-specific :func:`main` and :func:`main_comparison`
configurations and routes between them via ``--compare``.

Architecture (matches the multi-TSO theory in Schwenke / CIGRE 2026)
---------------------------------------------------------------------

    ┌──────────────────────────────────────────────────────────┐
    │              IEEE 39-bus network (plant)                 │
    │  Zone 1        │  Zone 2 (w/ DSOs) │  Zone 3             │
    │  TSOCtrl_1     │  TSOCtrl_2        │  TSOCtrl_3          │
    │  (4 gen incl.  │  ├── DSOCtrl_2_0  │  (4 gen)            │
    │   slack)       │  └── DSOCtrl_2_1  │                     │
    └──────────────────────────────────────────────────────────┘

Model of Q_PCC_set:

    TN backbone ─── primary bus ─── 3W trafo ─── MV bus ─── (dropped HV sub-network)
                       ▲                            ▲
                       │                            │
        Controllable Q_PCC,set                 Ward load
        actuator (DSO dispatch              = (-p_mv, -q_mv) cached
        commanded by TSO)                     (represents the dropped sub-net's
        ↑ what you asked about                 static draw at the MV side)

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Show every column
pd.set_option('display.max_columns', None)
# Show every row
pd.set_option('display.max_rows', None)
# Ensure the width is wide enough to prevent wrapping
pd.set_option('display.width', None)
# Show full content within a cell (don't truncate long strings)
pd.set_option('display.max_colwidth', None)

# ── Ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import MultiTSOConfig, apply_dso_v_relief
from experiments.helpers import ContingencyEvent
from experiments.results_io import new_run_dir
from experiments.runners import run_multi_tso_dso
from network.ieee39 import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sensitivity.network_reduction import THEVENIN_K_PER_CORRIDOR
from sbx_h.config import SBXConfig


# =============================================================================
#  Comparison: coordinated vs. uncoordinated Q_PCC
# =============================================================================

def main_comparison() -> None:
    """Run coordinated vs. uncoordinated Q_PCC scenarios and compare.

    Invoke from the project root::

        python experiments/000_M_TSO_M_DSO.py --compare
    """
    import dataclasses

    # ── Shared parameters (identical for both scenarios) ─────────────────
    base_kwargs = dict(
        n_total_s=60.0 * 60 * 6,      # 720-min full simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=20.0,    # DSO every 5 seconds (more inner iterations)
        dt_s=20.0,
        g_v=150000.0,  # TSO voltage tracking; drives PCC Q dispatch
        # ── DSO objective tuning ──
        dso_g_v=20000.0,  # reduced to avoid competing with Q tracking
        dso_gamma_oltc_q=0.0,  # OLTC Q-tracking attenuation: DER-primary, OLTC-backup
        # ── TSO weights (alpha=1, spectral rho(C)/2) ──
        g_w_der=10,   # single-DER zones; rho~C_jj=396 -> min 198
        g_w_gen=4e7,   # excluded from stability
        # ── DSO weights (alpha=1, rho(C_DER)=790 -> min 395) ──
        g_w_dso_der=1000,  # 8 correlated DER; sf~2.5 for smooth tracking
        g_w_dso_oltc=30,   # rho(C_OLTC)~1.1; higher for switching suppression
        use_fixed_zones=True,      # literature 3-area partition (not spectral)
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,  # refresh H_ij every N TSO steps
        verbose=1,
        live_plot_system=False,
        # ── Profile & contingency settings ───────────────────────────────
        start_time=datetime(2016, 4, 15, 12, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            # Example: trip line 0 at t=30 min, restore at t=60 min
            # ContingencyEvent(minute=100, element_type="line", element_index=8, action="trip"),
            # ContingencyEvent(minute=150, element_type="line", element_index=8, action="restore"),
            ContingencyEvent(minute=10, element_type="gen", element_index=5, action="trip"),
            ContingencyEvent(minute=180, element_type="gen", element_index=5, action="restore"),
            ContingencyEvent(minute=120, element_type="load", bus=5, p_mw=400, q_mvar=200, action="connect"),
            ContingencyEvent(minute=300, element_type="load", bus=5, p_mw=400, q_mvar=200, action="trip"),
            ContingencyEvent(minute=330, element_type="gen", element_index=4, action="trip"),
            ContingencyEvent(minute=420, element_type="gen", element_index=4, action="restore"),
            ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
            ContingencyEvent(minute=720, element_type="load", bus=7, p_mw=300, q_mvar=100, action="connect"),
            ContingencyEvent(minute=900, element_type="load", bus=7, p_mw=300, q_mvar=100, action="trip"),
        ],
    )

    # ── Scenario A: coordinated TSO-DSO ─────────────────────────────────
    cfg_a = MultiTSOConfig(
        **base_kwargs,
        g_q=200,
        g_w_pcc=100,
        g_w_tso_oltc=100,
        live_plot_controller=True,
        live_plot_cascade=True,
    )

    # ── Scenario B: local DSO control (DiscreteTapControl + cos phi=1) ──
    cfg_b = dataclasses.replace(
        cfg_a,
        dso_mode="local",       # local controllers instead of OFO
        g_w_pcc=1e5,           # TSO cannot dispatch Q_PCC (no coordination)
        g_q=0,
        g_w_tso_oltc=250,
        local_der_mode="cos_phi_1",  # unity power factor for HV-connected DER
        warmup_s=900.0,         # 15 min: let TSO OFO settle before baseline activates
        live_plot_controller=True,
        live_plot_cascade=True,
    )

    run_dir = new_run_dir(
        "run_multi_system_ofo_comparison",
        {"coordinated": cfg_a, "local_dso": cfg_b},
    )
    cfg_a.result_dir = str(run_dir.root)
    cfg_b.result_dir = str(run_dir.root)

    print("=" * 72)
    print("  Scenario A: Coordinated TSO-DSO (OFO, g_q=200)")
    print("=" * 72)
    log_a = run_multi_tso_dso(cfg_a)

    print()
    print("=" * 72)
    print("  Scenario B: Local DSO (DiscreteTapControl + cos φ = 1 HV-DER)")
    print("=" * 72)
    log_b = run_multi_tso_dso(cfg_b)
    with (run_dir.root / "records.pkl").open("wb") as handle:
        pickle.dump(
            {"coordinated": log_a, "local_dso": log_b},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ── Summary statistics ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Comparison Summary")
    print("=" * 72)
    v_sp = cfg_a.v_setpoint_pu
    for label, log in [("Coordinated", log_a), ("Local DSO", log_b)]:
        v_devs = []
        for r in log:
            for z in r.zone_v_min:
                v_devs.append(abs(r.zone_v_min[z] - v_sp))
                v_devs.append(abs(r.zone_v_max[z] - v_sp))
        v_devs = np.array(v_devs)
        violations_5pct = np.sum(v_devs > 0.05)
        print(f"  {label:15s}  mean|ΔV|={v_devs.mean():.4f} p.u.  "
              f"max|ΔV|={v_devs.max():.4f} p.u.  "
              f"steps>5%={violations_5pct}")

    # ── Extract generator limits for capability curve plot ──────────────
    # Build the network once to read gen table limits and zone assignments.
    net_tmp, _ = build_ieee39_net(
        ext_grid_vm_pu=cfg_a.v_setpoint_pu,
        scenario=cfg_a.scenario,
    )
    _, bus_zone_tmp = fixed_zone_partition_ieee39(net_tmp, verbose=False)
    # gen_info: list of dicts, one per generator, sorted by (zone, gen_idx)
    gen_info: List[Dict[str, Any]] = []
    for g_idx in net_tmp.gen.index:
        g_bus = int(net_tmp.gen.at[g_idx, 'bus'])
        # Generator may be on LV terminal bus behind a machine trafo;
        # walk up through trafos to find the TN bus that has a zone.
        zone = bus_zone_tmp.get(g_bus)
        if zone is None:
            # Check if connected via a 2W trafo (machine trafo)
            for ti in net_tmp.trafo.index:
                if int(net_tmp.trafo.at[ti, 'lv_bus']) == g_bus:
                    hv_bus = int(net_tmp.trafo.at[ti, 'hv_bus'])
                    zone = bus_zone_tmp.get(hv_bus)
                    if zone is not None:
                        break
        if zone is None:
            continue  # skip generators not assigned to any zone
        # Same capability parameters as run_multi_tso_dso (nameplate read
        # directly; build_ieee39_net guarantees sn_mva and max_p_mw are set).
        sn       = float(net_tmp.gen.at[g_idx, 'sn_mva'])
        p_max_mw = float(net_tmp.gen.at[g_idx, 'max_p_mw'])
        gen_info.append(dict(
            zone=zone,
            gen_idx=int(g_idx),
            name=net_tmp.gen.at[g_idx, 'name'] or f"Gen_{g_idx}",
            s_rated_mva=sn,
            p_max_mw=p_max_mw,
            p_min_mw=0.0,
            xd_pu=1.8,
            i_f_max_pu=2.7,
            beta=0.15,
            q0_pu=0.4,
        ))
    gen_info.sort(key=lambda g: (g["zone"], g["gen_idx"]))

    # ── Plot comparison ─────────────────────────────────────────────────
    from visualisation.plot_multi_tso import plot_coordination_comparison
    plot_coordination_comparison(
        log_a, log_b,
        label_a="Coordinated",
        label_b="Local DSO",
        v_setpoint_pu=cfg_a.v_setpoint_pu,
        contingencies=cfg_a.contingencies,
        gen_info=gen_info,
    )


# =============================================================================
#  Entry point
# =============================================================================

def make_config() -> MultiTSOConfig:
    """Run configuration for the default multi-TSO / multi-DSO run (edit here).

    Single place to change the horizon, objective weights, OFO timing,
    profile and contingency schedule for ``main()``.  ``main_comparison()``
    keeps its own paired config.
    """
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 24,      # 36-hour (2160-min) simulation
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=20.0,            # DSO-OFO each plant step (dt_s=60 >= 10)
        dt_s=20.0,
        g_v=1E7,                      # TSO voltage tracking; drives PCC Q dispatch
        g_q=250,                      # DSO Q-tracking
        tso_g_res_sg=0,
        tso_g_loss=0,
        # ── DSO objective tuning ──
        # BO-tuned: dso_v_priority = 0.7641326900293102 x the 1E5 reference.
        # This is the 4th reparam coordinate -- the only one written as a plain
        # field rather than through the preconditioner.  Note it is the one
        # coordinate the study did NOT identify (Spearman -0.149, p = 0.113),
        # so treat the 0.76 factor as unresolved rather than optimal.
        dso_g_v=1E5,#1E5,#76413.26900293102,  # was 1E5; avoid competing with Q tracking
        dso_gamma_oltc_q=0.0,         # DER-primary, OLTC-backup
        coordination_mode="sbx_h",
        sbx_config=SBXConfig(
            k_sched=2,
            q_band_mvar=10.0,
            p_support_eur_per_mvarh=5.0,
            v_hold_tolerance_pu=0.005,
            v_sag_threshold_pu=0.01,
            n_need=2,
            release_threshold_pu=0.001,
            escalation_cycles=4,
            w_track_factor=1.0,
        ),
        zone_v_setpoints_pu={1: 1.03, 2: 1.03, 3: 1.03},
        # ── TSO weights (w-shift closed-loop curvature) ──
        # 2026-08-13: the uniform ``zone_g_w_scale = {1: 0.3, 2: 0.3, 3: 0.3}``
        # was FOLDED INTO these values (x 0.3) and the knob set to None.  The
        # runner applies the zone scale to the whole TSO ``params.g_w`` vector
        # after controller construction, so for a *uniform* factor the two are
        # algebraically identical in the cascaded path -- this is a rename, not
        # a re-tuning; the run reproduces
        # ``results/007_tie_boundary/BEST_SO_FAR_2026-08-13.params.json``.
        #
        # Why remove it: a uniform g_w factor f scales lambda_max(M) by 1/f, so
        # it is the *same* direction as the BO coordinate ``tso_lambda`` (which
        # sets g_w through the curvature preconditioner).  Carrying both means
        # the coordinate does not mean what it says: with the scale in place a
        # search over lambda in [0.05, 1.20] actually explores an effective
        # [0.167, 4.0], i.e. past the hard OFO bound of 2.  Kept only for a
        # genuinely PER-ZONE (non-uniform) re-gain, which is what the field is
        # for; see 00_daily_log/2026-08-13_bo_thevenin_study_setup.md.
        g_w_der=15,          # 50   x 0.3
        g_w_gen=1e9,       # 5e9  x 0.3
        g_w_pcc=50,          # 150  x 0.3
        # KEPT at the hand-tuned value.  The 2026-08-03 switching calibration
        # recommended 2287.57 (median 5.625 tap ops/h, 6.2 % off its 6 ops/h
        # target) and it was written here, then reverted on measurement:
        #
        #   g_w_tso_oltc   worst ops/h   worst REVERSALS/h
        #   5000 (this)        6.429           0.804
        #   2287.57            8.036           6.429      <- 8x hunting
        #
        # calibrate_switching targets ``tap_ops_per_h`` only and is blind to
        # ``tap_reversals_per_h``, so it traded reversals for operation count.
        # Constraint g5b limits reversals to max(worst)*1.5 = 1.206 derived from
        # this reference, which 2287.57 violates 5.3x.  Since g_w_tso_oltc is
        # NOT a BO dimension it is fixed for the whole study, so that would have
        # made every trial infeasible -- observed in the Stage 4 smoke, where
        # g5a and g5b were both violated by 100 % of trials.
        #
        # 5000 gives 1.206 ops/h median (~1.5 real ops/day), far inside the
        # 20-30 ops/day maintenance ceiling, and is essentially reversal-free.
        # (x 0.3 with the rest of the TSO g_w block -- see the fold note above;
        # 5000 x 0.3 = 1500 is the same operating point, not a new calibration,
        # so the measured ops/h figures quoted here still apply.)
        g_w_tso_oltc=5E3, # 5E3 x 0.3
        # shunt
        install_tso_tertiary_shunts=True,
        shunt_dispatch="integrator", #"integrator"
        # Inert while shunt_dispatch="integrator" (read only on the MIQP path);
        # folded x 0.3 with the rest of the block so the record stays consistent.
        g_w_tso_shunt=3600,  # 12000 x 0.3
        tso_shunt_kind="msc_msr",  # one capacitor + one reactor bank per DSO
        tso_shunt_msc_n_levels=2,  # MSC steps 0..N
        tso_shunt_msr_n_levels=2,  # MSR steps 0..N
        tso_shunt_msc_q_step_mvar=25.0,  # Mvar per MSC step
        tso_shunt_msr_q_step_mvar=25.0,  # Mvar per MSR step
        # integrator tuning
        shunt_int_g_w=100,  # step = g_H/(2*g_w); SMALLER = bigger step — TUNE THIS
        shunt_int_delta_mvar=10.0,  # hysteresis half-width (must be < q_step/2 = 25)
        shunt_int_t_dwell_s=30*60.0,  # min seconds between commits per bank (anti-chatter)
        shunt_int_v_min_pu=0.90,  # HV feasibility band (overshoot guard)
        shunt_int_v_max_pu=1.10,
        # ── DSO weights ──
        g_w_dso_der=800, # 1000
        # NOT calibrated -- deliberately left at the hand-tuned value.  The
        # 2026-08-03 bisection returned g_w=54.74 but only reached 4.821 ops/h,
        # 19.6 % off the 6 ops/h target at --tol-rel 0.1, because the DSO
        # response *skips* the target band: g_w 52.33 -> 7.232 (9 ops) drops
        # straight to 54.74 -> 4.821 (6 ops), and 7 ops (5.625) / 8 ops (6.429)
        # are the only achievable values inside [5.4, 6.6].  The ladder is also
        # locally non-monotone there (62.64 -> 3.616 but 74.99 -> 4.018), so the
        # bisection's premise does not hold at this resolution.  150 gives
        # 1.206 ops/h ~ 1.5 real ops/day.
        g_w_dso_oltc=200, #200
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
        # Live plotting on (controller + cascade); system overview off.
        live_plot_controller=True,
        live_plot_cascade=True,
        live_plot_system=False,
        live_plot_tracking=False,
        live_plot_sbx=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # ── Boundary equivalent for neighbouring TS areas ───────────────────
        # Each neighbouring area is condensed either as a constant PQ injection
        # ("pq", the shipped default behind every published result) or as a
        # voltage source behind the measured per-corridor impedance
        # ("thevenin"; THEVENIN_K_PER_CORRIDOR is keyed by
        # (line_idx, far_end_bus), measured in 007f).
        #
        # The two settings below MUST be changed together.  The Thevenin H is
        # smaller, so at the PQ gain the loop is under-driven; re-gained, the
        # two boundaries land within ~3% of each other on the tuning objective
        # (007p).  PQ stays the reference because it needs no per-corridor
        # impedance agreed with the neighbour, not because it controls better.
        # Evidence: 00_daily_log/2026-08-11_tie_boundary_equivalent.md
        #
        # The Thevenin re-gain used to live in ``zone_g_w_scale`` as a uniform
        # 0.3 across all three zones.  It is now folded into the ``g_w_*`` block
        # above (2026-08-13) and the knob is None: a uniform factor is exactly
        # the loop-gain direction, so carrying it separately double-counts the
        # gain and de-calibrates the BO coordinate ``tso_lambda``.  Reinstate
        # ``zone_g_w_scale`` only for a genuinely per-zone (non-uniform) re-gain.
        #
        # For the record, the 007* per-zone scales were tuned against
        # ``make_cigre_config``, whose step weights are heavier (g_w_der 100 vs
        # 50, g_w_pcc 200 vs 80, g_w_tso_oltc 10000 vs 5000 -- pre-fold values
        # here), so applying 007p's raw 0.07/0.07/0.15 lands on an absolute gain
        # 2-2.5x too high and the loop oscillates.  The translation is
        # ``scale_here = scale_007p * (g_w_cigre / g_w_here)`` -> ~0.15, and
        # zone 3 -> ~0.3.  Reproduce 007p exactly by running against
        # make_cigre_config + VARIANTS["V4"] with
        # zone_g_w_scale={1: 0.07, 2: 0.07, 3: 0.15}.
        #
        # Shipped, known-good behaviour: tie_boundary_equivalent="pq" with the
        # PRE-fold weights (g_w_der 50, g_w_pcc 150, g_w_tso_oltc 5000) and no
        # zone scale.
        #
        tie_boundary_equivalent="thevenin",
        zone_g_w_scale=None,
        tie_thevenin_k=THEVENIN_K_PER_CORRIDOR,
        # ── Preconditioning of g_w ──────────────────────────────────────────
        # BO-tuned point, study v5_reparam_v2 trial 173 (2026-08-05).  All SIX
        # fields below are required together: the tuned point is *defined* in
        # terms of the preconditioner, because lambda has no scalar realisation
        # (kappa depends on the cached sensitivity H).  See
        # docs/tuning/RESULTS_bo_retuning_2026-08.md.
        #
        # Two of these fail SILENTLY if left at their old values -- no error,
        # just reference behaviour:
        #   * mode='cap' only ever *reduces* g_w to meet the target.  The tuned
        #     lambda (1.1975) is ABOVE the reference 0.5, so 'cap' cannot raise
        #     the gain and the coordinate stays inert.  'set' binds both ways.
        #   * scope='all' lets the integer OLTC columns block the target
        #     outright (TSO zone 1 reads integer_dominated at 1.085 while its
        #     continuous loop sits at 0.021), which also leaves it inert.
        precondition_g_w = False,
        precondition_mode = "set",                 # NOT "cap" -- see above
        precondition_lambda_scope = "preconditioned",   # NOT "all"
        precondition_granularity = "column",       # NOT "class"
        precondition_lambda_target_tso = 0.9, #1.1975421798462904,
        precondition_lambda_target_dso = 0.9, #1.0964048871681646,
        # tau_der_pcc = 0.01618211331204804, carried as (sqrt(tau), 1/sqrt(tau)).
        # The geometric mean is pinned at 1 so this moves only the DER/PCC ratio
        # and leaves the loop gain to lambda.  PCC weight ends up ~62x the DER
        # weight -- the coordinate with the strongest signal (Spearman +0.794).
        precondition_class_scales = {"der": 1.0, "pcc": 1.0}, #{"der": 0.12721, "pcc": 7.86102},
        precondition_exclude_classes = ("gen",),  # AVR setpoint left at config
        # ── Profile & contingency settings ──
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            # 400 Mvar reactive load step at the zone-2 / L14 boundary bus (bus 9)
            # at t=120 min: stresses zone 2's boundary voltage so the TSO-TSO
            # coordinator must act (zone 1, slack-rich, supports across L14).
            # ContingencyEvent(minute=20, element_type="load", bus=9,
            #                  p_mw=0, q_mvar=400, action="connect"),
            # --- further examples (disabled) ---
            ContingencyEvent(minute=30, element_type="gen",  element_index=2,  action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=2,  action="restore"),
            ContingencyEvent(minute=90, element_type="load", bus=11, p_mw=0, q_mvar=250, action="connect"),
            ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=0, q_mvar=250, action="trip"),
            ContingencyEvent(minute=150, element_type="load", bus=11, p_mw=150, q_mvar=100, action="connect"),
            ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=150, q_mvar=100, action="trip"),
            ContingencyEvent(minute=210, element_type="line", element_index=25, action="trip"),
            ContingencyEvent(minute=300, element_type="line", element_index=25, action="restore"),
        ],
    )
    return cfg


DSO_V_RELIEF_FACTORS: Dict[str, float] = {"DSO_2": 20.0, "DSO_4": 20.0}
"""Per-area voltage-authority factors applied by :func:`_apply_dso_v_relief`.

Each factor scales that area's ``dso_g_v`` **and** its ``dso_oltc`` step weight
by the same amount, so the OLTC loop gain ``dso_g_v / g_w_dso_oltc`` is
unchanged and the extra authority lands on the continuous DER block.  Drop an
area from the mapping (or set its factor to ``1.0``) to leave it untouched.

**Listed areas are the spread-limited ones, and that is the selection rule.**
An HV network's internal voltage spread ``max_i V_i - min_i V_i`` is the part of
its profile no tap changer can remove (the OLTC translates the whole profile),
so an area whose spread is a large fraction of the 0.20 p.u. corridor has almost
no margin left to place.  Measured 2026-08-18, spread max / min headroom:

    DSO_1  0.015 / +0.048     DSO_2  0.117 / -0.001
    DSO_3  0.037 / +0.039     DSO_4  0.147 / -0.001

Applying the factor to DSO_1 and DSO_3 as well (run I) bought 0.0016 and 0.0002
p.u. of ``V_max`` and cost DSO_3 **+53 %** interface-Q RMSE, so they are
deliberately absent: voltage authority on a network that is not spread-limited
only competes with interface-Q tracking.  Rule of thumb from that measurement --
apply above ~0.10 p.u. of spread, leave alone below ~0.04.

Rationale and full measurements:
``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md``.
"""


def _apply_dso_v_relief(
    cfg: MultiTSOConfig,
    factors: Optional[Dict[str, float]] = None,
) -> MultiTSOConfig:
    """:func:`configs.config.apply_dso_v_relief` with this module's defaults.

    The implementation lives in ``configs.config`` because the Stage-1 config
    builder needs it too and must not import an experiment module.
    """
    return apply_dso_v_relief(
        cfg, DSO_V_RELIEF_FACTORS if factors is None else factors)


def make_config_tuned() -> MultiTSOConfig:
    """Run configuration for the default multi-TSO / multi-DSO run (edit here).

    Single place to change the horizon, objective weights, OFO timing,
    profile and contingency schedule for ``main()``.  ``main_comparison()``
    keeps its own paired config.
    """
    cfg = MultiTSOConfig(
        # time
        n_total_s=60.0 * 60 * 24,      # 36-hour (2160-min) simulation
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=20.0,            # DSO-OFO each plant step (dt_s=60 >= 10)
        dt_s=20.0,
        # objective weights
        g_v=1E7,                      # TSO voltage tracking; drives PCC Q dispatch
        g_q=250,                      # DSO Q-tracking
        dso_gamma_oltc_q=0.0,         # DER-primary, OLTC-backup
        dso_g_v=84140,#1E5,
        # tso weights
        g_w_der=10.2,  # was 20
        g_w_pcc = 49.3,  # was 60
        g_w_gen=1e9,
        # ── DSO weights ──
        g_w_dso_der=550,  # campaign optimum 549.88 (was 617)
        # zeroing to be sure
        tso_g_res_sg=0,
        tso_g_loss=0,
        # shunt
        install_tso_tertiary_shunts=True,
        shunt_dispatch="integrator", #"integrator"
        tso_shunt_kind="msc_msr",  # one capacitor + one reactor bank per DSO
        tso_shunt_msc_n_levels=2,  # MSC steps 0..N
        tso_shunt_msr_n_levels=2,  # MSR steps 0..N
        tso_shunt_msc_q_step_mvar=25.0,  # Mvar per MSC step
        tso_shunt_msr_q_step_mvar=25.0,  # Mvar per MSR step
        shunt_int_g_w=100,  # step = g_H/(2*g_w); SMALLER = bigger step — TUNE THIS
        shunt_int_delta_mvar=10.0,  # hysteresis half-width (must be < q_step/2 = 25)
        shunt_int_t_dwell_s=30*60.0,  # min seconds between commits per bank (anti-chatter)
        shunt_int_v_min_pu=0.90,  # HV feasibility band (overshoot guard)
        shunt_int_v_max_pu=1.10,
        # OLTC weights and settings
        g_w_tso_oltc = 3783,  # unchanged — see note
        g_w_dso_oltc = 150,  # was 150
        # ── DSO_4 voltage relief (2026-08-18) ───────────────────────────────
        # DSO_4 is the long-line area (SUBNET_DEFS scale=2.44 -> 586 km,
        # X = 222.5 ohm = 1.84 p.u., unreinforced 305 mm2) carrying the same
        # load and the same 700 MW of DER as the compact ones.  Its INTERNAL
        # voltage spread (V_max - V_min across its ten HV buses) is 0.147 p.u.
        # -- 73 % of the whole [0.90, 1.10] band -- so tracking V_set = 1.03
        # puts the top of the profile on the upper bound by arithmetic
        # (1.03 + 0.147/2 = 1.103).  Measured: V_max pinned at 1.1001 for 58 %
        # of a 6 h day.
        #
        # The pair below raises DSO_4's voltage-tracking weight x20 while
        # raising its OLTC step weight by the SAME x20, so the OLTC loop gain
        # dso_g_v / g_w_dso_oltc is unchanged.  The extra authority therefore
        # lands on the CONTINUOUS DER block, where it reshapes which of the ten
        # DERs injects (aggregate Q_DER unchanged, 63.3 -> 60.9 Mvar) and so
        # shrinks the spread, rather than on the integer tap.
        #
        #   V_max  1.1010 -> 1.0629      V_min  0.9531 -> 1.0027
        #   spread 0.147  -> 0.058       tap    2.33   -> 0.67 ops/h, 0 reversals
        #   interface-Q RMSE 6.08 -> 6.59 Mvar (+8 %), losses 40.68 -> 39.86 MW
        #
        # Zone 1/2/3 EHV envelopes, DSO 1/2/3 voltages, their interface-Q RMSEs
        # and their tap rates are unchanged to four decimals -- this is local.
        #
        # DO NOT raise dso_g_v_per_area without the matching dso_oltc entry:
        # that configuration (x6.7 on g_v alone) limit-cycles the tap at 50.5
        # reversals/h.  The runner warns if the two ratios disagree.
        #
        # NOT re-measured on the tuning_mc scenario banks yet -- single 6 h
        # window, 2016-05-03 08:00.  See
        # docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md.
        #
        # The pair itself is applied by _apply_dso_v_relief() below, derived
        # from this config's own dso_g_v / g_w_dso_oltc, so the loop-gain
        # invariant survives an edit to either base value.
        local_oltc_max_step_per_dt=1,
        oltc_cooldown_s_mt=180.0,
        oltc_cooldown_s_nc=60.0,
        # preconditioning
        precondition_g_w=False,
        precondition_mode="set",  # NOT "cap" -- see above
        precondition_lambda_scope="preconditioned",  # NOT "all"
        precondition_granularity="column",  # NOT "class"
        precondition_lambda_target_tso=0.5012,  # 1.1975421798462904,
        precondition_lambda_target_dso=0.8509,  # 1.0964048871681646,
        precondition_class_scales={'der': 0.13225, 'pcc': 7.5617},
        precondition_exclude_classes=("gen",),  # AVR setpoint left at config
        # H update
        sensitivity_update_interval=1E6,
        # verbosity
        verbose=1,
        # Live plotting on (controller + cascade); system overview off.
        live_plot_controller=True,
        live_plot_cascade=True,
        live_plot_system=False,
        live_plot_tracking=False,
        live_plot_sbx=False,
        # Local Sensitivities
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # ── Boundary equivalent for neighbouring TS areas ───────────────────
        tie_boundary_equivalent="thevenin",
        tie_thevenin_k=THEVENIN_K_PER_CORRIDOR,
        zone_v_setpoints_pu={1: 1.03, 2: 1.03, 3: 1.03},
        coordination_mode="sbx_h",
        sbx_config=SBXConfig(
            k_sched=2,
            q_band_mvar=10.0,
            p_support_eur_per_mvarh=5.0,
            v_hold_tolerance_pu=0.005,
            v_sag_threshold_pu=0.01,
            n_need=2,
            release_threshold_pu=0.001,
            escalation_cycles=4,
            w_track_factor=1.0,
        ),
        # ── Profile & contingency settings ──
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_fixed_zones=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            # 400 Mvar reactive load step at the zone-2 / L14 boundary bus (bus 9)
            # at t=120 min: stresses zone 2's boundary voltage so the TSO-TSO
            # coordinator must act (zone 1, slack-rich, supports across L14).
            # ContingencyEvent(minute=20, element_type="load", bus=9,
            #                  p_mw=0, q_mvar=400, action="connect"),
            # --- further examples (disabled) ---
            ContingencyEvent(minute=30, element_type="gen",  element_index=2,  action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=2,  action="restore"),
            ContingencyEvent(minute=90, element_type="load", bus=11, p_mw=0, q_mvar=250, action="connect"),
            ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=0, q_mvar=250, action="trip"),
            # ContingencyEvent(minute=150, element_type="load", bus=11, p_mw=150, q_mvar=100, action="connect"),
            # ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=150, q_mvar=100, action="trip"),
            ContingencyEvent(minute=210, element_type="line", element_index=25, action="trip"),
            ContingencyEvent(minute=300, element_type="line", element_index=25, action="restore"),
        ],
    )
    return _apply_dso_v_relief(cfg)


# ---------------------------------------------------------------------------
#  Gauge normalisation of the weight group
# ---------------------------------------------------------------------------
#  ``tuning/reparam.py`` establishes (measured 2026-07-31, trajectory reproduced
#  to ~4e-10 including the integer tap sequence) that scaling
#
#      {g_v, tso_g_q_pcc, g_q, dso_g_v} u {g_w_*} u {g_z_*} u {shunt_int_g_w}
#
#  by a COMMON factor changes nothing: the MIQP feasible set contains no weight,
#  so the minimiser of ``w'G_w w + grad_f'w + z'G_z z`` is invariant when every
#  term scales together.  The common factor is therefore a gauge, and the only
#  quantities that mean anything are the RATIOS inside the group.
#
#  Consequence for "normalising g_v, g_q and dso_g_v": they cannot be normalised
#  independently.  Their ratios are identifiable (they are the objective
#  trade-off) and must be preserved; only the one common factor is free.  What
#  a normalisation can do is fix that factor at a value with a stated meaning,
#  and then report the other two in interpretable terms.
#
#  The convention used here is the project's own priority reading from
#  ``tuning.reparam.PriorityScales``: a weight becomes dimensionless as
#  ``pi = g * sigma^2`` with ``sigma`` the engineering tolerance on that output.
#  Fixing ``pi_v_ts = 1`` gives
#
#      g_v := 1 / sigma_v_ts^2 ,
#
#  i.e. "one unit of weighted objective = one TSO voltage tolerance squared",
#  and every other weight in the group is then read against that unit.  With the
#  hand-tuned ``g_v = 1e7`` this is a factor 4e-3.
#
#  Side benefit, and the only legitimate criterion for the gauge itself: the
#  G_w diagonal moves from [13, 1e9] (~1e8 spread, entirely above 1) to
#  [0.05, 4e6], which straddles 1 and is better conditioned for the solver.
SIGMA_V_TS_PU = 0.005
"""TSO voltage tolerance [pu] that fixes the gauge (``reparam.PriorityScales``)."""

_G_REF_G_V = 1e7
"""Hand-tuned ``g_v`` the un-normalised weights below are quoted against."""

GAUGE = 1.0 / (SIGMA_V_TS_PU ** 2) / _G_REF_G_V     # = 4e-3


def _gauged(**weights: float) -> Dict[str, float]:
    """Scale every member of the exact-scaling group by :data:`GAUGE`.

    Applied programmatically rather than by hand-multiplying literals so the
    un-normalised values stay visible and readable at the call site, and so the
    invariance cannot drift if one number is later edited.
    """
    return {k: v * GAUGE for k, v in weights.items()}


def _gauged_area(spec: Dict[Any, Dict[str, float]]) -> Dict[Any, Dict[str, float]]:
    """:func:`_gauged` for the nested per-area ``{area: {class: g_w}}`` maps."""
    return {a: {c: v * GAUGE for c, v in d.items()} for a, d in spec.items()}


def make_config_per_area() -> MultiTSOConfig:
    """``make_config_tuned`` with the weight gauge normalised and the per-area,
    per-class ``g_w`` design applied.

    Two changes against :func:`make_config_tuned`, both derived rather than
    hand-tuned:

    1. **Gauge normalised** — every member of the exact-scaling group is scaled
       by :data:`GAUGE` so that ``g_v = 1/sigma_v_ts^2``.  This is a no-op on the
       trajectory by construction (see the note above ``GAUGE``); it only makes
       the weights readable as priorities and improves the ``G_w`` conditioning.
       The un-normalised values are the literals passed to :func:`_gauged`, so
       the diff against ``make_config_tuned`` stays auditable.

    2. **Per-area ``g_w``** — ``zone_g_w_class`` / ``dso_g_w_class`` carry the
       per-column analytic design aggregated per control area, instead of one
       scalar per class for the whole system.  Needed because a single scalar
       per area (``zone_g_w_scale``) cannot express an area whose classes want
       to move in opposite directions: measured at this operating point, TSO
       zone 1 wants ``der`` well below and ``tso_oltc`` well above the global
       value, a spread no single factor absorbs.

    Regenerate the per-area block after any change to the boundary equivalent,
    the zone partition or the start time — all three change ``H``, and the
    design is a function of ``H``::

        python -m tuning_mc.stage_0_preconditioning \\
            --from-runner make_config_per_area --per-area

    and paste the ``zone_g_w_class`` / ``dso_g_w_class`` blocks it prints.  The
    global ``g_w_<class>`` scalars are kept as the fallback for any area or
    class the per-area block does not list.
    """
    # Undo the DSO_4 relief before the gauge/per-area rewrite and re-apply it
    # afterwards (see the tail of this function).  Applying it first would
    # leave a raised dso_g_v against the per-area dso_oltc design below, i.e.
    # the unmatched loop gain that limit-cycles the tap.
    # Do NOT hand-write the per-area dicts here: the ``dataclasses.replace``
    # below overwrites ``dso_g_w_class`` with ``_gauged_area(...)``, and
    # ``_apply_dso_v_relief`` then recomputes ``dso_g_v_per_area`` from this
    # config's own (gauged) ``dso_g_v``, so literals are silently discarded.
    # Strip here, re-apply after the rewrite -- that ordering is what keeps the
    # relief measured against each area's per-area ``dso_oltc`` base.
    cfg = dataclasses.replace(
        make_config_tuned(), dso_g_v_per_area=None, dso_g_w_class=None,
    )
    return _apply_dso_v_relief(dataclasses.replace(
        cfg,
        # ── Gauge-normalised weight group ───────────────────────────────────
        # Objective (output) weights.  Ratios preserved exactly; only the
        # common factor moves.  pi = g*sigma^2 with the reparam tolerances
        # (sigma_v_ts=0.005, sigma_v_ds=0.010, sigma_q=5.0 pu/pu/Mvar):
        #   pi_v_ts = 1.0     (the unit, by construction)
        #   pi_q    = 25.0    interface-Q priced 25x the TSO voltage unit
        #   pi_v_ds = 0.04    the DSO voltage schedule, 625x below its own Q
        # That 625x is the "interface-Q dominates the DSO objective" statement
        # in defensible form; change it by moving g_q/dso_g_v, never by moving
        # the gauge.
        **_gauged(
            g_v=1e7,
            g_q=250.0,
            dso_g_v=1e5,
            # ── TSO g_w ────────────────────────────────────────────────────
            # FALLBACKS ONLY.  zone_g_w_class / dso_g_w_class below cover every
            # area and class that exists at this operating point, so nothing
            # reads these unless a zone or DSO is added.  They are set to the
            # SAME design's global aggregate (geometric mean per class; max for
            # gen, which is a bound) rather than left at the hand-tuned values
            # of make_config_tuned -- a new zone should inherit the design's
            # regime, not a value from a different shape.  Hand-tuned values
            # for reference: der 13, pcc 20, gen 1e9, tso_oltc 5e3,
            # dso_der 1200, dso_oltc 180.
            g_w_der=13,
            g_w_pcc=20,
            g_w_gen=9.6e8,
            g_w_tso_oltc=8732,
            g_w_tso_shunt=1e4,      # inert under shunt_dispatch="integrator"
            # ── DSO g_w ────────────────────────────────────────────────────
            g_w_dso_der=1653,
            g_w_dso_oltc=224,
            # ── Shunt integrator: step = g_H/(2 g_w), g_H is linear in g_v ──
            shunt_int_g_w=150,#42.0,
            # ── Output-slack weights.  In the group: G_z must scale with G_w
            # and grad_f or the tracking/violation trade-off shifts. ──
            g_z_voltage=1e9,
            g_z_q_gen=1e2,
            g_z_q_pcc=1e6,
            # g_z_current / g_z_interface are 0.0 -> gauge-invariant, omitted.
        ),
        # ── Per-area, per-class g_w ─────────────────────────────────────────
        # Generated 2026-08-14 by
        #   python -m tuning_mc.stage_0_preconditioning \
        #       --from-runner make_config_per_area --per-area
        # at start_time 2016-01-05 08:00, tie_boundary_equivalent="thevenin",
        # local_sensitivities_tso/dso=True.  Quoted in the REFERENCE gauge
        # (g_v = 1e7) like every other weight above, so _gauged_area keeps them
        # consistent with the globals if SIGMA_V_TS_PU ever moves.
        #
        # Where each number comes from -- three different rules, not one:
        #   der / pcc  : curvature rule, per column, at the config's own
        #                lambda_target_tso=0.5012 and class_scales
        #                {der: 0.13225, pcc: 7.5617} (= tau 0.017484).
        #   gen        : per-step move budget, |du| <= 1e-3 pu at a 2 % per-bus
        #                reference error -- the AVR is excluded from the
        #                curvature rule, so this is its only rule.
        #   tso_oltc   : commit-threshold rule at a 1.5 % systematic offset.
        #
        # READ THIS BEFORE RUNNING.  The der/pcc numbers are a SUBSTANTIVE
        # re-gain, not a refinement.  ``make_config_tuned`` declares BO-tuned
        # precondition_* fields but sets ``precondition_g_w=False``, so the
        # tuned DER/PCC shape (~57x PCC over DER) is currently INERT and the
        # run actually uses g_w_der=13 / g_w_pcc=20, a ratio of 1.5.  Writing
        # the design into zone_g_w_class applies that tuned shape statically,
        # which is the point -- but it means DER gets ~5-19x more authority and
        # PCC ~7-10x less than the run does today.  To keep the analytic shape
        # instead of the BO-tuned one, regenerate with ``--tau 1.0``.
        zone_g_w_class=_gauged_area({
            1: {"der": 0.39,                  "gen": 4.9e8, "tso_oltc": 6420.0},
            2: {"der": 12.6, "pcc": 11.12,    "gen": 9.6e8, "tso_oltc": 1467.0},
            3: {"der": 2.62, "pcc": 5.05,     "gen": 5.9e8, "tso_oltc": 8732.0},
        }),
        dso_g_w_class=_gauged_area({
            "DSO_1": {"dso_der": 1057.0, "dso_oltc": 185.0},
            "DSO_2": {"dso_der": 1128.0, "dso_oltc": 181.0},
            "DSO_3": {"dso_der": 997.0, "dso_oltc": 189.0},
            "DSO_4": {"dso_der": 1219.0, "dso_oltc": 178.0},
        }),
        zone_g_w_scale=None,        # superseded by zone_g_w_class
    ))


def main() -> None:
    """
    Run the multi-TSO-DSO simulation with default settings and print results.

    Invoke from the project root:
        python experiments/000_M_TSO_M_DSO.py
    """

    # make_config_tuned(), NOT make_config_per_area(): the 2026-08-19 campaign
    # optimum is a GLOBAL weight set (its eval carries ``zone_g_w_class: None``).
    # make_config_per_area() overwrites every weight with its own analytic
    # per-area literals, which is a different operating point -- and not merely
    # a gauge rescaling of this one, so the trajectory changes.
    cfg = make_config_tuned()
    run_dir = new_run_dir("run_multi_system_ofo", cfg)
    log = run_multi_tso_dso(cfg)
    with (run_dir.root / "records.pkl").open("wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSimulation complete. {len(log)} steps recorded.")
    print(f"Results: {run_dir.root}")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        main_comparison()
    else:
        main()
