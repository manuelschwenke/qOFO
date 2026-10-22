#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/017_SBX_PLANNING.py
===============================
SBX v3 planning pre-pass: the DACF/IDCF emulation that produces the
PLANNING-ANCHORED hourly contract-voltage schedule (STATUS_SBX.md,
v3 amendment 2026-07-08).

For every planning interval (hourly) of the scenario horizon the script
builds the PLANNING VIEW of the system — the same network, the same
load/DER profiles and the same zonal generation dispatch as the
closed-loop experiments, but WITHOUT contingencies (the plan does not
know them) and WITHOUT the closed-loop controllers (generators sit at
their scheduled voltage setpoints; taps and shunts at their build
defaults) — runs one power flow, and records the tie-line terminal
voltages per corridor.  The output JSON is consumed through
``MultiTSOConfig.sbx_v_std_schedule_path``: planning then REPLACES the
settled-state snapshot as the source of ``v_std``.

Forecast-quality modes:

* ``perfect``      — planning power flow on the true profiles: the
                     optimistic bound (plan = what the profiles will do).
* ``persistence``  — day-ahead persistence: the plan for hour t uses the
                     profile values of t − 24 h (a classic naive
                     forecast); the resulting deviation floor IS the
                     forecast error.
* ``noise``        — true profiles with multiplicative injection noise
                     (``--sigma``, seeded): a tunable forecast error.

Outputs (``results/017_SBX_PLANNING/``):
``schedule_<mode>[_<sigma>]_<minutes>min.json`` +
``schedule_<mode>[...].png`` (per-corridor terminal-voltage schedules).

Run:
    python experiments/017_SBX_PLANNING.py --mode perfect --minutes 360
    python experiments/014_SBX_SINGLE_DEMO.py --cell D2 \
        --schedule results/017_SBX_PLANNING/schedule_perfect_360min.json

Author: Manuel Schwenke / Claude Code
Date: 2026-07-08 (SBX v3)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.profiles import (  # noqa: E402
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from network.ieee39 import add_hv_networks, build_ieee39_net  # noqa: E402
from network.ieee39.zonal_balancing import (  # noqa: E402
    apply_gen_dispatch,
    compute_zonal_gen_dispatch,
)
from network.zone_partition import fixed_zone_partition_ieee39  # noqa: E402
from sbx_h.corridor import build_corridor_registry, \
    corridor_sensitivities  # noqa: E402
from sbx_h.fail import rep1  # noqa: E402

# v6 (2026-07-13): decoupled from the archived deal-era 013 script —
# the pre-pass only ever needed the shared 005 scenario identity.
_005 = importlib.import_module("experiments.005_CIGRE_MULTI")

RESULT_DIR = REPO / "results" / "017_SBX_PLANNING"

PLAN_INTERVAL_S = 3600.0     # hourly planning intervals (DACF-style)
DEFAULT_MINUTES = 360.0      # full case-study horizon
BAND_FLOOR_MVAR = 5.0        # tier-1 band floor (013-era rule, kept)


def build_planning_net(cfg):
    """The planning view of the experiment network.

    Same build path as the runner (IEEE 39 + HV sub-networks, identical
    flags), but no contingency loads, no controllers, no operating-point
    initialisation: generators at the scheduled voltage setpoint, taps
    and shunts at their build defaults.  Deliberately a SIMPLIFIED model
    — the planning/reality residual is what the tier-1 band prices.
    """
    net, meta = build_ieee39_net(
        ext_grid_vm_pu=1.03, scenario=cfg.scenario, verbose=False,
    )
    zone_map, _ = fixed_zone_partition_ieee39(net, verbose=False)

    cfg.validate_integrator_mode()
    shunt_mode = cfg.shunt_dispatch
    if shunt_mode == "off" and cfg.install_tso_tertiary_shunts:
        shunt_mode = "miqp"
    if shunt_mode == "integrator":
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=True,
            tso_shunt_kind="msc_msr",
            msc_n_levels=cfg.tso_shunt_msc_n_levels,
            msr_n_levels=cfg.tso_shunt_msr_n_levels,
            msc_q_step_mvar=cfg.tso_shunt_msc_q_step_mvar,
            msr_q_step_mvar=cfg.tso_shunt_msr_q_step_mvar,
            verbose=False,
        )
    else:
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
            tso_tertiary_shunt_q_mvar=cfg.tso_tertiary_shunt_q_mvar,
            verbose=False,
        )
    existing = set(net.bus.index)
    for z in zone_map:
        zone_map[z] = [b for b in zone_map[z] if b in existing]

    # Dispatch zone map: fixed partition + HV buses + machine LV buses
    # (mirrors the runner's _extend_zone_map_for_dispatch).
    dispatch_map = {z: list(b) for z, b in zone_map.items()}
    for hv in meta.hv_networks:
        z_hv = int(hv.zone)
        dispatch_map[z_hv] = sorted(
            set(dispatch_map[z_hv]) | set(hv.bus_indices))
    for tidx, gidx in zip(meta.machine_trafo_indices,
                          meta.machine_trafo_gen_map):
        if gidx < 0:
            continue
        lv_bus = int(net.trafo.at[tidx, "lv_bus"])
        hv_bus = int(net.trafo.at[tidx, "hv_bus"])
        for z, buses in dispatch_map.items():
            if hv_bus in set(buses):
                if lv_bus not in set(buses):
                    dispatch_map[z] = sorted(set(buses) | {lv_bus})
                break

    # Planning voltage schedule at the generators (crude, by design):
    # each machine at its zone's scheduled setpoint (uniform
    # cfg.v_setpoint_pu unless zone_v_setpoints_pu overrides a zone).
    zone_sets = {z: set(b) for z, b in dispatch_map.items()}

    def _zone_of_bus(bus: int):
        for z, buses in zone_sets.items():
            if bus in buses:
                return z
        return None

    for g in net.gen.index:
        z = _zone_of_bus(int(net.gen.at[g, "bus"]))
        v_set = float(cfg.v_setpoint_pu)
        if z is not None and cfg.zone_v_setpoints_pu:
            v_set = float(cfg.zone_v_setpoints_pu.get(z, v_set))
        net.gen.at[g, "vm_pu"] = v_set

    return net, meta, zone_map, dispatch_map


def plan_operating_point(net, meta, cfg) -> None:
    """Schedule the controllable states of the planning view (F10 fix,
    option (a)): real planning optimises taps and compensation too, so
    the pre-pass mirrors the runner's operating-point phases per hour —
    STATCOM Q via the temp-PV-gen trick + machine 2W OLTC at the zone
    voltage schedule (phase 1), then coupler 3W OLTC at the DSO target
    (phase 2).  Taps persist across hours (hourly re-scheduling from
    the previous plan, as planners do).
    """
    from pandapower.control import DiscreteTapControl

    v_mt = float(cfg.v_setpoint_pu)
    v_dso = float(cfg.oltc_init_v_target_pu)
    tol = float(cfg.dso_oltc_init_tol_pu)

    # Phase 1: TN-side STATCOMs as temporary PV gens + machine OLTCs.
    statcom_mask = (
        net.sgen["name"].astype(str).str.contains("STATCOM")
        & (net.sgen["subnet"].astype(str) != "DN")
    )
    tmp_map = {}
    for si in net.sgen.index[statcom_mask]:
        bus = int(net.sgen.at[si, "bus"])
        sn = float(net.sgen.at[si, "sn_mva"])
        net.sgen.at[si, "in_service"] = False
        gi = pp.create_gen(
            net, bus=bus, p_mw=float(net.sgen.at[si, "p_mw"]),
            vm_pu=v_mt, sn_mva=sn, max_q_mvar=sn, min_q_mvar=-sn,
            in_service=True, name="_PLAN_TEMP",
        )
        tmp_map[int(gi)] = int(si)
    for tidx in meta.machine_trafo_indices:
        DiscreteTapControl(net, element_index=tidx,
                           vm_lower_pu=v_mt - tol, vm_upper_pu=v_mt + tol,
                           side="hv", element="trafo")
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, distributed_slack=cfg.distributed_slack,
             enforce_q_lims=cfg.enforce_q_lims_plant)
    for gi, si in tmp_map.items():
        net.sgen.at[si, "q_mvar"] = float(net.res_gen.at[gi, "q_mvar"])
        net.sgen.at[si, "in_service"] = True
    if tmp_map:
        net.gen.drop(index=list(tmp_map.keys()), inplace=True)
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(index=list(net.controller.index),
                            inplace=True)

    # Phase 2: coupler 3W OLTCs at the DSO voltage target.
    for hv in meta.hv_networks:
        for t3w in hv.coupling_trafo_indices:
            DiscreteTapControl(net, element_index=t3w,
                               vm_lower_pu=v_dso - tol,
                               vm_upper_pu=v_dso + tol,
                               side="mv", element="trafo3w")
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=100, distributed_slack=cfg.distributed_slack,
             enforce_q_lims=cfg.enforce_q_lims_plant)
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(index=list(net.controller.index),
                            inplace=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SBX v3 planning pre-pass (hourly v_std schedule).")
    ap.add_argument("--mode", type=str, default="perfect",
                    choices=("perfect", "persistence", "noise"))
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="relative injection noise (mode 'noise' only)")
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    ap.add_argument("--no-oltc-schedule", action="store_true",
                    help="skip the per-hour tap/compensation scheduling "
                         "(reproduces the F10 crude-plan gap)")
    ap.add_argument("--band-ensemble", type=int, default=8,
                    help="forecast-error ensemble size per hour for the "
                         "PLANNING-DERIVED tier-1 band (0 = no band "
                         "schedule; the closed loop then keeps its "
                         "constant band)")
    ap.add_argument("--band-z", type=float, default=2.0,
                    help="band = z * ensemble sigma + tracking + gap")
    ap.add_argument("--eps-track-pu", type=float, default=1.0e-3,
                    help="declared terminal tracking tolerance [pu] — "
                         "converted to Mvar through the corridor "
                         "stiffness |s_corr| (contract data)")
    ap.add_argument("--m-gap-mvar", type=float, default=5.0,
                    help="model-gap allowance [Mvar] per corridor "
                         "(planning view vs closed loop; backtested)")
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.mode + (f"_s{args.sigma:g}" if args.mode == "noise" else "")
    if args.no_oltc_schedule:
        tag += "_crude"
    out_stem = RESULT_DIR / f"schedule_{tag}_{args.minutes:.0f}min"

    # Scenario identity = the shared 005 configuration (profiles/start
    # time/flags); contingencies and bound overrides are IRRELEVANT to
    # the plan, so the pre-pass needs nothing from the closed-loop
    # experiment scripts.
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * args.minutes
    if not cfg.use_profiles:
        rep1("the planning pre-pass needs profile-driven scenarios "
             "(use_profiles=False)")

    print(f"=== 017 planning pre-pass: mode={args.mode}, "
          f"{args.minutes:.0f} min horizon, hourly intervals ===")
    net, meta, zone_map, dispatch_map = build_planning_net(cfg)
    registry = build_corridor_registry(net, zone_map)

    profiles_csv = cfg.profiles_csv or DEFAULT_PROFILES_CSV
    profiles = load_profiles(profiles_csv,
                             timestep_s=int(PLAN_INTERVAL_S))
    snapshot_base_values(net)
    start = cfg.start_time
    lo = start - timedelta(hours=24)          # persistence needs t-24h
    hi = start + timedelta(seconds=cfg.n_total_s + PLAN_INTERVAL_S)
    profiles = profiles.loc[lo:hi]

    gen_dispatch = compute_zonal_gen_dispatch(
        net, profiles.loc[start:hi], dispatch_map,
        gen_p_min_mw={int(g): 0.0 for g in net.gen.index},
    )

    rng = np.random.default_rng(args.seed)
    n_hours = int(np.ceil(args.minutes * 60.0 / PLAN_INTERVAL_S)) + 1
    schedule: Dict[str, List] = {
        f"{k[0]}-{k[1]}": [] for k in registry
    }
    series: Dict[str, List] = {f"{k[0]}-{k[1]}": [] for k in registry}

    for h in range(n_hours):
        t_from_s = h * PLAN_INTERVAL_S
        t_real = start + timedelta(seconds=t_from_s)
        t_plan = (t_real - timedelta(hours=24)
                  if args.mode == "persistence" else t_real)
        apply_profiles(net, profiles, t_plan)
        apply_gen_dispatch(net, gen_dispatch,
                           t_real if args.mode != "persistence"
                           else t_plan)
        if args.mode == "noise":
            for tbl, cols in (("load", ("p_mw", "q_mvar")),
                              ("sgen", ("p_mw", "q_mvar"))):
                frame = getattr(net, tbl)
                for col in cols:
                    frame[col] = frame[col] * (
                        1.0 + args.sigma * rng.standard_normal(len(frame))
                    )
        if args.no_oltc_schedule:
            pp.runpp(net, max_iteration=50, run_control=False,
                     calculate_voltage_angles=True, init="auto",
                     distributed_slack=cfg.distributed_slack,
                     enforce_q_lims=cfg.enforce_q_lims_plant)
        else:
            # F10 fix (a): the plan schedules taps and compensation too.
            plan_operating_point(net, meta, cfg)

        # Point-plan record: terminal voltages, reference-end flows and
        # (for the band's tracking term) the corridor stiffness.
        point: Dict[str, dict] = {}
        for key, corr in registry.items():
            va = [float(net.res_bus.at[ln.bus_a, "vm_pu"])
                  for ln in corr.lines]
            vb = [float(net.res_bus.at[ln.bus_b, "vm_pu"])
                  for ln in corr.lines]
            q_pt, p_pt = [], []
            for ln in corr.lines:
                side = ("from" if int(net.line.at[ln.line_idx,
                                                  "from_bus"]) == ln.bus_a
                        else "to")
                q_pt.append(float(net.res_line.at[ln.line_idx,
                                                  f"q_{side}_mvar"]))
                p_pt.append(float(net.res_line.at[ln.line_idx,
                                                  f"p_{side}_mw"]))
            point[f"{key[0]}-{key[1]}"] = {
                "va": va, "vb": vb, "q": sum(q_pt), "p": p_pt,
            }

        # Forecast-error ensemble (band derivation): injections sampled
        # around the SAME hour's point forecast, taps HELD at the point
        # plan (the schedule is fixed day-ahead; the realisation varies
        # around it) — one plain power flow per member.
        band_sigma: Dict[str, float] = {}
        if args.band_ensemble > 0:
            dev: Dict[str, List[float]] = {
                f"{k[0]}-{k[1]}": [] for k in registry
            }
            for _m in range(args.band_ensemble):
                apply_profiles(net, profiles, t_plan)
                apply_gen_dispatch(net, gen_dispatch,
                                   t_real if args.mode != "persistence"
                                   else t_plan)
                for tbl, cols in (("load", ("p_mw", "q_mvar")),
                                  ("sgen", ("p_mw", "q_mvar"))):
                    frame = getattr(net, tbl)
                    for col in cols:
                        frame[col] = frame[col] * (
                            1.0 + args.sigma
                            * rng.standard_normal(len(frame))
                        )
                pp.runpp(net, max_iteration=50, run_control=False,
                         calculate_voltage_angles=True, init="auto",
                         distributed_slack=cfg.distributed_slack,
                         enforce_q_lims=cfg.enforce_q_lims_plant)
                for key, corr in registry.items():
                    q_m = 0.0
                    for ln in corr.lines:
                        side = ("from" if int(net.line.at[
                            ln.line_idx, "from_bus"]) == ln.bus_a
                            else "to")
                        q_m += float(net.res_line.at[ln.line_idx,
                                                     f"q_{side}_mvar"])
                    tag_k = f"{key[0]}-{key[1]}"
                    dev[tag_k].append(q_m - point[tag_k]["q"])
            for tag_k, dvals in dev.items():
                band_sigma[tag_k] = float(np.std(dvals))

        for key, corr in registry.items():
            tag_k = f"{key[0]}-{key[1]}"
            pt = point[tag_k]
            if args.band_ensemble > 0:
                # Tracking term: corridor stiffness (contract data) ×
                # the declared terminal tracking tolerance.
                _, s_corr_a, s_corr_b = corridor_sensitivities(
                    corr, pt["va"], pt["vb"], pt["p"],
                )
                s_corr = max(abs(s_corr_a), abs(s_corr_b))
                band = max(
                    BAND_FLOOR_MVAR,
                    float(np.ceil(
                        args.band_z * band_sigma[tag_k]
                        + s_corr * args.eps_track_pu
                        + args.m_gap_mvar
                    )),
                )
                schedule[tag_k].append([t_from_s, pt["va"], pt["vb"],
                                        band])
            else:
                schedule[tag_k].append([t_from_s, pt["va"], pt["vb"]])
            series[tag_k].append((t_from_s / 60.0, pt["va"], pt["vb"]))
        if args.band_ensemble > 0:
            bands_now = {k: schedule[k][-1][3] for k in schedule}
            print(f"  hour {h:2d} (t = {t_from_s / 60.0:5.0f} min, "
                  f"plan basis {t_plan:%d.%m. %H:%M}): converged; "
                  f"bands {bands_now}")
        else:
            print(f"  hour {h:2d} (t = {t_from_s / 60.0:5.0f} min, "
                  f"plan basis {t_plan:%d.%m. %H:%M}): converged")

    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(schedule, indent=1),
                         encoding="utf-8")
    print(f"schedule written: {json_path}")

    # ── Plot: per-corridor terminal-voltage schedules ───────────────────
    fig, axes = plt.subplots(len(registry), 1,
                             figsize=(9, 2.4 * len(registry)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (key, corr) in zip(axes, sorted(registry.items())):
        tag_k = f"{key[0]}-{key[1]}"
        t = [row[0] for row in series[tag_k]]
        for k, ln in enumerate(corr.lines):
            ax.step(t, [row[1][k] for row in series[tag_k]],
                    where="post", lw=1.2,
                    label=f"L{ln.line_idx} bus {ln.bus_a} (A)")
            ax.step(t, [row[2][k] for row in series[tag_k]],
                    where="post", lw=1.2, ls="--",
                    label=f"L{ln.line_idx} bus {ln.bus_b} (B)")
        ax.set_ylabel("v_std / pu", fontsize=8)
        ax.set_title(f"corridor ({key[0]},{key[1]}) — planned terminal "
                     f"voltages ({args.mode})", fontsize=9, loc="left")
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    axes[-1].set_xlabel("time / min")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=160)
    print(f"plot written:    {out_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
