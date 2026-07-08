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
    python experiments/014_SBX_SINGLE_DEMO.py --scenario asym_z3 \
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
from sbx.corridor import build_corridor_registry  # noqa: E402
from sbx.fail import rep1  # noqa: E402

_013 = importlib.import_module("experiments.013_SBX_LADDER")

RESULT_DIR = REPO / "results" / "017_SBX_PLANNING"

PLAN_INTERVAL_S = 3600.0     # hourly planning intervals (DACF-style)


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SBX v3 planning pre-pass (hourly v_std schedule).")
    ap.add_argument("--mode", type=str, default="perfect",
                    choices=("perfect", "persistence", "noise"))
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="relative injection noise (mode 'noise' only)")
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--minutes", type=float, default=_013.DEFAULT_MINUTES)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.mode + (f"_s{args.sigma:g}" if args.mode == "noise" else "")
    out_stem = RESULT_DIR / f"schedule_{tag}_{args.minutes:.0f}min"

    # Scenario identity from the 013 family (profiles/start time/flags);
    # contingencies and bound overrides are IRRELEVANT to the plan.
    cfg = _013.make_config("asym_z3", "none", args.minutes)
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
        pp.runpp(net, max_iteration=50, run_control=False,
                 calculate_voltage_angles=True, init="auto",
                 distributed_slack=cfg.distributed_slack,
                 enforce_q_lims=cfg.enforce_q_lims_plant)
        for key, corr in registry.items():
            va = [float(net.res_bus.at[ln.bus_a, "vm_pu"])
                  for ln in corr.lines]
            vb = [float(net.res_bus.at[ln.bus_b, "vm_pu"])
                  for ln in corr.lines]
            tag_k = f"{key[0]}-{key[1]}"
            schedule[tag_k].append([t_from_s, va, vb])
            series[tag_k].append((t_from_s / 60.0, va, vb))
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
