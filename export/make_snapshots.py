"""
export/make_snapshots.py
========================
Build the IEEE 39-bus network at a chosen build phase and profile timestamp,
bring it to the experiment's initial operating point, and dump a dynamic
snapshot JSON (see :mod:`export.dynamic_snapshot`).

The construction sequence mirrors the plant-relevant steps of
``experiments/runners/multi_tso_dso.py`` (steps [1]-[4], [9], [10.1]-[10.2])
so that the snapshot equals the state every quasi-static experiment starts
from.  Controller installation (runner step [10.3], plant-side Q(V) loops)
is deliberately excluded: snapshots are controller-free states and the
OFO/PowerFactory loop commands DER Q explicitly.

Build phases
------------
``base``          -- IEEE 39 with machine trafos, load split, profiles.
                     No wind_replace, no 110 kV underlays.  (Gate A oracle.)
``wind_replace``  -- + generators replaced by STATCOM wind parks.
                     (Gate B oracle.)
``full``          -- + four 110 kV TUDA underlays with coupler 3W trafos,
                     DER, and TSO tertiary shunts.  (Gate C oracle and RMS
                     initial condition.)

Reference timestamps
--------------------
``t0``       -- the default experiment start (05.01.2016 08:00, winter
                morning; ``make_config`` in experiments/run_multi_system_ofo.py).
``peakres``  -- the 15-min profile timestamp maximising the system residual
                load  sum(P_load) - sum(P_sgen)  over the full year, i.e.
                the stress point for the synchronous fleet.

Usage
-----
    python -m export.make_snapshots --phase full --auto t0,peakres --verify
    python -m export.make_snapshots --phase base --at "2016-04-15 12:00"

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

# Allow "python export/make_snapshots.py" as well as "python -m export.make_snapshots".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import MultiTSOConfig
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from export.dynamic_snapshot import dump_dynamic_snapshot, verify_roundtrip
from network.ieee39 import add_hv_networks, build_ieee39_net, tag_der_q_modes
from network.ieee39.load_model import apply_zip_load_model
from network.ieee39.meta import IEEE39NetworkMeta
from network.ieee39.zonal_balancing import (
    apply_gen_dispatch,
    compute_zonal_gen_dispatch,
)
from network.zone_partition import fixed_zone_partition_ieee39

PHASES = ("base", "wind_replace", "full")

#: Installed-DER scenario used for the non-base phases when the caller does
#: not name one.  Tracks ``MultiTSOConfig.scenario`` so a reference snapshot
#: and the study it is meant to validate cannot silently disagree.
DEFAULT_DSO_DER_SCENARIO = MultiTSOConfig().scenario

#: Voltage setpoint used across the experiments (ext grid, AVRs, OLTC init).
V_SETPOINT_PU = 1.03

#: Default experiment start (make_config in experiments/run_multi_system_ofo.py).
DEFAULT_T0 = datetime(2016, 1, 5, 8, 0)

#: Canonical solver options: exactly the runner's step-[9] re-converge call
#: with the MultiTSOConfig defaults (distributed_slack=True,
#: enforce_q_lims_plant=True).  The stored solution is defined by this call.
SOLVER_OPTIONS: Dict[str, object] = {
    "run_control": False,
    "calculate_voltage_angles": True,
    "init": "auto",
    "max_iteration": 50,
    "distributed_slack": True,
    "enforce_q_lims": True,
}


@dataclass
class SnapshotState:
    """A fully built network frozen at one profile timestamp."""
    net: pp.pandapowerNet
    meta: IEEE39NetworkMeta
    zone_map: Dict[int, List[int]]
    solver_options: Dict[str, object]
    snapshot_time: datetime
    phase: str


# =====================================================================
#  Zone-map dispatch extension (mirrors the runner closure)
# =====================================================================

def _extend_zone_map_for_dispatch(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    zone_map: Dict[int, List[int]],
) -> None:
    """Extend ``zone_map`` in place with HV sub-network buses and machine
    LV terminal buses.

    Replicates ``_extend_zone_map_for_dispatch`` from
    ``experiments/runners/multi_tso_dso.py`` (a runner-local closure) so the
    zonal generator dispatch sees the same per-zone load/generation pools as
    the experiment.
    """
    for hv in meta.hv_networks:
        z = int(hv.zone)
        zone_map[z] = sorted(set(zone_map[z]) | set(hv.bus_indices))
    aux_lengths = {
        len(meta.internal_aux_bus_indices),
        len(meta.internal_aux_parent_buses),
        len(meta.internal_aux_line_indices),
    }
    if len(aux_lengths) != 1:
        raise ValueError("Internal auxiliary metadata lists have different lengths")
    for aux_bus, parent_bus in zip(
        meta.internal_aux_bus_indices, meta.internal_aux_parent_buses,
    ):
        owners = [z for z, buses in zone_map.items() if parent_bus in set(buses)]
        if len(owners) != 1:
            raise ValueError(
                f"Auxiliary bus {aux_bus} parent {parent_bus} belongs to "
                f"{len(owners)} zones; expected exactly one"
            )
        z = owners[0]
        zone_map[z] = sorted(set(zone_map[z]) | {int(aux_bus)})
    for tidx, gidx in zip(meta.machine_trafo_indices,
                          meta.machine_trafo_gen_map):
        if gidx < 0:
            continue
        lv_bus = int(net.trafo.at[tidx, "lv_bus"])
        hv_bus = int(net.trafo.at[tidx, "hv_bus"])
        for z, buses in zone_map.items():
            if hv_bus in set(buses):
                if lv_bus not in set(buses):
                    zone_map[z] = sorted(set(zone_map[z]) | {lv_bus})
                break


# =====================================================================
#  Operating-point initialisation (mirrors runner steps [10.1]-[10.2])
# =====================================================================

def _init_operating_point(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    *,
    verbose: int = 1,
) -> None:
    """STATCOM Q seed + machine 2W OLTC + coupler 3W OLTC initialisation.

    Phase 1: TN-side STATCOM wind parks are temporarily replaced by PV
    generators holding ``V_SETPOINT_PU`` while DiscreteTapControl moves the
    machine 2W OLTCs into the +/-0.01 pu band; the resulting Q is written
    back into the sgens.  Phase 2: coupler 3W OLTCs regulate their MV side
    to the same band.  All controllers are dropped afterwards -- the
    snapshot must be reproducible by a plain ``pp.runpp``.

    Mirrors steps [10.1]-[10.2] of ``run_multi_tso_dso`` with the config
    defaults (``oltc_init_v_target_pu=1.03``, ``dso_oltc_init_tol_pu=0.01``);
    step [10.3] (plant-side Q(V) loop install) is intentionally omitted.
    """
    from pandapower.control import DiscreteTapControl

    cfg = MultiTSOConfig()
    v_init_mt = V_SETPOINT_PU
    v_init_dso = float(cfg.oltc_init_v_target_pu)
    tol_pu = float(cfg.dso_oltc_init_tol_pu)

    # ── Phase 1: TN STATCOM Q via temp PV gens + machine 2W OLTC ─────────
    _statcom_mask = (
        net.sgen["name"].astype(str).str.contains("STATCOM")
        & (net.sgen["subnet"].astype(str) != "DN")
    ) if len(net.sgen) else pd.Series(dtype=bool)
    _statcom_idxs = net.sgen.index[_statcom_mask].tolist()

    _tmp_map: Dict[int, int] = {}
    for si in _statcom_idxs:
        bus = int(net.sgen.at[si, "bus"])
        p = float(net.sgen.at[si, "p_mw"])
        sn = float(net.sgen.at[si, "sn_mva"])
        net.sgen.at[si, "in_service"] = False
        gi = pp.create_gen(
            net, bus=bus, p_mw=p, vm_pu=V_SETPOINT_PU, sn_mva=sn,
            max_q_mvar=sn, min_q_mvar=-sn,
            in_service=True, name="_TEMP_INIT",
        )
        _tmp_map[int(gi)] = si

    for tidx in meta.machine_trafo_indices:
        DiscreteTapControl(
            net, element_index=tidx,
            vm_lower_pu=v_init_mt - tol_pu,
            vm_upper_pu=v_init_mt + tol_pu,
            side="hv", element="trafo",
        )

    if verbose:
        print(f"  [init 1] {len(_tmp_map)} STATCOM Q seeds + "
              f"{len(meta.machine_trafo_indices)} machine OLTC -> "
              f"{v_init_mt:.3f} +/-{tol_pu:.3f} pu")

    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, distributed_slack=True, enforce_q_lims=True)

    for gi, si in _tmp_map.items():
        net.sgen.at[si, "q_mvar"] = float(net.res_gen.at[gi, "q_mvar"])
        net.sgen.at[si, "in_service"] = True
    if _tmp_map:
        net.gen.drop(index=list(_tmp_map.keys()), inplace=True)
    net.controller.drop(index=net.controller.index, inplace=True)

    # ── Phase 2: coupler 3W OLTC (full phase only) ───────────────────────
    n_coup = sum(len(hv.coupling_trafo_indices) for hv in meta.hv_networks)
    if n_coup:
        for hv in meta.hv_networks:
            for t3w in hv.coupling_trafo_indices:
                DiscreteTapControl(
                    net, element_index=t3w,
                    vm_lower_pu=v_init_dso - tol_pu,
                    vm_upper_pu=v_init_dso + tol_pu,
                    side="mv", element="trafo3w",
                )
        if verbose:
            print(f"  [init 2] {n_coup} coupler 3W OLTC -> "
                  f"{v_init_dso:.3f} +/-{tol_pu:.3f} pu")
        pp.runpp(net, run_control=True, calculate_voltage_angles=True,
                 max_iteration=100, distributed_slack=True,
                 enforce_q_lims=True)
        net.controller.drop(index=net.controller.index, inplace=True)


# =====================================================================
#  Reference timestamps
# =====================================================================

def find_peak_residual_time(
    net: pp.pandapowerNet,
    profiles: pd.DataFrame,
) -> datetime:
    """Timestamp maximising system residual load  sum(P_load) - sum(P_sgen).

    Works on the base columns (profile-invariant) so it can be evaluated
    without running a power flow per timestep:

        P_load(t) = sum(const base_p) + sum_prof coeff_load[prof] * prof(t)
        P_sgen(t) = sum(const base_p) + sum_prof coeff_sgen[prof] * prof(t)

    Requires ``snapshot_base_values`` to have run (sgen base_p_mw).
    """
    if "base_p_mw" not in net.load.columns:
        raise ValueError("net.load lacks base_p_mw -- run snapshot_base_values")
    if len(net.sgen) and "base_p_mw" not in net.sgen.columns:
        raise ValueError("net.sgen lacks base_p_mw -- run snapshot_base_values")

    residual = pd.Series(0.0, index=profiles.index)

    prof_p = net.load["profile_p"] if "profile_p" in net.load.columns \
        else pd.Series(index=net.load.index, dtype=object)
    const_mask = prof_p.isna()
    residual += float(net.load.loc[const_mask, "base_p_mw"].sum())
    for prof_name in prof_p.dropna().unique():
        if prof_name not in profiles.columns:
            raise KeyError(f"load profile {prof_name!r} not in profiles CSV")
        coeff = float(net.load.loc[prof_p == prof_name, "base_p_mw"].sum())
        residual += coeff * profiles[prof_name]

    if len(net.sgen):
        sg_prof = net.sgen["profile"] if "profile" in net.sgen.columns \
            else pd.Series(index=net.sgen.index, dtype=object)
        residual -= float(net.sgen.loc[sg_prof.isna(), "base_p_mw"].sum())
        for prof_name in sg_prof.dropna().unique():
            if prof_name not in profiles.columns:
                raise KeyError(f"sgen profile {prof_name!r} not in profiles CSV")
            coeff = float(net.sgen.loc[sg_prof == prof_name, "base_p_mw"].sum())
            residual -= coeff * profiles[prof_name]

    return residual.idxmax().to_pydatetime()


# =====================================================================
#  State construction
# =====================================================================

def build_snapshot_state(
    phase: str,
    at: datetime,
    *,
    oltc_init: bool = True,
    shunt_kind: str = "msc_msr",
    msc_n_levels: int = 2,
    msr_n_levels: int = 2,
    msc_q_step_mvar: float = 25.0,
    msr_q_step_mvar: float = 25.0,
    bipolar_q_mvar: float = 50.0,
    load_model: Optional[str] = None,
    load_zip_anchor_vm_pu: Optional[float] = None,
    profiles_csv: Optional[str] = None,
    scenario: Optional[str] = None,
    verbose: int = 1,
) -> SnapshotState:
    """Build phase ``phase`` at profile timestamp ``at``.

    Shunt defaults mirror ``make_config()`` in
    ``experiments/run_multi_system_ofo.py`` (MSC/MSR banks, 2 levels of
    25 Mvar each); the load model defaults to the ``MultiTSOConfig``
    defaults (anchored ZIP since 2026-07-17).  ``at`` must lie exactly on
    the native 15-min profile grid -- reference snapshots must not depend
    on interpolation.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    _cfg_defaults = MultiTSOConfig()
    if load_model is None:
        load_model = _cfg_defaults.load_model
    if load_zip_anchor_vm_pu is None:
        load_zip_anchor_vm_pu = float(_cfg_defaults.load_zip_anchor_vm_pu)
    if load_model not in ("zip", "const_pq"):
        raise ValueError(
            f"load_model must be 'zip' or 'const_pq', got {load_model!r}"
        )

    # The non-base phases used to hardcode the (now deprecated) alias
    # ``wind_replace``, which resolves to ``base_410``.  Since the installed
    # DER capacity became a scenario choice, a reference snapshot built for a
    # ``rural_700`` study must say so -- otherwise the PowerFactory checkpoint
    # is parity-validated at 410 MW while the RMS runs at 700 MW.
    if scenario is None:
        scenario = "base" if phase == "base" else DEFAULT_DSO_DER_SCENARIO
    if verbose:
        print(f"[snapshot] phase={phase} scenario={scenario} at={at:%d.%m.%Y %H:%M}")

    # ── Runner steps [1]-[2]: build + zone partition ─────────────────────
    net, meta = build_ieee39_net(
        ext_grid_vm_pu=V_SETPOINT_PU, scenario=scenario,
        verbose=(verbose >= 2),
    )
    zone_map, _bus_zone = fixed_zone_partition_ieee39(net, verbose=(verbose >= 2))

    # ── Runner step [3]: HV underlays (full phase only) ──────────────────
    if phase == "full":
        if shunt_kind == "msc_msr":
            meta = add_hv_networks(
                net, meta,
                install_tso_tertiary_shunts=True,
                tso_shunt_kind="msc_msr",
                msc_n_levels=msc_n_levels,
                msr_n_levels=msr_n_levels,
                msc_q_step_mvar=msc_q_step_mvar,
                msr_q_step_mvar=msr_q_step_mvar,
                verbose=(verbose >= 2),
            )
        elif shunt_kind == "bipolar":
            meta = add_hv_networks(
                net, meta,
                install_tso_tertiary_shunts=True,
                tso_shunt_kind="bipolar",
                tso_tertiary_shunt_q_mvar=bipolar_q_mvar,
                verbose=(verbose >= 2),
            )
        elif shunt_kind == "none":
            meta = add_hv_networks(
                net, meta,
                install_tso_tertiary_shunts=False,
                verbose=(verbose >= 2),
            )
        else:
            raise ValueError(
                f"shunt_kind must be 'msc_msr', 'bipolar' or 'none', "
                f"got {shunt_kind!r}"
            )
        # add_hv_networks may drop buses; purge them from the zone map.
        existing = set(net.bus.index)
        for z in zone_map:
            zone_map[z] = [b for b in zone_map[z] if b in existing]

    # ── Runner step [3b]: plant load model (mirrors multi_tso_dso) ───────
    if load_model == "zip":
        apply_zip_load_model(
            net, anchor_vm_pu=load_zip_anchor_vm_pu, verbose=(verbose >= 1),
        )

    # ── Runner step [4]: droop tagging with the config defaults ──────────
    cfg = _cfg_defaults
    meta = tag_der_q_modes(
        net, meta,
        tso_q_mode=cfg.tso_q_mode, dso_q_mode=cfg.dso_q_mode,
        tso_qv_slope_pu=cfg.tso_qv_slope_pu,
        dso_qv_slope_pu=cfg.dso_qv_slope_pu,
        tso_qv_vref_pu=cfg.tso_qv_vref_pu,
        dso_qv_vref_pu=cfg.dso_qv_vref_pu,
        tso_qv_deadband_pu=cfg.tso_qv_deadband_pu,
        dso_qv_deadband_pu=cfg.dso_qv_deadband_pu,
        tso_cosphi=cfg.tso_cosphi, dso_cosphi=cfg.dso_cosphi,
        tso_cosphi_sign=cfg.tso_cosphi_sign,
        dso_cosphi_sign=cfg.dso_cosphi_sign,
        verbose=(verbose >= 2),
    )

    # ── Runner step [9]: profiles at the snapshot instant ────────────────
    csv_path = profiles_csv or DEFAULT_PROFILES_CSV
    profiles = load_profiles(csv_path, timestep_min=15)
    snapshot_base_values(net)

    ts = pd.Timestamp(at)
    if ts not in profiles.index:
        raise ValueError(
            f"snapshot time {at} is not on the native 15-min profile grid "
            f"({profiles.index[0]} .. {profiles.index[-1]}); reference "
            f"snapshots must not depend on interpolation"
        )
    apply_profiles(net, profiles, at)

    _extend_zone_map_for_dispatch(net, meta, zone_map)
    gen_p_min = {int(g): 0.0 for g in net.gen.index}
    dispatch = compute_zonal_gen_dispatch(
        net, profiles.loc[ts:ts], zone_map, gen_p_min_mw=gen_p_min,
    )
    apply_gen_dispatch(net, dispatch, at)

    pp.runpp(net, **SOLVER_OPTIONS)

    # ── Runner steps [10.1]-[10.2]: operating-point init ─────────────────
    if oltc_init:
        _init_operating_point(net, meta, verbose=verbose)

    # Final controller-free power flow: this call defines the stored
    # solution and must use exactly SOLVER_OPTIONS.
    pp.runpp(net, **SOLVER_OPTIONS)

    if verbose:
        vm = net.res_bus["vm_pu"]
        print(f"  [snapshot] converged: {len(net.bus)} buses, "
              f"vm in [{vm.min():.4f}, {vm.max():.4f}] pu, "
              f"slack P = {float(net.res_gen.loc[net.gen['slack'], 'p_mw'].sum()):.1f} MW")

    return SnapshotState(
        net=net, meta=meta, zone_map=zone_map,
        solver_options=dict(SOLVER_OPTIONS),
        snapshot_time=at, phase=phase,
    )


# =====================================================================
#  CLI
# =====================================================================

def _parse_at(text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse timestamp {text!r}; use 'YYYY-MM-DD HH:MM'"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dump dynamic IEEE 39 snapshots for the PowerFactory "
                    "RMS pipeline (build plan Phase 0).",
    )
    parser.add_argument("--phase", choices=PHASES, default="full")
    parser.add_argument(
        "--at", action="append", type=_parse_at, default=None, metavar="TIME",
        help="Explicit profile timestamp (repeatable), e.g. '2016-04-15 12:00'.",
    )
    parser.add_argument(
        "--auto", default="t0,peakres",
        help="Comma-separated automatic reference timestamps out of "
             "{t0, peakres}; ignored when empty. Default: t0,peakres.",
    )
    parser.add_argument("--out-dir", default="export/snapshots")
    parser.add_argument("--no-oltc-init", action="store_true",
                        help="Skip the phase-1/2 OLTC + STATCOM Q init "
                             "(taps stay at 0).")
    parser.add_argument("--shunt-kind", choices=("msc_msr", "bipolar", "none"),
                        default="msc_msr")
    parser.add_argument("--load-model", choices=("zip", "const_pq"),
                        default=None,
                        help="Plant load model; default = MultiTSOConfig "
                             "default (anchored ZIP since 2026-07-17).")
    parser.add_argument("--scenario", default=None,
                        help="Installed-DER scenario ('base_410' or "
                             "'rural_700'); default = MultiTSOConfig.scenario "
                             f"({DEFAULT_DSO_DER_SCENARIO!r}). The 'base' "
                             "phase always builds the bare IEEE 39 case.")
    parser.add_argument("--verify", action="store_true",
                        help="Round-trip verify each dumped snapshot.")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args(argv)

    # ── Resolve requested timestamps ─────────────────────────────────────
    requested: List[Tuple[str, datetime]] = []
    auto_tags = [t.strip() for t in args.auto.split(",") if t.strip()] \
        if args.auto else []
    for tag in auto_tags:
        if tag == "t0":
            requested.append(("t0", DEFAULT_T0))
        elif tag == "peakres":
            requested.append(("peakres", None))  # resolved after a probe build
        else:
            raise SystemExit(f"Unknown --auto tag {tag!r}; use t0, peakres")
    for at in (args.at or []):
        requested.append((f"at{at:%Y%m%d-%H%M}", at))
    if not requested:
        raise SystemExit("No timestamps requested (empty --auto and no --at).")

    # Resolve 'peakres' with one probe build (base values only, no dump).
    if any(t is None for _, t in requested):
        probe = build_snapshot_state(
            args.phase, DEFAULT_T0, oltc_init=False,
            shunt_kind=args.shunt_kind, load_model=args.load_model,
            scenario=args.scenario, verbose=0,
        )
        profiles = load_profiles(DEFAULT_PROFILES_CSV, timestep_min=15)
        t_peak = find_peak_residual_time(probe.net, profiles)
        requested = [(tag, t_peak if t is None else t) for tag, t in requested]
        if args.verbose:
            print(f"[snapshot] peak residual load at {t_peak:%d.%m.%Y %H:%M}")

    # ── Build + dump each snapshot ───────────────────────────────────────
    failures = 0
    for tag, at in requested:
        state = build_snapshot_state(
            args.phase, at,
            oltc_init=not args.no_oltc_init,
            shunt_kind=args.shunt_kind,
            load_model=args.load_model,
            scenario=args.scenario,
            verbose=args.verbose,
        )
        label = f"{state.phase}_{tag}_{at:%Y%m%d-%H%M}"
        path = dump_dynamic_snapshot(
            state.net, state.meta, state.zone_map, label, args.out_dir,
            solver_options=state.solver_options,
            snapshot_time=state.snapshot_time,
            phase=state.phase,
        )
        print(f"[snapshot] wrote {path}")
        if args.verify:
            report = verify_roundtrip(path)
            print("  " + report.summary().replace("\n", "\n  "))
            if not report.ok:
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
