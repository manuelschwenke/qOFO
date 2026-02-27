"""
AC-OPF MINLP Reference Runner
==============================

Runs the full nonlinear AC-OPF at each timestep as a "perfect-knowledge"
benchmark.  Uses the same network, profiles, contingencies, and objective
weights as the cascaded TSO-DSO controller so that results are directly
comparable.

Usage
-----
    python run_reference.py

Author: Claude (generated)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

os.environ["QT_API"] = "pyqt5"
import matplotlib as mpl
mpl.use('Qt5Agg')

import numpy as np
from numpy.typing import NDArray
import pandapower as pp
import idaes

# ── Project imports ───────────────────────────────────────────────────────────
from network.build_tuda_net import build_tuda_net, NetworkMetadata
from core.profiles import (
    load_profiles, apply_profiles, snapshot_base_values, DEFAULT_PROFILES_CSV,
)
from run_cascade import ContingencyEvent, _apply_contingency
from run_cascade import run_cascade
from optimisation.ac_opf_reference import (
    extract_network_data, build_ac_opf_model, solve_ac_opf,
    apply_opf_result_to_net, ACOPFResult, NetworkData,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReferenceRecord:
    """One timestep of AC-OPF reference results."""
    minute: int

    # Optimal decision variables
    q_der_mvar: NDArray[np.float64]         # all DER Q setpoints [Mvar]
    v_gen_pu: NDArray[np.float64]           # generator AVR setpoints [p.u.]
    q_gen_mvar: NDArray[np.float64]         # generator Q injection [Mvar]
    oltc_taps: Dict[str, int]               # oltc_key -> tap position
    shunt_steps: NDArray[np.int64]          # shunt step values

    # Solver info
    objective_value: float
    solver_status: str
    termination_condition: str
    solve_time_s: float

    # Plant voltages (from verification PF after applying optimal setpoints)
    plant_tn_voltages_pu: NDArray[np.float64]
    plant_dn_voltages_pu: NDArray[np.float64]

    # Voltage penalty (matching cascade's tso_v_penalty)
    v_penalty_tn: float     # g_v_tn * sum((V_tn - V_set)^2)
    v_penalty_dn: float     # g_v_dn * sum((V_dn - V_set)^2)

    # Contingency events that fired this minute
    contingency_events: Optional[List[str]] = None


@dataclass
class ReferenceResult:
    """Container for all reference simulation results."""
    log: List[ReferenceRecord]
    tn_bus_indices_pp: List[int]
    dn_bus_indices_pp: List[int]
    der_sgen_indices: List[int]
    ders_pp_buses: List[int]
    v_setpoint_pu: float
    g_v_tn: float
    g_v_dn: float
    g_u: float


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_reference(
    v_setpoint_pu: float = 1.05,
    n_minutes: int = 120,
    timestep_min: int = 1,
    start_time: datetime = datetime(2016, 6, 10, 0, 0),
    profiles_csv: str = DEFAULT_PROFILES_CSV,
    g_v_tn: float = 100000.0,
    g_v_dn: float = 10000.0,
    g_u: float = 0.1,
    v_min_pu: float = 0.90,
    v_max_pu: float = 1.10,
    solver: str = 'mindtpy',
    mip_solver: str = 'gurobi',
    nlp_solver: str = 'ipopt',
    verbose: int = 1,
    use_profiles: bool = True,
    contingencies: Optional[List[ContingencyEvent]] = None,
) -> ReferenceResult:
    """
    Run the AC-OPF MINLP reference at each timestep.

    Parameters
    ----------
    v_setpoint_pu : float
        Target voltage setpoint for all monitored buses.
    n_minutes : int
        Number of simulation minutes.
    timestep_min : int
        Resolution in minutes (each step solves one AC-OPF).
    start_time : datetime
        Simulation start time (for profile lookup).
    profiles_csv : str
        Path to profiles CSV file.
    g_v_tn, g_v_dn : float
        Voltage deviation penalty for TN (380 kV) / DN (110 kV) buses.
    g_u : float
        DER Q regularisation weight.
    solver : str
        Primary solver ('mindtpy', 'bonmin', 'scip', etc.).
    mip_solver, nlp_solver : str
        Sub-solvers for MindtPy OA.
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed).
    use_profiles : bool
        Whether to apply time-varying profiles.
    contingencies : list of ContingencyEvent, optional
        Contingency events to inject at specified timesteps.
    """
    print("=" * 72)
    print("  AC-OPF MINLP REFERENCE")
    print(f"  V_set={v_setpoint_pu:.3f} p.u.  |  {n_minutes} min  |  "
          f"g_v_tn={g_v_tn:.0f}  g_v_dn={g_v_dn:.0f}  g_u={g_u}")
    print(f"  Solver: {solver}  (MIP: {mip_solver}, NLP: {nlp_solver})")
    print("=" * 72)

    # ── 1) Build network ──────────────────────────────────────────────────
    net, meta = build_tuda_net(ext_grid_vm_pu=v_setpoint_pu, pv_nodes=True)

    # Remove pandapower controllers (OLTCs etc.) — we control everything
    if hasattr(net, 'controller') and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Initial power flow
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # ── 2) Identify monitored buses (same as run_cascade) ────────────────
    dn_buses = {int(b) for b in net.bus.index
                if str(net.bus.at[b, "subnet"]) == "DN"}

    tn_v_buses = sorted(int(b) for b in net.bus.index
                        if float(net.bus.at[b, "vn_kv"]) >= 300.0)
    dn_v_buses = sorted(int(b) for b in net.bus.index
                        if 100.0 <= float(net.bus.at[b, "vn_kv"]) < 200.0
                        and int(b) in dn_buses)

    # DER sgen indices (for output)
    der_sgens = [int(s) for s in net.sgen.index
                 if not str(net.sgen.at[s, "name"]).startswith("BOUND_")]

    # ── 3) Load profiles ─────────────────────────────────────────────────
    if use_profiles:
        profiles = load_profiles(profiles_csv, timestep_min=timestep_min)
        snapshot_base_values(net)
    else:
        profiles = None
        snapshot_base_values(net)

    # ── 4) Time loop ─────────────────────────────────────────────────────
    log: List[ReferenceRecord] = []
    total_solve_time = 0.0

    for minute in range(timestep_min, n_minutes + 1, timestep_min):
        t_now = start_time + timedelta(minutes=minute)

        # Apply profiles
        if use_profiles and profiles is not None:
            apply_profiles(net, profiles, t_now)

        # Apply contingencies
        ev_descs: Optional[List[str]] = None
        if contingencies:
            fired = [ev for ev in contingencies if ev.minute == minute]
            if fired:
                ev_descs = []
                for ev in fired:
                    ev_descs.append(_apply_contingency(net, ev, verbose))

        # Run power flow (provides warm-start for solver + updates _ppc)
        try:
            pp.runpp(net, run_control=True, calculate_voltage_angles=True)
        except pp.powerflow.LoadflowNotConverged:
            print(f"  [min {minute}] WARNING: power flow did not converge, skipping")
            continue

        # Extract network data
        nd = extract_network_data(net, meta, tn_v_buses, dn_v_buses)

        # Build Pyomo model
        model = build_ac_opf_model(
            nd, v_setpoint_pu=v_setpoint_pu,
            g_v_tn=g_v_tn, g_v_dn=g_v_dn, g_u=g_u,
            v_min_pu=v_min_pu, v_max_pu=v_max_pu,
        )

        # Solve
        result = solve_ac_opf(
            model, nd,
            solver_name=solver,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            verbose=(verbose >= 2),
        )
        total_solve_time += result.solve_time_s

        # Apply solution to network and run verification PF
        apply_opf_result_to_net(net, result, nd)
        try:
            pp.runpp(net, run_control=False, calculate_voltage_angles=True)
        except pp.powerflow.LoadflowNotConverged:
            print(f"  [min {minute}] WARNING: verification PF did not converge")

        # Extract plant voltages from verification PF
        plant_tn_v = np.array([float(net.res_bus.at[b, "vm_pu"])
                               for b in tn_v_buses])
        plant_dn_v = np.array([float(net.res_bus.at[b, "vm_pu"])
                               for b in dn_v_buses])

        # Compute voltage penalties
        v_pen_tn = g_v_tn * float(np.sum((plant_tn_v - v_setpoint_pu) ** 2))
        v_pen_dn = g_v_dn * float(np.sum((plant_dn_v - v_setpoint_pu) ** 2))

        # Record
        rec = ReferenceRecord(
            minute=minute,
            q_der_mvar=np.array(result.q_der_mvar),
            v_gen_pu=np.array(result.v_gen_pu),
            q_gen_mvar=np.array(result.q_gen_mvar),
            oltc_taps=result.oltc_taps,
            shunt_steps=np.array(result.shunt_steps, dtype=np.int64),
            objective_value=result.objective_value,
            solver_status=result.solver_status,
            termination_condition=result.termination_condition,
            solve_time_s=result.solve_time_s,
            plant_tn_voltages_pu=plant_tn_v,
            plant_dn_voltages_pu=plant_dn_v,
            v_penalty_tn=v_pen_tn,
            v_penalty_dn=v_pen_dn,
            contingency_events=ev_descs,
        )
        log.append(rec)

        # Progress output
        if verbose >= 1:
            tn_err = np.max(np.abs(plant_tn_v - v_setpoint_pu)) if len(plant_tn_v) > 0 else 0
            dn_err = np.max(np.abs(plant_dn_v - v_setpoint_pu)) if len(plant_dn_v) > 0 else 0
            taps_str = " ".join(f"{k}:{v}" for k, v in sorted(result.oltc_taps.items()))
            shunts_str = " ".join(str(s) for s in result.shunt_steps)
            print(f"  min {minute:4d}  "
                  f"TN_err={tn_err:.5f}  DN_err={dn_err:.5f}  "
                  f"obj={result.objective_value:.4f}  "
                  f"taps=[{taps_str}]  shunts=[{shunts_str}]  "
                  f"t={result.solve_time_s:.1f}s  {result.termination_condition}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  REFERENCE SUMMARY  --  V_set = {v_setpoint_pu:.3f} p.u.")
    print("=" * 72)
    if log:
        final = log[-1]
        v_tn = final.plant_tn_voltages_pu
        v_dn = final.plant_dn_voltages_pu
        print(f"  Final TN V: min={np.min(v_tn):.4f}  "
              f"mean={np.mean(v_tn):.4f}  max={np.max(v_tn):.4f}")
        print(f"  Final DN V: min={np.min(v_dn):.4f}  "
              f"mean={np.mean(v_dn):.4f}  max={np.max(v_dn):.4f}")
        print(f"  Max TN |V-V_set| = {np.max(np.abs(v_tn - v_setpoint_pu)):.5f} p.u.")
        print(f"  Max DN |V-V_set| = {np.max(np.abs(v_dn - v_setpoint_pu)):.5f} p.u.")
        obj_values = [r.objective_value for r in log if r.objective_value < float('inf')]
        if obj_values:
            print(f"  Objective: min={min(obj_values):.4f}  "
                  f"mean={np.mean(obj_values):.4f}  max={max(obj_values):.4f}")
        print(f"  Total solve time: {total_solve_time:.1f}s  "
              f"(avg {total_solve_time / len(log):.2f}s/step)")
    print("=" * 72)

    ders_buses = [
        int(net.sgen.at[s, 'bus']) for s in net.sgen.index
        if not str(net.sgen.at[s, 'name']).startswith('BOUND_')
    ]

    return ReferenceResult(
        log=log,
        tn_bus_indices_pp=tn_v_buses,
        dn_bus_indices_pp=dn_v_buses,
        der_sgen_indices=der_sgens,
        ders_pp_buses=ders_buses,
        v_setpoint_pu=v_setpoint_pu,
        g_v_tn=g_v_tn, g_v_dn=g_v_dn, g_u=g_u,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Comparison Utilities
# ═══════════════════════════════════════════════════════════════════════════════

# =============================================================================
# Weight-Free Physical Metric Helpers
# =============================================================================

def _rmsd(v: np.ndarray, v_set: float) -> float:
    """
    Root-mean-square voltage deviation in p.u. — completely weight-free.

    Parameters
    ----------
    v : np.ndarray
        Voltage magnitudes [p.u.].
    v_set : float
        Voltage setpoint [p.u.].

    Raises
    ------
    ValueError
        If the voltage array is empty (indicates a misconfigured bus set).
    """
    if v.size == 0:
        raise ValueError("_rmsd: voltage array is empty.")
    return float(np.sqrt(np.mean((v - v_set) ** 2)))


def _max_dev(v: np.ndarray, v_set: float) -> float:
    """
    Maximum absolute voltage deviation from setpoint in p.u.

    Parameters
    ----------
    v : np.ndarray
        Voltage magnitudes [p.u.].
    v_set : float
        Voltage setpoint [p.u.].

    Raises
    ------
    ValueError
        If the voltage array is empty.
    """
    if v.size == 0:
        raise ValueError("_max_dev: voltage array is empty.")
    return float(np.max(np.abs(v - v_set)))


# =============================================================================
# Common Objective Evaluation (apples-to-apples suboptimality gap)
# =============================================================================

def evaluate_common_objective(
    v_tn: np.ndarray,
    v_dn: np.ndarray,
    q_der_mvar: np.ndarray,
    v_setpoint_pu: float,
    g_v_tn: float,
    g_v_dn: float,
    g_u: float,
    s_base_mva: float = 1.0,
) -> float:
    """
    Evaluate the MINLP objective function at an arbitrary plant state.

    This is the *same* mathematical function that the AC-OPF MINLP minimises::

        J = g_v_tn * Σ_{i∈TN}(V_i − V_set)²
          + g_v_dn * Σ_{j∈DN}(V_j − V_set)²
          + g_u    * Σ_{k∈DER}(Q_k / s_base)²

    By evaluating this at **both** the MINLP-optimal and the cascade plant
    states, we obtain a true apples-to-apples suboptimality gap that is
    independent of the cascade's own (structurally different) objective.

    Parameters
    ----------
    v_tn : np.ndarray
        Voltage magnitudes at TN (380 kV) monitored buses [p.u.].
    v_dn : np.ndarray
        Voltage magnitudes at DN (110 kV) monitored buses [p.u.].
    q_der_mvar : np.ndarray
        DER reactive power setpoints [Mvar].
    v_setpoint_pu : float
        Voltage setpoint [p.u.].
    g_v_tn, g_v_dn : float
        Voltage deviation penalty weights for TN / DN buses.
    g_u : float
        DER Q regularisation weight.
    s_base_mva : float
        System base apparent power [MVA].  Default 1.0 (pandapower convention).

    Returns
    -------
    float
        Objective value J.
    """
    j_v_tn = g_v_tn * float(np.sum((v_tn - v_setpoint_pu) ** 2))
    j_v_dn = g_v_dn * float(np.sum((v_dn - v_setpoint_pu) ** 2))
    j_u = g_u * float(np.sum((q_der_mvar / s_base_mva) ** 2))
    return j_v_tn + j_v_dn + j_u


# =============================================================================
# Comparison Function
# =============================================================================

def compare_results(
    ref: ReferenceResult,
    cas,  # CascadeResult
    v_setpoint_pu: float,
) -> None:
    """
    Print a side-by-side comparison of the cascaded OFO against the MINLP
    reference using exclusively weight-free physical metrics.

    All quantities are observable physical outcomes (p.u., Mvar, tap counts)
    that exist independently of objective function weightings.  The weighted
    penalty terms stored in ReferenceRecord.v_penalty_tn/dn and
    IterationRecord.tso_v_penalty are intentionally NOT used here because
    they embed incompatible g_v values from the two formulations.

    Parameters
    ----------
    ref : ReferenceResult
        From ``run_reference()``.
    cas : CascadeResult
        From ``run_cascade()``.
    v_setpoint_pu : float
        Voltage setpoint [p.u.].

    Raises
    ------
    RuntimeError
        If no common timesteps exist between reference and cascade logs.
    """
    from run_cascade import IterationRecord

    print("\n" + "=" * 80)
    print("  CASCADE vs. REFERENCE COMPARISON  —  weight-free physical metrics")
    print("=" * 80)

    cascade_log = cas.log

    # -- Build minute-indexed lookups ------------------------------------------
    ref_by_min = {r.minute: r for r in ref.log}
    cas_by_min = {
        r.minute: r for r in cascade_log
        if r.plant_tn_voltages_pu is not None
    }
    common_mins = sorted(set(ref_by_min) & set(cas_by_min))

    if not common_mins:
        raise RuntimeError(
            "compare_results: no common timesteps found between reference and "
            "cascade logs.  Ensure both simulations cover the same time window."
        )

    # ==========================================================================
    # 1.  Per-timestep physical metric arrays
    # ==========================================================================

    ref_rmsd_tn:  list[float] = []
    ref_rmsd_dn:  list[float] = []
    ref_max_tn:   list[float] = []
    ref_max_dn:   list[float] = []
    ref_der_q_l1: list[float] = []   # Σ|Q_DER|  [Mvar]

    cas_rmsd_tn:  list[float] = []
    cas_rmsd_dn:  list[float] = []
    cas_max_tn:   list[float] = []
    cas_max_dn:   list[float] = []
    cas_der_q_l1: list[float] = []

    # OLTC switching: Σ_i |Δs_i| per timestep
    ref_oltc_switches: list[int] = []
    cas_oltc_switches: list[int] = []

    # State carry-forward variables for the cascade (TSO fires every 3 min)
    prev_ref_taps:     dict | None      = None
    prev_cas_tso_taps: np.ndarray | None = None
    prev_cas_dso_taps: np.ndarray | None = None
    last_cas_tso_q_der: np.ndarray | None = None
    last_cas_dso_q_der: np.ndarray | None = None

    for m in common_mins:
        rr = ref_by_min[m]
        cr = cas_by_min[m]

        # -- Voltage metrics ---------------------------------------------------
        ref_rmsd_tn.append(_rmsd(rr.plant_tn_voltages_pu, v_setpoint_pu))
        ref_rmsd_dn.append(_rmsd(rr.plant_dn_voltages_pu, v_setpoint_pu))
        ref_max_tn.append(_max_dev(rr.plant_tn_voltages_pu, v_setpoint_pu))
        ref_max_dn.append(_max_dev(rr.plant_dn_voltages_pu, v_setpoint_pu))

        cas_rmsd_tn.append(_rmsd(cr.plant_tn_voltages_pu, v_setpoint_pu))
        cas_rmsd_dn.append(_rmsd(cr.plant_dn_voltages_pu, v_setpoint_pu))
        cas_max_tn.append(_max_dev(cr.plant_tn_voltages_pu, v_setpoint_pu))
        cas_max_dn.append(_max_dev(cr.plant_dn_voltages_pu, v_setpoint_pu))

        # -- DER Q total absolute dispatch  Σ|Q_k|  [Mvar] --------------------
        # Reference: q_der_mvar contains ALL DER Q (TN + DN) for this step.
        ref_der_q_l1.append(float(np.sum(np.abs(rr.q_der_mvar))))

        # Cascade: carry forward last known vectors across TSO/DSO periods.
        if cr.tso_q_der_mvar is not None:
            last_cas_tso_q_der = cr.tso_q_der_mvar
        if cr.dso_q_der_mvar is not None:
            last_cas_dso_q_der = cr.dso_q_der_mvar

        cas_q_parts = []
        if last_cas_tso_q_der is not None:
            cas_q_parts.append(last_cas_tso_q_der)
        if last_cas_dso_q_der is not None:
            cas_q_parts.append(last_cas_dso_q_der)

        if not cas_q_parts:
            raise RuntimeError(
                f"compare_results: cascade has no DER Q state at minute {m}."
            )
        cas_der_q_l1.append(float(np.sum(np.abs(np.concatenate(cas_q_parts)))))

        # -- OLTC switching  Σ_i |Δs_i(t)|  [tap steps] ----------------------

        # Reference: oltc_taps is Dict[str, int] (trafo name → tap position).
        curr_ref_taps = rr.oltc_taps
        if prev_ref_taps is None:
            ref_oltc_switches.append(0)
        else:
            sw = sum(
                abs(curr_ref_taps.get(k, v) - v)
                for k, v in prev_ref_taps.items()
            )
            ref_oltc_switches.append(int(sw))
        prev_ref_taps = curr_ref_taps

        # Cascade: diff against previous known position when controller fires.
        cas_sw = 0
        if cr.tso_active and cr.tso_oltc_taps is not None:
            if prev_cas_tso_taps is not None:
                cas_sw += int(np.sum(np.abs(cr.tso_oltc_taps - prev_cas_tso_taps)))
            prev_cas_tso_taps = cr.tso_oltc_taps

        if cr.dso_active and cr.dso_oltc_taps is not None:
            if prev_cas_dso_taps is not None:
                cas_sw += int(np.sum(np.abs(cr.dso_oltc_taps - prev_cas_dso_taps)))
            prev_cas_dso_taps = cr.dso_oltc_taps

        cas_oltc_switches.append(cas_sw)

    # ==========================================================================
    # 2.  Control vector L2 norm  ||u||_2  (physical actuators, no Q_PCC_set)
    # ==========================================================================
    ref_norms: list[float] = []
    cas_norms: list[float] = []
    last_tso_physical: np.ndarray | None = None
    last_dso_physical: np.ndarray | None = None

    for m in common_mins:
        rr = ref_by_min[m]
        cr = cas_by_min[m]

        # Reference: concatenate all physical actuators in consistent order.
        u_ref = np.concatenate([
            rr.v_gen_pu,                                        # generator AVR [p.u.]
            np.array(list(rr.oltc_taps.values()), dtype=float), # 2W + 3W OLTCs [taps]
            rr.q_der_mvar,                                      # all DER Q [Mvar]
            rr.shunt_steps.astype(float),                       # shunts [steps]
        ])
        ref_norms.append(float(np.linalg.norm(u_ref)))

        # Cascade: carry forward last known physical state.
        # Q_PCC_set is deliberately excluded — it is an internal TSO→DSO
        # coordination signal with no counterpart in the MINLP formulation.
        if cr.tso_active:
            tso_parts = []
            if cr.tso_v_gen_pu is not None:
                tso_parts.append(cr.tso_v_gen_pu)
            if cr.tso_oltc_taps is not None:
                tso_parts.append(cr.tso_oltc_taps.astype(float))
            if cr.tso_q_der_mvar is not None:
                tso_parts.append(cr.tso_q_der_mvar)
            if cr.tso_shunt_states is not None:
                tso_parts.append(cr.tso_shunt_states.astype(float))
            if tso_parts:
                last_tso_physical = np.concatenate(tso_parts)

        if cr.dso_active:
            dso_parts = []
            if cr.dso_q_der_mvar is not None:
                dso_parts.append(cr.dso_q_der_mvar)
            if cr.dso_oltc_taps is not None:
                dso_parts.append(cr.dso_oltc_taps.astype(float))
            if cr.dso_shunt_states is not None:
                dso_parts.append(cr.dso_shunt_states.astype(float))
            if dso_parts:
                last_dso_physical = np.concatenate(dso_parts)

        cas_parts = []
        if last_tso_physical is not None:
            cas_parts.append(last_tso_physical)
        if last_dso_physical is not None:
            cas_parts.append(last_dso_physical)

        if not cas_parts:
            raise RuntimeError(
                f"compare_results: cascade has no physical actuator state at "
                f"minute {m}."
            )
        cas_norms.append(float(np.linalg.norm(np.concatenate(cas_parts))))

    ref_norms_arr = np.array(ref_norms)
    cas_norms_arr = np.array(cas_norms)

    # ==========================================================================
    # 3.  Per-minute comparison table (RMSD-primary, no weights)
    # ==========================================================================
    # Voltages scaled to milli-p.u. for readability (typical TN deviations are
    # in the 1–50 mp.u. range).
    print(
        f"\n  {'min':>5s}  {'Ref σ_TN':>9s}  {'Cas σ_TN':>9s}  "
        f"{'Gap%':>6s}  {'Ref ε_TN':>9s}  {'Cas ε_TN':>9s}  "
        f"{'ΔTap_Ref':>8s}  {'ΔTap_Cas':>8s}"
    )
    print(
        f"  {'':>5s}  {'[mp.u.]':>9s}  {'[mp.u.]':>9s}  "
        f"{'':>6s}  {'[mp.u.]':>9s}  {'[mp.u.]':>9s}  "
        f"{'[steps]':>8s}  {'[steps]':>8s}"
    )
    sep = f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}"
    print(sep)

    eps = 1e-12
    for i, m in enumerate(common_mins[:120]):
        gap = (cas_rmsd_tn[i] - ref_rmsd_tn[i]) / (ref_rmsd_tn[i] + eps) * 100.0
        print(
            f"  {m:5d}  "
            f"{ref_rmsd_tn[i]*1e3:9.4f}  {cas_rmsd_tn[i]*1e3:9.4f}  "
            f"{gap:+6.1f}%  "
            f"{ref_max_tn[i]*1e3:9.4f}  {cas_max_tn[i]*1e3:9.4f}  "
            f"{ref_oltc_switches[i]:8d}  {cas_oltc_switches[i]:8d}"
        )

    if len(common_mins) > 120:
        print(f"  ... ({len(common_mins) - 120} more timesteps omitted)")

    # ==========================================================================
    # 4.  Time-aggregated summary statistics
    # ==========================================================================
    rta  = np.array(ref_rmsd_tn)
    cta  = np.array(cas_rmsd_tn)
    rda  = np.array(ref_rmsd_dn)
    cda  = np.array(cas_rmsd_dn)
    rmta = np.array(ref_max_tn)
    cmta = np.array(cas_max_tn)
    rmda = np.array(ref_max_dn)
    cmda = np.array(cas_max_dn)

    print("\n" + "=" * 70)
    print("  Time-Aggregated Summary  (weight-free physical metrics)")
    print("=" * 70)
    print(f"  {'Metric':<38s}  {'MINLP Ref':>10s}  {'Cascade':>10s}  {'Gap %':>8s}")
    print(f"  {'-'*38}  {'-'*10}  {'-'*10}  {'-'*8}")

    def _row(label: str, r_val: float, c_val: float, unit: str = "") -> None:
        """Print one summary row with relative gap."""
        gap = (c_val - r_val) / (r_val + eps) * 100.0
        suffix = f"  [{unit}]" if unit else ""
        print(
            f"  {label:<38s}  {r_val:10.6f}  {c_val:10.6f}  {gap:+8.2f}%"
            f"{suffix}"
        )

    _row("Mean σ_V (TN)  [p.u.]",          float(np.mean(rta)),  float(np.mean(cta)))
    _row("Max  σ_V (TN)  [p.u.]",          float(np.max(rta)),   float(np.max(cta)))
    _row("Mean σ_V (DN)  [p.u.]",          float(np.mean(rda)),  float(np.mean(cda)))
    _row("Max  σ_V (DN)  [p.u.]",          float(np.max(rda)),   float(np.max(cda)))
    _row("Mean max|ΔV| TN  [p.u.]",        float(np.mean(rmta)), float(np.mean(cmta)))
    _row("Peak max|ΔV| TN  [p.u.]",        float(np.max(rmta)),  float(np.max(cmta)))
    _row("Mean max|ΔV| DN  [p.u.]",        float(np.mean(rmda)), float(np.mean(cmda)))
    _row("Peak max|ΔV| DN  [p.u.]",        float(np.max(rmda)),  float(np.max(cmda)))

    ref_q_l1_arr = np.array(ref_der_q_l1)
    cas_q_l1_arr = np.array(cas_der_q_l1)
    _row("Mean Σ|Q_DER|  [Mvar]",          float(np.mean(ref_q_l1_arr)),
                                            float(np.mean(cas_q_l1_arr)))

    ref_sw_total = int(np.sum(ref_oltc_switches))
    cas_sw_total = int(np.sum(cas_oltc_switches))
    print(
        f"  {'Total OLTC tap changes  [steps]':<38s}  "
        f"{ref_sw_total:>10d}  {cas_sw_total:>10d}  {'     N/A':>8s}"
    )

    _row("Mean ||u||_2  [mixed units]",
         float(np.mean(ref_norms_arr)), float(np.mean(cas_norms_arr)))

    print("=" * 70)

    # ==========================================================================
    # 5.  Common-objective suboptimality gap  J_MINLP(u*_cas) vs J_MINLP(u*_ref)
    # ==========================================================================
    #
    # Evaluate the MINLP's own objective function at both plant states.
    # This gives a fair, single-number comparison: the cascade can never
    # beat the MINLP reference, so gap ≥ 0 % (up to solver tolerance).

    # Use the reference's weights (same formulation we benchmark against).
    g_vt = ref.g_v_tn
    g_vd = ref.g_v_dn
    g_uu = ref.g_u

    ref_J: list[float] = []
    cas_J: list[float] = []
    gap_pct: list[float] = []

    # Reset carry-forward state for this second pass over common_mins
    _cf_tso_q: np.ndarray | None = None
    _cf_dso_q: np.ndarray | None = None

    for m in common_mins:
        rr = ref_by_min[m]
        cr = cas_by_min[m]

        # Reference plant state → J_ref(t)
        j_ref = evaluate_common_objective(
            rr.plant_tn_voltages_pu, rr.plant_dn_voltages_pu,
            rr.q_der_mvar,
            v_setpoint_pu, g_vt, g_vd, g_uu,
        )
        ref_J.append(j_ref)

        # Cascade plant state → J_cas(t)
        # Assemble the full DER Q vector (carry forward TSO/DSO pieces)
        if cr.tso_q_der_mvar is not None:
            _cf_tso_q = cr.tso_q_der_mvar
        if cr.dso_q_der_mvar is not None:
            _cf_dso_q = cr.dso_q_der_mvar

        cas_q_all = []
        if _cf_tso_q is not None:
            cas_q_all.append(_cf_tso_q)
        if _cf_dso_q is not None:
            cas_q_all.append(_cf_dso_q)
        cas_q_mvar = np.concatenate(cas_q_all) if cas_q_all else np.array([])

        j_cas = evaluate_common_objective(
            cr.plant_tn_voltages_pu, cr.plant_dn_voltages_pu,
            cas_q_mvar,
            v_setpoint_pu, g_vt, g_vd, g_uu,
        )
        cas_J.append(j_cas)

        # Relative gap (%)
        if abs(j_ref) > 1e-12:
            gap_pct.append((j_cas - j_ref) / j_ref * 100.0)
        else:
            gap_pct.append(0.0 if abs(j_cas) < 1e-12 else float('inf'))

    ref_J_arr = np.array(ref_J)
    cas_J_arr = np.array(cas_J)
    gap_arr = np.array(gap_pct)

    # -- Print per-timestep table -----------------------------------------------
    # The cascade starts far from the MINLP optimum and converges over time.
    # The "convergence ratio" η(t) = J_ref(t) / J_cas(t) shows how close the
    # cascade is to the oracle at each step: η = 1 means identical, η → 0
    # means the cascade is far away.
    print("\n" + "=" * 90)
    print("  COMMON-OBJECTIVE SUBOPTIMALITY ANALYSIS")
    print(f"  J(u) = {g_vt:g}·Σ(V_TN-V_set)² + {g_vd:g}·Σ(V_DN-V_set)²"
          f" + {g_uu:g}·Σ(Q_DER/s_base)²")
    print("=" * 90)
    print(f"  {'min':>5s}  {'J_ref':>12s}  {'J_cas':>12s}  "
          f"{'ΔJ':>12s}  {'η=J_ref/J_cas':>14s}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*14}")

    eta_list: list[float] = []
    for i in range(len(common_mins)):
        j_r = ref_J[i]
        j_c = cas_J[i]
        delta = j_c - j_r
        eta = j_r / j_c if abs(j_c) > 1e-15 else (1.0 if abs(j_r) < 1e-15 else 0.0)
        eta_list.append(eta)
        if i < 30 or i == len(common_mins) - 1:
            print(f"  {common_mins[i]:5d}  {j_r:12.4f}  {j_c:12.4f}  "
                  f"{delta:+12.4f}  {eta:14.6f}")
    if len(common_mins) > 31:
        print(f"  ... ({len(common_mins) - 31} rows omitted)")

    eta_arr = np.array(eta_list)

    # -- Aggregated summary ----------------------------------------------------
    print(f"\n  {'Metric':<42s}  {'MINLP Ref':>12s}  {'Cascade':>12s}")
    print(f"  {'-'*42}  {'-'*12}  {'-'*12}")
    _row2 = lambda lbl, rv, cv: print(
        f"  {lbl:<42s}  {rv:12.4f}  {cv:12.4f}"
    )
    _row2("J at t=1 (initial)",     ref_J[0],                  cas_J[0])
    _row2("J at t=T (final)",       ref_J[-1],                 cas_J[-1])
    _row2("Mean J(t)",              float(np.mean(ref_J_arr)), float(np.mean(cas_J_arr)))
    n_ss = max(1, len(ref_J_arr) // 10)
    _row2("Steady-state J (last 10%)",
          float(np.mean(ref_J_arr[-n_ss:])),
          float(np.mean(cas_J_arr[-n_ss:])))

    print(f"\n  Convergence ratio  η = J_ref / J_cas  (1.0 = oracle-optimal):")
    print(f"    η(t=1)     = {eta_list[0]:.6f}")
    print(f"    η(t=T)     = {eta_list[-1]:.6f}")
    print(f"    mean η     = {float(np.mean(eta_arr)):.6f}")
    ss_eta = float(np.mean(eta_arr[-n_ss:]))
    print(f"    η last 10% = {ss_eta:.6f}  "
          f"({'converged' if ss_eta > 0.90 else 'still converging' if ss_eta > 0.50 else 'far from optimum'})")

    # Absolute ΔJ
    delta_arr = cas_J_arr - ref_J_arr
    print(f"\n  Absolute suboptimality  ΔJ = J_cas − J_ref:")
    print(f"    ΔJ(t=1)     = {delta_arr[0]:+.4f}")
    print(f"    ΔJ(t=T)     = {delta_arr[-1]:+.4f}")
    print(f"    mean ΔJ     = {float(np.mean(delta_arr)):+.4f}")
    print(f"    ΔJ last 10% = {float(np.mean(delta_arr[-n_ss:])):+.4f}")
    print("=" * 90)

    # ==========================================================================
    # 6.  Generate all comparison plots
    # ==========================================================================
    from visualisation.plot_comparison import (
        plot_voltage_rmsd_comparison,
        plot_voltage_maxdev_comparison,
        plot_der_q_comparison,
        plot_oltc_switching_comparison,
        plot_control_effort_comparison,
        plot_common_objective_comparison,
        plot_3w_oltc_and_shunt_states,
        plot_2w_oltc_and_generator_states,
        plot_der_q_states,
    )

    plot_voltage_rmsd_comparison(
        common_mins,
        ref_rmsd_tn, cas_rmsd_tn,
        ref_rmsd_dn, cas_rmsd_dn,
    )
    plot_voltage_maxdev_comparison(
        common_mins,
        ref_max_tn, cas_max_tn,
        ref_max_dn, cas_max_dn,
    )
    plot_der_q_comparison(
        common_mins,
        list(ref_der_q_l1), list(cas_der_q_l1),
    )
    plot_oltc_switching_comparison(
        common_mins,
        ref_oltc_switches, cas_oltc_switches,
    )
    plot_control_effort_comparison(common_mins, ref_norms_arr, cas_norms_arr)
    plot_common_objective_comparison(
        common_mins, ref_J, cas_J, eta_list,
        g_v_tn=g_vt, g_v_dn=g_vd, g_u=g_uu,
    )
    plot_3w_oltc_and_shunt_states(common_mins, ref, cas)
    plot_2w_oltc_and_generator_states(common_mins, ref, cas)
    plot_der_q_states(common_mins, ref, cas)



# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Example contingency events (same as run_cascade main)
    events: List[ContingencyEvent] = [
        # ContingencyEvent(minute=100, element_type="line", element_index=3),
        # ContingencyEvent(minute=150, element_type="gen",  element_index=0),
    ]

    ref_result = run_reference(
        v_setpoint_pu=1.03,
        n_minutes=60*12,
        timestep_min=1,
        g_v_tn=100000,
        g_v_dn=1,
        g_u=0,
        solver='bonmin', # 'mindtpy', 'bonmin' -> works good, locally optimal
        mip_solver='gurobi',
        nlp_solver='ipopt',
        verbose=1,
        use_profiles=True,
        contingencies=events if events else None,
    )

    from core.cascade_config import CascadeConfig
    cas_config = CascadeConfig(
        v_setpoint_pu=1.03,
        n_minutes=60 * 12,
        tso_period_min=3,
        dso_period_min=1,
        verbose=1,
        g_v=200000,
        g_q=1,
        use_profiles=True,
        enable_reserve_observer=False,
        gw_oltc_cross_tso=0,
        gw_oltc_cross_dso=0,
        contingencies=events if events else [],
        reserve_cooldown_min=3,
        live_plot=False,
    )
    cas_result = run_cascade(cas_config)

    compare_results(ref_result, cas_result, v_setpoint_pu=1.05)


if __name__ == "__main__":
    main()
