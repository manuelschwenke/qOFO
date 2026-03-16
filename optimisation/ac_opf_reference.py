"""
AC-OPF MINLP Reference Model
=============================

Full nonlinear AC Optimal Power Flow formulated as a Mixed-Integer
Nonlinear Program (MINLP) using Pyomo.  Serves as a "perfect-knowledge"
benchmark for the cascaded TSO-DSO online feedback controller.

The model is built from a converged pandapower network and uses the same
decision variables, objective function weights, and actuator bounds as
the cascaded MIQP controller so that objective values are directly
comparable.

Solver: MindtPy with Outer Approximation (OA), decomposing the MINLP
into alternating NLP sub-problems (IPOPT) and MILP master problems
(Gurobi).

Author: Claude (generated)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import pandapower as pp
import pyomo.environ as pyo

from network.build_tuda_net import NetworkMetadata
from sensitivity.index_helper import (
    get_ppc_trafo3w_branch_indices,
    pp_bus_to_ppc_bus,
)
from core.actuator_bounds import (
    GeneratorParameters,
    compute_generator_q_limits,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BranchInfo:
    """One branch in the _ppc representation."""
    ppc_idx: int          # index into _ppc['branch']
    from_bus: int         # ppc bus index (from)
    to_bus: int           # ppc bus index (to)
    r_pu: float           # resistance [system p.u.]
    x_pu: float           # reactance  [system p.u.]
    b_sh_pu: float        # total line charging susceptance [system p.u.]
    rate_a_mva: float     # thermal limit [MVA]  (0 = unlimited)
    tap_ratio: float      # current tap ratio (transformers; 1 for lines)
    shift_rad: float      # phase shift [rad]
    is_oltc: bool         # True if this branch has a controllable tap
    oltc_key: Optional[str] = None      # unique key for OLTC ('2w_<idx>' or '3w_<idx>')
    tap_step_pct: float = 0.0
    tap_neutral: int = 0
    tap_min: int = 0
    tap_max: int = 0


@dataclass(frozen=True)
class ShuntInfo:
    """One switchable shunt element."""
    pp_idx: int           # pandapower shunt index
    ppc_bus: int          # ppc bus index
    q_mvar: float         # rated Q per step [Mvar] (>0 reactor, <0 capacitor)
    b_pu: float           # susceptance per step [system p.u.]
    max_step: int         # max step (typically 1)
    current_step: int     # current step from net


@dataclass(frozen=True)
class DERInfo:
    """One controllable DER (static generator)."""
    pp_sgen_idx: int      # pandapower sgen index
    pp_bus: int           # pandapower bus index
    ppc_bus: int          # ppc bus index
    p_mw: float           # current P dispatch [MW]
    q_mvar: float         # current Q [Mvar]
    sn_mva: float         # rated apparent power [MVA]
    p_max_mw: float       # installed capacity [MW]


@dataclass(frozen=True)
class GenInfo:
    """One conventional synchronous generator."""
    pp_gen_idx: int
    pp_bus: int           # pandapower terminal bus
    ppc_bus: int          # ppc bus index
    p_mw: float           # current P dispatch [MW]
    q_mvar: float         # current Q [Mvar]
    sn_mva: float
    vm_pu: float          # AVR voltage setpoint [p.u.]
    params: GeneratorParameters


@dataclass
class NetworkData:
    """All data needed to build the Pyomo model, extracted from pandapower."""
    n_buses_ppc: int
    s_base_mva: float
    slack_ppc_bus: int
    ext_grid_vm_pu: float

    # Bus data indexed by ppc bus index
    bus_vm_pu: Dict[int, float]     # initial voltage magnitudes
    bus_va_rad: Dict[int, float]    # initial voltage angles
    bus_pd_pu: Dict[int, float]     # total active load [p.u.]
    bus_qd_pu: Dict[int, float]     # total reactive load [p.u.]
    bus_gs_pu: Dict[int, float]     # shunt conductance [p.u.]
    bus_bs_pu: Dict[int, float]     # shunt susceptance [p.u.] (fixed, non-switchable)
    bus_type: Dict[int, int]        # 1=PQ, 2=PV, 3=slack

    # Branch data
    branches: List[BranchInfo]

    # Controllable elements
    ders: List[DERInfo]
    gens: List[GenInfo]
    shunts: List[ShuntInfo]

    # Bus index mapping: pandapower bus -> ppc bus
    pp_to_ppc: Dict[int, int]

    # Monitored voltage buses (ppc indices)
    tn_v_buses_ppc: List[int]       # 380 kV monitored buses
    dn_v_buses_ppc: List[int]       # 110 kV monitored buses

    # OLTC info (for building integer variables)
    oltc_keys: List[str]            # ordered unique keys
    oltc_branch_idx: Dict[str, int] # key -> branches list index


@dataclass(frozen=True)
class ACOPFResult:
    """Results from a single-timestep AC-OPF solve."""
    voltages_pu: Dict[int, float]       # ppc_bus -> V_pu
    angles_rad: Dict[int, float]        # ppc_bus -> theta_rad
    q_der_mvar: List[float]             # per DER (same order as NetworkData.ders)
    v_gen_pu: List[float]               # per generator
    q_gen_mvar: List[float]             # per generator
    oltc_taps: Dict[str, int]           # oltc_key -> tap position
    shunt_steps: List[int]              # per shunt (same order as NetworkData.shunts)
    objective_value: float
    solver_status: str
    solve_time_s: float
    termination_condition: str

    # Bus voltages at monitored locations (for easy comparison)
    tn_voltages_pu: NDArray[np.float64]
    dn_voltages_pu: NDArray[np.float64]


# ═══════════════════════════════════════════════════════════════════════════════
#  Network Data Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_network_data(
    net: pp.pandapowerNet,
    meta: NetworkMetadata,
    tn_v_buses_pp: List[int],
    dn_v_buses_pp: List[int],
) -> NetworkData:
    """
    Extract all bus, branch, generator, shunt, and DER data from a converged
    pandapower network into a solver-friendly structure.

    Parameters
    ----------
    net : pandapowerNet
        Must have a converged power flow (``pp.runpp`` called).
    meta : NetworkMetadata
        Topology metadata from ``build_tuda_net()``.
    tn_v_buses_pp, dn_v_buses_pp : list of int
        Pandapower bus indices for voltage monitoring.
    """
    ppc = net._ppc
    s_base = float(ppc['baseMVA'])
    bus_data = ppc['bus']
    branch_data = ppc['branch']
    n_buses = len(bus_data)
    n_branches = len(branch_data)

    # ── pp → ppc bus mapping ──────────────────────────────────────────────
    pp_to_ppc: Dict[int, int] = {}
    bus_lookup = net._pd2ppc_lookups.get('bus', None)
    if bus_lookup is not None:
        for pp_idx in net.bus.index:
            ppc_idx = int(bus_lookup[pp_idx])
            pp_to_ppc[int(pp_idx)] = ppc_idx
    else:
        for pp_idx in net.bus.index:
            pp_to_ppc[int(pp_idx)] = pp_bus_to_ppc_bus(net, int(pp_idx))

    # ── Bus state from _ppc (voltage init, bus types) ─────────────────────
    bus_vm, bus_va, bus_tp = {}, {}, {}
    for i in range(n_buses):
        bus_vm[i] = float(bus_data[i, 7])       # VM
        bus_va[i] = float(bus_data[i, 8]) * math.pi / 180.0  # VA (deg→rad)
        bus_tp[i] = int(bus_data[i, 1])               # type

    # Find slack bus
    slack_buses = [i for i in range(n_buses) if bus_tp[i] == 3]
    slack_ppc = slack_buses[0] if slack_buses else 0
    ext_grid_vm = bus_vm[slack_ppc]

    # ── Bus injections from pandapower DataFrames (NOT from _ppc) ─────────
    # We build Pd/Qd from net.load only (pure load, no generators/sgens/shunts)
    # so that controllable elements enter only through decision variables.
    bus_pd: Dict[int, float] = {i: 0.0 for i in range(n_buses)}  # p.u.
    bus_qd: Dict[int, float] = {i: 0.0 for i in range(n_buses)}  # p.u.
    bus_gs: Dict[int, float] = {i: 0.0 for i in range(n_buses)}  # p.u.
    bus_bs: Dict[int, float] = {i: 0.0 for i in range(n_buses)}  # p.u.

    # Loads → Pd, Qd
    for ld in net.load.index:
        if not net.load.at[ld, "in_service"]:
            continue
        pp_bus = int(net.load.at[ld, "bus"])
        ppc_bus = pp_to_ppc.get(pp_bus, -1)
        if ppc_bus >= 0:
            bus_pd[ppc_bus] += float(net.load.at[ld, "p_mw"]) / s_base
            bus_qd[ppc_bus] += float(net.load.at[ld, "q_mvar"]) / s_base

    # NON-switchable shunts (bus Gs, Bs) — these are fixed elements
    # (Switchable shunts are handled separately as decision variables)
    all_switchable_shunts = set(meta.tertiary_shunt_indices) | set(meta.tn_shunt_indices)
    for sh in net.shunt.index:
        if int(sh) in all_switchable_shunts:
            continue  # skip switchable — handled as decision variable
        if not net.shunt.at[sh, "in_service"]:
            continue
        pp_bus = int(net.shunt.at[sh, "bus"])
        ppc_bus = pp_to_ppc.get(pp_bus, -1)
        if ppc_bus >= 0:
            step = int(net.shunt.at[sh, "step"])
            q_mvar = float(net.shunt.at[sh, "q_mvar"]) * step
            bus_bs[ppc_bus] += q_mvar / s_base  # same convention as _ppc

    # ── Branch data ───────────────────────────────────────────────────────
    # Identify OLTC branches: machine trafos (2W) and coupler 3W HV branches
    oltc_ppc_to_key: Dict[int, str] = {}

    # 2W machine trafos
    if hasattr(net._pd2ppc_lookups, '__getitem__'):
        branch_lookup = net._pd2ppc_lookups.get('branch', {})
    else:
        branch_lookup = {}

    for mt_idx in meta.machine_trafo_indices:
        if mt_idx not in net.trafo.index:
            continue
        # Find ppc branch index for this 2W trafo
        trafo_range = branch_lookup.get('trafo', None)
        if trafo_range is not None:
            trafo_positions = list(net.trafo.index)
            pos = trafo_positions.index(mt_idx)
            ppc_br = trafo_range[0] + pos
            key = f"2w_{mt_idx}"
            oltc_ppc_to_key[ppc_br] = key

    # 3W coupler trafos (HV branch only has OLTC)
    for t3w_idx in meta.coupler_trafo3w_indices:
        hv_br, _, _, _ = get_ppc_trafo3w_branch_indices(net, t3w_idx)
        key = f"3w_{t3w_idx}"
        oltc_ppc_to_key[hv_br] = key

    branches: List[BranchInfo] = []
    oltc_keys: List[str] = []
    oltc_branch_idx: Dict[str, int] = {}

    for i in range(n_branches):
        f_bus = int(np.real(branch_data[i, 0]))
        t_bus = int(np.real(branch_data[i, 1]))
        r = float(np.real(branch_data[i, 2]))
        x = float(np.real(branch_data[i, 3]))
        b_sh = float(np.real(branch_data[i, 4]))
        rate_a = float(np.real(branch_data[i, 5]))
        tap = float(np.real(branch_data[i, 8]))
        shift = float(np.real(branch_data[i, 9])) * math.pi / 180.0
        if tap == 0.0:
            tap = 1.0  # lines have tap=0 in ppc meaning ratio=1

        is_oltc = i in oltc_ppc_to_key
        oltc_key = oltc_ppc_to_key.get(i, None)

        # Get OLTC tap parameters
        tap_step_pct = 0.0
        tap_neutral = 0
        tap_min = 0
        tap_max = 0
        if is_oltc and oltc_key is not None:
            if oltc_key.startswith("2w_"):
                mt_idx = int(oltc_key.split("_")[1])
                tap_step_pct = float(net.trafo.at[mt_idx, "tap_step_percent"])
                tap_neutral = int(net.trafo.at[mt_idx, "tap_neutral"])
                tap_min = int(net.trafo.at[mt_idx, "tap_min"])
                tap_max = int(net.trafo.at[mt_idx, "tap_max"])
            elif oltc_key.startswith("3w_"):
                t3w_idx = int(oltc_key.split("_")[1])
                tap_step_pct = float(net.trafo3w.at[t3w_idx, "tap_step_percent"])
                tap_neutral = int(net.trafo3w.at[t3w_idx, "tap_neutral"])
                tap_min = int(net.trafo3w.at[t3w_idx, "tap_min"])
                tap_max = int(net.trafo3w.at[t3w_idx, "tap_max"])

        br = BranchInfo(
            ppc_idx=i, from_bus=f_bus, to_bus=t_bus,
            r_pu=r, x_pu=x, b_sh_pu=b_sh, rate_a_mva=rate_a,
            tap_ratio=tap, shift_rad=shift,
            is_oltc=is_oltc, oltc_key=oltc_key,
            tap_step_pct=tap_step_pct, tap_neutral=tap_neutral,
            tap_min=tap_min, tap_max=tap_max,
        )
        branches.append(br)
        if is_oltc and oltc_key is not None and oltc_key not in oltc_branch_idx:
            oltc_keys.append(oltc_key)
            oltc_branch_idx[oltc_key] = len(branches) - 1

    # ── DER data ──────────────────────────────────────────────────────────
    dn_buses = {int(b) for b in net.bus.index
                if str(net.bus.at[b, "subnet"]) == "DN"}
    ders: List[DERInfo] = []
    for s in net.sgen.index:
        name = str(net.sgen.at[s, "name"])
        if name.startswith("BOUND_"):
            continue
        pp_bus = int(net.sgen.at[s, "bus"])
        ders.append(DERInfo(
            pp_sgen_idx=int(s),
            pp_bus=pp_bus,
            ppc_bus=pp_to_ppc[pp_bus],
            p_mw=float(net.sgen.at[s, "p_mw"]),
            q_mvar=float(net.sgen.at[s, "q_mvar"]),
            sn_mva=float(net.sgen.at[s, "sn_mva"]),
            p_max_mw=float(net.sgen.at[s, "base_p_mw"])
                if "base_p_mw" in net.sgen.columns
                else float(net.sgen.at[s, "p_mw"]),
        ))

    # ── Generator data ────────────────────────────────────────────────────
    gens: List[GenInfo] = []
    for g in net.gen.index:
        pp_bus = int(net.gen.at[g, "bus"])
        sn = float(net.gen.at[g, "sn_mva"])
        p_mw = float(net.gen.at[g, "p_mw"])
        vm = float(net.gen.at[g, "vm_pu"])
        q_mvar = float(net.res_gen.at[g, "q_mvar"]) if g in net.res_gen.index else 0.0
        gens.append(GenInfo(
            pp_gen_idx=int(g),
            pp_bus=pp_bus,
            ppc_bus=pp_to_ppc[pp_bus],
            p_mw=p_mw, q_mvar=q_mvar,
            sn_mva=sn, vm_pu=vm,
            params=GeneratorParameters(
                s_rated_mva=sn, p_max_mw=p_mw,
                xd_pu=1.2, i_f_max_pu=2.65,
                beta=0.15, q0_pu=0.4,
            ),
        ))

    # ── Shunt data ────────────────────────────────────────────────────────
    shunts: List[ShuntInfo] = []
    all_shunt_indices = list(meta.tertiary_shunt_indices) + list(meta.tn_shunt_indices)
    for sh_idx in all_shunt_indices:
        if sh_idx not in net.shunt.index:
            continue
        pp_bus = int(net.shunt.at[sh_idx, "bus"])
        q_mvar = float(net.shunt.at[sh_idx, "q_mvar"])
        vn_kv = float(net.bus.at[pp_bus, "vn_kv"])
        # Susceptance: Q = B * V^2 => B = Q / V^2 (in system p.u.)
        # q_mvar > 0 means reactor (absorbing Q), B > 0 (inductive susceptance)
        b_pu = q_mvar / (s_base)  # at nominal voltage
        max_step = int(net.shunt.at[sh_idx, "max_step"])
        current_step = int(net.shunt.at[sh_idx, "step"])
        shunts.append(ShuntInfo(
            pp_idx=sh_idx, ppc_bus=pp_to_ppc[pp_bus],
            q_mvar=q_mvar, b_pu=b_pu,
            max_step=max_step, current_step=current_step,
        ))

    # ── Monitored buses (pp → ppc) ───────────────────────────────────────
    tn_v_ppc = [pp_to_ppc[b] for b in tn_v_buses_pp if b in pp_to_ppc]
    dn_v_ppc = [pp_to_ppc[b] for b in dn_v_buses_pp if b in pp_to_ppc]

    return NetworkData(
        n_buses_ppc=n_buses, s_base_mva=s_base,
        slack_ppc_bus=slack_ppc, ext_grid_vm_pu=ext_grid_vm,
        bus_vm_pu=bus_vm, bus_va_rad=bus_va,
        bus_pd_pu=bus_pd, bus_qd_pu=bus_qd,
        bus_gs_pu=bus_gs, bus_bs_pu=bus_bs,
        bus_type=bus_tp,
        branches=branches,
        ders=ders, gens=gens, shunts=shunts,
        pp_to_ppc=pp_to_ppc,
        tn_v_buses_ppc=tn_v_ppc, dn_v_buses_ppc=dn_v_ppc,
        oltc_keys=oltc_keys, oltc_branch_idx=oltc_branch_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DER Q Capability  (VDE-AR-N 4120)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_der_q_bounds_mvar(
    p_mw: float, p_max_mw: float, sn_mva: float,
) -> Tuple[float, float]:
    """VDE-AR-N 4120 simplified Q capability for a single DER."""
    q_cap = 0.33
    p_thresh = 0.2
    if p_max_mw > 0.0:
        p_ratio = p_mw / p_max_mw
    else:
        p_ratio = 0.0
    if p_ratio >= p_thresh:
        q_factor = q_cap
    else:
        q_factor = q_cap * (p_ratio / p_thresh)
    q_max = q_factor * sn_mva
    q_min = -q_factor * sn_mva
    return q_min, q_max


# ═══════════════════════════════════════════════════════════════════════════════
#  Pyomo Model Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_ac_opf_model(
    nd: NetworkData,
    v_setpoint_pu: float = 1.05,
    g_v_tn: float = 100000.0,
    g_v_dn: float = 10000.0,
    g_u: float = 0.1,
    v_min_pu: float = 0.90,
    v_max_pu: float = 1.10,
    gen_vm_min_pu: float = 0.95,
    gen_vm_max_pu: float = 1.10,
) -> pyo.ConcreteModel:
    """
    Build the full AC-OPF MINLP as a Pyomo ConcreteModel.

    Parameters
    ----------
    nd : NetworkData
        Extracted from ``extract_network_data()``.
    v_setpoint_pu : float
        Target voltage magnitude for all monitored buses.
    g_v_tn, g_v_dn : float
        Voltage deviation penalty weights for TN (380 kV) and DN (110 kV).
    g_u : float
        DER Q regularisation weight (penalise Q toward zero).
    v_min_pu, v_max_pu : float
        Voltage magnitude limits for all buses.
    gen_vm_min_pu, gen_vm_max_pu : float
        AVR voltage setpoint limits for synchronous generators.
    """
    m = pyo.ConcreteModel("AC_OPF_Reference")
    s_base = nd.s_base_mva

    # ── Sets ──────────────────────────────────────────────────────────────
    all_buses = list(range(nd.n_buses_ppc))
    m.BUSES = pyo.Set(initialize=all_buses)
    m.TN_VMON = pyo.Set(initialize=nd.tn_v_buses_ppc)
    m.DN_VMON = pyo.Set(initialize=nd.dn_v_buses_ppc)
    m.DER = pyo.RangeSet(0, len(nd.ders) - 1) if nd.ders else pyo.Set(initialize=[])
    m.GEN = pyo.RangeSet(0, len(nd.gens) - 1) if nd.gens else pyo.Set(initialize=[])
    m.OLTC = pyo.Set(initialize=nd.oltc_keys)
    m.SHUNT = pyo.RangeSet(0, len(nd.shunts) - 1) if nd.shunts else pyo.Set(initialize=[])

    # ── Bus Variables ─────────────────────────────────────────────────────
    m.V = pyo.Var(m.BUSES, bounds=(v_min_pu, v_max_pu),
                  initialize=lambda m, b: nd.bus_vm_pu.get(b, 1.0))
    m.theta = pyo.Var(m.BUSES, bounds=(-math.pi, math.pi),
                      initialize=lambda m, b: nd.bus_va_rad.get(b, 0.0))

    # ── Slack bus P and Q (free variables) ────────────────────────────────
    m.P_slack = pyo.Var(bounds=(-10000.0, 10000.0), initialize=0.0)  # p.u.
    m.Q_slack = pyo.Var(bounds=(-10000.0, 10000.0), initialize=0.0)

    # ── DER Q setpoints [Mvar] ────────────────────────────────────────────
    der_q_bounds = {}
    for d_idx, der in enumerate(nd.ders):
        qmin, qmax = _compute_der_q_bounds_mvar(
            der.p_mw, der.p_max_mw, der.sn_mva)
        der_q_bounds[d_idx] = (qmin, qmax)

    m.Q_der = pyo.Var(
        m.DER,
        bounds=lambda m, d: der_q_bounds.get(d, (-100.0, 100.0)),
        initialize=0.0,
    )

    # ── Generator variables ───────────────────────────────────────────────
    m.V_gen = pyo.Var(
        m.GEN,
        bounds=(gen_vm_min_pu, gen_vm_max_pu),
        initialize=lambda m, g: nd.gens[g].vm_pu if g < len(nd.gens) else 1.05,
    )
    # Generator Q is a free variable bounded by capability curve
    # We compute fixed bounds here (at nominal V) and add nonlinear constraints too
    gen_q_bounds = {}
    for g_idx, gen in enumerate(nd.gens):
        qmin, qmax = compute_generator_q_limits(
            gen.params, gen.p_mw, v_pu=gen.vm_pu)
        gen_q_bounds[g_idx] = (qmin / s_base, qmax / s_base)  # p.u.

    m.Q_gen = pyo.Var(
        m.GEN,
        bounds=lambda m, g: gen_q_bounds.get(g, (-5.0, 5.0)),
        initialize=lambda m, g: nd.gens[g].q_mvar / s_base if g < len(nd.gens) else 0.0,
    )

    # ── OLTC tap positions (integer) ──────────────────────────────────────
    oltc_tap_bounds = {}
    oltc_tap_init = {}
    for key in nd.oltc_keys:
        br = nd.branches[nd.oltc_branch_idx[key]]
        oltc_tap_bounds[key] = (br.tap_min, br.tap_max)
        # Infer current tap from current tap_ratio:
        # tap_ratio = 1 + (s - neutral) * step_pct / 100
        if br.tap_step_pct > 0:
            s_init = round((br.tap_ratio - 1.0) / (br.tap_step_pct / 100.0)
                           + br.tap_neutral)
            s_init = max(br.tap_min, min(br.tap_max, s_init))
        else:
            s_init = br.tap_neutral
        oltc_tap_init[key] = s_init

    m.s_oltc = pyo.Var(
        m.OLTC,
        within=pyo.Integers,
        bounds=lambda m, k: oltc_tap_bounds.get(k, (-9, 9)),
        initialize=lambda m, k: oltc_tap_init.get(k, 0),
    )

    # ── Shunt steps (binary: 0 or 1) ─────────────────────────────────────
    m.s_shunt = pyo.Var(
        m.SHUNT,
        within=pyo.Integers,
        bounds=lambda m, s: (0, nd.shunts[s].max_step),
        initialize=lambda m, s: nd.shunts[s].current_step,
    )

    # ── OLTC tap ratio expressions ────────────────────────────────────────
    def _tau_rule(m, k):
        br = nd.branches[nd.oltc_branch_idx[k]]
        return 1.0 + (m.s_oltc[k] - br.tap_neutral) * br.tap_step_pct / 100.0
    m.tau = pyo.Expression(m.OLTC, rule=_tau_rule)

    # ── Precompute bus-to-branch incidence ────────────────────────────────
    # For each bus, list of (branch_idx, 'from' | 'to')
    bus_branches: Dict[int, List[Tuple[int, str]]] = {b: [] for b in all_buses}
    for br_idx, br in enumerate(nd.branches):
        bus_branches[br.from_bus].append((br_idx, 'from'))
        bus_branches[br.to_bus].append((br_idx, 'to'))

    # Precompute bus-to-DER, bus-to-gen, bus-to-shunt mappings
    bus_ders: Dict[int, List[int]] = {}  # ppc_bus -> [der_idx, ...]
    for d_idx, der in enumerate(nd.ders):
        bus_ders.setdefault(der.ppc_bus, []).append(d_idx)

    bus_gens: Dict[int, List[int]] = {}
    for g_idx, gen in enumerate(nd.gens):
        bus_gens.setdefault(gen.ppc_bus, []).append(g_idx)

    bus_shunts: Dict[int, List[int]] = {}
    for sh_idx, sh in enumerate(nd.shunts):
        bus_shunts.setdefault(sh.ppc_bus, []).append(sh_idx)

    # ── AC Power Balance Constraints ──────────────────────────────────────
    # Standard pi-model with tap ratio:
    #
    # For a branch (f, t) with series admittance g_s + j*b_s,
    # shunt susceptance b_sh/2 on each end, tap ratio tau, and shift:
    #
    # FROM side:
    #   P_f = V_f^2 * g_s / tau^2
    #       + V_f * V_t * (-g_s*cos(theta_f-theta_t-shift) - b_s*sin(theta_f-theta_t-shift)) / tau
    #   Q_f = -V_f^2 * (b_s + b_sh/2) / tau^2
    #       + V_f * V_t * (-g_s*sin(theta_f-theta_t-shift) + b_s*cos(theta_f-theta_t-shift)) / tau
    #
    # TO side:
    #   P_t = V_t^2 * g_s
    #       + V_f * V_t * (-g_s*cos(theta_t-theta_f+shift) - b_s*sin(theta_t-theta_f+shift)) / tau
    #   Q_t = -V_t^2 * (b_s + b_sh/2)
    #       + V_f * V_t * (-g_s*sin(theta_t-theta_f+shift) + b_s*cos(theta_t-theta_f+shift)) / tau

    def _get_tau(m, br: BranchInfo):
        """Return Pyomo expression for tap ratio or float constant."""
        if br.is_oltc and br.oltc_key is not None:
            return m.tau[br.oltc_key]
        else:
            return br.tap_ratio

    def p_balance_rule(m, i):
        """P injection = P generation - P load at bus i."""
        # Generation side
        p_gen = 0.0
        if i == nd.slack_ppc_bus:
            p_gen += m.P_slack

        for g_idx in bus_gens.get(i, []):
            gen = nd.gens[g_idx]
            p_gen += gen.p_mw / s_base  # P is fixed (not a decision variable)

        for d_idx in bus_ders.get(i, []):
            der = nd.ders[d_idx]
            p_gen += der.p_mw / s_base  # DER P is fixed by profile

        # Load side
        p_load = nd.bus_pd_pu.get(i, 0.0)

        # Fixed shunt conductance (from bus_gs, not switchable)
        # P_shunt = Gs * V^2  (but usually 0 for non-lossy shunts)
        p_shunt = nd.bus_gs_pu.get(i, 0.0) * m.V[i]**2

        # Power flow from branches
        p_flow = 0.0
        for br_idx, side in bus_branches.get(i, []):
            br = nd.branches[br_idx]
            z2 = br.r_pu**2 + br.x_pu**2
            if z2 < 1e-20:
                continue  # skip zero-impedance branches
            g_s = br.r_pu / z2
            b_s = -br.x_pu / z2
            tau = _get_tau(m, br)

            if side == 'from':
                f, t = br.from_bus, br.to_bus
                dtheta = m.theta[f] - m.theta[t] - br.shift_rad
                p_flow += m.V[f]**2 * g_s / tau**2
                p_flow += m.V[f] * m.V[t] * (
                    -g_s * pyo.cos(dtheta) - b_s * pyo.sin(dtheta)
                ) / tau
            else:  # 'to'
                f, t = br.from_bus, br.to_bus
                dtheta = m.theta[t] - m.theta[f] + br.shift_rad
                p_flow += m.V[t]**2 * g_s
                p_flow += m.V[f] * m.V[t] * (
                    -g_s * pyo.cos(dtheta) - b_s * pyo.sin(dtheta)
                ) / tau

        return p_gen - p_load - p_shunt == p_flow

    m.p_balance = pyo.Constraint(m.BUSES, rule=p_balance_rule)

    def q_balance_rule(m, i):
        """Q injection = Q generation - Q load at bus i."""
        # Generation side
        q_gen = 0.0
        if i == nd.slack_ppc_bus:
            q_gen += m.Q_slack

        for g_idx in bus_gens.get(i, []):
            q_gen += m.Q_gen[g_idx]  # already in p.u.

        for d_idx in bus_ders.get(i, []):
            q_gen += m.Q_der[d_idx] / s_base  # Mvar → p.u.

        # Switchable shunt injection: Q_sh = step * b_pu * V^2
        for sh_idx in bus_shunts.get(i, []):
            sh = nd.shunts[sh_idx]
            q_gen += -m.s_shunt[sh_idx] * sh.b_pu * m.V[i]**2
            # Negative because: reactor has q_mvar > 0 (absorbing),
            # so b_pu > 0, and Q_shunt = -B*V^2 means absorbing = reducing Q_inj.
            # Actually, let's be precise about sign convention:
            # pandapower shunt convention: q_mvar > 0 means inductive (absorbing)
            # In power balance: Q_inj = Q_gen - Q_load = sum of flows
            # A reactor absorbs Q, so it acts like a load: Q_shunt = -abs_Q
            # b_pu = q_mvar / s_base > 0 for reactor
            # Q_injection from shunt = -step * b_pu * V^2  (load-like)

        # Load side
        q_load = nd.bus_qd_pu.get(i, 0.0)

        # Fixed bus shunt susceptance (non-switchable part)
        q_fixed_shunt = -nd.bus_bs_pu.get(i, 0.0) * m.V[i]**2

        # Power flow from branches
        q_flow = 0.0
        for br_idx, side in bus_branches.get(i, []):
            br = nd.branches[br_idx]
            z2 = br.r_pu**2 + br.x_pu**2
            if z2 < 1e-20:
                continue
            g_s = br.r_pu / z2
            b_s = -br.x_pu / z2
            b_sh = br.b_sh_pu
            tau = _get_tau(m, br)

            if side == 'from':
                f, t = br.from_bus, br.to_bus
                dtheta = m.theta[f] - m.theta[t] - br.shift_rad
                q_flow += -m.V[f]**2 * (b_s + b_sh / 2.0) / tau**2
                q_flow += m.V[f] * m.V[t] * (
                    -g_s * pyo.sin(dtheta) + b_s * pyo.cos(dtheta)
                ) / tau
            else:
                f, t = br.from_bus, br.to_bus
                dtheta = m.theta[t] - m.theta[f] + br.shift_rad
                q_flow += -m.V[t]**2 * (b_s + b_sh / 2.0)
                q_flow += m.V[f] * m.V[t] * (
                    -g_s * pyo.sin(dtheta) + b_s * pyo.cos(dtheta)
                ) / tau

        return q_gen - q_load - q_fixed_shunt == q_flow

    m.q_balance = pyo.Constraint(m.BUSES, rule=q_balance_rule)

    # ── Slack bus constraints ─────────────────────────────────────────────
    m.slack_theta = pyo.Constraint(
        expr=m.theta[nd.slack_ppc_bus] == 0.0)
    m.slack_voltage = pyo.Constraint(
        expr=m.V[nd.slack_ppc_bus] == nd.ext_grid_vm_pu)

    # ── Generator terminal voltage = AVR setpoint ─────────────────────────
    def gen_v_rule(m, g):
        gen = nd.gens[g]
        return m.V[gen.ppc_bus] == m.V_gen[g]
    if nd.gens:
        m.gen_v_eq = pyo.Constraint(m.GEN, rule=gen_v_rule)

    # ── Generator capability curve (Milano) ───────────────────────────────
    # Stator: p^2 + q^2 <= s_max^2  (in machine p.u.)
    def gen_stator_rule(m, g):
        gen = nd.gens[g]
        p_pu = gen.p_mw / gen.sn_mva
        q_pu = m.Q_gen[g] * s_base / gen.sn_mva
        return p_pu**2 + q_pu**2 <= 1.0
    if nd.gens:
        m.gen_stator = pyo.Constraint(m.GEN, rule=gen_stator_rule)

    # Rotor: p^2 + (q + v^2/xd)^2 <= (v*i_f_max/xd)^2
    def gen_rotor_rule(m, g):
        gen = nd.gens[g]
        p_pu = gen.p_mw / gen.sn_mva
        q_pu = m.Q_gen[g] * s_base / gen.sn_mva
        v = m.V[gen.ppc_bus]
        xd = gen.params.xd_pu
        i_f_max = gen.params.i_f_max_pu
        return p_pu**2 + (q_pu + v**2 / xd)**2 <= (v * i_f_max / xd)**2
    if nd.gens:
        m.gen_rotor = pyo.Constraint(m.GEN, rule=gen_rotor_rule)

    # Under-excitation: q >= -q0*v^2 + beta*p_max  (machine p.u.)
    def gen_ue_rule(m, g):
        gen = nd.gens[g]
        q_pu = m.Q_gen[g] * s_base / gen.sn_mva
        v = m.V[gen.ppc_bus]
        return q_pu >= -gen.params.q0_pu * v**2 + gen.params.beta * (gen.p_mw / gen.sn_mva)
    if nd.gens:
        m.gen_ue = pyo.Constraint(m.GEN, rule=gen_ue_rule)

    # ── Objective ─────────────────────────────────────────────────────────
    def obj_rule(m):
        obj = 0.0
        # Voltage deviation at TN buses
        for b in m.TN_VMON:
            obj += g_v_tn * (m.V[b] - v_setpoint_pu)**2
        # Voltage deviation at DN buses
        for b in m.DN_VMON:
            obj += g_v_dn * (m.V[b] - v_setpoint_pu)**2
        # DER Q regularisation
        for d in m.DER:
            obj += g_u * (m.Q_der[d] / s_base)**2  # in p.u. squared
        return obj

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Solver Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def solve_ac_opf(
    model: pyo.ConcreteModel,
    nd: NetworkData,
    solver_name: str = 'mindtpy',
    mip_solver: str = 'gurobi',
    nlp_solver: str = 'ipopt',
    time_limit: float = 300.0,
    verbose: bool = False,
) -> ACOPFResult:
    """
    Solve the Pyomo MINLP model and extract results.

    Uses MindtPy with Outer Approximation by default, falling back to
    NLP relaxation + rounding if MindtPy is not available.
    """
    t0 = time.time()

    try:
        if solver_name == 'mindtpy':
            solver = pyo.SolverFactory('mindtpy')
            results = solver.solve(
                model,
                strategy='OA',
                mip_solver=mip_solver,
                nlp_solver=nlp_solver,
                time_limit=time_limit,
                tee=verbose,
            )
        elif solver_name == 'scip':
            solver = pyo.SolverFactory('scip', solver_io='nl')
            solver.set_executable(r'C:\Program Files\SCIPOptSuite 10.0.1\bin\scip.exe', validate=False)
            # Set gap tolerance and time limit via options
            solver.options['presolving/milp/nthreads'] = 4
            solver.options['limits/gap'] = 0.1  # 1% gap
            solver.options['limits/time'] = 60  # 2 min
            solver.options['display/verblevel'] = 4
            results = solver.solve(model, tee=verbose)
        else:
            solver = pyo.SolverFactory(solver_name)
            if time_limit:
                # ── Handle Bonmin's specific time limit keyword ──
                if solver_name == 'bonmin':
                    solver.options['bonmin.time_limit'] = time_limit
                else:
                    solver.options['time_limit'] = time_limit

            results = solver.solve(model, tee=verbose)

            # Check if it actually solved successfully
        if results.solver.termination_condition in [
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded
        ]:
            print(
                f"  [AC-OPF] Primary solver returned {results.solver.termination_condition}. Trying NLP relaxation...")
            results = _solve_nlp_relaxation(model, nlp_solver, time_limit, verbose)

    except Exception as e:
        print(f"  [AC-OPF] Primary solver failed ({e}), trying NLP relaxation...")
        results = _solve_nlp_relaxation(model, nlp_solver, time_limit, verbose)

    solve_time = time.time() - t0

    # Extract solution
    status = str(results.solver.status)
    termination = str(results.solver.termination_condition)

    try:
        obj_val = float(pyo.value(model.obj))
    except Exception:
        obj_val = float('inf')

    # Voltages and angles
    voltages = {b: float(pyo.value(model.V[b])) for b in model.BUSES}
    angles = {b: float(pyo.value(model.theta[b])) for b in model.BUSES}

    # DER Q
    q_der = [float(pyo.value(model.Q_der[d])) for d in model.DER] if nd.ders else []

    # Generator
    v_gen = [float(pyo.value(model.V_gen[g])) for g in model.GEN] if nd.gens else []
    q_gen = [float(pyo.value(model.Q_gen[g])) * nd.s_base_mva
             for g in model.GEN] if nd.gens else []

    # OLTC taps
    oltc_taps = {k: int(round(float(pyo.value(model.s_oltc[k]))))
                 for k in model.OLTC}

    # Shunt steps
    shunt_steps = [int(round(float(pyo.value(model.s_shunt[s]))))
                   for s in model.SHUNT] if nd.shunts else []

    # Monitored voltages
    tn_v = np.array([voltages.get(b, 1.0) for b in nd.tn_v_buses_ppc])
    dn_v = np.array([voltages.get(b, 1.0) for b in nd.dn_v_buses_ppc])

    return ACOPFResult(
        voltages_pu=voltages, angles_rad=angles,
        q_der_mvar=q_der, v_gen_pu=v_gen, q_gen_mvar=q_gen,
        oltc_taps=oltc_taps, shunt_steps=shunt_steps,
        objective_value=obj_val,
        solver_status=status, solve_time_s=solve_time,
        termination_condition=termination,
        tn_voltages_pu=tn_v, dn_voltages_pu=dn_v,
    )


def _solve_nlp_relaxation(
    model: pyo.ConcreteModel,
    nlp_solver: str = 'ipopt',
    time_limit: float = 120.0,
    verbose: bool = False,
) -> object:
    """
    Fallback: relax integer variables to continuous, solve NLP, round.
    Returns a Pyomo results object.
    """
    # Temporarily relax integer constraints
    relaxed_vars = []
    for v in model.component_objects(pyo.Var, active=True):
        for idx in v:
            if v[idx].domain is pyo.Integers or v[idx].domain is pyo.Binary:
                v[idx].domain = pyo.Reals
                relaxed_vars.append((v, idx))

    solver = pyo.SolverFactory(nlp_solver)
    if time_limit:
        solver.options['max_cpu_time'] = time_limit
    results = solver.solve(model, tee=verbose)

    # Round integer variables and fix them
    for v, idx in relaxed_vars:
        val = float(pyo.value(v[idx]))
        rounded = int(round(val))
        lb, ub = v[idx].bounds
        if lb is not None:
            rounded = max(int(lb), rounded)
        if ub is not None:
            rounded = min(int(ub), rounded)
        v[idx].fix(rounded)

    # Re-solve NLP with fixed integers
    results2 = solver.solve(model, tee=verbose)

    # Unfix for potential later use
    for v, idx in relaxed_vars:
        v[idx].unfix()
        v[idx].domain = pyo.Integers

    return results2


# ═══════════════════════════════════════════════════════════════════════════════
#  Apply Results to Network
# ═══════════════════════════════════════════════════════════════════════════════

def apply_opf_result_to_net(
    net: pp.pandapowerNet,
    result: ACOPFResult,
    nd: NetworkData,
) -> None:
    """
    Write optimal DER Q, OLTC taps, shunt states, and generator V back
    to the pandapower network for a verification power flow.
    """
    # DER Q setpoints
    for d_idx, der in enumerate(nd.ders):
        if d_idx < len(result.q_der_mvar):
            net.sgen.at[der.pp_sgen_idx, "q_mvar"] = result.q_der_mvar[d_idx]

    # Generator AVR
    for g_idx, gen in enumerate(nd.gens):
        if g_idx < len(result.v_gen_pu):
            net.gen.at[gen.pp_gen_idx, "vm_pu"] = result.v_gen_pu[g_idx]

    # OLTC tap positions
    for key, tap in result.oltc_taps.items():
        if key.startswith("2w_"):
            mt_idx = int(key.split("_")[1])
            net.trafo.at[mt_idx, "tap_pos"] = tap
        elif key.startswith("3w_"):
            t3w_idx = int(key.split("_")[1])
            net.trafo3w.at[t3w_idx, "tap_pos"] = tap

    # Shunt states
    for sh_idx, sh in enumerate(nd.shunts):
        if sh_idx < len(result.shunt_steps):
            net.shunt.at[sh.pp_idx, "step"] = result.shunt_steps[sh_idx]
