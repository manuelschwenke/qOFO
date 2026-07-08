"""
sbx/tie_line_model.py
=====================
Per-line π-model physics for SBX tie lines (plan v2 §3, normative).

Symbol map (code ↔ plan §2/§3)
------------------------------
* ``TieLineParams``        ↔ (r, x, b_sh, g_sh) of one tie line, per unit
                             on S_base = ``net.sn_mva`` and the terminal
                             ``vn_kv``.  ``b_sh_pu``/``g_sh_pu`` are the
                             PER-END (half) shunt values of the symmetric
                             π model, so the plan's Q-equation applies as
                             written with ``b_sh = b_sh_pu``.
* ``q_flow``               ↔ Q_A(V_A, V_B, P_A): reactive power at end A,
                             export-positive (pandapower load convention
                             at the A terminal), with δ resolved from the
                             P-equation on the small-|δ| branch.
* ``v_sched_for_q``        ↔ inverse solve: V_B such that Q_A = q_target.
* ``sensitivities``        ↔ (s_a, s_b, s_p) = TOTAL derivatives of Q_A
                             w.r.t. (V_A, V_B, P_A) — δ is implicit via
                             the P-equation; plain partials of Q_A would
                             be wrong (plan §3).

Equations (y = g + jb series admittance, b < 0; δ = δ_A − δ_B)
--------------------------------------------------------------
    P_A = (g + g_sh)·V_A² − V_A·V_B·(g·cos δ + b·sin δ)
    Q_A = −(b + b_sh)·V_A² − V_A·V_B·(g·sin δ − b·cos δ)

(The plan writes P_A without the g_sh term; all IEEE 39 tie lines have
g_sh = 0, so the two forms coincide there.  The g_sh term is carried for
correctness and validated by golden test 1.)

Units: voltages in p.u., P in MW and Q in Mvar at the interface
(converted internally on ``s_base_mva``).  Fail-fast via ``rep1``.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandapower as pp
from scipy.optimize import brentq

from sbx.fail import rep1

#: Guard margin of the transfer-limit assertion in :func:`q_flow`
#: (plan §3: |g·V_A² − P| ≤ (1 − margin)·V_A·V_B·√(g² + b²)).  A module
#: constant, not contract data: it only keeps the δ root find away from
#: the nose of the P–δ curve.
TRANSFER_LIMIT_MARGIN = 0.05

#: Central finite-difference steps of :func:`sensitivities` (plan §3).
FD_STEP_V_PU = 1.0e-4
FD_STEP_P_MW = 1.0e-2

#: Step-halving consistency bound of :func:`sensitivities` (plan §3: ≤ 1 %).
#: The absolute floor guards genuinely vanishing derivatives, where a
#: relative bound is meaningless (units: Mvar per p.u. / Mvar per MW).
FD_CONSISTENCY_REL = 0.01
FD_CONSISTENCY_FLOOR = 1.0e-6


@dataclass(frozen=True)
class TieLineParams:
    """π-model parameters of one tie line, per unit on ``s_base_mva``.

    ``b_sh_pu`` and ``g_sh_pu`` are the per-end (half) shunt values of the
    symmetric π model.  ``line_idx`` and ``vn_kv`` are provenance for
    diagnostics only.
    """

    r_pu: float
    x_pu: float
    b_sh_pu: float
    g_sh_pu: float
    s_base_mva: float
    line_idx: int = -1
    vn_kv: float = float("nan")

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in
                   (self.r_pu, self.x_pu, self.b_sh_pu, self.g_sh_pu,
                    self.s_base_mva)):
            rep1("TieLineParams contains non-finite entries",
                 line_idx=self.line_idx, r_pu=self.r_pu, x_pu=self.x_pu,
                 b_sh_pu=self.b_sh_pu, g_sh_pu=self.g_sh_pu,
                 s_base_mva=self.s_base_mva)
        if self.x_pu <= 0.0:
            rep1("TieLineParams: series reactance must be positive",
                 line_idx=self.line_idx, x_pu=self.x_pu)
        if self.r_pu < 0.0:
            rep1("TieLineParams: series resistance must be non-negative",
                 line_idx=self.line_idx, r_pu=self.r_pu)
        if self.b_sh_pu < 0.0 or self.g_sh_pu < 0.0:
            rep1("TieLineParams: shunt values must be non-negative",
                 line_idx=self.line_idx, b_sh_pu=self.b_sh_pu,
                 g_sh_pu=self.g_sh_pu)
        if self.s_base_mva <= 0.0:
            rep1("TieLineParams: s_base_mva must be positive",
                 line_idx=self.line_idx, s_base_mva=self.s_base_mva)

    @property
    def series_g_pu(self) -> float:
        """Series conductance g of y = 1/(r + jx) = g + jb."""
        return self.r_pu / (self.r_pu ** 2 + self.x_pu ** 2)

    @property
    def series_b_pu(self) -> float:
        """Series susceptance b of y = 1/(r + jx); b < 0 for x > 0."""
        return -self.x_pu / (self.r_pu ** 2 + self.x_pu ** 2)


def extract_tie_line_params(
    net: pp.pandapowerNet, line_idx: int
) -> TieLineParams:
    """Extract :class:`TieLineParams` from ``net.line`` (plan v2 §3).

    Handles ``length_km``, ``parallel`` and the standard-type-populated
    per-km columns; per unit on ``S_base = net.sn_mva`` and the terminal
    ``vn_kv``.  Line charging uses ``net.f_hz`` (Phase 0 finding A8: the
    IEEE 39 build runs at 60 Hz — never hard-code the frequency).
    """
    if line_idx not in net.line.index:
        rep1("tie line index not present in net.line", line_idx=line_idx)
    row = net.line.loc[line_idx]
    if not bool(row["in_service"]):
        rep1("tie line is out of service", line_idx=line_idx)

    for col in ("r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km",
                "g_us_per_km", "length_km", "parallel"):
        if col not in net.line.columns:
            rep1("net.line lacks a required column", column=col,
                 line_idx=line_idx)

    from_bus = int(row["from_bus"])
    to_bus = int(row["to_bus"])
    vn_from = float(net.bus.at[from_bus, "vn_kv"])
    vn_to = float(net.bus.at[to_bus, "vn_kv"])
    if not math.isclose(vn_from, vn_to, rel_tol=1e-9):
        rep1("tie line terminals have different voltage bases; the "
             "single-base π model does not apply",
             line_idx=line_idx, vn_from_kv=vn_from, vn_to_kv=vn_to)

    s_base = float(net.sn_mva)
    f_hz = float(net.f_hz)
    if s_base <= 0.0 or f_hz <= 0.0:
        rep1("net.sn_mva and net.f_hz must be positive",
             sn_mva=s_base, f_hz=f_hz)

    length = float(row["length_km"])
    parallel = int(row["parallel"])
    if length <= 0.0 or parallel < 1:
        rep1("tie line has non-physical length_km / parallel",
             line_idx=line_idx, length_km=length, parallel=parallel)

    z_base = vn_from ** 2 / s_base
    r_pu = float(row["r_ohm_per_km"]) * length / parallel / z_base
    x_pu = float(row["x_ohm_per_km"]) * length / parallel / z_base
    b_total_pu = (2.0 * math.pi * f_hz * float(row["c_nf_per_km"]) * 1e-9
                  * length * parallel * z_base)
    g_total_pu = (float(row["g_us_per_km"]) * 1e-6
                  * length * parallel * z_base)

    return TieLineParams(
        r_pu=r_pu, x_pu=x_pu,
        b_sh_pu=0.5 * b_total_pu, g_sh_pu=0.5 * g_total_pu,
        s_base_mva=s_base, line_idx=int(line_idx), vn_kv=vn_from,
    )


def _p_a_pu(
    delta_rad: float, v_a: float, v_b: float, params: TieLineParams
) -> float:
    """P at end A [p.u.], export-positive, at angle difference δ."""
    g, b = params.series_g_pu, params.series_b_pu
    return ((g + params.g_sh_pu) * v_a ** 2
            - v_a * v_b * (g * math.cos(delta_rad) + b * math.sin(delta_rad)))


def _q_a_pu(
    delta_rad: float, v_a: float, v_b: float, params: TieLineParams
) -> float:
    """Q at end A [p.u.], export-positive, at angle difference δ."""
    g, b = params.series_g_pu, params.series_b_pu
    return (-(b + params.b_sh_pu) * v_a ** 2
            - v_a * v_b * (g * math.sin(delta_rad) - b * math.cos(delta_rad)))


def q_flow(
    v_a_pu: float,
    v_b_pu: float,
    p_a_mw: float,
    params: TieLineParams,
    *,
    delta_max_rad: float = 0.6,
) -> float:
    """Q_A [Mvar] at end A for terminal voltages and active power P_A.

    δ is resolved from the P-equation on the small-|δ| branch via
    ``brentq`` over ``[-delta_max_rad, +delta_max_rad]`` (bracket
    asserted).  Sign convention: export from A positive at both P and Q
    (pandapower load convention at the A terminal).
    """
    if not (v_a_pu > 0.0 and v_b_pu > 0.0
            and math.isfinite(v_a_pu) and math.isfinite(v_b_pu)
            and math.isfinite(p_a_mw)):
        rep1("q_flow inputs must be finite with positive voltages",
             line_idx=params.line_idx, v_a_pu=v_a_pu, v_b_pu=v_b_pu,
             p_a_mw=p_a_mw)

    g, b = params.series_g_pu, params.series_b_pu
    p_pu = p_a_mw / params.s_base_mva

    # Transfer-limit assertion (plan §3), on the series part of P.
    p_series_pu = p_pu - params.g_sh_pu * v_a_pu ** 2
    transfer_pu = v_a_pu * v_b_pu * math.hypot(g, b)
    if abs(g * v_a_pu ** 2 - p_series_pu) > \
            (1.0 - TRANSFER_LIMIT_MARGIN) * transfer_pu:
        rep1("requested P exceeds the tie line's transfer capability "
             "(margin included)",
             line_idx=params.line_idx, p_a_mw=p_a_mw,
             p_series_pu=p_series_pu,
             limit_pu=(1.0 - TRANSFER_LIMIT_MARGIN) * transfer_pu,
             v_a_pu=v_a_pu, v_b_pu=v_b_pu)

    def h(delta: float) -> float:
        return _p_a_pu(delta, v_a_pu, v_b_pu, params) - p_pu

    h_lo, h_hi = h(-delta_max_rad), h(+delta_max_rad)
    if h_lo * h_hi > 0.0:
        rep1("δ bracket does not enclose the P-equation root",
             line_idx=params.line_idx, delta_max_rad=delta_max_rad,
             h_at_minus=h_lo, h_at_plus=h_hi, p_a_mw=p_a_mw,
             v_a_pu=v_a_pu, v_b_pu=v_b_pu)
    delta = brentq(h, -delta_max_rad, +delta_max_rad, xtol=1e-14, rtol=1e-15)

    return _q_a_pu(delta, v_a_pu, v_b_pu, params) * params.s_base_mva


def v_sched_for_q(
    v_a_pu: float,
    q_target_mvar: float,
    p_a_mw: float,
    params: TieLineParams,
    *,
    v_search_range_pu: Tuple[float, float] = (0.90, 1.10),
    delta_max_rad: float = 0.6,
) -> float:
    """V_B [p.u.] such that Q_A(v_a, V_B, p_a) = q_target (plan §3).

    Nested ``brentq`` over ``v_b ∈ v_search_range_pu``; the bracket is
    asserted with the endpoint Q values in the failure message.
    """
    lo, hi = v_search_range_pu

    def h(v_b: float) -> float:
        return q_flow(v_a_pu, v_b, p_a_mw, params,
                      delta_max_rad=delta_max_rad) - q_target_mvar

    h_lo, h_hi = h(lo), h(hi)
    if h_lo * h_hi > 0.0:
        rep1("v_b search range does not bracket the Q target",
             line_idx=params.line_idx, q_target_mvar=q_target_mvar,
             v_search_range_pu=v_search_range_pu,
             q_at_lo_mvar=h_lo + q_target_mvar,
             q_at_hi_mvar=h_hi + q_target_mvar,
             v_a_pu=v_a_pu, p_a_mw=p_a_mw)
    return float(brentq(h, lo, hi, xtol=1e-12, rtol=1e-15))


def _fd_triplet(
    v_a_pu: float, v_b_pu: float, p_a_mw: float, params: TieLineParams,
    h_v: float, h_p: float, delta_max_rad: float,
) -> Tuple[float, float, float]:
    """Central finite differences of q_flow at one step size."""
    s_a = (q_flow(v_a_pu + h_v, v_b_pu, p_a_mw, params,
                  delta_max_rad=delta_max_rad)
           - q_flow(v_a_pu - h_v, v_b_pu, p_a_mw, params,
                    delta_max_rad=delta_max_rad)) / (2.0 * h_v)
    s_b = (q_flow(v_a_pu, v_b_pu + h_v, p_a_mw, params,
                  delta_max_rad=delta_max_rad)
           - q_flow(v_a_pu, v_b_pu - h_v, p_a_mw, params,
                    delta_max_rad=delta_max_rad)) / (2.0 * h_v)
    s_p = (q_flow(v_a_pu, v_b_pu, p_a_mw + h_p, params,
                  delta_max_rad=delta_max_rad)
           - q_flow(v_a_pu, v_b_pu, p_a_mw - h_p, params,
                    delta_max_rad=delta_max_rad)) / (2.0 * h_p)
    return s_a, s_b, s_p


def sensitivities(
    v_a_pu: float,
    v_b_pu: float,
    p_a_mw: float,
    params: TieLineParams,
    *,
    delta_max_rad: float = 0.6,
) -> Tuple[float, float, float]:
    """TOTAL derivatives (s_a, s_b, s_p) of Q_A (plan §3).

    * ``s_a`` = dQ_A/dV_A [Mvar per p.u.] at fixed (V_B, P_A),
    * ``s_b`` = dQ_A/dV_B [Mvar per p.u.] at fixed (V_A, P_A),
    * ``s_p`` = dQ_A/dP_A [Mvar per MW] at fixed (V_A, V_B).

    δ is implicit via the P-equation (plain partials of the Q-equation
    would be wrong).  Central finite differences through :func:`q_flow`
    with step-halving consistency assertion (≤ 1 %, absolute floor for
    vanishing derivatives).
    """
    full = _fd_triplet(v_a_pu, v_b_pu, p_a_mw, params,
                       FD_STEP_V_PU, FD_STEP_P_MW, delta_max_rad)
    half = _fd_triplet(v_a_pu, v_b_pu, p_a_mw, params,
                       0.5 * FD_STEP_V_PU, 0.5 * FD_STEP_P_MW, delta_max_rad)
    for name, s_full, s_half in zip(("s_a", "s_b", "s_p"), full, half):
        tol = max(FD_CONSISTENCY_REL * abs(s_half), FD_CONSISTENCY_FLOOR)
        if abs(s_full - s_half) > tol:
            rep1("finite-difference sensitivity failed the step-halving "
                 "consistency check",
                 line_idx=params.line_idx, name=name,
                 s_full=s_full, s_half=s_half, tolerance=tol,
                 v_a_pu=v_a_pu, v_b_pu=v_b_pu, p_a_mw=p_a_mw)
    return half
