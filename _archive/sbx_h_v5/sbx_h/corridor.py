"""
sbx_h/corridor.py
===============
Corridor registry and corridor-level model for SBX (plan v2 §2.1, §3).

Symbol map (code ↔ plan)
------------------------
* ``Corridor`` c           ↔ the set of all tie lines between one area
                             pair; reference end A = area with the smaller
                             area id; corridor flow q_corr = Σ_ℓ q_ℓ, each
                             q_ℓ at the reference end, positive = export
                             from A (§2.1).
* ``build_corridor_registry`` ↔ registry built from the global net against
                             the BME area partition (§ Phase 0/1): every
                             inter-area branch must be a line; each line's
                             ends map to two distinct areas; corridors are
                             grouped by unordered area pair.
* ``corridor_q_flow``      ↔ q_corr evaluation (per-line sum at end A).
* ``corridor_solve_dv``    ↔ the Step-4 scalar root find: one common shift
                             ``dv`` over the acting side's corridor
                             terminals such that q_corr = q_sched.
* ``corridor_sensitivities`` ↔ per-line (s_a, s_b, s_p) plus the per-side
                             sums (sensitivity of q_corr to a COMMON shift
                             of that side's terminal voltages, far end
                             held) — the s_corr of the capability LP
                             (§2.4 / v2.2 D13).

Clarification against the plan's Step-4 formula (recorded in
STATUS_SBX.md Phase 1): q_corr — and hence the root-find target — is
ALWAYS evaluated at the reference end A, where ``q_sched`` is defined;
``dv`` is applied to the acting side's terminals whichever end that is.
The plan's ``q_flow(v_std[far], v_std[act] + dv, ·)`` reads literally
only for the acting-side-B case; evaluating at the acting end when A acts
would silently redefine the schedule by the line losses/charging.

Fail-fast throughout via ``rep1``.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import pandapower as pp
from scipy.optimize import brentq

from sbx_h.fail import rep1
from sbx_h.tie_line_model import TieLineParams, extract_tie_line_params, \
    q_flow, sensitivities


@dataclass(frozen=True)
class CorridorLine:
    """One tie line of a corridor, oriented A → B (A = smaller area id)."""

    line_idx: int
    bus_a: int
    bus_b: int
    params: TieLineParams


@dataclass(frozen=True)
class Corridor:
    """All tie lines between one unordered area pair (plan §2.1).

    ``area_a < area_b``; end A is the reference end: corridor flow and all
    schedules are export-from-A positive.
    """

    area_a: int
    area_b: int
    lines: Tuple[CorridorLine, ...]

    def __post_init__(self) -> None:
        if self.area_a >= self.area_b:
            rep1("Corridor requires area_a < area_b",
                 area_a=self.area_a, area_b=self.area_b)
        if not self.lines:
            rep1("Corridor must contain at least one tie line",
                 area_a=self.area_a, area_b=self.area_b)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def terminals_a(self) -> List[int]:
        """Reference-end terminal buses, ascending, deduplicated."""
        return sorted({ln.bus_a for ln in self.lines})

    @property
    def terminals_b(self) -> List[int]:
        return sorted({ln.bus_b for ln in self.lines})


def build_corridor_registry(
    net: pp.pandapowerNet,
    area_map: Mapping[int, Sequence[int]],
) -> Dict[Tuple[int, int], Corridor]:
    """Group inter-area tie lines into corridors (plan Phase 0/1).

    Parameters
    ----------
    net :
        The global pandapower network (in-service topology is read).
    area_map :
        ``{area_id: [bus indices]}`` — the BME area partition (byte-
        identical to the BME zone_map; v2.2 item 6).  Buses not listed
        (generator terminals, DN feeders) carry no area and never anchor
        a tie.

    Raises via ``rep1`` on: unknown buses, a bus in two areas, any
    cross-area branch that is not a line (trafo/trafo3w/impedance — out
    of scope), or an empty corridor set.
    """
    if not area_map:
        rep1("area_map must contain at least one area")

    bus_set = {int(b) for b in net.bus.index}
    bus_area: Dict[int, int] = {}
    for area, buses in area_map.items():
        for b in buses:
            b = int(b)
            if b not in bus_set:
                rep1("area_map references a bus that is not in net.bus",
                     area=area, bus=b)
            if b in bus_area:
                rep1("bus assigned to two areas",
                     bus=b, areas=(bus_area[b], int(area)))
            bus_area[b] = int(area)

    def _area(bus: object) -> int | None:
        return bus_area.get(int(bus))

    # Every cross-area branch must be a line (plan Phase 0 assert).
    offenders: List[str] = []
    for t in net.trafo.index:
        if not bool(net.trafo.at[t, "in_service"]):
            continue
        zh, zl = _area(net.trafo.at[t, "hv_bus"]), _area(net.trafo.at[t, "lv_bus"])
        if zh is not None and zl is not None and zh != zl:
            offenders.append(f"trafo {t} (areas {zh}-{zl})")
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        for t in net.trafo3w.index:
            if not bool(net.trafo3w.at[t, "in_service"]):
                continue
            areas = {_area(net.trafo3w.at[t, c])
                     for c in ("hv_bus", "mv_bus", "lv_bus")}
            areas.discard(None)
            if len(areas) > 1:
                offenders.append(f"trafo3w {t} (areas {sorted(areas)})")
    if hasattr(net, "impedance") and len(net.impedance):
        for i in net.impedance.index:
            if not bool(net.impedance.at[i, "in_service"]):
                continue
            zf = _area(net.impedance.at[i, "from_bus"])
            zt = _area(net.impedance.at[i, "to_bus"])
            if zf is not None and zt is not None and zf != zt:
                offenders.append(f"impedance {i} (areas {zf}-{zt})")
    if offenders:
        rep1("cross-area non-line branches found — SBX corridors model "
             "tie LINES only (plan v2 Phase 0)", offenders=offenders)

    grouped: Dict[Tuple[int, int], List[CorridorLine]] = {}
    for li in net.line.index:
        if not bool(net.line.at[li, "in_service"]):
            continue
        fb = int(net.line.at[li, "from_bus"])
        tb = int(net.line.at[li, "to_bus"])
        zf, zt = _area(fb), _area(tb)
        if zf is None or zt is None or zf == zt:
            continue
        params = extract_tie_line_params(net, int(li))
        if zf < zt:
            cl = CorridorLine(int(li), fb, tb, params)
            key = (zf, zt)
        else:
            cl = CorridorLine(int(li), tb, fb, params)
            key = (zt, zf)
        grouped.setdefault(key, []).append(cl)

    if not grouped:
        rep1("no inter-area tie line found — an SBX corridor registry "
             "cannot be empty", areas=sorted(area_map.keys()))

    registry: Dict[Tuple[int, int], Corridor] = {}
    for (za, zb), lines in sorted(grouped.items()):
        lines = sorted(lines, key=lambda ln: ln.line_idx)
        registry[(za, zb)] = Corridor(za, zb, tuple(lines))
    return registry


def _check_aligned(
    corridor: Corridor,
    name_to_seq: Mapping[str, Sequence[float]],
) -> None:
    """Assert per-line argument sequences align with ``corridor.lines``."""
    for name, seq in name_to_seq.items():
        if len(seq) != corridor.n_lines:
            rep1("per-line argument does not align with the corridor's "
                 "line list (order: ascending line_idx)",
                 corridor=(corridor.area_a, corridor.area_b),
                 argument=name, expected=corridor.n_lines, got=len(seq))
        if not all(math.isfinite(float(v)) for v in seq):
            rep1("per-line argument contains non-finite entries",
                 corridor=(corridor.area_a, corridor.area_b),
                 argument=name, values=list(seq))


def corridor_q_flow(
    corridor: Corridor,
    v_a_pu: Sequence[float],
    v_b_pu: Sequence[float],
    p_a_mw: Sequence[float],
    *,
    delta_max_rad: float = 0.6,
) -> float:
    """q_corr [Mvar] at the reference end A (per-line sum, plan §2.1).

    All sequences are per line, in ``corridor.lines`` order (ascending
    ``line_idx``); ``p_a_mw`` is the per-line active power at end A,
    export-positive.
    """
    _check_aligned(corridor, {"v_a_pu": v_a_pu, "v_b_pu": v_b_pu,
                              "p_a_mw": p_a_mw})
    return sum(
        q_flow(float(va), float(vb), float(p), ln.params,
               delta_max_rad=delta_max_rad)
        for ln, va, vb, p in zip(corridor.lines, v_a_pu, v_b_pu, p_a_mw)
    )


def corridor_solve_dv(
    corridor: Corridor,
    v_std_a_pu: Sequence[float],
    v_std_b_pu: Sequence[float],
    p_sched_mw: Sequence[float],
    q_target_mvar: float,
    acting_end: str,
    *,
    dv_search_range_pu: Tuple[float, float] = (-0.05, +0.05),
    delta_max_rad: float = 0.6,
) -> float:
    """Step-4 scalar root find (plan §2.2): common shift ``dv`` on the
    acting side's terminals such that q_corr(v_std ± dv) = q_target.

    ``acting_end`` ∈ {"a", "b"}; the far end holds its contract voltages.
    q_corr is always evaluated at the reference end A (see module
    docstring).  The bracket is asserted with endpoint corridor flows in
    the failure message.
    """
    if acting_end not in ("a", "b"):
        rep1("acting_end must be 'a' or 'b'", acting_end=acting_end)
    _check_aligned(corridor, {"v_std_a_pu": v_std_a_pu,
                              "v_std_b_pu": v_std_b_pu,
                              "p_sched_mw": p_sched_mw})
    if not math.isfinite(q_target_mvar):
        rep1("q_target_mvar must be finite", q_target_mvar=q_target_mvar)

    def h(dv: float) -> float:
        if acting_end == "a":
            va = [float(v) + dv for v in v_std_a_pu]
            vb = [float(v) for v in v_std_b_pu]
        else:
            va = [float(v) for v in v_std_a_pu]
            vb = [float(v) + dv for v in v_std_b_pu]
        return corridor_q_flow(corridor, va, vb, p_sched_mw,
                               delta_max_rad=delta_max_rad) - q_target_mvar

    lo, hi = dv_search_range_pu
    h_lo, h_hi = h(lo), h(hi)
    if h_lo * h_hi > 0.0:
        rep1("dv search range does not bracket the corridor Q target",
             corridor=(corridor.area_a, corridor.area_b),
             acting_end=acting_end, q_target_mvar=q_target_mvar,
             dv_search_range_pu=dv_search_range_pu,
             q_corr_at_lo_mvar=h_lo + q_target_mvar,
             q_corr_at_hi_mvar=h_hi + q_target_mvar)
    return float(brentq(h, lo, hi, xtol=1e-12, rtol=1e-15))


def corridor_sensitivities(
    corridor: Corridor,
    v_a_pu: Sequence[float],
    v_b_pu: Sequence[float],
    p_a_mw: Sequence[float],
    *,
    delta_max_rad: float = 0.6,
) -> Tuple[List[Tuple[float, float, float]], float, float]:
    """Per-line (s_a, s_b, s_p) and the per-side common-shift sums.

    Returns ``(per_line, s_corr_a, s_corr_b)`` where ``per_line[k]`` is
    the total-derivative triplet of line k (plan §3) and
    ``s_corr_a = Σ_ℓ s_a,ℓ`` / ``s_corr_b = Σ_ℓ s_b,ℓ`` are the corridor
    sensitivities of q_corr to a COMMON voltage shift of the A / B side
    terminals with the far side held (the s_corr of the capability LP,
    plan §2.4 / v2.2 D13).  Units: Mvar per p.u. and Mvar per MW.
    """
    _check_aligned(corridor, {"v_a_pu": v_a_pu, "v_b_pu": v_b_pu,
                              "p_a_mw": p_a_mw})
    per_line = [
        sensitivities(float(va), float(vb), float(p), ln.params,
                      delta_max_rad=delta_max_rad)
        for ln, va, vb, p in zip(corridor.lines, v_a_pu, v_b_pu, p_a_mw)
    ]
    s_corr_a = sum(s[0] for s in per_line)
    s_corr_b = sum(s[1] for s in per_line)
    return per_line, s_corr_a, s_corr_b
