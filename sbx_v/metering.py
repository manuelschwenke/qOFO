"""
sbx_v/metering.py
================
15-minute metering registers per AggregationArea (plan §7; [LF §5.5,
§8.3]).

Model of the four-quadrant metering [LF §5.5]: at every NVP the meter
accumulates reactive ENERGY separately per flow direction over each
15-minute window.  In this simulation the per-NVP signal is the signed
``q_hv_mvar`` (pandapower load convention at the EHV port, DP3), so the
two Q registers are

    e_q_pos_mvarh  ↔  (Q1 + Q2) of [LF Abb. 8.4]  (positive q_hv,
                       TS → DS, spannungssenkend / LOWERING energy)
    e_q_neg_mvarh  ↔  (Q3 + Q4)                    (negative q_hv,
                       DS → TS, spannungshebend / RAISING energy)

Saldierung per AggregationArea [LF §8.3]: 15-minute-sharp addition of
the work registers across the area's NVPs, conversion work → power, and
the signed mean ``Q = (Q1+Q2) − (Q3+Q4)`` per window (DP5: reference
points and settlement operate ONLY on this netted area value, never per
NVP).

Fail-fast (hard rule 1): recording intervals must be contiguous, must
not straddle window boundaries, and every closed window must be fully
covered; a trailing partial window is reported, never silently settled.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sbx_h.fail import rep1

#: Numerical tolerance [s] for interval/window boundary arithmetic.
_T_TOL_S = 1e-6


@dataclass(frozen=True)
class QuadrantRegisters:
    """Q-work registers of one NVP over one window [LF §5.5, Abb. 8.4].

    ``e_q_pos_mvarh`` = (Q1+Q2) analogue (positive ``q_hv``, LOWERING);
    ``e_q_neg_mvarh`` = (Q3+Q4) analogue (negative ``q_hv``, RAISING).
    Both non-negative by construction.
    """

    e_q_pos_mvarh: float
    e_q_neg_mvarh: float

    def __post_init__(self) -> None:
        for name in ("e_q_pos_mvarh", "e_q_neg_mvarh"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                rep1(f"register {name} must be finite and non-negative",
                     **{name: v})


@dataclass(frozen=True)
class AreaWindowRegister:
    """Netted 15-minute register of one AggregationArea [LF §8.3]."""

    area_id: str
    window_index: int
    t_start_s: float
    t_end_s: float
    e_q_pos_mvarh: float   # Σ over NVPs, (Q1+Q2) analogue
    e_q_neg_mvarh: float   # Σ over NVPs, (Q3+Q4) analogue
    q_mean_mvar: float     # signed netted mean = (pos − neg) / h_window


def aggregate_quadrant_registers(
    area_id: str,
    window_index: int,
    t_start_s: float,
    window_s: float,
    per_nvp: Sequence[QuadrantRegisters],
) -> AreaWindowRegister:
    """Saldierung of one window per [LF §8.3].

    Addition of the work registers across the area's NVPs, conversion
    work → power, signed mean ``Q = (Q1+Q2) − (Q3+Q4)``.
    """
    if not per_nvp:
        rep1("aggregation needs at least one NVP register",
             area_id=area_id, window_index=window_index)
    if window_s <= 0.0:
        rep1("window_s must be positive", window_s=window_s)
    e_pos = float(sum(r.e_q_pos_mvarh for r in per_nvp))
    e_neg = float(sum(r.e_q_neg_mvarh for r in per_nvp))
    h = window_s / 3600.0
    return AreaWindowRegister(
        area_id=area_id,
        window_index=window_index,
        t_start_s=t_start_s,
        t_end_s=t_start_s + window_s,
        e_q_pos_mvarh=e_pos,
        e_q_neg_mvarh=e_neg,
        q_mean_mvar=(e_pos - e_neg) / h,
    )


class AreaMeter:
    """Interval-recording meter of one AggregationArea.

    Feed one plant interval at a time via :meth:`record_step` (constant
    or varying ``dt_s``; intervals must be contiguous and must not
    straddle a window boundary — the plant step of the shared scenario
    is 60 s, the window 900 s, so alignment holds by construction).
    ``finalise`` returns the completed windows; a trailing partial
    window is exposed via :attr:`incomplete_tail_s` and never settled.
    """

    def __init__(self, area_id: str, n_nvp: int, window_s: float,
                 t_origin_s: float = 0.0) -> None:
        if not area_id:
            rep1("area_id must be non-empty", area_id=area_id)
        if n_nvp < 1:
            rep1("an AggregationArea needs at least one NVP", n_nvp=n_nvp)
        if window_s <= 0.0 or not math.isfinite(window_s):
            rep1("window_s must be positive", window_s=window_s)
        self.area_id = area_id
        self.n_nvp = int(n_nvp)
        self.window_s = float(window_s)
        self.t_origin_s = float(t_origin_s)
        self._t_next_s: Optional[float] = None
        # Per open window: per-NVP [e_pos, e_neg] accumulators.
        self._open: dict = {}
        self._closed: List[AreaWindowRegister] = []

    # ------------------------------------------------------------------

    def _window_of(self, t_s: float) -> int:
        return int(math.floor((t_s - self.t_origin_s + _T_TOL_S)
                              / self.window_s))

    def record_step(
        self,
        t_start_s: float,
        dt_s: float,
        q_hv_mvar_per_nvp: Sequence[float],
    ) -> None:
        """Record one interval of per-NVP signed boundary Q.

        Left-constant integration over ``[t_start_s, t_start_s+dt_s)``;
        the signed value is split into the two Q registers by sign
        (four-quadrant metering, [LF §5.5]).
        """
        if dt_s <= 0.0 or not math.isfinite(dt_s):
            rep1("dt_s must be positive", area_id=self.area_id, dt_s=dt_s)
        if len(q_hv_mvar_per_nvp) != self.n_nvp:
            rep1("per-NVP sample length mismatch", area_id=self.area_id,
                 got=len(q_hv_mvar_per_nvp), expected=self.n_nvp)
        if any(not math.isfinite(q) for q in q_hv_mvar_per_nvp):
            rep1("non-finite boundary Q sample", area_id=self.area_id,
                 t_start_s=t_start_s)
        if self._t_next_s is None:
            # First interval must start on the window grid.
            rel = (t_start_s - self.t_origin_s) / self.window_s
            if abs(rel - round(rel)) > _T_TOL_S:
                rep1("first metering interval must start on a window "
                     "boundary", area_id=self.area_id,
                     t_start_s=t_start_s, window_s=self.window_s)
        elif abs(t_start_s - self._t_next_s) > _T_TOL_S:
            rep1("metering intervals must be contiguous (gap or overlap "
                 "detected)", area_id=self.area_id,
                 t_start_s=t_start_s, expected_s=self._t_next_s)
        w0 = self._window_of(t_start_s)
        # End-of-interval lookup WITHOUT the snap tolerance of
        # _window_of (which would cancel the −tol and misclassify an
        # interval ending exactly on a boundary as straddling).
        w1 = int(math.floor(
            (t_start_s + dt_s - self.t_origin_s - _T_TOL_S)
            / self.window_s))
        if w0 != w1:
            rep1("metering interval straddles a window boundary",
                 area_id=self.area_id, t_start_s=t_start_s, dt_s=dt_s,
                 window_s=self.window_s)
        acc = self._open.setdefault(
            w0, {"t_cov_s": 0.0,
                 "nvp": [[0.0, 0.0] for _ in range(self.n_nvp)]})
        h = dt_s / 3600.0
        for j, q in enumerate(q_hv_mvar_per_nvp):
            if q >= 0.0:
                acc["nvp"][j][0] += q * h
            else:
                acc["nvp"][j][1] += (-q) * h
        acc["t_cov_s"] += dt_s
        self._t_next_s = t_start_s + dt_s
        if acc["t_cov_s"] >= self.window_s - _T_TOL_S:
            self._close(w0, acc)

    def _close(self, window_index: int, acc: dict) -> None:
        if abs(acc["t_cov_s"] - self.window_s) > 1e-3:
            rep1("window coverage mismatch at close", area_id=self.area_id,
                 window_index=window_index, t_cov_s=acc["t_cov_s"],
                 window_s=self.window_s)
        regs = [QuadrantRegisters(e_q_pos_mvarh=v[0], e_q_neg_mvarh=v[1])
                for v in acc["nvp"]]
        self._closed.append(aggregate_quadrant_registers(
            area_id=self.area_id,
            window_index=window_index,
            t_start_s=self.t_origin_s + window_index * self.window_s,
            window_s=self.window_s,
            per_nvp=regs,
        ))
        del self._open[window_index]

    # ------------------------------------------------------------------

    @property
    def incomplete_tail_s(self) -> float:
        """Coverage [s] of the still-open (never settled) tail window."""
        return float(sum(a["t_cov_s"] for a in self._open.values()))

    def finalise(self) -> List[AreaWindowRegister]:
        """Return the completed windows in order (the open tail, if any,
        is NOT included — check :attr:`incomplete_tail_s`)."""
        return sorted(self._closed, key=lambda r: r.window_index)
