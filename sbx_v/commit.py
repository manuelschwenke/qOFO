"""
sbx_v/commit.py
==============
Commit-instant integration (build plan §4 scheduling plane / §9
Phase 4): grant activation and expiry at window boundaries only, the
deterministic expiry ramp, the scheduled-envelope feedforward for the
DSO side, and the Incapability path into settlement.

Scheduled instants and R3
-------------------------
The priced segment structure (the ``AreaTierSpec`` tuple consumed by
:class:`sbx_v.miqp_cost.PricingSolver`) is a PURE FUNCTION of the
iteration index and of ledger state frozen at scheduled instants:

* **window commit instants** ``k = w · k_window`` — grants confirmed
  for window ``w`` activate, expired grants drop, the window's spec is
  built once and reused BY IDENTITY until the next scheduled instant;
* **expiry-ramp instants** — for a grant ending at the next boundary
  with no confirmed follow-up (evaluated ONCE at the ramp start
  ``k_end − ramp_steps`` and frozen), the granted segment bound shrinks
  deterministically to zero over the final ``ramp_steps`` iterations so
  ``q_pcc`` re-enters the band smoothly (plan §4 item 4).

No measurement ever changes the spec between these instants —
regression R3 asserts identity within stretches and the exact ramp
schedule.  A follow-up grant confirmed AFTER the ramp start does not
cancel the already-scheduled ramp (deterministic by freezing); it
activates normally at the boundary.

Emergency interaction: under the open-tail design (STATUS §1.2) every
magnitude beyond band + grant is already priced at the Grenzpreis, so a
Notfall-Abruf (:mod:`sbx_v.emergency`) changes consent bookkeeping and
logging, NOT the spec — the commit scheduler therefore never rebuilds
specs on an emergency call (R3 is preserved by construction).

Feedforward (MSR/MSC pattern reuse)
-----------------------------------
The MSR/MSC machinery steps a persistent per-interface offset at the
commit instant so the DSO does not counteract a KNOWN TSO-side change.
The SBX-V analogue of the "known change" is the scheduled granted
envelope: :meth:`CommitScheduler.scheduled_envelope_mvar` (band edge +
effective grant, per direction) and
:meth:`CommitScheduler.envelope_step_mvar` (its signed netted
per-iteration change — nonzero exactly at commit instants and during
expiry ramps).  The runner applies it as a synchronised lead on the
interface setpoints (Phase 5 wiring); in a neutral configuration both
are identically zero-change.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 4)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sbx_h.fail import rep1
from sbx_v.band import NormalBand
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction
from sbx_v.emergency import EmergencyHandler
from sbx_v.grants_ledger import GrantsLedger
from sbx_v.messages import (IncapabilityDeclaration, PotentialMessage,
                           Window)
from sbx_v.miqp_cost import AreaTierSpec, area_tier_spec
from sbx_v.need_flag import VerticalNeedDecision, VerticalNeedTracker
from sbx_v.pipeline import RequestPipeline
from sbx_v.settlement import (DSO_DELIVERS, TSO_DELIVERS, GrantRecord,
                             IncapabilityRecord)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AreaIterationInput:
    """Per-TSO-iteration data of one AggregationArea (feeds the need
    trackers; DP3 sign convention for the netted flow)."""

    q_pcc_netted_mvar: float
    v_dev_raising_pu: float    # undervoltage depth beyond the band [pu]
    v_dev_lowering_pu: float   # overvoltage depth beyond the band [pu]
    shortfall_raising_mvar: Optional[float] = None
    shortfall_lowering_mvar: Optional[float] = None


class CommitScheduler:
    """Window scheduler tying the Phase-3 pipeline to the Phase-1 MIQP
    cost layer at commit instants only (plan §4)."""

    def __init__(
        self,
        config: SBXVConfig,
        band_by_area: Mapping[str, NormalBand],
        pcc_rows_by_area: Mapping[str, Sequence[int]],
        *,
        ledger: Optional[GrantsLedger] = None,
        emergency: Optional[EmergencyHandler] = None,
    ) -> None:
        if not band_by_area:
            rep1("commit scheduler needs at least one AggregationArea")
        if set(band_by_area) != set(pcc_rows_by_area):
            rep1("band and PCC-row area sets must coincide",
                 bands=sorted(band_by_area),
                 rows=sorted(pcc_rows_by_area))
        if config.ramp_steps > config.k_window:
            rep1("ramp_steps must not exceed the window length in "
                 "iterations (plan §4: ramp over the FINAL steps of the "
                 "grant's last window)", ramp_steps=config.ramp_steps,
                 k_window=config.k_window)
        self.config = config
        self.bands = dict(band_by_area)
        self.pcc_rows = {a: tuple(int(r) for r in rows)
                         for a, rows in pcc_rows_by_area.items()}
        self.ledger = ledger if ledger is not None else \
            GrantsLedger(config.dq_grant_mvar)
        self.emergency = emergency
        self.pipeline = RequestPipeline(
            self.bands, self.ledger,
            reserve_margin_mvar=config.reserve_margin_mvar,
        )
        self._trackers: Dict[Tuple[str, Direction],
                             VerticalNeedTracker] = {
            (a, d): VerticalNeedTracker(
                a, d,
                n_persist=config.n_persist,
                n_clear=config.n_clear,
                sat_tol_mvar=config.sat_tol_mvar,
                v_dev_threshold_pu=config.v_dev_threshold_pu,
            )
            for a in sorted(band_by_area)
            for d in (Direction.RAISING, Direction.LOWERING)
        }
        #: Frozen expiry-ramp decisions: (window, area, direction) →
        #: grant magnitude being ramped (0.0 = no ramp), decided ONCE
        #: at the ramp start and never re-evaluated (R3 determinism).
        self._ramp_frozen: Dict[Tuple[int, str, str], float] = {}
        #: Spec cache: scheduled-instant key → spec tuple (identity
        #: within a stretch is the R3 guarantee).
        self._spec_cache: Dict[Tuple, Tuple[AreaTierSpec, ...]] = {}
        self._incap: List[Tuple[IncapabilityDeclaration, float]] = []
        #: Replay-comparable event log (primitives only).
        self.log: List[Tuple] = []

    # ------------------------------------------------------------------
    #  Window geometry
    # ------------------------------------------------------------------

    def window_of(self, k: int) -> int:
        if k < 0:
            rep1("iteration index must be non-negative", k=k)
        return k // self.config.k_window

    def is_commit_instant(self, k: int) -> bool:
        return k % self.config.k_window == 0

    def window_obj(self, w: int) -> Window:
        kw = self.config.k_window
        ts = self.config.tso_period_s
        return Window(index=w, k_start=w * kw, k_end=(w + 1) * kw,
                      t_start_s=w * kw * ts, t_end_s=(w + 1) * kw * ts)

    # ------------------------------------------------------------------
    #  Per-iteration entry point (need flags → pipeline)
    # ------------------------------------------------------------------

    def step(
        self,
        k: int,
        inputs: Mapping[str, AreaIterationInput],
        forecast_provider: Callable[
            [str, Direction, Window], Optional[PotentialMessage]],
    ) -> Dict[Tuple[str, Direction], VerticalNeedDecision]:
        """Update the need trackers with iteration ``k`` data and, on a
        set flag, run the request pipeline for the NEXT window (plan §4
        step 1; activation stays at the boundary).  Returns the
        decisions for logging."""
        if set(inputs) != set(self.bands):
            rep1("iteration inputs must cover exactly the configured "
                 "areas", got=sorted(inputs), expected=sorted(self.bands))
        w = self.window_of(k)
        decisions: Dict[Tuple[str, Direction], VerticalNeedDecision] = {}
        for (area, d), tracker in self._trackers.items():
            inp = inputs[area]
            edge = (self.bands[area].edge_mvar(d)
                    + self.ledger.granted_mvar(area, d, w))
            v_dev = (inp.v_dev_raising_pu if d is Direction.RAISING
                     else inp.v_dev_lowering_pu)
            shortfall = (inp.shortfall_raising_mvar
                         if d is Direction.RAISING
                         else inp.shortfall_lowering_mvar)
            decisions[(area, d)] = tracker.update(
                k, inp.q_pcc_netted_mvar, edge, v_dev,
                shortfall_estimate_mvar=shortfall,
            )
        if any(dec.flag for dec in decisions.values()):
            confirmations = self.pipeline.run_window(
                self.window_obj(w + 1), decisions, forecast_provider)
            for c in confirmations:
                self.log.append(("confirmed", k, c.aggregation_area_id,
                                 c.direction.value, c.n_quanta,
                                 c.window.index))
        return decisions

    # ------------------------------------------------------------------
    #  Spec schedule (R3)
    # ------------------------------------------------------------------

    def _effective_grant_mvar(self, area: str, d: Direction,
                              k: int) -> float:
        """Granted magnitude at iteration ``k`` including the frozen
        expiry-ramp schedule (plan §4 item 4)."""
        w = self.window_of(k)
        grant = self.ledger.granted_mvar(area, d, w)
        if grant <= 0.0:
            return 0.0
        kw = self.config.k_window
        rs = self.config.ramp_steps
        k_end = (w + 1) * kw
        ramp_start = k_end - rs
        if k < ramp_start:
            return grant
        key = (w, area, d.value)
        if key not in self._ramp_frozen:
            # Freeze the expiry decision ONCE at the ramp start: ramp
            # iff nothing is confirmed for the follow-up window.
            follow_up = self.ledger.granted_mvar(area, d, w + 1)
            self._ramp_frozen[key] = grant if follow_up <= 0.0 else 0.0
            if self._ramp_frozen[key] > 0.0:
                self.log.append(("ramp_scheduled", ramp_start, area,
                                 d.value, grant))
        ramped = self._ramp_frozen[key]
        if ramped <= 0.0:
            return grant
        # Linear descent over the final ramp_steps iterations,
        # reaching zero at the last in-window iteration.
        return grant * float(k_end - 1 - k) / float(rs)

    def _schedule_key(self, k: int) -> Tuple:
        """Scheduled-instant key: constant exactly where the spec is
        constant (window, plus the per-iteration ramp phase when and
        only when a ramp is active somewhere)."""
        w = self.window_of(k)
        kw = self.config.k_window
        ramp_start = (w + 1) * kw - self.config.ramp_steps
        if k < ramp_start:
            return (w, "main")
        ramping = any(
            self._effective_grant_mvar(a, d, k)
            != self.ledger.granted_mvar(a, d, w)
            for a in self.bands
            for d in (Direction.RAISING, Direction.LOWERING)
        )
        return (w, "ramp", k) if ramping else (w, "main")

    def specs_for(self, k: int) -> Optional[Tuple[AreaTierSpec, ...]]:
        """The priced segment structure at iteration ``k``.

        With ``miqp_pricing_enabled`` (the default) the band prices act
        from window 0 onward — beyond-band dispatch costs the
        Grenzpreis even without a grant (V-D9); active grants insert
        the Durchschnittspreis segment.  ``None`` (the PricingSolver
        neutral bypass, R1 byte-identity) is returned ONLY in the
        explicit neutral configuration ``miqp_pricing_enabled=False``.

        Within a scheduled stretch the SAME tuple object is returned
        (R3 identity); pass ``lambda: scheduler.specs_for(k_current)``
        as the PricingSolver's spec provider.
        """
        if not self.config.miqp_pricing_enabled:
            return None
        self.window_of(k)   # validates k ≥ 0
        key = self._schedule_key(k)
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        specs = tuple(
            area_tier_spec(
                area_id=a,
                pcc_output_rows=self.pcc_rows[a],
                band=self.bands[a],
                grant_raise_mvar=self._effective_grant_mvar(
                    a, Direction.RAISING, k),
                grant_lower_mvar=self._effective_grant_mvar(
                    a, Direction.LOWERING, k),
                config=self.config,
            )
            for a in sorted(self.bands)
        )
        self._spec_cache[key] = specs
        return specs

    # ------------------------------------------------------------------
    #  Scheduled-envelope feedforward (MSR/MSC pattern)
    # ------------------------------------------------------------------

    def scheduled_envelope_mvar(self, area: str, d: Direction,
                                k: int) -> float:
        """Band edge + effective grant at iteration ``k`` [Mvar]."""
        if area not in self.bands:
            rep1("unknown AggregationArea", area_id=area)
        return (self.bands[area].edge_mvar(d)
                + self._effective_grant_mvar(area, d, k))

    def envelope_step_mvar(self, area: str, k: int) -> float:
        """Signed netted scheduled envelope change from ``k−1`` to
        ``k`` — nonzero exactly at commit instants and during expiry
        ramps; the runner's synchronised feedforward lead."""
        if k == 0:
            return 0.0
        total = 0.0
        for d in (Direction.RAISING, Direction.LOWERING):
            delta = (self.scheduled_envelope_mvar(area, d, k)
                     - self.scheduled_envelope_mvar(area, d, k - 1))
            total += d.q_hv_sign * delta
        return total

    # ------------------------------------------------------------------
    #  Incapability path (→ settlement case 3a)
    # ------------------------------------------------------------------

    def declare_incapability(
        self, declaration: IncapabilityDeclaration,
    ) -> None:
        """Consume one Nichtverfügbarkeits-Meldung [AR §6.4.3; LF §6.9]:
        logged as a Reserve-Observer event; the provided remainder is
        recorded for settlement."""
        area = declaration.aggregation_area_id
        d = declaration.direction
        w = declaration.window.index
        granted = self.ledger.granted_mvar(area, d, w)
        if granted <= 0.0:
            rep1("incapability declared without an active grant",
                 area_id=area, direction=d.value, window_index=w,
                 q_shortfall_mvar=declaration.q_shortfall_mvar)
        provided = max(0.0, granted - declaration.q_shortfall_mvar)
        self._incap.append((declaration, provided))
        logger.warning(
            "SBX-V: IncapabilityDeclaration area %s direction %s "
            "window %d — shortfall %.1f of %.1f Mvar granted "
            "(provided %.1f Mvar) [AR §6.4.3].",
            area, d.value, w, declaration.q_shortfall_mvar, granted,
            provided,
        )
        self.log.append(("incapability", area, d.value, w,
                         declaration.q_shortfall_mvar, provided))

    def to_incapability_records(
        self, grant_records: Sequence[GrantRecord],
    ) -> Tuple[IncapabilityRecord, ...]:
        """Map declarations onto settlement inputs.

        Only TSO-delivers grants yield an :class:`IncapabilityRecord`
        (Tabelle 8.1 case 3a).  For DSO-delivers grants the declaration
        stays a logged Reserve-Observer event — the settlement detects
        the under-delivery from metering (Tabelle 8.2 case 2,
        STATUS §2.2)."""
        records: List[IncapabilityRecord] = []
        for declaration, provided in self._incap:
            area = declaration.aggregation_area_id
            d = declaration.direction
            w = declaration.window.index
            matching = [
                g for g in grant_records
                if g.area_id == area and g.direction is d
                and g.window_first <= w < g.window_end
            ]
            if not matching:
                rep1("incapability declaration without a matching "
                     "grant record", area_id=area, direction=d.value,
                     window_index=w)
            (grant,) = matching
            if grant.delivering_party == TSO_DELIVERS:
                records.append(IncapabilityRecord(
                    area_id=area, direction=d, window_index=w,
                    q_vh_provided_mvar=min(provided,
                                           grant.q_grant_mvar),
                ))
            else:
                logger.warning(
                    "SBX-V: incapability for a DSO-delivers grant "
                    "(area %s, direction %s, window %d) is settlement-"
                    "detected via Tabelle 8.2 case 2; declaration "
                    "remains a Reserve-Observer event.",
                    area, d.value, w,
                )
        return tuple(records)
