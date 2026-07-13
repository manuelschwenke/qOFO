"""
sbx_v/settlement.py
==================
Settlement per [LF §7] and the Fall-Kategorien of [LF Tabellen 8.1/8.2]
(plan §7, V-D8 — settlement fidelity is EXACT; the MIQP incentive of
:mod:`sbx_v.miqp_cost` may approximate, this module may not).

Worlds (delivering party) and tables
------------------------------------
* ``DSO_DELIVERS`` — the downstream DSO delivers to the requesting TSO:
  [LF Tabelle 8.2].  Reference point of Vorhalteleistung AND Blindarbeit
  is the OPPOSITE band edge [LF §7.2 case 1]; worked example
  [LF §8.2]: band ±50 Mvar, grant 100 Mvar beyond the upper edge ⇒
  Vorhalteleistung = 200 Mvar.
* ``TSO_DELIVERS`` — the upstream TSO delivers to the requesting DSO:
  [LF Tabelle 8.1].  Reference point is the SAME-side band edge
  [LF §7.2 case 2]; Blindarbeit inside the Normalbereich is deducted
  [LF §7.1].

All quantities operate on the NETTED per-window mean of one
AggregationArea (DP5; [LF §8.3] — see :mod:`sbx_v.metering`), in the side
coordinate ``s = sign_d · q`` (positive when operating on side ``d``,
DP3).

Case logic implemented (window classification)
----------------------------------------------
Tabelle 8.2 (``DSO_DELIVERS``, active grant, Abruf = the logged PCC-Q
reference; required — fail-fast when missing):

1. correct delivery (deviation within the ±``tolerance_frac``·VH
   15-min-mean tolerance [LF §4.7]) — capacity VH at the
   Leistungs-Durchschnittspreis, energy from the opposite edge to the
   operating point at the Arbeits-Durchschnittspreis;
2. under-delivery beyond tolerance — Leistungspreis suspended fully or
   pro rata ([LF §7.5.4]; provided fraction measured from the opposite
   edge), delivered energy still at the Durchschnittspreis;
3. over-delivery — payment capped at the called amounts [LF §7.5.5]
   (energy up to the Sollwert only);
4. call beyond the grant — within-grant parts at Durchschnittspreisen,
   the exceedance at Grenzpreisen (capacity exceedance = highest call
   beyond the grant maximum [LF §7.5.3], accrued per day).

Tabelle 8.1 (``TSO_DELIVERS``):

1. no grant, operation in band — free [LF §7.5.1];
2. grant, operation in band — capacity only;
3. call within the grant — capacity + energy beyond the own edge at
   Durchschnittspreisen;
3a. as 3 but the Vorhalteleistung was not available (voltage-limit
   violation at the NVP; consumed from an :class:`IncapabilityRecord`,
   Phase 3 wires the ``IncapabilityDeclaration`` message to it) —
   capacity none or pro rata, energy still at the Durchschnittspreis;
4. no grant, operation beyond the band — everything at Grenzpreisen
   (capacity = day-maximum provided exceedance);
4a. call beyond the grant — within like 3, exceedance at Grenzpreisen.

Ad-hoc downstream delivery (documented interpretation, STATUS §2.2):
with NO grant and the TSO's logged reference beyond the band edge, the
DSO delivers on posted potential ([LF §6]: the Potenzialmeldung implies
consent to the Abruf; V-D4: no gating).  Tabelle 8.2 has no such row;
treatment mirrors Tabelle 8.1 case 4 construction with the OWN edge as
reference: energy and day-maximum called exceedance at Grenzpreisen.
If instead the measured value exceeds the band WITHOUT a corresponding
reference, the DSO exceeded on its own initiative → Tabelle 8.1 case 4.

Capacity accrual is per day of Vorhaltung [LF §7.4]: a day touched by
an active grant accrues ``LP · VH``, scaled by the day's worst window
capacity fraction (full or pro-rata suspension, [LF §7.5.4]); grant
days are charged in full even when the grant covers only part of the
day (documented — scenarios are sub-day, plan §2: one
Verrechnungsperiode per scenario).

CSV output schema (``write_settlement_csv``)
--------------------------------------------
``<prefix>_windows.csv``: area_id, window_index, t_start_s, direction,
world, case, q_meas_mvar, q_set_mvar, e_avg_mvarh, e_grenz_mvarh,
pay_energy_avg_eur, pay_energy_grenz_eur, cap_frac, cap_exceed_mvar,
payer.
``<prefix>_days.csv``: area_id, direction, day_index, world, vh_mvar,
day_frac, pay_cap_avg_eur, exceed_mvar, pay_cap_grenz_eur, payer.
``<prefix>_totals.csv``: area_id, direction, world, payer,
pay_energy_avg_eur, pay_energy_grenz_eur, pay_cap_avg_eur,
pay_cap_grenz_eur, pay_total_eur.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 2)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sbx_h.fail import rep1
from sbx_v.band import NormalBand
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction

#: Delivering-party worlds.
DSO_DELIVERS = "dso_delivers"   # Tabelle 8.2 — downstream delivers
TSO_DELIVERS = "tso_delivers"   # Tabelle 8.1 — upstream delivers

#: Case labels (table-cased for traceability to the Leitfaden).
CASE_FREE = "free_band"          # [LF §7.5.1] / Tabelle 8.1 case 1
CASE_82_1 = "8.2-1"
CASE_82_2 = "8.2-2"
CASE_82_3 = "8.2-3"
CASE_82_4 = "8.2-4"
CASE_82_ADHOC = "8.2-4b_adhoc"   # documented interpretation (see above)
CASE_81_2 = "8.1-2"
CASE_81_3 = "8.1-3"
CASE_81_3A = "8.1-3a"
CASE_81_4 = "8.1-4"
CASE_81_4A = "8.1-4a"

#: Numerical tolerance [Mvar] for edge comparisons (NOT the contractual
#: tolerance of [LF §4.7], which comes from the config).
_Q_EPS_MVAR = 1e-9

_SECONDS_PER_DAY = 86400.0


# ----------------------------------------------------------------------
#  Inputs
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class WindowObservation:
    """One settled 15-minute window of one AggregationArea.

    ``q_meas_mvar`` — netted signed mean from :mod:`sbx_v.metering`;
    ``q_set_mvar`` — netted signed mean of the dispatched PCC-Q
    reference (the Abruf is the logged dispatch, plan §0 vocabulary);
    ``None`` when no reference was logged for the window.
    """

    area_id: str
    window_index: int
    t_start_s: float
    q_meas_mvar: float
    q_set_mvar: Optional[float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.q_meas_mvar):
            rep1("q_meas_mvar must be finite", area_id=self.area_id,
                 window_index=self.window_index)
        if self.q_set_mvar is not None and not math.isfinite(self.q_set_mvar):
            rep1("q_set_mvar must be finite when present",
                 area_id=self.area_id, window_index=self.window_index)


@dataclass(frozen=True)
class GrantRecord:
    """One confirmed grant (Vorhalteleistung extension beyond the band).

    ``q_grant_mvar`` is the granted MAGNITUDE beyond the own band edge;
    active over windows ``[window_first, window_end)`` (half-open).
    """

    area_id: str
    direction: Direction
    q_grant_mvar: float
    delivering_party: str
    window_first: int
    window_end: int

    def __post_init__(self) -> None:
        if self.q_grant_mvar <= 0.0 or not math.isfinite(self.q_grant_mvar):
            rep1("grant magnitude must be positive", area_id=self.area_id,
                 q_grant_mvar=self.q_grant_mvar)
        if self.delivering_party not in (DSO_DELIVERS, TSO_DELIVERS):
            rep1("delivering_party must be 'dso_delivers' or "
                 "'tso_delivers'", delivering_party=self.delivering_party)
        if self.window_end <= self.window_first:
            rep1("grant window range must be non-empty",
                 area_id=self.area_id, window_first=self.window_first,
                 window_end=self.window_end)


@dataclass(frozen=True)
class IncapabilityRecord:
    """Vorhalteleistung not (fully) available in one window
    (Tabelle 8.1 case 3a: voltage-limit violation at the NVP)."""

    area_id: str
    direction: Direction
    window_index: int
    q_vh_provided_mvar: float

    def __post_init__(self) -> None:
        if self.q_vh_provided_mvar < 0.0 or \
                not math.isfinite(self.q_vh_provided_mvar):
            rep1("provided Vorhalteleistung must be non-negative",
                 area_id=self.area_id,
                 q_vh_provided_mvar=self.q_vh_provided_mvar)


# ----------------------------------------------------------------------
#  Outputs
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class WindowSettlementRow:
    area_id: str
    window_index: int
    t_start_s: float
    direction: Optional[Direction]
    world: Optional[str]
    case: str
    q_meas_mvar: float
    q_set_mvar: Optional[float]
    e_avg_mvarh: float
    e_grenz_mvarh: float
    pay_energy_avg_eur: float
    pay_energy_grenz_eur: float
    cap_frac: float
    cap_exceed_mvar: float


@dataclass(frozen=True)
class DayCapacityRow:
    area_id: str
    direction: Direction
    day_index: int
    world: str
    vh_mvar: float
    day_frac: float
    pay_cap_avg_eur: float
    exceed_mvar: float
    pay_cap_grenz_eur: float


@dataclass(frozen=True)
class SettlementTotals:
    area_id: str
    direction: Direction
    world: str
    pay_energy_avg_eur: float
    pay_energy_grenz_eur: float
    pay_cap_avg_eur: float
    pay_cap_grenz_eur: float

    @property
    def pay_total_eur(self) -> float:
        return (self.pay_energy_avg_eur + self.pay_energy_grenz_eur
                + self.pay_cap_avg_eur + self.pay_cap_grenz_eur)


@dataclass(frozen=True)
class SettlementResult:
    window_rows: Tuple[WindowSettlementRow, ...]
    day_rows: Tuple[DayCapacityRow, ...]
    totals: Tuple[SettlementTotals, ...]


def payer_of(world: str) -> str:
    """The requesting party pays the delivering party [LF §7]."""
    if world == DSO_DELIVERS:
        return "tso"
    if world == TSO_DELIVERS:
        return "dso"
    rep1("unknown world", world=world)


# ----------------------------------------------------------------------
#  Engine
# ----------------------------------------------------------------------

class SettlementEngine:
    """Offline settlement over logged windows (plan §7, one
    Verrechnungsperiode per scenario)."""

    def __init__(
        self,
        config: SBXVConfig,
        bands: Mapping[str, NormalBand],
    ) -> None:
        self.config = config
        self.bands = dict(bands)
        wpd = _SECONDS_PER_DAY / config.window_s
        if abs(wpd - round(wpd)) > 1e-9:
            rep1("window_s must divide one day for the daily "
                 "Leistungspreis accrual [LF §7.4]",
                 window_s=config.window_s)
        self._windows_per_day = int(round(wpd))

    # ------------------------------------------------------------------

    def settle(
        self,
        observations: Sequence[WindowObservation],
        grants: Sequence[GrantRecord] = (),
        incapabilities: Sequence[IncapabilityRecord] = (),
    ) -> SettlementResult:
        self._assert_inputs(observations, grants, incapabilities)
        incap_by_key = {
            (r.area_id, r.direction, r.window_index): r
            for r in incapabilities
        }
        window_rows: List[WindowSettlementRow] = []
        # cap_frac per (grant identity, window) for the daily accrual.
        cap_frac: Dict[Tuple[int, int], float] = {}
        # Day-max exceedance per (area, direction, world, day).
        exceed: Dict[Tuple[str, Direction, str, int], float] = {}

        grants_by_area: Dict[str, List[GrantRecord]] = {}
        for g in grants:
            grants_by_area.setdefault(g.area_id, []).append(g)

        for obs in sorted(observations,
                          key=lambda o: (o.area_id, o.window_index)):
            band = self._band_of(obs.area_id)
            day = self._day_of(obs)
            rows_before = len(window_rows)
            for d in (Direction.RAISING, Direction.LOWERING):
                grant = self._active_grant(
                    grants_by_area.get(obs.area_id, ()), d,
                    obs.window_index)
                if grant is not None and \
                        grant.delivering_party == DSO_DELIVERS:
                    row = self._settle_82(obs, band, d, grant)
                    cap_frac[(id(grant), obs.window_index)] = row.cap_frac
                elif grant is not None:
                    incap = incap_by_key.get(
                        (obs.area_id, d, obs.window_index))
                    row = self._settle_81(obs, band, d, grant, incap)
                    cap_frac[(id(grant), obs.window_index)] = row.cap_frac
                else:
                    row = self._settle_no_grant(obs, band, d)
                if row is None:
                    continue
                window_rows.append(row)
                if row.cap_exceed_mvar > 0.0:
                    key = (obs.area_id, d, row.world, day)
                    exceed[key] = max(exceed.get(key, 0.0),
                                      row.cap_exceed_mvar)
            if len(window_rows) == rows_before:
                # Fully free window [LF §7.5.1] — reported, not priced.
                window_rows.append(WindowSettlementRow(
                    area_id=obs.area_id, window_index=obs.window_index,
                    t_start_s=obs.t_start_s, direction=None, world=None,
                    case=CASE_FREE, q_meas_mvar=obs.q_meas_mvar,
                    q_set_mvar=obs.q_set_mvar, e_avg_mvarh=0.0,
                    e_grenz_mvarh=0.0, pay_energy_avg_eur=0.0,
                    pay_energy_grenz_eur=0.0, cap_frac=1.0,
                    cap_exceed_mvar=0.0))

        day_rows = self._accrue_capacity(observations, grants, cap_frac,
                                         exceed)
        totals = self._totalise(window_rows, day_rows)
        return SettlementResult(
            window_rows=tuple(window_rows),
            day_rows=tuple(day_rows),
            totals=tuple(totals),
        )

    # ------------------------------------------------------------------
    #  Tabelle 8.2 — downstream DSO delivers (primary direction)
    # ------------------------------------------------------------------

    def _settle_82(
        self,
        obs: WindowObservation,
        band: NormalBand,
        d: Direction,
        grant: GrantRecord,
    ) -> WindowSettlementRow:
        if obs.q_set_mvar is None:
            rep1("Tabelle 8.2 settlement needs the logged PCC-Q "
                 "reference (Abruf) for every granted window",
                 area_id=obs.area_id, window_index=obs.window_index,
                 direction=d)
        edge_own = band.edge_mvar(d)
        edge_opp = band.edge_mvar(d.opposite)
        vh = edge_opp + edge_own + grant.q_grant_mvar   # [LF §7.2 case 1]
        grant_max_s = edge_own + grant.q_grant_mvar
        tol = self.config.tolerance_frac * vh           # [LF §4.7]
        h = self.config.window_s / 3600.0
        s_meas = d.q_hv_sign * obs.q_meas_mvar
        s_set = d.q_hv_sign * obs.q_set_mvar

        e_avg = e_grenz = 0.0
        cap_frac = 1.0
        cap_exceed = 0.0
        if s_set > grant_max_s + _Q_EPS_MVAR:
            case = CASE_82_4                            # [LF §7.5.3]
            e_avg = max(0.0, min(s_meas, grant_max_s) + edge_opp) * h
            e_grenz = max(0.0, s_meas - grant_max_s) * h
            cap_exceed = s_set - grant_max_s
        elif s_meas < s_set - tol:
            case = CASE_82_2                            # [LF §7.5.4]
            e_avg = max(0.0, s_meas + edge_opp) * h
            cap_frac = min(1.0, max(0.0, (s_meas + edge_opp) / vh))
        elif s_meas > s_set + tol:
            case = CASE_82_3                            # [LF §7.5.5]
            e_avg = max(0.0, min(s_meas, s_set) + edge_opp) * h
        else:
            case = CASE_82_1
            e_avg = max(0.0, s_meas + edge_opp) * h

        return WindowSettlementRow(
            area_id=obs.area_id, window_index=obs.window_index,
            t_start_s=obs.t_start_s, direction=d, world=DSO_DELIVERS,
            case=case, q_meas_mvar=obs.q_meas_mvar,
            q_set_mvar=obs.q_set_mvar, e_avg_mvarh=e_avg,
            e_grenz_mvarh=e_grenz,
            pay_energy_avg_eur=(
                e_avg * self.config.price_arb_avg_eur_per_mvarh),
            pay_energy_grenz_eur=(
                e_grenz * self.config.price_arb_grenz_eur_per_mvarh),
            cap_frac=cap_frac, cap_exceed_mvar=cap_exceed)

    # ------------------------------------------------------------------
    #  Tabelle 8.1 — upstream TSO delivers (reverse direction)
    # ------------------------------------------------------------------

    def _settle_81(
        self,
        obs: WindowObservation,
        band: NormalBand,
        d: Direction,
        grant: GrantRecord,
        incap: Optional[IncapabilityRecord],
    ) -> WindowSettlementRow:
        edge_own = band.edge_mvar(d)
        vh = grant.q_grant_mvar                         # [LF §7.2 case 2]
        grant_max_s = edge_own + grant.q_grant_mvar
        h = self.config.window_s / 3600.0
        s_meas = d.q_hv_sign * obs.q_meas_mvar

        e_avg = e_grenz = 0.0
        cap_frac = 1.0
        cap_exceed = 0.0
        if incap is not None:
            # Case 3a dominates: Vorhalteleistung counted as not (fully)
            # provided; energy beyond the band still at Durchschnitt.
            case = CASE_81_3A
            e_avg = max(0.0, s_meas - edge_own) * h
            cap_frac = min(1.0, incap.q_vh_provided_mvar / vh)
        elif s_meas <= edge_own + _Q_EPS_MVAR:
            case = CASE_81_2
        elif s_meas <= grant_max_s + _Q_EPS_MVAR:
            case = CASE_81_3
            e_avg = (s_meas - edge_own) * h             # [LF §7.1]
        else:
            case = CASE_81_4A
            e_avg = (grant_max_s - edge_own) * h
            e_grenz = (s_meas - grant_max_s) * h
            cap_exceed = s_meas - grant_max_s

        return WindowSettlementRow(
            area_id=obs.area_id, window_index=obs.window_index,
            t_start_s=obs.t_start_s, direction=d, world=TSO_DELIVERS,
            case=case, q_meas_mvar=obs.q_meas_mvar,
            q_set_mvar=obs.q_set_mvar, e_avg_mvarh=e_avg,
            e_grenz_mvarh=e_grenz,
            pay_energy_avg_eur=(
                e_avg * self.config.price_arb_avg_eur_per_mvarh),
            pay_energy_grenz_eur=(
                e_grenz * self.config.price_arb_grenz_eur_per_mvarh),
            cap_frac=cap_frac, cap_exceed_mvar=cap_exceed)

    # ------------------------------------------------------------------
    #  No grant on this side
    # ------------------------------------------------------------------

    def _settle_no_grant(
        self,
        obs: WindowObservation,
        band: NormalBand,
        d: Direction,
    ) -> Optional[WindowSettlementRow]:
        edge_own = band.edge_mvar(d)
        h = self.config.window_s / 3600.0
        s_meas = d.q_hv_sign * obs.q_meas_mvar
        if s_meas <= edge_own + _Q_EPS_MVAR:
            return None                                 # free [LF §7.5.1]
        s_set = (None if obs.q_set_mvar is None
                 else d.q_hv_sign * obs.q_set_mvar)
        if s_set is not None and s_set > edge_own + _Q_EPS_MVAR:
            # TSO called beyond the band on posted potential — ad-hoc
            # downstream delivery (documented interpretation).
            case = CASE_82_ADHOC
            world = DSO_DELIVERS
            cap_exceed = s_set - edge_own               # highest call
        else:
            # DSO exceeded on its own — the upstream TSO delivered.
            case = CASE_81_4
            world = TSO_DELIVERS
            cap_exceed = s_meas - edge_own              # provided
        e_grenz = (s_meas - edge_own) * h
        return WindowSettlementRow(
            area_id=obs.area_id, window_index=obs.window_index,
            t_start_s=obs.t_start_s, direction=d, world=world,
            case=case, q_meas_mvar=obs.q_meas_mvar,
            q_set_mvar=obs.q_set_mvar, e_avg_mvarh=0.0,
            e_grenz_mvarh=e_grenz, pay_energy_avg_eur=0.0,
            pay_energy_grenz_eur=(
                e_grenz * self.config.price_arb_grenz_eur_per_mvarh),
            cap_frac=1.0, cap_exceed_mvar=cap_exceed)

    # ------------------------------------------------------------------
    #  Daily capacity accrual [LF §7.4]
    # ------------------------------------------------------------------

    def _accrue_capacity(
        self,
        observations: Sequence[WindowObservation],
        grants: Sequence[GrantRecord],
        cap_frac: Dict[Tuple[int, int], float],
        exceed: Dict[Tuple[str, Direction, str, int], float],
    ) -> List[DayCapacityRow]:
        rows: List[DayCapacityRow] = []
        obs_windows = {(o.area_id, o.window_index) for o in observations}
        day_of_window = {
            (o.area_id, o.window_index): self._day_of(o)
            for o in observations
        }
        # Granted capacity at the Durchschnittspreis, per grant per day.
        emitted: Dict[Tuple[str, Direction, str, int], DayCapacityRow] = {}
        for g in grants:
            band = self._band_of(g.area_id)
            if g.delivering_party == DSO_DELIVERS:
                vh = (band.edge_mvar(g.direction.opposite)
                      + band.edge_mvar(g.direction) + g.q_grant_mvar)
            else:
                vh = g.q_grant_mvar
            by_day: Dict[int, float] = {}
            for w in range(g.window_first, g.window_end):
                if (g.area_id, w) not in obs_windows:
                    rep1("granted window has no metering observation — "
                         "settlement input incomplete",
                         area_id=g.area_id, window_index=w,
                         direction=g.direction)
                day = day_of_window[(g.area_id, w)]
                frac = cap_frac.get((id(g), w), 1.0)
                by_day[day] = min(by_day.get(day, 1.0), frac)
            for day, frac in sorted(by_day.items()):
                key = (g.area_id, g.direction, g.delivering_party, day)
                prev = emitted.get(key)
                if prev is not None:
                    # Sequential re-issued grants on the same day (plan
                    # §6: re-issue rather than renegotiate) accrue ONE
                    # day of Vorhaltung [LF §7.4]: the day's capacity is
                    # the largest held magnitude, suspension follows the
                    # worst window (documented interpretation).
                    vh_day = max(prev.vh_mvar, vh)
                    frac = min(prev.day_frac, frac)
                else:
                    vh_day = vh
                x = exceed.pop(key, 0.0) + (
                    0.0 if prev is None else prev.exceed_mvar)
                emitted[key] = DayCapacityRow(
                    area_id=g.area_id, direction=g.direction,
                    day_index=day, world=g.delivering_party,
                    vh_mvar=vh_day,
                    day_frac=frac,
                    pay_cap_avg_eur=(
                        vh_day * frac
                        * self.config.price_lp_avg_eur_per_mvar_day),
                    exceed_mvar=x,
                    pay_cap_grenz_eur=(
                        x * self.config.price_lp_grenz_eur_per_mvar_day),
                )
        rows.extend(emitted.values())
        # Remaining exceedances (no-grant cases 8.1-4 / 8.2-4b_adhoc):
        # Grenzpreis capacity without any Durchschnitt base.
        for (area_id, d, world, day), x in sorted(
                exceed.items(), key=lambda kv: (kv[0][0], kv[0][3])):
            rows.append(DayCapacityRow(
                area_id=area_id, direction=d, day_index=day, world=world,
                vh_mvar=0.0, day_frac=1.0, pay_cap_avg_eur=0.0,
                exceed_mvar=x,
                pay_cap_grenz_eur=(
                    x * self.config.price_lp_grenz_eur_per_mvar_day)))
        rows.sort(key=lambda r: (r.area_id, r.day_index,
                                 r.direction.value, r.world))
        return rows

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _band_of(self, area_id: str) -> NormalBand:
        band = self.bands.get(area_id)
        if band is None:
            rep1("no NormalBand configured for area", area_id=area_id)
        return band

    def _day_of(self, obs: WindowObservation) -> int:
        return int(math.floor(obs.t_start_s / _SECONDS_PER_DAY + 1e-9))

    @staticmethod
    def _active_grant(
        grants: Sequence[GrantRecord],
        direction: Direction,
        window_index: int,
    ) -> Optional[GrantRecord]:
        hits = [g for g in grants
                if g.direction is direction
                and g.window_first <= window_index < g.window_end]
        if len(hits) > 1:
            rep1("overlapping grants for one (area, direction, window)",
                 area_id=hits[0].area_id, direction=direction,
                 window_index=window_index)
        return hits[0] if hits else None

    def _assert_inputs(
        self,
        observations: Sequence[WindowObservation],
        grants: Sequence[GrantRecord],
        incapabilities: Sequence[IncapabilityRecord],
    ) -> None:
        if not observations:
            rep1("settlement needs at least one window observation")
        seen = set()
        for o in observations:
            key = (o.area_id, o.window_index)
            if key in seen:
                rep1("duplicate window observation", area_id=o.area_id,
                     window_index=o.window_index)
            seen.add(key)
        # Grants must not overlap per (area, direction) — the
        # GrantsLedger invariant (plan §3), asserted here as well.
        by_ad: Dict[Tuple[str, Direction], List[GrantRecord]] = {}
        for g in grants:
            by_ad.setdefault((g.area_id, g.direction), []).append(g)
        for (area_id, d), gs in by_ad.items():
            gs = sorted(gs, key=lambda g: g.window_first)
            for a, b in zip(gs, gs[1:]):
                if b.window_first < a.window_end:
                    rep1("overlapping grants", area_id=area_id,
                         direction=d, first=a.window_first,
                         second=b.window_first)
        for r in incapabilities:
            g = self._active_grant(
                [g for g in grants if g.area_id == r.area_id],
                r.direction, r.window_index)
            if g is None or g.delivering_party != TSO_DELIVERS:
                rep1("incapability record without a matching "
                     "TSO-delivers grant (Tabelle 8.1 case 3a)",
                     area_id=r.area_id, direction=r.direction,
                     window_index=r.window_index)
            if r.q_vh_provided_mvar > g.q_grant_mvar + _Q_EPS_MVAR:
                rep1("provided Vorhalteleistung exceeds the grant",
                     area_id=r.area_id,
                     q_vh_provided_mvar=r.q_vh_provided_mvar,
                     q_grant_mvar=g.q_grant_mvar)

    @staticmethod
    def _totalise(
        window_rows: Sequence[WindowSettlementRow],
        day_rows: Sequence[DayCapacityRow],
    ) -> List[SettlementTotals]:
        acc: Dict[Tuple[str, Direction, str], List[float]] = {}
        for r in window_rows:
            if r.direction is None:
                continue
            key = (r.area_id, r.direction, r.world)
            a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0])
            a[0] += r.pay_energy_avg_eur
            a[1] += r.pay_energy_grenz_eur
        for r in day_rows:
            key = (r.area_id, r.direction, r.world)
            a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0])
            a[2] += r.pay_cap_avg_eur
            a[3] += r.pay_cap_grenz_eur
        return [
            SettlementTotals(area_id=k[0], direction=k[1], world=k[2],
                             pay_energy_avg_eur=v[0],
                             pay_energy_grenz_eur=v[1],
                             pay_cap_avg_eur=v[2],
                             pay_cap_grenz_eur=v[3])
            for k, v in sorted(acc.items(),
                               key=lambda kv: (kv[0][0], kv[0][1].value,
                                               kv[0][2]))
        ]


# ----------------------------------------------------------------------
#  CSV output (schema in the module docstring)
# ----------------------------------------------------------------------

def write_settlement_csv(result: SettlementResult, path_prefix: str) -> None:
    """Write ``<prefix>_windows.csv``, ``<prefix>_days.csv`` and
    ``<prefix>_totals.csv`` (tidy, comma-separated, one header row)."""
    with open(f"{path_prefix}_windows.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area_id", "window_index", "t_start_s", "direction",
                    "world", "case", "q_meas_mvar", "q_set_mvar",
                    "e_avg_mvarh", "e_grenz_mvarh", "pay_energy_avg_eur",
                    "pay_energy_grenz_eur", "cap_frac", "cap_exceed_mvar",
                    "payer"])
        for r in result.window_rows:
            w.writerow([
                r.area_id, r.window_index, repr(r.t_start_s),
                "" if r.direction is None else r.direction.value,
                "" if r.world is None else r.world, r.case,
                repr(r.q_meas_mvar),
                "" if r.q_set_mvar is None else repr(r.q_set_mvar),
                repr(r.e_avg_mvarh), repr(r.e_grenz_mvarh),
                repr(r.pay_energy_avg_eur), repr(r.pay_energy_grenz_eur),
                repr(r.cap_frac), repr(r.cap_exceed_mvar),
                "" if r.world is None else payer_of(r.world)])
    with open(f"{path_prefix}_days.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area_id", "direction", "day_index", "world",
                    "vh_mvar", "day_frac", "pay_cap_avg_eur",
                    "exceed_mvar", "pay_cap_grenz_eur", "payer"])
        for r in result.day_rows:
            w.writerow([r.area_id, r.direction.value, r.day_index,
                        r.world, repr(r.vh_mvar), repr(r.day_frac),
                        repr(r.pay_cap_avg_eur), repr(r.exceed_mvar),
                        repr(r.pay_cap_grenz_eur), payer_of(r.world)])
    with open(f"{path_prefix}_totals.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area_id", "direction", "world", "payer",
                    "pay_energy_avg_eur", "pay_energy_grenz_eur",
                    "pay_cap_avg_eur", "pay_cap_grenz_eur",
                    "pay_total_eur"])
        for r in result.totals:
            w.writerow([r.area_id, r.direction.value, r.world,
                        payer_of(r.world), repr(r.pay_energy_avg_eur),
                        repr(r.pay_energy_grenz_eur),
                        repr(r.pay_cap_avg_eur),
                        repr(r.pay_cap_grenz_eur),
                        repr(r.pay_total_eur)])
