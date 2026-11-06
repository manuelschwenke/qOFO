"""
experiments/ch_11_case2_multisystem/ref/scenarios.py
====================================================

Candidate disturbance schedules for the reference run of Case 2
(``ch:case3:nominal``) on the reference period (5 Jan 2016 from 08:00).

**Scenario A is the reference scenario (author decision 2026-10-27)**, chosen
after the screening of A/B/C with L2 and O2 (daily log 2026-10-27e): G4 out
(area 3), 250 Mvar connected at bus 12 (area 2), tie line 17-27 out (areas 1/3)
-- every area hit by one event; with L2 area 2 leaves [0.95, 1.05] pu 67 % of
the time.  It replaces the six events of ``tab:setup:reference:events``
(the author asked for a challenging but not unrealistic scenario: one generator
outage, one line trip and one load step-in).  B and C are kept as the screened
alternatives.  Each candidate places every event strictly
between two transmission dispatches (author decision 2026-10-27), at different
offsets, so the sub-transmission layer acts on the old set-point until the
next dispatch.

Element indices (pandapower, from ``run_mc.enumerate_elements`` and the fixed
zone partition): machines G4 = 2 (area 3), G3 = 1 (area 2); lines 25 = 17-27
(tie, areas 1/3), 7 = 4-14 (area 2), 14 = 9-39 (tie, areas 2/1, to the G1
equivalent); load buses 11 = IEEE 12, 7 = IEEE 8,
3 = IEEE 4 (area 2).

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from experiments.helpers.records import ContingencyEvent

T_TS = 180.0
#: Simulation step (``dt_s``).  The runner fires an event only at a step whose
#: time equals the event time (``multi_tso_dso``), so an event off this grid
#: never fires -- F's second step at 271:10 did not (runs 0020/0021).
DT_S = 20.0


def _at(minute: int, offset_s: float) -> float:
    t = 60.0 * minute + float(offset_s)
    phase = t % T_TS
    if not 0.0 < phase < T_TS:
        raise ValueError(f"event at {t} s lies on a dispatch instant")
    if t % DT_S != 0.0:
        raise ValueError(f"event at {t} s lies off the {DT_S:g}-s step grid and would never fire")
    return t


def _gen(i: int, t: float, action: str = "trip") -> ContingencyEvent:
    return ContingencyEvent(minute=0, time_s=t, element_type="gen",
                            element_index=i, action=action)


def _line(i: int, t: float, action: str = "trip") -> ContingencyEvent:
    return ContingencyEvent(minute=0, time_s=t, element_type="line",
                            element_index=i, action=action)


def _load(bus: int, q: float, t: float, action: str = "connect") -> ContingencyEvent:
    return ContingencyEvent(minute=0, time_s=t, element_type="load", bus=bus,
                            p_mw=0.0, q_mvar=q, action=action)


#: Candidate schedules; one generator outage, one line trip, one load step-in
#: each, staggered by 90 min, persisting to the end of the horizon.
SCENARIOS: Dict[str, Dict[str, object]] = {
    "A": dict(
        title="REFERENCE: G4 out (area 3), 250 Mvar at bus 12 (area 2), tie 17-27 out",
        events=lambda: [_gen(2, _at(30, 40)), _load(11, 250.0, _at(120, 100)),
                        _line(25, _at(210, 140))]),
    "B": dict(
        title="area-2 stress: G3 out, line 4-14 out, 250 Mvar at bus 8",
        events=lambda: [_gen(1, _at(30, 40)), _line(7, _at(120, 100)),
                        _load(7, 250.0, _at(210, 140))]),
    # Declared 2026-10-30 before any run (docs/analysis/
    # 2026-10-30_scenarios_B_D_declaration.md): B with the load moved by the
    # location rule to IEEE 4 (load bus 3) -- no MSC, no TS tap at the bus,
    # area 2's machine out.  Reported together with B, whatever the outcome.
    "D": dict(
        title="area-2 Q-scarce, rule bus: G3 out, line 4-14 out, 250 Mvar at bus 4",
        events=lambda: [_gen(1, _at(30, 40)), _line(7, _at(120, 100)),
                        _load(3, 250.0, _at(210, 140))]),
    # Declared 2026-10-30 before any run (docs/analysis/
    # 2026-10-30_scenario_E_declaration.md): D with the tie to G1 (line 9-39)
    # instead of 4-14, and a deficit beyond area 2's own TS resources -- with
    # their raising limits the area-2 level stays 27 mpu below 1.03 pu (plant
    # bound), restorable with 13 % of the STS DER capability.
    "E": dict(
        title="area-2 Q-scarce beyond own TS resources: G3 out, line 9-39 out, "
              "600 Mvar at bus 4",
        events=lambda: [_gen(1, _at(30, 40)), _line(14, _at(120, 100)),
                        _load(3, 600.0, _at(210, 140))]),
    # Declared 2026-11-05 before any run (docs/analysis/
    # 2026-11-05_scenario_F_declaration.md; author: "two subsequent 400 Mvar
    # steps", "a scenario where we see a huge difference and really need sts
    # resources"): E's first two events, then two 400 Mvar steps -- IEEE 4 (the
    # location rule) and IEEE 8 (B's load bus) -- 59 min apart.  After the first
    # step area 2's own TS resources still restore the level (bound +12 mpu);
    # after the second they leave it 90 mpu low, and restoring it takes ~55 % of
    # the area-2 STS DER capability.  Two load buses, because connect events with
    # equal (bus, P, Q) share one dormant load (contingency.py).  FM: the same with
    # 2 x 350 Mvar, declared as F's fallback (~41 %).  Step 2 moved from 271:10
    # (off the 20-s step grid, never fired in runs 0020/0021) to 271:00
    # (amendment 2026-11-06 in the declaration).
    "F": dict(
        title="area-2 deficit far beyond own TS resources: G3 out, line 9-39 out, "
              "400 Mvar at bus 4, then 400 Mvar at bus 8",
        events=lambda: [_gen(1, _at(30, 40)), _line(14, _at(120, 100)),
                        _load(3, 400.0, _at(210, 140)), _load(7, 400.0, _at(270, 60))]),
    "FM": dict(
        title="F's fallback: G3 out, line 9-39 out, 350 Mvar at bus 4, then 350 Mvar at bus 8",
        events=lambda: [_gen(1, _at(30, 40)), _line(14, _at(120, 100)),
                        _load(3, 350.0, _at(210, 140)), _load(7, 350.0, _at(270, 60))]),
    # Declared 2026-11-06 before any run (docs/analysis/2026-11-06_scenario_E2_
    # declaration.md; author: "with 9-39 in operation, and maybe G3 and load step
    # overlapping, then G3 coming back, then a line drop later"): E without the
    # 9-39 opening, which raised the controllers' mismatch loop gain most (CEN
    # 3.1 at its former cap), and at most two disturbances at a time.  The line
    # (3-4, line 4) is the area-2 line or tie other than 9-39 whose opening leaves
    # the largest area-2 deficit at the TS-a2 bound, with every controller's
    # mismatch loop gain < 2 and the frozen-control minimum >= 0.90 pu.
    "E2": dict(
        title="area-2 Q-scarce without 9-39: G3 out, 600 Mvar at bus 4, G3 back, "
              "line 3-4 out",
        events=lambda: [_gen(1, _at(30, 40)), _load(3, 600.0, _at(90, 100)),
                        _gen(1, _at(150, 140), "restore"), _line(4, _at(240, 60))]),
    # Renewable-dominated period under unit commitment (author decision
    # 2026-10-31): the earliest 6-h window of the profile year, outside the
    # excluded days, starting at a full hour between 06:00 and 12:00, in which
    # a single synchronous machine is committed (G10, hydro; G1 equivalent
    # aside).  A's and E's schedules on that period; an outage of a machine
    # that is not committed changes nothing.
    "RA": dict(
        title="renewable-dominated (one machine committed), A's events",
        start="2016-03-13T09:00:00",
        events=lambda: [_gen(2, _at(30, 40)), _load(11, 250.0, _at(120, 100)),
                        _line(25, _at(210, 140))]),
    "RE": dict(
        title="renewable-dominated (one machine committed), E's events",
        start="2016-03-13T09:00:00",
        events=lambda: [_gen(1, _at(30, 40)), _line(14, _at(120, 100)),
                        _load(3, 600.0, _at(210, 140))]),
    # Period with G3 offline (author decision 2026-10-31; rule declared before
    # the search in docs/analysis/2026-10-31_ref_period_G3_offline_declaration.md):
    # the earliest 6-h window outside the excluded days, starting at a full hour
    # between 06:00 and 12:00, with G3 not committed and >= 3 machines
    # committed, that is operable under LOC without events and whose post-event
    # states of A's and E's schedules are solvable.  Selected: 08 Jan 06:00
    # (G10, G9, G4, G7 committed); E's G3 outage is void there.
    "GA": dict(
        title="G3 offline (area 2 without a synchronous machine), A's events",
        start="2016-01-08T06:00:00",
        events=lambda: [_gen(2, _at(30, 40)), _load(11, 250.0, _at(120, 100)),
                        _line(25, _at(210, 140))]),
    "GE": dict(
        title="G3 offline (area 2 without a synchronous machine), E's events",
        start="2016-01-08T06:00:00",
        events=lambda: [_gen(1, _at(30, 40)), _line(14, _at(120, 100)),
                        _load(3, 600.0, _at(210, 140))]),
    "C": dict(
        title="G3 out (area 2), 250 Mvar at bus 12 (area 2), tie 17-27 out",
        events=lambda: [_gen(1, _at(30, 40)), _load(11, 250.0, _at(120, 100)),
                        _line(25, _at(210, 140))]),
    # The six events of tab:setup:reference:events, on the dispatch grid as
    # written in the thesis; kept for comparison only.
    "thesis6": dict(
        title="six thesis events on the dispatch grid (reference only)",
        events=lambda: [
            ContingencyEvent(minute=30, element_type="gen", element_index=2, action="trip"),
            ContingencyEvent(minute=180, element_type="gen", element_index=2, action="restore"),
            _load(11, 250.0, 120 * 60.0), _load(11, 250.0, 330 * 60.0, "trip"),
            ContingencyEvent(minute=210, element_type="line", element_index=25, action="trip"),
            ContingencyEvent(minute=300, element_type="line", element_index=25, action="restore"),
        ]),
}


#: The reference scenario of ch:case3:nominal (author decision 2026-10-27).
REFERENCE = "A"


def events(name: str) -> List[ContingencyEvent]:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; valid: {sorted(SCENARIOS)}")
    return list(SCENARIOS[name]["events"]())


def start_time(name: str) -> Optional[datetime]:
    """Start of the scenario's period, or None for the reference period."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; valid: {sorted(SCENARIOS)}")
    s = SCENARIOS[name].get("start")
    return None if s is None else datetime.fromisoformat(str(s))


def describe(name: str) -> List[str]:
    """Human-readable schedule for README and report."""
    out = []
    for ev in sorted(events(name), key=lambda e: e.effective_time_s):
        t = ev.effective_time_s
        what = (f"{ev.element_type} {ev.element_index}" if ev.element_index >= 0
                else f"load {ev.q_mvar:g} Mvar at bus {ev.bus}")
        out.append(f"{int(t // 60)}:{int(t % 60):02d} {ev.action} {what} "
                   f"(phase {t % T_TS:.0f} s)")
    return out
