"""
experiments/ch_11_case2_multisystem/ref/scenarios.py
====================================================

Candidate disturbance schedules for the reference run of Case 2
(``ch:case3:nominal``) on the reference period (5 Jan 2016 from 08:00).

The schedule is NOT settled: the author asked (2026-10-27) for a challenging
but not unrealistic scenario -- one generator outage, one line trip and one load
step-in -- to be chosen after a screening, instead of the six events of
``tab:setup:reference:events``.  Each candidate places every event strictly
between two transmission dispatches (author decision 2026-10-27), at different
offsets, so the sub-transmission layer acts on the old set-point until the
next dispatch.

Element indices (pandapower, from ``run_mc.enumerate_elements`` and the fixed
zone partition): machines G4 = 2 (area 3), G3 = 1 (area 2); lines 25 = 17-27
(tie, areas 1/3), 7 = 4-14 (area 2); load buses 11 = IEEE 12, 7 = IEEE 8
(area 2).

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

from typing import Dict, List

from experiments.helpers.records import ContingencyEvent

T_TS = 180.0


def _at(minute: int, offset_s: float) -> float:
    t = 60.0 * minute + float(offset_s)
    phase = t % T_TS
    if not 0.0 < phase < T_TS:
        raise ValueError(f"event at {t} s lies on a dispatch instant")
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
        title="G4 out (area 3), 250 Mvar at bus 12 (area 2), tie 17-27 out",
        events=lambda: [_gen(2, _at(30, 40)), _load(11, 250.0, _at(120, 100)),
                        _line(25, _at(210, 140))]),
    "B": dict(
        title="area-2 stress: G3 out, line 4-14 out, 250 Mvar at bus 8",
        events=lambda: [_gen(1, _at(30, 40)), _line(7, _at(120, 100)),
                        _load(7, 250.0, _at(210, 140))]),
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


def events(name: str) -> List[ContingencyEvent]:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; valid: {sorted(SCENARIOS)}")
    return list(SCENARIOS[name]["events"]())


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
