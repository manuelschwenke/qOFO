"""
tuning_mc/scenarios_mc_v2.py
============================
Scenario banks for the 0815 campaign.  Replaces :mod:`tuning_mc.scenarios_mc`,
which is kept unchanged so the 0814 results stay reproducible.

What changed, and why
---------------------
The 0814 bank was five hand-picked 90-min windows.  Four defects were measured
on it, not suspected:

1. **One window owned the objective.**  ``mc_undervolt_ramp_winter``
   contributed ~60 % of the aggregate ``f_ts``, so "the mean over five windows"
   was in practice one window.
2. **That window has exactly zero DER reactive capability**, the stratum in
   which ``tau``, ``lambda_dso`` and ``dso_g_v_ratio`` are structurally inert
   (VDE-AR-N-4120-v2 gives *zero* Q below P/Sn = 0.1).  The dominant window was
   the one carrying no signal about half the search space.
3. **The bank was easier than the holdout** -- ``f_ts`` ~2.04 in sample against
   ~3.95 out of sample -- so only candidate-to-candidate comparisons
   transferred, and the interface-tracking gain did not transfer at all.
4. **Wear and hunting are not measurable at 90 min.**  Taps are integers: one
   tap in a 90-min window is 0.667 ops/h, so a limit of 1.2054/h falls *between*
   the one- and two-reversal levels and the constraint is effectively binary.

Three banks, two tiers
----------------------
**Tier 1 -- design (12 x 90 min).**  The excitation-role design of the 0814 set
is kept (quiet control; impulsive generator outage; load ramp up; load ramp
down; sign-reversing) but each role is replicated across seasons, and the
operating point filling each role x season cell is chosen **from the Screen-1
capability measurement** by :mod:`tuning_mc.select_windows_v2` rather than by
hand.  The resulting capability mix tracks the profile year:

    stratum   bank            2016 profile year
    none      2/12 (17 %)     18.6 %
    partial   4/12 (33 %)     33.7 %
    full      6/12 (50 %)     47.7 %

so the zero-capability stratum is present -- it is a real 19 % of the year and
excluding it would be its own bias -- without dominating.  Every window records
its stratum in :data:`WINDOW_META`, so ``f_q`` can be reported per stratum
instead of only in aggregate.

**Tier 1 -- confirmation (6 x 90 min).**  Drawn from *the same cells* by *the
same rule*, in even ISO weeks.  The 0814 holdout was not comparable in
difficulty to its design bank; this one is, by construction.

**Tier 2 -- audit (4 x 12 h).**  Profile-driven, spanning seasons, two quiet and
two carrying a single realistic event.  This is where wear and hunting are
measured: at 12 h the quantisation step is 0.083 ops/h, so a limit of 1.25/h
(30 taps/day/transformer) sits 15 steps above zero and is properly resolved.
Subsumes the old ``wear_day_set``.  Placed in odd ISO weeks and on calendar
days disjoint from Tier 1, so the confirmation set is untouched by it.

Parity convention is unchanged: design and audit in odd ISO weeks, confirmation
in even ones.

Disturbances are designed, operating points are selected
--------------------------------------------------------
The events below are *injected by design* -- a generator trip does not occur in
a profile year, and its justification is the N-1 requirement, not a
measurement.  The operating points they are injected at are *selected from
data*.  Conflating the two is what made the first hand-picked set
indefensible; the event constructions are therefore carried over from
``scenarios_mc.py`` **unchanged**, so that season and capability are the only
things that differ between a 0814 window and its 0815 counterpart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List

from tuning.scenarios import ContingencyEvent, ScenarioSpec, _load_ramp

__all__ = [
    "MC_NETWORK",
    "WINDOW_META",
    "tier1_design_set",
    "tier1_confirm_set",
    "tier2_audit_set",
    "stratum_of",
]

#: One network, per the Stage-1 specification: the design is per-plant, and
#: ``H`` -- hence every weight Stage 0 derives -- changes with the network.
MC_NETWORK = "rural_700"

STABILISE_MIN = 15
EVENT_MIN = 75
TOTAL_MIN = STABILISE_MIN + EVENT_MIN          # 90 min, unchanged from 0814
_S = STABILISE_MIN

AUDIT_H = 12                                   # Tier-2 window length

# ---------------------------------------------------------------------------
# Windows, as selected by tuning_mc/select_windows_v2.py from
# results/tuning_mc/stage1a/screen1.json.  Regenerate with
#     python -m tuning_mc.select_windows_v2
# The provenance of every timestamp is in
# results/tuning_mc/campaign_0815/window_selection.json.
# ---------------------------------------------------------------------------

#: ``(role, season, start, stratum)``.  Odd ISO weeks.
DESIGN_WINDOWS: list[tuple[str, str, datetime, str]] = [
    ('quiet',     'summer', datetime(2016, 6, 6, 2, 0),   'none'),     # wk 23, Q 0
    ('quiet',     'spring', datetime(2016, 3, 3, 10, 0),  'full'),     # wk 9,  Q 3699
    ('gen_trip',  'spring', datetime(2016, 4, 16, 14, 0), 'full'),     # wk 15, Q 3699
    ('gen_trip',  'summer', datetime(2016, 8, 31, 14, 0), 'full'),     # wk 35, Q 3699
    ('gen_trip',  'winter', datetime(2016, 1, 4, 8, 0),   'partial'),  # wk 1,  Q 1261
    ('ramp_up',   'winter', datetime(2016, 2, 3, 20, 0),  'none'),     # wk 5,  Q 0
    ('ramp_up',   'autumn', datetime(2016, 11, 26, 10, 0), 'partial'), # wk 47, Q 710
    ('ramp_up',   'spring', datetime(2016, 5, 12, 10, 0), 'full'),     # wk 19, Q 3699
    ('ramp_down', 'summer', datetime(2016, 7, 18, 14, 0), 'full'),     # wk 29, Q 3699
    ('ramp_down', 'spring', datetime(2016, 4, 2, 10, 0),  'partial'),  # wk 13, Q 710
    ('reversal',  'spring', datetime(2016, 3, 19, 10, 0), 'full'),     # wk 11, Q 3699
    ('reversal',  'autumn', datetime(2016, 10, 1, 14, 0), 'partial'),  # wk 39, Q 710
]

#: Same cells, same rule, even ISO weeks.
#:
#: **Nine windows, at least two per stratum.**  The first version had six with
#: only one ``partial`` window, and measured 2026-08-15 that single window
#: carried **49 % of the confirmation aggregate** -- with n = 1 there is no
#: averaging, so one extreme draw becomes half the answer.  Extending the cell
#: list changes which windows the selector picks (the maximin spread now runs
#: over nine cells), so this bank is not a superset of the six-window one and
#: results on the two are not directly comparable.
CONFIRM_WINDOWS: list[tuple[str, str, datetime, str]] = [
    ('quiet',     'summer', datetime(2016, 6, 2, 4, 0),   'none'),     # wk 22, Q 0
    ('ramp_up',   'winter', datetime(2016, 1, 27, 18, 0), 'none'),     # wk 4,  Q 0
    ('gen_trip',  'winter', datetime(2016, 2, 23, 12, 0), 'partial'),  # wk 8,  Q 1179
    ('ramp_up',   'autumn', datetime(2016, 11, 20, 12, 0), 'partial'), # wk 46, Q 710
    ('reversal',  'autumn', datetime(2016, 9, 5, 10, 0),  'partial'),  # wk 36, Q 710
    ('gen_trip',  'spring', datetime(2016, 4, 10, 14, 0), 'full'),     # wk 14, Q 3699
    ('ramp_down', 'summer', datetime(2016, 7, 17, 14, 0), 'full'),     # wk 28, Q 3699
    ('reversal',  'spring', datetime(2016, 5, 4, 14, 0),  'full'),     # wk 18, Q 3699
    ('ramp_up',   'spring', datetime(2016, 3, 21, 12, 0), 'full'),     # wk 12, Q 3699
]

#: Tier 2.  Odd ISO weeks, calendar days disjoint from Tier 1.
AUDIT_WINDOWS: list[tuple[str, str, datetime, str]] = [
    ('quiet',    'winter', datetime(2016, 12, 25, 6, 0), 'partial'),   # wk 51, Q 2051
    ('quiet',    'summer', datetime(2016, 6, 26, 6, 0),  'full'),      # wk 25, Q 2988
    ('gen_trip', 'spring', datetime(2016, 4, 29, 6, 0),  'full'),      # wk 17, Q 2988
    ('ramp',     'autumn', datetime(2016, 10, 30, 6, 0), 'partial'),   # wk 43, Q 1833
]


# ---------------------------------------------------------------------------
# Event constructions -- carried over from scenarios_mc.py unchanged
# ---------------------------------------------------------------------------

def _events_quiet(_s: int) -> tuple:
    """Quiescent control.  Keeps the search from buying stress performance
    with chatter at rest."""
    return ()


def _events_gen_trip(s: int) -> tuple:
    """Impulsive disturbance the continuous actuators can absorb -- the
    excitation ``lambda_tso`` is identified from."""
    return (
        ContingencyEvent(minute=s + 5, element_type="gen",
                         element_index=2, action="trip"),
        ContingencyEvent(minute=s + 45, element_type="gen",
                         element_index=2, action="restore"),
    )


def _events_ramp_up(s: int) -> tuple:
    """One-way ramp: exhausts continuous reactive range and hands authority to
    the tap changers, which is what identifies the engage thresholds."""
    return tuple(
        _load_ramp(bus=5, start_min=s + 5, n_steps=4, step_min=10,
                   p_mw=150.0, q_mvar=75.0)
        + _load_ramp(bus=7, start_min=s + 10, n_steps=3, step_min=10,
                     p_mw=100.0, q_mvar=50.0)
    )


def _events_ramp_down(s: int) -> tuple:
    """The same, opposite sign.  Without it the tuning is biased toward one
    tap direction."""
    return tuple(
        _load_ramp(bus=27, start_min=s + 5, n_steps=3, step_min=12,
                   p_mw=-120.0, q_mvar=-60.0)
    )


def _events_reversal(s: int) -> tuple:
    """Load on, then the same load off 30 min later, so the controlled error
    crosses zero and a tap that moved up has a reason to move back down.  The
    only construction in the set that can produce a reversal."""
    return tuple(
        _load_ramp(bus=5, start_min=s + 5, n_steps=3, step_min=8,
                   p_mw=140.0, q_mvar=70.0)
        + [ContingencyEvent(minute=s + 40 + 8 * k, element_type="load",
                            bus=5, p_mw=140.0, q_mvar=70.0, action="trip")
           for k in range(3)]
    )


_ROLE_EVENTS = {
    "quiet": _events_quiet,
    "gen_trip": _events_gen_trip,
    "ramp_up": _events_ramp_up,
    "ramp_down": _events_ramp_down,
    "reversal": _events_reversal,
}

# ---------------------------------------------------------------------------
# Tier-2 events: ONE realistic disturbance in 12 h, not an event-dense window.
# The point of the audit is a wear rate that may legitimately be extrapolated
# to a day, so the window must look like a day: mostly profile, one event.
# ---------------------------------------------------------------------------

def _audit_events(role: str) -> tuple:
    h = 60                                       # minutes per hour, for reading
    if role == "quiet":
        return ()
    if role == "gen_trip":
        # One N-1 outage, cleared after an hour.
        return (
            ContingencyEvent(minute=4 * h, element_type="gen",
                             element_index=2, action="trip"),
            ContingencyEvent(minute=5 * h, element_type="gen",
                             element_index=2, action="restore"),
        )
    if role == "ramp":
        # One sustained load build-up over the evening ramp, no restore.
        return tuple(_load_ramp(bus=5, start_min=7 * h, n_steps=4,
                                step_min=20, p_mw=120.0, q_mvar=60.0))
    raise KeyError(role)


# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------

#: ``scenario name -> {role, season, stratum, tier, iso_week}``.
#:
#: Not decoration: ``tau``, ``lambda_dso`` and ``dso_g_v_ratio`` are
#: *structurally* inert in the ``none`` stratum, so an aggregate ``f_q`` that
#: mixes strata averages a signal with a constant.  Reporting per stratum is
#: the only way that aggregate can be read.
WINDOW_META: dict[str, dict[str, Any]] = {}


def _register(name: str, role: str, season: str, stratum: str, tier: str,
              start: datetime) -> None:
    WINDOW_META[name] = {"role": role, "season": season, "stratum": stratum,
                         "tier": tier, "iso_week": start.isocalendar()[1],
                         "start": start.isoformat()}


def _spec(name: str, start: datetime, contingencies=(),
          duration_s: float = TOTAL_MIN * 60) -> ScenarioSpec:
    return ScenarioSpec(
        name=name, start_time=start, duration_s=duration_s,
        scenario=MC_NETWORK, contingencies=tuple(contingencies),
    )


def _build(windows, tier: str, prefix: str) -> List[ScenarioSpec]:
    out = []
    for role, season, start, stratum in windows:
        name = f"{prefix}_{role}_{season}"
        _register(name, role, season, stratum, tier, start)
        out.append(_spec(name, start, _ROLE_EVENTS[role](_S)))
    return out


def tier1_design_set() -> List[ScenarioSpec]:
    """12 x 90 min, ``rural_700``, odd ISO weeks.  The search bank."""
    return _build(DESIGN_WINDOWS, "tier1_design", "d")


def tier1_confirm_set() -> List[ScenarioSpec]:
    """6 x 90 min, same cells and rule, even ISO weeks.  Never tuned on."""
    return _build(CONFIRM_WINDOWS, "tier1_confirm", "c")


def tier2_audit_set() -> List[ScenarioSpec]:
    """4 x 12 h, profile-driven, two quiet and two with one event.

    **The authoritative wear and hunting measurement.**  A rate read off a
    90-min event-dense window may not be extrapolated to a day: the 0814
    campaign's ``tap_ops_per_h`` of 6.03 would be 145 taps/day against a budget
    of 30, and that arithmetic is exactly the mistake this tier removes.
    """
    out = []
    for role, season, start, stratum in AUDIT_WINDOWS:
        name = f"a_{role}_{season}"
        _register(name, role, season, stratum, "tier2_audit", start)
        out.append(_spec(name, start, _audit_events(role),
                         duration_s=AUDIT_H * 3600.0))
    return out


def stratum_of(scenario_name: str) -> str:
    """Capability stratum of a window, or ``'unknown'`` for a foreign name."""
    return WINDOW_META.get(scenario_name, {}).get("stratum", "unknown")


# Populate WINDOW_META at import time so consumers can read strata without
# having to build the specs first.
for _f in (tier1_design_set, tier1_confirm_set, tier2_audit_set):
    _f()
