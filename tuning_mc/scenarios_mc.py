"""
tuning_mc/scenarios_mc.py
=========================
Scenario set for the Monte-Carlo tuning campaign.

Why not reuse ``tuning.scenarios.tune_set_v2``
----------------------------------------------
Copying the BO set would be the cheap option and it is the wrong one, for
three reasons that are specific and checkable:

1. **Wrong network mix.**  ``tune_v2`` is 3/4 ``base_410``; only
   ``v2_overvoltage_rural`` runs ``rural_700``.  The campaign is specified on
   ``rural_700``, and the design is per-plant: ``H``, and therefore every
   weight Stage 0 derives, changes with the network.
2. **Wrong window length.**  75 min, where 90 min is specified.  The window
   also sets the resolution of the wear metric: taps are integers, so
   ``ops/h`` is quantised at one tap per window -- 0.804/h at 75 min, 0.667/h
   at 90 min.  Neither resolves a 30-taps-per-*day* budget, which is why the
   wear scenario below is 12 h.
3. **Documented excitation gaps.**  Measured on ``tune_v2`` during the
   2026-08-13 study:

   * **zero** TSO tap reversals in every scenario, so the hunting constraint
     ``g5b`` is vacuous on the set it is calibrated against (the holdout
     produced 1.607-2.411/h);
   * the MSC/MSR banks never commit on ``v2_overvoltage_rural``, so the shunt
     coordinate carried no signal (Spearman rho = +0.064, p = 0.44);
   * ``v2_undervoltage_ramp`` starts at a winter evening where PV-based TS-DER
     have exactly zero reactive capability, making ``tau`` structurally inert
     there.

Design principles
-----------------
**Excitation before coverage.**  A weight can only be identified from a
scenario in which its actuator's activity actually varies.  Each entry below
exists to make one search direction identifiable, and the set is stated as a
season x disturbance-class design rather than a list of hand-picked cases:

===========================  ==============================================
scenario                     the direction it exists to excite
===========================  ==============================================
``mc_quiet_summer_night``    none -- the quiescent control.  Keeps the search
                             from buying stress performance with chatter at
                             rest.
``mc_gen_trip_spring``       continuous loop gain (``lambda_tso``): an
                             impulsive disturbance the continuous actuators
                             can absorb, at spring noon where DER reactive
                             capability is full so ``tau`` is live.
``mc_undervolt_ramp_winter`` tap engage thresholds: a one-way ramp exhausts
                             continuous reactive range and hands authority to
                             the tap changers.  ``tau`` is inert here (PV at
                             zero) -- deliberately kept, because it is the
                             case where TS voltage is genuinely stressed.
``mc_overvolt_ramp_summer``  the same, opposite sign.  Without it the tuning
                             is biased toward one tap direction.
``mc_reversal_spring``       hunting: the error changes sign mid-window, so
                             reversals actually occur and ``g5b`` / the
                             hunting criterion stop being vacuous.
===========================  ==============================================

**Tune/holdout disjointness** follows the existing calendar convention: tune
dates sit in odd ISO weeks, holdout dates in even ones, so no week is shared.

**The wear budget is measured separately.**  ``wear_day_set`` is a single 12-h
run with real profiles and no injected contingencies -- the only construction
in which "30 tap operations per day" is a meaningful measurement.  Event-dense
90-min windows over-state a real day by roughly an order of magnitude and must
not be extrapolated to it.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from tuning.scenarios import ContingencyEvent, ScenarioSpec, _load_ramp

__all__ = ["tune_set_mc", "holdout_set_mc", "wear_day_set", "MC_NETWORK"]

#: The campaign runs on one network, per the Stage-1 specification.
MC_NETWORK = "rural_700"

STABILISE_MIN = 15
EVENT_MIN = 75
TOTAL_MIN = STABILISE_MIN + EVENT_MIN          # 90 min, as specified
_S = STABILISE_MIN

# Tune dates -- odd ISO weeks.
_T_SUMMER_NIGHT = datetime(2016, 7, 10, 3, 0)    # ISO wk 27, min load, high DER
_T_SUMMER_NOON = datetime(2016, 7, 10, 12, 0)    # ISO wk 27, PV peak
_T_SPRING_NOON = datetime(2016, 4, 15, 12, 0)    # ISO wk 15, mid load
_T_WINTER_EVE = datetime(2016, 1, 21, 18, 0)     # ISO wk 3, peak, PV at zero

# Holdout dates -- even ISO weeks, disjoint by construction.
_H_SUMMER_NIGHT = datetime(2016, 7, 17, 3, 0)    # ISO wk 28
_H_SPRING_NOON = datetime(2016, 4, 22, 12, 0)    # ISO wk 16
_H_WINTER_EVE = datetime(2016, 1, 28, 18, 0)     # ISO wk 4


def _spec(name: str, start: datetime, contingencies=()) -> ScenarioSpec:
    return ScenarioSpec(
        name=name, start_time=start, duration_s=TOTAL_MIN * 60,
        scenario=MC_NETWORK, contingencies=tuple(contingencies),
    )


def tune_set_mc() -> List[ScenarioSpec]:
    """Five 90-min ``rural_700`` scenarios; one per excitation requirement."""
    return [
        _spec("mc_quiet_summer_night", _T_SUMMER_NIGHT),
        _spec("mc_gen_trip_spring", _T_SPRING_NOON, (
            ContingencyEvent(minute=_S + 5, element_type="gen",
                             element_index=2, action="trip"),
            ContingencyEvent(minute=_S + 45, element_type="gen",
                             element_index=2, action="restore"),
        )),
        _spec("mc_undervolt_ramp_winter", _T_WINTER_EVE, tuple(
            _load_ramp(bus=5, start_min=_S + 5, n_steps=4, step_min=10,
                       p_mw=150.0, q_mvar=75.0)
            + _load_ramp(bus=7, start_min=_S + 10, n_steps=3, step_min=10,
                         p_mw=100.0, q_mvar=50.0)
        )),
        _spec("mc_overvolt_ramp_summer", _T_SUMMER_NOON, tuple(
            _load_ramp(bus=27, start_min=_S + 5, n_steps=3, step_min=12,
                       p_mw=-120.0, q_mvar=-60.0)
        )),
        # Sign-reversing: load on, then the same load off again 30 min later,
        # so the controlled error crosses zero and a tap that moved up has a
        # reason to move back down.  This is the only construction in the set
        # that can produce a reversal, and hence the only one under which a
        # hunting criterion carries information.
        _spec("mc_reversal_spring", _T_SPRING_NOON, tuple(
            _load_ramp(bus=5, start_min=_S + 5, n_steps=3, step_min=8,
                       p_mw=140.0, q_mvar=70.0)
            + [ContingencyEvent(minute=_S + 40 + 8 * k, element_type="load",
                                bus=5, p_mw=140.0, q_mvar=70.0, action="trip")
               for k in range(3)]
        )),
    ]


def holdout_set_mc() -> List[ScenarioSpec]:
    """Disjoint confirmation set (even ISO weeks). Never tuned on."""
    return [
        _spec("ho_quiet_summer_night", _H_SUMMER_NIGHT),
        _spec("ho_gen_trip_spring", _H_SPRING_NOON, (
            ContingencyEvent(minute=_S + 5, element_type="gen",
                             element_index=2, action="trip"),
            ContingencyEvent(minute=_S + 45, element_type="gen",
                             element_index=2, action="restore"),
        )),
        _spec("ho_undervolt_ramp_winter", _H_WINTER_EVE, tuple(
            _load_ramp(bus=5, start_min=_S + 5, n_steps=4, step_min=10,
                       p_mw=150.0, q_mvar=75.0)
        )),
    ]


def wear_day_set() -> List[ScenarioSpec]:
    """One 12-h profile-driven run, no injected events.

    The only construction in which "tap operations per day" is measurable:
    a real day is mostly quiet, so a rate read off an event-dense 90-min
    window over-states it by roughly an order of magnitude.  Used to check the
    switching budget of a *finished* candidate, never inside the search loop.
    """
    return [
        ScenarioSpec(
            name="mc_wear_day", start_time=_T_SUMMER_NIGHT,
            duration_s=12 * 3600.0, scenario=MC_NETWORK, contingencies=(),
        ),
    ]
