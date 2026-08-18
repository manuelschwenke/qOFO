"""
tuning/scenarios.py
===================
Declarative scenario specifications for offline controller tuning.

A :class:`ScenarioSpec` defines the time-series operating point under
which a candidate controller configuration is evaluated.  It is
overlaid onto a baseline :class:`MultiTSOConfig` at runtime — no
controller weights are touched.

Two scenario sources:

* :func:`design_set` — five deterministic, named scenarios used as the
  BO inner loop.  Designed to span the operating envelope (quiet, gen
  trip, load step, dual disturbance, off-peak/winter) at minimum
  simulated cost.
* :func:`validation_set` — randomised scenarios for evaluating the
  tuned controller AFTER BO converges.  Reproducible via the seed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np

from configs.config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent
from network.ieee39.scenarios import SCENARIO_REGISTRY

#: Network scenarios a :class:`ScenarioSpec` may name.  ``"wind_replace"`` is
#: accepted by ``build_ieee39_net`` only as a deprecated alias for
#: ``"base_410"`` and is deliberately excluded here so specs name the current
#: identifier directly.
VALID_NETWORK_SCENARIOS: frozenset[str] = frozenset(SCENARIO_REGISTRY)


@dataclass(frozen=True)
class ScenarioSpec:
    """Specification of one simulation scenario.

    Attributes overlay the corresponding fields on a baseline
    :class:`MultiTSOConfig` via :meth:`overlay_on`.  All other config
    fields (controller weights, stability flags, output paths, etc.)
    are preserved.
    """

    name: str
    start_time: datetime
    duration_s: float
    contingencies: tuple[ContingencyEvent, ...] = ()
    scenario: str = "base_410"
    use_profiles: bool = True
    tso_period_s: float = 180.0
    dso_period_s: float = 20.0
    dt_s: float = 20.0
    """Plant step [s].

    Previously **not** part of the spec, so it silently inherited the baseline
    (60 s) while ``dso_period_s`` was set to 10 s.  Because the runner tests
    ``time_s % period_s < 1``, that made the DSO fire on *every* plant step —
    one DSO step per TSO step, not the 18 the spec implied.  The cascade
    therefore ran with no timescale separation at all, which is the premise of
    the whole hierarchy.  Making ``dt_s`` explicit closes that gap; at
    ``dt_s=20`` with ``tso_period_s=180`` the ratio is the intended 9:1."""

    def overlay_on(self, cfg: MultiTSOConfig) -> MultiTSOConfig:
        """Return a new :class:`MultiTSOConfig` with this scenario's
        time-series fields applied."""
        return dataclasses.replace(
            cfg,
            n_total_s=float(self.duration_s),
            start_time=self.start_time,
            dt_s=float(self.dt_s),
            tso_period_s=float(self.tso_period_s),
            dso_period_s=float(self.dso_period_s),
            scenario=self.scenario,
            use_profiles=self.use_profiles,
            contingencies=list(self.contingencies),
        )

    @property
    def event_times_s(self) -> tuple[float, ...]:
        """Disturbance onsets [s], for event-anchored settling analysis.

        A settling time is only meaningful relative to a disturbance, so the
        analysis windows are anchored here rather than on a fixed grid — a
        uniform grid mostly measures quiet intervals in which nothing stepped.
        Deduplicated and sorted; empty for a quiescent scenario, which is
        correct (there is nothing to settle from).
        """
        return tuple(sorted({
            float(ev.effective_time_s) for ev in self.contingencies
        }))

    def __post_init__(self) -> None:
        # Fail loudly on an unknown network scenario.  ``validation_set`` used
        # to emit ``"base"`` (not in the registry) for ~20 % of its draws;
        # ``build_ieee39_net`` raised, ``run_one`` swallowed it into a sentinel
        # cost, and a fifth of every validation campaign was silently scored as
        # a power-flow failure.
        if self.scenario not in VALID_NETWORK_SCENARIOS:
            raise ValueError(
                f"Unknown network scenario {self.scenario!r} for "
                f"ScenarioSpec {self.name!r}; valid: "
                f"{sorted(VALID_NETWORK_SCENARIOS)}"
            )
        if self.dso_period_s < self.dt_s:
            raise ValueError(
                f"ScenarioSpec {self.name!r}: dso_period_s="
                f"{self.dso_period_s} is below dt_s={self.dt_s}, so the DSO "
                f"would fire every plant step and the stated period would be "
                f"fiction.  Set dso_period_s >= dt_s."
            )


# ---------------------------------------------------------------------------
# Design set: 5 deterministic scenarios
# ---------------------------------------------------------------------------
# All scenarios are 75 min: 15 min of *stabilisation* (no events) so the
# controller settles the operating point before any disturbance, then
# 60 min of *event window* during which contingencies fire.  Equal
# duration across scenarios eliminates the T^2 ITAE bias (same physical
# tracking error → same metric contribution regardless of scenario).

_STABILISE_MIN = 15
_EVENT_WINDOW_MIN = 60
_TOTAL_MIN = _STABILISE_MIN + _EVENT_WINDOW_MIN     # 75 min

_T0 = datetime(2016, 4, 15, 12, 0)         # spring noon, mid-load
_T_WINTER = datetime(2016, 1, 14, 18, 0)   # winter evening, peak


def design_set() -> List[ScenarioSpec]:
    """Five named, deterministic scenarios spanning the operating
    envelope.  Each is 75 min total (15 min stabilisation + 60 min
    event window)."""
    return [
        ScenarioSpec(
            name="nominal_quiet",
            start_time=_T0,
            duration_s=_TOTAL_MIN * 60,
            contingencies=(),
        ),
        ScenarioSpec(
            name="gen_trip_recovery",
            start_time=_T0,
            duration_s=_TOTAL_MIN * 60,
            contingencies=(
                ContingencyEvent(
                    # Trip 5 min into the event window.
                    minute=_STABILISE_MIN + 5, element_type="gen",
                    element_index=2, action="trip",
                ),
                ContingencyEvent(
                    # Restore 35 min later, leaves 20 min recovery.
                    minute=_STABILISE_MIN + 40, element_type="gen",
                    element_index=2, action="restore",
                ),
            ),
        ),
        ScenarioSpec(
            name="load_step",
            start_time=_T0,
            duration_s=_TOTAL_MIN * 60,
            contingencies=(
                ContingencyEvent(
                    minute=_STABILISE_MIN + 5, element_type="load",
                    bus=5, p_mw=300.0, q_mvar=150.0,
                    action="connect",
                ),
                ContingencyEvent(
                    minute=_STABILISE_MIN + 45, element_type="load",
                    bus=5, p_mw=300.0, q_mvar=150.0,
                    action="trip",
                ),
            ),
        ),
        ScenarioSpec(
            name="dual_disturbance",
            start_time=_T0,
            duration_s=_TOTAL_MIN * 60,
            contingencies=(
                ContingencyEvent(
                    minute=_STABILISE_MIN + 5, element_type="gen",
                    element_index=5, action="trip",
                ),
                ContingencyEvent(
                    minute=_STABILISE_MIN + 15, element_type="load",
                    bus=2, p_mw=200.0, q_mvar=100.0,
                    action="connect",
                ),
                ContingencyEvent(
                    minute=_STABILISE_MIN + 35, element_type="gen",
                    element_index=5, action="restore",
                ),
                ContingencyEvent(
                    minute=_STABILISE_MIN + 50, element_type="load",
                    bus=2, p_mw=200.0, q_mvar=100.0,
                    action="trip",
                ),
            ),
        ),
        ScenarioSpec(
            name="winter_peak",
            start_time=_T_WINTER,
            duration_s=_TOTAL_MIN * 60,
            contingencies=(),
        ),
    ]


# ---------------------------------------------------------------------------
# Validation set: randomised
# ---------------------------------------------------------------------------

# Generators known to survive the ``wind_replace`` cull, and load buses
# with sufficient capacity for ±300 MW disturbances.
_VALID_GEN_INDICES = (2, 5)
_VALID_LOAD_BUSES = (5, 7, 27)


def validation_set(seed: int, n: int = 200) -> List[ScenarioSpec]:
    """Reproducible randomised scenarios for post-BO validation.

    Each scenario:

    * random ``start_time`` uniform over 2016 (full annual variation)
    * duration uniform in ``{30, 60, 90}`` minutes
    * 0–2 random contingencies (gen trip OR load step) with timing
      uniform within the scenario duration
    * ``scenario`` string in ``{'base_410', 'rural_700'}`` with 80/20 split

    Reproducibility: identical ``(seed, n)`` produces identical output.

    .. warning::
       The ``{30, 60, 90}`` minute duration draw reintroduces the very
       :math:`T^2` ITAE bias that :func:`design_set` fixes its duration to
       eliminate: with :math:`\\mathrm{ITAE} \\approx \\bar e\\,T^2/2` the
       multiplier spans 450 / 1800 / 4050, a 9x spread that has nothing to do
       with control quality.  Use duration-normalised metrics (or a fixed
       duration) before drawing generalisation conclusions from this set.
    """
    rng = np.random.default_rng(seed)
    scenarios: List[ScenarioSpec] = []

    for i in range(n):
        # start time
        day_of_year = int(rng.integers(1, 366))
        hour = int(rng.integers(0, 24))
        start = datetime(2016, 1, 1, hour, 0) + timedelta(days=day_of_year - 1)

        # duration
        duration_min = int(rng.choice((30, 60, 90)))
        duration_s = duration_min * 60

        # contingencies
        n_cont = int(rng.choice((0, 1, 2), p=(0.3, 0.5, 0.2)))
        events: list[ContingencyEvent] = []
        for _ in range(n_cont):
            kind = str(rng.choice(("gen", "load")))
            t_trip = float(rng.uniform(0.1, 0.6) * duration_min)
            t_restore = float(rng.uniform(t_trip + 5, max(t_trip + 6, duration_min - 1)))
            if kind == "gen":
                gi = int(rng.choice(_VALID_GEN_INDICES))
                events.append(ContingencyEvent(
                    minute=int(t_trip), element_type="gen",
                    element_index=gi, action="trip",
                ))
                events.append(ContingencyEvent(
                    minute=int(t_restore), element_type="gen",
                    element_index=gi, action="restore",
                ))
            else:
                bus = int(rng.choice(_VALID_LOAD_BUSES))
                p = float(rng.uniform(100, 400))
                q = float(rng.uniform(50, 200))
                events.append(ContingencyEvent(
                    minute=int(t_trip), element_type="load",
                    bus=bus, p_mw=p, q_mvar=q, action="connect",
                ))
                events.append(ContingencyEvent(
                    minute=int(t_restore), element_type="load",
                    bus=bus, p_mw=p, q_mvar=q, action="trip",
                ))

        # 2026-07-31: was ``"wind_replace" if rng.random() < 0.8 else "base"``.
        # ``"base"`` is not in SCENARIO_REGISTRY, so ``build_ieee39_net``
        # raised for ~20 % of draws and ``run_one`` turned each into a
        # sentinel cost — a fifth of every validation campaign was silently
        # counted as a power-flow failure.  ``"wind_replace"`` is a deprecated
        # alias for ``"base_410"``.  Both are now named directly, and the
        # 80/20 split is kept as a genuine two-network stratification.
        scenario_str = "base_410" if rng.random() < 0.8 else "rural_700"

        scenarios.append(ScenarioSpec(
            name=f"val_{i:03d}",
            start_time=start,
            duration_s=duration_s,
            contingencies=tuple(events),
            scenario=scenario_str,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# v2 sets: excitation-checked tune / holdout split
# ---------------------------------------------------------------------------
# Motivation (2026-07-31 audit).  The original ``design_set`` cannot identify
# the OLTC weights: taps were frozen in 77 % of clean runs, and a direct
# measurement of ``nominal_quiet`` produced 1 TSO tap and 0 DSO taps.  A weight
# whose actuator never moves has no leverage on any objective, so no budget can
# recover it.
#
# What forces OLTC action is not an impulsive trip — continuous actuators absorb
# that within 2-3 TSO steps — but *sustained drift after continuous authority
# saturates*.  Hence the ramp scenarios below: stepped load changes with **no
# restore**, so DER and generator reactive reserve runs out and the tap changer
# becomes the only remaining authority.
#
# Admission is gated rather than assumed: ``tuning/scripts/audit_design_set.py``
# runs each candidate at the reference weights and rejects any that fails to
# excite the actuator classes being tuned.

# Every v2 tune start time must fall in an ODD ISO week, because even weeks are
# reserved for the holdout (see _HOLDOUT_WEEK_PARITY).  The legacy `_T_WINTER`
# (2016-01-14) is ISO week 2 — a holdout week — so the v2 set uses the following
# Thursday instead.  Enforced by `test_tune_and_holdout_calendars_are_disjoint`.
_T_WINTER_V2 = datetime(2016, 1, 21, 18, 0)  # ISO week 3, winter evening peak
_T_SUMMER = datetime(2016, 7, 10, 3, 0)      # ISO week 27, summer night minimum
# _T_SPRING = datetime(2016, 4, 15, 12, 0)     # ISO week 15, spring noon mid-load
_T_SPRING = datetime(2016, 1, 5, 8, 0)     # ISO week 15, spring noon mid-load

def _load_ramp(
    bus: int,
    start_min: int,
    n_steps: int,
    step_min: int,
    p_mw: float,
    q_mvar: float,
) -> list[ContingencyEvent]:
    """Monotone stepped load change with **no restore**.

    The absence of a restore is the point: a trip/restore pair lets the
    continuous actuators ride through, whereas a one-way ramp eventually
    exhausts their reactive range and hands authority to the tap changers.
    """
    return [
        ContingencyEvent(
            minute=start_min + k * step_min, element_type="load", bus=bus,
            p_mw=p_mw, q_mvar=q_mvar, action="connect",
        )
        for k in range(n_steps)
    ]


def tune_set_v2() -> List[ScenarioSpec]:
    """Design set for the reparameterised tuning, 75 min each.

    Spans quiescent / impulsive / sustained behaviour across **both** installed-
    DER networks.  ``rural_700`` had never been tuned on at all — every scenario
    in the legacy set ran ``base_410`` through the deprecated ``wind_replace``
    alias.
    """
    total_s = _TOTAL_MIN * 60
    s0 = _STABILISE_MIN
    return [
        # Quiescent reference.  Keeps a low-excitation case in the set so tuning
        # cannot drift toward a controller that only behaves well under stress.
        ScenarioSpec(
            name="v2_quiet_spring", start_time=_T_SPRING,
            duration_s=total_s, scenario="rural_700",
        ),
        # Impulsive: the classic contingency pair, absorbable by the continuous
        # actuators.
        ScenarioSpec(
            name="v2_gen_trip", start_time=_T_SPRING, duration_s=total_s,
            scenario="base_410",
            contingencies=(
                ContingencyEvent(minute=s0 + 5, element_type="gen",
                                 element_index=2, action="trip"),
                ContingencyEvent(minute=s0 + 40, element_type="gen",
                                 element_index=2, action="restore"),
            ),
        ),
        # Sustained undervoltage: one-way load steps at the winter evening peak.
        ScenarioSpec(
            name="v2_undervoltage_ramp", start_time=_T_WINTER_V2,
            duration_s=total_s, scenario="base_410",
            contingencies=tuple(
                _load_ramp(bus=5, start_min=s0 + 5, n_steps=4, step_min=10,
                           p_mw=150.0, q_mvar=75.0)
                + _load_ramp(bus=7, start_min=s0 + 10, n_steps=3, step_min=10,
                             p_mw=100.0, q_mvar=50.0)
            ),
        ),
        # Sustained overvoltage on the high-DER network: summer night minimum
        # load with 700 MW installed DER at cos phi = 1 lifts HV voltages,
        # forcing taps in the opposite direction.
        ScenarioSpec(
            name="v2_overvoltage_rural", start_time=_T_SUMMER,
            duration_s=total_s, scenario="rural_700",
            contingencies=tuple(
                _load_ramp(bus=27, start_min=s0 + 5, n_steps=3, step_min=12,
                           p_mw=-120.0, q_mvar=-60.0)
            ),
        ),
        # ── WITHDRAWN 2026-08-03: "both layers stressed at once" ────────────
        #
        # A `v2_dual_rural` case (gen-5 trip + a load ramp at bus 5, winter
        # peak, rural_700) **diverged at the reference weights**, both at
        # 3 x 120 MW/60 Mvar and after softening to 2 x 70 MW/35 Mvar.  A
        # scenario the known-good controller cannot complete is not a
        # discriminator: every candidate fails it identically, so it costs ~3
        # min of wall clock per trial and yields no information about the
        # weights.
        #
        # It is withdrawn rather than softened further.  Softening because the
        # reference cannot complete a run is legitimate; softening repeatedly
        # until a chosen scenario passes is fitting the test set.  The four
        # scenarios above already satisfy every set-level excitation criterion
        # (27 TSO / 22 DSO tap moves, reserve driven to 0.080 by the ramp), so
        # nothing is lost in identifiability.
        #
        # OPEN ITEM: the set therefore has no case with both layers stressed
        # *simultaneously* on the high-DER network.  Adding one needs the
        # divergence understood first — is rural_700 + winter peak + a gen trip
        # genuinely outside the controller's envelope, or is the ramp hitting a
        # DSO capability limit?  Until then this is a stated gap, not a covered
        # case.
    ]


#: ISO-week parity reserved for the holdout.  SimBench profiles are strongly
#: autocorrelated within a day, so a random day-level split leaks between tune
#: and holdout; splitting on calendar blocks does not.
_HOLDOUT_WEEK_PARITY = 0        # even ISO weeks are holdout, odd are tuning


def holdout_set_v2(seed: int, n: int = 40) -> List[ScenarioSpec]:
    """Randomised holdout drawn only from calendar blocks the tune set avoids.

    Evaluate **once**, on the selected point only.  If tune-to-holdout
    degradation exceeds the pre-registered threshold, that is an overfitting
    result to report — not a licence to re-tune, which would consume the holdout
    and leave no independent evidence at all.

    Fixed duration, unlike :func:`validation_set`, which draws
    ``{30, 60, 90}`` min and so reintroduces a 9x ``T^2`` ITAE bias.
    """
    rng = np.random.default_rng(seed)
    out: List[ScenarioSpec] = []
    total_s = _TOTAL_MIN * 60
    attempts = 0
    while len(out) < n and attempts < 50 * n:
        attempts += 1
        day = int(rng.integers(1, 366))
        start = (datetime(2016, 1, 1, int(rng.integers(0, 24)), 0)
                 + timedelta(days=day - 1))
        if start.isocalendar().week % 2 != _HOLDOUT_WEEK_PARITY:
            continue                        # reserved for tuning
        network = "base_410" if rng.random() < 0.8 else "rural_700"

        events: list[ContingencyEvent] = []
        kind = float(rng.random())
        t0 = _STABILISE_MIN + int(rng.integers(3, 15))
        if kind < 0.35:
            gi = int(rng.choice(_VALID_GEN_INDICES))
            events += [
                ContingencyEvent(minute=t0, element_type="gen",
                                 element_index=gi, action="trip"),
                ContingencyEvent(minute=t0 + int(rng.integers(20, 40)),
                                 element_type="gen", element_index=gi,
                                 action="restore"),
            ]
        elif kind < 0.75:
            events += _load_ramp(
                bus=int(rng.choice(_VALID_LOAD_BUSES)), start_min=t0,
                n_steps=int(rng.integers(2, 5)), step_min=10,
                p_mw=float(rng.uniform(80, 180)),
                q_mvar=float(rng.uniform(40, 90)),
            )
        # else: quiescent

        out.append(ScenarioSpec(
            name=f"hold_{len(out):03d}", start_time=start,
            duration_s=total_s, scenario=network,
            contingencies=tuple(events),
        ))
    return out
