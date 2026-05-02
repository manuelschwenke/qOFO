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

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent


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
    scenario: str = "wind_replace"
    use_profiles: bool = True
    tso_period_s: float = 180.0
    dso_period_s: float = 10.0

    def overlay_on(self, cfg: MultiTSOConfig) -> MultiTSOConfig:
        """Return a new :class:`MultiTSOConfig` with this scenario's
        time-series fields applied."""
        return dataclasses.replace(
            cfg,
            n_total_s=float(self.duration_s),
            start_time=self.start_time,
            tso_period_s=float(self.tso_period_s),
            dso_period_s=float(self.dso_period_s),
            scenario=self.scenario,
            use_profiles=self.use_profiles,
            contingencies=list(self.contingencies),
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
    * ``scenario`` string in ``{'wind_replace', 'base'}`` with 80/20
      split

    Reproducibility: identical ``(seed, n)`` produces identical output.
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

        scenario_str = "wind_replace" if rng.random() < 0.8 else "base"

        scenarios.append(ScenarioSpec(
            name=f"val_{i:03d}",
            start_time=start,
            duration_s=duration_s,
            contingencies=tuple(events),
            scenario=scenario_str,
        ))

    return scenarios
