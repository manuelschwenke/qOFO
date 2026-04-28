"""
tuning/runner.py
================
One BO trial = one design scenario.

Wraps :func:`run_multi_tso_dso` with:

* controller-param overlay via :func:`tuning.parameters.apply_to_config`
* scenario overlay via :meth:`tuning.scenarios.ScenarioSpec.overlay_on`
* stdout / stderr suppression (BO must be silent)
* failure handling (any exception → high-cost sentinel metrics)
* timing diagnostics (used by Task-3 trial ``user_attrs``)

Note on stdout suppression
--------------------------
We use :func:`contextlib.redirect_stdout`.  If
:func:`run_multi_tso_dso` (or any of its dependencies) routes output
through Python's :mod:`logging` instead of bare ``print``, that output
will leak through.  This is acceptable: BO trials may produce a small
amount of log noise.  Configure ``logging`` at the call site if a
fully-silent run is required.
"""

from __future__ import annotations

import contextlib
import io
import time
import traceback
from dataclasses import dataclass

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import MultiTSOIterationRecord
from tuning._sim_loader import get_run_multi_tso_dso
from tuning.metrics import (
    CostWeights,
    NoiseFloors,
    TrajectoryMetrics,
    extract_metrics,
)
from tuning.parameters import apply_to_config
from tuning.scenarios import ScenarioSpec


@dataclass(frozen=True)
class RunResult:
    """One scenario evaluation: metrics + diagnostics."""

    scenario_name: str
    metrics: TrajectoryMetrics
    wall_time_s: float
    failure_reason: str = ""


def run_one(
    params: dict[str, float],
    scenario: ScenarioSpec,
    baseline_cfg: MultiTSOConfig,
    cost_weights: CostWeights | None = None,
    noise_floors: NoiseFloors | None = None,
) -> RunResult:
    """Evaluate one ``(params, scenario)`` pair.

    Pipeline
    --------
    1. ``cfg = scenario.overlay_on(baseline_cfg)``
    2. ``cfg = apply_to_config(cfg, params)``  (also applies
       ``FIXED_OVERRIDES``: live plots/observer/analysis off, integral
       mode off, etc.)
    3. With stdout / stderr suppressed: ``log = run_multi_tso_dso(cfg)``
    4. ``metrics = extract_metrics(log, cfg, ...)``

    Any exception during step 3 is caught, recorded in
    :attr:`RunResult.failure_reason`, and a sentinel
    :class:`TrajectoryMetrics` with high ``cost_J`` is returned.  BO
    never crashes due to a bad parameter combination.

    Parameters
    ----------
    params
        BO param dict matching ``BO_DIMS`` keys.  Validation is deferred
        to :func:`apply_to_config` (raises :class:`ValueError` on bad
        keys -- *this* exception is **not** caught and propagates).
    scenario
        :class:`ScenarioSpec` defining the operating point.
    baseline_cfg
        Baseline :class:`MultiTSOConfig`.  ``FIXED_OVERRIDES`` will
        force headless / deterministic flags on.
    cost_weights, noise_floors
        Forwarded to :func:`extract_metrics`.

    Returns
    -------
    RunResult
    """
    cost_weights = cost_weights or CostWeights()
    noise_floors = noise_floors or NoiseFloors()

    cfg = scenario.overlay_on(baseline_cfg)
    cfg = apply_to_config(cfg, params)

    run_fn = get_run_multi_tso_dso()

    t0 = time.perf_counter()
    failure = ""
    log: list[MultiTSOIterationRecord] = []

    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), \
                contextlib.redirect_stderr(buf_err):
            log = run_fn(cfg)
    except Exception as e:
        failure = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        log = []

    wall = time.perf_counter() - t0
    metrics = extract_metrics(log, cfg, cost_weights, noise_floors)

    return RunResult(
        scenario_name=scenario.name,
        metrics=metrics,
        wall_time_s=wall,
        failure_reason=failure,
    )
