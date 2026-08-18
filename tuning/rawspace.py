"""
tuning/rawspace.py
==================
Raw-weight search space, with the curvature preconditioner **off**.

Why this exists
---------------
The reparameterised space in :mod:`tuning.reparam` tunes a target
``lambda_max(M)`` per layer, and ``lambda`` has no route into a config other
than the preconditioner: ``apply_reparam_to_config`` therefore forces
``precondition_g_w=True`` on every trial.  When the operating requirement is
``precondition_g_w = False`` in the shipped controller, that space cannot
express the result -- a tuned point from it is only reproducible with the
preconditioner running.  This module searches the weights themselves instead.

The two defects that motivated :mod:`tuning.reparam` are handled here *without*
the preconditioner:

1. **Exact scaling redundancy.**  Scaling every objective weight, every ``g_w``,
   every ``g_z`` and the shunt integrator gain by a common factor reproduces the
   trajectory to ~4e-10 (measured 2026-07-31).  It is quotiented out the same
   way as in ``reparam.py``: by pinning the gauge ``{g_v, g_q, g_w_gen,
   tso_g_q_pcc}``.  With the gauge fixed, a common factor on the ``g_w`` block
   is no longer redundant -- it *is* the loop gain -- so all five coordinates
   below are identifiable.
2. **A box that excluded the known-good point** (``g_v = 1e7`` against a box of
   ``[1e2, 1e5]``).  Every coordinate here is a **ratio to the baseline**, so
   the reference sits at 1.0 by construction and cannot fall outside its own
   search space.

Coordinates
-----------
Chosen to mirror the reparameterised study one-for-one, so the two are
comparable at equal budget:

======================  ===========================================
raw coordinate          reparameterised counterpart
======================  ===========================================
``g_w_der_ratio``       ``tso_lambda`` x ``tau_der_pcc`` (gain+shape)
``g_w_pcc_ratio``       -- " --
``g_w_dso_der_ratio``   ``dso_lambda``
``dso_v_priority``      identical coordinate
``shunt_int_gain``      identical coordinate
======================  ===========================================

Deliberately **not** searched, exactly as in the reparameterised study:
``g_w_tso_oltc`` / ``g_w_dso_oltc`` (integer switching prices -- monotone in
switching rate, calibrated by 1-D bisection against an operational taps/hour
budget, see :mod:`tuning.bisect_switching`), and ``g_w_gen`` (the AVR carries
0.6-2.3 % of TSO curvature).

Known limitation, stated up front
---------------------------------
``g_w_der`` and ``g_w_pcc`` are **global scalars**: one value for every TSO
zone.  The preconditioner assigned per-zone -- indeed per-column -- weights, and
those differ materially: measured 2026-08-14 at the tuned point, TSO zone 1
wanted ``g_w_der = 0.70`` against ~2.9 in zones 2 and 3, and the DSO ``g_w`` had
a 1.8x spread *within* one controller.  That structure is not representable
here.  The gap between the best raw point and the preconditioned optimum is the
price of the restriction, and measuring it is the point of the pilot study.

Author: Manuel Schwenke (with Claude Code), 2026-08-14
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import numpy as np
import optuna

from configs.config import MultiTSOConfig
from tuning._types import BOParam
from tuning.metrics import MetricScales
from tuning.objective import cvar_aggregate
from tuning.objectives_v2 import (
    CONSTRAINT_KEY,
    CONSTRAINT_NAMES,
    ConstraintLimits,
    PerfWeights,
    _run_scenario,
    _worst_settling_s,
    feasibility_constraints,
    performance_scalar,
)
from tuning.runner import RunResult

__all__ = [
    "RAW_DIMS",
    "RawReference",
    "apply_raw_to_config",
    "make_raw_objective",
    "raw_space_fingerprint",
    "raw_search_space",
    "sample_raw_coords",
]


#: Bumped when the *meaning* of a coordinate changes.
RAW_SPACE_VERSION: int = 1

#: Half-width, in decades, of the multiplicative window around the baseline.
#: Same convention as ``reparam._WINDOW_DECADES``: bounds relative to the
#: reference, so the reference is the centre and is always representable.
_WINDOW_DECADES: float = 1.5

_LO = 10.0 ** -_WINDOW_DECADES
_HI = 10.0 ** _WINDOW_DECADES

#: ``coordinate -> config field`` it scales.
RAW_FIELDS: dict[str, str] = {
    "g_w_der_ratio": "g_w_der",
    "g_w_pcc_ratio": "g_w_pcc",
    "g_w_dso_der_ratio": "g_w_dso_der",
    "dso_v_priority": "dso_g_v",
    "shunt_int_gain": "shunt_int_g_w",
}

RAW_DIMS: tuple[BOParam, ...] = tuple(
    BOParam(name, log=True, low=_LO, high=_HI) for name in RAW_FIELDS
)


@dataclass(frozen=True)
class RawReference:
    """Baseline values the ratio coordinates multiply, plus the pinned gauge."""

    values: dict[str, float]          # config field -> reference value
    gauge: dict[str, float]           # pinned fields -> value

    @classmethod
    def from_config(cls, cfg: MultiTSOConfig) -> "RawReference":
        values = {}
        for field in RAW_FIELDS.values():
            v = float(getattr(cfg, field))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(
                    f"Reference {field}={v!r} is not a positive finite number; "
                    f"a ratio coordinate cannot be defined about it."
                )
            values[field] = v
        return cls(
            values=values,
            gauge={
                "g_v": float(cfg.g_v),
                "g_q": float(cfg.g_q),
                "g_w_gen": float(cfg.g_w_gen),
                "tso_g_q_pcc": float(getattr(cfg, "tso_g_q_pcc", 0.0) or 0.0),
            },
        )


def raw_search_space() -> dict[str, tuple[float, float, bool]]:
    """``{name: (low, high, log)}`` for the raw space."""
    return {p.name: (float(p.low), float(p.high), bool(p.log))
            for p in RAW_DIMS}


def raw_space_fingerprint(ref: RawReference) -> str:
    """Digest of space + reference, so a resumed study cannot silently mix.

    The reference is part of the identity here: the same ratio means a
    different weight if the baseline moves.
    """
    payload = {
        "version": RAW_SPACE_VERSION,
        "dims": sorted((p.name, float(p.low), float(p.high), bool(p.log))
                       for p in RAW_DIMS),
        "reference": {k: float(v) for k, v in sorted(ref.values.items())},
        "gauge": {k: float(v) for k, v in sorted(ref.gauge.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def sample_raw_coords(trial: optuna.Trial) -> dict[str, float]:
    """Sample one point from the raw ratio space."""
    return {
        p.name: trial.suggest_float(p.name, float(p.low), float(p.high),
                                    log=bool(p.log))
        for p in RAW_DIMS
    }


def resolve_weights(coords: dict[str, float],
                    ref: RawReference) -> dict[str, float]:
    """``{config field: absolute weight}`` for one coordinate vector."""
    return {
        field: float(coords[coord]) * ref.values[field]
        for coord, field in RAW_FIELDS.items()
    }


def apply_raw_to_config(
    cfg: MultiTSOConfig,
    coords: dict[str, float],
    ref: RawReference,
    *,
    fixed_overrides: dict[str, Any] | None = None,
) -> MultiTSOConfig:
    """Overlay raw ratio coordinates onto a config, preconditioner **off**.

    The gauge is written back explicitly rather than merely inherited, so the
    resolved config states its own numeraire and a later edit to the baseline
    cannot silently move it.
    """
    expected = set(RAW_FIELDS)
    given = set(coords)
    if given - expected:
        raise ValueError(f"Unknown raw coords: {sorted(given - expected)}")
    if expected - given:
        raise KeyError(f"Missing raw coords: {sorted(expected - given)}")
    for k, v in coords.items():
        f = float(v)
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(f"Coordinate {k} must be finite and positive: {v}")

    overlay: dict[str, Any] = dict(ref.gauge)
    overlay.update(resolve_weights(coords, ref))
    # The whole point of this space: no curvature preconditioning, so the
    # weights above are the ones the controllers actually run with.
    overlay["precondition_g_w"] = False

    if fixed_overrides:
        overlay.update(fixed_overrides)
    return dataclasses.replace(cfg, **overlay)


def make_raw_objective(
    baseline_cfg: MultiTSOConfig,
    ref: RawReference,
    scenarios: Sequence,
    *,
    fixed_overrides: dict | None = None,
    limits: ConstraintLimits | None = None,
    weights: PerfWeights | None = None,
    scales: MetricScales | None = None,
    short_circuit: bool = True,
    cvar_pct: float = 100.0,
    perf_exclude: frozenset[str] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective over the raw space.

    Scoring, constraint vector and aggregation are the *same functions* the
    reparameterised study used (:mod:`tuning.objectives_v2`), so the objective
    values of the two studies are directly comparable as long as the weight
    profile, scenario set, limits and ``cvar_pct`` match.
    """
    limits = limits or ConstraintLimits()
    weights = weights or PerfWeights()
    scales = scales or MetricScales()
    excluded = frozenset(perf_exclude or ())
    unknown = excluded - {getattr(s, "name", "") for s in scenarios}
    if unknown:
        raise ValueError(
            f"perf_exclude names scenarios not in the set: {sorted(unknown)}"
        )
    if excluded and len(excluded) >= len(scenarios):
        raise ValueError("perf_exclude would leave no scenario in the aggregate")

    def objective(trial: optuna.Trial) -> float:
        coords = sample_raw_coords(trial)
        cfg = apply_raw_to_config(baseline_cfg, coords, ref,
                                  fixed_overrides=fixed_overrides)
        # The resolved weights, so a trial can be read without re-deriving them.
        for field, value in resolve_weights(coords, ref).items():
            trial.set_user_attr(f"w__{field}", float(value))

        results: List[RunResult] = []
        settling: list[float] = []
        for scenario in scenarios:
            res, records = _run_scenario(scenario, cfg, scales)
            results.append(res)
            settling.append(_worst_settling_s(records, scenario.event_times_s))
            trial.set_user_attr(f"wall_s__{scenario.name}",
                                float(res.wall_time_s))
            if res.failure_reason:
                trial.set_user_attr(f"err__{scenario.name}",
                                    res.failure_reason[:500])
            if short_circuit and not res.metrics.feasible:
                trial.set_user_attr("short_circuited_at", scenario.name)
                break

        g = feasibility_constraints(
            results, cfg, limits,
            settling_s_by_scenario=settling if any(
                math.isfinite(s) and s > 0.0 for s in settling) else None,
        )
        trial.set_user_attr(CONSTRAINT_KEY, [float(v) for v in g])
        for name, value in zip(CONSTRAINT_NAMES, g):
            trial.set_user_attr(f"c__{name}", float(value))

        values: list[float] = []
        excluded_values: list[float] = []
        for res in results:
            total, parts = performance_scalar(res.metrics, weights, scales)
            if res.scenario_name in excluded:
                excluded_values.append(total)
            else:
                values.append(total)
            trial.set_user_attr(f"perf__{res.scenario_name}", float(total))
            for key, part in parts.items():
                trial.set_user_attr(
                    f"perf__{res.scenario_name}__{key}", float(part))
        for name, val in zip(sorted(excluded), excluded_values):
            trial.set_user_attr(f"perf_excluded__{name}", float(val))

        agg = cvar_aggregate(values, pct=cvar_pct)
        trial.set_user_attr("cvar_perf", float(agg))
        trial.set_user_attr(
            "mean_perf", float(np.mean(values)) if values else float("inf"))
        trial.set_user_attr("n_perf_scenarios", len(values))
        return float(agg)

    return objective
