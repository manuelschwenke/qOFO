r"""
tuning/parameters.py
====================
The canonical controller-weight vector, and its overlay onto ``MultiTSOConfig``.

One vector, nine weights
------------------------
After the 2026-09-10 reset there is exactly one definition of "the controller
weights".  :data:`WEIGHT_FIELDS` names them; a *parameter set* is a mapping over
exactly these keys plus the timing fields, and nothing else in the codebase may
carry a competing default (``configs.config`` raises on an unset weight).

The gauge
---------
The MIQP objective ``w^T G_w w + grad_f^T w + z^T G_z z`` is positively
homogeneous of degree one in the whole weight vector: scaling ``Q``, ``G_w``,
``G_z`` and the shunt integrator's gain by a common ``c`` multiplies the
objective by ``c`` and leaves the argmin — hence the trajectory, *including the
integer tap sequence* — unchanged.  Measured 2026-07-31: reproduced to ~4e-10.

So weights map many-to-one onto controllers: an entire ray through the origin
is one controller.  A search over all nine spends budget on a direction that
provably changes nothing, and a reported ``g_v = 1e7`` means nothing on its own.

:data:`GAUGE_FIELD` is therefore **pinned**, and the remaining eight coordinates
are searched.  Objective weights are searched as dimensionless priorities
``pi = g * sigma^2`` (:class:`tuning.stage1_screen.PriorityScales`) so that
"voltage matters more than interface-Q" is a unit-free statement the reader can
check, rather than a ratio of raw magnitudes in different units.

What is *not* here
------------------
* **Scoring weights.**  Candidate selection is by stated operational constraints
  plus a lexicographic order on the controlled outputs, so there is no composite
  cost and no ``w_*`` to justify.  The retired ``CostWeights``/``PerfWeights``
  scalarisations were themselves untuned parameter sets, and the interface-Q
  term in them was gameable: a prior BO run drove ``g_w_pcc`` to ~270 because
  freezing the PCC setpoint makes interface-Q error vanish trivially.
* **Search bounds.**  The box is centred on the Stage-I point and lives with the
  campaign that draws it, not in this module — bounds that outlive the design
  they were drawn around are how the previous campaign came to exclude its own
  known-good point (``g_v = 1e7`` against a box of ``[1e2, 1e5]``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

from configs.config import MultiTSOConfig

__all__ = [
    "WEIGHT_FIELDS",
    "GAUGE_COUPLED_FIELDS",
    "SHUNT_FIELDS",
    "SEARCHED_FIELDS",
    "GAUGE_FIELD",
    "TIMING_FIELDS",
    "FIXED_OVERRIDES",
    "apply_to_config",
    "params_from_config",
    "weight_fingerprint",
]

#: The single weight the gauge is fixed on.  Every other weight is meaningful
#: only as a ratio to this one.  Pinned, never searched.
GAUGE_FIELD: str = "g_v"

#: The nine controller weights.  ``g_w_tso_shunt`` is deliberately absent: the
#: shunt is dispatched by a separate integrator outside the MIQP, so it is a
#: structural choice rather than a tuned weight.
WEIGHT_FIELDS: tuple[str, ...] = (
    "g_v",              # TS voltage tracking            [1/pu^2]   (gauge)
    "g_q",              # STS interface-Q tracking       [1/Mvar^2]
    "dso_g_v",          # STS voltage tracking           [1/pu^2]
    "g_w_der",          # TS DER step price              [1/Mvar^2]
    "g_w_pcc",          # PCC setpoint step price        [1/Mvar^2]
    "g_w_gen",          # AVR setpoint step price        [1/pu^2]
    "g_w_tso_oltc",     # machine-transformer tap price  [1/tap^2]
    "g_w_dso_der",      # STS DER step price             [1/Mvar^2]
    "g_w_dso_oltc",     # STS coupling-transformer tap   [1/tap^2]
)

#: The eight coordinates a search may move.
SEARCHED_FIELDS: tuple[str, ...] = tuple(
    f for f in WEIGHT_FIELDS if f != GAUGE_FIELD
)

#: Weights that are **not searched** but that must scale with the gauge.
#:
#: These sit in the invariance group with everything in :data:`WEIGHT_FIELDS`
#: -- scaling the whole group by a common factor reproduces the trajectory --
#: but they are set by their own rule rather than by the search, so they were
#: originally left out of the parameter set entirely.  That was a defect.
#:
#: Measured 2026-09-11: pinning the gauge at ``g_v = 40000`` (from ``1e7``)
#: while ``shunt_int_g_w`` stayed at its old-gauge value of 10 made the
#: integrator advance ~250x too slowly for its own commit threshold.  The
#: accumulated state peaked at 0.395 Mvar against a threshold of 22.5, so the
#: eight shunt banks **never committed once** -- an entire actuator class was
#: silently absent from the whole campaign.
#:
#: Carrying them in the parameter set is what stops that recurring: the gauge
#: and everything coupled to it now move together or not at all.
GAUGE_COUPLED_FIELDS: tuple[str, ...] = (
    "shunt_int_g_w",
)

#: Switched-shunt integrator design, carried in the parameter set for the same
#: reason as the cadence: the commit behaviour of the bulk banks is part of the
#: design, and leaving it in ``configs.config`` defaults is how it came to be
#: silently wrong.  ``shunt_int_g_w`` is NOT here -- it is gauge-coupled and
#: lives in :data:`GAUGE_COUPLED_FIELDS`; these three are free design choices.
SHUNT_FIELDS: tuple[str, ...] = (
    "shunt_int_delta_mvar",
    "shunt_int_t_dwell_s",
    "shunt_int_daily_budget",
)

#: Cadence, carried in a parameter set so that a weight set and the periods it
#: was designed against cannot drift apart.  ``N_inner = tso_period_s /
#: dso_period_s`` is a *measured* quantity (Experiment B) and is not a field.
TIMING_FIELDS: tuple[str, ...] = ("tso_period_s", "dso_period_s", "dt_s")

#: Fields always pinned when a parameter set is applied.  Headless and
#: structural only — **no weights**.  A weight pinned here would be a silent
#: parameter set of exactly the kind the reset removed.
FIXED_OVERRIDES: dict[str, Any] = {
    # Structural choices, not tuned.
    "int_max_step":            1,
    # 1 -> 5 on 2026-10-14 (author decision): aligned with the evaluated runs
    # (configs/config.py int_cooldown = 5).  Campaigns before this date,
    # including the one that selected 399e30d3b60b, ran with 1.
    "int_cooldown":            5,
    # Headless / deterministic.
    "verbose":                 0,
    "live_plot_controller":    False,
    "live_plot_cascade":       False,
    "live_plot_system":        False,
    "run_stability_analysis":  False,
    # The runtime curvature preconditioner REGENERATES g_w at controller
    # construction, overwriting whatever a parameter set supplied.  It must be
    # off, or the weights that run are not the weights that were selected.
    "precondition_g_w":        False,
}


def apply_to_config(cfg: MultiTSOConfig, params: dict[str, float]) -> MultiTSOConfig:
    """Return ``cfg`` with a weight/timing mapping and :data:`FIXED_OVERRIDES`.

    ``params`` must contain **every** weight in :data:`WEIGHT_FIELDS` — the
    gauge included, since it is pinned rather than absent — and may contain the
    timing fields.  Unknown keys and missing weights both raise: a partial
    overlay that silently inherits a weight from somewhere else is the failure
    mode this reset exists to remove.

    ``dataclasses.replace`` is used so ``__post_init__`` re-derives everything
    downstream of the weights (per-area voltage relief in particular).
    """
    allowed = (set(WEIGHT_FIELDS) | set(TIMING_FIELDS)
               | set(GAUGE_COUPLED_FIELDS) | set(SHUNT_FIELDS))
    given = set(params)
    extra = sorted(given - allowed)
    if extra:
        raise ValueError(
            f"unknown parameter-set keys: {extra}. Allowed: "
            f"{sorted(allowed)}")
    missing = sorted(set(WEIGHT_FIELDS) - given)
    if missing:
        raise KeyError(
            f"parameter set is incomplete: {missing} missing. Every weight "
            f"must be stated explicitly; there is no default to fall back on.")

    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"non-numeric value for {k}: {v!r}")
        v_f = float(v)
        if not math.isfinite(v_f):
            raise ValueError(f"non-finite value for {k}: {v}")
        if v_f <= 0.0:
            raise ValueError(
                f"non-positive value for {k}: {v}. Every weight divides or "
                f"scales a curvature term; zero is not a valid price.")

    # ``shunt_int_daily_budget`` is an int on the dataclass; casting every key
    # to float would hand the bank config a float where it validates an int.
    _int_fields = {"shunt_int_daily_budget"}
    overlay: dict[str, Any] = {
        **{k: (int(v) if k in _int_fields else float(v))
           for k, v in params.items()},
        **FIXED_OVERRIDES,
    }
    return dataclasses.replace(cfg, **overlay)


def params_from_config(cfg: MultiTSOConfig) -> dict[str, float]:
    """Extract the nine weights from a config.  Pure extractor, no validation."""
    return {f: float(getattr(cfg, f)) for f in WEIGHT_FIELDS}


def weight_fingerprint(params: dict[str, float]) -> str:
    """Stable short digest of a weight vector, for run provenance and caching."""
    payload = {k: repr(float(params[k])) for k in sorted(params)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
