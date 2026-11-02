r"""
tuning/parameters.py
====================
The canonical controller-weight vector, and its overlay onto ``MultiTSOConfig``.

One vector, ten weights
-----------------------
After the 2026-09-10 reset there is exactly one definition of "the controller
weights".  :data:`WEIGHT_FIELDS` names them; a *parameter set* is a mapping over
exactly these keys plus the timing fields, and nothing else in the codebase may
carry a competing default (``configs.config`` raises on an unset weight).  The
tenth, the bank step price ``g_w_tso_shunt``, joined on 2026-10-22 when the
switched-shunt banks became integer inputs of the TS-MIQP.

The gauge
---------
The MIQP objective ``w^T G_w w + grad_f^T w + z^T G_z z`` is positively
homogeneous of degree one in the whole weight vector: scaling ``Q``, ``G_w``
and ``G_z`` by a common ``c`` multiplies the objective by ``c`` and leaves the
argmin — hence the trajectory, *including the integer tap and bank sequence* —
unchanged.  Measured 2026-07-31: reproduced to ~4e-10.

So weights map many-to-one onto controllers: an entire ray through the origin
is one controller.  A search over all ten spends budget on a direction that
provably changes nothing, and a reported ``g_v = 1e7`` means nothing on its own.

:data:`GAUGE_FIELD` is therefore **pinned**, and the remaining nine coordinates
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

#: The ten controller weights.
#:
#: ``g_w_tso_shunt`` was added on 2026-10-22.  Since the switched-shunt banks
#: became ordinary integer inputs of the TS-MIQP, their step price plays the
#: role the tap prices play: Stage I places it from a stated engagement
#: deviation (``tuning.stage1_screen.design_stage1``), and Stage II searches it
#: in a narrow box around that point.  Before that date it was not searched but
#: carried as a separate "gauge-coupled" field, because it had to scale with
#: the gauge (measured 2026-09-11: pinning ``g_v = 40000`` while the bank price
#: kept its old-gauge value made the banks never commit in the whole
#: campaign).  As a weight it scales with the gauge by construction.
WEIGHT_FIELDS: tuple[str, ...] = (
    "g_v",              # TS voltage tracking            [1/pu^2]   (gauge)
    "g_q",              # STS interface-Q tracking       [1/Mvar^2]
    "dso_g_v",          # STS voltage tracking           [1/pu^2]
    "g_w_der",          # TS DER step price              [1/Mvar^2]
    "g_w_pcc",          # PCC setpoint step price        [1/Mvar^2]
    "g_w_gen",          # AVR setpoint step price        [1/pu^2]
    "g_w_tso_oltc",     # machine-transformer tap price  [1/tap^2]
    "g_w_tso_shunt",    # MSC/MSR bank step price        [1/level^2]
    "g_w_dso_der",      # STS DER step price             [1/Mvar^2]
    "g_w_dso_oltc",     # STS coupling-transformer tap   [1/tap^2]
)

#: The nine coordinates a search may move.
SEARCHED_FIELDS: tuple[str, ...] = tuple(
    f for f in WEIGHT_FIELDS if f != GAUGE_FIELD
)

#: Switched-shunt design, carried in the parameter set for the same reason as
#: the cadence: the commit behaviour of the bulk banks is part of the design,
#: and leaving it in ``configs.config`` defaults is how it came to be silently
#: wrong.  The step price is NOT here -- it is a weight
#: (:data:`WEIGHT_FIELDS`).  What remains is the wear limit: the minimum
#: holding time, which the 2026-10-21 study found to be the one parameter the
#: law's benefit actually turns on (1800 s keeps the engagement, 900 s pays
#: +5.6% interface RMSE for an engage-release cycle).
SHUNT_FIELDS: tuple[str, ...] = (
    "shunt_t_dwell_s",
)

#: Cadence, carried in a parameter set so that a weight set and the periods it
#: was designed against cannot drift apart.
#:
#: ``N_inner = tso_period_s / dso_period_s`` *follows* from the two periods
#: and is not a field: ``T_TS`` = 180 s by definition and ``T_STS`` = 20 s
#: from the modal bound, so ``N_inner`` = 9.  Experiment C
#: (``ch:param:experiment:e``) checks ``n_s,95 <= 3*N_inner`` at the final
#: weights; it does not measure ``N_inner``.
#:
#: This said "a *measured* quantity (Experiment B)" until 2026-10-23.  That
#: described the superseded chapter, where the multiplier was measured on the
#: isolated subordinate loop; the 2026-09-29 restructure made it derived and
#: also remapped the experiment letters (former B + D -> B, E -> C,
#: F1/F2 -> D1/D2), so the old letter pointed at the weight selection.
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
    "live_plot_overview":      False,
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
    allowed = set(WEIGHT_FIELDS) | set(TIMING_FIELDS) | set(SHUNT_FIELDS)
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

    overlay: dict[str, Any] = {
        **{k: float(v) for k, v in params.items()},
        **FIXED_OVERRIDES,
    }
    return dataclasses.replace(cfg, **overlay)


def params_from_config(cfg: MultiTSOConfig) -> dict[str, float]:
    """Extract the ten weights from a config.  Pure extractor, no validation."""
    return {f: float(getattr(cfg, f)) for f in WEIGHT_FIELDS}


def weight_fingerprint(params: dict[str, float]) -> str:
    """Stable short digest of a weight vector, for run provenance and caching."""
    payload = {k: repr(float(params[k])) for k in sorted(params)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
