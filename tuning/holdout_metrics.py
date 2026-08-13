"""
tuning/holdout_metrics.py
=========================
Interface-Q tracking as an **RMS intensity**, for the Stage-5 holdout.

Why this exists
---------------
``TrajectoryMetrics`` reports interface-Q tracking only as ITAE
(``itae_q_pcc`` / ``itae_q_tie``, units min*Mvar) — a time-weighted
*accumulation*.  The holdout is specified to score "interface-Q RMS", an
*intensity* in Mvar, which is a different statistic and not derivable from the
ITAE.

Note on a premise that does **not** apply: ITAE is duration-sensitive, but
:func:`tuning.scenarios.holdout_set_v2` deliberately uses a *fixed* duration
(unlike ``validation_set``, whose ``{30, 60, 90}`` min draw reintroduces a 9x
``T^2`` ITAE bias).  So RMS is not needed to defend against a duration bias
here.  It is reported because it is the requested statistic and because an
intensity is the directly interpretable quantity for an interface ("this
interface misses its setpoint by X Mvar rms"), with ITAE retained alongside for
continuity with the tuning objective.

Same signal, deliberately
-------------------------
The per-step error series built here is the *same* one
``metrics._itae_q_pcc`` / ``_itae_q_tie`` consume: per step, the mean absolute
setpoint error across the interfaces present at that step.  That equivalence is
pinned by a test which reconstructs the ITAE from these series and compares it
against ``extract_metrics`` output, so the two can never silently diverge.

This lives in its own module rather than in ``metrics.py`` so that adding it
cannot perturb a tuning run that already has ``metrics`` imported.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "q_pcc_error_series",
    "q_tie_error_series",
    "rms_of_series",
]


def _time_min(records: Sequence[Any]) -> NDArray[np.float64]:
    return np.array([r.time_s / 60.0 for r in records], dtype=float)


def q_pcc_error_series(
    records: Sequence[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Per-step mean ``|Q_set - Q_actual|`` across all DSO PCC interfaces.

    Returns ``(t_min, err_per_step, any_interfaces)``.  ``err_per_step`` is NaN
    at steps where no interface reported finite values.  ``any_interfaces`` is
    False when the network has no PCC interfaces at all — which must be reported
    as 0.0 rather than NaN, since a network with nothing to track has not got
    anything wrong (mirrors ``metrics._itae_q_pcc``).
    """
    if not records:
        return np.zeros(0), np.zeros(0), False
    t_min = _time_min(records)
    err = np.full(len(records), np.nan)
    any_keys = False
    for i, r in enumerate(records):
        keys = set(r.dso_trafo_q_set_mvar) & set(r.dso_trafo_q_actual_mvar)
        if not keys:
            continue
        any_keys = True
        e = [abs(r.dso_trafo_q_set_mvar[k] - r.dso_trafo_q_actual_mvar[k])
             for k in keys
             if math.isfinite(r.dso_trafo_q_set_mvar[k])
             and math.isfinite(r.dso_trafo_q_actual_mvar[k])]
        if e:
            err[i] = float(np.mean(e))
    return t_min, err, any_keys


def q_tie_error_series(
    records: Sequence[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Per-step mean ``|Q_tie - Q_tie_set|`` across all zone pairs.

    Setpoint resolution matches ``metrics._itae_q_tie``: per-pair setpoints are
    read from ``zone_tie_q_set_mvar`` when the runner populates it, else the
    Phase-B target of 0 Mvar is used.
    """
    if not records:
        return np.zeros(0), np.zeros(0), False
    t_min = _time_min(records)
    err = np.full(len(records), np.nan)
    any_pairs = False
    for i, r in enumerate(records):
        pair_q = r.zone_tie_q_mvar
        pair_set = getattr(r, "zone_tie_q_set_mvar", {}) or {}
        if not pair_q:
            continue
        any_pairs = True
        e: List[float] = []
        for pair, q in pair_q.items():
            if q is None or not math.isfinite(float(q)):
                continue
            sp = float(pair_set.get(pair, 0.0))
            e.append(abs(float(q) - sp))
        if e:
            err[i] = float(np.mean(e))
    return t_min, err, any_pairs


def rms_of_series(
    err_per_step: NDArray[np.float64],
    any_interfaces: bool,
) -> float:
    """Root-mean-square of a per-step error series [Mvar].

    Semantics deliberately mirror ``metrics._itae``:

    * ``0.0`` when there are no interfaces to track at all — not a failure;
    * ``nan`` when interfaces exist but fewer than two finite samples survive,
      so a diverged trajectory propagates to ``inf`` through the normaliser
      instead of scoring as *perfect* tracking.  The two-sample floor matches
      ``_itae``'s trapezoid requirement so RMS and ITAE agree on which runs are
      admissible; a one-sample RMS would otherwise report a finite score where
      ITAE reports ``nan``.
    """
    if not any_interfaces:
        return 0.0
    if err_per_step.size == 0:
        return float("nan")
    finite = np.isfinite(err_per_step)
    if int(finite.sum()) < 2:
        return float("nan")
    vals = err_per_step[finite]
    return float(np.sqrt(np.mean(vals ** 2)))
