"""
IEEE 39 wind_replace Integration for the Stability Observer
============================================================

Drop-in integration of :class:`stability_observer.StabilityObserver`
into the ``run_multi_tso_dso`` experiment script in
``experiments/000_M_TSO_M_DSO.py``.

This is a *passive* add-on: the observer listens to every cross-sensitivity
refresh that the coordinator already performs, computes the per-block
stability-minimum ``g_w`` at that operating point, and accumulates the
trajectory.  At end of simulation it writes a JSON summary and per-zone
plots alongside the existing ``stability_analysis_t{min}.md`` report.

No controller behaviour is affected.  The only runtime cost is one
``eigvalsh`` call per tracked zone per refresh — negligible compared to
the MIQP solve.

Integration (three-line patch to the experiment script)
--------------------------------------------------------
After the coordinator is constructed (around line 1012 of the experiment
script)::

    from stability_integration_ieee39 import attach_observer
    observer = attach_observer(coordinator, zone_defs, config)

Inside the main loop, right after ``coordinator.step(...)`` fires with
``refresh_H=True`` (around line 1658)::

    if refresh_H:
        observer.record(time_s=time_s)

After the simulation loop completes (near line 1860, right after the
existing delayed stability report)::

    observer.write_results(config.result_dir)

That is the entire integration.  The observer's output files live next to
``stability_analysis_t60min.md`` / ``tuned_params_t60min.json`` and cover
the full profile trajectory rather than a single snapshot.

Scenario-specific defaults
--------------------------
The ``attach_observer`` factory here is tuned for the ``wind_replace``
scenario:

- **ratio_priors** = (1, 2, 3, 10) mapping (DER, PCC, V_gen, OLTC).
  Physical prior: machine-transformer OLTCs need the most proximal
  regularisation because each tap move is a ~0.01 pu perturbation.
  V_gen setpoints next (AVR bandwidth), PCC setpoints (tracked by DSO),
  then DER Q (fast STATCOM inverters, smallest prior).
- **safety_margin** = 0.30 — 30% inflation over the computed minimum.
  Covers the discrete-relaxation gap (MIQP integer moves not captured
  by the continuous-relaxation analysis) and drift between refreshes.
- **method** = "block" — block-scalar g_w, matching your
  ``multi_tso_coordinator.py`` configuration which uses one scalar per
  actuator class.

Override these in the call to ``attach_observer`` if needed.

Author: Manuel Schwenke (drafted with Claude 2026-04-18)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .stability_observer import StabilityObserver, write_trajectory_report


def attach_observer(
    coordinator: Any,
    zone_defs: Dict[int, Any],
    config: Any,
    *,
    ratio_priors: Sequence[float] = (1.0, 2.0, 3.0, 10.0),
    safety_margin: float = 0.30,
    method: str = "lmi",
    tracked_zone_ids: Optional[Sequence[int]] = None,
    verbose: int = 1,
) -> StabilityObserver:
    """
    Build and return a :class:`StabilityObserver` tied to the coordinator.

    Parameters
    ----------
    coordinator : MultiTSOCoordinator
        The already-constructed coordinator (ZoneDefinition list + net).
    zone_defs : dict[int, ZoneDefinition]
        Same zone definitions passed to the coordinator.
    config : MultiTSOConfig
        The experiment config.  Uses ``config.g_v`` and ``config.g_q``.
    ratio_priors : sequence of 4 floats
        Physical-prior ratios for the four actuator blocks
        ``(DER, PCC, V_gen, OLTC)``.  See module docstring.
    safety_margin : float
        Fractional inflation of the computed minimum (default 0.30).
    method : {"block", "lmi"}
        ``"block"`` for block-scalar tuning (thesis default),
        ``"lmi"`` for the per-actuator Gershgorin bound.
    tracked_zone_ids : sequence of int, optional
        Restrict observation to a subset of zones.  Default: all zones.
    verbose : int
        If ``>= 1``, print a one-line banner describing the observer.

    Returns
    -------
    StabilityObserver
    """
    observer = StabilityObserver(
        coordinator=coordinator,
        zone_defs=zone_defs,
        g_v=float(config.g_v),
        g_q=float(config.g_q),
        ratio_priors=ratio_priors,
        safety_margin=safety_margin,
        tracked_zone_ids=tracked_zone_ids,
        method=method,
    )
    if verbose >= 1:
        n_zones = len(observer.tracked_zone_ids)
        print()
        print(f"[9a] Stability observer attached "
              f"(tracking {n_zones} zones, method={method}, "
              f"safety_margin={safety_margin:.2f}).")
    return observer


def write_observer_results_alongside_report(
    observer: StabilityObserver,
    result_dir: str,
    *,
    basename: str = "stability_observer",
    verbose: int = 1,
) -> None:
    """
    Save observer results to ``result_dir`` alongside the existing
    stability-analysis markdown report.

    Writes:
    - ``stability_observer.json`` — all per-zone aggregations
    - ``stability_observer_zone{z}.png`` — time-series + histogram
    - ``stability_observer_report.md`` — human-readable table

    Parameters
    ----------
    observer : StabilityObserver
    result_dir : str
        Same directory used by ``_run_delayed_stability_analysis``.
    basename : str
        Filename prefix (default ``stability_observer``).
    verbose : int
    """
    os.makedirs(result_dir, exist_ok=True)
    observer.write_results(result_dir, basename=basename, plot=True)

    md_path = os.path.join(result_dir, f"{basename}_report.md")
    write_trajectory_report(md_path, observer.trajectories)

    if verbose >= 1:
        n = sum(len(t.records) for t in observer.trajectories.values())
        print(f"  Stability observer report:   {md_path}")
        print(f"  Observer JSON + plots:       {result_dir}/"
              f"{basename}.json (+{len(observer.tracked_zone_ids)} PNGs, "
              f"{n} total observations)")


# --------------------------------------------------------------------------- #
#  In-loop helper: refresh H for the observer without disturbing the cache
# --------------------------------------------------------------------------- #

def observer_record_fresh(
    observer: StabilityObserver,
    coordinator: Any,
    *,
    time_s: float,
) -> None:
    """
    Record one observer snapshot with a freshly-computed H, without
    disturbing the coordinator's cached ``_H_blocks`` that the controller
    relies on at its own (slower) refresh cadence.

    The pattern: swap in an empty dict, call the coordinator's normal
    ``compute_cross_sensitivities()`` to populate it, let the observer
    read from it, then restore the original (stale) dict.  The coordinator
    class itself is not modified; only its private ``_H_blocks`` attribute
    is rebound (and rebound back) by reference, exception-safe via
    ``try/finally``.

    Use from inside the main TSO loop instead of the gated ``if refresh_H:
    observer.record(...)`` pattern when the controller's refresh cadence
    (``config.sensitivity_update_interval``) is intentionally much slower
    than the desired observation cadence.
    """
    saved = coordinator._H_blocks
    coordinator._H_blocks = {}
    try:
        coordinator.compute_cross_sensitivities()
        observer.record(time_s=time_s)
    finally:
        coordinator._H_blocks = saved


# --------------------------------------------------------------------------- #
#  End-of-simulation helper: derive tuned g_w values from the trajectory
# --------------------------------------------------------------------------- #

@dataclass
class ObserverTunedGw:
    """
    Container for g_w values derived from an observer trajectory, in the
    exact shape consumed by :class:`ZoneDefinition` in
    ``multi_tso_coordinator.py``.

    Attributes per zone:
        g_w_der   : float
        g_w_pcc   : float
        g_w_gen   : float   (aka g_w_vgen in some configs)
        g_w_oltc  : float
    """
    per_zone: Dict[int, Dict[str, float]] = field(default_factory=dict)
    statistic: str = "percentile"   # "max" | "percentile" | "mean"
    percentile: float = 95.0


def derive_tuned_gw(
    observer: StabilityObserver,
    *,
    statistic: str = "percentile",
    percentile: float = 95.0,
) -> ObserverTunedGw:
    """
    Reduce the observer's trajectory to one ``g_w`` value per block per zone.

    Usage to rewrite the ``ZoneDefinition`` defaults for a re-run::

        tuned = derive_tuned_gw(observer, statistic="percentile",
                                percentile=95.0)
        for z, values in tuned.per_zone.items():
            zone_defs[z].g_w_der  = values["DER"]
            zone_defs[z].g_w_pcc  = values["PCC"]
            zone_defs[z].g_w_gen  = values["V_gen"]
            zone_defs[z].g_w_oltc = values["OLTC"]

    Notes
    -----
    ``statistic="max"`` is the worst-case guarantee (conservative).
    ``statistic="percentile"`` with 95th percentile is the typical
    thesis-robust choice.  ``statistic="mean"`` is for diagnostic
    comparison only; do not use for deployed tuning.
    """
    out = ObserverTunedGw(statistic=statistic, percentile=percentile)
    for z, traj in observer.trajectories.items():
        if not traj.records:
            continue
        gw_vec = traj.aggregate(statistic=statistic, percentile=percentile)
        zone_tuned = {}
        for k, name in enumerate(traj.layout.names):
            sl = traj.layout.block_slice(k)
            if sl.stop == sl.start:
                continue   # block has no actuators in this zone
            zone_tuned[name] = float(gw_vec[sl].mean())
        out.per_zone[z] = zone_tuned
    return out


# --------------------------------------------------------------------------- #
#  Worked example: the exact diff for 000_M_TSO_M_DSO.py
# --------------------------------------------------------------------------- #

INTEGRATION_DIFF = '''\
# ============================================================================
# Integration diff for experiments/000_M_TSO_M_DSO.py (wind_replace scenario)
# ============================================================================

# --- IMPORT (add near the top of the file, with the other controller imports)

from stability_integration_ieee39 import (
    attach_observer,
    write_observer_results_alongside_report,
    derive_tuned_gw,
)

# --- STEP 7b (immediately after coordinator is constructed, around line 1012)

    coordinator = MultiTSOCoordinator(
        zones=list(zone_defs.values()),
        net=net,
        verbose=verbose,
    )
    for z, ctrl in tso_controllers.items():
        coordinator.register_tso_controller(z, ctrl)

+   # Attach passive stability observer (runs alongside the controller).
+   observer = attach_observer(coordinator, zone_defs, config, verbose=verbose)

# --- MAIN LOOP (inside the tso-step branch, around line 1658)

            tso_outputs = coordinator.step(
                measurements,
                step,
                recompute_cross_sensitivities=refresh_H,
            )

+           # Passive stability recording — runs only when the coordinator
+           # actually refreshed the cross-sensitivity blocks.
+           if refresh_H:
+               observer.record(time_s=time_s)

# --- END OF SIMULATION (after the existing delayed stability report,
#     near line 1860)

            stab_result = _run_delayed_stability_analysis(...)

+   # Write observer trajectory report (full-simulation view).
+   write_observer_results_alongside_report(
+       observer, config.result_dir, verbose=verbose,
+   )
+
+   # Print the tuning recommendation for the next run.
+   tuned = derive_tuned_gw(observer, statistic="percentile", percentile=95.0)
+   if verbose >= 1 and tuned.per_zone:
+       print()
+       print("[9b] Stability observer tuning recommendation (p95):")
+       for z, vals in sorted(tuned.per_zone.items()):
+           print(f"  Zone {z}: "
+                 f"DER={vals[\'DER\']:.0f}, "
+                 f"PCC={vals[\'PCC\']:.0f}, "
+                 f"V_gen={vals[\'V_gen\']:.0f}, "
+                 f"OLTC={vals[\'OLTC\']:.0f}")
'''


if __name__ == "__main__":
    print(INTEGRATION_DIFF)
