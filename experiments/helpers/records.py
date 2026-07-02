# -*- coding: utf-8 -*-
"""
experiments/records.py
======================
Data-classes and helper lambdas shared across the cascade and multi-zone
simulations.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from analysis.stability_analysis import MultiZoneStabilityResult
from controller.dso_controller import DSOControllerConfig
from controller.tso_controller import TSOControllerConfig

# ---------------------------------------------------------------------------
#  Array-to-string helpers
# ---------------------------------------------------------------------------

A2S = lambda a: np.array2string(a, precision=2, suppress_small=True)
A3S = lambda a: np.array2string(a, precision=3, suppress_small=True)


# ---------------------------------------------------------------------------
#  Iteration log data-classes
# ---------------------------------------------------------------------------


@dataclass
class ContingencyEvent:
    """
    A scheduled network contingency.

    Parameters
    ----------
    minute : int
        Simulation minute at which the event occurs.
    element_type : str
        Pandapower element table: ``"line"``, ``"gen"``, ``"ext_grid"``,
        or ``"load"`` (for load connection / shedding).
    element_index : int
        Row index in the corresponding ``net.<element_type>`` table.

        For ``"load"`` events three addressing modes are supported by
        :func:`run.contingency.prepare_load_contingencies`:

        1. Pass ``element_index >= 0`` directly → trip / restore the
           existing ``net.load`` row at that index (no dormant load is
           created).  ``"connect"`` is rejected.
        2. Pass ``name=<str>`` → look up the unique row whose
           ``net.load["name"]`` matches.  ``"connect"`` is rejected.
        3. Legacy "create dormant + trip" pattern (default ``element_index = -1``
           and ``name = None``): supply ``bus`` + ``p_mw`` + ``q_mvar``; a
           dormant load is pre-created at the first ``"connect"`` event,
           subsequent ``"trip"`` / ``"restore"`` events at the same
           ``(bus, p_mw, q_mvar)`` triple reuse that load.
    action : str
        ``"trip"`` sets ``in_service = False``;
        ``"restore"`` / ``"connect"`` sets ``in_service = True``.
    name : str | None
        Optional load name for ``"load"`` events.  When set,
        :func:`prepare_load_contingencies` resolves ``element_index`` by
        exact match on ``net.load["name"]``.  Required to be unique.
    bus : int | None
        Bus index for ``"load"`` events (dormant-load mode only).
    p_mw : float
        Active power [MW] of the contingency load (dormant-load mode only).
    q_mvar : float
        Reactive power [Mvar] of the contingency load (dormant-load mode only).
    """

    minute: int
    element_type: str  # "line" | "gen" | "ext_grid" | "load"
    element_index: int = -1
    action: str = "trip"  # "trip" | "restore" | "setpoint_change" | "connect"
    new_setpoint: float = np.nan
    time_s: Optional[float] = None
    """Event time in seconds.  Overrides ``minute`` when set."""
    name: Optional[str] = None
    """Element name for ``"load"`` events; resolved to ``element_index``
    by :func:`prepare_load_contingencies`."""
    bus: Optional[int] = None
    """Bus index for load contingency events (dormant-load mode)."""
    p_mw: float = np.nan
    """Active power [MW] of contingency load (dormant-load mode)."""
    q_mvar: float = np.nan
    """Reactive power [Mvar] of contingency load (dormant-load mode)."""

    @property
    def effective_time_s(self) -> float:
        """Event time in seconds (``time_s`` if set, else ``minute * 60``)."""
        return self.time_s if self.time_s is not None else self.minute * 60.0


@dataclass
class IterationRecord:
    minute: int
    time_s: float = 0.0
    """Simulation time in seconds for this record."""
    tso_active: bool = False
    dso_active: bool = False
    # TSO optimisation variables [Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]
    tso_q_der_mvar: Optional[NDArray[np.float64]] = None
    tso_q_pcc_set_mvar: Optional[NDArray[np.float64]] = None
    tso_v_gen_pu: Optional[NDArray[np.float64]] = None
    tso_oltc_taps: Optional[NDArray[np.int64]] = None
    tso_shunt_states: Optional[NDArray[np.int64]] = None
    tso_objective: Optional[float] = None
    tso_solver_status: Optional[str] = None
    tso_solve_time_s: Optional[float] = None
    # DSO optimisation variables [Q_DER | s_OLTC | s_shunt]
    dso_q_der_mvar: Optional[NDArray[np.float64]] = None
    dso_oltc_taps: Optional[NDArray[np.int64]] = None
    dso_shunt_states: Optional[NDArray[np.int64]] = None
    dso_q_setpoint_mvar: Optional[NDArray[np.float64]] = None  # Q set by TSO
    dso_q_actual_mvar: Optional[NDArray[np.float64]] = (
        None  # actual Q at interface after PF
    )
    dso_objective: Optional[float] = None
    dso_solver_status: Optional[str] = None
    dso_solve_time_s: Optional[float] = None
    # Plant measurements after PF
    plant_tn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_dn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_tn_currents_ka: Optional[NDArray[np.float64]] = None  # TN line i_from_ka
    plant_dn_currents_ka: Optional[NDArray[np.float64]] = None  # DN line i_from_ka
    tso_q_gen_mvar: Optional[NDArray[np.float64]] = None  # synchronous gen Q output
    # Penalty terms (computed from plant measurements after PF)
    tso_v_penalty: Optional[float] = None  # g_v * sum((V - V_set)^2)
    dso_q_penalty: Optional[float] = None  # g_q * sum((Q - Q_set)^2)
    # Contingency events that fired this minute: list of (verbose_desc, short_label)
    contingency_events: Optional[List[tuple]] = None


@dataclass
class CascadeResult:
    """Container for all cascade simulation results."""

    log: List[IterationRecord]
    tso_config: TSOControllerConfig
    dso_config: DSOControllerConfig


# ---------------------------------------------------------------------------
#  Multi-TSO / Multi-DSO iteration record
# ---------------------------------------------------------------------------


@dataclass
class MultiTSOIterationRecord:
    """
    One timestep's worth of simulation data for the multi-zone TSO/DSO runner.

    Stores per-zone TSO outputs and plant measurements.
    DSO outputs are stored in the ``dso_*`` fields (indexed by dso_id string).
    """

    step:           int
    time_s:         float
    tso_active:     bool = False
    dso_active:     bool = False

    # Per-zone TSO outputs: zone_id -> array
    zone_q_der:         Dict[int, NDArray] = field(default_factory=dict)
    zone_q_pcc_set:     Dict[int, NDArray] = field(default_factory=dict)
    zone_v_gen:         Dict[int, NDArray] = field(default_factory=dict)
    zone_q_gen:         Dict[int, Any] = field(default_factory=dict)
    zone_p_gen:         Dict[int, NDArray] = field(default_factory=dict)
    zone_oltc_taps:     Dict[int, NDArray] = field(default_factory=dict)
    zone_tso_objective: Dict[int, Optional[float]] = field(default_factory=dict)
    zone_tso_status:    Dict[int, Optional[str]]   = field(default_factory=dict)
    zone_tso_solve_s:   Dict[int, Optional[float]] = field(default_factory=dict)

    # Plant voltages per zone (after PF)
    zone_v_min:  Dict[int, float] = field(default_factory=dict)
    zone_v_max:  Dict[int, float] = field(default_factory=dict)
    zone_v_mean: Dict[int, float] = field(default_factory=dict)
    zone_v_rms_err_pu: Dict[int, float] = field(default_factory=dict)
    """Per-zone spatial RMS of ``(V_i - v_setpoint_pu)`` over the zone's
    voltage-observed EHV buses (``zd.v_bus_indices``).  Added for the CIGRE
    per-zone voltage-tracking-error figure (005_CIGRE_MULTI).  Defaulted so
    older pickles load; the plotter falls back to ``|zone_v_mean - v_set|``
    when this dict is empty."""

    # Per-zone stability diagnostic from coordinator
    zone_contraction_lhs: Dict[int, float] = field(default_factory=dict)

    # DSO outputs
    dso_q_der:         Dict[str, NDArray] = field(default_factory=dict)
    dso_q_actual_mvar: Dict[str, Optional[float]] = field(default_factory=dict)
    dso_q_set_mvar:    Dict[str, Optional[float]] = field(default_factory=dict)
    dso_objective:     Dict[str, Optional[float]] = field(default_factory=dict)
    dso_status:        Dict[str, Optional[str]]   = field(default_factory=dict)

    # DSO network-group aggregates
    dso_group_q_der_mvar:     Dict[str, float] = field(default_factory=dict)
    dso_group_q_der_min_mvar: Dict[str, float] = field(default_factory=dict)
    dso_group_q_der_max_mvar: Dict[str, float] = field(default_factory=dict)
    dso_group_v_min_pu:       Dict[str, float] = field(default_factory=dict)
    dso_group_v_mean_pu:      Dict[str, float] = field(default_factory=dict)
    dso_group_v_max_pu:       Dict[str, float] = field(default_factory=dict)

    # Transformer-level DSO interface and OLTC data
    dso_trafo_q_set_mvar:    Dict[str, float] = field(default_factory=dict)
    dso_trafo_q_actual_mvar: Dict[str, float] = field(default_factory=dict)
    dso_trafo_tap_pos:       Dict[str, int] = field(default_factory=dict)

    # DSO-reported PCC Q-capability envelope, in absolute Mvar at the HV side
    # of the interface trafo (i.e. q_iface_now + reported delta).  Same key
    # convention as ``dso_trafo_q_set_mvar``.  Used by the cascade live plot
    # to overlay the achievable band behind the setpoint and actual traces.
    dso_trafo_q_cap_min_mvar: Dict[str, float] = field(default_factory=dict)
    dso_trafo_q_cap_max_mvar: Dict[str, float] = field(default_factory=dict)

    # Explicit grouping metadata for fail-fast plotting
    dso_trafo_group:      Dict[str, str] = field(default_factory=dict)
    dso_controller_group: Dict[str, str] = field(default_factory=dict)

    # Global multi-zone stability snapshot (computed periodically)
    stability_result: Optional[MultiZoneStabilityResult] = None

    # Load-balance aggregates (for live plot)
    total_load_p_mw:    float = 0.0
    total_load_q_mvar:  float = 0.0
    total_sgen_p_mw:    float = 0.0
    total_gen_p_mw:     float = 0.0
    total_gen_q_mvar:   float = 0.0
    residual_load_p_mw: float = 0.0

    # ── Live-plot observables (added for the three-figure rework) ───────────

    # Per-zone line loadings (max / mean / min loading %) over zd.line_indices.
    zone_line_loading_max_pct:  Dict[int, float] = field(default_factory=dict)
    zone_line_loading_mean_pct: Dict[int, float] = field(default_factory=dict)
    zone_line_loading_min_pct:  Dict[int, float] = field(default_factory=dict)

    # Per-zone active-power line loss [MW], ground truth (net.res_line.pl_mw)
    # summed over exactly zd.line_indices -- the SAME EHV line set a zone's
    # TSO loss objective (TSOControllerConfig.g_loss, current_line_indices)
    # targets, so this is the right ground-truth check for "did g_loss lower
    # this zone's own losses" (mirrors total_losses_mw's role at the whole-
    # network level).  Does NOT include the zone's PCC/OLTC transformers or
    # its DSO sub-networks' own HV lines -- those sit outside the objective's
    # current_line_indices.
    zone_losses_mw: Dict[int, float] = field(default_factory=dict)

    # Per-zone TSO DER active power (array, one entry per TSO DER).
    zone_tso_der_p_mw: Dict[int, NDArray] = field(default_factory=dict)

    # Per-zone aggregate P/Q balance terms (all in MW / Mvar).
    zone_balance_der_p_mw:            Dict[int, float] = field(default_factory=dict)
    zone_balance_der_q_mvar:          Dict[int, float] = field(default_factory=dict)
    zone_balance_gen_p_mw:            Dict[int, float] = field(default_factory=dict)
    zone_balance_gen_q_mvar:          Dict[int, float] = field(default_factory=dict)
    zone_balance_load_p_mw:           Dict[int, float] = field(default_factory=dict)
    zone_balance_load_q_mvar:         Dict[int, float] = field(default_factory=dict)
    zone_balance_tso_dso_p_out_mw:    Dict[int, float] = field(default_factory=dict)
    zone_balance_tso_dso_q_out_mvar:  Dict[int, float] = field(default_factory=dict)

    # Inter-zone tie-line Q flow.  Keyed by ordered pair (zone_i, zone_j)
    # with i < j, positive = Q leaves zone i into zone j, summed over
    # all physical boundary lines between the two zones.
    zone_tie_q_mvar: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # ── Horizontal TSO-TSO tie-coordination observables (two-loop ΔV_ref) ────
    # All keyed by tie_id (pandapower line index of the individual tie line);
    # the coordination fields are populated when config.enable_tie_coordination
    # is set (tie_q_mvar is populated unconditionally post-PF).  Feed the
    # TIE-COORDINATION live plot (config.live_plot_tie_coordination).
    tie_dvref:       Dict[int, float] = field(default_factory=dict)
    """Agreed boundary-voltage difference ΔV_ref [p.u.] per tie (outer loop)."""
    tie_dv_realized: Dict[int, float] = field(default_factory=dict)
    """Realised boundary-voltage difference V_i − V_j [p.u.] per tie."""
    tie_grad_i:      Dict[int, float] = field(default_factory=dict)
    """Zone-i boundary objective-gradient γ_i = ∇J_i·h_b/‖h_b‖² per tie."""
    tie_grad_j:      Dict[int, float] = field(default_factory=dict)
    """Zone-j boundary objective-gradient γ_j per tie."""
    tie_grad_combined: Dict[int, float] = field(default_factory=dict)
    """Combined gradient G = κ·γ_i − (1−κ)·γ_j = ∂(J_i+J_j)/∂ΔV_ref per tie."""
    zone_reserve_scarcity: Dict[int, float] = field(default_factory=dict)
    """Per-zone aggregate reactive-reserve scarcity in [0,1] (0 abundant, 1
    saturated) — the μ_i signal driving the reserve extension."""
    zone_reserve_headroom_cap_mvar: Dict[int, float] = field(default_factory=dict)
    """Per-zone remaining positive-Q injection headroom H_cap [Mvar]."""
    zone_reserve_headroom_ind_mvar: Dict[int, float] = field(default_factory=dict)
    """Per-zone remaining negative-Q / absorption headroom H_ind [Mvar]."""
    zone_reserve_headroom_min_mvar: Dict[int, float] = field(default_factory=dict)
    """Per-zone limiting directional headroom min(H_cap, H_ind) [Mvar]."""
    tie_v_i:     Dict[int, float] = field(default_factory=dict)
    """Realised boundary voltage [p.u.] at the zone-i endpoint per tie line."""
    tie_v_j:     Dict[int, float] = field(default_factory=dict)
    """Realised boundary voltage [p.u.] at the zone-j endpoint per tie line."""
    tie_q_mvar:  Dict[int, float] = field(default_factory=dict)
    """Per-tie-line reactive flow [Mvar] at the zone-i endpoint (into line)."""

    # Per-zone TSO shunt states (MSC/MSR tap positions; empty array if none).
    zone_tso_shunt_states: Dict[int, NDArray] = field(default_factory=dict)

    # Per-HV-group line loading stats and balance aggregates.
    dso_group_i_max_pct:    Dict[str, float] = field(default_factory=dict)
    dso_group_i_mean_pct:   Dict[str, float] = field(default_factory=dict)
    dso_group_i_min_pct:    Dict[str, float] = field(default_factory=dict)
    dso_group_der_p_mw:     Dict[str, float] = field(default_factory=dict)
    dso_group_load_p_mw:    Dict[str, float] = field(default_factory=dict)
    dso_group_load_q_mvar:  Dict[str, float] = field(default_factory=dict)

    # Per-interface-trafo actual active power (HV side, parallel to
    # existing dso_trafo_q_actual_mvar).
    dso_trafo_p_actual_mw: Dict[str, float] = field(default_factory=dict)

    # ── Comparison metrics (used by 002_M_TSO_M_DSO_COMPARE.py) ──────────────

    total_losses_mw: float = 0.0
    """Sum of res_line.pl_mw + res_trafo.pl_mw + res_trafo3w.pl_mw across the
    whole combined network.  Filled at every step from the converged PF."""

    gen_q_headroom_mvar: Dict[int, NDArray] = field(default_factory=dict)
    """Per-zone array, parallel to ``zone_q_gen``: ``q_max(g) - |q_actual(g)|``
    for each synchronous machine in that zone (positive = remaining
    capability, zero = saturated, negative = capability violated)."""

    gen_q_reserve: Dict[int, NDArray] = field(default_factory=dict)
    """Per-zone array, parallel to ``zone_q_gen``: normalised reactive-power
    reserve of each synchronous machine,
    ``min(q_max - q, q - q_min) / (q_max - q_min)`` from the Milano §12.2.1
    capability band at the current P / terminal V.  Ranges 0 (at a limit)
    to 0.5 (mid-band); NaN where the band has zero width.  Feeds the
    TRACKING ERRORS & RESERVES live plot (``live_plot_tracking``)."""

    tso_der_q_reserve: Dict[int, NDArray] = field(default_factory=dict)
    """Per-zone array, parallel to ``zone_q_der``: normalised reactive-power
    reserve of each TSO-connected DER, same definition as
    ``gen_q_reserve`` but using the VDE-AR-N-4120 / STATCOM capability band
    from ``ActuatorBounds.compute_der_q_bounds`` at the current P."""

    # ── Slack-Q saturation diagnostic (added 2026-05-02) ────────────────────────
    # Populated post-PF in run_multi_tso_dso() to expose slack saturation in
    # pickled logs.  Defaults keep older logs loadable.
    slack_p_mw: float = 0.0
    slack_q_mvar: float = 0.0
    slack_q_at_limit: bool = False

    # ── Voltage-stability reachability margin (added 2026-06-08) ────────────────
    # Populated post-PF in run_multi_tso_dso() when config.enable_reachability_guard
    # is set, by analysis.reachability.ReachabilityMonitor.check_step().  Records
    # the per-step nose-curve margin so the full trajectory is available even when
    # no violation occurs.  Defaults keep older logs loadable.
    reach_sigma_min_J: Optional[float] = None
    """Smallest singular value of the full power-flow Jacobian (nose proximity)."""
    reach_lambda_min_JR: Optional[float] = None
    """Minimum real eigenvalue of the reduced Q-V Jacobian; positive on the
    stable upper branch, non-positive at/beyond the saddle-node."""
    reach_critical_bus: Optional[int] = None
    """Pandapower bus index with the largest participation in the critical
    (minimum-eigenvalue) voltage-stability mode."""
