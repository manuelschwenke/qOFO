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
from typing import Any, Dict, List, Optional

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
        For ``"load"`` events this is filled automatically by
        :func:`run.contingency.prepare_load_contingencies`; pass ``-1``.
    action : str
        ``"trip"`` sets ``in_service = False``;
        ``"restore"`` / ``"connect"`` sets ``in_service = True``.
    bus : int | None
        Bus index for ``"load"`` events (ignored for other types).
    p_mw : float
        Active power [MW] of the contingency load (``"load"`` only).
    q_mvar : float
        Reactive power [Mvar] of the contingency load (``"load"`` only).
    """

    minute: int
    element_type: str  # "line" | "gen" | "ext_grid" | "load"
    element_index: int = -1
    action: str = "trip"  # "trip" | "restore" | "setpoint_change" | "connect"
    new_setpoint: float = np.nan
    time_s: Optional[float] = None
    """Event time in seconds.  Overrides ``minute`` when set."""
    bus: Optional[int] = None
    """Bus index for load contingency events."""
    p_mw: float = np.nan
    """Active power [MW] of contingency load."""
    q_mvar: float = np.nan
    """Reactive power [Mvar] of contingency load."""

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
