# -*- coding: utf-8 -*-
"""
run/records.py
==============
Data-classes and helper lambdas shared across the cascade simulation.

Moved here from the root ``run_cascade.py`` as part of the ``run/`` package
refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

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
    A scheduled network contingency (line trip or generator outage).

    Parameters
    ----------
    minute : int
        Simulation minute at which the event occurs.
    element_type : str
        Pandapower element table: ``"line"`` or ``"gen"``.
    element_index : int
        Row index in the corresponding ``net.<element_type>`` table.
    action : str
        ``"trip"`` sets ``in_service = False``;
        ``"restore"`` sets ``in_service = True``.
    """

    minute: int
    element_type: str  # "line" or "gen" or "ext_grid"
    element_index: int
    action: str = "trip"  # "trip" | "restore" | "setpoint_change"
    new_setpoint: float = np.nan


@dataclass
class IterationRecord:
    minute: int
    tso_active: bool
    dso_active: bool
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
