"""
Actuator Bounds Module
======================

This module defines the ActuatorBounds class, which calculates operating-point-
dependent bounds for all actuators controlled by an OFO controller.

The bounds are used as input constraints in the MIQP optimisation problem.
DER reactive power bounds depend on the current active power output via the
capability curve defined in grid codes (e.g., VDE-AR-N 4120).

Synchronous generator limits are modelled using the detailed capability
curve from Milano (2010), §12.2.1, comprising three thermal constraints:
    (i)   Stator current limit   –  p² + q² ≤ s_max²
    (ii)  Rotor current limit    –  p² + (q + v²/xd)² ≤ (v·i_f_max/xd)²
    (iii) Under-excitation limit –  q ≥ −q₀(v) + β·p_max

References
----------
[1] F. Milano, *Power System Modelling and Scripting*, Springer, 2010,
    Chapter 12, Eqs. (12.7)–(12.11).

Author: Manuel Schwenke
Date: 2025-02-05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════════
#  Synchronous Generator Capability Curve  (Milano §12.2.1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratorParameters:
    """
    Physical parameters of a synchronous generator for capability-curve
    computation.

    All electrical quantities are in *per-unit on the machine MVA base*
    unless noted otherwise.

    Attributes
    ----------
    s_rated_mva : float
        Rated apparent power [MVA].  Used to convert between p.u. and
        physical units.
    p_max_mw : float
        Maximum active power output [MW].  Equals the turbine rating.
    xd_pu : float
        Direct-axis synchronous reactance [p.u.].
        Typical: 1.0–1.8 for turbo-generators, 0.6–1.2 for salient-pole.
    i_f_max_pu : float
        Maximum field current [p.u.].
        Typical: 2.6–2.73 for turbo-generators (Milano eq. 12.10),
                 1.7–1.79 for salient-pole (Milano eq. 12.9).
    beta : float
        Under-excitation limit slope parameter [p.u./p.u.].
        Typical: 0.1–0.2 (Milano p. 293).
    q0_pu : float
        Under-excitation limit offset at nominal voltage [p.u.].
        Typical: ≈ 0.4 (Milano p. 293).
        Voltage-dependent: q₀(v) ≈ q0_pu · v² (proportional to V²).
    """
    s_rated_mva: float
    p_max_mw: float
    xd_pu: float = 1.2
    i_f_max_pu: float = 2.65
    beta: float = 0.15
    q0_pu: float = 0.4


def compute_generator_q_limits(
    params: GeneratorParameters,
    p_mw: float,
    v_pu: float = 1.0,
) -> tuple[float, float]:
    """
    Compute the reactive power limits of a synchronous generator at a
    given operating point using the detailed capability curve.

    The three constraints (all in p.u. on the machine base) are:

    1. **Stator current limit** (Milano eq. 12.7):
       p² + q² ≤ s_max²

    2. **Rotor current limit** (Milano eq. 12.8):
       p² + (q + v²/xd)² ≤ (v · i_f_max / xd)²

    3. **Under-excitation limit** (Milano eq. 12.11):
       q ≥ −q₀(v) + β · p_max

    Parameters
    ----------
    params : GeneratorParameters
        Machine parameters.
    p_mw : float
        Current active power output [MW].
    v_pu : float
        Terminal voltage magnitude [p.u.].  Default 1.0.

    Returns
    -------
    q_min_mvar : float
        Minimum reactive power (under-excited / absorbing) [Mvar].
    q_max_mvar : float
        Maximum reactive power (over-excited / injecting) [Mvar].
    """
    s_base = params.s_rated_mva
    if s_base <= 0:
        return 0.0, 0.0

    # Convert to per-unit on machine base
    p_pu = p_mw / s_base
    p_max_pu = params.p_max_mw / s_base
    s_max_pu = 1.0  # by definition of s_rated

    xd = params.xd_pu
    i_f_max = params.i_f_max_pu

    # ------------------------------------------------------------------
    # q_max: minimum of stator limit and rotor limit (both give upper Q)
    # ------------------------------------------------------------------

    # (1) Stator current limit:  q_max_stator = sqrt(s_max² − p²)
    disc_stator = s_max_pu ** 2 - p_pu ** 2
    if disc_stator > 0:
        q_max_stator = math.sqrt(disc_stator)
    else:
        q_max_stator = 0.0

    # (2) Rotor current limit:
    #     p² + (q + v²/xd)² ≤ (v·i_f_max/xd)²
    #     => q ≤ -v²/xd + sqrt((v·i_f_max/xd)² − p²)
    rotor_radius = v_pu * i_f_max / xd
    disc_rotor = rotor_radius ** 2 - p_pu ** 2
    if disc_rotor > 0:
        q_max_rotor = -v_pu ** 2 / xd + math.sqrt(disc_rotor)
    else:
        q_max_rotor = -v_pu ** 2 / xd  # degenerate: p exceeds rotor circle

    q_max_pu = min(q_max_stator, q_max_rotor)

    # ------------------------------------------------------------------
    # q_min: maximum of stator limit (lower) and under-excitation limit
    # ------------------------------------------------------------------

    # (1) Stator current limit (lower branch):  q_min_stator = -sqrt(s_max² − p²)
    q_min_stator = -q_max_stator  # symmetric for stator

    # (3) Under-excitation limit (Milano eq. 12.11):
    #     q ≥ -q₀(v) + β · p_max
    #     q₀(v) ≈ q0_pu · v²  (voltage-dependent offset)
    q0_v = params.q0_pu * v_pu ** 2
    q_min_ue = -q0_v + params.beta * p_max_pu

    q_min_pu = max(q_min_stator, q_min_ue)

    # Ensure q_min ≤ q_max
    if q_min_pu > q_max_pu:
        mid = 0.5 * (q_min_pu + q_max_pu)
        q_min_pu = mid
        q_max_pu = mid

    # Convert back to Mvar
    q_min_mvar = q_min_pu * s_base
    q_max_mvar = q_max_pu * s_base

    return q_min_mvar, q_max_mvar


class ActuatorBounds:
    """
    Calculator for operating-point-dependent actuator bounds.

    This class computes the lower and upper bounds for all actuators at the
    current operating point.  For DERs, the reactive power bounds depend on
    the current active power output.  For synchronous generators, the Q
    bounds depend on P and terminal voltage via the detailed capability
    curve (Milano §12.2.1).  For discrete actuators (OLTCs, shunts), the
    bounds are fixed.

    Attributes
    ----------
    der_indices : NDArray[np.int64]
        Indices of controllable DERs.
    der_s_rated_mva : NDArray[np.float64]
        Rated apparent power of each DER in MVA.
    der_p_max_mw : NDArray[np.float64]
        Maximum active power (installed capacity) of each DER in MW.
    oltc_indices : NDArray[np.int64]
        Indices of controllable OLTCs.
    oltc_tap_min : NDArray[np.int64]
        Minimum tap position for each OLTC.
    oltc_tap_max : NDArray[np.int64]
        Maximum tap position for each OLTC.
    shunt_indices : NDArray[np.int64]
        Indices of controllable shunts.
    shunt_q_mvar : NDArray[np.float64]
        Rated reactive power of each shunt in Mvar (positive for reactors).
    gen_params : list[GeneratorParameters] | None
        Per-generator physical parameters for capability-curve computation.
        ``None`` if no synchronous generators are modelled.
    """

    def __init__(
        self,
        der_indices: NDArray[np.int64],
        der_s_rated_mva: NDArray[np.float64],
        der_p_max_mw: NDArray[np.float64],
        oltc_indices: NDArray[np.int64],
        oltc_tap_min: NDArray[np.int64],
        oltc_tap_max: NDArray[np.int64],
        shunt_indices: NDArray[np.int64],
        shunt_q_mvar: NDArray[np.float64],
        gen_params: Optional[list[GeneratorParameters]] = None,
    ) -> None:
        """
        Initialise ActuatorBounds with static actuator parameters.

        Parameters
        ----------
        der_indices : NDArray[np.int64]
            Indices of controllable DERs.
        der_s_rated_mva : NDArray[np.float64]
            Rated apparent power of each DER in MVA.
        der_p_max_mw : NDArray[np.float64]
            Maximum active power of each DER in MW.
        oltc_indices : NDArray[np.int64]
            Indices of controllable OLTCs.
        oltc_tap_min : NDArray[np.int64]
            Minimum tap position for each OLTC.
        oltc_tap_max : NDArray[np.int64]
            Maximum tap position for each OLTC.
        shunt_indices : NDArray[np.int64]
            Indices of controllable shunts.
        shunt_q_mvar : NDArray[np.float64]
            Rated reactive power of each shunt in Mvar.
        gen_params : list[GeneratorParameters] | None, optional
            Per-generator physical parameters for the detailed capability
            curve.  If ``None`` (default), generator Q bounds are not
            available and :meth:`compute_gen_q_bounds` will raise.
        """
        self.der_indices = der_indices
        self.der_s_rated_mva = der_s_rated_mva
        self.der_p_max_mw = der_p_max_mw
        self.oltc_indices = oltc_indices
        self.oltc_tap_min = oltc_tap_min
        self.oltc_tap_max = oltc_tap_max
        self.shunt_indices = shunt_indices
        self.shunt_q_mvar = shunt_q_mvar
        self.gen_params = gen_params
    
    def compute_der_q_bounds(
        self,
        der_p_current_mw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute DER reactive power bounds based on current active power output.
        
        The bounds are derived from the capability curve according to
        VDE-AR-N 4120 for HV-connected DERs. The reactive power range
        depends on the ratio P/P_installed.
        
        Parameters
        ----------
        der_p_current_mw : NDArray[np.float64]
            Current active power output of each DER in MW.
        
        Returns
        -------
        q_min_mvar : NDArray[np.float64]
            Minimum reactive power for each DER in Mvar (underexcited/consuming).
        q_max_mvar : NDArray[np.float64]
            Maximum reactive power for each DER in Mvar (overexcited/producing).
        
        Notes
        -----
        Sign convention:
            - Positive Q: overexcited (producing/injecting reactive power)
            - Negative Q: underexcited (consuming/absorbing reactive power)
        
        The capability curve is approximated as per VDE-AR-N 4120 Figure 1:
            - At P/P_inst >= 0.2: Q/P_inst in [-0.33, +0.33]
            - At P/P_inst < 0.2: Q range reduced proportionally
        """
        n_der = len(self.der_indices)
        q_min_mvar = np.zeros(n_der)
        q_max_mvar = np.zeros(n_der)
        
        for i in range(n_der):
            p_current = der_p_current_mw[i]
            p_max = self.der_p_max_mw[i]
            s_rated = self.der_s_rated_mva[i]
            
            # Compute P ratio (normalised to installed capacity)
            if p_max > 0.0:
                p_ratio = p_current / p_max
            else:
                p_ratio = 0.0
            
            # Compute Q capability based on VDE-AR-N 4120 curve
            q_min, q_max = self._compute_single_der_q_capability(
                p_ratio=p_ratio,
                s_rated_mva=s_rated,
            )
            
            q_min_mvar[i] = q_min
            q_max_mvar[i] = q_max
        
        return q_min_mvar, q_max_mvar
    
    def _compute_single_der_q_capability(
        self,
        p_ratio: float,
        s_rated_mva: float,
    ) -> tuple[float, float]:
        """
        Compute Q capability for a single DER based on its P ratio.
        
        This implements a simplified version of the VDE-AR-N 4120 capability
        curve for HV-connected generation units.
        
        Parameters
        ----------
        p_ratio : float
            Ratio of current active power to installed capacity (P/P_inst).
        s_rated_mva : float
            Rated apparent power of the DER in MVA.
        
        Returns
        -------
        q_min : float
            Minimum reactive power in Mvar.
        q_max : float
            Maximum reactive power in Mvar.
        """
        # VDE-AR-N 4120 specifies Q/P_inst ratio limits
        # Simplified: constant Q capability factor of 0.33 for P >= 0.2
        # Below P = 0.2, capability reduces (here: linear reduction)
        
        q_capability_factor = 0.33  # Q/P_inst at full capability
        p_threshold = 0.2  # Below this, capability is reduced
        
        if p_ratio >= p_threshold:
            # Full Q capability available
            q_factor = q_capability_factor
        else:
            # Reduced capability at low active power
            # Linear interpolation from 0 at P=0 to full at P=0.2
            q_factor = q_capability_factor * (p_ratio / p_threshold)
        
        # Q limits based on rated apparent power
        # Note: This is a simplification; actual curve is more complex
        q_max = q_factor * s_rated_mva  # Overexcited limit
        q_min = -q_factor * s_rated_mva  # Underexcited limit
        
        return q_min, q_max
    
    def get_oltc_tap_bounds(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Get the fixed tap position bounds for all OLTCs.
        
        Returns
        -------
        tap_min : NDArray[np.int64]
            Minimum tap position for each OLTC.
        tap_max : NDArray[np.int64]
            Maximum tap position for each OLTC.
        """
        return self.oltc_tap_min.copy(), self.oltc_tap_max.copy()
    
    def get_shunt_state_bounds(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Get the fixed state bounds for all shunts.
        
        Shunts have three possible states:
            -1: Capacitor (reactive power production)
             0: Off
            +1: Reactor (reactive power consumption)
        
        Returns
        -------
        state_min : NDArray[np.int64]
            Minimum state for each shunt (always -1).
        state_max : NDArray[np.int64]
            Maximum state for each shunt (always +1).
        """
        n_shunts = len(self.shunt_indices)
        state_min = np.full(n_shunts, -1, dtype=np.int64)
        state_max = np.full(n_shunts, +1, dtype=np.int64)
        return state_min, state_max
    
    def compute_gen_q_bounds(
        self,
        gen_p_mw: NDArray[np.float64],
        gen_v_pu: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute synchronous generator Q bounds from the detailed
        capability curve (Milano §12.2.1).

        The bounds depend on the current active power dispatch *and*
        the terminal voltage, accounting for stator current, rotor
        current, and under-excitation thermal limits.

        Parameters
        ----------
        gen_p_mw : NDArray[np.float64]
            Current active power output of each generator [MW].
        gen_v_pu : NDArray[np.float64]
            Terminal voltage magnitude of each generator [p.u.].

        Returns
        -------
        q_min_mvar : NDArray[np.float64]
            Minimum reactive power for each generator [Mvar].
        q_max_mvar : NDArray[np.float64]
            Maximum reactive power for each generator [Mvar].

        Raises
        ------
        RuntimeError
            If ``gen_params`` was not provided at construction.
        """
        if self.gen_params is None:
            raise RuntimeError(
                "Generator parameters not configured. "
                "Pass gen_params to ActuatorBounds constructor."
            )
        n_gen = len(self.gen_params)
        q_min = np.zeros(n_gen, dtype=np.float64)
        q_max = np.zeros(n_gen, dtype=np.float64)
        for i, gp in enumerate(self.gen_params):
            q_min[i], q_max[i] = compute_generator_q_limits(
                gp, p_mw=gen_p_mw[i], v_pu=gen_v_pu[i],
            )
        return q_min, q_max

    @property
    def n_ders(self) -> int:
        """Return the number of controllable DERs."""
        return len(self.der_indices)

    @property
    def n_gens(self) -> int:
        """Return the number of synchronous generators."""
        return len(self.gen_params) if self.gen_params is not None else 0

    @property
    def n_oltcs(self) -> int:
        """Return the number of controllable OLTCs."""
        return len(self.oltc_indices)

    @property
    def n_shunts(self) -> int:
        """Return the number of controllable shunts."""
        return len(self.shunt_indices)
