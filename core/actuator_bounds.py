"""
Actuator Bounds Module
======================

This module defines the ActuatorBounds class, which calculates operating-point-
dependent bounds for all actuators controlled by an OFO controller.

The bounds are used as input constraints in the MIQP optimisation problem.
DER reactive power bounds depend on the current active power output via the
capability curve defined in grid codes (e.g., VDE-AR-N 4120).

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
from numpy.typing import NDArray


class ActuatorBounds:
    """
    Calculator for operating-point-dependent actuator bounds.
    
    This class computes the lower and upper bounds for all actuators at the
    current operating point. For DERs, the reactive power bounds depend on
    the current active power output. For discrete actuators (OLTCs, shunts),
    the bounds are fixed.
    
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
        """
        self.der_indices = der_indices
        self.der_s_rated_mva = der_s_rated_mva
        self.der_p_max_mw = der_p_max_mw
        self.oltc_indices = oltc_indices
        self.oltc_tap_min = oltc_tap_min
        self.oltc_tap_max = oltc_tap_max
        self.shunt_indices = shunt_indices
        self.shunt_q_mvar = shunt_q_mvar
    
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
    
    @property
    def n_ders(self) -> int:
        """Return the number of controllable DERs."""
        return len(self.der_indices)
    
    @property
    def n_oltcs(self) -> int:
        """Return the number of controllable OLTCs."""
        return len(self.oltc_indices)
    
    @property
    def n_shunts(self) -> int:
        """Return the number of controllable shunts."""
        return len(self.shunt_indices)
