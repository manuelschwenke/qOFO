"""
Message Module
==============

This module defines message classes for communication between TSO and DSO
controllers in the cascaded OFO framework.

Three message types are defined:
    - SetpointMessage: TSO to DSO, contains reactive power setpoints
    - CapabilityMessage: DSO to TSO, contains reactive power capability bounds
    - ShuntDisturbanceMessage: TSO to DSO, signals a TSO-owned shunt step
      change at a tertiary bus inside the DSO's network so the DSO can
      apply a rank-1 Sherman–Morrison update to its cached Jacobian
      inverse without re-measuring or re-running pp.runpp.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
from numpy.typing import NDArray


class SetpointMessage:
    """
    Message from TSO controller to subordinate DSO controller.
    
    This message communicates the desired reactive power setpoints at the
    TSO-DSO interface transformers. The DSO controller shall track these
    setpoints whilst respecting local constraints.
    
    Attributes
    ----------
    source_controller_id : str
        Identifier of the sending controller (TSO).
    target_controller_id : str
        Identifier of the receiving controller (DSO).
    iteration : int
        OFO iteration index at which this message was generated.
    interface_transformer_indices : NDArray[np.int64]
        Pandapower transformer indices for the TSO-DSO interfaces.
    q_setpoints_mvar : NDArray[np.float64]
        Reactive power setpoints in Mvar for each interface transformer.
        Sign convention: pandapower load convention at HV port
        (positive = Q flowing from HV bus into transformer).
    """
    
    def __init__(
        self,
        source_controller_id: str,
        target_controller_id: str,
        iteration: int,
        interface_transformer_indices: NDArray[np.int64],
        q_setpoints_mvar: NDArray[np.float64],
    ) -> None:
        """
        Initialise a SetpointMessage.
        
        Parameters
        ----------
        source_controller_id : str
            Identifier of the sending controller.
        target_controller_id : str
            Identifier of the receiving controller.
        iteration : int
            OFO iteration index.
        interface_transformer_indices : NDArray[np.int64]
            Interface transformer indices.
        q_setpoints_mvar : NDArray[np.float64]
            Reactive power setpoints in Mvar.
        """
        self.source_controller_id = source_controller_id
        self.target_controller_id = target_controller_id
        self.iteration = iteration
        self.interface_transformer_indices = interface_transformer_indices
        self.q_setpoints_mvar = q_setpoints_mvar
    
    @property
    def n_interfaces(self) -> int:
        """Return the number of interface transformers in this message."""
        return len(self.interface_transformer_indices)


class ShuntDisturbanceMessage:
    """
    Message from TSO controller to subordinate DSO controller signalling
    that a TSO-owned shunt physically located inside the DSO's network
    has just changed step.

    The recipient DSO does **not** rebuild its Jacobian from a fresh
    ``pp.runpp`` call; instead it applies a rank-1 Sherman–Morrison
    update to its cached reduced Jacobian inverse for the susceptance
    change at the shunt bus.  The cached operating point (V, θ) is
    preserved.  The DSO's H cache is invalidated so the next
    ``step(measurement)`` call rebuilds H from the updated
    ``dV_dQ_reduced`` — refreshing the OLTC sensitivities of the
    affected 3-winding transformer with the correct shunt-coupling term.

    The DSO never sees the shunt as a *control variable*
    (``shunt_bus_indices`` stays ``[]``); the message only refreshes the
    DSO's cached model.

    Attributes
    ----------
    source_controller_id : str
        Identifier of the sending controller (TSO).
    target_controller_id : str
        Identifier of the receiving controller (DSO).
    iteration : int
        OFO iteration index at which this message was generated.
    shunt_bus_indices : NDArray[np.int64]
        Pandapower bus indices of the shunts that changed step.
    shunt_steps : NDArray[np.int64]
        New shunt states (post-switch), one per entry in
        ``shunt_bus_indices``.  Convention: ``-1`` capacitor on,
        ``0`` off, ``+1`` reactor on.
    shunt_q_steps_mvar : NDArray[np.float64]
        Rated reactive power per step at V = 1 pu [Mvar] for each
        shunt (used by the SMW update to compute ΔY_bb).
    """

    def __init__(
        self,
        source_controller_id: str,
        target_controller_id: str,
        iteration: int,
        shunt_bus_indices: NDArray[np.int64],
        shunt_steps: NDArray[np.int64],
        shunt_q_steps_mvar: NDArray[np.float64],
    ) -> None:
        """
        Initialise a ShuntDisturbanceMessage.

        Parameters
        ----------
        source_controller_id : str
            Identifier of the sending controller.
        target_controller_id : str
            Identifier of the receiving controller.
        iteration : int
            OFO iteration index.
        shunt_bus_indices : NDArray[np.int64]
            Pandapower bus indices of the shunts that changed step.
        shunt_steps : NDArray[np.int64]
            Post-switch states (e.g. ``{-1, 0, +1}`` for bipolar).
        shunt_q_steps_mvar : NDArray[np.float64]
            Rated Q per step at V = 1 pu [Mvar].
        """
        self.source_controller_id = source_controller_id
        self.target_controller_id = target_controller_id
        self.iteration = iteration
        self.shunt_bus_indices = np.asarray(shunt_bus_indices, dtype=np.int64)
        self.shunt_steps = np.asarray(shunt_steps, dtype=np.int64)
        self.shunt_q_steps_mvar = np.asarray(shunt_q_steps_mvar, dtype=np.float64)

    @property
    def n_shunts(self) -> int:
        """Return the number of shunts referenced by this message."""
        return len(self.shunt_bus_indices)


class CapabilityMessage:
    """
    Message from DSO controller to superordinate TSO controller.
    
    This message communicates the current reactive power capability bounds
    at the TSO-DSO interface. The bounds are operating-point-dependent and
    derived from DER capabilities and local network constraints.
    
    The TSO controller uses these bounds as constraints when determining
    the reactive power setpoints for the next iteration.
    
    Attributes
    ----------
    source_controller_id : str
        Identifier of the sending controller (DSO).
    target_controller_id : str
        Identifier of the receiving controller (TSO).
    iteration : int
        OFO iteration index at which this message was generated.
    interface_transformer_indices : NDArray[np.int64]
        Pandapower transformer indices for the TSO-DSO interfaces.
    q_min_mvar : NDArray[np.float64]
        Minimum achievable Q change at each interface in Mvar (delta from
        current operating point).
        Sign convention: pandapower load convention at HV port
        (positive = Q flowing from HV bus into transformer).
    q_max_mvar : NDArray[np.float64]
        Maximum achievable Q change at each interface in Mvar (delta from
        current operating point).
        Sign convention: pandapower load convention at HV port
        (positive = Q flowing from HV bus into transformer).
    """
    
    def __init__(
        self,
        source_controller_id: str,
        target_controller_id: str,
        iteration: int,
        interface_transformer_indices: NDArray[np.int64],
        q_min_mvar: NDArray[np.float64],
        q_max_mvar: NDArray[np.float64],
    ) -> None:
        """
        Initialise a CapabilityMessage.
        
        Parameters
        ----------
        source_controller_id : str
            Identifier of the sending controller.
        target_controller_id : str
            Identifier of the receiving controller.
        iteration : int
            OFO iteration index.
        interface_transformer_indices : NDArray[np.int64]
            Interface transformer indices.
        q_min_mvar : NDArray[np.float64]
            Minimum reactive power capabilities in Mvar.
        q_max_mvar : NDArray[np.float64]
            Maximum reactive power capabilities in Mvar.
        """
        self.source_controller_id = source_controller_id
        self.target_controller_id = target_controller_id
        self.iteration = iteration
        self.interface_transformer_indices = interface_transformer_indices
        self.q_min_mvar = q_min_mvar
        self.q_max_mvar = q_max_mvar
    
    @property
    def n_interfaces(self) -> int:
        """Return the number of interface transformers in this message."""
        return len(self.interface_transformer_indices)
