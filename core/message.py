"""
Message Module
==============

This module defines message classes for communication between TSO and DSO
controllers in the cascaded OFO framework.

Two message types are defined:
    - SetpointMessage: TSO to DSO, contains reactive power setpoints
    - CapabilityMessage: DSO to TSO, contains reactive power capability bounds

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
        Sign convention: positive = Q flowing into HV bus from transformer.
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
        Minimum reactive power capability in Mvar for each interface.
        Sign convention: positive = Q flowing into HV bus from transformer.
    q_max_mvar : NDArray[np.float64]
        Maximum reactive power capability in Mvar for each interface.
        Sign convention: positive = Q flowing into HV bus from transformer.
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
