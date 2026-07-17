"""
Network Module
==============

Provides the benchmark network builder and the TSO–DSO split function.

Classes
-------
NetworkMetadata
    Structured metadata describing the elements of the combined network.
CouplerPowerFlow
    Converged power flow at a single 3-winding coupler transformer.
SplitResult
    Result container for the TN/DN network split.

Functions
---------
build_tuda_net
    Build the combined 380/110/20 kV TSO–DSO benchmark network.
split_network
    Split the combined network into separate TN and DN models.
validate_split
    Assert that the split reproduces the combined operating point.
"""

from network.build_tuda_net import build_tuda_net, NetworkMetadata
from network.split_tn_dn_net import (
    split_network,
    validate_split,
    CouplerPowerFlow,
    SplitResult,
)

__all__ = [
    "build_tuda_net",
    "NetworkMetadata",
    "split_network",
    "validate_split",
    "CouplerPowerFlow",
    "SplitResult",
]
