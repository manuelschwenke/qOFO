"""
IEEE 39-Bus New England Test Network
=====================================

Build the IEEE 39-bus network with optional scenario modifications and
110 kV HV sub-network attachment.

Public API
----------
build_ieee39_net
    Build the base IEEE 39-bus network with a selected scenario.
add_hv_networks
    Attach TUDA-style 110 kV HV sub-networks to the 345 kV grid.
IEEE39NetworkMeta
    Immutable index catalogue for the IEEE 39-bus network.
HVNetworkInfo
    Tracking information for one attached 110 kV sub-network.
remove_generators
    Remove synchronous generators and associated machine transformers.
"""

from network.ieee39.meta import IEEE39NetworkMeta, HVNetworkInfo
from network.ieee39.helpers import remove_generators
from network.ieee39.build import build_ieee39_net
from network.ieee39.hv_networks import add_hv_networks

__all__ = [
    "build_ieee39_net",
    "add_hv_networks",
    "IEEE39NetworkMeta",
    "HVNetworkInfo",
    "remove_generators",
]
