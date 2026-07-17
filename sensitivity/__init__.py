"""
Sensitivity Module
==================

This module provides Jacobian-based sensitivity calculations for the OFO
controllers.

Functions
---------
compute_voltage_sensitivity
    Compute ∂V/∂Q sensitivity matrix.
compute_transformer_q_sensitivity
    Compute ∂Q_tr/∂Q_DER sensitivity matrix.
compute_oltc_voltage_sensitivity
    Compute ∂V/∂tap sensitivity.
compute_oltc_q_sensitivity
    Compute ∂Q_tr/∂tap sensitivity.
compute_branch_current_sensitivity
    Compute ∂I/∂Q sensitivity matrix.
"""

# Placeholder imports - to be implemented
# from sensitivity.jacobian import (
#     compute_voltage_sensitivity,
#     compute_transformer_q_sensitivity,
#     compute_oltc_voltage_sensitivity,
#     compute_oltc_q_sensitivity,
#     compute_branch_current_sensitivity,
# )

from sensitivity.sensitivity_updater import SensitivityUpdater
from sensitivity.network_reduction import (
    DSOLocalNetResult,
    TSOLocalNetResult,
    build_dso_local_net,
    build_tso_local_net,
)

__all__ = [
    "SensitivityUpdater",
    "build_tso_local_net",
    "build_dso_local_net",
    "TSOLocalNetResult",
    "DSOLocalNetResult",
    # "compute_voltage_sensitivity",
    # "compute_transformer_q_sensitivity",
    # "compute_oltc_voltage_sensitivity",
    # "compute_oltc_q_sensitivity",
    # "compute_branch_current_sensitivity",
]
