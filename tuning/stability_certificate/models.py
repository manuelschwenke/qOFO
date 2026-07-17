"""Data models for the offline OFO IQC/LMI certificate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CertificateStatus(str, Enum):
    """Machine-readable certificate outcome."""

    CERTIFIED = "certified"
    NOT_CERTIFIED = "not_certified"
    NOT_APPLICABLE = "not_applicable"
    SOLVER_ERROR = "solver_error"


@dataclass(frozen=True)
class LMIResult:
    """Result of one fixed-model LMI rate computation."""

    status: CertificateStatus
    rho: float | None
    spectral_radius: float | None = None
    solver: str | None = None
    max_lmi_eigenvalue: float | None = None
    p_condition: float | None = None
    multipliers: dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class LocalSectorCertificate:
    """Projected-gradient sector certificate for one local loop."""

    controller: str
    layer: str
    n_total: int
    n_active: int
    n_null: int
    m: float | None
    L: float | None
    condition_number: float | None
    current_rho_formula: float | None
    nominal_iqc: LMIResult
    robust_iqc: dict[str, LMIResult]
    stable_uniform_gw_scale_min: float | None
    optimal_uniform_gw_scale: float | None
    optimal_uniform_rho: float | None
    scale_interpretation: str
    scope: str


@dataclass(frozen=True)
class CoupledCertificate:
    """Certificate and diagnostics for the coupled continuous hierarchy."""

    dimension: int
    spectral_radius_full: float
    spectral_norm_full: float
    symmetry_defect: float
    potential_gradient_compatible: bool
    n_active: int
    n_neutral: int
    neutral_tolerance: float
    projected_full_state_iqc: LMIResult
    inactive_constraint_linear_lmi: LMIResult
    active_mode_linear_lmi: LMIResult
    scope: str


@dataclass(frozen=True)
class HierarchyCertificate:
    """Top-level offline result for one frozen cached-model snapshot."""

    generated_at: str
    config_factory: str
    alpha_convention: str
    operating_point: str
    current_parameters: dict[str, Any]
    local_continuous: list[LocalSectorCertificate]
    coupled_continuous: CoupledCertificate
    discrete_miqp: dict[str, Any]
    bo_guidance: dict[str, Any]
    assumptions: list[str]
    risks: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))
