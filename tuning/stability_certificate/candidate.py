"""Fast per-candidate curvature reconstruction and LMI evaluation."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from configs.config import MultiTSOConfig

from .iqc import _fixed_rho_sector
from .linear import _fixed_rho_linear, active_invariant_block
from .snapshot import (
    CONTINUOUS_BO_FIELDS,
    CachedCurvatureSnapshot,
    rebuild_stability_result,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    params: dict[str, float]
    fixed_g_w_gen: float
    objective: float
    dynamic_rate_cost: float
    log_distance_cost: float
    local_rho: dict[str, float]
    local_lmi_certified: dict[str, bool]
    coupled_active_rho: float
    coupled_active_lmi_certified: bool
    n_coupled_neutral: int
    c3_gamma: float
    c3_certified: bool
    all_candidate_lmis_certified: bool
    numerical_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_active_eigenvalues(values: np.ndarray) -> tuple[np.ndarray, bool]:
    eigs = np.asarray(values, dtype=float)
    if eigs.size == 0:
        return eigs, True
    tolerance = max(1e-10, 1e-9 * max(float(np.max(np.abs(eigs))), 1.0))
    has_negative = bool(np.any(eigs < -tolerance))
    return eigs[eigs > tolerance], not has_negative


def _sector_candidate_check(values: np.ndarray) -> tuple[float, bool, str | None]:
    active, psd = _positive_active_eigenvalues(values)
    if active.size == 0 or not psd:
        return 1.0, False, "No positive-semidefinite active curvature sector."
    m = float(np.min(active))
    L = float(np.max(active))
    rho = max(abs(1.0 - m), abs(1.0 - L))
    if rho >= 1.0:
        return rho, False, "Analytical active-mode rate is not below one."
    rho_test = 0.5 * (rho + 1.0)
    feasible, solver, residual, _ = _fixed_rho_sector(
        m,
        L,
        rho_test,
        delta=0.0,
    )
    note = None
    if not feasible:
        note = (
            f"Sector LMI failed validation at rho={rho_test:.8g}; "
            f"solver={solver}, residual={residual}."
        )
    return rho, bool(feasible), note


def _rate_cost(rho: float) -> float:
    return -math.log10(max(1.0 - min(rho, 1.0 - 1e-14), 1e-14))


def evaluate_candidate(
    snapshot: CachedCurvatureSnapshot,
    baseline: MultiTSOConfig,
    params: dict[str, float],
    *,
    closeness_weight: float = 0.05,
) -> CandidateEvaluation:
    """Rebuild M(G_w), rerun fixed-rate LMIs, and score one BO candidate."""

    if set(params) != set(CONTINUOUS_BO_FIELDS):
        raise ValueError(
            f"candidate keys must be exactly {sorted(CONTINUOUS_BO_FIELDS)}"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in params.values()):
        raise ValueError("candidate G_w values must be finite and positive")

    candidate = dataclasses.replace(
        baseline,
        **{name: float(value) for name, value in params.items()},
    )
    if candidate.g_w_gen != baseline.g_w_gen:
        raise AssertionError("g_w_gen changed in a stability BO candidate")

    result = rebuild_stability_result(snapshot, candidate)
    M = np.asarray(result.M_full_c, dtype=float)
    local_rho: dict[str, float] = {}
    local_certified: dict[str, bool] = {}
    notes: list[str] = []

    offset = 0
    for zone in result.zones:
        n_continuous = int(zone.n_continuous)
        block = M[offset : offset + n_continuous, offset : offset + n_continuous]
        symmetric = 0.5 * (block + block.T)
        rho, certified, note = _sector_candidate_check(
            np.linalg.eigvalsh(symmetric)
        )
        name = f"TSO zone {zone.zone_id}"
        local_rho[name] = rho
        local_certified[name] = certified
        if note:
            notes.append(f"{name}: {note}")
        offset += n_continuous

    for dso in result.c1_dso:
        name = str(dso.dso_id)
        rho, certified, note = _sector_candidate_check(
            np.asarray(dso.Phi_c_eigenvalues, dtype=float)
        )
        local_rho[name] = rho
        local_certified[name] = certified
        if note:
            notes.append(f"{name}: {note}")

    A = np.eye(M.shape[0]) - M
    active_A, n_neutral, _ = active_invariant_block(A)
    if active_A.size:
        coupled_rho = float(np.max(np.abs(np.linalg.eigvals(active_A))))
    else:
        coupled_rho = 0.0
    coupled_certified = False
    if active_A.size and coupled_rho < 1.0:
        rho_test = 0.5 * (coupled_rho + 1.0)
        feasible, solver, residual, _ = _fixed_rho_linear(active_A, rho_test)
        coupled_certified = bool(feasible)
        if not feasible:
            notes.append(
                "Coupled active LMI failed validation at "
                f"rho={rho_test:.8g}; solver={solver}, residual={residual}."
            )
    elif active_A.size:
        notes.append("Coupled active spectral radius is not below one.")

    rates = list(local_rho.values())
    if active_A.size:
        rates.append(coupled_rho)
    rate_costs = [_rate_cost(rho) for rho in rates]
    dynamic_cost = (
        0.5 * max(rate_costs) + 0.5 * float(np.mean(rate_costs))
        if rate_costs
        else 100.0
    )
    log_ratios = [
        math.log10(float(params[name]) / float(getattr(baseline, name)))
        for name in CONTINUOUS_BO_FIELDS
    ]
    distance_cost = float(np.mean(np.square(log_ratios)))
    all_lmis = all(local_certified.values()) and coupled_certified
    c3_certified = bool(result.c3_discrete.stable)
    objective = dynamic_cost + closeness_weight * distance_cost
    if not all_lmis:
        objective += 100.0
    if not c3_certified:
        objective += 100.0 + 10.0 * max(
            0.0,
            float(result.c3_discrete.Gamma_spectral_radius) - 1.0,
        )

    return CandidateEvaluation(
        params={name: float(params[name]) for name in CONTINUOUS_BO_FIELDS},
        fixed_g_w_gen=float(candidate.g_w_gen),
        objective=float(objective),
        dynamic_rate_cost=float(dynamic_cost),
        log_distance_cost=float(distance_cost),
        local_rho=local_rho,
        local_lmi_certified=local_certified,
        coupled_active_rho=coupled_rho,
        coupled_active_lmi_certified=coupled_certified,
        n_coupled_neutral=n_neutral,
        c3_gamma=float(result.c3_discrete.Gamma_spectral_radius),
        c3_certified=c3_certified,
        all_candidate_lmis_certified=all_lmis and c3_certified,
        numerical_notes=tuple(notes),
    )
