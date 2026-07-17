"""Assemble IQC/LMI diagnostics from the actual multi-system OFO setup."""

from __future__ import annotations

import dataclasses
import importlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.stability_analysis import MultiZoneStabilityResult
from configs.config import MultiTSOConfig
from tuning.ceilings import _run_one_step_stability

from .iqc import robust_rate_sweep, sector_rate_certificate
from .linear import (
    active_invariant_block,
    linear_rate_certificate,
    projected_linear_rate_certificate,
)
from .models import (
    CertificateStatus,
    CoupledCertificate,
    HierarchyCertificate,
    LMIResult,
    LocalSectorCertificate,
)


DEFAULT_CONFIG_FACTORY = "experiments.run_multi_system_ofo:make_config"
DEFAULT_DELTAS = (0.01, 0.05, 0.10)


def load_config_factory(spec: str = DEFAULT_CONFIG_FACTORY) -> MultiTSOConfig:
    """Load ``module:callable`` and create the experiment configuration."""

    if ":" not in spec:
        raise ValueError("config factory must use the form 'module:callable'")
    module_name, callable_name = spec.split(":", 1)
    factory: Callable[[], Any] = getattr(importlib.import_module(module_name), callable_name)
    config = factory()
    if not isinstance(config, MultiTSOConfig):
        raise TypeError(f"{spec} returned {type(config).__name__}, not MultiTSOConfig")
    return config


def _active_eigenvalues(eigenvalues: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(eigenvalues, dtype=float)
    if values.size == 0:
        return values, 0
    tolerance = max(1e-10, 1e-9 * max(float(np.max(np.abs(values))), 1.0))
    active = values[values > tolerance]
    return active, int(values.size - active.size)


def _scale_interpretation(scale: float | None) -> str:
    if scale is None or not np.isfinite(scale):
        return "not available"
    if 0.5 <= scale <= 2.0:
        return "close to the current tuning (within a factor of two)"
    if 0.1 <= scale <= 10.0:
        return "same order of magnitude as the current tuning"
    return "far from the current tuning (more than one order of magnitude)"


def _local_certificate(
    name: str,
    layer: str,
    eigenvalues: np.ndarray,
    *,
    n_total: int,
    inherited_filter: bool,
    deltas: tuple[float, ...],
) -> LocalSectorCertificate:
    active, n_null_local = _active_eigenvalues(eigenvalues)
    n_null = max(n_total - int(active.size), n_null_local)
    if active.size == 0:
        unavailable = LMIResult(
            CertificateStatus.NOT_APPLICABLE,
            None,
            note="No strictly positive curvature mode was available.",
        )
        return LocalSectorCertificate(
            controller=name,
            layer=layer,
            n_total=n_total,
            n_active=0,
            n_null=n_total,
            m=None,
            L=None,
            condition_number=None,
            current_rho_formula=None,
            nominal_iqc=unavailable,
            robust_iqc={},
            stable_uniform_gw_scale_min=None,
            optimal_uniform_gw_scale=None,
            optimal_uniform_rho=None,
            scale_interpretation="not available",
            scope="No active-mode certificate could be formed.",
        )

    m, L = float(np.min(active)), float(np.max(active))
    nominal = sector_rate_certificate(m, L)
    robust = robust_rate_sweep(m, L, deltas)
    scale_opt = 0.5 * (m + L)
    scope_bits = [
        "Full local projected-gradient state" if n_null == 0 else (
            "Active curvature subspace only; neutral directions prevent a "
            "full-state linear-rate claim"
        )
    ]
    if inherited_filter:
        scope_bits.append(
            "DSO eigenvalues inherit the upstream stability analyser's null-mode filter"
        )
    if n_null:
        conditional_note = (
            "Conditional active-curvature-subspace certificate; neutral directions are excluded."
        )
        nominal = dataclasses.replace(nominal, note=conditional_note)
        robust = {
            delta: dataclasses.replace(value, note=conditional_note)
            for delta, value in robust.items()
        }
    return LocalSectorCertificate(
        controller=name,
        layer=layer,
        n_total=n_total,
        n_active=int(active.size),
        n_null=n_null,
        m=m,
        L=L,
        condition_number=L / m,
        current_rho_formula=max(abs(1.0 - m), abs(1.0 - L)),
        nominal_iqc=nominal,
        robust_iqc=robust,
        stable_uniform_gw_scale_min=0.5 * L,
        optimal_uniform_gw_scale=scale_opt,
        optimal_uniform_rho=(L - m) / (L + m),
        scale_interpretation=_scale_interpretation(scale_opt),
        scope="; ".join(scope_bits) + ".",
    )


def certificate_from_result(
    result: MultiZoneStabilityResult,
    config: MultiTSOConfig,
    *,
    config_factory: str = DEFAULT_CONFIG_FACTORY,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
) -> HierarchyCertificate:
    """Build the certificate from the frozen cached-model snapshot."""

    M = np.asarray(result.M_full_c, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("stability result does not contain a square M_full_c")
    A = np.eye(M.shape[0]) - M

    local: list[LocalSectorCertificate] = []
    offset = 0
    for zone in result.zones:
        n = int(zone.n_continuous)
        block = 0.5 * (M[offset : offset + n, offset : offset + n] + M[offset : offset + n, offset : offset + n].T)
        local.append(
            _local_certificate(
                f"TSO zone {zone.zone_id}",
                "TSO",
                np.linalg.eigvalsh(block) if n else np.array([]),
                n_total=n,
                inherited_filter=False,
                deltas=deltas,
            )
        )
        offset += n
    if offset != M.shape[0]:
        raise ValueError(
            f"zone continuous dimensions sum to {offset}, but M_full_c is {M.shape[0]}"
        )

    for dso in result.c1_dso:
        local.append(
            _local_certificate(
                str(dso.dso_id),
                "DSO",
                np.asarray(dso.Phi_c_eigenvalues, dtype=float),
                n_total=int(dso.n_continuous),
                inherited_filter=True,
                deltas=deltas,
            )
        )

    active_A, n_neutral, neutral_tolerance = active_invariant_block(A)
    active_lmi = linear_rate_certificate(active_A)
    if n_neutral:
        active_lmi = dataclasses.replace(
            active_lmi,
            note=(
                active_lmi.note
                + " Conditional diagnostic on the invariant non-neutral Schur subspace; "
                "it is not a full projected-map certificate."
            ),
        )
    symmetry_defect = float(
        np.linalg.norm(M - M.T, ord="fro") / max(np.linalg.norm(M, ord="fro"), 1e-15)
    )
    symmetric_part_min = float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T))))
    coupled = CoupledCertificate(
        dimension=M.shape[0],
        spectral_radius_full=float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0,
        spectral_norm_full=float(np.linalg.norm(A, ord=2)) if A.size else 0.0,
        symmetry_defect=symmetry_defect,
        potential_gradient_compatible=(
            symmetry_defect <= 1e-8 and symmetric_part_min >= -1e-10
        ),
        n_active=active_A.shape[0],
        n_neutral=n_neutral,
        neutral_tolerance=neutral_tolerance,
        projected_full_state_iqc=projected_linear_rate_certificate(A),
        inactive_constraint_linear_lmi=linear_rate_certificate(A),
        active_mode_linear_lmi=active_lmi,
        scope=(
            "Frozen cached sensitivities and frozen integer actuators. The projection IQC "
            "additionally assumes one fixed convex continuous feasible set in the G_w metric."
        ),
    )

    c3 = result.c3_discrete
    discrete_count = int(sum(zone.n_discrete for zone in result.zones))
    shunt_note = (
        "The configured TSO shunts use the separate hysteretic integrator, so any shunt "
        "entries assembled by the legacy C3 analyser are not an IQC/MIQP certificate for "
        "that integrator."
        if getattr(config, "shunt_dispatch", "miqp") == "integrator"
        else "Configured TSO shunts are dispatched in the MIQP."
    )
    legacy_c3_status = (
        CertificateStatus.CERTIFIED.value
        if bool(c3.stable)
        else CertificateStatus.NOT_CERTIFIED.value
    )
    discrete = {
        "paper_iqc_applicable": False,
        "status": CertificateStatus.NOT_APPLICABLE.value,
        "legacy_c3_status": legacy_c3_status,
        "method": "project-specific discrete small-gain check, not the Lessard IQC",
        "n_tso_discrete_variables": discrete_count,
        "gamma_spectral_radius": float(c3.Gamma_spectral_radius),
        "gamma_row_sums": np.asarray(c3.Gamma_row_sums, dtype=float).tolist(),
        "all_reported_gw_margins_nonnegative": all(
            margin >= 0.0
            for per_zone in c3.g_margin.values()
            for margin in per_zone.values()
        ),
        "note": shunt_note,
    }

    parameter_names = (
        "g_v",
        "g_q",
        "dso_g_v",
        "g_w_der",
        "g_w_pcc",
        "g_w_gen",
        "g_w_tso_oltc",
        "g_w_tso_shunt",
        "g_w_dso_der",
        "g_w_dso_oltc",
        "shunt_int_g_w",
    )
    current_parameters = {
        name: getattr(config, name) for name in parameter_names if hasattr(config, name)
    }
    scales = {
        item.controller: item.optimal_uniform_gw_scale
        for item in local
        if item.optimal_uniform_gw_scale is not None
    }
    full_status = coupled.projected_full_state_iqc.status
    bo_guidance = {
        "ready_as_hard_stability_constraint": full_status == CertificateStatus.CERTIFIED,
        "recommended_use": (
            "Use the full projected-map LMI as a hard BO feasibility constraint and the "
            "certified rho as a secondary objective."
            if full_status == CertificateStatus.CERTIFIED
            else "Use active-mode rates and uniform G_w scale factors as BO priors/diagnostics, "
            "not as a hard proof of full hierarchical stability."
        ),
        "local_uniform_gw_scale_factors": scales,
        "warning": (
            "A local uniform scale multiplies every continuous G_w entry in that controller; "
            "it is not a per-actuator-class optimum. Per-class tuning remains a numerical BO problem."
        ),
    }

    return HierarchyCertificate(
        generated_at=datetime.now(timezone.utc).isoformat(),
        config_factory=config_factory,
        alpha_convention=(
            "alpha = 1 exactly; the effective step size is absorbed into diagonal G_w"
        ),
        operating_point=(
            f"initial nominal snapshot ({config.start_time.isoformat()}), cached sensitivities, "
            "contingencies disabled for extraction"
        ),
        current_parameters=current_parameters,
        local_continuous=local,
        coupled_continuous=coupled,
        discrete_miqp=discrete,
        bo_guidance=bo_guidance,
        assumptions=[
            "The controller uses its cached affine sensitivity model; the plant Jacobian is not substituted.",
            "Continuous actuator constraints define a fixed convex set in the G_w metric for the projection-IQC claim.",
            "Integer OLTC/shunt variables are frozen for every continuous certificate.",
            "Reported robustness deltas are explicit hypothetical relative cached-gradient errors, not measured bounds.",
            "The snapshot is nominal and does not cover topology, operating-point, delay, or sensitivity drift unless included in delta.",
        ],
        risks=[
            "Zero-curvature directions prevent a full-state linear rate unless constraints or added regularisation remove them.",
            "The coupled map is generally non-symmetric, so the local gradient-sector theorem cannot be promoted to the hierarchy without a separate interconnection/projection LMI.",
            "State-dependent ramp bounds, slack variables, active-set changes, and asynchronous TSO/DSO timing can violate the fixed-projection abstraction.",
            "The Lessard IQC does not certify switching, dwell-time logic, quantisation, or MIQP integer decisions.",
        ],
    )


def analyse_config(
    config: MultiTSOConfig,
    *,
    config_factory: str = DEFAULT_CONFIG_FACTORY,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
) -> HierarchyCertificate:
    """Run the short automatic extraction and build the certificate."""

    # The existing extraction hook writes its legacy reports. Isolate them in
    # a temporary directory so this diagnostic cannot overwrite experiment data.
    with tempfile.TemporaryDirectory(prefix="ofo_stability_certificate_") as tmp:
        extraction_config = dataclasses.replace(config, result_dir=tmp)
        result = _run_one_step_stability(extraction_config)
    return certificate_from_result(
        result,
        config,
        config_factory=config_factory,
        deltas=deltas,
    )


def write_json(certificate: HierarchyCertificate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate.to_dict(), indent=2), encoding="utf-8")
