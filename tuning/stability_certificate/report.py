"""Human-readable stability-certificate report."""

from __future__ import annotations

from pathlib import Path

from .models import HierarchyCertificate, LMIResult


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _lmi(result: LMIResult) -> str:
    rho = _number(result.rho)
    return f"{result.status.value} (rho={rho}, solver={result.solver or 'n/a'})"


def to_markdown(certificate: HierarchyCertificate) -> str:
    coupled = certificate.coupled_continuous
    lines = [
        "# Offline IQC/LMI stability certificate",
        "",
        f"- Generated: `{certificate.generated_at}`",
        f"- Configuration: `{certificate.config_factory}`",
        f"- Step convention: {certificate.alpha_convention}",
        f"- Operating point: {certificate.operating_point}",
        "",
        "## Verdict",
        "",
        f"- Full frozen projected continuous hierarchy: **{_lmi(coupled.projected_full_state_iqc)}**",
        f"- Full frozen linear map (constraints inactive): **{_lmi(coupled.inactive_constraint_linear_lmi)}**",
        f"- Conditional non-neutral linear subspace: **{_lmi(coupled.active_mode_linear_lmi)}**",
        f"- Integer/MIQP Lessard-IQC applicability: **no**; complementary C3 model status: **{certificate.discrete_miqp['legacy_c3_status']}**",
        "",
        "A `not_certified` result is not evidence of instability. It means that the stated sufficient LMI and assumptions did not produce a proof.",
        "",
        "## Current parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {_number(float(value)) if isinstance(value, (int, float)) else value} |"
        for name, value in certificate.current_parameters.items()
    )
    lines.extend(
        [
            "",
            "## Local projected-gradient diagnostics",
            "",
            "The scale is the multiplier on the controller's entire current continuous diagonal `G_w`; alpha remains exactly one.",
            "",
            "| Controller | Modes active/total | m | L | Current rho | IQC rho | Stable G_w scale > | Best G_w scale | Best rho | Interpretation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in certificate.local_continuous:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.controller,
                    f"{item.n_active}/{item.n_total}",
                    _number(item.m),
                    _number(item.L),
                    _number(item.current_rho_formula),
                    _number(item.nominal_iqc.rho),
                    _number(item.stable_uniform_gw_scale_min),
                    _number(item.optimal_uniform_gw_scale),
                    _number(item.optimal_uniform_rho),
                    item.scale_interpretation,
                ]
            )
            + " |"
        )
    lines.extend(["", "### Robustness sweeps", ""])
    for item in certificate.local_continuous:
        values = ", ".join(
            f"delta={delta}: {_lmi(result)}" for delta, result in item.robust_iqc.items()
        ) or "no sweep"
        lines.append(f"- **{item.controller}:** {values}. Scope: {item.scope}")

    lines.extend(
        [
            "",
            "## Coupled continuous hierarchy",
            "",
            f"- Dimension: {coupled.dimension}; active modes: {coupled.n_active}; neutral modes: {coupled.n_neutral}.",
            f"- Full-map spectral radius: {_number(coupled.spectral_radius_full)}; spectral norm: {_number(coupled.spectral_norm_full)}.",
            f"- Symmetry defect: {_number(coupled.symmetry_defect)}; common-potential gradient interpretation: {coupled.potential_gradient_compatible}.",
            f"- Scope: {coupled.scope}",
            "",
            "## Integer actuators",
            "",
            f"- Method: {certificate.discrete_miqp['method']}.",
            f"- Lessard-IQC status: {certificate.discrete_miqp['status']}; legacy C3 model status: {certificate.discrete_miqp['legacy_c3_status']}.",
            f"- TSO discrete variables seen by C3: {certificate.discrete_miqp['n_tso_discrete_variables']}.",
            f"- Gamma spectral radius: {_number(certificate.discrete_miqp['gamma_spectral_radius'])}.",
            f"- Note: {certificate.discrete_miqp['note']}",
            "",
            "## BO guidance",
            "",
            f"- Ready as a hard full-hierarchy stability constraint: **{certificate.bo_guidance['ready_as_hard_stability_constraint']}**.",
            f"- {certificate.bo_guidance['recommended_use']}",
            f"- {certificate.bo_guidance['warning']}",
            "",
            "## Assumptions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in certificate.assumptions)
    lines.extend(["", "## Risks / unresolved points", ""])
    lines.extend(f"- {item}" for item in certificate.risks)
    return "\n".join(lines) + "\n"


def write_markdown(certificate: HierarchyCertificate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(certificate), encoding="utf-8")
