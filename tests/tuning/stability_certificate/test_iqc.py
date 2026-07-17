from __future__ import annotations

import pytest

from tuning.stability_certificate.iqc import sector_rate_certificate
from tuning.stability_certificate.models import CertificateStatus


def test_nominal_sector_lmi_matches_quadratic_rate() -> None:
    result = sector_rate_certificate(0.2, 1.5)

    assert result.status == CertificateStatus.CERTIFIED
    assert result.rho == pytest.approx(0.8, abs=2e-4)
    assert result.max_lmi_eigenvalue is not None
    assert result.max_lmi_eigenvalue <= 2e-5


def test_relative_gradient_error_cannot_improve_certified_rate() -> None:
    nominal = sector_rate_certificate(0.2, 1.5)
    robust = sector_rate_certificate(0.2, 1.5, delta=0.05)

    assert robust.status == CertificateStatus.CERTIFIED
    assert robust.rho is not None and nominal.rho is not None
    assert robust.rho >= nominal.rho - 2e-4


def test_oversized_step_is_not_certified() -> None:
    result = sector_rate_certificate(0.2, 2.2)

    assert result.status == CertificateStatus.NOT_CERTIFIED
    assert result.rho is None


def test_sector_requires_positive_strong_convexity() -> None:
    result = sector_rate_certificate(0.0, 1.0)

    assert result.status == CertificateStatus.NOT_APPLICABLE
