from __future__ import annotations

import numpy as np
import pytest

from tuning.stability_certificate.linear import (
    active_invariant_block,
    linear_rate_certificate,
    projected_linear_rate_certificate,
)
from tuning.stability_certificate.models import CertificateStatus


def test_linear_lmi_matches_diagonal_spectral_radius() -> None:
    result = linear_rate_certificate(np.diag([0.4, 0.8]))

    assert result.status == CertificateStatus.CERTIFIED
    assert result.rho == pytest.approx(0.8, abs=5e-4)


def test_projection_iqc_certifies_contracting_diagonal_map() -> None:
    result = projected_linear_rate_certificate(np.diag([0.4, 0.8]))

    assert result.status == CertificateStatus.CERTIFIED
    assert result.rho is not None and result.rho < 1.0


def test_full_linear_lmi_rejects_neutral_mode() -> None:
    result = linear_rate_certificate(np.diag([0.5, 1.0]))

    assert result.status == CertificateStatus.NOT_CERTIFIED


def test_active_schur_block_separates_neutral_mode() -> None:
    active, n_neutral, tolerance = active_invariant_block(np.diag([0.5, 1.0]))

    assert active.shape == (1, 1)
    assert active[0, 0] == pytest.approx(0.5)
    assert n_neutral == 1
    assert tolerance > 0.0


def test_active_schur_treats_roundoff_near_one_as_neutral() -> None:
    active, n_neutral, _ = active_invariant_block(
        np.diag([0.5, 1.0 + 4e-9])
    )

    assert active.shape == (1, 1)
    assert n_neutral == 1
