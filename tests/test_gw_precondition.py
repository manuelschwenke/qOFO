"""
Unit tests for :mod:`controller.gw_precondition` (Tier-2 g_w preconditioning).

Proves the defining properties of the curvature rule:

* **Gain (cap-only).**  ``precondition_g_w`` only ever *reduces* a loop's
  ``lambda_max`` to the target; a loop already at/below the target is left
  untouched (it can never make a loop more aggressive).
* **Shape.**  Column-granularity preconditioning equalises the per-actuator
  contributions ``||a_i||^2 / g_w_i`` (Zagorowska Eq. 16 diagonal scaling).
* **Curvature.**  ``curvature_spectrum`` has the ``M ∝ G_w^{-1}`` scaling
  that makes one global ``kappa`` sufficient.

Plus the safety contracts: integer/non-listed classes are never touched,
the integer-dominated case declines to act, near-uncontrollable columns are
floored, and the result is scale-covariant in ``H`` and ``g_v``.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller.gw_precondition import (
    curvature_spectrum,
    precondition_g_w,
)


def _toy_problem(seed: int = 0):
    """Small synthetic curvature problem with ``lambda_max_before`` ~ O(1)s.

    Layout: ``der=[0,1]``, ``pcc=[2]`` (continuous), ``oltc=[3,4]`` (integer,
    large g_w → negligible curvature).  Unit ``g_v`` and O(1) ``H`` columns
    give ``lambda_max_before`` of a few, so targets like 0.5/0.9/1.2 are
    *below* it (→ the "reduced" branch).
    """
    rng = np.random.default_rng(seed)
    n_v, n_u = 4, 5
    H_v = rng.standard_normal((n_v, n_u))
    g_v = np.ones(n_v)
    # OLTC g_w large enough that its curvature stays negligible even under
    # the H/g_v rescalings used by the covariance test.
    g_w0 = np.array([1.0, 1.0, 1.0, 1e6, 1e6])
    class_map = {
        "der": np.array([0, 1], dtype=np.int64),
        "pcc": np.array([2], dtype=np.int64),
        "tso_oltc": np.array([3, 4], dtype=np.int64),
    }
    return H_v, g_v, g_w0, class_map


# ---------------------------------------------------------------------------
# curvature_spectrum
# ---------------------------------------------------------------------------

def test_curvature_scales_inverse_with_g_w():
    """``M ∝ G_w^{-1}``: scaling all g_w by kappa scales lambda_max by 1/kappa."""
    H_v, g_v, g_w0, _ = _toy_problem()
    lam1 = curvature_spectrum(H_v, g_v, g_w0).lambda_max
    for kappa in (2.0, 10.0, 0.5):
        lam_k = curvature_spectrum(H_v, g_v, g_w0 * kappa).lambda_max
        assert lam_k == pytest.approx(lam1 / kappa, rel=1e-9)


def test_curvature_spectrum_psd_and_bounds():
    H_v, g_v, g_w0, _ = _toy_problem()
    spec = curvature_spectrum(H_v, g_v, g_w0)
    assert np.all(spec.eigenvalues >= -1e-12)
    assert spec.lambda_max == pytest.approx(float(spec.eigenvalues.max()))
    assert spec.lambda_min_pos > 0.0
    assert spec.cond >= 1.0


def test_curvature_rejects_nonpositive_g_w():
    H_v, g_v, g_w0, _ = _toy_problem()
    bad = g_w0.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError):
        curvature_spectrum(H_v, g_v, bad)


# ---------------------------------------------------------------------------
# precondition_g_w — gain (cap-only: reduces a hot loop to the target)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("granularity", ["class", "column"])
@pytest.mark.parametrize("target", [0.5, 0.9, 1.2])
def test_precondition_reduces_hot_loop_to_target(granularity, target):
    H_v, g_v, g_w0, class_map = _toy_problem()
    # sanity: the toy loop is hotter than every target tried here
    assert curvature_spectrum(H_v, g_v, g_w0).lambda_max > 1.2
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map,
        preconditionable_classes=["der", "pcc"],
        lambda_target=target,
        granularity=granularity,
    )
    assert res.status == "reduced" and res.applied is True
    assert res.lambda_max_after == pytest.approx(target, rel=5e-3)
    lam = curvature_spectrum(H_v, g_v, res.g_w_new).lambda_max
    assert lam == pytest.approx(target, rel=5e-3)
    # never raised: after <= before
    assert res.lambda_max_after <= res.lambda_max_before + 1e-9


def test_within_margin_is_noop():
    """A loop already at/below the target must be left untouched (cap-only)."""
    H_v, g_v, _, class_map = _toy_problem()
    # Large continuous g_w → already well-damped (lambda_max << target).
    g_w0 = np.array([50.0, 50.0, 50.0, 1e6, 1e6])
    lam_before = curvature_spectrum(H_v, g_v, g_w0).lambda_max
    assert lam_before < 0.9
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map, ["der", "pcc"], lambda_target=0.9,
    )
    assert res.status == "within_margin" and res.applied is False
    assert res.kappa == 1.0
    assert np.allclose(res.g_w_new, g_w0)
    assert res.lambda_max_after == pytest.approx(lam_before, rel=1e-9)


def test_precondition_leaves_integer_classes_untouched():
    H_v, g_v, g_w0, class_map = _toy_problem()
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map, ["der", "pcc"], lambda_target=0.9,
    )
    assert res.status == "reduced"
    oltc = class_map["tso_oltc"]
    assert np.allclose(res.g_w_new[oltc], g_w0[oltc])
    cont = np.concatenate([class_map["der"], class_map["pcc"]])
    assert not np.allclose(res.g_w_new[cont], g_w0[cont])
    assert set(res.preconditioned_classes) == {"der", "pcc"}


def test_column_granularity_equalises_contributions():
    """Column preconditioning ⇒ ||a_i||^2 / g_w_i constant (= 1/kappa)."""
    H_v, g_v, g_w0, class_map = _toy_problem()
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map, ["der", "pcc"],
        lambda_target=0.9, granularity="column",
    )
    sqrt_gv = np.sqrt(g_v)
    A = sqrt_gv[:, None] * H_v
    col_sq = np.einsum("ij,ij->j", A, A)
    cont = np.concatenate([class_map["der"], class_map["pcc"]])
    contrib = col_sq[cont] / res.g_w_new[cont]
    assert np.allclose(contrib, contrib[0], rtol=1e-6)
    assert contrib[0] == pytest.approx(1.0 / res.kappa, rel=1e-6)


# ---------------------------------------------------------------------------
# Safety contracts
# ---------------------------------------------------------------------------

def test_integer_dominated_declines_to_act():
    """When the fixed (integer) curvature alone exceeds the target, the
    preconditioner must DECLINE (no-op + flag), never touch the continuous
    actuators (the z1/z3 pathology the smoke exposed)."""
    g_v = np.array([1.0, 1.0])
    # col 0 = continuous 'der' (weak, row 1); col 1 = integer 'oltc'
    # (strong, row 0) → oltc alone gives lambda_max = 1.0 > target 0.9,
    # and the full loop lambda_max ≈ 1.0 > 0.9 (so not 'within_margin').
    H_v = np.array([[0.0, 1.0],
                    [1e-3, 0.0]])
    g_w0 = np.array([2.0, 1.0])
    class_map = {
        "der": np.array([0], dtype=np.int64),
        "oltc": np.array([1], dtype=np.int64),
    }
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map,
        preconditionable_classes=["der"],   # oltc is integer → fixed
        lambda_target=0.9, granularity="column",
    )
    assert res.status == "integer_dominated" and res.applied is False
    assert res.lambda_floor == pytest.approx(1.0, rel=1e-6)
    assert res.kappa == 1.0
    assert np.allclose(res.g_w_new, g_w0)
    assert res.lambda_max_after == pytest.approx(res.lambda_max_before, rel=1e-9)


def test_near_zero_column_is_floored_not_collapsed():
    """A column with ~0 voltage sensitivity must not drive g_w -> 0."""
    H_v, g_v, g_w0, class_map = _toy_problem()
    H_v = H_v.copy()
    H_v[:, 2] = 0.0  # the 'pcc' actuator has no voltage sensitivity
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map, ["der", "pcc"],
        lambda_target=0.9, granularity="column",
    )
    assert np.all(np.isfinite(res.g_w_new)) and np.all(res.g_w_new > 0.0)
    assert res.g_w_new[2] > 0.0


def test_no_preconditionable_class_is_noop():
    H_v, g_v, g_w0, class_map = _toy_problem()
    res = precondition_g_w(
        H_v, g_v, g_w0, class_map, ["does_not_exist"], lambda_target=0.9,
    )
    assert res.status == "no_class" and res.applied is False
    assert np.allclose(res.g_w_new, g_w0)
    assert res.kappa == 1.0
    assert res.preconditioned_classes == ()


def test_scale_covariance_in_H_and_g_v():
    """Rescaling H or g_v changes the gauge but the reduced target is hit."""
    H_v, g_v, g_w0, class_map = _toy_problem()
    for H_scale, gv_scale in [(10.0, 1.0), (1.0, 100.0), (3.0, 1.0)]:
        res = precondition_g_w(
            H_scale * H_v, gv_scale * g_v, g_w0, class_map,
            preconditionable_classes=["der", "pcc"], lambda_target=0.9,
        )
        assert res.status == "reduced"
        assert res.lambda_max_after == pytest.approx(0.9, rel=5e-3)


def test_invalid_inputs_raise():
    H_v, g_v, g_w0, class_map = _toy_problem()
    with pytest.raises(ValueError):
        precondition_g_w(H_v, g_v, g_w0, class_map, ["der"], lambda_target=2.5)
    with pytest.raises(ValueError):
        precondition_g_w(H_v, g_v, g_w0, class_map, ["der"], granularity="bogus")
