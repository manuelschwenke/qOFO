"""The per-DSO voltage relief must move ``dso_g_v`` and ``g_w_dso_oltc`` together.

Why this invariant matters
--------------------------
With ``dso_gamma_oltc_q = 0`` the subordinate OLTC is driven only by the voltage
gradient, so ``dso_g_v / g_w_dso_oltc`` **is** the OLTC loop gain.  Raising
``dso_g_v`` alone pushes the integer commit threshold into a limit cycle --
measured 50.5 tap reversals/h at a factor of 6.7, against 0.00 at baseline
(``configs/config.py``, ``dso_v_relief_factors``).  Holding the ratio instead
moves the extra authority onto the continuous DER block.

Rewritten 2026-09-10 with the weight reset.  It previously ran against
``experiments.run_multi_system_ofo.make_config`` and every JSON parameter set in
``configs/paramsets/``, and asserted that a bare ``MultiTSOConfig()`` *carries*
the relief.  All three premises are gone: the parameter sets were deleted, the
weights no longer have defaults, and ``dso_v_relief_factors`` now defaults to
``None`` precisely because a per-area x10 applied silently on every config
construction was a hidden parameter set.  The invariant itself is unchanged and
is what Stage III of the campaign relies on, so it is tested here against an
explicit config instead of against whatever weight set happened to ship.
"""

from __future__ import annotations

import dataclasses

import pytest

from configs.config import MultiTSOConfig, apply_dso_v_relief

#: Areas the relief is exercised on.  Arbitrary: the invariant is per-area and
#: does not depend on which areas a given design happens to relieve.
RELIEVED = ("DSO_2", "DSO_4")

#: A complete weight vector.  Deliberately explicit and deliberately not any
#: campaign answer -- round numbers chosen so that a factor applied to one of
#: them is visible by eye in a failure message.
WEIGHTS = dict(
    g_v=1.0e4, g_q=100.0, dso_g_v=2.0e3,
    g_w_der=10.0, g_w_pcc=20.0, g_w_gen=1.0e8, g_w_tso_oltc=500.0,
    g_w_dso_der=100.0, g_w_dso_oltc=50.0,
)


def make_cfg(**overrides) -> MultiTSOConfig:
    """A weight-complete config, relieved on :data:`RELIEVED` unless overridden."""
    kw = dict(WEIGHTS)
    kw["dso_v_relief_factors"] = {d: 10.0 for d in RELIEVED}
    kw.update(overrides)
    return MultiTSOConfig(**kw)


# ---------------------------------------------------------------------------
# The reset itself
# ---------------------------------------------------------------------------

def test_bare_config_carries_no_relief_and_no_weights() -> None:
    """A bare config is inert: no weights, and no per-area multiplier.

    This is the reset.  Before 2026-09-10 ``MultiTSOConfig()`` returned a config
    with nine weights and a x10 voltage relief on DSO_2 and DSO_4 already
    installed, so every run that did not override them inherited a parameter set
    nobody had chosen.
    """
    cfg = MultiTSOConfig()
    assert cfg.dso_v_relief_factors is None
    assert cfg.dso_g_v_per_area is None
    assert cfg.dso_g_w_class is None
    assert set(cfg.missing_weights()) == set(MultiTSOConfig.REQUIRED_WEIGHTS)
    with pytest.raises(ValueError, match="missing controller weights"):
        cfg.require_weights()


# ---------------------------------------------------------------------------
# The pairing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dso_id", RELIEVED)
def test_relief_holds_the_oltc_loop_gain(dso_id: str) -> None:
    cfg = make_cfg()
    factor = cfg.dso_v_relief_factors[dso_id]
    gv = float(cfg.dso_g_v_per_area[dso_id])
    oltc = float(cfg.dso_g_w_class[dso_id]["dso_oltc"])

    assert gv == pytest.approx(float(cfg.dso_g_v) * factor)
    assert oltc == pytest.approx(float(cfg.g_w_dso_oltc) * factor)
    # The ratio is the point: it must match the unrelieved loop gain.
    assert gv / oltc == pytest.approx(
        float(cfg.dso_g_v) / float(cfg.g_w_dso_oltc))


def test_relief_is_idempotent_under_replace() -> None:
    """``dataclasses.replace`` re-runs the derivation; it must not compound."""
    cfg = make_cfg()
    once = dict(cfg.dso_g_v_per_area)
    again = dataclasses.replace(dataclasses.replace(cfg))
    assert dict(again.dso_g_v_per_area) == pytest.approx(once)
    for d in RELIEVED:
        assert again.dso_g_w_class[d]["dso_oltc"] == pytest.approx(
            cfg.dso_g_w_class[d]["dso_oltc"])


def test_relief_tracks_an_edit_to_either_base_weight() -> None:
    """Editing a base weight re-derives the relieved pair from it."""
    cfg = make_cfg()
    bumped = dataclasses.replace(cfg, dso_g_v=float(cfg.dso_g_v) * 3.0)
    for d in RELIEVED:
        f = bumped.dso_v_relief_factors[d]
        assert bumped.dso_g_v_per_area[d] == pytest.approx(
            float(bumped.dso_g_v) * f)
        # The OLTC half must follow the new base too, or the gain moves.
        assert bumped.dso_g_w_class[d]["dso_oltc"] == pytest.approx(
            float(bumped.g_w_dso_oltc) * f)


def test_relief_is_scoped_to_the_listed_areas() -> None:
    cfg = make_cfg()
    assert set(cfg.dso_g_v_per_area) == set(RELIEVED)


def test_factor_one_and_none_are_no_ops() -> None:
    for factors in (None, {}, {d: 1.0 for d in RELIEVED}):
        cfg = make_cfg(dso_v_relief_factors=factors)
        assert not (cfg.dso_g_v_per_area or {}), factors
        assert not (cfg.dso_g_w_class or {}), factors


def test_relief_preserves_another_areas_per_area_design() -> None:
    """A per-area design for an unrelieved area must survive the derivation."""
    cfg = make_cfg(
        dso_g_v_per_area={"DSO_1": 7.0},
        dso_g_w_class={"DSO_1": {"dso_der": 11.0}},
    )
    assert cfg.dso_g_v_per_area["DSO_1"] == pytest.approx(7.0)
    assert cfg.dso_g_w_class["DSO_1"]["dso_der"] == pytest.approx(11.0)
    for d in RELIEVED:
        assert d in cfg.dso_g_v_per_area


def test_explicit_call_replaces_rather_than_layers() -> None:
    """``apply_dso_v_relief`` re-derives from the bases, never from its output."""
    cfg = make_cfg()
    twice = apply_dso_v_relief(cfg, {d: 10.0 for d in RELIEVED})
    for d in RELIEVED:
        assert twice.dso_g_v_per_area[d] == pytest.approx(
            float(cfg.dso_g_v) * 10.0)


# ---------------------------------------------------------------------------
# Bounds the controller enforces, asserted on the config that feeds it
# ---------------------------------------------------------------------------

def test_per_area_gamma_is_within_the_controller_bound() -> None:
    """Every designed gamma must be one the controller will accept.

    The table lives in config data and the bound in the controller, so nothing
    but this test connects them; a value past the cap would fail at controller
    construction, i.e. minutes into a run.
    """
    from controller.dso_controller import GAMMA_OLTC_Q_MAX

    cfg = make_cfg()
    for dso_id, gamma in (cfg.dso_gamma_oltc_q_per_area or {}).items():
        assert 0.0 <= float(gamma) <= GAMMA_OLTC_Q_MAX, (
            f"{dso_id}: gamma {gamma!r} outside [0, {GAMMA_OLTC_Q_MAX:g}]")
    assert 0.0 <= float(cfg.dso_gamma_oltc_q) <= GAMMA_OLTC_Q_MAX


@pytest.mark.parametrize("dso_id", RELIEVED)
def test_relieved_area_is_not_left_q_inert(dso_id: str) -> None:
    """The voltage relief must not leave the tap unable to act on interface Q.

    Holding ``dso_g_v / g_w_dso_oltc`` preserves the voltage commit threshold of
    the OLTC, but the factor on ``g_w_dso_oltc`` is uncompensated in its
    interface-Q threshold ``(g_w_oltc + norm(a_oltc)**2)/(2 g_q gamma dQ/ds)``,
    which therefore rises by the full factor -- 108-244 Mvar to commit on
    DSO_2/DSO_4 at gamma = 1, against ~6 Mvar of measured RMSE (2026-08-20).

    Skipped at gamma = 0 -- the pinned structural value -- where the tap has no
    Q gradient at all and no weight makes that threshold finite.
    """
    cfg = make_cfg()
    per_gamma = cfg.dso_gamma_oltc_q_per_area or {}
    gamma_area = float(per_gamma.get(dso_id, cfg.dso_gamma_oltc_q))
    gamma_base = float(cfg.dso_gamma_oltc_q)
    if gamma_area <= 0.0 or gamma_base <= 0.0:
        pytest.skip("gamma_oltc_q = 0: the OLTC carries no interface-Q gradient")

    q_area = float((cfg.dso_g_q_per_area or {}).get(dso_id, cfg.g_q))
    inflation = ((cfg.dso_g_w_class[dso_id]["dso_oltc"] / (q_area * gamma_area))
                 / (float(cfg.g_w_dso_oltc) / (float(cfg.g_q) * gamma_base)))
    assert inflation <= 1.0 + 1e-9, (
        f"{dso_id}: the voltage relief leaves the interface-Q commit threshold "
        f"of the tap at x{inflation:.2f} the unrelieved one. Compensate on the "
        f"Q channel with dso_gamma_oltc_q_per_area (OLTC only, preferred) or "
        f"dso_g_q_per_area (which also moves the DER block).")
