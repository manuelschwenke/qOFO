"""
Tests for the per-DSO voltage-relief pair (``dso_g_v_per_area`` +
``dso_g_w_class[...]['dso_oltc']``).

The pair exists because ``dso_gamma_oltc_q = 0.0`` makes the DSO OLTC a purely
voltage-driven actuator: the tap's gradient is
``2 * dso_g_v * (V - V_set)^T * dV/ds`` and, being integer, it commits only when
that exceeds ``g_w_dso_oltc + g_u``.  The ratio ``dso_g_v / g_w_dso_oltc`` is
therefore the OLTC loop gain, and raising ``dso_g_v`` on its own drives the tap
into a limit cycle — measured 2026-08-18 on DSO_4 at 50.5 tap reversals/h
against 0.00 at baseline (docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md).

What is locked here is the *invariant*, not the tuned factor: whichever way the
config is built, a relieved area's ``g_v`` and its ``dso_oltc`` weight must move
by the same factor, and the other areas' per-area designs must survive.

Since 2026-08-21 the relief is a **config field**
(:attr:`MultiTSOConfig.dso_v_relief_factors`) applied in ``__post_init__``
rather than a post-hoc call, so the tests below also pin the property that
motivated the change: it is idempotent, and a ``dataclasses.replace`` re-derives
it instead of squaring the factor.
"""

from __future__ import annotations

import dataclasses

import pytest

from configs.config import MultiTSOConfig, apply_dso_v_relief
from configs.paramsets import apply_paramset, available
from experiments.run_multi_system_ofo import make_config


def _builders():
    """The runner config and every parameter set applied on top of it."""
    out = [("make_config", make_config)]
    for name in available():
        out.append((name,
                    lambda n=name: apply_paramset(make_config(), n,
                                                  verbose=False)))
    return out


BUILDERS = _builders()
BUILDER_IDS = [n for n, _ in BUILDERS]
RELIEVED = sorted(make_config().dso_v_relief_factors)


@pytest.mark.parametrize("builder", [b for _, b in BUILDERS], ids=BUILDER_IDS)
@pytest.mark.parametrize("dso_id", RELIEVED)
def test_relief_holds_the_oltc_loop_gain(builder, dso_id):
    """g_v and the OLTC step weight move by the same factor in every builder.

    The ``per_area`` set rewrites ``dso_g_w_class`` wholesale with its analytic
    per-area block *and* rescales the whole weight group by the gauge, so it is
    the case that would silently break if the relief were written as two
    hand-matched literals.
    """
    cfg = builder()
    factor = cfg.dso_v_relief_factors[dso_id]
    gv = cfg.dso_g_v_per_area[dso_id] / float(cfg.dso_g_v)
    gw = cfg.dso_g_w_class[dso_id]["dso_oltc"] / float(cfg.g_w_dso_oltc)
    assert gv == pytest.approx(factor)
    assert gw == pytest.approx(factor), (
        f"{dso_id}: g_v raised x{gv:g} but the OLTC step weight x{gw:g}; the "
        f"OLTC loop gain moved x{gv / gw:g} and the integer tap may limit-cycle"
    )


@pytest.mark.parametrize("builder", [b for _, b in BUILDERS], ids=BUILDER_IDS)
def test_relief_is_idempotent_under_replace(builder):
    """``dataclasses.replace`` re-derives the relief; it must not compound.

    This is the whole reason the relief became a field.  As a post-hoc call it
    read an existing per-area ``dso_oltc`` entry as its base, so applying it
    twice squared the factor -- a trap every config factory had to work around
    by stripping the per-area maps first and re-applying last.
    """
    cfg = builder()
    again = dataclasses.replace(cfg)
    assert again.dso_g_v_per_area == cfg.dso_g_v_per_area
    assert again.dso_g_w_class == cfg.dso_g_w_class
    # ... and three copies deep, which is what a builder chain actually does.
    thrice = dataclasses.replace(dataclasses.replace(again))
    assert thrice.dso_g_v_per_area == cfg.dso_g_v_per_area
    assert thrice.dso_g_w_class == cfg.dso_g_w_class


def test_relief_tracks_an_edit_to_either_base_weight():
    """Moving ``dso_g_v`` or ``g_w_dso_oltc`` moves the relieved pair with it.

    The property an absolute per-area literal cannot have, and the reason
    ``tuning_mc.stage_1_search`` needs the relief applied to the config's own
    weights: ``dso_g_v_ratio`` is a search coordinate, so an absolute relief
    would let the OLTC loop gain drift as the search walked.
    """
    cfg = make_config()
    moved = dataclasses.replace(cfg, dso_g_v=2.0 * cfg.dso_g_v,
                                g_w_dso_oltc=3.0 * cfg.g_w_dso_oltc)
    for dso_id, factor in cfg.dso_v_relief_factors.items():
        assert moved.dso_g_v_per_area[dso_id] == pytest.approx(
            2.0 * cfg.dso_g_v * factor)
        assert moved.dso_g_w_class[dso_id]["dso_oltc"] == pytest.approx(
            3.0 * cfg.g_w_dso_oltc * factor)


def test_relief_is_scoped_to_the_spread_limited_areas():
    """Only the areas listed in ``dso_v_relief_factors`` get a voltage override.

    DSO_1 (spread 0.015) and DSO_3 (0.037) must stay out: measured 2026-08-18,
    giving them the factor bought 0.0016 / 0.0002 p.u. of V_max and cost DSO_3
    +53 % interface-Q RMSE.
    """
    cfg = make_config()
    assert set(cfg.dso_g_v_per_area) == set(cfg.dso_v_relief_factors)
    assert "DSO_1" not in cfg.dso_g_v_per_area
    assert "DSO_3" not in cfg.dso_g_v_per_area


def test_relief_preserves_other_areas_per_area_design():
    """The per-area ``dso_der`` / ``dso_oltc`` design for DSO_1..3 survives."""
    cfg = apply_paramset(make_config(), "per_area", verbose=False)
    for dso in ("DSO_1", "DSO_2", "DSO_3"):
        assert dso in cfg.dso_g_w_class
        assert set(cfg.dso_g_w_class[dso]) == {"dso_der", "dso_oltc"}
    # DSO_4 keeps its analytic dso_der entry too -- the relief merges, it does
    # not replace the area's block.
    assert "dso_der" in cfg.dso_g_w_class["DSO_4"]


def test_factor_one_is_a_no_op():
    """A factor of 1.0 (or an empty mapping) disables the relief cleanly."""
    cfg = dataclasses.replace(make_config(), dso_v_relief_factors=None,
                              dso_g_v_per_area=None, dso_g_w_class=None)
    assert cfg.dso_g_v_per_area is None
    assert apply_dso_v_relief(cfg, {"DSO_4": 1.0}) is cfg
    assert apply_dso_v_relief(cfg, {}) is cfg


def test_explicit_call_replaces_rather_than_layers():
    """``apply_dso_v_relief`` re-derives from the base scalars.

    Applied on top of a config that already carries the default relief, it must
    land on ``base * new_factor`` for the areas it names and REMOVE the entries
    it does not, rather than multiplying into whatever was there.
    """
    cfg = make_config()
    assert set(cfg.dso_v_relief_factors) == {"DSO_2", "DSO_4"}
    out = apply_dso_v_relief(cfg, {"DSO_4": 3.0})
    assert set(out.dso_v_relief_factors) == {"DSO_4"}
    assert out.dso_g_v_per_area == {
        "DSO_4": pytest.approx(float(cfg.dso_g_v) * 3.0)}
    assert out.dso_g_w_class["DSO_4"]["dso_oltc"] == pytest.approx(
        float(cfg.g_w_dso_oltc) * 3.0)
    assert "DSO_2" not in out.dso_g_v_per_area
    assert "DSO_2" not in (out.dso_g_w_class or {})


def test_relief_merges_into_an_existing_spec():
    """An existing per-area spec for another class/area is not clobbered."""
    cfg = dataclasses.replace(
        make_config(),
        dso_v_relief_factors={"DSO_4": 3.0},
        dso_g_v_per_area={"DSO_2": 1.0},
        dso_g_w_class={"DSO_2": {"dso_der": 42.0},
                       "DSO_4": {"dso_der": 7.0}},
    )
    assert cfg.dso_g_v_per_area["DSO_2"] == 1.0
    assert cfg.dso_g_w_class["DSO_2"] == {"dso_der": 42.0}
    assert cfg.dso_g_w_class["DSO_4"]["dso_der"] == 7.0
    assert cfg.dso_g_w_class["DSO_4"]["dso_oltc"] == pytest.approx(
        float(cfg.g_w_dso_oltc) * 3.0)
    assert cfg.dso_g_v_per_area["DSO_4"] == pytest.approx(
        float(cfg.dso_g_v) * 3.0)


@pytest.mark.parametrize("builder", [b for _, b in BUILDERS], ids=BUILDER_IDS)
def test_per_area_gamma_is_within_the_controller_bound(builder):
    """Every designed gamma must be one the controller will accept.

    The table lives in config data and the bound in the controller, so nothing
    but this test connects them; a value past the cap would fail at controller
    construction, i.e. minutes into a run.
    """
    from controller.dso_controller import GAMMA_OLTC_Q_MAX

    cfg = builder()
    table = cfg.dso_gamma_oltc_q_per_area or {}
    for dso_id, gamma in table.items():
        assert 0.0 <= float(gamma) <= GAMMA_OLTC_Q_MAX, (
            f"{dso_id}: gamma {gamma!r} outside [0, {GAMMA_OLTC_Q_MAX:g}]"
        )
    assert 0.0 <= float(cfg.dso_gamma_oltc_q) <= GAMMA_OLTC_Q_MAX


@pytest.mark.parametrize("builder", [b for _, b in BUILDERS], ids=BUILDER_IDS)
@pytest.mark.parametrize("dso_id", RELIEVED)
def test_relieved_area_is_not_left_q_inert(builder, dso_id):
    """The voltage relief must not leave the tap unable to act on interface Q.

    Holding ``dso_g_v / g_w_dso_oltc`` preserves the OLTC's VOLTAGE commit
    threshold, but the factor on ``g_w_dso_oltc`` is uncompensated in its
    INTERFACE-Q threshold ``(g_w_oltc + ||a_oltc||^2)/(2 g_q gamma |dQ/ds|)``,
    which therefore rises by the full factor -- 108-244 Mvar to commit on
    DSO_2/DSO_4 at gamma = 1, against ~6 Mvar of measured RMSE (2026-08-20).

    Asserted on the weights, which is what the config controls; the linear
    ``1/gamma`` reading ignores the ``gamma^2 ||a_q||^2`` self-cost term, so it
    is a proxy for the threshold, not the threshold.  The measured Mvar values
    are in the daily log.

    Skipped at gamma = 0, where the tap has no Q gradient at all and no weight
    makes that threshold finite.
    """
    cfg = builder()
    per_gamma = cfg.dso_gamma_oltc_q_per_area or {}
    gamma_area = float(per_gamma.get(dso_id, cfg.dso_gamma_oltc_q))
    gamma_base = float(cfg.dso_gamma_oltc_q)
    if gamma_area <= 0.0 or gamma_base <= 0.0:
        pytest.skip("gamma_oltc_q = 0: the OLTC carries no interface-Q gradient")

    v_factor = cfg.dso_v_relief_factors[dso_id]
    oltc_base = float(cfg.g_w_dso_oltc)
    q_area = float((cfg.dso_g_q_per_area or {}).get(dso_id, cfg.g_q))
    inflation = ((cfg.dso_g_w_class[dso_id]["dso_oltc"] / (q_area * gamma_area))
                 / (oltc_base / (float(cfg.g_q) * gamma_base)))
    assert inflation <= 1.0 + 1e-9, (
        f"{dso_id}: the x{v_factor:g} voltage relief leaves the tap's "
        f"interface-Q commit threshold x{inflation:.2f} the unrelieved one. "
        f"Compensate on the Q channel with dso_gamma_oltc_q_per_area (OLTC "
        f"only, preferred) or dso_g_q_per_area (also moves the DER block)."
    )


def test_gamma_gain_is_applied_by_both_the_controller_and_stage_0():
    """``gamma > 1`` must scale in *both* places or the design is against the
    wrong self-cost.

    Until 2026-08-20 both sites read ``if gamma < 1.0``, so a gain was a silent
    no-op.  Fixing only one of them would be worse than fixing neither: Stage 0
    would design ``g_w_dso_oltc`` against a ``||a_i||^2`` the MIQP never sees,
    which is the disagreement the gamma block in ``stage_0_preconditioning``
    exists to prevent.
    """
    import inspect

    import controller.dso_controller as dc
    import tuning_mc.stage_0_preconditioning as s0

    for mod in (dc, s0):
        src = inspect.getsource(mod)
        assert "if gamma < 1.0" not in src, (
            f"{mod.__name__} still guards the gamma scaling with '< 1.0', so a "
            f"gamma above 1 is silently ignored there"
        )
        assert "gamma != 1.0" in src, (
            f"{mod.__name__} no longer applies the gamma scaling at all"
        )


def test_bare_config_carries_the_relief_by_default():
    """A plain ``MultiTSOConfig()`` already has the relief installed.

    The field default is the runner's factor set, so a config built anywhere --
    a test, a tuning stage, an ad-hoc script -- gets the spread-limited areas'
    voltage authority without having to remember a post-hoc call.
    """
    cfg = MultiTSOConfig(dso_g_v=1.0e5, g_w_dso_oltc=183.0)
    factors = cfg.dso_v_relief_factors
    # The factor itself is a tuning point and moves; what is locked is that a
    # bare config HAS one and that it is installed on both weights.
    assert factors, "MultiTSOConfig lost its default voltage relief"
    assert set(factors) == {"DSO_2", "DSO_4"}
    assert cfg.dso_g_v_per_area == {
        d: pytest.approx(1.0e5 * f) for d, f in factors.items()}
    assert cfg.dso_g_w_class == {
        d: {"dso_oltc": pytest.approx(183.0 * f)} for d, f in factors.items()}
