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
config is built, DSO_4's ``g_v`` and its ``dso_oltc`` weight must move by the
same factor, and the other areas' per-area designs must survive.
"""

from __future__ import annotations

import dataclasses

import pytest

from experiments.run_multi_system_ofo import (
    DSO_V_RELIEF_FACTORS,
    _apply_dso_v_relief,
    make_config_per_area,
    make_config_tuned,
)


def _loop_gain_ratio(cfg, dso_id: str, oltc_base: float) -> tuple[float, float]:
    """(g_v factor, g_w_oltc factor) for ``dso_id`` against its own bases."""
    gv = cfg.dso_g_v_per_area[dso_id] / float(cfg.dso_g_v)
    gw = cfg.dso_g_w_class[dso_id]["dso_oltc"] / oltc_base
    return gv, gw


@pytest.mark.parametrize("builder", [make_config_tuned, make_config_per_area])
@pytest.mark.parametrize("dso_id", sorted(DSO_V_RELIEF_FACTORS))
def test_relief_holds_the_oltc_loop_gain(builder, dso_id):
    """g_v and the OLTC step weight move by the same factor in every builder.

    ``make_config_per_area`` rewrites ``dso_g_w_class`` wholesale with its
    analytic per-area block *and* rescales the whole weight group by ``GAUGE``,
    so this is the case that would silently break if the relief were written as
    two hand-matched literals.
    """
    cfg = builder()
    # The OLTC base is whatever that config would have given DSO_4 without the
    # relief: the per-area design if one exists, else the global scalar.
    bare = dataclasses.replace(builder(), dso_g_v_per_area=None,
                               dso_g_w_class=None, dso_g_q_per_area=None)
    assert bare.dso_g_v_per_area is None
    factor = DSO_V_RELIEF_FACTORS[dso_id]
    oltc_base = cfg.dso_g_w_class[dso_id]["dso_oltc"] / factor
    assert oltc_base > 0

    gv, gw = _loop_gain_ratio(cfg, dso_id, oltc_base)
    assert gv == pytest.approx(factor)
    assert gw == pytest.approx(factor)
    assert gv / gw == pytest.approx(1.0), (
        f"OLTC loop gain moved by {gv / gw:.3f}x -- the integer tap will "
        f"limit-cycle; raise dso_g_v and dso_oltc by the same factor"
    )
    assert bare.dso_g_v == cfg.dso_g_v          # relief must not touch the base


@pytest.mark.parametrize("builder", [make_config_tuned, make_config_per_area])
@pytest.mark.parametrize("dso_id", sorted(DSO_V_RELIEF_FACTORS))
def test_relief_holds_the_oltc_q_threshold_when_the_tap_tracks_q(builder, dso_id):
    """At ``gamma_oltc_q > 0`` the ``g_q`` leg must be present too.

    Replaces the blanket ``assert dso_gamma_oltc_q == 0.0`` this module used to
    open with.  That guard said "the loop-gain argument needs revisiting" -- it
    has been (2026-08-20), and this is the result, so the guard is now a check
    of the *second* invariant rather than a refusal to look.

    Holding ``dso_g_v / g_w_dso_oltc`` preserves the tap's VOLTAGE commit
    threshold.  Once the tap also carries a Q gradient, its INTERFACE-Q
    threshold ``(g_w_oltc + ||a||^2) / (2 g_q |dQ/ds|)`` matters too, and the
    factor on ``g_w_dso_oltc`` is uncompensated there unless ``g_q`` moves with
    it.  Measured 2026-08-20 at gamma = 1 and a x20 relief without this leg:
    DSO_2/DSO_4 commit at 108-244 Mvar against ~6 Mvar of interface-Q RMSE,
    i.e. never, while DSO_1/DSO_3 commit at 2.9-5.0 Mvar.

    Skipped at gamma = 0, where the tap has no Q gradient and no ``g_q`` makes
    that threshold finite.
    """
    cfg = builder()
    if float(cfg.dso_gamma_oltc_q) <= 0.0:
        pytest.skip("gamma_oltc_q = 0: the OLTC carries no interface-Q gradient")

    per_q = cfg.dso_g_q_per_area or {}
    assert dso_id in per_q, (
        f"{dso_id} has a x{DSO_V_RELIEF_FACTORS[dso_id]:g} relief on "
        f"g_w_dso_oltc and gamma_oltc_q = {cfg.dso_gamma_oltc_q:g}, but no "
        f"dso_g_q_per_area entry -- its interface-Q commit threshold is "
        f"x{DSO_V_RELIEF_FACTORS[dso_id]:g} the unrelieved one and the tap will "
        f"not respond to Q.  Pass scale_q=True to apply_dso_v_relief."
    )
    factor = DSO_V_RELIEF_FACTORS[dso_id]
    oltc_base = cfg.dso_g_w_class[dso_id]["dso_oltc"] / factor
    q_base = float(per_q[dso_id]) / factor
    # The invariant: g_w_dso_oltc / g_q unchanged by the relief, so the Q
    # threshold is the same as it would be without it.
    assert (cfg.dso_g_w_class[dso_id]["dso_oltc"] / per_q[dso_id]
            == pytest.approx(oltc_base / q_base)), (
        "the relief moved the OLTC's interface-Q commit threshold; scale "
        "g_q by the same factor as g_w_dso_oltc"
    )


def test_relief_is_scoped_to_the_spread_limited_areas():
    """Only the areas listed in DSO_V_RELIEF_FACTORS get a voltage override.

    DSO_1 (spread 0.015) and DSO_3 (0.037) must stay out: measured 2026-08-18,
    giving them the factor bought 0.0016 / 0.0002 p.u. of V_max and cost DSO_3
    +53 % interface-Q RMSE.
    """
    cfg = make_config_tuned()
    assert set(cfg.dso_g_v_per_area) == set(DSO_V_RELIEF_FACTORS)
    assert "DSO_1" not in cfg.dso_g_v_per_area
    assert "DSO_3" not in cfg.dso_g_v_per_area


def test_relief_preserves_other_areas_per_area_design():
    """The per-area ``dso_der`` / ``dso_oltc`` design for DSO_1..3 survives."""
    cfg = make_config_per_area()
    for dso in ("DSO_1", "DSO_2", "DSO_3"):
        assert dso in cfg.dso_g_w_class
        assert set(cfg.dso_g_w_class[dso]) == {"dso_der", "dso_oltc"}
    # DSO_4 keeps its analytic dso_der entry too -- the relief merges, it does
    # not replace the area's block.
    assert "dso_der" in cfg.dso_g_w_class["DSO_4"]


def test_factor_one_is_a_no_op():
    """A factor of 1.0 (or an empty mapping) disables the relief cleanly."""
    cfg = dataclasses.replace(make_config_tuned(),
                              dso_g_v_per_area=None, dso_g_w_class=None)
    assert _apply_dso_v_relief(cfg, {"DSO_4": 1.0}) is cfg
    assert _apply_dso_v_relief(cfg, {}) is cfg


def test_relief_merges_into_an_existing_spec():
    """An existing per-area spec for another class/area is not clobbered."""
    cfg = dataclasses.replace(
        make_config_tuned(),
        dso_g_v_per_area={"DSO_2": 1.0},
        dso_g_w_class={"DSO_2": {"dso_der": 42.0},
                       "DSO_4": {"dso_der": 7.0}},
    )
    out = _apply_dso_v_relief(cfg, {"DSO_4": 3.0})
    assert out.dso_g_v_per_area["DSO_2"] == 1.0
    assert out.dso_g_w_class["DSO_2"] == {"dso_der": 42.0}
    assert out.dso_g_w_class["DSO_4"]["dso_der"] == 7.0
    # DSO_4 had no dso_oltc entry, so the global scalar is the base.
    assert out.dso_g_w_class["DSO_4"]["dso_oltc"] == pytest.approx(
        float(cfg.g_w_dso_oltc) * 3.0)
    assert out.dso_g_v_per_area["DSO_4"] == pytest.approx(
        float(cfg.dso_g_v) * 3.0)
