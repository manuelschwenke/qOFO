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
    DSO_GAMMA_OLTC_Q_PER_AREA,
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
def test_relief_does_not_inflate_the_oltc_q_threshold(builder, dso_id):
    """At ``gamma_oltc_q > 0`` the relief's x-factor must be compensated.

    Replaces the blanket ``assert dso_gamma_oltc_q == 0.0`` this module opened
    with until 2026-08-20.  That guard said "the loop-gain argument needs
    revisiting"; it has been, twice, and this is where it landed.

    Holding ``dso_g_v / g_w_dso_oltc`` preserves the tap's VOLTAGE commit
    threshold.  Once the tap also carries a Q gradient its INTERFACE-Q
    threshold ``(g_w_oltc + ||a||^2) / (2 g_q gamma |dQ/ds|)`` matters too, and
    the x-factor on ``g_w_dso_oltc`` is uncompensated there unless something on
    the Q channel moves with it.  Measured at gamma = 1, x20 relief, nothing
    compensating: DSO_2/DSO_4 commit at 108-244 Mvar against ~6 Mvar of
    interface-Q RMSE, i.e. never.

    **Two instruments can compensate, and this test accepts either**, because
    they are different claims and the config may legitimately use one or the
    other:

    ``dso_g_q_per_area``
        raises that area's interface-Q objective weight.  Shared by every
        column, so it also moves the continuous DER block -- measured to
        oscillate it at the full factor.
    ``dso_gamma_oltc_q_per_area``
        raises the Q gain on the OLTC columns only, leaving DER untouched.
        The instrument in service since 2026-08-20.

    What is locked is the *outcome*: the relieved area's tap must not end up
    harder to trigger on interface-Q than it would be with no relief at all.
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
    if gamma_area <= 0.0:
        pytest.skip("gamma_oltc_q = 0: the OLTC carries no interface-Q gradient")

    # The two bases are NOT symmetric, and getting that wrong is how the first
    # version of this test came out vacuous: it divided the g_q base by the
    # voltage factor as well, so both sides carried the same assumed number and
    # every factor passed.
    #
    #   g_w_dso_oltc -- may carry a per-area analytic design
    #                   (make_config_per_area), so its base is the area's own
    #                   entry divided by the voltage factor;
    #   g_q, gamma   -- have no per-area *design*, only per-area overrides, so
    #                   their bases are the global scalars.
    v_factor = DSO_V_RELIEF_FACTORS[dso_id]
    oltc_base = cfg.dso_g_w_class[dso_id]["dso_oltc"] / v_factor
    q_area = float((cfg.dso_g_q_per_area or {}).get(dso_id, cfg.g_q))
    gamma_base = float(cfg.dso_gamma_oltc_q)
    assert gamma_base > 0.0, (
        "the global gamma_oltc_q is the fallback this comparison is measured "
        "against; at 0 there is no unrelieved baseline to compare to"
    )

    inflation = ((cfg.dso_g_w_class[dso_id]["dso_oltc"] / (q_area * gamma_area))
                 / (oltc_base / (float(cfg.g_q) * gamma_base)))
    assert inflation <= 1.0 + 1e-9, (
        f"{dso_id}: the x{v_factor:g} voltage relief leaves the tap's "
        f"interface-Q commit threshold x{inflation:.2f} the unrelieved one. "
        f"Compensate on the Q channel -- dso_gamma_oltc_q_per_area (OLTC only, "
        f"preferred) or dso_g_q_per_area (also moves the DER block)."
    )


def test_per_area_gamma_is_within_the_controller_bound():
    """Every designed gamma must be one the controller will accept.

    The table lives in an experiment module and the bound in the controller, so
    nothing but this test connects them; a value past the cap would fail at
    controller construction, i.e. minutes into a run.
    """
    from controller.dso_controller import GAMMA_OLTC_Q_MAX

    for dso_id, gamma in DSO_GAMMA_OLTC_Q_PER_AREA.items():
        assert 0.0 <= float(gamma) <= GAMMA_OLTC_Q_MAX, (
            f"{dso_id}: gamma {gamma!r} outside [0, {GAMMA_OLTC_Q_MAX:g}]"
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


def test_q_leg_at_the_voltage_factor_preserves_the_threshold_exactly():
    """``scale_q=True`` is the setting at which the two factors cancel.

    Constructed here rather than read off a module default, so it keeps testing
    the identity after ``DSO_V_RELIEF_SCALE_Q`` is retuned (it moved 20 -> 5 on
    2026-08-20 when the full factor oscillated the DSO-DER block).
    """
    bare = dataclasses.replace(make_config_tuned(), dso_g_v_per_area=None,
                               dso_g_w_class=None, dso_g_q_per_area=None)
    cfg = _apply_dso_v_relief(bare, scale_q=True)
    for dso_id in DSO_V_RELIEF_FACTORS:
        ratio_area = (cfg.dso_g_w_class[dso_id]["dso_oltc"]
                      / cfg.dso_g_q_per_area[dso_id])
        ratio_global = float(cfg.g_w_dso_oltc) / float(cfg.g_q)
        assert ratio_area == pytest.approx(ratio_global), (
            f"{dso_id}: at scale_q=True the interface-Q commit threshold must "
            f"equal the unrelieved one"
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
