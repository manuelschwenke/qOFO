"""Unit tests for ``tuning/reparam.py`` and the OLTC switching bisection.

The properties pinned here are the ones whose absence made the original search
space unidentifiable: the reference point must be representable, the gauge must
not be searched, and the shape knob must be orthogonal to the gain.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from configs.config import MultiTSOConfig
from tuning.bisect_switching import (
    BisectionResult,
    SwitchingProbe,
    calibrate_switching_price,
)
from tuning.metrics import TrajectoryMetrics
from tuning.reparam import (
    BO_DIMS_V2,
    Gauge,
    PriorityScales,
    priority_report,
    apply_reparam_to_config,
    coords_from_config,
    reparam_search_space,
)
from tuning.runner import RunResult


@pytest.fixture
def gauge(baseline_cfg: MultiTSOConfig) -> Gauge:
    return Gauge.from_config(baseline_cfg)


def _mid_coords() -> dict[str, float]:
    out = {}
    for p in BO_DIMS_V2:
        lo, hi = float(p.low), float(p.high)
        out[p.name] = math.sqrt(lo * hi) if p.log else 0.5 * (lo + hi)
    return out


# ---------------------------------------------------------------------------
# Gauge / representability
# ---------------------------------------------------------------------------

def test_reference_point_is_representable(baseline_cfg, gauge) -> None:
    """The whole point of gauge-fixing: the reference is inside the box.

    The raw-weight space failed this — the hand-tuned point had ``g_v = 1e7``
    against a box of ``[1e2, 1e5]``, so no trial could ever propose it.
    """
    coords = coords_from_config(baseline_cfg, gauge)
    space = reparam_search_space()
    for name, value in coords.items():
        lo, hi, _ = space[name]
        assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"


def test_gauge_weights_are_pinned_not_searched(baseline_cfg, gauge) -> None:
    """``g_v`` / ``g_q`` / ``g_w_gen`` are the numeraire, so they never move.

    Pinning them is what removes the exact scaling redundancy measured on
    2026-07-31 (trajectories identical to ~4e-10 under a common rescaling).
    """
    names = {p.name for p in BO_DIMS_V2}
    assert "g_v" not in names and "g_q" not in names
    assert "g_w_gen" not in names          # AVR is 0.6-2.3 % of curvature

    for coords in (_mid_coords(), {**_mid_coords(), "dso_v_priority": 5.0}):
        out = apply_reparam_to_config(baseline_cfg, coords, gauge)
        assert out.g_v == pytest.approx(gauge.g_v)
        assert out.g_q == pytest.approx(gauge.g_q)
        assert out.g_w_gen == pytest.approx(gauge.g_w_gen)
        assert out.tso_g_q_pcc == pytest.approx(gauge.tso_g_q_pcc)


def test_oltc_weights_are_not_coordinates() -> None:
    """They price integer switching and are bisected, not searched."""
    names = {p.name for p in BO_DIMS_V2}
    assert "g_w_tso_oltc" not in names
    assert "g_w_dso_oltc" not in names


# ---------------------------------------------------------------------------
# Coordinate semantics
# ---------------------------------------------------------------------------

def test_lambda_routes_to_per_layer_targets(baseline_cfg, gauge) -> None:
    coords = {**_mid_coords(), "tso_lambda": 0.4, "dso_lambda": 0.8}
    out = apply_reparam_to_config(baseline_cfg, coords, gauge)
    assert out.precondition_g_w is True
    assert out.precondition_lambda_target_tso == pytest.approx(0.4)
    assert out.precondition_lambda_target_dso == pytest.approx(0.8)


def test_lambda_uses_set_mode_and_continuous_scope(baseline_cfg, gauge) -> None:
    """Both are required for ``lambda`` to be a usable coordinate.

    Under ``mode='cap'`` every target above the current ``lambda_max`` is the
    same no-op; under ``lambda_scope='all'`` the integer OLTC columns — whose
    rank-1 term is an upper bound, not a real per-tick effect — can block the
    target outright (TSO zone 1: ``integer_dominated`` at 1.085 against a
    continuous loop of 0.021).
    """
    out = apply_reparam_to_config(baseline_cfg, _mid_coords(), gauge)
    assert out.precondition_mode == "set"
    assert out.precondition_lambda_scope == "preconditioned"


def test_tau_is_gauge_fixed_to_geometric_mean_one(baseline_cfg, gauge) -> None:
    """Shape must be orthogonal to gain, or the two coordinates alias."""
    for tau in (0.25, 1.0, 16.0):
        out = apply_reparam_to_config(
            baseline_cfg, {**_mid_coords(), "tau_der_pcc": tau}, gauge)
        scales = out.precondition_class_scales
        assert scales["der"] / scales["pcc"] == pytest.approx(tau)
        assert math.sqrt(scales["der"] * scales["pcc"]) == pytest.approx(1.0)


def test_tau_one_is_the_analytic_preconditioner(baseline_cfg, gauge) -> None:
    out = apply_reparam_to_config(
        baseline_cfg, {**_mid_coords(), "tau_der_pcc": 1.0}, gauge)
    assert out.precondition_class_scales["der"] == pytest.approx(1.0)
    assert out.precondition_class_scales["pcc"] == pytest.approx(1.0)


def test_priority_ratio_round_trips(baseline_cfg, gauge) -> None:
    """factor -> raw weight -> factor must be the identity."""
    for f in (0.05, 1.0, 20.0):
        out = apply_reparam_to_config(
            baseline_cfg, {**_mid_coords(), "dso_v_priority": f}, gauge)
        back = coords_from_config(out, gauge)
        assert back["dso_v_priority"] == pytest.approx(f, rel=1e-9)


def test_reference_sits_at_the_centre_of_every_ratio_coordinate(
    baseline_cfg, gauge,
) -> None:
    """Representability is structural, not something to check afterwards.

    Ratio coordinates are defined *relative to* the reference, so it is at 1.0 —
    the geometric centre of a log range — by construction.  Absolute round-number
    bounds are what let the previous space exclude the hand-tuned point.
    """
    coords = coords_from_config(baseline_cfg, gauge)
    space = reparam_search_space()
    for name in ("tau_der_pcc", "dso_v_priority"):
        lo, hi, is_log = space[name]
        assert is_log
        assert coords[name] == pytest.approx(1.0)
        assert math.sqrt(lo * hi) == pytest.approx(1.0)


def test_tso_interface_q_objective_is_not_a_coordinate() -> None:
    """It is zero in the reference, so a log coordinate cannot represent it."""
    assert "pi_qpcc" not in {p.name for p in BO_DIMS_V2}
    assert "tso_g_q_pcc" not in {p.name for p in BO_DIMS_V2}


def test_priority_report_exposes_the_dso_imbalance(baseline_cfg) -> None:
    """``pi = g * sigma^2`` makes an inversion visible that raw weights hide.

    At the hand-tuned baseline the DSO weights interface-Q tracking ~500x above
    its voltage schedule — the opposite of what the raw ``g_q = 200`` vs
    ``dso_g_v = 1e5`` suggests at a glance.
    """
    cfg = dataclasses.replace(baseline_cfg, g_q=200.0, dso_g_v=1e5)
    pi = priority_report(cfg)
    assert pi["pi_q_dso"] / pi["pi_v_ds"] == pytest.approx(500.0, rel=0.01)


def test_unknown_or_missing_coordinates_raise(baseline_cfg, gauge) -> None:
    with pytest.raises(ValueError, match="Unknown reparam coords"):
        apply_reparam_to_config(
            baseline_cfg, {**_mid_coords(), "bogus": 1.0}, gauge)
    partial = _mid_coords()
    partial.pop("tso_lambda")
    with pytest.raises(KeyError, match="Missing reparam coords"):
        apply_reparam_to_config(baseline_cfg, partial, gauge)


def test_non_positive_coordinate_raises(baseline_cfg, gauge) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        apply_reparam_to_config(
            baseline_cfg, {**_mid_coords(), "tau_der_pcc": 0.0}, gauge)


# ---------------------------------------------------------------------------
# OLTC switching bisection
# ---------------------------------------------------------------------------

def _fake_runner(rate_of_g):
    """Runner stub whose switching rate is a declared function of ``g_w``."""
    def run(params, scenario, cfg, cost_weights=None, noise_floors=None):
        g = float(cfg.g_w_dso_oltc)
        return RunResult(
            scenario_name=scenario.name,
            metrics=TrajectoryMetrics(
                n_records=100, tap_ops_per_h_dso=rate_of_g(g)),
            wall_time_s=0.0,
        )
    return run


def test_bisection_finds_the_target_rate(baseline_cfg) -> None:
    from tuning.scenarios import design_set
    scenarios = design_set()[:2]
    # Monotone decreasing in g, crossing 10 ops/day at g = 1e3.
    res = calibrate_switching_price(
        "g_w_dso_oltc", target_ops_per_day=10.0,
        baseline_cfg=baseline_cfg, scenarios=scenarios,
        lo=1.0, hi=1e5, runner=_fake_runner(lambda g: 10.0 * (1e3 / g) ** 0.5),
        verbose=False,
    )
    assert res.status == "bracketed"
    assert res.within_tolerance
    assert res.g_w == pytest.approx(1e3, rel=0.5)


def test_bisection_reports_plateau_high_when_budget_is_slack(baseline_cfg) -> None:
    """Already quieter than the budget at the cheapest weight.

    The returned value must not be read as "the tuned value" — the constraint
    simply does not bind, and this weight is not what limits switching.
    """
    from tuning.scenarios import design_set
    res = calibrate_switching_price(
        "g_w_dso_oltc", target_ops_per_day=10.0,
        baseline_cfg=baseline_cfg, scenarios=design_set()[:1],
        runner=_fake_runner(lambda g: 2.0), verbose=False,
    )
    assert res.status == "plateau_high"


def test_bisection_reports_plateau_low_when_unreachable(baseline_cfg) -> None:
    from tuning.scenarios import design_set
    res = calibrate_switching_price(
        "g_w_dso_oltc", target_ops_per_day=1.0,
        baseline_cfg=baseline_cfg, scenarios=design_set()[:1],
        runner=_fake_runner(lambda g: 50.0), verbose=False,
    )
    assert res.status == "plateau_low"


def test_bisection_brackets_before_searching(baseline_cfg) -> None:
    """Both ends are probed first, because the tails are exactly flat."""
    from tuning.scenarios import design_set
    res = calibrate_switching_price(
        "g_w_dso_oltc", target_ops_per_day=10.0,
        baseline_cfg=baseline_cfg, scenarios=design_set()[:1],
        lo=2.0, hi=2e4, runner=_fake_runner(lambda g: 10.0 * (1e3 / g) ** 0.5),
        verbose=False,
    )
    assert res.ladder[0].g_w == pytest.approx(2.0)
    assert res.ladder[1].g_w == pytest.approx(2e4)


def test_probe_hands_the_runner_a_complete_param_set_with_the_swept_value(
    baseline_cfg,
) -> None:
    """The contract the ``runner`` stubs above hide.

    Every other test in this section injects a fake runner, so the real
    ``run_one`` path went unexercised: it overlays ``params`` via
    ``apply_to_config``, which demands *all* 8 BO dims, and the historical
    ``params or {}`` therefore raised ``KeyError`` on the first probe.

    Two properties are pinned:

    1. the dict is complete, so ``apply_to_config`` accepts it;
    2. ``params[field]`` carries the *swept* weight, not the baseline one.
       Property 2 fails silently -- the rate would be constant in ``g_w`` and
       the bisection would return a bogus ``plateau_*`` rather than an error.
    """
    from tuning.parameters import BO_DIMS, apply_to_config
    from tuning.scenarios import design_set

    expected_names = {p.name for p in BO_DIMS}
    seen: list[dict[str, float]] = []

    def _validating_runner(params, scenario, cfg, cost_weights=None,
                           noise_floors=None):
        # Raises KeyError/ValueError if params is not exactly the BO dim set.
        applied = apply_to_config(cfg, params)
        seen.append(dict(params))
        return RunResult(
            scenario_name=scenario.name,
            metrics=TrajectoryMetrics(
                n_records=100,
                tap_ops_per_h_dso=10.0 * (1e3 / applied.g_w_dso_oltc) ** 0.5,
            ),
            wall_time_s=0.0,
        )

    res = calibrate_switching_price(
        "g_w_dso_oltc", target_ops_per_day=10.0,
        baseline_cfg=baseline_cfg, scenarios=design_set()[:1],
        lo=2.0, hi=2e4, runner=_validating_runner, verbose=False,
    )

    assert seen, "runner was never called"
    for params in seen:
        assert set(params) == expected_names

    # The swept weight reached the runner: the first two probes bracket the
    # ends, so those params must carry lo and hi -- not the baseline value.
    assert seen[0]["g_w_dso_oltc"] == pytest.approx(2.0)
    assert seen[1]["g_w_dso_oltc"] == pytest.approx(2e4)
    assert len({p["g_w_dso_oltc"] for p in seen}) > 1, \
        "ladder was inert: every probe ran at the same weight"
    assert res.status == "bracketed"


def test_bisection_rejects_bad_arguments(baseline_cfg) -> None:
    from tuning.scenarios import design_set
    sc = design_set()[:1]
    with pytest.raises(ValueError, match="field must be one of"):
        calibrate_switching_price("g_w_der", 10.0, baseline_cfg, sc)
    with pytest.raises(ValueError, match="target_ops_per_day must be positive"):
        calibrate_switching_price("g_w_dso_oltc", 0.0, baseline_cfg, sc)
    with pytest.raises(ValueError, match="need 0 < lo < hi"):
        calibrate_switching_price("g_w_dso_oltc", 10.0, baseline_cfg, sc,
                                  lo=100.0, hi=1.0)
