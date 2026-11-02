"""
Tests for the voltage-stability / nose-curve reachability guard
(:mod:`analysis.reachability`).

Test strategy
-------------
A two-bus system (slack + one PQ load bus joined by a predominantly reactive
line) has a closed-form P-V / Q-V nose.  Increasing the load drives the
operating point toward the saddle-node:

* On the **upper (stable) branch** the reduced Q-V Jacobian ``J_R`` has a
  positive minimum eigenvalue and the full Jacobian is well conditioned;
  ``check_reachability`` must report ``on_stable_branch=True``.
* The **lower (unstable) branch** is reached by seeding Newton-Raphson with a
  depressed voltage guess (``init_vm_pu``).  There the minimum eigenvalue of
  ``J_R`` is negative, so the guard must report ``on_stable_branch=False`` and
  the time-series monitor must raise :class:`ReachabilityViolation`.

The single PQ bus is necessarily the critical bus, which also exercises the
participation-factor / critical-bus diagnostic.

Author: Manuel Schwenke / Claude Code
Date: 2026-06-08
"""

from __future__ import annotations

import numpy as np
import pytest
import pandapower as pp

from analysis.reachability import (
    ReachabilityMonitor,
    ReachabilityResult,
    ReachabilityViolation,
    check_reachability,
)


# =============================================================================
# Fixtures / builders
# =============================================================================


def _build_two_bus(p_mw: float, q_mvar: float) -> pp.pandapowerNet:
    """Two-bus system: slack bus 0 (1.0 p.u.) feeding a PQ load at bus 1 over a
    mainly reactive line.  The load magnitude sets the distance to the nose.
    """
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0, name="slack")
    b1 = pp.create_bus(net, vn_kv=110.0, name="load")
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, b0, b1, length_km=1.0,
        r_ohm_per_km=2.0, x_ohm_per_km=20.0, c_nf_per_km=0.0, max_i_ka=10.0,
    )
    pp.create_load(net, b1, p_mw=p_mw, q_mvar=q_mvar)
    return net


def _solve_upper(p_mw: float, q_mvar: float) -> pp.pandapowerNet:
    """Converge to the upper branch from a flat start."""
    net = _build_two_bus(p_mw, q_mvar)
    pp.runpp(net, calculate_voltage_angles=True)
    return net


def _solve_lower(p_mw: float, q_mvar: float) -> pp.pandapowerNet:
    """Converge to the lower branch by seeding a depressed voltage guess."""
    net = _build_two_bus(p_mw, q_mvar)
    pp.runpp(
        net, calculate_voltage_angles=True,
        init_vm_pu=np.array([1.0, 0.25]),
        init_va_degree=np.array([0.0, -40.0]),
    )
    return net


# Loading well inside the feasible region for the chosen line / load.  At
# pf = 3.0 the two-bus system has both an upper-branch solution (v ~ 0.75 p.u.)
# and a lower-branch solution (v ~ 0.48 p.u.).
_P_BASE, _Q_BASE = 50.0, 25.0
_PF_FEASIBLE = 3.0


# =============================================================================
# Upper-branch behaviour
# =============================================================================


def test_upper_branch_is_stable() -> None:
    net = _solve_upper(_P_BASE * _PF_FEASIBLE, _Q_BASE * _PF_FEASIBLE)
    res = check_reachability(net, step_index=7)

    assert isinstance(res, ReachabilityResult)
    assert res.on_stable_branch is True
    assert res.lambda_min_JR > 0.0          # modal criterion: stable upper branch
    assert res.sigma_min_J > 1e-6           # full Jacobian non-singular
    assert np.isfinite(res.cond_J)
    assert res.step_index == 7
    # Only one PQ bus exists -> it must be the critical bus (pandapower bus 1).
    assert res.critical_bus == 1


def test_margin_shrinks_toward_the_nose() -> None:
    """The minimum eigenvalue of J_R and sigma_min(J) must both decrease
    monotonically as the load is ramped toward the saddle-node."""
    lambdas = []
    sigmas = []
    for pf in np.linspace(1.0, 3.3, 12):
        net = _solve_upper(_P_BASE * pf, _Q_BASE * pf)
        res = check_reachability(net)
        assert res.on_stable_branch is True
        lambdas.append(res.lambda_min_JR)
        sigmas.append(res.sigma_min_J)
    assert np.all(np.diff(lambdas) < 0.0)
    assert np.all(np.diff(sigmas) < 0.0)
    assert lambdas[-1] > 0.0                 # still upper branch at last point


# =============================================================================
# Lower-branch behaviour (at / beyond the nose)
# =============================================================================


def test_lower_branch_is_flagged() -> None:
    net = _solve_lower(_P_BASE * _PF_FEASIBLE, _Q_BASE * _PF_FEASIBLE)
    # Confirm the seed really landed on the lower branch.
    assert float(net.res_bus.vm_pu.iloc[1]) < 0.6

    res = check_reachability(net, step_index=9)
    assert res.on_stable_branch is False
    assert res.lambda_min_JR <= 1e-6         # zero/negative eigenvalue => nose
    assert res.critical_bus == 1


# =============================================================================
# Time-series monitor: passes on the upper branch, raises at the nose
# =============================================================================


def test_monitor_passes_upper_then_raises_at_nose() -> None:
    """Ramp the Q-load upward on the upper branch (all stable), then feed the
    post-nose lower-branch equilibrium and assert the guard aborts there."""
    monitor = ReachabilityMonitor()

    # Upper-branch ramp: every step must pass.
    upper_steps = list(np.linspace(1.0, 3.0, 6))
    for k, pf in enumerate(upper_steps):
        net = _solve_upper(_P_BASE * pf, _Q_BASE * pf)
        res = monitor.check_step(net, step_index=k, time_s=float(k))
        assert res.on_stable_branch is True

    # Post-nose lower-branch equilibrium: must raise at this first violation.
    nose_step = len(upper_steps)
    net_nose = _solve_lower(_P_BASE * _PF_FEASIBLE, _Q_BASE * _PF_FEASIBLE)
    with pytest.raises(ReachabilityViolation) as excinfo:
        monitor.check_step(net_nose, step_index=nose_step, time_s=float(nose_step))

    msg = str(excinfo.value)
    assert f"step {nose_step}" in msg
    assert "critical bus" in msg

    # The full margin trajectory (including the violating step) is recorded.
    df = monitor.to_dataframe()
    assert len(df) == len(upper_steps) + 1
    assert bool(df["on_stable_branch"].iloc[-1]) is False
    assert df["on_stable_branch"].iloc[:-1].all()


# =============================================================================
# Fail-fast input validation
# =============================================================================


def test_raises_on_none_net() -> None:
    with pytest.raises(ValueError):
        check_reachability(None)


def test_raises_when_not_converged() -> None:
    net = _build_two_bus(_P_BASE, _Q_BASE)  # never solved -> not converged
    with pytest.raises(ValueError):
        check_reachability(net)


def _case9_shared_slack_after_trip() -> pp.pandapowerNet:
    """IEEE 9 with every machine sharing the slack and one machine tripped:
    the distributed-slack solution then differs materially from the single-
    slack solution with the scheduled machine powers (the situation after a
    machine outage in the Monte-Carlo campaign, 2026-10-08)."""
    import pandapower.networks as pn

    net = pn.case9()
    net.ext_grid["slack_weight"] = 1.0
    net.gen["slack_weight"] = 1.0
    net.gen.at[net.gen.index[0], "in_service"] = False
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True)
    return net


def test_single_slack_copy_reproduces_the_distributed_operating_point() -> None:
    """The canonical copy must describe the plant's operating point, not the
    single-slack point with the scheduled machine powers (corrected
    2026-10-08)."""
    import copy as _copy
    from analysis.reachability import single_slack_copy

    net = _case9_shared_slack_after_trip()
    work = single_slack_copy(net)
    pp.runpp(work, calculate_voltage_angles=True, distributed_slack=False)
    np.testing.assert_allclose(work.res_bus.vm_pu, net.res_bus.vm_pu, atol=1e-8)
    np.testing.assert_allclose(work.res_bus.va_degree, net.res_bus.va_degree,
                               atol=1e-6)

    # The previous copy (scheduled machine powers) lands elsewhere.
    old = _copy.deepcopy(net)
    pp.runpp(old, calculate_voltage_angles=True, distributed_slack=False)
    assert np.max(np.abs(old.res_bus.va_degree - net.res_bus.va_degree)) > 1e-3


def test_distributed_slack_canonicalisation_matches_single_slack() -> None:
    """A distributed-slack solve augments the internal Jacobian; the guard must
    canonicalise it and return the margin of a single-slack solve AT THE SAME
    OPERATING POINT (machines at their solved powers).  Until 2026-10-08 this
    test compared against a single-slack solve with the scheduled powers, which
    only coincided because no machine shared the slack."""
    net = _case9_shared_slack_after_trip()
    res_dist = check_reachability(net)

    net2 = _case9_shared_slack_after_trip()
    on = net2.gen["in_service"].astype(bool)
    net2.gen.loc[on, "p_mw"] = net2.res_gen.loc[on, "p_mw"]
    pp.runpp(net2, calculate_voltage_angles=True, distributed_slack=False)
    res_single = check_reachability(net2)

    assert res_dist.on_stable_branch == res_single.on_stable_branch
    assert res_dist.critical_bus == res_single.critical_bus
    np.testing.assert_allclose(res_dist.lambda_min_JR, res_single.lambda_min_JR, rtol=1e-6)
    np.testing.assert_allclose(res_dist.sigma_min_J, res_single.sigma_min_J, rtol=1e-6)


def _flat_start_fails(monkeypatch, warm_start_fails: bool = False) -> list:
    """Make every solve without ``init`` (the flat start of the single-slack
    copy) raise; return the list of ``init`` values the guard solved with."""
    import analysis.reachability as reach
    from pandapower.powerflow import LoadflowNotConverged

    original = reach.pp.runpp
    calls: list = []

    def runpp(work, *args, **kwargs):
        calls.append(kwargs.get("init"))
        if kwargs.get("init") is None or warm_start_fails:
            raise LoadflowNotConverged("forced failure")
        return original(work, *args, **kwargs)

    monkeypatch.setattr(reach.pp, "runpp", runpp)
    return calls


def test_flat_start_failure_falls_back_to_the_plant_solution(monkeypatch) -> None:
    """When the flat start of the single-slack copy fails, the copy is solved
    from the plant's solution instead and gives the same margin (fallback
    added 2026-10-30)."""
    res_ref = check_reachability(_case9_shared_slack_after_trip())
    net = _case9_shared_slack_after_trip()
    calls = _flat_start_fails(monkeypatch)
    res = check_reachability(net)

    assert calls == [None, "results"]
    assert res.on_stable_branch == res_ref.on_stable_branch
    assert res.critical_bus == res_ref.critical_bus
    # pandapower stores the Jacobian of the last iteration, i.e. before the
    # final update: after a flat start (3 iterations here) that point is still
    # about 1e-4 from the solution, and lambda differs by 2.6e-5 relative from
    # the warm starts, which agree among themselves to 3e-6.
    np.testing.assert_allclose(res.lambda_min_JR, res_ref.lambda_min_JR, rtol=1e-4)
    np.testing.assert_allclose(res.sigma_min_J, res_ref.sigma_min_J, rtol=1e-4)


def test_both_starts_failing_raises(monkeypatch) -> None:
    """If the fallback fails as well, the guard still raises, with the message
    prefix the Monte-Carlo failure classification matches on."""
    net = _case9_shared_slack_after_trip()
    _flat_start_fails(monkeypatch, warm_start_fails=True)
    with pytest.raises(ValueError, match="Failed to re-converge a single-slack copy"):
        check_reachability(net)
