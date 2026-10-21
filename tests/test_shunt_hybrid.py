"""Joint MIQP and closed-loop tests of regular and slow bank requests."""
from dataclasses import replace
import numpy as np
import pytest

from controller.shunt_hybrid import BankModel, HybridContext, HybridShuntSolver
from controller.shunt_integrator import MSC, ShuntBankConfig
from controller.shunt_support import (FleetMargins, STSShuntFeedback,
                                      STSShuntSupportConfig)
from optimisation.miqp_solver import MIQPProblem, MIQPSolver


def make_case(target=40., measured=20., voltage=1., gradient=0., step_v=.0001,
              movement_weight=100., capability=100.,
              support_gate="voltage_allowance", support_exchange=0.,
              **bank_settings):
    settings = dict(shunt_idx=0, bus_idx=1, interface_trafo3w_idx=2,
        dso_id="STS", kind="MSC", q_step_mvar=25., n_levels=2,
        g_w=.01, delta=5., t_dwell_s=30., daily_switch_budget=10,
        y_h_min=.9, y_h_max=1.1, t_persistence_s=20., max_sample_gap_s=15.)
    settings.update(bank_settings)
    bank = MSC(ShuntBankConfig(**settings))
    f = STSShuntFeedback(np.array([2]), np.array([measured]), np.array([target]),
        np.array([-capability]), np.array([capability]), np.array([100.]), 0.)
    model = BankModel(bank, np.array([step_v, 25.]), np.array([2]),
        np.array([0]), np.array([25.]), np.array([0.]), f)
    ctx = HybridContext(0., [model], np.array([gradient, 0.]),
                        np.array([0]), np.array([1.]))
    p = MIQPProblem(2, 0, 2, 1., np.diag([1., 10000.]), np.eye(2),
        np.array([0., gradient]), np.array([[0., 1.], [1., 0.]]),
        np.array([target, 0.]), np.array([-100., -.1]), np.array([100., .1]),
        np.array([voltage, measured]), np.array([.9, -200.]),
        np.array([1.1, 200.]), [])
    solver = HybridShuntSolver(MIQPSolver(verbose=False), lambda: ctx,
        movement_weight=movement_weight,
        support_config=STSShuntSupportConfig(tracking_weight=10, reserve_weight=100),
        support_gate=support_gate, support_exchange=support_exchange)
    return solver, p, ctx, bank


def make_fleet_case(q_gen=45., gen_half=50., q_der=35., der_half=40.,
                    gen_step=-10., weight=100., margin_steps=1., step_v=.0001,
                    gradient=0., movement_weight=100., der_grad=0.,
                    pin_pcc=False, **bank_settings):
    """One machine and one TS DER, so the directional fleet margins can be
    exercised in isolation.

    Controls are ``[Q_DER | Q_PCC_set | other]`` and outputs ``[V | Q_PCC |
    Q_gen]``, matching the canonical orderings the solver assumes: the DER block
    heads the control vector and the machine block tails the output vector. The
    shared objective carries no reserve term, so the Q_gen entry of the output
    gradient is zero.

    Defaults put the fleet short on the *up* margin: machine 45 of [-50, +50]
    leaves 5, DER 35 of [-40, +40] leaves 5, so 10 Mvar against a threshold of
    one 25 Mvar bank step. An MSC engagement relieves it.
    """
    settings = dict(shunt_idx=0, bus_idx=1, interface_trafo3w_idx=2,
        dso_id="STS", kind="MSC", q_step_mvar=25., n_levels=2,
        g_w=.01, delta=5., t_dwell_s=30., daily_switch_budget=10,
        y_h_min=.9, y_h_max=1.1, t_persistence_s=20., max_sample_gap_s=15.)
    settings.update(bank_settings)
    bank = MSC(ShuntBankConfig(**settings))
    f = STSShuntFeedback(np.array([2]), np.array([20.]), np.array([20.]),
        np.array([-100.]), np.array([100.]), np.array([100.]), 0.)
    model = BankModel(bank, np.array([step_v, 25., gen_step]), np.array([2]),
        np.array([1]), np.array([25.]), np.array([0.]), f)
    fleet = FleetMargins(np.array([q_gen, q_der]),
                         np.array([-gen_half, -der_half]),
                         np.array([gen_half, der_half]),
                         np.array([True, True]), 1)
    ctx = HybridContext(0., [model], np.array([gradient, 0., 0.]),
                        np.array([0]), np.array([1.]), fleet, 1, 1)
    # The DER control bounds ARE its capability, as _get_control_limits sets
    # them from compute_der_q_bounds; a mismatch would make the solver move the
    # DER merely to re-enter its own box.
    lo = np.array([-der_half, 0. if pin_pcc else -100., -.1])
    hi = np.array([der_half, 0. if pin_pcc else 100., .1])
    p = MIQPProblem(3, 0, 3, 1., np.diag([1., 1., 10000.]), np.eye(3),
        np.array([der_grad, 0., gradient]),
        np.array([[0., 0., 1.], [0., 1., 0.], [0., 0., 0.]]),
        np.array([q_der, 0., 0.]), lo, hi,
        np.array([1., 20., q_gen]), np.array([.9, -200., -1e4]),
        np.array([1.1, 200., 1e4]), [])
    solver = HybridShuntSolver(MIQPSolver(verbose=False), lambda: ctx,
        movement_weight=movement_weight,
        support_config=STSShuntSupportConfig(tracking_weight=10, reserve_weight=100),
        ts_reserve_weight=weight, ts_reserve_margin_steps=margin_steps)
    return solver, p, ctx, bank


def advance(solver, p, ctx, t):
    ctx.time_s = t
    for model in ctx.models:
        if model.feedback is not None:
            model.feedback = replace(model.feedback, observed_at_s=t)
    return solver.solve(p)


def test_regular_ofo_can_switch_before_slow_confirmation():
    solver, p, ctx, b = make_case(voltage=.95, gradient=-100, step_v=.02,
                                 movement_weight=1)
    r = advance(solver, p, ctx, 0)
    c, _, dq = solver.take_commit()
    assert c.request_kind == "objective" and c.new_level == 1
    assert r.additional_output_change[0] == pytest.approx(.02)
    assert dq[0] == 25
    assert all(m.since is None for m in solver.memories.values())
    # A strong new regular request cannot bypass the same bank dwell.
    advance(solver, p, ctx, 10)
    assert solver.take_commit() is None


def test_regular_weight_controls_usage():
    solver, p, ctx, _ = make_case(voltage=.95, gradient=-100, step_v=.02,
                                  movement_weight=10)
    advance(solver, p, ctx, 0)
    assert solver.take_commit() is None


@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("alpha", [1., .2])
def test_slow_support_waits_then_transfers_reference_and_stops(reserve, alpha):
    target, measured, cap = (45., 45., 50.) if reserve else (40., 20., 100.)
    solver, p, ctx, b = make_case(target=target, measured=measured, capability=cap)
    p = replace(p, alpha=alpha)
    physical_bank, x, commits = 0., measured, []
    delegated = target
    for t in range(0, 100, 10):
        m = ctx.models[0]
        p = replace(p, u_current=np.array([delegated, 0.]),
                    y_current=np.array([1., x+physical_bank]))
        m.interface_offset = np.array([physical_bank])
        m.feedback = replace(m.feedback, q_measured=np.array([x+physical_bank]),
            q_requested=np.array([target]), q_min=np.array([-cap+physical_bank]),
            q_max=np.array([cap+physical_bank]))
        r = advance(solver, p, ctx, t)
        new_u = p.u_current+np.r_[alpha*r.w_continuous, r.w_integer]
        chosen = solver.take_commit()
        if chosen:
            c, _, dq = chosen
            assert t >= 20
            physical_bank += dq[0]
            commits.append(c)
        delegated = new_u[0]
        # Realised scalar STS response with a fixed total reference.
        limit = 50. if reserve else 20.
        x = float(np.clip(delegated, -limit, limit))
        assert delegated+physical_bank == pytest.approx(target, abs=1e-6)
    assert len(commits) == 1
    assert commits[0].request_kind == ("sts_reserve" if reserve else "sts_tracking")
    assert x == pytest.approx(20. if reserve else 15., abs=1e-6)
    assert x+physical_bank == pytest.approx(target)


def test_memories_are_independent_when_tracking_error_appears():
    solver, p, ctx, b = make_case(target=45, measured=45, capability=50)
    b._last_switch_t = 0
    advance(solver, p, ctx, 0)
    reserve = solver.memories[(0, "sts_reserve")]
    first = reserve.value
    ctx.models[0].feedback = replace(ctx.models[0].feedback, q_requested=np.array([65.]))
    advance(solver, p, ctx, 10)
    assert reserve.since == 0 and abs(reserve.value) >= abs(first)
    assert solver.memories[(0, "sts_tracking")].since == 10


def test_joint_solver_limits_competing_banks_to_one_step():
    solver, p, ctx, b = make_case(voltage=.95, gradient=-100, step_v=.02,
                                 movement_weight=1)
    b2 = MSC(replace(b.config, shunt_idx=1))
    ctx.models.append(replace(ctx.models[0], bank=b2))
    r = advance(solver, p, ctx, 0)
    assert np.sum(np.abs(solver.diagnostics["bank_moves"])) == 1
    assert r.additional_output_change[0] == pytest.approx(.02)
    assert solver.take_commit() is not None


def test_joint_voltage_constraint_is_hard_despite_soft_slack():
    solver, p, ctx, _ = make_case(voltage=1.095, gradient=-100, step_v=.02,
                                  movement_weight=.01)
    # Other controls cannot compensate the bank step.
    p = replace(p, u_lower=np.array([-100., 0.]), u_upper=np.array([100., 0.]))
    advance(solver, p, ctx, 0)
    assert solver.take_commit() is None


def test_support_cannot_exceed_voltage_tracking_allowance():
    solver, p, ctx, _ = make_case(step_v=.02)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
        assert solver.take_commit() is None
    assert solver.diagnostics["rejected_support"] > 0
    assert solver.diagnostics["support_candidates"][0]["rejection"] == "voltage_tracking"


def test_shared_objective_gate_admits_support_that_outweighs_its_cost():
    solver, p, ctx, _ = make_case(step_v=.02, support_gate="shared_objective",
                                  support_exchange=1e9)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    commit, _, dq = solver.take_commit()
    assert commit.request_kind == "sts_tracking" and commit.new_level == 1
    assert dq[0] == 25
    candidate = solver.diagnostics["support_candidates"][0]
    assert candidate["rejection"] is None
    # The pu bound no longer governs: the same case is refused under it.
    assert candidate["max_voltage_error_increase_pu"] > .001


def test_default_exchange_admits_support_that_does_not_worsen_the_objective():
    # The default exchange is zero, so support may act only when the shared
    # objective does not worsen. A step that improves it needs no paid trade,
    # even though the pu bound would have refused the same step.
    solver, p, ctx, _ = make_case(step_v=.02, gradient=-100.,
                                  support_gate="shared_objective")
    # The ordinary control cannot absorb the bank's voltage effect here.
    p = replace(p, u_lower=np.array([-100., 0.]), u_upper=np.array([100., 0.]))
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    commit, _, _ = solver.take_commit()
    assert commit.request_kind == "sts_tracking" and commit.new_level == 1
    candidate = solver.diagnostics["support_candidates"][0]
    assert candidate["rejection"] is None
    assert candidate["shared_objective_increase"] < 0
    assert candidate["max_voltage_error_increase_pu"] > .001


def test_shared_objective_gate_refuses_support_that_costs_more_than_it_buys():
    # A positive output gradient on the bank's own voltage makes engagement
    # uphill in the shared objective, which the gate must then price.
    solver, p, ctx, _ = make_case(step_v=.02, gradient=1000.,
                                  support_gate="shared_objective",
                                  support_exchange=0.)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
        assert solver.take_commit() is None
    candidate = solver.diagnostics["support_candidates"][0]
    assert candidate["support_cost_reduction"] > 0
    assert candidate["shared_objective_increase"] > 0
    assert candidate["rejection"] == "shared_objective"


def test_shared_objective_comparison_excludes_step_size_prices():
    # The candidate pays the bank movement weight and a whole-step move of the
    # delegated request, both of which regularise the step rather than express
    # f. A parked regular path must not make every candidate look expensive.
    increases = []
    for weight in (1., 1e5):
        solver, p, ctx, _ = make_case(step_v=.02, gradient=1000.,
                                      movement_weight=weight,
                                      support_gate="shared_objective",
                                      support_exchange=1e9)
        for t in (0, 10, 20):
            advance(solver, p, ctx, t)
        assert solver.take_commit() is not None
        increases.append(
            solver.diagnostics["support_candidates"][0]["shared_objective_increase"])
    assert increases[0] == pytest.approx(increases[1], abs=1e-6)


def test_original_recovery_remains_available_when_corridor_unreachable():
    solver, p, ctx, _ = make_case(voltage=1.2)
    p = replace(p, u_lower=np.array([-100., 0.]), u_upper=np.array([100., 0.]))
    r = advance(solver, p, ctx, 0)
    assert r.is_feasible and solver.take_commit() is None
    assert solver.diagnostics["regular_fallback"]


def test_joint_solution_accounts_for_other_actuator_motion():
    solver, p, ctx, _ = make_case(voltage=1.095, gradient=-100, step_v=.008,
                                  movement_weight=.01)
    r = advance(solver, p, ctx, 0)
    assert solver.take_commit() is not None
    y = p.y_current+p.H_tilde @ r.w_continuous+r.additional_output_change
    assert y[0] <= 1.1+1e-7
    assert r.w_continuous[1] < 0


def test_disappearing_need_clears_both_memories():
    solver, p, ctx, _ = make_case(target=45, measured=45, capability=50)
    advance(solver, p, ctx, 0)
    ctx.models[0].feedback = replace(ctx.models[0].feedback,
        q_measured=np.array([0.]), q_requested=np.array([0.]))
    advance(solver, replace(p, u_current=np.zeros(2)), ctx, 10)
    assert all(m.value == 0 and m.since is None for m in solver.memories.values())


def test_solver_rejects_unapplied_previous_physical_decision():
    solver, p, ctx, _ = make_case(voltage=.95, gradient=-100, step_v=.02,
                                  movement_weight=1)
    advance(solver, p, ctx, 0)
    with pytest.raises(RuntimeError, match="not applied"):
        advance(solver, p, ctx, 10)


def test_release_uses_same_dwell_and_budget_as_engagement():
    solver, p, ctx, b = make_case(voltage=.95, gradient=-100, step_v=.02,
        movement_weight=1, daily_switch_budget=2)
    advance(solver, p, ctx, 0)
    assert solver.take_commit()[0].new_level == 1
    ctx.output_gradient = np.array([100., 0.])
    p = replace(p, grad_f=np.array([0., 100.]), y_current=np.array([1.05, 20.]))
    advance(solver, p, ctx, 10)
    assert solver.take_commit() is None
    advance(solver, p, ctx, 30)
    commit, _, dq = solver.take_commit()
    assert commit.direction == -1 and commit.new_level == 0 and dq[0] == -25
    ctx.output_gradient = np.array([-100., 0.])
    p = replace(p, grad_f=np.array([0., -100.]), y_current=np.array([.95, 20.]))
    advance(solver, p, ctx, 60)
    assert solver.take_commit() is None
    assert b._switches_in_window == 2


def test_long_sample_gap_restarts_confirmation():
    solver, p, ctx, _ = make_case()
    for t in (0, 10, 30, 40):
        advance(solver, p, ctx, t)
        assert solver.take_commit() is None
    advance(solver, p, ctx, 50)
    assert solver.take_commit()[0].request_kind == "sts_tracking"


def test_current_command_cannot_use_old_tracking_pressure():
    solver, p, ctx, _ = make_case()
    for t in (0, 10):
        advance(solver, p, ctx, t)
    # The regular MIQP now requests the measured value. Old unmet demand
    # cannot justify transferring a bank step out of this new request.
    p = replace(p, u_current=np.array([20., 0.]))
    advance(solver, p, ctx, 20)
    assert solver.memories[(0, "sts_tracking")].ready(20, ctx.models[0].bank) == 1
    assert solver.take_commit() is None


def test_stale_feedback_clears_memory():
    solver, p, ctx, _ = make_case()
    advance(solver, p, ctx, 0)
    ctx.time_s = 100.
    solver.solve(p)
    assert solver.take_commit() is None
    assert solver.memories[(0, "sts_tracking")].value == 0


@pytest.mark.parametrize("backend", ["GUROBI", "SCIP"])
def test_support_candidate_with_explicit_solver_backend(backend):
    import cvxpy as cp
    if backend not in cp.installed_solvers():
        pytest.skip(f"{backend} is not installed")
    solver, p, ctx, _ = make_case(target=45, measured=45, capability=50)
    solver.solver = MIQPSolver(solver=backend, verbose=False)
    for t in (0, 10, 20):
        r = advance(solver, p, ctx, t)
    c, _, dq = solver.take_commit()
    assert c.request_kind == "sts_reserve"
    assert p.u_current[0]+r.w_continuous[0]+dq[0] == pytest.approx(45, abs=1e-5)


def test_ts_fleet_reserve_commits_when_a_directional_margin_is_short():
    solver, p, ctx, _ = make_fleet_case()
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    commit, _, _ = solver.take_commit()
    assert commit.request_kind == "ts_reserve" and commit.new_level == 1
    candidate = solver.diagnostics["support_candidates"][0]
    assert candidate["kind"] == "ts_reserve" and candidate["rejection"] is None
    assert candidate["fleet_cost_candidate"] < candidate["fleet_cost_regular"]
    # The margins are recorded whether or not either side is short.
    assert solver.diagnostics["fleet_margin_up_mvar"] == pytest.approx(10.)
    assert solver.diagnostics["fleet_margin_down_mvar"] == pytest.approx(170.)


def test_ts_fleet_reserve_counts_der_margin_not_only_machines():
    # The machine alone is short (5 Mvar), but a roomy DER lifts the fleet
    # margin above the threshold, so no request is made.
    solver, p, ctx, _ = make_fleet_case(q_der=0., der_half=40.)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    assert solver.take_commit() is None
    assert not solver.diagnostics["support_candidates"]
    assert solver.diagnostics["fleet_margin_up_mvar"] == pytest.approx(45.)


def test_ts_fleet_reserve_with_room_makes_no_request():
    solver, p, ctx, _ = make_fleet_case(q_gen=0., q_der=0.)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    assert solver.take_commit() is None
    assert not solver.diagnostics["support_candidates"]
    assert all(m.since is None for k, m in solver.memories.items()
               if k[1] == "ts_reserve")


def test_ts_fleet_reserve_takes_no_credit_for_relief_the_ordinary_controls_give():
    # The DER control is driven to its own rail, which restores the fleet margin
    # in the regular solution, so the bank adds nothing and is refused.
    solver, p, ctx, _ = make_fleet_case(der_grad=1e4)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    assert solver.take_commit() is None
    candidate = solver.diagnostics["support_candidates"][0]
    assert candidate["rejection"] == "no_fleet_relief"
    assert candidate["fleet_cost_regular"] == pytest.approx(0.)


def test_ts_fleet_reserve_is_absent_from_the_shared_objective():
    # The supervisory weight must not reach the continuous actuators: the
    # regular solve has to be identical with the cost on and off.
    results = []
    for weight in (0., 1e4):
        solver, p, ctx, _ = make_fleet_case(weight=weight, gradient=-50.)
        results.append(advance(solver, p, ctx, 0).w_continuous.copy())
        assert solver.take_commit() is None
    assert np.allclose(results[0], results[1], atol=1e-9)


def test_ts_fleet_reserve_leaves_the_delegated_request_free():
    # The delegated control is pinned to its current value, so the transfer an
    # STS request applies would be out of bounds. A fleet request does not
    # apply it - the offset-only convention - so it is still admitted.
    solver, p, ctx, _ = make_fleet_case(pin_pcc=True)
    for t in (0, 10, 20):
        advance(solver, p, ctx, t)
    commit, _, _ = solver.take_commit()
    assert commit.request_kind == "ts_reserve"
    assert all(c["rejection"] != "delegated_bounds"
               for c in solver.diagnostics["support_candidates"])


def test_ts_fleet_reserve_memory_is_a_confirmation_timer():
    # A threshold condition carries no gradient, so one observation saturates
    # the accumulator and readiness is governed by persistence alone.
    solver, p, ctx, _ = make_fleet_case()
    advance(solver, p, ctx, 0)
    memory = solver.memories[(0, "ts_reserve")]
    assert memory.value == pytest.approx(25.)
    assert memory.direction == 1
    assert memory.ready(0., ctx.models[0].bank) == 0      # persistence not met
    assert memory.ready(20., ctx.models[0].bank) == 1


@pytest.mark.parametrize("cached", [False, True])
def test_stage1_rejects_incomplete_hybrid_curvature(tmp_path, cached):
    from types import SimpleNamespace
    from tuning.stage1_screen import build_operating_point, load_or_build_operating_points
    cfg = SimpleNamespace(shunt_dispatch="hybrid")
    with pytest.raises(ValueError, match="hybrid bank columns"):
        if cached:
            load_or_build_operating_points(cfg, [], tmp_path)
        else:
            build_operating_point(cfg, "test")


@pytest.mark.parametrize("settings", [
    {"coordination_mode": "sbx_h"}, {"tso_pcc_capability_on_output": True},
    {"shunt_sts_support": False}, {"g_w_tso_shunt": 0},
    {"shunt_hybrid_voltage_error_allowance_pu": -1},
    {"shunt_hybrid_support_gate": "pu_allowance"},
    {"shunt_hybrid_support_exchange": -1},
    {"shunt_ts_reserve_weight": -1},
    {"shunt_ts_reserve_margin_steps": 0},
])
def test_hybrid_rejects_unsupported_configurations(settings):
    from configs.config import MultiTSOConfig
    values = dict(shunt_dispatch="hybrid", shunt_sts_support=True,
        shunt_int_t_persistence_s=360, coordination_mode="none",
        install_tso_tertiary_shunts=True)
    values.update(settings)
    with pytest.raises(ValueError):
        MultiTSOConfig(**values).validate_integrator_mode()


@pytest.mark.parametrize("settings", [
    {"shunt_int_drive": "sideways"},
    {"shunt_int_drive": "local_voltage", "shunt_dispatch": "hybrid"},
    {"shunt_int_drive": "local_voltage", "shunt_local_g": 0},
    {"shunt_int_drive": "local_voltage", "shunt_local_v_deadband_pu": -0.01},
])
def test_local_voltage_drive_rejects_unsupported_configurations(settings):
    from configs.config import MultiTSOConfig
    values = dict(shunt_dispatch="integrator", install_tso_tertiary_shunts=True,
                  coordination_mode="none")
    values.update(settings)
    with pytest.raises(ValueError):
        MultiTSOConfig(**values).validate_integrator_mode()


def test_local_voltage_drive_excludes_the_shared_objective_terms():
    # The autonomous rule must not be combinable with STS support, which also
    # replaces the driving signal.
    from configs.config import MultiTSOConfig
    cfg = MultiTSOConfig(shunt_dispatch="integrator", install_tso_tertiary_shunts=True,
                         coordination_mode="none", shunt_int_drive="local_voltage",
                         shunt_sts_support=True, shunt_int_t_persistence_s=360)
    with pytest.raises(ValueError, match="autonomous"):
        cfg.validate_integrator_mode()


def test_regular_path_commits_immediately_by_default():
    solver, p, ctx, _ = make_case(voltage=.95, gradient=-100, step_v=.02,
                                  movement_weight=1)
    advance(solver, p, ctx, 0)
    commit, _, _ = solver.take_commit()
    assert commit.request_kind == "objective"
    assert solver.diagnostics["regular_pending"]["count"] == 1


def test_regular_path_waits_for_its_confirmations():
    solver, p, ctx, _ = make_case(voltage=.95, gradient=-100, step_v=.02,
                                  movement_weight=1)
    solver.regular_confirmations = 3
    for t, expected in ((0, 1), (5, 2)):
        advance(solver, p, ctx, t)
        assert solver.take_commit() is None
        assert solver.diagnostics["regular_pending"]["count"] == expected
        assert solver.diagnostics["bank_moves"] == [0]      # dispatched held
    advance(solver, p, ctx, 10)
    commit, _, _ = solver.take_commit()
    assert commit.request_kind == "objective" and commit.new_level == 1


def test_regular_confirmation_resets_when_the_deviation_is_recovered():
    # An instant in which the MIQP holds every bank clears the count, which is
    # what makes the criterion "not recovered by the other actuators".
    solver, p, ctx, _ = make_case(voltage=.95, gradient=-100, step_v=.02,
                                  movement_weight=1)
    solver.regular_confirmations = 3
    advance(solver, p, ctx, 0)
    assert solver.diagnostics["regular_pending"]["count"] == 1
    ctx.output_gradient = np.array([0., 0.])                # deviation resolved
    p_quiet = replace(p, grad_f=np.array([0., 0.]))
    advance(solver, p_quiet, ctx, 5)
    assert solver.take_commit() is None
    assert not solver._regular_pending
    ctx.output_gradient = np.array([-100., 0.])             # and it returns
    advance(solver, p, ctx, 10)
    assert solver.diagnostics["regular_pending"]["count"] == 1
