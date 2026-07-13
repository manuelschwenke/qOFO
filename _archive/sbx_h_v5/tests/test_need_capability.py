"""
SBX Phase 3 — tests for ``sbx_h.need`` and ``sbx_h.capability``.

Acceptance (plan v2 §4 Phase 3, amended by v2.2 item 4):
* base case → no need flags; joint-box LP returns t ≥ 1 (full quantum
  offers on all corridors);
* stressed case → flag fires with the correct sign after EXACTLY
  ``n_need`` consecutive iterations;
* t shrinks monotonically as stress increases (3-point check);
* the capability LP never fails silently (``rep1`` on solver status);
* offers on all corridors of a violating area are (0, 0).

The need tests run on real base-case voltages of the IEEE 39 net (bounds
tightened to create deterministic violations without re-running power
flows); the capability tests use a small synthetic area so every
constraint activation is analytically checkable.  The joint-box LP on a
real zone's cached H is exercised by the Phase 5 closed-loop smoke test.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from optimisation.miqp_solver import MIQPResult, MIQPSolver
from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.capability import CapabilityResult, CorridorCoupling, \
    joint_box_capability
from sbx_h.config import SBXConfig
from sbx_h.fail import SBXError
from sbx_h.need import NeedTracker, assert_relieving_sign


# ---------------------------------------------------------------------------
#  Need flag
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zone1_voltages():
    net, _ = build_ieee39_net(scenario="wind_replace")
    zone_map, _ = fixed_zone_partition_ieee39(net)
    pp.runpp(net)
    buses = zone_map[1]
    v = [float(net.res_bus.at[b, "vm_pu"]) for b in buses]
    return buses, v


def _run_tracker(cfg, buses, v, v_min, v_max, n_iters):
    tracker = NeedTracker(cfg, area_id=1)
    lo = [v_min] * len(buses)
    hi = [v_max] * len(buses)
    return [tracker.update(k, buses, v, lo, hi) for k in range(n_iters)]


def test_need_no_flag_in_base_case(zone1_voltages):
    buses, v = zone1_voltages
    cfg = SBXConfig()
    decisions = _run_tracker(cfg, buses, v, 0.90, 1.10, 3 * cfg.n_need)
    assert all(not d.flag and d.direction == 0 for d in decisions)


def test_need_fires_after_exactly_n_need_undervoltage(zone1_voltages):
    buses, v = zone1_voltages
    cfg = SBXConfig()
    # Tightened lower bound: base-case zone-1 voltages (~1.016-1.03 pu)
    # violate v_min = 1.04 by > threshold -> persistent undervoltage.
    decisions = _run_tracker(cfg, buses, v, 1.04, 1.10, cfg.n_need + 2)
    for k, d in enumerate(decisions):
        assert d.direction == +1
        assert d.flag == (k + 1 >= cfg.n_need), (
            f"iteration {k}: consecutive={d.consecutive}"
        )
    # Corridor request sign (§2.3): import need -> more negative q_corr
    # when the area is the reference end A, more positive when end B.
    d = decisions[-1]
    assert d.request_sign("a") == -1
    assert d.request_sign("b") == +1


def test_need_fires_overvoltage_with_export_sign(zone1_voltages):
    buses, v = zone1_voltages
    cfg = SBXConfig()
    decisions = _run_tracker(cfg, buses, v, 0.90, 1.00, cfg.n_need)
    d = decisions[-1]
    assert d.flag and d.direction == -1
    assert d.request_sign("a") == +1
    assert d.request_sign("b") == -1


def test_need_resets_on_recovery_and_gap(zone1_voltages):
    buses, v = zone1_voltages
    # Persistence semantics need n_need > 1; pinned explicitly (the
    # SBXConfig default is Manuel's experimental knob).
    cfg = SBXConfig(n_need=5)
    tracker = NeedTracker(cfg, area_id=1)
    lo_viol = [1.04] * len(buses)
    lo_ok = [0.90] * len(buses)
    hi = [1.10] * len(buses)
    for k in range(cfg.n_need - 1):
        tracker.update(k, buses, v, lo_viol, hi)
    # Recovery iteration resets the count ...
    d = tracker.update(cfg.n_need - 1, buses, v, lo_ok, hi)
    assert d.consecutive == 0 and not d.flag
    # ... so the violation must persist n_need times again.
    for k in range(cfg.n_need):
        d = tracker.update(cfg.n_need + k, buses, v, lo_viol, hi)
    assert d.flag
    # A gap in the iteration sequence also resets.
    d = tracker.update(cfg.n_need * 5, buses, v, lo_viol, hi)
    assert d.consecutive == 1 and not d.flag


def test_need_request_sign_without_flag_raises(zone1_voltages):
    buses, v = zone1_voltages
    cfg = SBXConfig()
    d = _run_tracker(cfg, buses, v, 0.90, 1.10, 1)[-1]
    with pytest.raises(SBXError, match="without a set need flag"):
        d.request_sign("a")


def test_relieving_sign_assert(zone1_voltages):
    buses, v = zone1_voltages
    cfg = SBXConfig()
    d = _run_tracker(cfg, buses, v, 1.04, 1.10, cfg.n_need)[-1]
    assert d.direction == +1
    assert_relieving_sign(d, dv_worst_per_dq_request=+2e-4)  # relieving
    with pytest.raises(SBXError, match="relieving sign"):
        assert_relieving_sign(d, dv_worst_per_dq_request=-2e-4)


# ---------------------------------------------------------------------------
#  Joint-box capability LP (v2.2 D13)
# ---------------------------------------------------------------------------


def _synthetic_area(v_meas: float):
    """Small analytically checkable area: 3 actuators, 2 buses, 2 corridors.

    With v_meas = 1.00 the binding constraint is the voltage box:
    du = (10t, 10t, 0) satisfies both corridor equalities at every sign
    vertex and gives |H_loc du| = 0.015 t ≤ 0.045 → t ≈ 3 ≥ 1.
    """
    u_now = np.zeros(3)
    u_min = np.full(3, -50.0)
    u_max = np.full(3, +50.0)
    v = np.array([v_meas, v_meas])
    v_min = np.array([0.95, 0.95])
    v_max = np.array([1.05, 1.05])
    h_loc = np.array([
        [0.0010, 0.0005, 0.0000],
        [0.0005, 0.0010, 0.0002],
    ])
    couplings = (
        CorridorCoupling(key=(1, 2), control_row=np.array([1.0, 0.0, 0.5]),
                         dq_quant_mvar=10.0),
        CorridorCoupling(key=(1, 3), control_row=np.array([0.0, 1.0, -0.5]),
                         dq_quant_mvar=10.0),
    )
    return u_now, u_min, u_max, v, v_min, v_max, h_loc, couplings


def test_capability_base_case_full_offers():
    u, ulo, uhi, v, vlo, vhi, h, coup = _synthetic_area(1.00)
    res = joint_box_capability(u, ulo, uhi, v, vlo, vhi, h, coup,
                               MIQPSolver(), voltage_margin_pu=0.005)
    assert not res.skipped_due_to_violation
    assert res.t >= 1.0
    for key in ((1, 2), (1, 3)):
        lo, hi_ = res.offers_mvar[key]
        assert lo == pytest.approx(-10.0)
        assert hi_ == pytest.approx(+10.0)


def test_capability_t_shrinks_monotonically_under_stress():
    ts = []
    for v_meas in (1.00, 1.035, 1.043):
        u, ulo, uhi, v, vlo, vhi, h, coup = _synthetic_area(v_meas)
        res = joint_box_capability(u, ulo, uhi, v, vlo, vhi, h, coup,
                                   MIQPSolver(), voltage_margin_pu=0.005)
        assert not res.skipped_due_to_violation
        ts.append(res.t)
    assert ts[0] > ts[1] > ts[2] > 0.0, f"t sequence not decreasing: {ts}"
    # The most stressed point can no longer support the full quantum.
    assert ts[2] < 1.0


def test_capability_violating_area_offers_zero():
    u, ulo, uhi, v, vlo, vhi, h, coup = _synthetic_area(1.048)  # > 1.045
    res = joint_box_capability(u, ulo, uhi, v, vlo, vhi, h, coup,
                               MIQPSolver(), voltage_margin_pu=0.005)
    assert res.skipped_due_to_violation
    assert res.t == 0.0
    assert all(offer == (0.0, 0.0) for offer in res.offers_mvar.values())


def test_capability_lp_failure_is_not_silent():
    u, ulo, uhi, v, vlo, vhi, h, coup = _synthetic_area(1.00)

    class _FailingSolver(MIQPSolver):
        def solve(self, problem):
            return MIQPResult(
                w_continuous=np.zeros(problem.n_total),
                w_integer=np.array([], dtype=np.int64),
                z=np.zeros(problem.n_outputs),
                objective_value=np.inf,
                status="infeasible",
                solve_time_s=0.0,
            )

    with pytest.raises(SBXError, match="never fails silently"):
        joint_box_capability(u, ulo, uhi, v, vlo, vhi, h, coup,
                             _FailingSolver(), voltage_margin_pu=0.005)


def test_capability_single_corridor():
    u, ulo, uhi, v, vlo, vhi, h, coup = _synthetic_area(1.00)
    res = joint_box_capability(u, ulo, uhi, v, vlo, vhi, h, coup[:1],
                               MIQPSolver(), voltage_margin_pu=0.005)
    assert res.t >= 1.0
    assert set(res.offers_mvar.keys()) == {(1, 2)}
