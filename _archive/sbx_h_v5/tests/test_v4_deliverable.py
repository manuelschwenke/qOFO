"""
SBX v4 'deliverable SBX' (2026-07-09) — tests for the three amendments:

(i)   per-bus voltage-tracking weights (controller-side priority for
      the corridor-terminal contract references) — config validation
      here; the closed-loop effect is validated by the 014 run.
(iii) delivery-conditioned requesting: while the last settled cycle is
      sign_mismatch / magnitude_off, new requests are suppressed and
      the non-delivery counts towards the unwind dwell.
(A)   capability mode "auto": the D13 joint box first, per-corridor
      fallback when it collapses (finding F2, collinear corridor
      couplings) — the cross-corridor side effects are priced by the
      existing tier-3 attribution instead of geometrically forbidden.
"""
from __future__ import annotations

import numpy as np
import pytest

from optimisation.miqp_solver import MIQPSolver
from sbx_h.capability import CorridorCoupling, area_capability
from sbx_h.config import SBXConfig
from sbx_h.scheduler import CONSISTENCY_SIGN_MISMATCH

from tests.sbx_h.test_scheduler import (  # noqa: F401
    BOUND_UNDER,
    Harness,
    plant,
    ref_config,
)


# ---------------------------------------------------------------------------
#  (iii) delivery gate
# ---------------------------------------------------------------------------


def test_delivery_gate_suppresses_and_unwinds(plant):
    """The harness feeds tie_q = 0 while the schedule sits far away —
    physically 'nothing is delivered'.  v4: after the first deal the
    consistency turns sign_mismatch, further requests are suppressed
    (need flag STILL set), and the non-delivery counts as dwell so the
    surplus unwinds while the violation persists."""
    cfg = ref_config()         # gate ON by default
    assert cfg.delivery_gate
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER  # need never clears
    h.run_cycles(3 + cfg.n_need + cfg.m_release + 4)

    key = (1, 2)
    recs = h.sched.records[key]
    deals = [r for r in recs if r.deal.dq_deal_mvar != 0.0]
    assert deals, "no initial deal executed"
    # After the first deal the elapsed-cycle classification flips to
    # sign_mismatch and the gate holds every later request.
    first = recs.index(deals[0])
    post = recs[first + 1:]
    assert any(r.consistency == CONSISTENCY_SIGN_MISMATCH for r in post)
    suppressed = [r for r in post if r.request_suppressed_a]
    assert suppressed, "gate never suppressed a request"
    # The RATCHET is broken: after unwinding to zero the gate releases
    # (no_surplus carries no evidence), so the mechanism PROBES again —
    # a bounded deal/mismatch/unwind retry loop instead of the pre-v4
    # climb to the contract cap.  Assert the bound, not a hard stop.
    quantum = h.contracts[key].dq_quant_mvar
    assert max(abs(r.surplus_mvar) for r in post) <= 2 * quantum + 1e-9
    # ... and undelivered surplus is repeatedly wound back although the
    # need flag never clears.
    assert any(r.unwound_mvar != 0.0 for r in post)
    assert abs(h.sched.corridor_state(key).surplus_mvar) <= quantum
    assert post[-1].need_a  # violation still present, honestly flagged


# ---------------------------------------------------------------------------
#  (A) capability modes
# ---------------------------------------------------------------------------


def _collinear_area():
    """Two corridors whose control rows are collinear (the F2 geometry:
    electrically adjacent corridor terminals) — the D13 joint box
    demands moving the two flows in opposite directions and collapses."""
    u_now = np.zeros(3)
    u_min = np.full(3, -50.0)
    u_max = np.full(3, +50.0)
    v = np.array([1.00, 1.00])
    v_min = np.array([0.95, 0.95])
    v_max = np.array([1.05, 1.05])
    h_loc = np.array([[0.0005, 0.0005, 0.0],
                      [0.0005, 0.0005, 0.0002]])
    couplings = (
        CorridorCoupling(key=(1, 3), control_row=np.array([1.0, 0.0, 0.5]),
                         dq_quant_mvar=10.0),
        CorridorCoupling(key=(2, 3), control_row=np.array([2.0, 0.0, 1.0]),
                         dq_quant_mvar=10.0),
    )
    return u_now, u_min, u_max, v, v_min, v_max, h_loc, couplings


def test_joint_box_collapses_on_collinear_couplings():
    args = _collinear_area()
    res = area_capability(*args, MIQPSolver(), voltage_margin_pu=0.005,
                          mode="joint_box")
    assert res.mode == "joint_box"
    assert all(hi < 1.0 for (_lo, hi) in res.offers_mvar.values()), \
        f"expected F2 collapse, got {res.offers_mvar}"


def test_auto_mode_falls_back_to_per_corridor():
    args = _collinear_area()
    res = area_capability(*args, MIQPSolver(), voltage_margin_pu=0.005,
                          mode="auto", dq_min_deal_mvar=1.0)
    assert res.mode == "per_corridor"
    for key, (lo, hi) in res.offers_mvar.items():
        assert hi == pytest.approx(10.0), (key, lo, hi)
        assert lo == pytest.approx(-10.0)


def test_auto_mode_keeps_joint_box_when_feasible():
    from tests.sbx_h.test_need_capability import _synthetic_area

    u, ulo, uhi, v, vlo, vhi, h, coup = _synthetic_area(1.00)
    res = area_capability(u, ulo, uhi, v, vlo, vhi, h, coup,
                          MIQPSolver(), voltage_margin_pu=0.005,
                          mode="auto", dq_min_deal_mvar=1.0)
    assert res.mode == "joint_box"
    assert res.t >= 1.0


# ---------------------------------------------------------------------------
#  (i) per-bus tracking weights — config plumbing
# ---------------------------------------------------------------------------


def test_g_v_per_bus_validation():
    from controller.tso_controller import TSOControllerConfig

    kwargs = dict(
        der_indices=[], pcc_trafo_indices=[], pcc_dso_controller_ids=[],
        oltc_trafo_indices=[], shunt_bus_indices=[],
        shunt_q_steps_mvar=[], voltage_bus_indices=[1, 2, 3],
        current_line_indices=[],
    )
    cfg = TSOControllerConfig(
        g_v_per_bus=np.array([1.0, 20.0, 1.0]), **kwargs)
    assert cfg.g_v_per_bus[1] == 20.0
    with pytest.raises(ValueError, match="length"):
        TSOControllerConfig(g_v_per_bus=np.array([1.0, 2.0]), **kwargs)
    with pytest.raises(ValueError, match="positive"):
        TSOControllerConfig(
            g_v_per_bus=np.array([1.0, -1.0, 1.0]), **kwargs)
