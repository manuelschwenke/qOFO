"""
SBX-V Phase 3 acceptance tests (build plan §9 Phase 3):

* deterministic replay (identical scenario ⇒ identical request/grant
  log);
* ledger invariants (summation, feasibility cap, boundary activation);
* no grant exceeds the feasibility answer;
* PARTIAL and REJECT paths exercised (incl. all_or_nothing);
* the codified missing-message substitute (loud, potential := band);
* DP1 potentials wrapper (delta box → absolute netted box → direction
  split per DP3) and the gesichert flag;
* the A ∧ B need flag with persistence, gap reset and clearing
  hysteresis; request sizing with the day-ahead cap.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core.message import CapabilityMessage
from sbx_h.fail import SBXError
from sbx_v.band import NormalBand
from sbx_v.directions import Direction
from sbx_v.feasibility import check_feasibility
from sbx_v.grants_ledger import GrantsLedger
from sbx_v.messages import (
    VERDICT_ACCEPT,
    VERDICT_PARTIAL,
    VERDICT_REJECT,
    GrantConfirmation,
    ReserveRequest,
    Window,
)
from sbx_v.need_flag import VerticalNeedTracker, size_request_quanta
from sbx_v.pipeline import RequestPipeline
from sbx_v.potentials import build_potential_message, substitute_potential

AREA = "dso_area_1"
BAND = NormalBand(area_id=AREA, q_raise_mvar=50.0, q_lower_mvar=50.0)
DQ = 30.0


def win(i: int) -> Window:
    return Window(index=i, k_start=5 * i, k_end=5 * (i + 1),
                  t_start_s=900.0 * i, t_end_s=900.0 * (i + 1))


def capability(q_min: float, q_max: float, n_nvp: int = 3
               ) -> CapabilityMessage:
    """Netted delta box split evenly over the area's NVPs."""
    return CapabilityMessage(
        source_controller_id="dso_1", target_controller_id="tso_1",
        iteration=0,
        interface_transformer_indices=np.arange(n_nvp, dtype=np.int64),
        q_min_mvar=np.full(n_nvp, q_min / n_nvp),
        q_max_mvar=np.full(n_nvp, q_max / n_nvp),
    )


def potential(direction: Direction, q_pot: float, window: Window,
              ledger=None, *, is_forecast=True):
    """Convenience: a posted potential with a given magnitude."""
    from sbx_v.messages import PotentialMessage
    return PotentialMessage(
        aggregation_area_id=AREA, direction=direction,
        q_pot_mvar=q_pot, q_vh_flagged_mvar=0.0, window=window,
        is_forecast=is_forecast,
    )


# ---------------------------------------------------------------------------
#  Potentials (DP1 wrapper + substitute)
# ---------------------------------------------------------------------------


def test_potential_wrapper_direction_split():
    ledger = GrantsLedger(DQ)
    w = win(1)
    # Operating point +20 Mvar (LOWERING side), delta box [-150, +90]:
    # absolute box [-130, +110] → LOWERING pot 110, RAISING pot 130.
    cap = capability(-150.0, +90.0)
    p_low = build_potential_message(AREA, Direction.LOWERING, cap,
                                    20.0, w, BAND, ledger)
    p_rai = build_potential_message(AREA, Direction.RAISING, cap,
                                    20.0, w, BAND, ledger)
    assert p_low.q_pot_mvar == pytest.approx(110.0)
    assert p_rai.q_pot_mvar == pytest.approx(130.0)
    assert not p_low.is_substitute


def test_potential_gesichert_flag_marks_grants():
    ledger = GrantsLedger(DQ)
    w = win(2)
    req = ReserveRequest(aggregation_area_id=AREA,
                         direction=Direction.RAISING, n_quanta=2,
                         window=w)
    check_feasibility(req, potential(Direction.RAISING, 200.0, w),
                      50.0, ledger)
    ledger.confirm(GrantConfirmation(
        order_id="ord:x", aggregation_area_id=AREA,
        direction=Direction.RAISING, n_quanta=2, window=w))
    p = build_potential_message(AREA, Direction.RAISING,
                                capability(-200.0, 50.0), 0.0, w, BAND,
                                ledger)
    assert p.q_vh_flagged_mvar == pytest.approx(60.0)   # 2 × 30 Mvar


def test_missing_message_substitute_is_loud(caplog):
    with caplog.at_level(logging.WARNING, logger="sbx_v.potentials"):
        p = substitute_potential(AREA, Direction.RAISING, win(3), BAND)
    assert p.is_substitute and p.q_pot_mvar == pytest.approx(50.0)
    assert p.q_vh_flagged_mvar == 0.0
    assert any("codified substitute" in r.message for r in caplog.records)
    # The wrapper applies it for a missing capability message too.
    ledger = GrantsLedger(DQ)
    p2 = build_potential_message(AREA, Direction.RAISING, None, 0.0,
                                 win(3), BAND, ledger)
    assert p2.is_substitute


# ---------------------------------------------------------------------------
#  Feasibility verdicts + ledger invariants
# ---------------------------------------------------------------------------


def test_feasibility_accept_partial_reject_and_cap():
    ledger = GrantsLedger(DQ)
    w = win(4)
    # Potential 150 beyond zero, band 50 → headroom 100 → 3 quanta.
    pot = potential(Direction.RAISING, 150.0, w)
    r_ok = ReserveRequest(aggregation_area_id=AREA,
                          direction=Direction.RAISING, n_quanta=3,
                          window=w)
    assert check_feasibility(r_ok, pot, 50.0, ledger).verdict \
        == VERDICT_ACCEPT
    r_big = ReserveRequest(aggregation_area_id=AREA,
                           direction=Direction.RAISING, n_quanta=5,
                           window=w)
    reply = check_feasibility(r_big, pot, 50.0, ledger)
    assert reply.verdict == VERDICT_PARTIAL
    assert reply.n_quanta_offered == 3
    # all_or_nothing turns the same partial answer into REJECT.
    r_aon = ReserveRequest(aggregation_area_id=AREA,
                           direction=Direction.RAISING, n_quanta=5,
                           window=w, all_or_nothing=True)
    reply = check_feasibility(r_aon, pot, 50.0, ledger)
    assert reply.verdict == VERDICT_REJECT
    assert reply.n_quanta_offered == 0
    # Zero headroom (substitute-grade potential) → REJECT.
    r1 = ReserveRequest(aggregation_area_id=AREA,
                        direction=Direction.RAISING, n_quanta=1,
                        window=w)
    assert check_feasibility(
        r1, potential(Direction.RAISING, 50.0, w), 50.0, ledger,
    ).verdict == VERDICT_REJECT


def test_ledger_caps_confirmations_at_feasibility():
    ledger = GrantsLedger(DQ)
    w = win(5)
    req = ReserveRequest(aggregation_area_id=AREA,
                         direction=Direction.RAISING, n_quanta=2,
                         window=w)
    check_feasibility(req, potential(Direction.RAISING, 140.0, w),
                      50.0, ledger)                    # 3 possible, 2 offered
    ledger.confirm(GrantConfirmation(
        order_id="ord:a", aggregation_area_id=AREA,
        direction=Direction.RAISING, n_quanta=2, window=w))
    assert ledger.granted_mvar(AREA, Direction.RAISING, 5) \
        == pytest.approx(60.0)
    # A second confirmation beyond the offered quanta must fail.
    with pytest.raises(SBXError, match="exceed the last accepted"):
        ledger.confirm(GrantConfirmation(
            order_id="ord:b", aggregation_area_id=AREA,
            direction=Direction.RAISING, n_quanta=1, window=w))
    # Window-boundary activation: adjacent windows see nothing.
    assert ledger.granted_mvar(AREA, Direction.RAISING, 4) == 0.0
    assert ledger.granted_mvar(AREA, Direction.RAISING, 6) == 0.0
    ledger.assert_invariants(5)


def test_confirmation_without_feasibility_raises():
    ledger = GrantsLedger(DQ)
    with pytest.raises(SBXError, match="without a recorded feasibility"):
        ledger.confirm(GrantConfirmation(
            order_id="ord:c", aggregation_area_id=AREA,
            direction=Direction.RAISING, n_quanta=1, window=win(6)))


# ---------------------------------------------------------------------------
#  Need flag (A ∧ B) and request sizing
# ---------------------------------------------------------------------------


def test_need_flag_a_and_b_with_hysteresis():
    tr = VerticalNeedTracker(AREA, Direction.RAISING, n_persist=3,
                             n_clear=2, sat_tol_mvar=1.0,
                             v_dev_threshold_pu=0.005)
    edge = 50.0
    # A alone (saturated, no voltage deviation) never fires.
    for it in range(5):
        d = tr.update(it, -50.0, edge, 0.0)
    assert not d.flag
    # A ∧ B persists → fires after exactly n_persist joint iterations.
    for k, it in enumerate(range(5, 9)):
        d = tr.update(it, -50.0, edge, 0.008,
                      shortfall_estimate_mvar=45.0)
        if k < 2:
            assert not d.flag
    assert d.flag and d.shortfall_mvar == pytest.approx(45.0)
    # Hysteresis: one clear iteration is not enough ...
    d = tr.update(9, -20.0, edge, 0.0)
    assert d.flag
    # ... two consecutive clear iterations drop the flag.
    d = tr.update(10, -20.0, edge, 0.0)
    assert not d.flag
    # An iteration gap resets the persistence counters.
    tr.update(20, -50.0, edge, 0.008)
    d = tr.update(21, -50.0, edge, 0.008)
    assert not d.flag


def test_request_sizing_covers_and_caps():
    w = win(7)
    pot = potential(Direction.RAISING, 140.0, w)   # headroom 90 → 3 quanta
    assert size_request_quanta(45.0, DQ, pot, 50.0, 0.0) == 2
    assert size_request_quanta(200.0, DQ, pot, 50.0, 0.0) == 3   # capped
    assert size_request_quanta(200.0, DQ, pot, 50.0, 60.0) == 1  # grants
    assert size_request_quanta(10.0, DQ,
                               potential(Direction.RAISING, 50.0, w),
                               50.0, 0.0) == 0                   # no room


# ---------------------------------------------------------------------------
#  Pipeline: deterministic replay + paths
# ---------------------------------------------------------------------------


def _run_pipeline_scenario():
    ledger = GrantsLedger(DQ)
    # The Reserve-Observer margin is subtracted in the feasibility
    # answer but NOT in request sizing — that asymmetry is what
    # exercises the PARTIAL path end-to-end.
    pipe = RequestPipeline({AREA: BAND}, ledger,
                           reserve_margin_mvar=30.0)

    def forecast(area_id, direction, window):
        # Window 1: sizing headroom 140-50 = 90 (3 quanta), feasibility
        # headroom 90-30 = 60 (2 quanta); window 2: none beyond band.
        q = 140.0 if window.index == 1 else 50.0
        return potential(direction, q, window)

    from sbx_v.need_flag import VerticalNeedDecision
    need = {(AREA, Direction.RAISING): VerticalNeedDecision(
        iteration=4, flag=True, cond_a_count=3, cond_b_count=3,
        shortfall_mvar=70.0)}
    pipe.run_window(win(1), need, forecast)
    pipe.run_window(win(2), need, forecast)       # no headroom → logged
    # Re-invocation of an already-requested window is a no-op.
    pipe.run_window(win(1), need, forecast)
    return pipe, ledger


def test_pipeline_deterministic_replay_and_paths():
    pipe1, ledger1 = _run_pipeline_scenario()
    pipe2, ledger2 = _run_pipeline_scenario()
    # Byte-identical logs (Phase-3 acceptance).
    assert pipe1.log == pipe2.log
    assert repr(ledger1.entries()) == repr(ledger2.entries())
    # Window 1: sized to ceil(70/30) = 3, PARTIAL-capped at 2, granted.
    assert ("request", 1, AREA, "raising", 3, False) in pipe1.log
    assert ("reply", 1, AREA, "raising", VERDICT_PARTIAL, 2) in pipe1.log
    assert ("grant", 1, AREA, "raising", 2) in pipe1.log
    assert ledger1.granted_mvar(AREA, Direction.RAISING, 1) \
        == pytest.approx(60.0)
    # Window 2: the day-ahead cap leaves no room → logged, no request.
    assert ("no_headroom", 2, AREA, "raising", 70.0, 50.0, False) \
        in pipe1.log
    # No grant beyond the feasibility answer anywhere.
    ledger1.assert_invariants(1)
    ledger1.assert_invariants(2)
    # Settlement export carries the v1 delivering party.
    recs = ledger1.to_grant_records()
    assert len(recs) == 1 and recs[0].q_grant_mvar == pytest.approx(60.0)
