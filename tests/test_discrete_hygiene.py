"""
BME Phase 5 — tests for :mod:`controller.discrete_hygiene` and the
TSO controller's post-solve gate (spec §3.8, §5 Phase 5).

Coverage against the spec's Phase 5 test list:
* slotting determinism: exactly one committer per tick, full rotation;
  the constructed two-area counter-switch scenario is exercised at the
  GATE level — without slotting both zones commit their discrete moves
  in the same tick, with slotting exactly the slot owner does (the
  closed-loop plant variant lands with the Phase 6 scenario runs);
* ε-acceptance rejects a constructed marginal-benefit switch and
  accepts a constructed clear-benefit switch (frozen-integer QP
  objective supplied as an oracle);
* ledger schema round-trips (append-only, one-time realised-ΔΦ fill);
* the estimator-masking/notice hook path is pure Phase 3 transport
  (delay/drop pinned there); v1's hook is a documented no-op.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from controller.discrete_hygiene import (
    LedgerEntry,
    SlottingSchedule,
    SwitchingLedger,
    epsilon_accepts,
)
from controller.tso_controller import TSOController


# ======================================================================
#  Slotting (§3.8.2, D5)
# ======================================================================

def test_slotting_round_robin():
    s = SlottingSchedule([3, 1, 2])  # rotation order is ascending ids
    owners = [s.slot_owner(k) for k in range(6)]
    assert owners == [1, 2, 3, 1, 2, 3]
    for k in range(6):
        committers = [z for z in (1, 2, 3) if s.may_commit(z, k)]
        assert len(committers) == 1
        assert committers[0] == owners[k]


def test_slotting_slot_length():
    s = SlottingSchedule([1, 2], slot_length=2)
    assert [s.slot_owner(k) for k in range(6)] == [1, 1, 2, 2, 1, 1]


def test_slotting_validation():
    with pytest.raises(ValueError, match="at least one zone"):
        SlottingSchedule([])
    with pytest.raises(ValueError, match="duplicate"):
        SlottingSchedule([1, 1])
    with pytest.raises(ValueError, match="slot_length"):
        SlottingSchedule([1, 2], slot_length=0)
    s = SlottingSchedule([1, 2])
    with pytest.raises(ValueError, match="unknown zone"):
        s.may_commit(9, 0)
    with pytest.raises(ValueError, match="tick"):
        s.slot_owner(-1)


# ======================================================================
#  ε-acceptance (§3.8.3, D6)
# ======================================================================

def test_epsilon_accepts_clear_benefit():
    ok, pred, cost = epsilon_accepts(
        obj_miqp=-10.0, obj_frozen=-2.0,
        delta_int_abs=np.array([1.0]), switch_costs=np.array([0.5]),
        epsilon_switch=1.0,
    )
    assert ok and pred == -8.0 and cost == 0.5


def test_epsilon_rejects_marginal_benefit():
    ok, pred, cost = epsilon_accepts(
        obj_miqp=-2.4, obj_frozen=-2.0,
        delta_int_abs=np.array([1.0]), switch_costs=np.array([0.5]),
        epsilon_switch=1.0,
    )
    assert not ok and pred == pytest.approx(-0.4)


def test_epsilon_zero_is_pure_sign_test():
    ok, _, _ = epsilon_accepts(
        obj_miqp=-2.0, obj_frozen=-2.0,
        delta_int_abs=np.array([2.0]), switch_costs=np.array([0.0]),
        epsilon_switch=0.0,
    )
    assert ok  # ties accepted at the boundary (≤)
    with pytest.raises(ValueError, match="non-negative"):
        epsilon_accepts(0, 0, np.array([-1.0]), np.array([0.0]), 0.0)


# ======================================================================
#  Ledger (§3.8.3)
# ======================================================================

def _entry(step=0, accepted=True, reason="accepted"):
    return LedgerEntry(
        step=step, zone=1, devices=("oltc:7",), delta_int=(1,),
        predicted_dphi=-3.0, accepted=accepted, reason=reason,
        slot_owner=1, epsilon_switch=0.5, switch_cost=0.1,
    )


def test_ledger_roundtrip_and_fill():
    led = SwitchingLedger()
    i0 = led.append(_entry(step=0))
    i1 = led.append(_entry(step=1, accepted=False, reason="epsilon_reject"))
    led.fill_realised(i0, -2.7)
    recs = led.to_records()
    led2 = SwitchingLedger.from_records(recs)
    assert led2.to_records() == recs
    assert led2.entries()[i0].realised_dphi == -2.7
    assert led2.entries()[i1].realised_dphi is None
    with pytest.raises(ValueError, match="append-only"):
        led.fill_realised(i0, -1.0)
    with pytest.raises(IndexError):
        led.fill_realised(99, 0.0)
    with pytest.raises(ValueError, match="reason"):
        _entry(reason="whatever")


# ======================================================================
#  Post-solve gate (controller-level, minimally constructed)
# ======================================================================

def _gate_controller(*, ledger, may_commit, epsilon=1.0,
                     cost_oltc=0.5, tick=4, slot_owner=2):
    """A TSOController shell carrying exactly the state the gate reads
    (the full construction needs a net; the gate itself is pure)."""
    c = TSOController.__new__(TSOController)
    c.controller_id = "TSO_test"
    c.bme_mode = True
    c._int_idx_arr = np.array([5, 6], dtype=np.int64)  # u-indices of ints
    c._oltc_int_indices = {5}                          # 5=oltc, 6=shunt
    c._bme_hygiene = {
        "zone": 1, "ledger": ledger, "epsilon": float(epsilon),
        "cost_oltc": float(cost_oltc), "cost_shunt": 0.0,
    }
    c._bme_slot_ctx = (tick, slot_owner, may_commit)
    return c


def _result(obj, w_integer):
    return SimpleNamespace(
        objective_value=float(obj),
        w_integer=np.asarray(w_integer, dtype=np.float64),
    )


def test_gate_accepts_clear_benefit():
    led = SwitchingLedger()
    c = _gate_controller(ledger=led, may_commit=True)
    miqp = _result(-10.0, [1, 0])
    frozen = _result(-2.0, [0, 0])
    out = c._post_solve_gate(miqp, solve_frozen=lambda: frozen)
    assert out is miqp
    e = led.entries()[0]
    assert e.accepted and e.reason == "accepted"
    assert e.devices == ("oltc:5",) and e.delta_int == (1,)
    assert e.predicted_dphi == -8.0 and e.switch_cost == 0.5
    assert c.bme_ledger_indices_this_step == (0,)


def test_gate_rejects_marginal_benefit():
    led = SwitchingLedger()
    c = _gate_controller(ledger=led, may_commit=True)
    miqp = _result(-2.4, [0, -1])
    frozen = _result(-2.0, [0, 0])
    out = c._post_solve_gate(miqp, solve_frozen=lambda: frozen)
    assert out is frozen
    e = led.entries()[0]
    assert not e.accepted and e.reason == "epsilon_reject"
    assert e.devices == ("shunt:6",)


def test_gate_slot_blocked_even_with_clear_benefit():
    led = SwitchingLedger()
    c = _gate_controller(ledger=led, may_commit=False)
    miqp = _result(-100.0, [1, 1])
    frozen = _result(-2.0, [0, 0])
    out = c._post_solve_gate(miqp, solve_frozen=lambda: frozen)
    assert out is frozen
    e = led.entries()[0]
    assert not e.accepted and e.reason == "slot_blocked"
    assert e.slot_owner == 2


def test_gate_no_move_no_ledger_no_frozen_solve():
    led = SwitchingLedger()
    c = _gate_controller(ledger=led, may_commit=False)
    miqp = _result(-1.0, [0, 0])

    def boom():
        raise AssertionError("frozen solve must not run without a move")

    out = c._post_solve_gate(miqp, solve_frozen=boom)
    assert out is miqp and len(led) == 0


def test_gate_missing_slot_context_raises():
    led = SwitchingLedger()
    c = _gate_controller(ledger=led, may_commit=True)
    c._bme_slot_ctx = None
    with pytest.raises(RuntimeError, match="slot context"):
        c._post_solve_gate(
            _result(-5.0, [1, 0]),
            solve_frozen=lambda: _result(-2.0, [0, 0]),
        )


def test_gate_inactive_without_hygiene():
    c = TSOController.__new__(TSOController)
    c.controller_id = "TSO_test"
    c.bme_mode = False
    miqp = _result(-5.0, [1])
    assert c._post_solve_gate(miqp, solve_frozen=None) is miqp


def test_two_zone_counter_switch_scenario():
    """Gate-level rendering of the spec's two-area scenario: both zones
    propose a discrete move on the same tick from the same (stale)
    signals. Without slotting both commit; with slotting exactly the
    slot owner does — deterministically."""
    def run(with_slotting):
        led = SwitchingLedger()
        committed = []
        for z in (1, 2):
            owner = SlottingSchedule([1, 2]).slot_owner(0)  # tick 0 → zone 1
            may = (z == owner) if with_slotting else True
            c = _gate_controller(
                ledger=led, may_commit=may, epsilon=0.0, cost_oltc=0.0,
                tick=0, slot_owner=owner if with_slotting else -1,
            )
            c._bme_hygiene["zone"] = z
            miqp = _result(-5.0, [1, 0])
            frozen = _result(-2.0, [0, 0])
            out = c._post_solve_gate(miqp, solve_frozen=lambda: frozen)
            if out is miqp:
                committed.append(z)
        return committed, led

    both, _ = run(with_slotting=False)
    assert both == [1, 2]        # counter-switching in the same tick
    only, led = run(with_slotting=True)
    assert only == [1]           # exactly the slot owner commits
    reasons = [e.reason for e in led.entries()]
    assert reasons == ["accepted", "slot_blocked"]
