"""Per-bank ledger of the switched-shunt interface feedforward (2026-11-02)."""
import pytest

from experiments.runners._multi_tso_helpers import _ledger_feedforward


def test_release_removes_what_engagement_added():
    ledger = {}
    itf = [0, 1, 2]
    on1 = _ledger_feedforward(ledger, 7, 0, +1, itf, [-2.6, -19.9, -3.5])
    on2 = _ledger_feedforward(ledger, 7, 1, +1, itf, [-2.8, -21.0, -3.7])
    # The pre-switch model overestimates a release; the ledger ignores it.
    off2 = _ledger_feedforward(ledger, 7, 2, -1, itf, [3.1, 23.6, 4.1])
    off1 = _ledger_feedforward(ledger, 7, 1, -1, itf, [3.0, 22.0, 4.0])
    assert on1 == [-2.6, -19.9, -3.5]
    assert off2 == [2.8, 21.0, 3.7]
    for k in range(3):
        assert on1[k] + on2[k] + off2[k] + off1[k] == pytest.approx(0.0, abs=1e-12)
    assert ledger[7] == []


def test_banks_are_independent():
    ledger = {}
    _ledger_feedforward(ledger, 1, 0, +1, [0, 1], [-1.0, -2.0])
    _ledger_feedforward(ledger, 2, 0, +1, [3, 4], [-5.0, -6.0])
    assert _ledger_feedforward(ledger, 1, 1, -1, [0, 1], [9.0, 9.0]) == [1.0, 2.0]
    assert len(ledger[2]) == 1


def test_fails_fast_on_inconsistency():
    ledger = {}
    with pytest.raises(ValueError, match="level"):
        _ledger_feedforward(ledger, 1, 1, -1, [0], [1.0])
    _ledger_feedforward(ledger, 1, 0, +1, [0, 1], [-1.0, -1.0])
    with pytest.raises(ValueError, match="Couplers"):
        _ledger_feedforward(ledger, 1, 1, -1, [0], [1.0])
    with pytest.raises(ValueError, match="direction"):
        _ledger_feedforward({1: [{0: -1.0}]}, 1, 1, 0, [0], [0.0])
