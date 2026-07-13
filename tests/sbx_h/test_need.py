"""
SBX-H v6 — tests for the ``sbx_h.need`` violation indicator
(set threshold + persistence, hysteresis/preventive release, direction
handling, gap reset).  The capability-LP tests of the former deal layer
live in ``_archive/sbx_h_v5/``.
"""
from __future__ import annotations

import pytest

from sbx_h.config import SBXConfig
from sbx_h.need import NeedTracker

BUS = [10, 11]
LO = [1.0, 1.0]
HI = [2.0, 2.0]


def cfg(**over) -> SBXConfig:
    return SBXConfig(**{"n_need": 2, "v_viol_threshold_pu": 0.005,
                        **over})


def _v(depth10=0.0, depth11=0.0):
    """Voltages with the requested UNDER-depths at the two buses."""
    return [1.0 - depth10, 1.0 - depth11]


def test_no_flag_inside_bounds():
    tr = NeedTracker(cfg(), 1)
    for it in range(3):
        d = tr.update(it, BUS, _v(), LO, HI)
        assert not d.flag and d.direction == 0 and d.worst_bus is None


def test_flag_fires_after_exactly_n_need():
    tr = NeedTracker(cfg(n_need=3), 1)
    for it in range(2):
        assert not tr.update(it, BUS, _v(depth10=0.01), LO, HI).flag
    d = tr.update(2, BUS, _v(depth10=0.01), LO, HI)
    assert d.flag and d.direction == +1 and d.worst_bus == 10
    assert d.depth_under_pu == pytest.approx(0.01)


def test_overvoltage_direction():
    tr = NeedTracker(cfg(n_need=1), 1)
    d = tr.update(0, BUS, [2.02, 1.5], LO, HI)
    assert d.flag and d.direction == -1 and d.worst_bus == 10
    assert d.depth_over_pu == pytest.approx(0.02)


def test_direction_change_restarts_persistence():
    tr = NeedTracker(cfg(n_need=2), 1)
    tr.update(0, BUS, _v(depth10=0.01), LO, HI)
    d = tr.update(1, BUS, [2.02, 1.5], LO, HI)   # flip to overvoltage
    assert not d.flag and d.consecutive == 1


def test_gap_resets_persistence():
    tr = NeedTracker(cfg(n_need=2), 1)
    tr.update(0, BUS, _v(depth10=0.01), LO, HI)
    d = tr.update(5, BUS, _v(depth10=0.01), LO, HI)  # iteration gap
    assert not d.flag and d.consecutive == 1


def test_hysteresis_preventive_release():
    tr = NeedTracker(cfg(n_need=1, release_threshold_pu=0.001), 1)
    assert tr.update(0, BUS, _v(depth10=0.010), LO, HI).flag
    # 3 mpu: below the set threshold, above release → stays latched.
    d = tr.update(1, BUS, _v(depth10=0.003), LO, HI)
    assert d.flag and d.direction == +1 and d.worst_bus == 10
    # 0.5 mpu: below release → clears.
    d = tr.update(2, BUS, _v(depth10=0.0005), LO, HI)
    assert not d.flag and d.direction == 0
    # Re-latching needs the SET threshold again.
    assert not tr.update(3, BUS, _v(depth10=0.003), LO, HI).flag


def test_default_release_equals_set_threshold():
    tr = NeedTracker(cfg(n_need=1), 1)
    assert tr.update(0, BUS, _v(depth10=0.010), LO, HI).flag
    # Without hysteresis, dipping below the (single) threshold clears.
    assert not tr.update(1, BUS, _v(depth10=0.003), LO, HI).flag


def test_input_validation():
    from sbx_h.fail import SBXError

    tr = NeedTracker(cfg(), 1)
    with pytest.raises(SBXError, match="align"):
        tr.update(0, BUS, [1.0], LO, HI)
    with pytest.raises(SBXError, match="at least one"):
        tr.update(0, [], [], [], [])
