"""
Regression tests for the local-relay OLTC rate limiter of the multi-TSO runner
=============================================================================

:class:`experiments.runners._multi_tso_helpers._OLTCRateLimiter` is called
after every plant power flow of a step (post-profile, post-contingency,
end-of-step), always against the snapshot taken at the start of the step.

Fault fixed 2026-11-06: a relay move accepted by the first clamp of a step was
reverted by a later clamp of the SAME step, because the cooldown gate saw
``time_s - last_change_time_s == 0 < cooldown``; the lock then restarted.
Under SVR the end-of-step power flow runs every step, so the coupler relays
were effectively frozen (thesis ch:case3:nominal, run 0025, scenario E2).

The tests drive the limiter the way the runner does, with a pandapower net that
carries one ``DiscreteTapControl``; the relay's own action inside a power flow
is emulated by writing ``tap_pos`` directly.

Author: Manuel Schwenke / Claude Code (2026-11-06)
"""

from __future__ import annotations

import pandapower.networks as pn
import pytest
from pandapower.control import DiscreteTapControl

from experiments.runners._multi_tso_helpers import _OLTCRateLimiter

DT = 20.0          # runner step [s]
COOLDOWN = 120.0   # coupler lock of the Case-2 runs (oltc_cooldown_s_nc)


@pytest.fixture
def net_and_tap():
    net = pn.example_simple()
    tid = int(net.trafo.index[0])
    net.trafo.at[tid, "tap_pos"] = 0
    DiscreteTapControl(net, element_index=tid, vm_lower_pu=0.99,
                       vm_upper_pu=1.01, side="lv")
    return net, tid


def _limiter() -> _OLTCRateLimiter:
    return _OLTCRateLimiter(max_step=1, cooldown_s=30.0,
                            cooldown_by_table={"trafo": COOLDOWN})


def _step(lim, net, tid, t, relay_positions):
    """One runner step: snapshot, then one clamp per power flow; the relay
    leaves the tap at ``relay_positions[i]`` in power flow ``i``."""
    lim.snapshot(net)
    for pos in relay_positions:
        net.trafo.at[tid, "tap_pos"] = pos
        lim.clamp(net, t)
    return int(net.trafo.at[tid, "tap_pos"])


def test_move_survives_end_of_step_clamp(net_and_tap):
    """The fault: post-profile move +1, end-of-step PF keeps +1 -> stays +1."""
    net, tid = net_and_tap
    lim = _limiter()
    assert _step(lim, net, tid, 1000.0, [1, 1]) == 1


def test_relay_keeps_moving_every_cooldown_under_svr_pattern(net_and_tap):
    """With an end-of-step PF in every step (SVR), a relay that wants to go
    up moves once per cooldown, not never."""
    net, tid = net_and_tap
    lim = _limiter()
    positions = []
    for k in range(13):                      # 0 ... 240 s
        t = 1000.0 + k * DT
        p0 = int(net.trafo.at[tid, "tap_pos"])
        positions.append(_step(lim, net, tid, t, [p0 + 1, p0 + 1]))
    # moves at t = 1000, 1120, 1240 s
    assert positions[0] == 1
    assert positions[5] == 1                 # 1100 s: still locked
    assert positions[6] == 2                 # 1120 s: lock elapsed
    assert positions[12] == 3                # 1240 s


def test_cooldown_still_blocks_a_move_in_a_later_step(net_and_tap):
    net, tid = net_and_tap
    lim = _limiter()
    assert _step(lim, net, tid, 1000.0, [1]) == 1
    assert _step(lim, net, tid, 1020.0, [2]) == 1          # within 120 s
    assert _step(lim, net, tid, 1100.0, [2, 2]) == 1       # still within
    assert _step(lim, net, tid, 1120.0, [2]) == 2          # elapsed


def test_max_step_against_snapshot_holds_across_clamps(net_and_tap):
    """Two power flows of one step cannot add up to two tap steps."""
    net, tid = net_and_tap
    lim = _limiter()
    assert _step(lim, net, tid, 1000.0, [1, 2]) == 1
    assert _step(lim, net, tid, 1200.0, [3, 3]) == 2       # 1 -> 3 clamped to 2


def test_move_undone_in_same_step_does_not_start_the_lock(net_and_tap):
    """Post-profile +1, end-of-step PF back to 0: no move, so the relay may
    move in the next step."""
    net, tid = net_and_tap
    lim = _limiter()
    assert _step(lim, net, tid, 1000.0, [1, 0]) == 0
    assert _step(lim, net, tid, 1020.0, [1]) == 1


def test_undone_move_restores_previous_lock(net_and_tap):
    """A move undone within its step must not erase an earlier, still
    running lock."""
    net, tid = net_and_tap
    lim = _limiter()
    assert _step(lim, net, tid, 1000.0, [1]) == 1          # lock until 1120
    assert _step(lim, net, tid, 1140.0, [2, 1]) == 1       # accepted, undone
    # the lock of 1000 s is restored (elapsed), so the next move is allowed
    assert _step(lim, net, tid, 1160.0, [2]) == 2
    # and a new lock runs from 1160 s
    assert _step(lim, net, tid, 1180.0, [3]) == 2
