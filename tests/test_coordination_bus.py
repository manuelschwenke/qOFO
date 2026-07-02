"""
BME Phase 3 — tests for :mod:`core.coordination_bus`.

Spec references (docs/BME_STATUS.md; BME spec §3.4, §3.8, §3.9, §5 Phase 3):
* delay semantics — a message published at k is visible at k + d, not
  earlier (and not later: signals are per-step, no stale carry-over at
  the bus level);
* cold start — the receiver logs and runs uncoordinated for exactly d
  steps;
* a missing expected signal after warm-up RAISES when drops are
  disabled;
* the hold-last-filtered-value policy engages and is logged when drop
  simulation is enabled (extended cold start if the very first signal
  is dropped);
* determinism under a fixed seed.

The bus is pure in-process machinery — no pandapower net is needed here.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.coordination_bus import (
    CoordinationBus,
    MarginalReceiver,
    MarginalSignal,
    SwitchNotice,
)

ZONES = [1, 2, 3]
NB = 4  # boundary registry length used throughout these tests


def _mu(sender: int, step: int) -> np.ndarray:
    """Deterministic, sender/step-distinct μ vector."""
    return np.arange(1.0, NB + 1.0) * sender + 0.01 * step


def _sig(sender: int, step: int) -> MarginalSignal:
    return MarginalSignal(
        zone_id=sender, step=step,
        mu=_mu(sender, step), v_b_meas=np.full(NB, 1.0),
    )


def _publish_all(bus, steps, senders=ZONES):
    for k in steps:
        for z in senders:
            bus.publish_marginal(_sig(z, k))


# ======================================================================
#  Dataclass and constructor validation (fail-fast, §8)
# ======================================================================

def test_signal_validation():
    with pytest.raises(ValueError, match="1-D"):
        MarginalSignal(1, 0, np.zeros((2, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        MarginalSignal(1, 0, np.array([1.0, np.nan]), np.zeros(2))
    with pytest.raises(ValueError, match="registry length"):
        MarginalSignal(1, 0, np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="at least one moved device"):
        SwitchNotice(1, 0, np.zeros(4), ())
    # signals are frozen against mutation after construction
    sig = _sig(1, 0)
    with pytest.raises(ValueError):
        sig.mu[0] = 99.0


def test_bus_constructor_validation():
    with pytest.raises(ValueError, match="at least two zones"):
        CoordinationBus([1], NB)
    with pytest.raises(ValueError, match="duplicate"):
        CoordinationBus([1, 1, 2], NB)
    with pytest.raises(ValueError, match="delay_steps"):
        CoordinationBus(ZONES, NB, delay_steps=-1)
    with pytest.raises(ValueError, match="drop_probability"):
        CoordinationBus(ZONES, NB, drop_probability=1.5)
    with pytest.raises(ValueError, match="requires an explicit seed"):
        CoordinationBus(ZONES, NB, drop_probability=0.5)


def test_publish_validation():
    bus = CoordinationBus(ZONES, NB, delay_steps=1)
    with pytest.raises(ValueError, match="not registered"):
        bus.publish_marginal(_sig(99, 0))
    with pytest.raises(ValueError, match="boundary registry length"):
        bus.publish_marginal(MarginalSignal(1, 0, np.zeros(3), np.zeros(3)))
    bus.publish_marginal(_sig(1, 0))
    with pytest.raises(ValueError, match="already published"):
        bus.publish_marginal(_sig(1, 0))


def test_receiver_constructor_validation():
    bus = CoordinationBus(ZONES, NB)
    with pytest.raises(ValueError, match="not registered"):
        MarginalReceiver(99, bus)
    with pytest.raises(ValueError, match="beta"):
        MarginalReceiver(1, bus, beta=0.0)
    with pytest.raises(ValueError, match="unknown zones"):
        MarginalReceiver(1, bus, expected_senders=[2, 7])
    with pytest.raises(ValueError, match="own marginal"):
        MarginalReceiver(1, bus, expected_senders=[1, 2])


# ======================================================================
#  Delay semantics (§5 Phase 3: visible at k + d, not earlier)
# ======================================================================

def test_delay_semantics():
    d = 2
    bus = CoordinationBus(ZONES, NB, delay_steps=d)
    _publish_all(bus, steps=[0])
    # Not visible before k + d
    assert bus.marginals_visible(1, 0) == {}
    assert bus.marginals_visible(1, 1) == {}
    # Visible at exactly k + d, senders = the OTHER zones
    vis = bus.marginals_visible(1, 2)
    assert sorted(vis) == [2, 3]
    np.testing.assert_array_equal(vis[2].mu, _mu(2, 0))
    # Per-step signals: nothing was published at k = 1, so nothing is
    # visible at k + d + 1 (no stale carry-over at the bus level).
    assert bus.marginals_visible(1, 3) == {}
    # Self signals are never delivered back to the sender
    assert 1 not in bus.marginals_visible(1, 2)


def test_delay_zero_same_step():
    """d = 0 (same-step exchange) is required by the Phase 4
    distributed-equals-centralised identity test."""
    bus = CoordinationBus(ZONES, NB, delay_steps=0)
    _publish_all(bus, steps=[5])
    vis = bus.marginals_visible(3, 5)
    assert sorted(vis) == [1, 2]


def test_notice_delay_and_multiplicity():
    d = 1
    bus = CoordinationBus(ZONES, NB, delay_steps=d)
    bus.publish_notice(SwitchNotice(1, 0, np.zeros(NB), ("oltc_3",)))
    bus.publish_notice(SwitchNotice(1, 0, np.zeros(NB), ("shunt_5",)))
    assert bus.notices_visible(2, 0) == []
    got = bus.notices_visible(2, 1)
    assert len(got) == 2
    assert {n.devices[0] for n in got} == {"oltc_3", "shunt_5"}
    assert bus.notices_visible(1, 1) == []  # never back to the sender


# ======================================================================
#  Cold start and filter recursion (§3.8, §3.4)
# ======================================================================

def test_cold_start_exactly_d_steps_then_filtered_sum():
    d, beta = 2, 0.3
    bus = CoordinationBus(ZONES, NB, delay_steps=d)
    recv = MarginalReceiver(1, bus, beta=beta, start_step=0)
    _publish_all(bus, steps=range(5))

    # Exactly d cold-start steps, logged
    for k in range(d):
        out = recv.update(k)
        assert out.coordinated is False
        assert out.mu_neighbour_sum is None
    assert [e.kind for e in recv.events] == ["cold_start"] * d

    # Warm-up step: first samples initialise the filters (β = 1 once)
    out = recv.update(d)
    assert out.coordinated is True
    np.testing.assert_allclose(
        out.mu_neighbour_sum, _mu(2, 0) + _mu(3, 0), rtol=0, atol=1e-15,
    )

    # One recursion step: μ^filt = (1 − β)·μ(0) + β·μ(1) per sender
    out = recv.update(d + 1)
    expected = sum(
        (1 - beta) * _mu(z, 0) + beta * _mu(z, 1) for z in (2, 3)
    )
    np.testing.assert_allclose(
        out.mu_neighbour_sum, expected, rtol=1e-14,
    )


def test_beta_one_disables_smoothing():
    """β = 1 (identity-test configuration): the filtered value IS the
    latest delivered sample."""
    d = 1
    bus = CoordinationBus(ZONES, NB, delay_steps=d)
    recv = MarginalReceiver(2, bus, beta=1.0, start_step=0)
    _publish_all(bus, steps=range(4))
    recv.update(0)
    for k in range(1, 4):
        out = recv.update(k)
        np.testing.assert_array_equal(
            out.mu_neighbour_sum, _mu(1, k - d) + _mu(3, k - d),
        )


def test_receiver_must_step_consecutively():
    bus = CoordinationBus(ZONES, NB, delay_steps=1)
    recv = MarginalReceiver(1, bus, start_step=0)
    with pytest.raises(ValueError, match="start_step"):
        recv.update(3)
    _publish_all(bus, steps=range(3))
    recv.update(0)
    with pytest.raises(ValueError, match="consecutively"):
        recv.update(2)


# ======================================================================
#  Missing signals: raise without drops, hold-last with drops (§3.8)
# ======================================================================

def test_missing_signal_after_warmup_raises():
    d = 2
    bus = CoordinationBus(ZONES, NB, delay_steps=d)
    recv = MarginalReceiver(1, bus, start_step=0)
    # Zone 3 never publishes step 1
    for k in range(4):
        for z in ZONES:
            if z == 3 and k == 1:
                continue
            bus.publish_marginal(_sig(z, k))
    recv.update(0)
    recv.update(1)
    recv.update(2)  # sees step-0 signals: fine
    with pytest.raises(RuntimeError, match="zone 3.*missing|missing"):
        recv.update(3)  # step-1 signal from zone 3 is absent


def test_hold_last_and_extended_cold_under_drops():
    """With drop simulation enabled: a dropped signal freezes the
    receiver's filter state for that sender (hold-last-FILTERED-value),
    and a sender whose first signal was dropped contributes exactly
    zero until one arrives — both logged per occurrence."""
    d, beta, steps = 1, 0.3, 12
    bus = CoordinationBus(
        ZONES, NB, delay_steps=d, drop_probability=0.5, seed=7,
    )
    recv = MarginalReceiver(1, bus, beta=beta, start_step=0)
    _publish_all(bus, steps=range(steps))

    saw_hold = saw_extended = False
    for k in range(steps):
        before = {z: recv.mu_filtered(z) for z in (2, 3)}
        out = recv.update(k)
        if k < d:
            assert out.coordinated is False
            continue
        assert out.coordinated is True
        held = {
            e.sender for e in recv.events
            if e.kind == "hold_last" and e.step == k
        }
        extended = {
            e.sender for e in recv.events
            if e.kind == "extended_cold" and e.step == k
        }
        saw_hold |= bool(held)
        saw_extended |= bool(extended)
        expected_sum = np.zeros(NB)
        for z in (2, 3):
            after = recv.mu_filtered(z)
            if z in held:
                # hold-last-filtered: state frozen
                np.testing.assert_array_equal(after, before[z])
            elif z in extended:
                assert after is None  # still no state — contributes zero
            else:
                # delivered: β-recursion (or first-sample initialisation)
                if before[z] is None:
                    np.testing.assert_array_equal(after, _mu(z, k - d))
                else:
                    np.testing.assert_allclose(
                        after,
                        (1 - beta) * before[z] + beta * _mu(z, k - d),
                        rtol=1e-14,
                    )
            if after is not None:
                expected_sum += after
        np.testing.assert_allclose(
            out.mu_neighbour_sum, expected_sum, rtol=1e-14,
        )
    # seed=7, p=0.5 over 22 deliveries: the hold policy must have fired
    assert saw_hold, "no hold_last event occurred — adjust seed"
    # extended_cold fires iff some sender's FIRST publish (step 0) was
    # dropped on its way to receiver 1 — consistency with the drop log
    first_drops = {
        e.sender for e in bus.drop_log
        if e.step == 0 and e.receiver == 1 and e.channel == "marginal"
    }
    assert saw_extended == bool(first_drops)


def test_all_dropped_gives_zero_sum():
    """p = 1: every delivery is lost — after warm-up the receiver is
    'coordinated' but every sender stays in extended cold start and the
    neighbour sum is exactly zero (a neutral price)."""
    d = 1
    bus = CoordinationBus(
        ZONES, NB, delay_steps=d, drop_probability=1.0, seed=1,
    )
    recv = MarginalReceiver(1, bus, start_step=0)
    _publish_all(bus, steps=range(3))
    recv.update(0)
    for k in (1, 2):
        out = recv.update(k)
        assert out.coordinated is True
        np.testing.assert_array_equal(out.mu_neighbour_sum, np.zeros(NB))
    kinds = [e.kind for e in recv.events]
    assert kinds.count("extended_cold") == 4  # 2 senders × 2 warm steps
    assert len(bus.drop_log) > 0


def test_determinism_under_fixed_seed():
    """Same seed → identical drop pattern and identical filtered sums;
    a different seed produces a different pattern."""
    def run(seed):
        bus = CoordinationBus(
            ZONES, NB, delay_steps=1, drop_probability=0.5, seed=seed,
        )
        recv = MarginalReceiver(1, bus, start_step=0)
        _publish_all(bus, steps=range(20))
        sums = []
        for k in range(20):
            out = recv.update(k)
            if out.coordinated:
                sums.append(out.mu_neighbour_sum)
        drops = [
            (e.sender, e.receiver, e.step, e.channel)
            for e in bus.drop_log
        ]
        return drops, np.vstack(sums)

    drops_a, sums_a = run(42)
    drops_b, sums_b = run(42)
    assert drops_a == drops_b
    np.testing.assert_array_equal(sums_a, sums_b)

    drops_c, _ = run(43)
    assert drops_a != drops_c


def test_dropped_notice_is_lost_and_logged():
    bus = CoordinationBus(
        ZONES, NB, delay_steps=1, drop_probability=1.0, seed=3,
    )
    bus.publish_notice(SwitchNotice(2, 0, np.zeros(NB), ("oltc_1",)))
    assert bus.notices_visible(1, 1) == []
    assert any(
        e.channel == "notice" and e.sender == 2 for e in bus.drop_log
    )
