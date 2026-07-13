"""
SBX-H v5 "evidence-based SBX" — unit/protocol tests (2026-07-10).

Covers the four v5 moves implemented after the 015 helpfulness campaign
(findings G1–G6 in STATUS_SBX.md):

* Move 1 — C1 arming: a need flag emits a request only when the area
  cannot help itself (optimistic self-help lift < depth).
* Move 2 — voltage-referenced delivery verification: the acting side's
  terminals must track their shifted references; undelivered cycles
  gate further requests AND suspend the tier-2 billing.
* Move 3 — preventive release (need-flag hysteresis) and gap-sized
  multi-quantum requests with matching/offer support.

The v4 reference behaviour remains pinned in test_scheduler.REF_CFG;
this module pins the same TIMING and exercises the v5 semantics.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sbx_h.config import SBXConfig
from sbx_h.fail import SBXError
from sbx_h.matching import KIND_UNILATERAL, match
from sbx_h.need import NeedTracker

from tests.sbx_h.test_messages_matching import CONTRACT, QUANT, _msg
from tests.sbx_h.test_scheduler import (  # noqa: F401  (plant via conftest)
    BOUND_OK, BOUND_UNDER, N_U, Harness, plant,
)

#: v5 protocol-test timing (same as REF_CFG) with the v5 SEMANTIC
#: defaults left active.
V5_CFG = dict(k_sched=2, dq_quant_rate_mvar_per_15min=30.0,
              n_need=1, m_release=1)


def v5_config(**overrides) -> SBXConfig:
    return SBXConfig(**{**V5_CFG, **overrides})


# ───────────────────────────────────────────────────────────────────────
#  Config validation
# ───────────────────────────────────────────────────────────────────────


def test_v5_config_validation():
    with pytest.raises(SBXError, match="delivery_check"):
        v5_config(delivery_check="bogus")
    with pytest.raises(SBXError, match="release_threshold_pu"):
        v5_config(v_viol_threshold_pu=0.005, release_threshold_pu=0.01)
    with pytest.raises(SBXError, match="request_sizing"):
        v5_config(request_sizing="foo")
    with pytest.raises(SBXError, match="k_max_quanta_per_request"):
        v5_config(k_max_quanta_per_request=0)
    with pytest.raises(SBXError, match="c1_arming_factor"):
        v5_config(c1_arming_factor=0.0)
    with pytest.raises(SBXError, match="v_delivery_tol_pu"):
        v5_config(v_delivery_tol_pu=0.0)


# ───────────────────────────────────────────────────────────────────────
#  Move 3a — preventive release (need-flag hysteresis)
# ───────────────────────────────────────────────────────────────────────


def test_need_hysteresis_preventive_release():
    cfg = v5_config(v_viol_threshold_pu=0.005, release_threshold_pu=0.001,
                    n_need=1)
    tr = NeedTracker(cfg, area_id=1)
    bus, lo, hi = [10], [1.0], [2.0]
    d = tr.update(0, bus, [0.990], lo, hi)      # depth 10 mpu > set
    assert d.flag and d.direction == +1
    d = tr.update(1, bus, [0.997], lo, hi)      # 3 mpu: below set, above
    assert d.flag and d.direction == +1         # release → STAYS flagged
    assert d.worst_bus == 10
    d = tr.update(2, bus, [0.9995], lo, hi)     # 0.5 mpu ≤ release
    assert not d.flag and d.direction == 0      # → cleared
    # Re-latching needs the SET threshold again.
    d = tr.update(3, bus, [0.997], lo, hi)      # 3 mpu < set
    assert not d.flag


def test_need_release_default_reproduces_v4():
    tr = NeedTracker(v5_config(v_viol_threshold_pu=0.005), area_id=1)
    bus, lo, hi = [10], [1.0], [2.0]
    assert tr.update(0, bus, [0.990], lo, hi).flag
    # v4: dipping below the (single) threshold clears immediately.
    assert not tr.update(1, bus, [0.997], lo, hi).flag


# ───────────────────────────────────────────────────────────────────────
#  Move 3b — sized requests: matching accepts integer multiples
# ───────────────────────────────────────────────────────────────────────


def test_matching_accepts_integer_multiples_of_quantum():
    deal = match(_msg(1, request=-2 * QUANT),
                 _msg(2, offer=(-3 * QUANT, 3 * QUANT)),
                 CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_UNILATERAL
    assert deal.dq_deal_mvar == pytest.approx(-2 * QUANT)


def test_matching_rejects_non_multiples():
    with pytest.raises(SBXError, match="multiple"):
        match(_msg(1, request=-1.5 * QUANT), _msg(2), CONTRACT, 0.0, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  Move 1 — C1 arming
# ───────────────────────────────────────────────────────────────────────


def test_c1_arming_blocks_selfsufficient_area(plant):
    """Ample H-weighted own headroom at the violated buses → the need
    flag stays, but NO request is emitted (c1_unarmed recorded)."""
    cfg = v5_config(request_sizing="single", delivery_check="magnitude")
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER
    orig = h.cycle_data

    def patched():
        data = orig()
        n_v = len(data[1].v_bus_indices)
        # Self-help lift = Σ |1e-2| · 500 per actuator ≫ any depth.
        data[1] = dataclasses.replace(
            data[1], h_loc=np.full((n_v, N_U), 1.0e-2))
        return data

    h.cycle_data = patched
    h.run_cycles(2)
    for key in ((1, 2), (1, 3)):
        recs = h.sched.records[key]
        assert all(r.deal.dq_deal_mvar == 0.0 for r in recs)
        assert any(r.c1_unarmed_a or r.c1_unarmed_b for r in recs)


def test_c1_stall_arming_overrides_optimistic_bound(plant):
    """The model bound says 'plenty of self-help' (big h_loc), but the
    violation depth does not improve — after c1_stall_cycles flagged
    boundaries the measured-stall clause arms the request anyway.

    This is the arming path that matters on REAL controllers, where
    setpoint headroom (AVR voltage boxes) overstates the physically
    deliverable Q of saturated machines (first D2S1 v5 run)."""
    cfg = v5_config(request_sizing="single", delivery_check="magnitude",
                    c1_stall_cycles=2)
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER
    orig = h.cycle_data

    def patched():
        data = orig()
        n_v = len(data[1].v_bus_indices)
        data[1] = dataclasses.replace(
            data[1], h_loc=np.full((n_v, N_U), 1.0e-2))
        return data

    h.cycle_data = patched
    h.run_cycles(4)                 # boundaries at cycles 1, 2, 3
    key = (1, 2)
    recs = {r.cycle: r for r in h.sched.records[key]}
    # Cycles 1–2: flag set, bound blocks (unarmed, no deal).
    assert recs[1].c1_unarmed_a and recs[1].deal.dq_deal_mvar == 0.0
    assert recs[2].c1_unarmed_a and recs[2].deal.dq_deal_mvar == 0.0
    # Cycle 3: stalled (depth unchanged for > 2 boundaries) → armed.
    assert recs[3].deal.dq_deal_mvar != 0.0
    assert not recs[3].c1_unarmed_a


def test_c1_arming_fires_for_exhausted_area(plant):
    """Zero self-help (the harness's zero h_loc) → armed → the v4-style
    unilateral deal executes."""
    cfg = v5_config(request_sizing="single", delivery_check="magnitude")
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(2)
    executed = [r for r in h.sched.records[(1, 2)]
                if r.deal.dq_deal_mvar != 0.0]
    assert executed and executed[0].deal.requester == 1
    assert abs(executed[0].deal.dq_deal_mvar) == pytest.approx(
        h.contracts[(1, 2)].dq_quant_mvar)


# ───────────────────────────────────────────────────────────────────────
#  Move 3c — gap-sized requests reach k_max quanta (offers scale too)
# ───────────────────────────────────────────────────────────────────────


def test_gap_sized_request_executes_multi_quantum_deal(plant):
    cfg = v5_config(delivery_check="magnitude")     # sizing "gap" default
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(2)
    key = (1, 2)
    executed = [r for r in h.sched.records[key]
                if r.deal.dq_deal_mvar != 0.0]
    assert executed, "no deal executed under gap sizing"
    # Harness geometry: depth ≈ 10–30 mpu, dv/dq = 2e-4 pu/Mvar,
    # quantum 12 Mvar → gap/lift_per_quantum ≫ k_max → capped at k_max.
    expect = cfg.k_max_quanta_per_request \
        * h.contracts[key].dq_quant_mvar
    assert abs(executed[0].deal.dq_deal_mvar) == pytest.approx(expect)


# ───────────────────────────────────────────────────────────────────────
#  Move 2 — voltage-referenced delivery gate + tier-2 suspension
# ───────────────────────────────────────────────────────────────────────


def test_voltage_gate_suppresses_after_undelivered_cycle(plant):
    """A plant that never moves (constant v_std feeds) fails the
    voltage verification → exactly one deal executes, then requests
    are suppressed and the undelivered paid surplus is NOT billed."""
    cfg = v5_config(request_sizing="single", v_delivery_tol_pu=1e-5)
    h = Harness(plant, cfg)                 # track_refs = False
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(4)
    key = (1, 2)
    recs = h.sched.records[key]
    executed = [r for r in recs if r.deal.dq_deal_mvar != 0.0]
    # Bounded probe loop (v4 design, kept in v5): deal → undelivered →
    # suppress + unwind → evidence resets → probe again.  The gate's
    # guarantee is NO STACKING (surplus ≤ one quantum), not zero
    # retries.
    assert executed and len(executed) < len(recs)
    quantum = h.contracts[key].dq_quant_mvar
    assert max(abs(r.surplus_mvar) for r in recs) \
        <= quantum + 1e-9
    assert any(r.delivered is False for r in recs)
    assert any(r.request_suppressed_a or r.request_suppressed_b
               for r in recs)
    setts = h.sched.settlements[key]
    undelivered = [s for s in setts if s.delivered_frac == 0.0]
    assert undelivered, "no undelivered settlement recorded"
    assert all(s.tier2_eur == 0.0 for s in undelivered)
    assert any(s.paid_mvarh > 0.0 for s in undelivered), \
        "paid surplus existed but was billed despite non-delivery"


def test_voltage_gate_stays_open_for_tracking_plant(plant):
    """A perfectly tracking plant (feeds follow the references incl.
    the acting dv) verifies delivery → deals keep executing and the
    tier-2 billing stays active."""
    cfg = v5_config(request_sizing="single", v_delivery_tol_pu=1e-5)
    h = Harness(plant, cfg)
    h.track_refs = True
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(4)
    key = (1, 2)
    recs = h.sched.records[key]
    assert sum(1 for r in recs if r.deal.dq_deal_mvar != 0.0) >= 2
    assert all(r.delivered is not False for r in recs)
    setts = h.sched.settlements[key]
    assert any(s.tier2_eur > 0.0 for s in setts)
    assert all(s.delivered_frac == 1.0 for s in setts)
