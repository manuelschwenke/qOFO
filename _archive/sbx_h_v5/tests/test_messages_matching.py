"""
SBX Phase 4 — tests for ``sbx_h.messages`` and ``sbx_h.matching``.

Acceptance (plan v2 §4 Phase 4): unilateral clip, mutual min,
opposite-sign scarcity, dust rejection, contract cap, checksum mismatch
abort, determinism (both sides byte-identical deal records).
"""
from __future__ import annotations

import dataclasses

import pytest

from sbx_h.contract import CorridorContract
from sbx_h.fail import SBXError
from sbx_h.matching import (
    KIND_MUTUAL,
    KIND_NONE,
    KIND_SCARCITY,
    KIND_UNILATERAL,
    REASON_CONTRACT_CAP,
    REASON_DUST,
    REASON_OPPOSITE_SIGNS,
    match,
)
from sbx_h.messages import (
    SBX_MESSAGE_VERSION,
    PeerCairMessage,
    assert_checksums_match,
)

#: Test corridor (1,2) with two lines; quantum = 10 Mvar (15-min cycle).
CONTRACT = CorridorContract(
    area_a=1, area_b=2,
    line_indices=(2, 14),
    v_std_a_pu=(1.02201, 1.02866),
    v_std_b_pu=(1.01523, 1.04547),
    q_band_mvar=5.0,
    dq_quant_rate_mvar_per_15min=10.0,
    dq_contract_max_mvar=50.0,
    dq_min_deal_mvar=1.0,
    p_surplus_eur_per_mvarh=5.0,
    kappa_penalty=2.0,
    k_sched=5,
    t_cycle_min=15.0,
)
QUANT = CONTRACT.dq_quant_mvar   # 10.0


def _msg(sender, *, request=None, offer=(-10.0, 10.0), cycle=7):
    receiver = 2 if sender == 1 else 1
    return PeerCairMessage(
        version=SBX_MESSAGE_VERSION,
        sender_area=sender,
        receiver_area=receiver,
        corridor=(1, 2),
        cycle=cycle,
        offer_range_mvar=offer,
        request_mvar=request,
        p_sched_mw=(373.4, 71.2) if sender == 1 else None,
    )


# ---------------------------------------------------------------------------
#  Message validation and checksum protocol
# ---------------------------------------------------------------------------


def test_message_reference_end_carries_p_sched():
    with pytest.raises(SBXError, match="must carry p_sched"):
        PeerCairMessage(SBX_MESSAGE_VERSION, 1, 2, (1, 2), 0,
                        (-10.0, 10.0), None, None)
    with pytest.raises(SBXError, match="only the reference-end sender"):
        PeerCairMessage(SBX_MESSAGE_VERSION, 2, 1, (1, 2), 0,
                        (-10.0, 10.0), None, (1.0,))


def test_message_offer_must_contain_zero():
    with pytest.raises(SBXError, match="must contain zero"):
        _msg(2, offer=(2.0, 10.0))


def test_message_checksum_roundtrip_and_mismatch_abort():
    m1 = _msg(1, request=+QUANT)
    m2 = _msg(1, request=+QUANT)
    assert m1.canonical_serialisation() == m2.canonical_serialisation()
    assert_checksums_match(m1.checksum(), m2.checksum(),
                           corridor=(1, 2), cycle=7, what="messages")
    m3 = _msg(1, request=-QUANT)
    with pytest.raises(SBXError, match="checksum mismatch"):
        assert_checksums_match(m1.checksum(), m3.checksum(),
                               corridor=(1, 2), cycle=7, what="messages")


# ---------------------------------------------------------------------------
#  Matching rules (Step 3)
# ---------------------------------------------------------------------------


def test_no_request_no_deal():
    deal = match(_msg(1), _msg(2), CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_NONE and deal.dq_deal_mvar == 0.0


def test_unilateral_deal_full_quantum():
    deal = match(_msg(1, request=-QUANT), _msg(2), CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_UNILATERAL
    assert deal.dq_deal_mvar == pytest.approx(-QUANT)
    assert deal.paid
    assert deal.requester == 1 and deal.supporter == 2


def test_unilateral_deal_clipped_to_offer():
    deal = match(_msg(1, request=-QUANT),
                 _msg(2, offer=(-4.0, 4.0)), CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_UNILATERAL
    assert deal.dq_deal_mvar == pytest.approx(-4.0)
    assert deal.paid


def test_unilateral_dust_rejected():
    deal = match(_msg(1, request=-QUANT),
                 _msg(2, offer=(-0.5, 0.5)), CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_UNILATERAL
    assert deal.dq_deal_mvar == 0.0 and not deal.paid
    assert deal.reason == REASON_DUST


def test_mutual_deal_min_of_requests_unpaid():
    # A needs import (sign -1 at end A), B needs export (sign -1 at
    # end B is an overvoltage in B: request_sign('b') = -1) -> same sign.
    deal = match(_msg(1, request=-QUANT), _msg(2, request=-QUANT),
                 CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_MUTUAL
    assert deal.dq_deal_mvar == pytest.approx(-QUANT)
    assert not deal.paid
    assert deal.requester is None and deal.supporter is None


def test_opposite_signs_scarcity():
    deal = match(_msg(1, request=-QUANT), _msg(2, request=+QUANT),
                 CONTRACT, 0.0, 0.0)
    assert deal.kind == KIND_SCARCITY
    assert deal.dq_deal_mvar == 0.0
    assert deal.reason == REASON_OPPOSITE_SIGNS


def test_contract_cap_rejects_deal():
    # Surplus already at +45 Mvar; a +10 deal would breach the 50 cap.
    deal = match(_msg(1, request=+QUANT), _msg(2), CONTRACT,
                 q_sched_mvar=45.0, q_std_mvar=0.0)
    assert deal.kind == KIND_UNILATERAL
    assert deal.dq_deal_mvar == 0.0
    assert deal.reason == REASON_CONTRACT_CAP
    # Towards the standard the same request is fine.
    deal2 = match(_msg(1, request=-QUANT), _msg(2), CONTRACT,
                  q_sched_mvar=45.0, q_std_mvar=0.0)
    assert deal2.dq_deal_mvar == pytest.approx(-QUANT)


def test_request_magnitude_must_be_quantum():
    with pytest.raises(SBXError, match="quantum"):
        match(_msg(1, request=-3.0), _msg(2), CONTRACT, 0.0, 0.0)


def test_matching_determinism_byte_identical():
    """Both sides assemble (msg_A, msg_B) by ROLE, so the records match."""
    # Side A's view of the exchange:
    deal_at_a = match(_msg(1, request=-QUANT), _msg(2, offer=(-7.0, 7.0)),
                      CONTRACT, 12.0, 10.0)
    # Side B constructs the identical inputs independently:
    deal_at_b = match(_msg(1, request=-QUANT), _msg(2, offer=(-7.0, 7.0)),
                      CONTRACT, 12.0, 10.0)
    assert deal_at_a.canonical_serialisation() \
        == deal_at_b.canonical_serialisation()
    assert_checksums_match(deal_at_a.checksum(), deal_at_b.checksum(),
                           corridor=(1, 2), cycle=7, what="deal records")


def test_matching_rejects_role_confusion():
    with pytest.raises(SBXError, match="msg_a from end A"):
        match(_msg(2), _msg(2), CONTRACT, 0.0, 0.0)
    with pytest.raises(SBXError, match="different cycles"):
        match(_msg(1, cycle=7), _msg(2, cycle=8), CONTRACT, 0.0, 0.0)
