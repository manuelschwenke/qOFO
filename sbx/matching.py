"""
sbx/matching.py
===============
Deterministic matching for SBX (plan v2 §2.2 Step 3; Phase 4).

A PURE function of the two corridor messages plus the frozen contract
and the current schedule state — no randomness, no side effects.  Both
areas compute it independently; equality is proven via the canonical
serialisation checksum of the resulting :class:`DealRecord`
(``sbx.messages.assert_checksums_match``), ``rep1`` on mismatch.

Rules, in corridor-flow sign space (positive = export from end A):

* Both request, same sign → **mutual deal**:
  ``dq_deal = sign · min(|req_A|, |req_B|)``, unpaid.
* Exactly one requests → **unilateral deal**:
  ``dq_deal = clip(request, supporter offer_range)``, paid; rejected
  below ``dq_min_deal_mvar`` (dust).
* Both request, opposite signs → **no deal**, ``ScarcityEvent`` logged.
* Always: reject any ``dq_deal`` that would violate
  ``|q_sched + dq_deal − q_std| ≤ dq_contract_max_mvar``.
* At most one schedule update per corridor per cycle (one outcome per
  invocation, by construction).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 4)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from sbx.contract import CorridorContract
from sbx.fail import rep1
from sbx.messages import PeerCairMessage

#: Outcome kinds (DealRecord.kind).
KIND_MUTUAL = "mutual"
KIND_UNILATERAL = "unilateral"
KIND_NONE = "none"
KIND_SCARCITY = "scarcity"

#: Rejection reasons (DealRecord.reason; None for executed deals).
REASON_NO_REQUEST = "no_request"
REASON_OPPOSITE_SIGNS = "opposite_signs"
REASON_DUST = "below_dust_threshold"
REASON_CONTRACT_CAP = "contract_cap"


@dataclass(frozen=True)
class DealRecord:
    """Deterministic outcome of one corridor matching cycle.

    ``dq_deal_mvar`` is 0 for every non-executed outcome.  ``requester``
    / ``supporter`` are area ids of a UNILATERAL deal (supporter = the
    non-requesting side); both are ``None`` for mutual deals, where the
    two areas request complementary changes and nobody supplies for the
    other.  ``paid`` marks tier-2 relevance (unilateral only, §2.5).
    """

    corridor: Tuple[int, int]
    cycle: int
    kind: str
    dq_deal_mvar: float
    paid: bool
    requester: Optional[int]
    supporter: Optional[int]
    reason: Optional[str]

    def canonical_serialisation(self) -> str:
        payload = {
            "corridor": list(self.corridor),
            "cycle": self.cycle,
            "kind": self.kind,
            "dq_deal_mvar": repr(float(self.dq_deal_mvar)),
            "paid": self.paid,
            "requester": self.requester,
            "supporter": self.supporter,
            "reason": self.reason,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def checksum(self) -> str:
        return hashlib.sha256(
            self.canonical_serialisation().encode("utf-8")
        ).hexdigest()


def match(
    msg_a: PeerCairMessage,
    msg_b: PeerCairMessage,
    contract: CorridorContract,
    q_sched_mvar: float,
    q_std_mvar: float,
) -> DealRecord:
    """Step-3 deterministic matching (plan §2.2) — one outcome per cycle.

    ``msg_a`` must be sent by the reference end (area A of the corridor),
    ``msg_b`` by area B; both areas assemble the identical argument tuple
    (their own message plus the received one, ordered by role), so the
    result — and its checksum — is identical on both sides.
    """
    a, b = contract.area_a, contract.area_b
    if msg_a.corridor != (a, b) or msg_b.corridor != (a, b):
        rep1("messages do not belong to this contract's corridor",
             contract=(a, b), msg_a=msg_a.corridor, msg_b=msg_b.corridor)
    if msg_a.sender_area != a or msg_b.sender_area != b:
        rep1("matching expects msg_a from end A and msg_b from end B",
             end_a=a, end_b=b,
             sender_a=msg_a.sender_area, sender_b=msg_b.sender_area)
    if msg_a.cycle != msg_b.cycle:
        rep1("messages stem from different cycles",
             cycle_a=msg_a.cycle, cycle_b=msg_b.cycle)
    if not (math.isfinite(q_sched_mvar) and math.isfinite(q_std_mvar)):
        rep1("schedule state must be finite",
             q_sched_mvar=q_sched_mvar, q_std_mvar=q_std_mvar)

    quantum = contract.dq_quant_mvar
    for msg in (msg_a, msg_b):
        if msg.request_mvar is not None and \
                not math.isclose(abs(msg.request_mvar), quantum,
                                 rel_tol=0.0, abs_tol=1e-9):
            rep1("request magnitude must equal the per-cycle quantum "
                 "(plan §2.2 Step 2)",
                 sender=msg.sender_area, request_mvar=msg.request_mvar,
                 dq_quant_mvar=quantum)

    cycle = msg_a.cycle
    key = (a, b)
    req_a, req_b = msg_a.request_mvar, msg_b.request_mvar

    def _cap_ok(dq: float) -> bool:
        return abs(q_sched_mvar + dq - q_std_mvar) \
            <= contract.dq_contract_max_mvar + 1e-9

    def _rejected(kind: str, reason: str, requester=None, supporter=None
                  ) -> DealRecord:
        return DealRecord(corridor=key, cycle=cycle, kind=kind,
                          dq_deal_mvar=0.0, paid=False,
                          requester=requester, supporter=supporter,
                          reason=reason)

    if req_a is None and req_b is None:
        return _rejected(KIND_NONE, REASON_NO_REQUEST)

    if req_a is not None and req_b is not None:
        sign_a = math.copysign(1.0, req_a)
        sign_b = math.copysign(1.0, req_b)
        if sign_a != sign_b:
            # Opposite needs — nobody can supply; log scarcity (§2.2).
            return _rejected(KIND_SCARCITY, REASON_OPPOSITE_SIGNS)
        dq = sign_a * min(abs(req_a), abs(req_b))
        if not _cap_ok(dq):
            return _rejected(KIND_MUTUAL, REASON_CONTRACT_CAP)
        return DealRecord(corridor=key, cycle=cycle, kind=KIND_MUTUAL,
                          dq_deal_mvar=dq, paid=False,
                          requester=None, supporter=None, reason=None)

    if req_a is not None:
        requester, supporter = a, b
        request = float(req_a)
        offer_lo, offer_hi = msg_b.offer_range_mvar
    else:
        requester, supporter = b, a
        request = float(req_b)
        offer_lo, offer_hi = msg_a.offer_range_mvar

    dq = min(max(request, offer_lo), offer_hi)
    if abs(dq) < contract.dq_min_deal_mvar:
        return _rejected(KIND_UNILATERAL, REASON_DUST,
                         requester=requester, supporter=supporter)
    if not _cap_ok(dq):
        return _rejected(KIND_UNILATERAL, REASON_CONTRACT_CAP,
                         requester=requester, supporter=supporter)
    return DealRecord(corridor=key, cycle=cycle, kind=KIND_UNILATERAL,
                      dq_deal_mvar=dq, paid=True,
                      requester=requester, supporter=supporter, reason=None)
