"""
sbx_h/messages.py
===============
Peer CAIR message for SBX (plan v2 §2.2 Step 2; Phase 4).

One message each way per corridor and cycle.  Fields per plan §2.2:

* ``offer_range_mvar = (dq_off_min, dq_off_max)`` — signed range of
  corridor-flow CHANGES this area can support, from the joint-box
  capability LP (§2.4 / v2.2 D13), already clipped to ±dq_quant.
* ``request_mvar`` (optional) — signed corridor-flow change, magnitude
  a positive integer multiple of the per-cycle quantum (v5 gap-sized
  requests; exactly one quantum under ``request_sizing = "single"``),
  present iff the need flag (§2.3) is set AND the v5 C1 arming holds;
  sign from the violation direction.
* ``p_sched_mw`` — per-line cycle-averaged measured P of the previous
  cycle (persistence), carried by the REFERENCE-END sender only (the
  area that is end A of the corridor); ``None`` from the other side.

Schema mirrors the vertical CAIR dataclass style in its BME form
(``core/coordination_bus.py``): frozen dataclass, versioned, validated
in ``__post_init__``, fail-fast via ``rep1``.  Sign convention
throughout: corridor-flow space, positive = export from the reference
end A (plan §2.1).

Canonical serialisation and checksum (§2.2 Step 3): both sides serialise
deterministically (sorted keys, ``repr`` floats — bit-exact round trip)
and exchange SHA-256 checksums; :func:`assert_checksums_match` calls
``rep1`` on mismatch.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 4)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from sbx_h.fail import rep1

#: Message schema version; bumped on any field change.
SBX_MESSAGE_VERSION = 1


@dataclass(frozen=True)
class PeerCairMessage:
    """One area's per-corridor, per-cycle SBX message (plan §2.2 Step 2)."""

    version: int
    sender_area: int
    receiver_area: int
    corridor: Tuple[int, int]
    cycle: int
    offer_range_mvar: Tuple[float, float]
    request_mvar: Optional[float]
    p_sched_mw: Optional[Tuple[float, ...]]

    def __post_init__(self) -> None:
        if self.version != SBX_MESSAGE_VERSION:
            rep1("unsupported SBX message version",
                 got=self.version, expected=SBX_MESSAGE_VERSION)
        a, b = self.corridor
        if not (a < b):
            rep1("corridor key must be (smaller, larger) area pair",
                 corridor=self.corridor)
        if {self.sender_area, self.receiver_area} != {a, b}:
            rep1("sender/receiver must be exactly the corridor's two areas",
                 sender=self.sender_area, receiver=self.receiver_area,
                 corridor=self.corridor)
        if self.cycle < 0:
            rep1("cycle index must be non-negative", cycle=self.cycle)
        lo, hi = self.offer_range_mvar
        if not (math.isfinite(lo) and math.isfinite(hi)):
            rep1("offer_range_mvar must be finite",
                 offer_range_mvar=self.offer_range_mvar)
        if not (lo <= 0.0 <= hi):
            # Zero change is always supportable; an offer excluding it is
            # a capability-side bug, not a negotiable position.
            rep1("offer_range_mvar must contain zero",
                 offer_range_mvar=self.offer_range_mvar)
        if self.request_mvar is not None:
            if not math.isfinite(self.request_mvar) or self.request_mvar == 0.0:
                rep1("request_mvar must be a finite non-zero signed value",
                     request_mvar=self.request_mvar)
        # p_sched is carried by the reference-end (area_a) sender only.
        if self.sender_area == a:
            if self.p_sched_mw is None or len(self.p_sched_mw) == 0:
                rep1("reference-end sender must carry p_sched_mw",
                     corridor=self.corridor, sender=self.sender_area)
            if not all(math.isfinite(p) for p in self.p_sched_mw):
                rep1("p_sched_mw contains non-finite entries",
                     corridor=self.corridor, p_sched_mw=self.p_sched_mw)
        elif self.p_sched_mw is not None:
            rep1("only the reference-end sender may carry p_sched_mw",
                 corridor=self.corridor, sender=self.sender_area)

    # ------------------------------------------------------------------
    #  Canonical serialisation and checksum (Step 3 protocol)
    # ------------------------------------------------------------------

    def canonical_serialisation(self) -> str:
        """Deterministic, bit-exact string form (sorted keys, repr floats)."""
        payload = {
            "version": self.version,
            "sender_area": self.sender_area,
            "receiver_area": self.receiver_area,
            "corridor": list(self.corridor),
            "cycle": self.cycle,
            "offer_range_mvar": [repr(float(x))
                                 for x in self.offer_range_mvar],
            "request_mvar": (None if self.request_mvar is None
                             else repr(float(self.request_mvar))),
            "p_sched_mw": (None if self.p_sched_mw is None
                           else [repr(float(p)) for p in self.p_sched_mw]),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def checksum(self) -> str:
        """SHA-256 of the canonical serialisation."""
        return hashlib.sha256(
            self.canonical_serialisation().encode("utf-8")
        ).hexdigest()


def assert_checksums_match(
    local_checksum: str,
    remote_checksum: str,
    *,
    corridor: Tuple[int, int],
    cycle: int,
    what: str,
) -> None:
    """Step-3 checksum comparison; ``rep1`` on mismatch (protocol abort)."""
    if local_checksum != remote_checksum:
        rep1("SBX checksum mismatch — the two sides computed different "
             f"{what}; the corridor protocol must abort this cycle",
             corridor=corridor, cycle=cycle,
             local=local_checksum, remote=remote_checksum)
