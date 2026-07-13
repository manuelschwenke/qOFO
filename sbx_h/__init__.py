"""
sbx_h — Scheduled Boundary Exchange, horizontal (SBX-H v6)
==========================================================
TSO–TSO reactive-power coordination for self-contained areas, reduced
to the mechanism the 015 evidence campaign (STATUS_SBX.md findings
G1–G7) showed to carry the value:

* **Contract layer** — agreed per-corridor boundary-voltage schedules
  ``v_std`` (constant snapshot, hourly planning schedule, or planned
  SUPPORT intervals where one side holds a deliberately raised
  voltage), tracked by each area's own controller with priority
  weight.  Implied standard flows ``q_std`` follow from the contracted
  π-line model.
* **Metering + attributed settlement** — per elapsed cycle: in-band
  netting (tier 1) and beyond-band deviations attributed per line to
  the A-side / B-side terminal state or the P-transfer (C_A/C_B/C_P
  decomposition); the dominant voltage side pays (causer-pays).  The
  ex-post remuneration of over-performance (architecture candidate A1,
  ``docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md``) hooks in here once its
  review closes.
* **Escalation indicator** — persistent violations / persistent
  beyond-band exceedance are flagged for a slow re-planning loop
  (candidate A4); no runtime action is taken.

The v5 runtime deal layer (requests, offers/capability LPs, matching,
delivery gate, unwind, tier-2 billing) was REMOVED on 2026-07-12:
commanded quanta proved unverifiable against natural flow shifts (G3),
physically marginal (G4) and never armed by an honest exhaustion test
(G7).  The complete v5 code is archived in ``_archive/sbx_h_v5/``.

BME modules, the vertical CAIR path, the OFO MIQP assembly and the
solver wrappers are never modified by this package.
"""
