"""
sbx_h — Scheduled Boundary Exchange, horizontal (SBX-H v6)
==========================================================
TSO–TSO reactive-power coordination for self-contained areas, reduced
to the mechanism the 015 evidence campaign (STATUS_SBX.md findings
G1–G7) showed to carry the value:

* **Contract layer** â€” agreed per-corridor boundary-voltage schedules
  ``v_std`` (controller-intended terminal references by default, an
  explicit planning schedule when supplied, and planned SUPPORT
  intervals where one side holds a deliberately raised voltage). Current
  experiments use a constant 1.03 pu schedule on every side,
  tracked by each area's own controller with the ordinary voltage
  weight by default. Implied standard flows ``q_std`` follow from the
  contracted pi-line model.
* **Metering + support-energy settlement** — per elapsed cycle the Q
  baseline is evaluated at the active scheduled terminal voltages and
  measured P transfer. If exactly one side violates its symmetric
  schedule band, the other side holds, and beyond-band Q has the
  relieving sign, the violating side pays the holder for delivered
  support energy. Strength is reported
  only as an optional ex-post diagnostic, not sold as a product.
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
