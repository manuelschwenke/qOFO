"""
sbx — Scheduled Boundary Exchange (SBX Minimal)
===============================================
Application-oriented multi-TSO reactive-power coordination for
self-contained TSO areas: capability messages, a deterministic scheduling
rule, voltage setpoints and settlement at fixed contract prices.  No price
discovery.  Horizontal analogue of the vertical TSO–DSO CAIR +
setpoint-scheduling mechanism.

Normative documents: the SBX Minimal Build Plan v2 (2026-07-07) with the
v2.2 amendment, both recorded in ``STATUS_SBX.md`` at the repository root.
BME modules, the vertical CAIR path, the OFO MIQP assembly and the solver
wrappers are never modified by this package.
"""
