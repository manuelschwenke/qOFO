"""
sbxv — SBX-V: vertical band-and-request coordination (TSO–DSO)
===============================================================
Implementation of the SBX-V build plan
(`SBXV_TSO_DSO_Coordination_Build_Plan.md` v1.0, 2026-07-08) on top of
the unchanged CAIR cascade.  Status tracking: `STATUS_SBXV.md`.

Phase 1 modules: :mod:`sbx_v.config`, :mod:`sbx_v.directions`,
:mod:`sbx_v.band`, :mod:`sbx_v.miqp_cost`.

Hard rule 5: no modification of CAIR, SBX-H, BME, or MSR/MSC modules —
everything here integrates via import, wrapping, or external
composition (see :class:`sbx_v.miqp_cost.PricingSolver`).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 1)
"""
