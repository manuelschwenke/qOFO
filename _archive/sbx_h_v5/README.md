# SBX-H v5 code archive (frozen 2026-07-12)

Snapshot of `sbx_h/`, `tests/sbx_h/` and `visualisation/plot_sbx.py`
taken immediately BEFORE the v6 cleanup that removed the runtime deal
layer (request/grant/matching/offers/capability LPs, delivery gate,
C1 arming, unwind, tier-2 billing).

Why removed: the 015 helpfulness campaign (STATUS_SBX.md findings
G1–G7) showed the deal layer is unverifiable at quantum scale (G3),
physically marginal (G4: +0.166 pu·step ceiling at delivery ratio
0.16) and never armed by an honest exhaustion test (G7: the contract
tier alone recovers every constructed scenario). The v6 architecture
(docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md, recommendation A1+A4)
keeps scheduled boundary voltages + ex-post attributed settlement +
escalation indicator, and adds planned support as a schedule product.

NOT an importable package — reference copy only. The v5 state also
exists in git history once the 2026-07-10/12 work is committed.

## experiments/ (added 2026-07-13)

The deal-era experiment scripts, archived with the mechanism they
exercised: `013_SBX_LADDER.py` (Phase-7 deal campaign),
`014_SBX_SINGLE_DEMO.py` (v5 single-run demo with deal markers),
`016_SBX_ABLATION.py` (quantum/cycle ablation, finding F9).  Their
RESULT directories remain under `results/` and their findings in
STATUS_SBX.md.  `017_SBX_PLANNING.py` stayed in `experiments/`
(decoupled from 013; still the v6 planning pre-pass), and
`014_SBX_SINGLE_DEMO.py` was replaced by a v6 version.
