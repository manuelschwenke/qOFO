# 2026-07-09 — SBX-V E2 sweep build + V-D9 conformance fix

**What:** Built the E2 band-width Pareto sweep (`experiments/019_SBXV_E2.py`) and, in the
process, found and fixed a V-D9 conformance gap in the Phase-4 spec schedule.

**The bug (found by design review while building E2):** `CommitScheduler.specs_for` returned
`None` whenever no grant was active — the MIQP never felt the band prices in the no-grant
regime, although V-D9 prescribes "0 in band, Grenzpreis beyond, when no grant is active" and
the settlement charged the ad-hoc exceedance (incentive/settlement inconsistency; E1 finding 4
was the observable symptom: unpriced beyond-band dispatch).

**The fix:** band prices act from window 0; the neutral/R1 bypass is an explicit switch
`SBXVConfig.miqp_pricing_enabled=False` (used by the R1 arms of the smoke and 018). Updated
Phase-4 tests (band priced from window 0, grant activation shown by segment structure instead
of None→spec). Suite 101 green; R1 smoke re-run byte-identical; 30-min stressed probe shows
the steering (no-grant exceedance 1.7 Mvarh — pulled back to the edge). E1 §5.3 numbers
predate the fix and will be re-run.

**E2 script:** arms none / ±{0,25,50,75,100} Mvar / `ar41414_default` (adapter now derives
contracted P from Σ `sn_hv_mva`; an AR Anhang C rejection is a recorded finding); scenarios
from the 012/006 Monte-Carlo idiom (seeded random profile start + contingency schedule — the
stress that fires condition B); per-cell JSON resume; metrics per plan E2 including the
persistent-exceedance metric (no-grant Grenzpreis windows, longest consecutive run, Mvarh);
tidy `e2_sweep.csv` + 4-panel `e2_pareto.png`. Campaign (3 seeds × 7 arms × 120 min) running.

**Why:** Plan §9 Phase 5 E2; the fix restores V-D1/V-D9 separation-of-planes correctness
(prices always, feasibility never).
