# Reactive limit of the network equivalent G1: operability rule (declared before the tests)

Author decision 2026-10-31, ~07:20 (reply "Operability rule" to the status report
`2026-10-31_commitment_base_status.md`, §3 decision 1). Declared and committed by Claude Code
before any of the tests in §3 ran.

## 1. Rule

$\bar Q_{G1}$ is the **smallest multiple of 100 Mvar** for which the plant is operable without
disturbances under the uncoordinated variant LOC (code L2), on the new base (merit-order unit
commitment, `unit_commitment = True`):

- **Windows:** the five Ch 9 design windows W1, W2, W3, W4, W6, at the start times of
  `tuning/scenarios_mc_v2.py`, each over 12 h.
- **Configuration:** the Case-2 registry (`variants.resolved("L2", ...)`), no events, plant step
  dt = 180 s (the MC prescreen step), noise seed 20261027.
- **Pass:** all 240 steps with a converged power flow and no exception. The voltage band is not
  part of the rule.
- G1's box is symmetric, $Q_{G1} \in [-\bar Q_{G1}, +\bar Q_{G1}]$.

The REF period (05 Jan 2016, 08:00–14:00) is the first 6 h of W1. Its commitment is the same
(all five machines, both over 6 h and over 12 h), so W1 covers it.

## 2. Why this form

- LOC is the reference without coordination. If the plant is inoperable under LOC without any
  disturbance, every comparison in that window is a statement about feasibility, not about
  control quality.
- Design windows only. The MC start times (test data) do not enter the rule. In the MC, the
  existing prescreen (L2, no events, dt = 180 s; `run_mc.py`) discards draws in which the base
  case is inoperable. Failures after events are outcomes (`fail_pf` / `fail_abort`). They are
  never a reason to discard, as declared on 2026-10-07.
- The 100-Mvar grid fixes the resolution before the result is known.

## 3. Evidence before the rule (night 30/31 Oct; LOC, no events, dt = 180 s)

| window | committed | G1 max at ±500 | ±500 | ±1000 | unlimited |
|---|---|---|---|---|---|
| W1 05 Jan 08:00 | 5 | +506 (at limit) | fails ≈16:54 | — | — |
| W2 22 Feb 13:00 | 5 | +385 | ok | — | — |
| W3 04 Jan 08:00 | 5 | +216 | ok | — | — |
| W4 26 Nov 10:00 | 5 | +94 | ok | — | — |
| W6 18 Jul 14:00 | 4 (G9 off) | +509 (at limit) | fails after 75 min | fails after 114 min | ok, max +1062 |

## 4. Result of the declared rule (tests 31 Oct 07:25–08:00)

LOC, no events, dt = 180 s. "Fails k/240" means the power flow diverged at step k, i.e. at
08:00 + 3k min in W1.

| window | ±1100 | ±1200 | unlimited |
|---|---|---|---|
| W1 05 Jan 08:00 | fails 233/240 (≈19:39), G1 max 1068 | fails 235/240 (≈19:45), G1 max 1181 | **fails 236/240 (≈19:48), G1 max 1181** |
| W2 22 Feb 13:00 | ok, G1 max 385 | — | — |
| W3 04 Jan 08:00 | ok, G1 max 216 | — | — |
| W4 26 Nov 10:00 | ok, G1 max 94 | — | — |
| W6 18 Jul 14:00 | ok, G1 max 1062 | see §5 | ok, G1 max 1062 (night run) |

**The declared rule cannot be met.** W1 loses voltage stability in the last 12–21 min of its
evening peak under LOC **even with G1 unlimited** (V_min 0.927 pu). No value of
$\bar Q_{G1}$ makes W1 operable, so the rule as written in §1 has no solution.

## 5. Amendment A1 (31 Oct ~08:05)

**Status:** written after the results of §4 and before the W6 test at ±1200 and the W1
diagnostics. Claude Code made it under the author's standing instruction to proceed
autonomously. **It needs the author's confirmation.**

**Rule A1.** $\bar Q_{G1}$ is the smallest multiple of 100 Mvar at which the limit is not the
cause of an inoperability in any design window (configuration as in §1):

- (i) every window that is operable with G1 unlimited is also operable at $\bar Q_{G1}$;
- (ii) in a window that fails with G1 unlimited, G1's converged Q stays below $\bar Q_{G1}$
  until that failure.

**Why.** The purpose stated in §2 is that the limit of the equivalent does not decide
operability by itself. W1's evening collapse occurs without any limit, so it is not an effect
of the limit. A1 keeps the purpose and drops the part that no value of the limit can meet.

**Application.**
- W6 needs 1062 Mvar, so (i) requires ≥ 1100.
- W1 needs 1181 Mvar before it collapses, so (ii) requires ≥ 1200.
- W2–W4 stay below 400 Mvar.
- **$\bar Q_{G1}$ = 1200 Mvar**, provided W6 passes at ±1200. W6 passes when unlimited and
  stays below 1200, so its trajectory is the unlimited one.
- At ±1200, W1 collapses one step (3 min) earlier than unlimited (235 vs 236 records),
  while G1's converged Q never exceeds 1181. The limit can act transiently in the
  iterations of the failing solve. That step lies at the collapse edge of the unlimited
  run.

**Diagnostics** (running, results in §6):
- W1 under LOC on the old base, to see whether the evening collapse comes from the new
  dispatch;
- W1 under OFO (code O2) at ±1200 and unlimited, to see whether coordination holds the
  window.

## 6. Diagnostics and re-test on the fixed dispatch (31 Oct 08:00–09:35)

**Before the dispatch fix** (commit 114b0ac):

| case | result |
|---|---|
| W1, LOC, **old base** (no commitment, G1 unlimited) | ok, G1 Q ≤ +5 Mvar |
| W1, OFO, G1 unlimited | ok, G1 max 1386 Mvar |
| W1, OFO, ±1200 | fails 237/240 (≈19:51) |
| W1, OFO, ±500 | fails 191/240 (≈17:33) |
| W6, OFO, ±500 | fails after 75 min |
| W6, OFO+, ±500 | fails after 87 min |

- Coordination does not hold W1 or W6 at ±500.
- **Cause of the W1 evening collapse** (`w1d.py`):
  - the 12-h window commits all five machines for the midday peak;
  - the dispatch loaded hydro to its cap before the other machines' minima;
  - at 19:45 the equivalent therefore had to absorb a 1577 MW export.
- The dispatch bug is fixed in 35b447e.

**After the fix** (35b447e), ±1200:

| window | LOC | OFO |
|---|---|---|
| W1 | ok, G1 max 927, V_min 0.959 | ok, G1 max 913 |
| W6 | ok, G1 max 1087 | — |

**Consequence for the value.**
- On the fixed dispatch, W1 is operable under LOC, so the original rule of §1 has a solution
  again. Amendment A1 is not needed for W1.
- W6 needs 1087 Mvar, so §1 and A1 both give ≥ 1100, unless W6 also passes with the limit
  binding at 1000 or below (not tested).
- **The author kept ±1200 (31 Oct ~08:30) as an effectively non-binding bound.** It does not
  bind in any operable state found, and neither would ±1100. The difference has no effect on
  any run observed.
- Recorded here rather than changed again.
