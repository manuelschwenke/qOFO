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

## 4. Tests and result

Filled in after the runs (§4 of the daily log of the day).
