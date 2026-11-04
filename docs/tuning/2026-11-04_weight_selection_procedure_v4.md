# Controller weight selection, procedure v4 (Ch 8, Experiment B)

**Status:** FROZEN 2026-11-04 ≈ 13:15, before the first simulation of its campaign, except
for one open item: the tap hold at bank switching (below), which is decided by the author
after the parallel evaluation and is then entered here before the launch. Applied once by
`experiments/ch_09_parameter_selection/ch_9_14_procedure_v4.py`. Its output is the weight
set of the commitment base, whatever it is: `configs/paramsets/ch9_selected_uc_2026_11_v4.json`.

**Why v4.** Procedure v3 with amendments 1 and 2 (run 0006) selected 8ad82442269b with GO.
- The held-out W5 (night) counted 31 tap reversals, 22 of them at a bank switching.
- The ε_op re-measurement (thesis run 0007) showed that the screen bank W1–W6 contains no
  commitment state with the zone-2 machine G3 off. In such states the candidates' TS λ is
  about 3.5, above the contraction bound 2. The author's GA run oscillates accordingly.

The requirements were defined by the author on 2026-11-04 and are recorded with their
evidence in `docs/tuning/2026-11-04_procedure_v4_requirements_draft.md`; this note states
the rules only.

## Design envelope

- **Commitment states with G3 committed** (author: "Yes, scope filter").
  - No must-run; the merit order is unchanged.
  - States with G3 off are an additional analysis ("low synchronous generation": RA, RE,
    GA, GE), not part of the tuning.
  - The Case-2 MC samples only G3-on windows (a declared filter, implemented before the MC).

## Changes against v3 (amendments 1 and 2)

| Rule | v3 | v4 |
|---|---|---|
| Hunting tolerance (taps / banks) | 1 / 0 | **0 / 0** |
| Early stop | 2nd tap or 1st bank reversal | **1st tap or 1st bank reversal** |
| Hunting dwell | < 2 h | **< 1 h** (author: 1–2 h reversals following the load profile "does not count as hunting"), taps and banks |
| Settling acceptance | p95 ≤ 36 (4 periods) | **p95 ≤ 27 (3 TS periods)**, median residual ≤ 1 Mvar |
| Stage-I verification margin | p95 ≤ 34 | **p95 ≤ 25**, residual ≤ 0.8 Mvar |
| λ screen | λ_max(B) ≤ 1.546 on W1–W6, all columns | **λ_max(B) ≤ 1.546 without the bank columns, on W1–W6 plus the 83 G3-on fresh points of run 0007** (89 points) |
| Go: e_v,TS | ≤ 1.10 × 0.00827 | **≤ 0.01** |
| Hosts | 30 here + 36 on e5power | **10 here** (pinned to 10 logical processors) **+ 30 on e5power** |

Unchanged from v3:
- the ceiling 1.546 = 1.6 / 1.035 (author: "Keep 1.546");
- θ_TS,min ≤ 0.009 pu and the θ_STS band [0.005, 0.015] pu, both on W1–W6;
- the start-up exemption of 900 s; event merging within 30 min; a move at an event
  reversed minutes later counts;
- the engagement filters and levels: couplers in every search window; banks commit within
  2 h after an event in ≥ 2 search windows; fallback `sts+bank` → `sts` → `none`;
- the search windows W1, W3, W4, W6; completion on W2; W5 held out;
- II-a with 512 Sobol draws (same seed as v3); II-b up to 4 rounds; Stage II-c on the
  bank price; the ranking by e_v,TS with the 2 % band and tie-break e_v,DS;
- Stage I `stage1_v3_uc_2026_11_pq10` (θ_STS 0.01). Its verification from run 0005 is
  reused: p95 24.0 ≤ 25 and residual 0.37 Mvar ≤ 0.8, so it passes the v4 margin.
- `shunt_ff_ledger = True`.

### Why the banks leave the λ screen (B1b, author decision)

- Bank price floor with the bank columns in B, at the v3 leader's other weights:
  - ≈ 510 on the 89 G3-on points, set by the zone-3 bank column at F1022_14, the one
    point with a zone-3 machine off;
  - 18 without that point;
  - ≥ 29 with G3-off states.
- In run 0006 no candidate with a bank price ≥ 25 committed a bank after an event, so
  banks in λ conflict with the bank-engagement requirement.
- The banks are discrete and dwell-locked: a linear contraction argument does not describe
  their steps. The zero-tolerance hunting filter judges them on the simulated windows
  instead.
- The tap changers stay in B, as in v3.
- Precedent in the code: `central_lambda_rule = "continuous"` of variant M.

## Tap hold at bank switching (author ≈ 13:30: "Hold n = 1")

- `--config-set sts_coupler_hold_after_bank=1` in every simulated window.
  - After a TS-MIQP bank commit, the hosting STS holds all its coupler OLTCs for the one STS step of the commit instant, step 1 included.
  - The STS learns of the commit from the `ShuntDisturbanceMessage` it already receives, not from the plant.
  - Code: commit 3b4aab0, a port of the parallel session's change (run 0014, note `2026-11-04c_sts_coupler_hold_after_bank.md`).
- Evidence (run 0014, II-c candidates on W1, W3, W4, W6):
  - the hold removes every coupler move at a bank instant;
  - 17.9 goes 2 → 0 counted reversals, the leader 1 → 0;
  - e_v,TS unchanged; e_v,DS +0.0002 (mainly W4).
- The settling test is unaffected: the subordinate loop is isolated, with the TS layer frozen.
- This note is thereby complete. It is frozen as of the launch.

## Erratum (2026-11-04 15:20, after the launch; wording only, author: "yes, add erratum")

- The table states the hunting dwell as "< 1 h". The rule the code applies, and the one
  run 0008 is judged by, is **≤ 1 h**:
  - `tuning/stage2_search.py::_split_reversals` skips a move pair only when
    `(t1 - t0) > dwell_s`, so a reversal with a dwell of exactly 3600 s counts as hunting;
  - the abort hook and the result rows use the same function;
  - v3 used the same convention at 7200 s ("< 2 h" in its note, ≤ 2 h in the code).
- Nothing in the code or the run changes. Occasion: in the II-a status at ≈ 15:10, three
  finished reversals have a dwell of exactly 3600 s; the rejection of one candidate
  (d5c278b98a71, W6) rests on one of them.

## Dry run (2026-11-04 13:11)

- Stage-I point: λ v4 1.405 (v3 rule 1.447).
- II-a draws passing the screen: v4 176 / 512 (v3 rule 100 / 512).

## Go criteria for the follow-ups

Written to `GO.json` or `NOGO.json`:
- e_v,TS ≤ 0.01 on W1–W4 and W6;
- couplers act in every search window;
- banks commit after events in ≥ 2 search windows;
- all limits met, including hunting tolerance 0 and settling p95 ≤ 27.

The follow-ups (settling figure, D1+D2, Ch 10 REF runs) are not armed automatically; the
author decides at the result.

## Risks stated before the campaign

- **Zero tolerance with the 1-h dwell.** In run 0006, 11 of 264 candidates had 0 counted
  reversals under the 2-h dwell. The 1-h dwell and the tap hold should raise that number.
  If no candidate at `sts+bank` reaches zero, the fallback levels apply and the go criterion
  on banks fails.
- **Compute.** II-a has 176 candidates against v3's 101, on 40 instead of 66 workers. The
  earlier early stop partly compensates. Expected total ≈ 14–18 h.
- **Banks outside λ.** Bank-induced oscillation is caught only where the windows exercise
  it: by the zero-tolerance filter on the windows, and in G3-off states by the additional
  analysis.
