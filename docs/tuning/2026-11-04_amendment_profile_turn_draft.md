# Amendment draft: profile-turn exemption of reversals (fallback for v4)

**Status:** DRAFT 2026-11-04 ≈ 15:50, planned at the author's request
("can you plan this already?"). It is not applied, and run 0008 (procedure v4) continues
unchanged.
- The rule's parameters were fixed before the dry evaluation below was run.
- However, they were fixed after the partial II-a results of run 0008 had been seen.
  If the amendment is adopted, the thesis must state this order.

## Purpose

- A tap that moves shortly before a turn of the exogenous injection profile, and back
  shortly after it, follows the profile. Such a reversal is not hunting.
  - Under v4, with zero tolerance and the ≤ 1 h dwell, it is counted.
  - In run 0008 this happens at the 16:00 wind peak of W6.
- The exemption is defined by cause, not by window, and applies in every window: search,
  completion, W5, and the MC reporting.
- Reversals at a bank switching stay counted. They are the tap–bank coupling that the hold
  is meant to remove.

## Rule

**Signal.** The residual load of control area a,

  r_a(t) = Σ_{loads ∈ a} P_base · s(t) − Σ_{IBG ∈ a} P_base · s(t)   [MW],

- s(t) are the scenario's profile factors (`data/profiles.csv`), with the accounting of
  `network/ieee39/zonal_balancing._zone_residual_series` (`BOUND_` sgens skipped).
- It is exogenous and known before the simulation.
- It is used only to evaluate a run. No controller sees it, so the controllers' model-only
  view is untouched.

**Areas.**
- STS coupler of DSO_k → DSO_k (its HV network).
- TS OLTC of zone z → zone z, including its HV networks (the dispatch zone map).
- Bank of zone z → zone z.

**Turn.** t* is a turn of r_a if r_a(t*) is the minimum (injection peak) or the maximum
(injection trough) of r_a on [t* − H, t* + H], and r_a moves by at least δ·C_a away from
r_a(t*) on each side within H.
- H = 1 h, the dwell horizon.
- δ = 0.05.
- C_a is the installed IBG active power of area a.

**Exemption.** A reversal (t0, t1) of device d is *profile-explained* if all of the
following hold. A reversal is a pair of opposite moves, as counted under v4: dwell ≤ 1 h,
not event-explained, not start-up exempt.
1. A turn t* of r_{a(d)} lies in [t0 − L, t1], with L = 900 s.
2. If d is a tap changer: no bank move anywhere in the network lies within ±60 s of t0 or
   t1. Otherwise the reversal is *bank-coupled* and counted.
3. It is d's first such reversal for this t*. Any further reversal around the same turn is
   counted, so a limit cycle that straddles a turn is still caught.

**Parameters and why.**

| Parameter | Value | Reason |
|---|---|---|
| H | 1 h | the dwell horizon of v4: a slower reversal is not counted anyway |
| δ | 5 % of C_a | removes the 15-min wiggles of the load profiles; chosen a priori, not fitted |
| L | 900 s | one minimum TS tap hold, (1 + `tso_int_cooldown`) · T_TS = 5 · 180 s: a first move up to one hold after the turn still answers the pre-turn trend |
| bank tolerance | 60 s | three STS steps; the coupler reaction after a bank commit was observed at +20 s |

## Decisions for the author

- **D1. Banks.** Exempt bank reversals at a turn too (recommended for consistency: the
  dwell rule already treats taps and banks alike), or taps only.
- **D2. Trigger.**
  - (a) Fallback only: adopt if run 0008 ends NOGO, or without an `sts+bank` candidate.
    This is recommended.
  - (b) Adopt regardless. Then the selection changes, because the feasible set grows, so a
    new campaign is needed even if run 0008 ends GO.
- **D3. Application** (if adopted):
  - (a) A full campaign v4.1 with the exemption from II-a on: ≈ 14–18 h. Recommended:
    one rule throughout, simplest to describe.
  - (b) Re-entry from run 0008. Rescore every stored row, re-simulate only the
    candidates whose stopping reversal is exempt, then rerun the selection and everything
    after it.
    - This saves part of II-a (≈ 3 h). II-b onward is still repeated.
    - The early stop truncated the other windows of every rejected candidate, so stored
      rows alone cannot be rescored to a verdict.
- **Not part of this draft: the W4 start-up boundary.**
  - In 12 W4 rejections, the TS OLTC 1[1] moves at exactly 900 s and reverses at 1800 s.
  - The start-up exemption requires t0 < 900 s.
  - Including t0 = 900 s would be a separate amendment, with its own decision.

## Implementation plan

- **Where.** Not in `F:\qOFO_res` while run 0008 is running.
  - Its later stages start fresh local shards that import this checkout's code, while
    e5power keeps the code copied at the launch.
  - Use a separate worktree from HEAD, merged only after run 0008 has ended.
- **Code:**
  1. `tuning/profile_turns.py`:
     - `area_residuals(net, meta, zone_defs, profiles, start, hours, dt_s=60)`;
     - `find_turns(r, cap_mw, side_s=3600, prom_frac=0.05)`;
     - `area_of(device_key)`.
  2. `tuning/stage2_search.py`:
     - `_split_reversals(..., turns=None, turn_lag_s=900, bank_moves=None, bank_tol_s=60)`
       gets a fourth category, profile-explained;
     - `hunting_summary` / `shunt_summary` report `n_profile_explained` and its detail.
     - `turns=None` reproduces v4 exactly.
  3. `ch_9_3_stage2_screen.py`:
     - compute the turns per window from a `pre_loop_hook` that captures net, meta and
       `zone_defs` and continues;
     - store them in the row; pass them to the abort hook and the summaries;
     - options `--profile-turn-exempt` and `--profile-turn-banks`.
  4. Driver `ch_9_15_procedure_v4_1.py`: v4 plus the two options, patched as v4 patches v3.
- **Tests:**
  - a synthetic peak gives one turn;
  - the prominence filter removes wiggles;
  - the lag window;
  - bank-coupled reversals stay counted;
  - one exemption per device per turn;
  - the area map;
  - regression: with the option off, every stored row of run 0008 rescores to its stored
    counts.
- **Effort:** ≈ 2–3 h including tests and a smoke run. It can be done while run 0008 is
  running, in the separate worktree.

## Dry evaluation (2026-11-04 ≈ 15:45, read-only, run 0008 II-a rows of 109 candidates)

Scripts are in `analysis/profile_turn_dry/`:
- `pt_capture.py`: a controller build per window and the residuals; no simulation.
- `pt_eval.py`: the rule applied to the rows.
- `pt_residuals.json`: the captured residuals.
- The e5power rows must be fetched into the same folder as `v4_2a_remote_rows.jsonl`.

Turns found by the rule (hours into the window):
- **W6:** 2.0 h in every area, the 16:00 wind peak.
- **W1:** zone 1 at 8.0 h.
- **W2:** zones 1–3 at 1.0 h; zone 3 also at 6.0 and 10.0 h.
- **W3:** zone 2 at 6.0 and 7.0 h; DSO_1–4 at 6.0 h.
- **W4:** zone 1 at 7.0 h; zone 2 at 2.75 h.
- **W5:** none.

| Variant | W6 tap reversals exempt / bank-coupled / still counted | Candidates whose every counted reversal is exempt | Still rejected |
|---|---|---|---|
| taps only | 24 / 37 / 9 | 24 | 83 |
| taps and banks | 24 / 37 / 9 (taps) and 29 bank reversals exempt (26 W6, 3 W4) | 45 | 62 |

- Two candidates have no reversal so far, independent of the variant.
- A candidate whose every counted reversal is exempt is not thereby feasible. Its other
  windows were cut by the early stop and must be simulated in full. These counts are an
  upper bound on the yield.
- No reversal in W1, W3 or W4 is exempted: none lies at a turn.
- The 13 W4 start-up reversals stay counted.
- The main limiter remains the bank-coupled reversals (37 in W6). This amendment does not
  address them; the hold does.

## Risks

- Declared after partial results (see Status). The parameter reasons are structural, but a
  reviewer may still read it as fitted to W6.
- The residual is a P balance, so a turn driven by reactive power is not seen. The load
  Q profiles do not follow P exactly: `mv_rural_qload` turns negative in the evening.
- Turn detection runs on 15-min profile data that is interpolated linearly. A plateau
  shifts t* by up to 15 min; L covers that.
- The W6 voltage-error floor (e_v,TS ≈ 0.015 for all candidates) is unaffected.
- For the MC, the turns must be computed per sampled window in the same way. The function
  is window-independent.
