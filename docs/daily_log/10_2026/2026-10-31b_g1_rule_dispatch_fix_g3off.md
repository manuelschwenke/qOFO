# G1 limit by the operability rule, dispatch fix (technical minima first), REF period with G3 offline

Timestamp: 2026-10-31, ~07:20–10:00 Europe/Berlin. Branch `res/commitment-base-2026-10-31`,
worktree `F:\qOFO_res`. Side session of the Case-2 main session, on the author's decisions of
the morning:
- G1 limit set by the operability rule (~07:20);
- stop the old-base STEP, and start a new tuning campaign autonomously if the check requires
  it (~07:55);
- keep ±1200 Mvar, since it is effectively non-binding (~08:30);
- declare and queue a REF period with G3 offline (~08:50).

## What changed (key structures)

| commit | file | change |
|---|---|---|
| 2091fdf, b79d4a6 | `docs/analysis/2026-10-31_g1_operability_rule.md` | rule declared before the tests; result; amendment A1 |
| 114b0ac | `configs/config.py` | `slack_q_limit_mvar` 500 → **1200** Mvar (rule A1) |
| 35b447e | `network/ieee39/zonal_balancing.py` | **bug fix** in `compute_merit_order_dispatch` (details below) |
| 0c3f5c2 | `docs/analysis/2026-10-31_ref_period_G3_offline_declaration.md` | rule for the REF period with G3 offline, declared before the search |

**Dispatch fix (35b447e).**
- **Before:** in each zone, the first committed machine was loaded up to its cap
  (`clip(rest, pmin, cap)` in merit order) before the later machines' technical minima were
  honoured. When the residual fell inside the window, the zone over-generated while hydro
  stayed at its cap, and the surplus went to the network equivalent.
- **Now:** every committed machine first gets its technical minimum. Only the residual above
  the sum of the minima is loaded in merit order up to the caps.
- **Unchanged:** commitment, inter-zone deficit handling, the equivalent as balance.
- **Effect** (`disp_cmp.py`):

| window | max change in machine P | G1 export peak, before → after |
|---|---|---|
| REF period, 6 h | 83 MW (13:45) | — |
| W1, 12 h | 395 MW | −1577 → −1273 MW |
| W6, 12 h | 280 MW | unchanged |
| RA period | 0 MW | — |

## Findings

**G1 limit**

| case | result |
|---|---|
| W1, LOC, G1 unlimited | fails at ≈19:48 after drawing 1181 Mvar from G1, so the declared rule has no solution |
| W6, LOC | needs 1062 Mvar |
| W2–W4, LOC | need < 400 Mvar |
| old-base MC (808 runs, G1 unlimited) | never more than 1000 Mvar; above 500 in 62 runs |
| new-base REF period incl. post-event states | ≤ 381 Mvar |

- Amendment A1 gives **±1200 Mvar**.
- The limit does not bind in any operable state found so far. The new base therefore differs
  from the old one through commitment, not through G1.

**W1 evening collapse**
- The old base runs W1 under LOC with G1 at about 0 Mvar.
- On the new base, the 12-h window commits all five machines for the midday peak (residual
  sum 2686 MW). At 19:45 the residual is 773 MW, and area 2 alone has a surplus of 574 MW.
- With the machines at their minima, the system must export more than 1.2 GW through G1, and
  LOC loses voltage stability.
- Under OFO, before the fix, G1 unlimited passes with 1386 Mvar from G1, but ±1200 fails at
  19:51.
- **Cause:** commitment is fixed per run window from the window's peak, so a 12-h window with
  a large residual swing over-commits in the evening. A 30-min window at 19:00 commits only
  G10 and G4 and is healthy (V_min ≈ 1.0).
- Re-test on the fixed dispatch: see the table at the end.

## Runs

**Stopped**
- **Old-base STEP** at 07:58, on the author's instruction: 404 / 1350 units, partial results
  in `F:\qOFO_weekend\results\ch_11_case2_multisystem\step\0001_...`. The weekend chain then
  logged the end of Case 1, and the queued old-base E started at 07:58.
- **`chain_res.py` waiter** (07:28): its note carried ±500 Mvar. Replaced by `chain_res2.py`.

**New-base REF**
- First start at 08:03 (runs 0001–0003). Stopped at 08:53 because of the dispatch fix.
- Restarted at 08:54 at 35b447e: A, E, B in parallel (3 × 6 workers), then D.

**e5power:** re-validation and MC not deployed. They wait for the W1 re-test on the fixed
dispatch, because W1 is a re-validation window under OFO.

## Open

- **Commitment window length** is a modelling choice with a visible effect (W1 evening). Fixed
  per window from the peak; an hourly schedule with minimum up and down times would remove it.
  That would be an architectural change and needs the author.

## Results after the fix (09:00–10:00)

**W1 and W6 re-test, ±1200**

| window | LOC | OFO |
|---|---|---|
| W1 | ok, G1 max 927 | ok, G1 max 913 |
| W6 | ok, G1 max 1087 | — |

- The re-validation windows are operable without events.
- Before the fix: W1 under OFO at ±1200 failed at 19:51. Coordination did not hold a binding
  ±500 either (W1 OFO fails ≈17:33; W6 OFO and OFO+ fail after 75 and 87 min).

**G3-offline REF period**
- Selected: **08 Jan 2016 06:00** (G10, G9, G4, G7 committed), by the declared rule.
- Search table: `docs/analysis/2026-10-31_ref_period_G3_offline_declaration.md`.
- Scenarios **GA** and **GE** in `ref/scenarios.py` (be66584). Commit 01117ea carries only the
  document; its registry edit did not apply and followed in be66584.

**Launches**

| what | host | when | details |
|---|---|---|---|
| re-validation | e5power | 09:38 | run 0010, 70 units, 20 workers; chain `chain_res2.py e5power`: MC follows if c08b is confirmed |
| GA, GE | e5wimi | queued | `chain_res3.py`: GA when the second of A/E/B ends, GE at the third (6 workers each) |
