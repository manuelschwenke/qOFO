# 2026-11-08b — Network redesign for reactive scarcity: candidates and static screen

Session: "network redesign for STS use" (author request 2026-11-08, after MC run 0002 showed
OFO+ ≈ OFO). Worktree **`F:\qOFO_net`**, branch **`res/network-north-2026-11-08`** (from
`res/commitment-base-2026-10-31` @ 3cdb0e3). **Nothing committed; nothing merged.** The base
scenario `rural_700` is unchanged by every edit below (unit tests).

## 1. Why (diagnosis of MC run 0002, `F:\qOFO_res\...\mc\0002_..._run_mc_newbase`)

- OFO+ − OFO, area 2: e_v −0.11 mpu (≈ 2 %). The gain grows as the area's TS reserve falls
  (lowest reserve quartile −0.27 mpu, highest −0.05), but the reserve never gets near zero.
- Area 2 is reactively long: load Q ≈ 168 Mvar, 345 kV charging ≈ 230 Mvar, the STSs feed
  ≈ 110 Mvar in even under OFO, ties export ≈ 40 Mvar, G3 uses ≈ 40 of ±405 Mvar.
- STS IBG capability is 0 in 21 % of steps (VDE-AR-N 4120 v2 dead zone at night/low wind).

## 2. Code changes (all opt-in by scenario name)

| File | Change |
|---|---|
| `core/actuator_bounds.py` | `RESERVED_OP_DIAGRAM = "VDE-AR-N-4120-v2-reserved"`; `_compute_single_der_q_capability` returns [−0.33, +0.41]·S_n at every P ("reserved capability", author's term) |
| `controller/der_qv_local_loop.py` | `_qv_capability` (plant clip, local loop, PF export) mirrors it |
| `network/ieee39/constants.py` | `coupling_wp_op_diagram` per DSO capacity scenario; new entries `rural_700_res`, `rural_700_north`, `rural_700_north_res`, `rural_700_res_no939`, `rural_700_north_res_no939` |
| `network/ieee39/hv_networks.py` | coupling-bus parks (3 × 70 MVA per STS, ×2 in DSO_3) take that op diagram; `_reinit_pf` gets a last rung: warm-started continuation in IBG P (50 %→100 %), keeps the last solved Q seed, restores `p_mw` (reached only if the three old rungs fail) |
| `network/ieee39/scenarios/wind_replace.py` | `ts_wind_layout="north"`: same machines removed, parks W1 (ex-G8), W2 (ex-G2), W4 (ex-G6) not built, one park of `NORTH_WIND_MW` = 1500 MW at IEEE bus 26 (grid bus 25, zone 1), profile WP10, S_n = P_n, 4120 v2 |
| `network/ieee39/scenarios/scarcity.py` (new) | `apply_scarcity`: wind replacement + lines not built (IEEE pairs); used for 9–39 |
| `network/ieee39/scenarios/__init__.py` | registry entries for the five names |
| `experiments/runners/multi_tso_dso.py` | bootstrap Jacobian: an unconverged build state takes the existing profiled-start fallback (was a `ValueError`); the fallback solves its copy once if unconverged. Bootstrap object is replaced after Phase 2 (never used for control); runs that converge today are bit-identical |
| `tests/test_network_scarcity_variants.py` (new) | 19 tests: reserved box at every P, 4120 v2 unchanged, base layout unchanged, north layout, 9–39 absent, coupling-park diagrams |
| `experiments/ch_08_setup/network_scarcity/screen_network.py`, `report_screen.py` (new) | the screen below |

Found on the way (not changed):
- At the un-profiled build state (every IBG at rated P, ≈ 2.7 GW exported to G1) the north layout
  has no PF solution above 95 % of rated IBG output (1.8 GW at B26: above 80 %). Real OPs are
  screened below.
- **The TS controllers track voltages only.** Tie lines are "Phase A: monitored only"
  (`controller/multi_tso_coordinator.py`, output layout `[V_bus | Q_PCC | I_line | Q_gen]`), and the
  paramset has no tie weight. Q imported over ties is free for the controller; CLAUDE.md lists
  boundary Q as a controlled output.
- The base network already has 345 kV lines above rating: LOC over 40 MC episodes, max line
  loading median 77 %, p95 117 %, max 174 %, > 100 % in 8 % of steps. Ratings are not modelled as
  constraints.
- The extreme wind hours of the year (XWL/XWH below) do not initialise in the **base** either.

## 3. Screen (`screen_network.py`)

Runner initialisation (LOC, `pre_loop_hook`, no tuned weight involved) at each OP, then:
voltages, 345 kV loading vs case39 ratings, area Q balance and own reserve (machines without G1 +
TS IBGs), ties, STS interface and IBG capability (4120 v2 and reserved), N-1 of every admissible
line and committed non-slack machine (IBG Q(V) loops frozen = instant after the outage; runner's
`max_iter=300` loop solve is 5–9 s and kept for a subset), and the **own-reserve ladder**: area
reactive demand raised over its 345 kV load buses, only the area's own TS resources respond
(machines with Q box, TS IBGs as voltage sources inside capability), STSs frozen. `scarce_v` =
added demand at which the area's lowest voltage leaves 0.95 pu.

Candidates: B `rural_700`, R `_res`, N `_north`, NR `_north_res`, RT `_res_no939`, NRT
`_north_res_no939`.

- Pilot = run `results/ch_08_setup/network_scarcity/0001_2026-11-08_133154_screen_network_pilot`
  (14 OPs).
- Full = run `0002_2026-11-08_134251_screen_network_full` (376 MC prescreen starts + W1–W6,
  OP-H/M + 24 extreme hours = 408 OPs × 6), launched 13:42 on e5wimi, 14 workers.

## 4. Pilot result (14 OPs, ≈ 10 initialised per candidate; direction only)

| | B | R | N / NR | RT / NRT |
|---|---|---|---|---|
| area 2 `scarce_v` p50/p10 [Mvar] | 600/390 | 600/390 | 600/290 | 400/260–370 |
| area 2 own up p10 [Mvar] | 142 | 142 | 48 | 131–135 |
| share of +300 Mvar met over ties | 0.64 | 0.64 | 0.72 | 0.50–0.55 |
| STS up, area 2, p10, 4120 → reserved | 110 → 455 | same | same | 86–98 → 430–442 |
| max line loading p50/p95 | 100/144 % | same | 114/144 % | 135–147/213–216 % |
| N-1 non-converged cases | 5 | 5 | 4 | 13–17 |
| G3 off-line (UC) | 20 % | 20 % | 10 % | 0–11 % |

Reading: reserved capability is a pure gain. The north layout lowers area 2's own reserve but
barely its voltage margin, because the ties (G1 next door via 9–39 and 2–3) carry the extra
demand. Removing 9–39 is the strongest voltage lever but overloads the remaining corridor.

## 5. Author decisions 2026-11-08 ~14:00 and the G1 question

- **"No, network only."** No tie-Q term in the TS objective; the choice comes from the screens.
- Author question: should the equivalent's Q be limited like an interconnection (345 kV,
  line-type P²+Q² limit)?
  - Model facts: G1 is a 10 GVA machine at a 10.5 kV terminal behind a 10 GVA step-up
    transformer (x ≈ 0.0012 pu on 100 MVA, effectively stiff) at IEEE bus 39, which also carries
    the equivalent's load (1104 MW / 556 Mvar at peak). Its interconnection is 1–39 (1000 MVA)
    and 9–39 (900 MVA), 168 km each; 9–39 is the tie into area 2. Q box ±1200 Mvar via
    `SlackQLimit` (operability rule of 2026-10-31).
  - Data (MC 0002, all steps of the 100 window episodes): G1 Q median 10–20 Mvar, p99 ≈ 255,
    max 375 Mvar. Lines 1–39 / 9–39 at ≤ ≈ 60 % (p95). Thermal Q headroom of 9–39 at its actual
    P: 710–870 Mvar. **A line-rating (P²+Q²) limit would not bind.**
  - Own-reserve ladder, +300 Mvar in area 2: G1 covers ≈ 28 % (base) / 34 % (north) of it,
    64 % with G3 off-line. A binding limit would be an exchange limit (few hundred Mvar), and is
    a scarcity lever.
  - Test: the 2026-10-31 operability rule (LOC, W1–W4, W6, 12 h, dt 180 s, no events) at G1
    limits unlimited / 1200 / 600 / 400 for `rural_700`, `rural_700_res`,
    `rural_700_north_res`: e5power run `0001_2026-11-08_143033_g1_operability_g1limits_e5power`
    (worktree `F:\qOFO_runs\qOFO_net`, 30 workers; local attempt 0003 stopped, ≈ 1 h per run under
    the e5wimi load). Screen candidates BG600, RG400, NRG600, NRG400 (G1 limit as a config
    override) follow on e5wimi after run 0002.
- The author's final MC will run with 30 cores on e5power and 10 on e5wimi.

## 6. Selection rule (declared 14:35, before the full-year results)

A candidate (network scenario + G1 limit) is **admissible** if, against the base B on the same
operating points:
- H1 initialisation on the MC prescreen OPs ≥ base rate − 2 percentage points;
- H2 N-1 (frozen): no outage element that fails to converge in more than 5 % of OPs unless it
  also does in the base;
- H3 post-contingency 345 kV V_min p5 ≥ 0.90 pu;
- H4 max 345 kV line loading p95 ≤ base p95 + 10 percentage points;
- H5 the G1 operability rule passes in W1–W4, W6 at its G1 limit.

Among admissible candidates, the **scarcest** wins, ranked by
- S1 the share of MC OPs with area-2 `scarce_v` ≤ 400 Mvar (the largest MC load step),
- then S2 the median area-2 `scarce_v`.

Reported but not ranked: the area-2 STS upward capability in those scarce OPs (p10); it
should be ≥ 150 Mvar with reserved capability.

## 7. Results (15:15)

### 7.1 Full-year screen, run 0002 (376 MC prescreen OPs; `evaluate_rule.py`, `rule_evaluation.txt`)

| | B | R | N / NR | RT / NRT |
|---|---|---|---|---|
| H1 init rate | 96.8 % ✓ | 96.8 % ✓ | 98.4 % ✓ | 89 % ✗ |
| H2 N-1 elements failing in > 5 % of OPs | 1-2, 1-39, G4 (base) ✓ | ✓ | 1-2, 1-39 ✓ | + 2-3, G3, G4, G10 ✗ |
| H3 post-contingency V_min p5 | 0.9055 ✓ | 0.9055 ✓ | **0.8994 ✗** (by 0.0006) | ✗ |
| H4 max line loading p95 | 130 % ✓ | 130 % ✓ | 129 % ✓ | 180 % ✗ |
| S1 share area-2 `scarce_v` ≤ 400 | 4.1 % | 4.1 % | **15.4 %** | 25–37 % |
| S1 with G3 committed | 0 % | 0 % | 0.6 % | 25–32 % |
| G3 off-line (UC) | 22 % | 22 % | 15 % | 7–16 % |

- All scarcity of B, R, N, NR is in G3-off states. In NR, with G3 off, area 2 has no TS reactive
  source at all; nearly every G3-off OP is scarce.
- NR's low post-contingency voltages are also G3-off states (outages of 1-2, 1-39, 2-25, 2-3,
  15-16), frozen droops.
- Under 4120 v2 the area-2 STS has zero upward capability in ≥ 10 % of OPs; reserved: p10 344 Mvar.

### 7.2 G1 operability rule, e5power run 0001 (pulled to `_remote_e5power/`; local 0003 archived)

| scenario | W1 G1 need | W6 G1 need | ±1200 | ±600 | ±400 |
|---|---|---|---|---|---|
| `rural_700` (B) | 1035 | **1259** | **W6 fails** (step 39) | W1, W6 fail | W1, W6 fail |
| `rural_700_res` (R) | 1035 | 1259 | W6 fails | W1, W6 fail | W1, W6 fail |
| `rural_700_north_res` (NR) | 347 | 531 | ok | ok | ok (W6 at the limit, V_min 0.926) |

W2–W4 need ≤ 220 Mvar in every case.
- **The current base violates the operability rule at its own ±1200** (W6 needs 1259 Mvar under
  LOC; on 2026-10-31 it needed 1087). Independent of the redesign; W6 is a tuning window. By the
  rule B/R would need ±1300.
- **The north layout removes most of the system's dependence on the equivalent** (W1 −66 %,
  W6 −58 %). With it, a G1 exchange limit is admissible under the declared rule down to at
  least ±400 (rule §1 = smallest operable multiple of 100; ±300/200/100 under test).

Running (15:15): e5power run 0002 (rule for NR at ±300/200/100), e5power run 0003 (screen NRG600,
NRG400 on the same 408 OPs), local run 0004 (N-1 with the runner's Q(V) loops on the 48 lowest
voltage OPs of B and NR; additional information for H3, the rule stays as declared).

## 8. Author decision 2026-11-08 ~15:30: north + reserved

**`rural_700_north_res` is accepted.** H3 is amended *after the results were known*: NR's
post-contingency V_min p5 of 0.8994 pu (0.0006 below the declared 0.90) is accepted. Author's
reasons: a rare excursion to 0.8994 is acceptable; OFO's output constraints would handle it if it
read such a measurement; with OFO+ (STS support) the value is likely higher.
Still open: the G1 limit for this network (runs above).

## 9. G1 not limited; north+reserved made the default (author ~15:40–16:00)

- Author: "maybe we just do not limit G1". Kept at the config's ±1200 Mvar, non-binding for
  NR (needs ≤ 531 Mvar in W1–W4, W6). e5power runs 0002 (rule at ±300/200/100) and 0003 (screen
  NRG600/NRG400) stopped, partial results kept there with `STOPPED_BY_AUTHOR.txt`.
- Author: "create north+reserved as a new network layout, then plan the tuning analysis".
  - `configs/config.py`: `MultiTSOConfig.scenario` default `"rural_700_north_res"`.
  - `tuning/scenarios_mc_v2.py`: `MC_NETWORK = MultiTSOConfig.scenario` (was the literal
    `"rural_700"`; the tuning windows W1–W6 are built from it).
  - `network/ieee39/scenarios/rural_700_north_res.md`: the layout and its screen evidence.
  - Screen-only scenarios stay registered (reproducibility of runs 0001–0004).

## 10. Closed-loop smoke on NR (e5power run 0004 `smoke_variants_north_res`, 60 min, +250 Mvar at bus 12)

Paramset as selected (not re-tuned). With a 60-min window the unit commitment keeps G3 off at
W1, W6 and the extreme OP MC20261155 (2016-05-27 13:00).
- **SVR / SVR+: calibration aborted** ("zone 2: pilot bus 7 is insensitive") — zone 2 had no
  machine and no TS IBG, so s_p = 0. Old network: W2 always existed.
  - Fix `controller/svr_controller.py`: `no_resources` (Σ headroom = 0 at calibration) → the
    SVR holds (no RVR update, no division by s_p); with STS access the couplers join s_p at
    the first capability message as before. An insensitive pilot *with* resources still raises.
    Test `test_zone_without_any_resource_holds_instead_of_raising`; SVR tests 41 passed.
- OFO+ and CEN: W1 ok (180/180, area-2 V_min 0.925 / 0.975). W6 OFO+ fails at step 99 (after the
  load step) and MC20261155 OFO+/CEN at step 1: `ControllerNotConverged`, 41 of 44 IBG Q(V)
  loops do not settle; at MC20261155 zone 2 starts at V ∈ [0.885, 0.981] with no TS resource and
  the TS-OFO commits all three MSCs. Near voltage-stability limit, not one bad loop.
- Comparison (runs 0005, 0006): `rural_700` cannot initialise MC20261155 at all (60-min window);
  `rural_700_north` (no reserved) fails OFO+ there like NR → reserved capability is not the cause.

## 11. Observation for the evaluation design (not the network)

- Base MC 0002: 36 of the first 130 drawn episodes have G3 decommitted, only 10 of them are in
  the 100-episode comparison window — 26 were replaced under rule (b) because some variant
  failed. NR's G3-off episodes are largely the same seeds (21 of the first 130).
- NR's scarcity is ≈ entirely in G3-off hours (S1 with G3 on 0.6 %). Under rule (b) those
  episodes tend to leave the comparison window, so e_v differences would again be measured
  mostly where area 2 is not scarce. Failure counts of MC 0002 already differ: OFO+ 3,
  OFO 14, SVR 16, SVR+ 9, LOC 3, CEN 1 (excl. the 6 infeasible).
- Running: e5power run 0007 `mini_mc_north_res` — LOC, OFO, OFO+, SVR+ on NR for the 21
  G3-off and 9 G3-on seeds among the first 130 (own event catalogue of NR: same starts and
  times, events can differ from MC 0002 → distributional comparison, not paired).

## 12. H3 tail with the Q(V) loops acting (local run 0004 `screen_network_loops_tail`)

N-1 on the 48 lowest-voltage OPs of B and NR, solved as the runner does after a disturbance
(relays held, IBG Q(V) loops and slack limit iterated) instead of frozen droops:

| | frozen p5 / min | loops p5 / min | below 0.90 pu, frozen → loops |
|---|---|---|---|
| B | 0.877 / 0.856 | 0.911 / 0.906 | 11 → 0 of 48 |
| NR | 0.838 / 0.817 | 0.903 / 0.865 | 20 → 2 of 48 |

Supports the author's acceptance of H3 (additional information; the declared rule used frozen).

## 13. Full test suite on the new default

1056 passed, 19 skipped, 3 failed (1 h 44 min, e5wimi).
- `test_public_registry_contains_only_capacity_named_scenarios`: pinned the registry to
  {base_410, rural_700}. Updated to the seven registered names (all capacity-prefixed, each with a
  capacity entry); new `test_config_default_is_a_registered_scenario`. 26 passed.
- `test_ch9_procedure_v2::test_engagement_bounds`, `::test_the_largest_rho_within_the_margins_is_chosen`:
  **pre-existing test-order dependence**, unrelated. `ch_9_11_procedure_v3` overwrites globals of
  the v2 module at import (`V2.VERIFY_P95`, `V2.stage_select`, ...). Reproduced on the unchanged
  base `F:\qOFO_res` (3cdb0e3): `pytest tests/test_procedure_v4.py tests/experiments/test_ch9_procedure_v2.py`
  → 3 failed; alone they pass in both checkouts. Not fixed here.

## 14. Mini-MC on NR (e5power run 0007, pulled to `_remote_e5power/0007_2026-11-08_160221_mini_mc_north_res`)

`run_mc.variant_task` on NR for the 21 seeds among the first 130 MC draws where NR's commitment
keeps G3 off, plus 9 G3-on seeds; variants LOC, OFO, OFO+, SVR+; paramset as selected (not
re-tuned). Events drawn from NR's own element catalogue (same starts and times as MC 0002,
elements can differ) → distributional comparison with MC 0002 on the same seeds. 4 of the 30
seeds were "infeasible" (all six variants fail) in MC 0002.

Non-ok runs:

| seeds | network | LOC | OFO | OFO+ | SVR+ |
|---|---|---|---|---|---|
| G3 off in NR (21; 20 in MC 0002) | base (MC 0002) | 7 | 14 | 7 | 10 |
| | NR | 2 | 6 | 1 | 3 |
| G3 on (9) | base | 0 | 2 | 0 | 1 |
| | NR | 0 | 0 | 0 | 0 |

OFO+ − OFO, area-2 e_v, paired where both complete:
- NR, G3 off: median −2.91 mpu, mean −7.10 (n = 14), OFO median 9.27 mpu;
- NR, G3 on: −0.78 (n = 9);
- base, G3 off −0.16 (n = 6); base, G3 on −0.39 (n = 7).

Reading: NR is more robust than the base in exactly the scarce episodes (fewer failures for every
variant), and the STS benefit there is an order of magnitude larger. OFO's failures are
oscillations (`fail_osc`). Small sample, old weights.

## 15. Committed and handed off (author ~18:10)

- Author asked whether a 200-300 MW park at W2's old bus would push G3 out more often:
  - estimate with the runner's UC rule over the year: G3 off in 16.5 % of 180-min windows
    → 19.6 / 20.1 / 20.9 % for 200 / 250 / 300 MW;
  - G3 is off only in windy hours, so the park would have its full Q in every G3-off window
    and remove the zero-resource state that carries the STS benefit;
  - author: "keep it like this".
- Commit **62c4944** on `res/network-north-2026-11-08` (fast-forward from
  res/commitment-base-2026-10-31; a trial merge onto minimal/thesis-core is conflict-free; not
  merged). The e5power worktree `F:\qOFO_runs\qOFO_net` was moved to the same commit.
- Brief sent to the session "STS capabilities tuning analysis" (delivered 2026-11-08).

## 16. Open (author)

1. Adopt reserved capability for the coupling parks (`_res`)?
2. Network lever for area 2: north layout, 9–39 not built, neither — after the full screen.
3. Tie-Q as a controlled output of the TS objective (design change; would be tuned in the new
   campaign). With it, an area's own reserve is the binding scarcity measure.
4. The "STS capabilities tuning analysis" session waits for the network decision.
