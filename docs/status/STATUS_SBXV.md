# STATUS — SBX-V (Vertical Band-and-Request Coordination, TSO–DSO)

Build plan: `SBXV_TSO_DSO_Coordination_Build_Plan.md` v1.0 (2026-07-08).
Status file per hard rule 6. Dated entries, newest phase at the top of each section.

## Phase overview

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo survey, `<report>` values, DP check | **done** (2026-07-09) |
| 1 | Band + MIQP cost layer | **done** (2026-07-09) — R1 at the solver seam; closed-loop R1 re-run due at Phase-4 wiring |
| 2 | Metering + settlement | **done** (2026-07-09) — all Fall-Kategorien + worked example encoded from the Leitfaden text (R4 established) |
| 3 | Request pipeline + grants ledger | **done** (2026-07-09) — deterministic replay, ledger invariants, PARTIAL/REJECT paths, codified substitute all green |
| 4 | Commit-instant integration (+ emergency) | **done at mechanism level** (2026-07-09) — R3 green; closed-loop DSO tracking invariant deferred to the Phase-5 runner wiring (§4.3) |
| 5 | Experiments E1–E4 | **in progress** (2026-07-09) — runner wiring done, closed-loop R1 green, E1 script running; E2–E4 open |
| 6 | Reporting (optional) | open |

---

## Phase 5 — Runner wiring + experiments (2026-07-09, in progress)

### 5.1 Runner integration (`coordination_mode="sbxv"`)

New module `sbxv/adapter.py` (`SBXVRunnerAdapter`, mirrors the `sbx.adapter` pattern); runner
changes in `experiments/runners/multi_tso_dso.py` — all strictly guarded by
`coordination_mode == "sbxv"` and inert otherwise:

1. mode whitelist + validation block (SBXVConfig instance/`tso_period_s` match);
2. adapter construction BEFORE the main loop (metering and the `PricingSolver` proxies must
   exist from t = 0); exposed to experiments via the `pre_loop_hook` state as `sbxv_runtime`
   (the SBX-H capture pattern);
3. `before_solve` hook (TSO branch, before `coordinator.step`): need trackers from zone
   measurements → request pipeline; arms the spec provider for the current iteration;
4. `after_solve` hook (right after `coordinator.step`): captures the dispatched netted PCC-Q
   reference per AggregationArea (the logged Abruf) into per-window accumulators;
5. plant-step metering hook (end of each step): records `[t−dt, t)` with the post-power-flow
   per-NVP boundary Q.

AggregationAreas are derived from the zone controllers' configs (`pcc_dso_controller_ids` —
one area per DSO, netting its interface transformers; PCC output rows = `n_v + position`).
`MultiTSOConfig` gains `sbxv_config` (typed loosely, mirroring `sbx_config`).
`adapter.finalise()` assembles observations (metered windows × captured reference means),
filters grants beyond the metered horizon (loud, listed in `dropped_grants`), and runs the
Phase-2 `SettlementEngine`.

**Feedforward decision (documented):** the v1 wiring applies NO setpoint offset — unlike an
MSR/MSC switch there is no plant-side jump the DSO could counteract; the reference moves
through the priced MIQP in micro-steps. The scheduled-envelope lead remains available from
`CommitScheduler`; E1's commit-instant tracking metric decides whether it is ever needed.

### 5.2 Closed-loop R1 — GREEN (2026-07-09)

`tests/sbxv/smoke_sbxv_closed_loop.py` (smoke, not pytest-collected): 45-min horizon on the
shared 005 scenario, arms `none` vs `sbxv` with an unreachable need flag → **1263 recorded
control arrays byte-identical over 135 steps** (Q_PCC set, DER Q, V_gen, OLTC taps, losses).
The settlement plane ran alongside (12 windows metered); notably the BASELINE already produces
`8.2-4b_adhoc` windows — some netted interfaces operate beyond the default ±50 Mvar band with
a commanded reference beyond the edge, so the ad-hoc Grenzpreis reporting is live even without
grants. First E1 economics will quantify this.

### 5.5 E2 — band-width Pareto complete (2026-07-09)

`experiments/019_SBXV_E2.py`, 3 MC seeds (012/006 harness: random profile-year start +
random contingency schedule) × 7 arms × 120 min; all 21 cells green. Outputs:
`results/019_SBXV_E2/e2_sweep.csv`, `e2_pareto.png`, per-cell JSON journals.

**Findings (means over seeds; payments per 2 h at placeholder prices):**

| Band | Payments [€] | No-grant exceedance [Mvarh] | Longest run [windows] |
|---|---|---|---|
| ±0 | 14 468 | 198.2 | 8.0 (= full horizon) |
| ±25 | 5 861 | 49.3 | 5.7 |
| ±50 | 924 | 6.2 | 1.3 |
| ±75 | 157 | 0.6 | 0.7 |
| ±100 | 0 | 0 | 0 |
| `ar41414_default` | 930 | 6.4 | 1.0 |

1. **Clean Pareto front:** payments and exceedance fall steeply and monotonically with band
   width; the knee sits between ±25 and ±50 Mvar for this scenario family. TS voltage
   quality and reserve margin are IDENTICAL across all arms and the baseline — the
   commercial layer redistributes cost without degrading physics (V-D1 separation confirmed
   in closed loop).
2. **Persistent-exceedance indicator works as intended** [LF Präambel]: at ±0 the exceedance
   is persistent over the full horizon (8 consecutive windows); at the knee it collapses to
   isolated single windows.
3. **`ar41414_default` ran** (spread assertion passed; contracted P from Σ sn_hv_mva) and
   prices out economically ≈ ±50 symmetric — but it is scenario-sensitive: clean in seed 1,
   4 exceedance windows (2 622 €) in seed 2. Band sizing is a quantile question → E4.
4. **Grant pipeline dormant in all 21 cells** (0 requests): even the contingency draws never
   push transmission voltages beyond the band, so condition B (a REAL voltage violation)
   never fires — with `g_v = 1e7` the zones hold voltage by paying the Grenzpreis instead
   (the ±0 arm demonstrates this: 14.5 k€ rather than any violation). Exercising
   request→grant→settlement in closed loop is E3's job (targeted largest-in-feed trip near
   an interface).

### 5.7 V-D2 revision — operative Normalbereich = AR 4141-4 preset (Manuel, 2026-07-10)

The operative band is now the E VDE-AR-N 4141-4 default: **5 % raising / 10 % lowering of
the contracted P** per AggregationArea [AR §5.2.1] (`SBXVConfig.band_preset` default flipped
to `"ar41414_default"`; the fixed ±50 box remains available as `"fixed"` and is what the E2
sweep arms explicitly use). Contracted P is derived from the rated interface capacity
(Σ `sn_hv_mva` of the area's coupling transformers); the Anhang C spread assertion (≥ 70
Mvar) applies per area and fails fast. The runner's `[sbxv]` startup print and the E1
summary now report the RESOLVED per-area band edges. Fits the §5.6 narrative: the preset IS
the standard's own simplified planning product — the band comes from the connection
agreement, no ORPF needed. Unit suite 102 green; E1 re-running with the preset band.

### 5.6 Scoping decision — ungesichert-centred thesis narrative (Manuel, 2026-07-10)

The thesis focuses on the CONTROL CONCEPT; the market/planning side (an ORPF-based forecast
that would determine Vorhalteleistung need in advance) is out of scope. It is therefore
ACCEPTED that the closed-loop demonstrations settle essentially in the ungesichert tier:

- The demonstrated vertical mechanism is: ex-ante band (Normalbereich) + no-gating priced
  dispatch on posted potential (V-D4) + Leitfaden-exact settlement + the persistent-
  exceedance metric as the feedback signal INTO planning [LF Präambel] — which is precisely
  the regulatory reading (Vorhalteleistung is a planning product; its absence at operation
  time routes the exchange through the Grenzpreis tier).
- The request→grant pipeline (Phases 3/4) stays implemented and mechanism-tested; the thesis
  presents it as the codified process whose activation presupposes the (out-of-scope)
  planning plane. No closed-loop grant demonstration is required.
- Vertical analogue of the SBX-H planning pre-run (017_SBX_PLANNING determines the standard
  tie-line flows q_std): the BAND is the vertical planning product. E4's Anhang-C quantile
  sizing over the MC draws plays exactly the role of the simplified pre-run — it anchors the
  Normalbereich without building an ORPF.
- Consequence for E3: re-scoped to its control-concept core (already in the plan text):
  time-to-band-recovery after a targeted contingency and WHAT THE UNGESICHERT SEGMENT
  ABSORBS — not grant activation for its own sake. Optional.

### 5.4 V-D9 conformance fix (2026-07-09, before E2)

Building the E2 sweep exposed a conformance gap in the Phase-4 spec schedule:
`CommitScheduler.specs_for` returned `None` whenever no grant was active anywhere — but V-D9
prices beyond-band dispatch at the Grenzpreis EVEN WITHOUT a grant ("0 in band, Grenzpreis
beyond, when no grant is active"). As wired, the dispatch could exceed the band unpriced
(settlement charged it, the MIQP never felt it — E1 finding 4 showed exactly this).

**Fix:** the band prices act from window 0 onward; the neutral/R1 bypass is now an EXPLICIT
configuration switch `SBXVConfig.miqp_pricing_enabled = False` (metering, need flags,
pipeline and settlement still run — reporting only). R1 smoke re-run after the fix: still
1263 arrays byte-identical (the neutral arm sets the switch). Unit suite 101 green. A 30-min
stressed probe (E2 seed 1, band ±50) confirms the steering: the no-grant exceedance drops to
1.7 Mvarh — the Grenzpreis pulls the dispatch back to the band edge where the unpriced E1 run
drifted freely. **Consequence: the E1 numbers of §5.3 predate this fix; E1 is re-run after
the E2 campaign.**

Also added for E2: adapter support for the `ar41414_default` preset (contracted P derived
from the rated interface capacity Σ `sn_hv_mva` of the area's coupling transformers; the
AR Anhang C spread assertion applies per area and a rejection is recorded as a finding).

### 5.3b E1 — with the AR 4141-4 preset band (2026-07-10, current; V-D2 rev. §5.7)

Resolved Normalbereich (contracted P = 900 MW per area from Σ `sn_hv_mva` = 3 × 300 MVA):
**raising 45 / lowering 90 Mvar** for every DSO area (spread 135 Mvar ≥ 70 ✓). R1
re-confirmed (3363 arrays); physics identical across arms (violation 0, reserve 0.304,
commit-instant tracking 1.69 < 1.85 Mvar). Payments **1 103.12 €** over 2 ad-hoc windows —
slightly MORE than under ±50 (1 021.82 €): the preset is asymmetric and its RAISING edge
(5 % = 45 Mvar) is tighter than ±50, and the scenario's exceedance at the affected interface
is raising-side (the DS injects > 45 Mvar netted). The 5 %-raising side is the binding edge
of the standard's default band in this system. `results/018_SBXV_E1/` holds these outputs.

### 5.3a E1 — post-fix re-run (2026-07-10, superseded by §5.3b)

Same three arms, 120 min, AFTER the §5.4 V-D9 fix: R1 re-confirmed (3363 arrays
byte-identical); violation energy 0 and reserve 0.304 across all arms; DSO tracking
1.85 Mvar mean, commit-instant subset 1.69 Mvar (invariant holds — no feedforward lead
needed). Active arm payments 1 021.82 € (vs 1 042.23 € pre-fix): the always-on Grenzpreis
pulls the dispatch marginally toward the band edge in the 2 ad-hoc windows; the voltage
objective dominates as intended (V-D4 economics). Still 0 requests — condition B never
fires in the benign scenario (see E2 finding 4). `results/018_SBXV_E1/` overwritten with
the post-fix outputs.

### 5.3 E1 — first run complete (120-min calibration horizon, 2026-07-09; PRE-FIX, see §5.4)

`experiments/018_SBXV_E1.py` — arms `none` / `sbxv_neutral` (R1 re-check inside the
experiment) / `sbxv` (active v1 defaults). Outputs: `results/018_SBXV_E1/e1_{windows,days,
totals}.csv` + `e1_summary.json`.

**Findings (V1 defaults, placeholder prices, 120 min):**

| Metric | none | sbxv_neutral | sbxv |
|---|---|---|---|
| TS violation energy | 0 pu·s (0 s) | 0 | 0 |
| min DER reserve margin | 0.304 | 0.304 | 0.304 |
| DSO tracking error, mean (all / commit instants) | 1.85 / 1.70 Mvar | identical | identical |
| Requests / grants | — | 0 / 0 | 0 / 0 |
| Payments | — | — | 1042.23 € (2 × `8.2-4b_adhoc`, 30 × `free_band` of 32 windows) |

1. **R1 re-confirmed at 120 min:** 3363 recorded control arrays byte-identical
   (`none` vs `sbxv_neutral`).
2. **Parity of the active arm:** the 005 profile keeps transmission voltages inside the band
   over this horizon, so condition B never fires, no request is emitted, and the active sbxv
   dispatch equals the baseline — the mechanism is provably non-invasive when unneeded.
3. **DSO tracking invariant at commit instants (Phase-4 deferred item): HOLDS** — the
   commit-instant error subset (1.70 Mvar mean) is *below* the all-steps mean (1.85 Mvar);
   no commit-instant transient exists, confirming the §5.1 decision to apply no feedforward
   lead in v1.
4. **Planning-deficit indicator live:** 2 of 32 windows settle as ad-hoc Grenzpreis windows
   (references beyond the ±50 Mvar default band without a grant) at 1042 € per 2 h — the
   [LF Präambel] persistent-exceedance economics are visible even in the benign scenario.
5. **Consequence for E2/E3:** exercising the grant pipeline in closed loop needs either the
   tighter band arms of the E2 Pareto sweep or the E3 contingency (condition B requires a
   real transmission-voltage violation). This mirrors the SBX-H finding F1 (mechanism pays
   under stress, idles otherwise).

E2 (band Pareto), E3 (contingency latency), E4 (Anhang-C sizing) open; the full 360-min
case-study E1 run is a one-command re-run (`python experiments/018_SBXV_E1.py 360`).

---

## Phase 4 — Commit-instant integration + emergency (2026-07-09)

New modules: `sbxv/commit.py`, `sbxv/emergency.py`; config validators for the Phase-4 knobs
(`n_clear`, `sat_tol_mvar`, `v_dev_threshold_pu`, `reserve_margin_mvar` — fields added by the
parallel session, validation completed here); tests `tests/sbxv/test_commit_phase4.py`
(21 tests). Full regression run 186 green (SBX-V 100 + SBX-H/solver 86, hard rule 10; R4
re-ran within the SBX-V suite). No existing module was modified.

### 4.1 What was implemented

- **`CommitScheduler`** (`sbxv/commit.py`) — the plan-§4 scheduling plane tied together:
  - per-iteration `step()`: need-tracker updates from `AreaIterationInput` (netted PCC flow,
    per-direction voltage deviations), flag → `RequestPipeline.run_window` for the NEXT window
    (one request per (area, direction, window); re-issue in later windows) — **activation stays
    at the boundary**;
  - `specs_for(k)`: the priced segment structure as a pure function of the iteration index and
    of ledger state frozen at scheduled instants. No grants anywhere ⇒ `None` ⇒ the
    `PricingSolver` neutral bypass (R1 tie-in). Within a stretch the SAME tuple object is
    returned (R3 identity is testable with `is`);
  - **expiry ramp** (plan §4 item 4): a grant ending at the next boundary with no confirmed
    follow-up shrinks linearly over the final `ramp_steps` iterations
    (`grant · (k_end − 1 − k)/ramp_steps`, reaching 0 at the last in-window iteration). The
    ramp decision is evaluated ONCE at the ramp start and frozen — a follow-up confirmed after
    the ramp start does not cancel the scheduled ramp (determinism; unit-tested);
  - **scheduled-envelope feedforward** (MSR/MSC pattern reuse): `scheduled_envelope_mvar`
    (band edge + effective grant per direction) and `envelope_step_mvar` (signed netted
    per-iteration change) — nonzero exactly at commit instants (+grant on activation) and
    during ramps (−grant/ramp_steps per step), zero elsewhere; the runner applies it as the
    synchronised lead on interface setpoints, mirroring the `q_itf_sh_offset` pattern;
  - **Incapability path**: `declare_incapability` (rep1 without an active grant; loud
    Reserve-Observer log; provided remainder = granted − shortfall) and
    `to_incapability_records`: TSO-delivers grants yield settlement `IncapabilityRecord`s
    (Tabelle 8.1 case 3a — end-to-end test through the Phase-2 engine produces `8.1-3a` with
    the pro-rata capacity fraction); DSO-delivers declarations stay logged events (settlement
    detects the under-delivery from metering, Tabelle 8.2 case 2, per §2.2).
- **`EmergencyHandler`** (`sbxv/emergency.py`, flag-gated per plan): any `EmergencyCall` with
  `emergency_call_enabled = False` fails fast; enabled calls register an immediate best-effort
  extension until the window end, loudly logged, event-scripted only. **Design consequence of
  the open-tail pricing (§1.2):** everything beyond band + grant is already Grenzpreis-priced,
  so a Notfall-Abruf changes consent bookkeeping and logging, NOT the spec — R3 is preserved by
  construction (unit-tested: an emergency call does not rebuild the spec tuple).

### 4.2 Phase-4 acceptance status

| Acceptance item | Status |
|---|---|
| R3 — MIQP constraint/price set changes only at commit instants | green at the mechanism level: spec-tuple identity within stretches; changes exactly at window boundaries and the pre-scheduled ramp iterations; emergency never rebuilds |
| Grant activation/expiry at window boundaries only | green (`test_grant_activates_only_at_the_boundary`, ramp reaches 0 at the boundary) |
| DSO feedforward reuse | scheduled-envelope API implemented and timing-tested; closed-loop application is the Phase-5 wiring |
| DSO tracking error invariant at commit instants (MSR test pattern) | **deferred — closed-loop property** (see §4.3) |
| Incapability event propagates to settlement case 3a | green end-to-end (declaration → record → engine → `8.1-3a`, pro-rata) |
| `emergency.py` behind `emergency_call_enabled` | green (fail-fast when disabled; activation window and log when enabled) |

### 4.3 Open Phase-4 item (moves into the Phase-5 wiring)

The DSO tracking-error invariant at commit instants and the closed-loop R1/R2 re-runs require
the runner integration (`coordination_mode="sbxv"` in `experiments/runners/multi_tso_dso.py`:
install `PricingSolver` proxies on the TSO controllers, drive `CommitScheduler` per TSO
iteration, meter from plant measurements, apply the envelope feedforward to `SetpointMessage`s,
export settlement CSVs). This is deliberately bundled with Phase 5/E1 — it is the same wiring
E1 needs, and doing it once avoids touching the 4000-line runner twice. The mechanism-level R3
guarantees transfer directly: the runner consumes `specs_for(k)` and cannot introduce
mid-window spec changes.

---

## Phase 3 — Request pipeline + grants ledger (2026-07-09)

New modules: `sbxv/messages.py`, `sbxv/potentials.py`, `sbxv/need_flag.py`,
`sbxv/feasibility.py`, `sbxv/grants_ledger.py`, `sbxv/pipeline.py`; tests
`tests/sbxv/test_pipeline_phase3.py` (9 tests; SBX-V total 42, full sbx+sbxv run 142 green).
No existing module was modified (hard rule 5) — Phase 3 is purely additive, so the
CAIR/SBX-H/BME/MSR suites are untouched by construction (hard rule 10).

### 3.1 What was implemented

- **Messages (plan §3):** frozen dataclasses `Window`, `PotentialMessage`, `ReserveRequest`
  (deterministic `request_id = req:<area>:<direction>:w<window>`; `all_or_nothing` as the single
  v1 condition), `FeasibilityReply` (ACCEPT/PARTIAL/REJECT), `BindingOrder`
  (`order_id = ord:<request_id>`), `GrantConfirmation`, `IncapabilityDeclaration` (consumed by
  the Phase-2 settlement via `IncapabilityRecord`; wired in Phase 4).
- **Potentials (DP1 wrapper):** `build_potential_message` wraps the operative
  `core.message.CapabilityMessage` untouched — absolute netted box = Σ_NVP q_meas + [Σq_min,
  Σq_max] (DP5 netting), direction split per DP3 (`LOWERING` potential = reachable positive
  netted extreme, `RAISING` = reachable negative extreme, both as magnitudes), gesichert share
  `q_vh_flagged = min(granted, pot)` [AR §6.4.3], `is_forecast` for the day-ahead plane
  [AR §6.3]. **The one codified fail-fast exception (hard rule 1):** `substitute_potential`
  replaces a MISSING message by *potential := 0 beyond the band* (q_pot = band edge)
  [AR §6.3.2 Schritt 2] — explicit `is_substitute=True` flag and a loud `logger.warning`.
- **Need flag (plan §6, SBX-H v2 philosophy — no shadow prices):** `VerticalNeedTracker` with
  Condition A (netted PCC dispatch saturated at the free-or-granted segment edge, DP3 sign,
  `sat_tol_mvar`), Condition B (persistent transmission-bus deviation beyond the Sollspannungs
  band in the corresponding direction), flag = A ∧ B after `n_persist` consecutive TSO
  iterations; clearing hysteresis (BOTH conditions clear for `n_clear` consecutive iterations);
  iteration-gap reset (the `sbx.need.NeedTracker` pattern). `size_request_quanta`: smallest
  covering n, capped by the day-ahead posted potential beyond band + existing grants (0 ⇒ the
  request is not emitted, logged as `no_headroom`).
- **Feasibility (Machbarkeitsprüfung [LF §6.4]):** headroom = posted potential − band edge −
  already granted − Reserve-Observer margin; ACCEPT/PARTIAL/REJECT with `all_or_nothing`
  partial → REJECT; every reply is recorded in the ledger.
- **Grants ledger:** append-only; `confirm` fail-fasts (rep1) on any confirmation without a
  recorded feasibility reply or exceeding the last accepted offer; `granted_mvar` is
  window-boundary-exact (adjacent windows see nothing); `assert_invariants` sweep per window;
  `to_grant_records` exports Phase-2 `GrantRecord`s (v1 delivering party = DSO).
- **Pipeline (plan §4 steps 1–3):** `RequestPipeline.run_window` — sorted deterministic
  iteration over (area, direction), one request per (area, direction, window) with re-issue in
  later windows [LF §6.9 spirit], request → feasibility → binding order → confirmation in one
  pass (no negotiation loop), fail-fast on a `None` forecast (the substitute is the potentials
  plane's job), primitive-tuple event log for replay comparison, ledger invariant sweep at
  window close.

### 3.2 Phase-3 acceptance status

| Acceptance item | Status |
|---|---|
| Deterministic replay (same inputs ⇒ identical request/grant log) | green — two scenario drives produce identical `pipeline.log` and ledger reprs |
| Ledger invariants | green — sum consistency, feasibility cap (over-confirmation rep1s), window-boundary activation |
| No grant exceeds the feasibility answer | green — asserted in `confirm` and in `assert_invariants`; end-to-end in the pipeline test |
| PARTIAL and REJECT paths exercised | green — PARTIAL via the reserve-margin asymmetry (sizing ignores the margin, feasibility subtracts it); REJECT via zero headroom and via `all_or_nothing` partial |
| Missing-message substitute loud + flagged | green — `caplog` asserts the warning; `is_substitute=True`; wrapper applies it for `capability=None` |

### 3.3 Design notes

- **Sizing vs feasibility asymmetry (intentional):** request sizing caps at the *day-ahead
  posted* potential beyond band + grants but does not know the Reserve-Observer margin (TSO-side
  information); the feasibility answer subtracts it (DSO-side). PARTIAL therefore occurs
  naturally when the margin binds — this is the mechanism the end-to-end test exercises.
- **Config wiring deferred to Phase 4:** the new knobs (`sat_tol_mvar`, `v_dev_threshold_pu`,
  `n_clear`, `reserve_margin_mvar`) are constructor parameters of the Phase-3 objects; they move
  into `SBXVConfig` when the commit-instant integration wires the trackers to the runner.
- **Transport-free (plan §12):** the pipeline holds both roles in-process, but every step passes
  through the §3 message dataclasses so the information accounting (who could know what) stays
  explicit.

---

## Phase 2 — Metering + settlement (2026-07-09)

New modules: `sbxv/metering.py`, `sbxv/settlement.py`; tests
`tests/sbxv/test_metering_settlement.py` (33 tests, all green; SBX-V total 70). Both regulatory
documents were read and the settlement rules were implemented from the Leitfaden text itself
(`docs/regulatory/guideline_q.pdf`), not from the plan's summary.

### 2.1 What was implemented

- **Metering** ([LF §5.5, §8.3]): four-quadrant register model per NVP — the signed `q_hv` is
  split into an `(Q1+Q2)` register (positive, LOWERING energy) and an `(Q3+Q4)` register
  (negative, RAISING energy) per 15-min window; Saldierung per AggregationArea = register-wise
  addition across NVPs, work → power conversion, signed mean `Q = (Q1+Q2) − (Q3+Q4)`. The
  numeric example of [LF Abb. 8.3] (4 Zählpunkte × 6 quarter-hours ⇒ means 16, −16, −20, 4, −8,
  −4) is encoded verbatim as a unit test. Interval recording is fail-fast: contiguity, no
  window-straddling, full coverage at close; a partial tail window is reported, never settled.
- **Settlement** ([LF §7], Tabellen 8.1/8.2, V-D8 exact):
  - Tabelle 8.2 (DSO delivers, primary): opposite-edge reference for Vorhalteleistung AND
    Blindarbeit [LF §7.2 case 1]; cases 1 (correct), 2 (under-delivery → Leistungspreis
    suspended fully/pro rata, energy still paid), 3 (over-delivery → energy capped at the
    Sollwert), 4 (call beyond grant → Durchschnitt within, Grenzpreis beyond; capacity
    exceedance = highest call [LF §7.5.3]). Worked example [LF §8.2] (band ±50, grant 100 ⇒
    VH = 200 Mvar) is a unit test.
  - Tabelle 8.1 (TSO delivers, reverse): same-side reference [LF §7.2 case 2], in-band energy
    deduction [LF §7.1]; cases 1, 2, 3, 3a (capacity suspension via `IncapabilityRecord` —
    Phase 3 wires the `IncapabilityDeclaration` message to it), 4, 4a.
  - Tolerance [LF §4.7]: ±10 % of the VH magnitude on the 15-min mean (boundary unit-tested).
  - Daily Leistungspreis accrual [LF §7.4]: per day of Vorhaltung; the day's worst window
    capacity fraction governs (full or pro-rata suspension, [LF §7.5.4]).
  - Tidy CSV output (`<prefix>_windows/days/totals.csv`), schema documented in the module
    docstring; `payer` column (requester pays deliverer).

### 2.2 Documented interpretations (beyond the Leitfaden letter)

- **Ad-hoc downstream delivery (`8.2-4b_adhoc`)**: Tabelle 8.2 has no no-grant row, but SBX-V's
  V-D4 (no gating; the CAIR posting doubles as Potenzialmeldung whose posting implies consent
  [LF §6]) makes grant-free calls beyond the band a normal closed-loop event. Treatment mirrors
  the Tabelle 8.1 case-4 construction with the OWN band edge as reference: energy and
  day-maximum CALLED exceedance at Grenzpreisen. A measured exceedance WITHOUT a corresponding
  logged reference is classified Tabelle 8.1 case 4 instead (the DSO exceeded on its own; the
  upstream delivered).
- **Re-issued grants on one day**: sequential grants (plan §6 re-issue pattern) accrue ONE day
  of Vorhaltung — largest held magnitude, worst-window suspension fraction.
- **Sub-day grants**: a day touched by an active grant accrues the full daily Leistungspreis
  (scenarios are sub-day; one Verrechnungsperiode per scenario, plan §2).

### 2.3 Phase-2 acceptance status

| Acceptance item | Status |
|---|---|
| All Fall-Kategorien of Tabellen 8.1/8.2 unit-tested | green — 8.1: 1, 2, 3, 3a (two variants), 4, 4a; 8.2: 1 (two variants), 2 (two variants), 3, 4; plus ad-hoc interpretation |
| 200 Mvar worked example verbatim | green (`TestWorkedExample`) |
| Tolerance and suspension logic covered | green (boundary test at ±10 % · VH; pro-rata and full suspension; worst-window day rule) |
| CSV output schema documented | module docstring of `sbxv/settlement.py`; round-trip test |
| R4 (settlement = Leitfaden tables) established | to be re-run every later phase |

---

## Phase 1 — Band + MIQP cost layer (2026-07-09)

New modules: `sbxv/__init__.py`, `sbxv/config.py`, `sbxv/directions.py`, `sbxv/band.py`,
`sbxv/miqp_cost.py`; tests `tests/sbxv/` (37 tests, all green). No existing module was modified
(hard rule 5); existing SBX-H + solver suites re-run green (86 tests).

### 1.1 Integration design (documented decision)

The controllers call `self.solver.solve(problem)` with a fully built `MIQPProblem`
([base_controller.py:581](controller/base_controller.py), `:720` for the frozen-integer companion
solve). SBX-V therefore wraps the **solver instance** from outside
(`sbxv.miqp_cost.PricingSolver`), installed at wiring time via
`controller.solver = PricingSolver(controller.solver, spec_provider, g_z_tier=…)`:

- **Neutral path (R1):** when the spec provider returns nothing, the problem passes through
  UNTOUCHED — one inner solve on the identical problem object. Byte-identity is by construction;
  tested at the seam (`test_bypass_is_byte_identical`). The full closed-loop R1 against the CAIR
  baseline on the 005 scenario is re-run when the runner wiring lands (Phase 4).
- **Priced path:** the problem is augmented per (AggregationArea, direction) with one tier
  boundary output row on the NETTED signed q (DP5) and one continuous variable per cost segment
  (pure linear objective, exact convex piecewise-linear pricing per §5), solved, and the result
  stripped back to the original shape before the controller sees it. The per-solve
  `TierDecomposition` (plan §5 reconstruction, asserted) is retained on the proxy for
  logging/metering.

### 1.2 Design notes / plan clarifications

- **Open tail (V-D1 dominance):** plan §5 bounds `q_ug` by the posted-potential headroom. A
  bounded final segment would make the tier row a hard constraint beyond the potential — i.e.
  tighten feasibility, contradicting V-D1 ("postings change prices, never feasibility"; the
  capability box, in practice a *soft* output bound with `g_z_q_pcc = 1e-2`, stays the only
  physical constraint). Resolution: every side ends with an open tail at the Grenzpreis slope
  (no higher tier exists); the posted-potential cap governs attribution/settlement, not
  feasibility. Asserted in `SideSpec.__post_init__`.
- **Tier-row slack:** the solver's shared-slack encoding penalises every output row
  quadratically; an exactly hard tier row is impossible without touching the solver. With slack
  weight `g_z_tier` (config, default 1e4) the incentive distortion is bounded by
  `slope/(2·g_z_tier)` ≈ 4e-4 Mvar at default prices; the reconstruction assert includes the
  observed slack and `rep1`s beyond the crossover bound.
- **V-D9 anchoring:** `build_side_spec` implements both incentive models. With no active grant
  they coincide (anchor at the own edge, Grenzpreis beyond). With a grant,
  `leitfaden_exact_when_granted` re-anchors to the OPPOSITE edge (constant cost offsets are
  irrelevant to the argmin, so anchoring encodes the exact Leitfaden marginal prices); the
  worked-example geometry (band ±50, grant 100 ⇒ 200 Mvar Durchschnittspreis span) is a unit
  test. Simultaneous grants in BOTH directions under the Leitfaden model would double-charge the
  band — undefined in the plan, `rep1` (report, do not improvise).
- **α-exactness:** auxiliary segment variables ride the solver's micro-step scaling
  (`x = α·w`, gradient entry `slope·α`); algebra verified in tests with `alpha = 1` (the runner's
  value).

### 1.3 Phase-1 acceptance status

| Acceptance item | Status |
|---|---|
| R1 neutral-config byte-identical dispatch | green at the solver seam (bypass); closed-loop rerun scheduled for Phase 4 wiring |
| Solver status asserted | `strip_result` preserves failures verbatim; controller raises (existing fail-fast path) |
| Reconstruction invariant tested | `strip_result` asserts §5 reconstruction on every priced solve; unit-tested incl. multi-NVP netting and integer blocks |
| Synthetic two-step preference test | `test_price_above_pull_pins_dispatch_at_the_edge` (stays in band), `test_price_below_pull_crosses_at_the_priced_optimum`, ordered tier filling, RAISING mirror case |

---

## Phase 0 — Repository survey (2026-07-09)

No code changes were made. All file:line references against the working tree at
commit `c64351c` (plus uncommitted SBX-H Phase-7 edits).

### 0.1 Reuse points

| # | Plan item | Location | Notes |
|---|---|---|---|
| 1 | `rep1()` fail-fast helper | `sbx/fail.py:29` (`rep1`), `sbx/fail.py:25` (`SBXError`) | Reuse via import (hard rule 5). Raises `SBXError(RuntimeError)` with sorted `key=value` diagnostics. Note the class name is SBX-generic, not SBX-H-specific; no wrapper needed. |
| 2 | Vertical CAIR message class | `core/message.py:178` (`CapabilityMessage`) | Fields: `source/target_controller_id`, `iteration`, `interface_transformer_indices`, `q_min_mvar`, `q_max_mvar`. **The bounds are DELTAS from the current operating point**, sign convention: pandapower load convention at the HV port (positive = Q flowing from the EHV bus into the transformer, i.e. into the DS). No window, no direction, no gesichert flag → **DP1 triggered** (see §0.4). |
| 3 | Capability box entry into the TSO MIQP | `controller/tso_controller.py:661` (`receive_capability`), stored per PCC at `:589–590` (wide default ±1e6 Mvar until first message) and `:694–695`; enters the MIQP as a **soft output bound on physical Q_PCC** in `_get_output_limits` (`:1706–1725`), anchored at the *current measured* interface Q (`q_iface_now + pcc_capability_min/max`), under `pcc_capability_on_output=True` (default, "Strategy D"); legacy hard input-bound path in `_compute_input_bounds` (`:1598–1608`). | The capability box is a **moving box re-anchored at the measured Q every TSO iteration**, not an absolute window box. The §5 decomposition must reconcile absolute band edges with this delta-anchored row (design note for Phase 1; interacts with DP1/DP2 but requires no CAIR change — the absolute box is `q_meas + [q_min, q_max]`, available to `sbxv/miqp_cost.py` from the same quantities the controller already holds). |
| 4 | SBX-H scheduler cadence | `sbx/scheduler.py` (`run_cycle` fires at iterations `c · k_sched`); `sbx/config.py:55` `k_sched = 2` (default), `sbx/config.py:59` `tso_period_s = 180.0` | As run in `experiments/013_SBX_LADDER.py:221–224` and `experiments/runners/multi_tso_dso.py:1163` the defaults are used → **SBX-H cycle = 2 × 180 s = 6 min**, not 15 min (the 15-min wording in the `sbx/config.py` module docstring is historical). → **DP4 triggered** (see §0.4). |
| 5 | SBX-H need-flag timers | `sbx/need.py:85` (`NeedTracker`): persistence counted in **consecutive TSO iterations** (`n_need`, default 1; direction change or iteration gap resets); release dwell `m_release` (default 1, counted in scheduler cycles, `sbx/scheduler.py` Step 5) | There is no wall-clock timer; `t_persist_s` maps to `n_need · tso_period_s`. Hysteresis-on-clearing pattern = `m_release` dwell. Reuse the `NeedTracker` consecutive-count pattern; SBX-V condition A (segment-edge saturation) needs a second tracker instance of the same shape. |
| 6 | SBX-H quantum constant | `sbx/config.py:64` `dq_quant_rate_mvar_per_15min = 30.0`; per-cycle quantum property `sbx/config.py:147` (`dq_quant_mvar = rate · t_cycle/15 min`) | Quantum-as-rate; for a 15-min SBX-V window this resolves to exactly 30 Mvar. |
| 7 | MSR/MSC commit-instant + feedforward hooks | `controller/shunt_integrator.py:162` (`ShuntCommit` record), `:256` (`ShuntBank.step`, dwell/budget/feasibility guards); runner atomic commit sequence `experiments/runners/multi_tso_dso.py:3885–3946`: (i) `apply_shunt_commit` on the plant net, (ii) DSO interface feedforward via `dso_ctrl.sensitivities.compute_dQtrafo3w_hv_dQ_shunt` accumulated into the persistent per-interface offset `q_itf_sh_offset` (`:3912–3915`), (iii) rank-1 SMW refresh of the TSO cached Jacobian (`:3918–3922`), (iv) `ShuntDisturbanceMessage` to the DSO (`:3924–3939`); the persistent offset is added to every outgoing `SetpointMessage` (`:3948–3970`) so the DSO does not counteract a committed switch. | This is the synchronised-feedforward pattern §4 prescribes for grant activation/expiry. For SBX-V the analogue is a step in the PCC-Q *reference*, not a physical switch, so only the offset-into-`SetpointMessage` half of the pattern applies; no SMW refresh is needed (no plant-side topology change). |
| 8 | Experiment-script naming | `experiments/` uses `NNN_TITLE.py` (`000`–`017`); shared runner `experiments/runners/multi_tso_dso.py`; shared scenario factory `experiments/005_CIGRE_MULTI.py` `make_cigre_config()` (IEEE 39 `wind_replace` TS, fixed 3-zone partition via `network/zone_partition.fixed_zone_partition_ieee39`, CIGRE HV DS networks at the interfaces; reused by 011/012/013 — this is the R2 anchor) | Plan §1.7 (`experiments/sbxv_…py`) adapted to repo convention: **`018_SBXV_….py`** (next free number). Package name `sbxv/` fits alongside `sbx/` unchanged. |
| 9 | TSO MIQP objective normalisation / units | `optimisation/miqp_solver.py:686` (`build_miqp_problem`): min `∇fᵀw + wᵀG_w w + zᵀG_z z` with `w = Δu/α` for continuous variables (`α = 1.0` in this runner, `experiments/runners/multi_tso_dso.py:1584`), `w = Δu` for integer variables; gradient assembled in `controller/tso_controller.py:2003` (`_compute_objective_gradient`) as weighted squared-error terms: `g_v = 1e7` per pu² (TS voltage tracking), `g_q = 200` per Mvar² (DSO Q tracking), `dso_g_v = 1e5` (`experiments/005_CIGRE_MULTI.py:128–133`); slack weight `g_z` per output; usage/change regularisation `g_u`, `g_w` per actuator class | The objective carries **no currency dimension** — all terms are heterogeneous weighted squared errors in pu²/Mvar² with empirically tuned weights. A linear €-term needs one explicit conversion constant → **DP2 triggered** (see §0.4). |
| 10 | Regulatory anchor documents | `docs/regulatory/` **does not exist**; no Leitfaden or E VDE-AR-N 4141-4 file found anywhere in the repo | → **BLOCKED entry B1** (see §0.5). |
| 11 | Boundary-Q sign convention at the PCC | Measurement point: `net.res_trafo3w.q_hv_mvar` at the EHV port (`experiments/runners/_multi_tso_helpers.py:406–411`, `:499–508`); `CapabilityMessage` docstring (`core/message.py:202–208`) fixes load convention: positive = Q from the EHV bus into the transformer (TS → DS) | Single documented convention exists. Proposed `directions.py` mapping (to be confirmed under DP3 before Phase 1): `RAISING` (spannungshebend/übererregt — the DS delivers Q into the TS) ↔ **negative** netted `q_hv_mvar`; `LOWERING` (spannungssenkend/untererregt — the DS absorbs Q) ↔ **positive** netted `q_hv_mvar`. |
| 12 | AggregationArea / multi-NVP netting | Each DSO area has **3 interface transformers** (`experiments/runners/_multi_tso_helpers.py:358` "3 per HV sub-network"); netting precedent (sum of `q_hv_mvar` over a PCC group) at `:560–567` | Netting is non-trivial in this scenario → **DP5 triggered** (see §0.4). |

### 0.2 Resolved `<report>` configuration values

| Key | Resolved value | Source |
|---|---|---|
| `window_s` | 900 (plan-fixed) = **5 TSO iterations** at the shared scenario's `tso_period_s = 180 s` | `experiments/005_CIGRE_MULTI.py:125` |
| `dq_grant_mvar` | **30.0** | SBX-H quantum rate 30 Mvar per 15 min (`sbx/config.py:64`); a 15-min window makes rate and quantum coincide. |
| `t_persist_s` | **180.0** (= `n_need · tso_period_s` with the SBX-H default `n_need = 1`) | `sbx/config.py:71`, `sbx/need.py`. Persistence is counted in TSO iterations, so `t_persist_s` should be stored but asserted to be an integer multiple of `tso_period_s`. Note: 180 s is short relative to a 900-s window; a larger value (e.g. 540 s = 3 iterations) is defensible — kept at the SBX-H-derived default pending E2 evidence. |
| `ramp_steps` | **3** (proposed, no repo anchor exists) | 3 TSO iterations = 9 min; bounds the per-iteration segment-edge change to `dq_grant_mvar/3 = 10 Mvar` per step for a one-quantum grant, same order as the SBX-H per-cycle quantum. Flagged for approval alongside DP2. |
| `price_arb_avg_eur_per_mvarh` | **5.0** (placeholder) | Mirrors SBX-H `p_surplus_eur_per_mvarh = 5.0` (`sbx/config.py:68`, itself marked PLACEHOLDER — calibrate). Exogenous constant per plan §2. |
| `price_arb_grenz_eur_per_mvarh` | **10.0** (placeholder, = 2 × avg) | Mirrors SBX-H `kappa_penalty = 2.0` (`sbx/config.py:69`); `> avg` assertion satisfied. |
| `price_lp_avg_eur_per_mvar_day` | **25.0** (placeholder) | No repo anchor; chosen so one day of Leistungspreis on one quantum (750 €) is the same order as ~6 h of Arbeitspreis on that quantum. Exogenous, settlement-side only. |
| `price_lp_grenz_eur_per_mvar_day` | **50.0** (placeholder, = 2 × avg) | Same ratio as the Arbeitspreis pair. |

All four prices are administratively set constants (plan §2, [LF §1/§4.1]); the placeholders keep
the avg < grenz ordering assertions satisfied and can be re-pinned once B1 is resolved.

### 0.3 Cadence note (DP4 detail)

SBX-H as run uses a 6-min cycle (`k_sched = 2`), not 15 min. SBX-V's `window_s = 900` therefore does
**not** align with the SBX-H scheduler cadence as currently configured; it aligns with 5 TSO
iterations directly. Proposed alignment (no SBX-H change, hard rule 5): SBX-V keeps its own window
counter (`k_window = round(window_s / tso_period_s)`, asserted exact) and does not reuse the SBX-H
`run_cycle` cadence; only the *pattern* (fire at `c · k_window`, consume elapsed-window averages) is
copied. The quantum-as-rate philosophy transfers unchanged (V-D3).

### 0.4 Decision points triggered

| DP | Status | Finding |
|---|---|---|
| DP1 | **Triggered — sanctioned wrapper path applies** | `CapabilityMessage` (`core/message.py:178`) lacks window and direction fields, and its `q_min/q_max` are deltas from the current operating point. `sbxv/potentials.py` will wrap it: absolute box = measured PCC Q + delta bounds; direction split at the band-edge signs; window stamped from the SBX-V window counter. No modification of the original class. Wrapping is possible → no STOP. |
| DP2 | **APPROVED by Manuel, 2026-07-09** (constant + calibration anchor as proposed) | The TSO MIQP objective is dimensionless (weighted squared errors, `g_v = 1e7` per pu², `g_q = 200` per Mvar², `α = 1.0`; see §0.1 item 9). Proposal: one explicit constant `obj_per_eur_per_step` in `sbxv/config.py` converting €-per-window linear prices into objective units per TSO step: `c_lin = price_eur_per_mvarh · (tso_period_s/3600) · obj_per_eur_per_step` applied to the `q_vh`/`q_ug` segment variables. Calibration anchor (proposed): choose `obj_per_eur_per_step` such that the Grenzpreis on one quantum (30 Mvar) costs the same as a voltage-band violation of `v_viol_threshold_pu = 5 mpu` on one bus under `g_v` (i.e. the mechanism is used just before hard violations would persist): `obj_per_eur_per_step = g_v · (0.005)² / (price_grenz · 30 · 0.05) ≈ 16.7` objective-units per €. **Waiting for approval before Phase 1 wires any price term.** Settlement (Phase 2) is unaffected — it is in real €, offline. |
| DP3 | **CONFIRMED by Manuel, 2026-07-09** | Single documented convention: `res_trafo3w.q_hv_mvar`, load convention at the EHV port (positive = TS → DS). Confirmed executable mapping for `sbxv/directions.py`: **positive `q_hv` = DS acts under-excited (absorbs Q) = `LOWERING`** (spannungssenkend); **negative `q_hv` = DS injects Q into the TS = `RAISING`** (spannungshebend). This assignment is now locked for every register, band edge, and settlement case. |
| DP4 | **Triggered — alignment proposed in §0.3** | SBX-H cadence 6 min (as run) vs SBX-V 15-min windows. Proposal: independent SBX-V window counter at 5 TSO iterations; quantum unchanged at 30 Mvar per window. No SBX-H modification. |
| DP5 | **CONFIRMED by Manuel, 2026-07-09 — option (b), netted AggregationArea** | Each DS area has 3 PCC NVPs (`_multi_tso_helpers.py:358`), so per-AggregationArea netting is real, not degenerate. Intended implementation: one `AggregationArea` per DSO area netting its 3 interface transformers (sum of `q_hv_mvar`, precedent at `:560–567`); band edges, grants, opposite-edge settlement references, and 15-min registers all live at the netted area level; per-NVP values are never given a reference point. Please confirm. |

### 0.5 Blocked entries

- **B1 — RESOLVED 2026-07-09:** Manuel added both documents:
  `docs/regulatory/guideline_q.pdf` (Leitfaden, 07.11.2024) and
  `docs/regulatory/draft_vde-ar-n-4141-4.pdf` (E VDE-AR-N 4141-4:2026-07). Phase 2 is unblocked.
  *(Original entry, for the record: the folder did not exist at Phase-0 survey time; the plan
  requires consulting both documents for every settlement and process detail.)*

### 0.6 Additional findings (informational)

- `sbx/messages.py:18` explicitly refers to "the vertical CAIR dataclass style in its BME form
  (`core/coordination_bus.py`)" — a second, BME-side message family exists there; SBX-V wraps the
  operative `core/message.py` `CapabilityMessage` (the one the TSO MIQP actually consumes), not the
  BME bus form.
- The TSO controller already carries per-PCC soft-bound slack machinery (`g_z` plumbing) and a
  Q_PCC tracking term (`g_q_tso`, `controller/tso_controller.py:2081–2109`); the SBX-V piecewise
  segments enter as additional *linear* gradient contributions on the PCC columns plus per-window
  segment bounds — no new hard constraints (V-D1), which is compatible with the existing
  `build_miqp_problem` interface (linear term via `grad_f`).
- Neutral-config equivalence (R1) has a clean hook: with all SBX-V prices at 0 and band =
  capability, the added gradient contribution is exactly 0 and the segment bounds reproduce the
  existing soft output bound → byte-identical dispatch is achievable without touching the solver.

---

## 2026-07-10 — Package rename: sbxv → sbx_v

`sbxv/` renamed to `sbx_v/` (and `tests/sbxv` → `tests/sbx_v`) alongside
`sbx` → `sbx_h`, on Manuel's request. Imports and dotted references
rewritten repo-wide; `coordination_mode = "sbx_v"` accepted as an alias
of the internal `"sbxv"`; `MultiTSOConfig.sbxv_config` and the
`sbxv_runtime` hook key unchanged. Historical entries in this file keep
the old paths. `pytest tests/sbx_h tests/sbx_v` → 165 passed after the
rename. Details:
`docs/daily_log/07_2026/2026-07-10_sbx_h_rename_and_015_helpfulness.md`.
