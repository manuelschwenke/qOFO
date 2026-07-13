# 2026-07-09 — SBX-V Phase 3: request pipeline + grants ledger

**Task:** SBX-V build plan §9 Phase 3 (plan:
`C:\Users\Manuel Schwenke\Downloads\SBXV_TSO_DSO_Coordination_Build_Plan.md`).
Phases 0–2 were completed earlier (Phase 2 in a parallel session); this entry covers Phase 3.

## What was changed

Six new modules (purely additive — no existing file touched, hard rule 5):

| File | Content |
|---|---|
| `sbxv/messages.py` | Frozen message dataclasses: `Window`, `PotentialMessage`, `ReserveRequest` (deterministic `request_id`, `all_or_nothing`), `FeasibilityReply` (ACCEPT/PARTIAL/REJECT), `BindingOrder`, `GrantConfirmation`, `IncapabilityDeclaration` |
| `sbxv/potentials.py` | DP1 sanctioned wrapper around `core.message.CapabilityMessage`: absolute netted box = q_meas + delta bounds (DP5), direction split per DP3, gesichert flag from the ledger; `substitute_potential` = the ONE codified missing-message exception (q_pot := band edge, `is_substitute=True`, loud warning) [AR §6.3.2 Schritt 2] |
| `sbxv/need_flag.py` | `VerticalNeedTracker`: flag = A ∧ B (A = netted PCC saturation at the free-or-granted edge; B = persistent voltage deviation in the corresponding direction), `n_persist` consecutive-iteration counting with gap reset (the `sbx.need.NeedTracker` pattern), clearing hysteresis over `n_clear`; `size_request_quanta` (smallest covering n, capped by day-ahead potential beyond band + grants) |
| `sbxv/feasibility.py` | Machbarkeitsprüfung [LF §6.4]: headroom = potential − band − granted − reserve margin; `all_or_nothing` partial → REJECT; replies recorded in the ledger |
| `sbxv/grants_ledger.py` | Append-only `GrantsLedger`: confirmation fail-fasts beyond the last accepted offer or without a feasibility record; window-exact `granted_mvar`; `assert_invariants`; export to Phase-2 `GrantRecord`s |
| `sbxv/pipeline.py` | `RequestPipeline.run_window`: deterministic sorted iteration, one request per (area, direction, window), request → feasibility → order → confirmation in one pass, primitive-tuple replay log, fail-fast on `None` forecasts, invariant sweep per window |

Tests: `tests/sbxv/test_pipeline_phase3.py` (9 tests) covering the plan's acceptance items:
deterministic replay (byte-identical logs), ledger invariants, grant ≤ feasibility, PARTIAL
(via the reserve-margin asymmetry) and REJECT (zero headroom; all_or_nothing) paths, the loud
substitute, the DP1/DP3 direction split, gesichert flagging, need-flag persistence/hysteresis/
gap-reset, and request sizing with the day-ahead cap.

## Key method / structure

- The pipeline is transport-free (plan §12) but every step passes through the §3 message
  dataclasses, so the information accounting stays explicit.
- Intentional asymmetry: request **sizing** (TSO side) does not know the Reserve-Observer
  margin; the **feasibility** answer (DSO side) subtracts it — this is what makes the PARTIAL
  verdict reachable end-to-end and is exercised by the replay test.
- New knobs (`sat_tol_mvar`, `v_dev_threshold_pu`, `n_clear`, `reserve_margin_mvar`) stay
  constructor parameters for now; they move into `SBXVConfig` at Phase-4 wiring.

## Test result

`pytest tests/sbxv tests/sbx` → **142 passed** (SBX-H 100 + SBX-V 42). One test-side fix during
development: the original PARTIAL scenario had headroom for 3 quanta (140 − 50 = 90) so the
pipeline correctly ACCEPTed; the reserve margin (30 Mvar) was added to force the PARTIAL cap.

## Reason

Phase 3 of the SBX-V plan: the scheduling plane (Potenzialmeldung → need flag → Reserve
request → Machbarkeitsprüfung → binding order → grant) as deterministic, replayable in-memory
machinery, ready for the Phase-4 commit-instant integration into the multi-TSO/DSO runner.
