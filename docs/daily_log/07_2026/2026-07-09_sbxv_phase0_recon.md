# 2026-07-09 — SBX-V Phase 0: repository reconnaissance

**What:** Executed Phase 0 of `SBXV_TSO_DSO_Coordination_Build_Plan.md` (v1.0, 2026-07-08).
Survey only — **no code changes**. Findings written to `STATUS_SBXV.md` (repo root).

**Method:** Located every reuse point named in plan §9 Phase 0 with file:line references;
resolved the `<report>` configuration values of plan §8 from the SBX-H (`sbx/`) package and the
shared 005 scenario; checked all five STOP decision points.

**Key results:**

- `rep1()` → `sbx/fail.py:29`; vertical CAIR message → `core/message.py:178`
  (`CapabilityMessage`, **delta** bounds, no window/direction → DP1 wrapper path).
- Capability box enters the TSO MIQP as a soft output bound on physical Q_PCC, re-anchored at the
  measured interface Q each iteration (`controller/tso_controller.py:1706–1725`).
- SBX-H cadence as run is 6 min (`k_sched = 2`), not 15 min → DP4; proposed independent SBX-V
  window counter (5 TSO iterations at 180 s).
- Quantum 30 Mvar/15 min → `dq_grant_mvar = 30.0`; need-flag persistence `n_need = 1` →
  `t_persist_s = 180 s`.
- MSR/MSC commit + DSO feedforward pattern for grant activation:
  `experiments/runners/multi_tso_dso.py:3885–3970` (`q_itf_sh_offset` into `SetpointMessage`).
- MIQP objective is currency-free (weighted squared errors, `g_v = 1e7`, `α = 1`) → DP2
  conversion-constant proposal written, awaiting approval.
- Each DS area has 3 PCC NVPs → DP5 (netting per AggregationArea) needs confirmation.
- **Blocked B1:** `docs/regulatory/` does not exist — Leitfaden and E VDE-AR-N 4141-4 must be
  added before Phase 2 (settlement fidelity).

**Why:** Phase 0 gate of the build plan; establishes the reuse map and the open decisions before
any `sbxv/` code is written.
