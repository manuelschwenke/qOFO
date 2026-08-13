# 2026-07-12 — SBX-H v6: deal layer removed, planned support added

Session: Claude Code (Fable), on Manuel's request ("clean sbx-h up to
only have the necessary mechanism left … planned support could be
agreed upon in advance e.g. by demanding a higher boundary voltage
from the neighbour").

## What changed

**Removed (evidence: 015 campaign findings G1–G7, v5 postscript).**
The entire runtime deal layer: `sbx_h/capability.py` (joint-box /
per-corridor / modal LPs), `sbx_h/matching.py`, `sbx_h/messages.py`,
the scheduler's Steps 2–5 (requests, offers, matching, dv execution,
unwind, delivery gate, C1 arming), `corridor_solve_dv`, the need
tracker's request-sign machinery, settlement tier 2 (paid-surplus
billing incl. the v5 `delivered` verification), and every deal-layer
config knob (quantum rate/cap/dust, capability modes, delivery/arming/
sizing knobs, `m_release`, `voltage_margin_pu`).  Complete v5 snapshot
archived in `_archive/sbx_h_v5/` (code + tests + plotter); v4/v5
evaluation outputs preserved under `results/015_SBX_COMPARE/
{v4_baseline,v5_baseline}/`.

**Kept (the necessary mechanism).** Contract layer (corridor registry,
π-line model, `CorridorContract` with v_std/q_band schedules, q_std,
priority terminal tracking via the adapter), metering, the violation
indicator (`need.py`, now indicator-only with the v5 hysteresis), and
the settlement reduced to tier-1 netting + the ATTRIBUTED deviation
tier (C_A/C_B/C_P per-line decomposition, causer-pays at
κ·`p_dev_eur_per_mvarh`; UNATTRIBUTED and ΔP-neutral handling
unchanged).  The A1 over-performance remuneration (architecture doc)
hooks into exactly this decomposition once its review closes.

**Added.**
1. `sbx_h.contract.with_planned_support(contract, t_from, t_to,
   dv_a_pu, dv_b_pu)` — planned support as a SCHEDULE product: the
   named side holds its corridor terminals shifted by dv during the
   window; composes interval-wise with existing planning schedules;
   settlement automatically references the raised promise.  Runner
   plumbing: `MultiTSOConfig.sbx_support_intervals`.
2. A4 escalation indicator: `SBXConfig.escalation_cycles`; a violation
   flag or beyond-band exceedance persisting past it is recorded
   (`scheduler.escalations`, `CorridorCycleRecord.escalation`) as the
   re-planning signal.  No runtime action — deliberate.
3. Config renames for the slim surface: `p_surplus_eur_per_mvarh` →
   `p_dev_eur_per_mvarh` (deviation-tier price basis).

**Adapted.** `sbx_h/{__init__,config,contract,corridor,need,scheduler,
settlement,adapter}.py` (scheduler and adapter substantially smaller —
the adapter no longer reads ANY controller-internal object, only
measurements and config; the former `numerical_h` runner restriction
is gone), `visualisation/plot_sbx.py` (Figure 6: q_std ± band,
deviation staircase, escalation markers), runner validation +
construction, `configs/multi_tso_config.py`.

**Tests.** Suite rewritten for v6: `test_scheduler.py` (steady state
in-band, indicator + escalation, beyond-band UNATTRIBUTED, planned-
support schedule switching incl. q_std shift direction),
`test_settlement.py` (netting, causer-pays attribution per side,
ΔP-neutral, UNATTRIBUTED, rolling window, outputs),
`test_contract.py` (planned-support composition + validation),
`test_need.py` (indicator + hysteresis), golden tie-line tests minus
the dv round trip.  **35 (sbx_h) + 102 (sbx_v) passed**; deal-era
tests archived.

**015 rewritten (v6).** Cells D2/D1/D0 (deficit levels) × arms
none / sbx (contract) / sbx_support (+2.5 mpu on the supporters' sides
of corridors (1,3)/(2,3) during the stress window, agreed in advance);
decomposition contract value = none − sbx, support benefit = sbx −
support; flags V1–V6 (indicator iff deficit, contract value,
support non-harm + reported benefit, escalation fires in D2, solver
health, settlement completion).  Results appended to STATUS_SBX.md
once run.

## Reason

The measured verdict (G1–G7) was that the deal layer is an
unnecessary component: unverifiable at quantum scale, physically
marginal, never armed by an honest exhaustion test.  v6 keeps exactly
what carried the value and turns "support" into what the evidence
says it should be: a planning-time schedule product plus an ex-post
attributed settlement — consistent with the SBX-V (vertical)
philosophy and with architecture candidates A1+A4
(`docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md`).

## Follow-up: deal-era experiment scripts (not done this pass)

`experiments/013_SBX_LADDER.py` (Phase-7 deal campaign),
`014_SBX_SINGLE_DEMO.py` (single-run + live Figure 6) and
`016_SBX_ABLATION.py` (k_sched/quantum ablation) are v5/deal-era
scripts: they import cleanly but reference removed record fields
(`.deal`, `.surplus`, `dq_quant`, scarcity/unwind) and will fail at
RUNTIME. They were left in place because `017_SBX_PLANNING.py` (the
v_std/q_band planning pre-pass — still valid and useful under v6)
imports `013.make_config` + constants at module load; hard-guarding or
moving 013 would break 017's import. Migrating this quartet to v6 (or
decoupling 017's config scaffolding from 013, then archiving
013/014/016) is a separate task. `015_SBX_COMPARE.py` is the current
standalone v6 experiment (imports only 005 + sbx_h). Flagged to Manuel.

## 2026-07-13 — Follow-up CLOSED: experiment migration to v6

`017_SBX_PLANNING.py` decoupled from 013 (scenario identity now
directly from `005.make_cigre_config()` + local DEFAULT_MINUTES /
BAND_FLOOR_MVAR constants — the pre-pass never needed anything else).
`013_SBX_LADDER.py` and `016_SBX_ABLATION.py` moved (git mv) to
`_archive/sbx_h_v5/experiments/` — they exercised the removed deal
layer; their results/findings remain in `results/` and STATUS_SBX.md.
`014_SBX_SINGLE_DEMO.py` REPLACED by a v6 version (single run of a
015 cell, live Figure 6, `--support` for the planned-support arm,
`--schedule` for 017 planning JSONs; old version archived).  Verified:
compile + import smoke green, no dangling references, end-to-end
headless run (D0, 60 min: 5 cycles/corridor, 0 beyond-band, 0
escalations, settlement written).  Also: 015 v6 demonstration ran
2026-07-13 (18/18 flags PASS — results in STATUS_SBX.md v6 entry).
