# HANDOVER — SBX Minimal Phase 7 (experiments)

**From:** Claude Code session of 2026-07-07 (Phases 2–6 implementation).
**To:** a fresh Claude Code session (different account, same machine).
**Scope:** run SBX plan v2 §4 **Phase 7** — the comparison experiments.
Phases 0–6 are COMPLETE; do not re-implement them.

Read this file, then `STATUS_SBX.md` (repo root — the full decision
trail: plan amendment v2.2 verbatim, gate decisions, Phase 5/6 findings
F1–F5), then skim `sbx/` module docstrings. The build plan v2 itself is
NOT in the repo (it lived in the chat); everything normative for
Phase 7 is restated verbatim in §3 below.

---

## 1. Environment (this machine)

- Python: `C:\Users\Manuel Schwenke\.conda\envs\qOFO_clean\python.exe`
  (conda env `qOFO_clean`, Python 3.12).
- Repo root: `Z:\Python_Projekte\qOFO_GH` (a network share; runs are
  I/O-tolerant but avoid excessive small-file churn).
- Tests: `python -m pytest tests/sbx` → **57 passed** at handover.
- Full closed-loop check: `python tests/sbx/smoke_sbx_closed_loop.py`
  (three arms × 360 min ≈ 25 min wall; prints C1–C8 PASS/FAIL and the
  settlement summary; exit code 0 = green at handover).
- LP/QP solvers installed: CLARABEL, GUROBI, HIGHS, OSQP, SCIP, SCIPY,
  SCS. The SBX capability LP is pinned to HiGHS (see §4 caveat 2).

## 2. What exists (Phases 0–6, all green)

Package `sbx/`: `fail.py` (`rep1` fail-fast helper), `config.py`
(frozen `SBXConfig`, §5 defaults as amended — note
`voltage_margin_pu = 0.01`, calibrated), `tie_line_model.py` (π-model
q_flow / v_sched_for_q / total-derivative sensitivities; golden-tested
against pandapower), `corridor.py` (registry from the BME 3-area IEEE 39
partition: corridors (1,2) lines 2+14, (1,3) line 25, (2,3) lines 5+18;
Step-4 dv root find), `contract.py` (frozen `CorridorContract`,
`V_STD_DECIMALS = 5`, `q_std_mvar`), `need.py` (consecutive-violation
tracker + relieving-sign assert), `capability.py` (v2.2 D13 joint-box
LP on the existing `MIQPSolver`), `messages.py` (frozen versioned
`PeerCairMessage`, SHA-256 checksums), `matching.py` (pure deterministic
`match`, `DealRecord`), `scheduler.py` (six-step cycle, all corridors
parallel per v2.2; per-cycle `CorridorCycleRecord` incl. `q_meas_mvar`
and the contract-`consistency` label; `settlements` per corridor),
`settlement.py` (§2.5 three tiers, attribution, conservation assert,
`write_settlement_outputs` → ledger CSV + Markdown summary),
`adapter.py` (`SBXRunnerAdapter` — the runner bridge; border-actuator
diagnostic; `terminal_history`).

Runner integration (`experiments/runners/multi_tso_dso.py`):
`MultiTSOConfig.coordination_mode = "sbx"`, plus fields `sbx_config`
(an `SBXConfig` or None) and `sbx_warmup_s` (default 1800 s — contracts
freeze at the first TSO tick ≥ warmup, at the SETTLED closed-loop
state; revised A7). Validation: `"sbx"` excludes
`enable_tie_coordination`, `single_zone_partition`, `numerical_h`,
`local_sensitivities_*`. The adapter is exposed to experiments via the
`pre_loop_hook` state dict key **`sbx_runtime`** (`{"adapter": ...,
"config": ...}`; adapter is None until the warmup tick — read it AFTER
`run_multi_tso_dso` returns).

Data you get per run (from `adapter = captured["sbx_runtime"]["adapter"]`):
- `adapter.scheduler.records[key]` — per-cycle `CorridorCycleRecord`
  (q_std / q_meas / q_sched / surplus / paid / unpaid / deal /
  unwound / acting_area / dv / need flags / offers / t_a / t_b /
  consistency),
- `adapter.scheduler.settlements[key]` + `settlement_engines[key]`
  (per-cycle `CycleSettlement`, cumulative ledger; write files with
  `sbx.settlement.write_settlement_outputs(engines, result_dir, name)`),
- `adapter.scheduler.scarcity_events`,
- `adapter.terminal_history` — per TSO tick `(it, {bus: v_meas},
  {bus: v_ref})`,
- `adapter.border_actuators`.
Zone-level series (violations, Φ, losses, taps, solver status) come
from the returned `MultiTSOIterationRecord` list as in 011/012.

Reference experiment scaffolding to copy:
- `experiments/011_BME_LADDER.py` — rung pattern, metrics.csv, figures,
  `record_bme_phi` (works on EVERY mode → the uniform Φ metric).
- `experiments/012_BME_MONTECARLO.py` — paired-scenario MC design.
- `tests/sbx/smoke_sbx_closed_loop.py` — the working example of an SBX
  run + evaluation (arm construction, hook capture, cycle↔time
  mapping, settlement outputs). Calibrated stress: 500 Mvar Q sink at
  bus 15 (zone 3), zone-3 `v_min = 1.00` via `zone_v_min_pu`,
  contingency connect/trip via `ContingencyEvent` (dormant-load mode).

## 3. Phase 7 — normative spec (plan v2 §4, with v2.2 items 5/6)

> **Phase 7 — Experiments.** Mechanism switch `{AUTONOMOUS, SBX, BME}`
> on identical nets/scenarios/seeds. Scenario families: asymmetric
> scarcity (one area stressed) and symmetric scarcity
> (`ScarcityEvent`s expected). Metrics: global objective, violation
> count/depth, exchanged |ΔQ|, payments per area, deals / scarcity
> events / unwinds, price of autonomy
> `(J^SBX − J^BME)/J^BME` and `(J^AUT − J^SBX)/J^AUT`.
> Plots per repo convention: `q_sched` vs `q_meas` with band shading,
> terminal voltages vs references, need flags, cumulative payments.
> Optional ablation (flag, off by default):
> `k_sched ∈ {3, 15, 30}` min equivalents — the rate-defined quantum
> makes this free; settlement then uses rolling `n_settle_cycles`
> windows so averaging is preserved at short cycles.
> **Acceptance:** asymmetric family mean ordering
> `J^BME ≤ J^SBX ≤ J^AUT` with the recovered gap fraction reported;
> symmetric family degrades gracefully to near-autonomous, scarcity
> logged, `q_sched` settling everywhere.
> **v2.2 item 6:** 3-area triangle partition byte-identical to BME is
> the headline; `spillover_mvar` metric per corridor and cycle
> (deal on corridor X shifting measured flow on corridor Y — compute
> from `records[key].q_meas_mvar` vs the no-deal counterfactual or the
> sbx_inert arm); Phase 7 on identical partition/scenarios/seeds.
> **Definition of done (§8):** one reproducible experiment script
> producing the Phase 7 table and plots; `pytest tests/sbx` clean;
> `git diff --stat` clean of BME paths / vertical CAIR / MIQP assembly
> / solver wrappers.

Arm configs (identical scenario, ONLY the mechanism differs — copy
`make_config` from the smoke test): AUTONOMOUS = `"none"`;
SBX = `"sbx"` (+ `sbx_warmup_s`, HiGHS pinned inside the adapter);
BME = `"bme"` with the 011 calibrated constants (gradient scale 1e5,
w_band 1e4, edges 1.02/1.04, ε/c_oltc/c_shunt from 011 header).
STRONG RECOMMENDATION (Phase 5 finding F1, see §4): carry `sbx_inert`
as a fourth diagnostic arm (contract pinning, no deals — construct
with `SBXConfig(v_viol_threshold_pu=0.5)`), so the pinning cost and
the deal benefit are separable in the J-comparison. J = the uniform Φ
metric (`record_bme_phi=True` on every arm, identical band settings).

For BME arms respect the BME-side constraints (see the runner
validation block): `refresh_shared_jac_on_tso=True`,
`local_sensitivities_*=False`, no `enable_tie_coordination`; use MC
seeds/scenario pairing exactly as 012 does.

## 4. Caveats and findings you MUST fold into the design

1. **F1 — pinning cost ≫ deal benefit (asymmetric smoke):** zone-3
   violation exposure: autonomous 1.645 < sbx 1.980 < sbx_inert 1.999
   pu·step. The Step-4 invariant holds the REQUESTER's own terminals
   at v_std; the deals recovered only +0.019. Expect
   `J^SBX ≤ J^AUT` to FAIL on violation-dominated scenarios unless the
   objective J is Φ (losses + band) rather than raw violation depth,
   or the sbx_inert decomposition is reported. Do not silently tune
   around this — surface it to Manuel with the numbers.
2. **HiGHS backend:** the joint-box LP stalls on OSQP (`user_limit`,
   mixed Mvar/pu column scales). The adapter already pins
   `MIQPSolver(solver="HIGHS")` — do not change zone-MIQP solvers.
3. **F2 — zone 3 joint-box collapse:** zone 3's capability t = 0 every
   cycle (its corridor terminals 14/16/17 are electrically adjacent →
   opposite-sign vertices infeasible). Zone 3 can request, never
   support. Symmetric-scarcity scenarios stressing zones 1+2 (both
   requesting from zone 3) will therefore yield SCARCITY/no-deal even
   when physical capability exists — that IS the v2.2 D13 behaviour,
   log it, don't "fix" it.
4. **F4 — tier-3 attribution under stress:** during a contingency the
   natural flow shift (~100 Mvar on (2,3)) dwarfs the schedule; the
   §2.5 decomposition classifies these cycles (check the settlement
   `attribution` and the record `consistency` labels — `magnitude_off`
   / `UNATTRIBUTED` are the markers). Settlement conservation is
   asserted; if a run dies with `SBXError` in settlement, that is a
   real bug — report, don't catch.
4b. **F6 — tier 3 dominates the money (STATUS §6.3):** in the smoke
   run tier-3 charges fired in 20/21 cycles on EVERY corridor (incl.
   the never-dealing (1,2)) and exceed tier-2 by 8–30× — the 5-Mvar
   band sits far below the 20–60 Mvar tracking-residual noise floor of
   these stiff ties. Phase 7 payment metrics are meaningless until
   D-P7-5 (band recalibration / tier-3 persistence condition) is
   decided with Manuel.
5. **Timing/cycle mapping:** TSO tick every 180 s; `k_sched = 5` →
   15-min cycles; the adapter REBASES its iteration counter at the
   contract-freeze tick, so cycle c's boundary is at
   `t = sbx_warmup_s + c·900 s`. The smoke test shows the mapping.
6. **Runtime:** 360 sim-min ≈ 7 min wall per arm (this machine).
   Plan MC campaigns accordingly; run arms sequentially in one process
   (the runner is not parallel-safe in-process) but you may launch
   separate Python processes per scenario.
7. **Do-not-touch (plan hard rule 5):** BME modules, vertical CAIR
   path, OFO MIQP assembly, solver wrappers. The runner and `sbx/` are
   yours. British English throughout. Fail fast via `sbx.fail.rep1` —
   no silent defaults. Document every session in `docs/daily_log/` and
   keep `STATUS_SBX.md` current (dated entries).

## 5. Open decision points (ask Manuel before locking the design)

- **D-P7-1:** which J for the price-of-autonomy — Φ (recommended,
  uniform with BME ladder) or violation exposure? F1 makes this
  material.
- **D-P7-2:** keep `sbx_inert` as a published fourth arm or diagnostic
  only?
- **D-P7-3:** tier-2 payer rule for EXPORT-need requesters (§2.5
  "importing side pays" makes the supporter pay in that case —
  recorded in STATUS Phase 6; symmetric-scarcity scenarios may hit it).
- **D-P7-4:** the k_sched ablation (off by default) — in scope now or
  deferred?
- **D-P7-5 (from F6, blocking the payment metrics):** recalibrate
  `q_band_mvar` to the realised per-corridor noise (e.g. ~2σ of the
  no-deal `|q_meas − q_sched|` — measurable from an sbx_inert run)
  and/or add a persistence condition to tier 3.
- Calibration horizons: Manuel prefers ~120-min sweeps for calibration
  runs, full 360-min only for the case study.

## 6. State at handover

- `pytest tests/sbx`: 57 passed.
- Three-arm smoke (`results/sbx_phase5_smoke/smoke_result.pkl` +
  `smoke_sbx_settlement_{ledger.csv,summary.md}`): **C1–C8 PASS with
  settlement active**; all 22 sbx-arm cycles settled, conservation
  clean; totals and finding F6 in `STATUS_SBX.md` §6.2/§6.3.
- Working tree: NOT committed (Manuel commits himself; large
  pre-existing uncommitted BME work is intermixed — do not commit or
  revert anything without his say-so).
- Status ledger: `STATUS_SBX.md`; daily logs
  `docs/daily_log/2026-07-07_sbx_phase*.md`.
