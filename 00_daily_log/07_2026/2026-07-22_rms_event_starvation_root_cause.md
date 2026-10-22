# ROOT CAUSE FOUND: RMS closed-loop tracking offset = PF event-queue starvation

**Date:** 2026-07-22 (evening)
**Area:** Gate E / PF co-simulation event layer
**Method:** post-mortem forensics on run 0038's PF artefacts (event folder + ElmRes), no re-run
**Scripts:** scratchpad `evt_forensics.py`, `evt_forensics2.py`

---

## The question

Why does the RMS closed loop miss its DSO interface-Q setpoints by 12–26 Mvar
(profiles on) while the static loop tracks to ≤0.6 Mvar — given that the u→y
open-loop test showed the plant reproduces the static plant to ~3 Mvar under
identical commands? Specifically (user): assert that Q_DS^set is actually
commanded in the RMS plant, and assure the re-anchoring works.

## Findings (all from run 0038's own artefacts)

1. **Commands ARE issued, correctly and on time — at source.** The event
   folder holds 30 `qset` EvtParams per DSO park on the exact 20 s grid
   (5363 events total: 1590 EvtLod, 3772 EvtParam, 1 EvtTap — counts match
   30×(53 load + 44 Pref_in + 40 qset + 40 Vanchor) + TSO extras exactly).
   Values evolve as the DSO MIQP decides. The ±1.0 pu capability override
   DID reach the RMS limiter (`qmin=-1.000, qmax=+1.000` in every QVPRE
   params vector; Kdroop=16.67, db=0.01, Tf=0.02).

2. **PF executes those events with a linearly growing delay.** Row-exact
   ElmRes matching (six-decimal value identification): event scheduled at
   t=60 fired at t=120.01; t=80→140.01; t=100→160.01; t=120→240.01;
   t=180→360.01; t=240→480.01; t=280→520.01. Law: an event created in
   dispatch interval k fires **k + 3·⌊k/3⌋ intervals late** — the lag grows
   by 3 intervals every 3 intervals. **Every event scheduled after t=300
   has firing time ≥600 and never executed** (~half of all events pending
   at run end). The ElmRes time axis is monotone 0→600 at uniform 10 ms —
   no rewind/segmentation; this is genuinely late execution, not a clock
   artefact.

3. **The plant itself is exact.** When the sched-160 qset event (Δ=+0.0565 pu
   on the 20 MVA park) finally landed at t=282, park Q stepped 7.79→8.92 Mvar
   = 0.0565×20 exactly, instantly. Plant + QVPRE law verified once more.

4. **Re-anchoring is correct at source, corrupted in execution.** Every
   Vanchor event value equals the park terminal voltage at its creation
   instant to 1e-4 (verified row by row). Fired late, it re-anchors to a
   60–240 s stale voltage; after t=300 anchors stopped landing entirely.

5. **Why the DSO's commands "froze" in the records:** the OFO feeds back the
   *measured* DER state; the plant was executing stale commands, so the
   measurement froze, so u_{k+1} = stale_measurement + σ froze too (σ_norm
   correctly showed 2–6 ≠ 0 the whole time). Controller logic is consistent —
   it was integrating against a plant that hadn't executed its last ~6
   commands.

## Root-cause statement

**PowerFactory admits newly created simulation events into a RUNNING RMS
simulation at a bounded rate (~90 per ComSim.Execute, empirically).** The
profile machinery creates ~177 events per 20 s interval (53 EvtLod + 44
Pref_in + 40 qset + 40 Vanchor) — about twice the admission rate — so the
backlog grows by ~1 interval of lag per interval of runtime. The exact PF
internal mechanism (admission cap vs. queue-rebuild cadence) is unconfirmed;
the 2:1 drain ratio and the 3-interval quantisation are the empirical facts.

## What this explains retroactively

- **Profiles-ON-only manifestation:** without profiles the run creates
  ~85 events/interval — under the admission rate → 0029 tracked to 2–5 Mvar.
- **The 2026-07-21 "120 s DSO response delay" mystery:** those runs created
  88/interval (44 qset + 44 Vanchor) — marginal → episodic delay.
- **The DSO_4 runaway amplification** in profile runs (0034): tap decisions
  taken against a plant executing 2-minute-old commands.
- **Part of the linear per-interval slowdown** of long profile runs.

## Consequence for existing results

- **The u→y validation (2.94 Mvar plant floor) ran under the same starvation**
  (profiles on, 177 events/interval) — its commands landed dilated 2:1. The
  static u trajectory is smooth, so the endpoint error stayed small, but the
  number is NOT a clean plant floor. Re-run after the fix.
- All profiles-ON closed-loop RMS runs (0034, 0035, 0038) are dynamically
  invalid beyond t≈150 s: the plant executed roughly the first half of the
  command stream, stretched 2:1.
- Profiles-OFF runs (0015, 0029) are unaffected (event volume under the rate).

## Proposed fix (to discuss)

**Persistent event pool — reschedule instead of create.** Keep ONE EvtLod
per load and ONE EvtParam per (element, parameter); each interval rewrite
`time` + `value`/`dP,dQ` on the existing objects instead of creating new
ones. The folder stays at ~180 objects, created before/at the first
interval — nothing needs mid-run admission. This is the same fix already
proposed for the event-accumulation performance bug; the starvation finding
upgrades it from performance to **correctness-critical**.

**Prerequisite probe (PF semantics, ~10 min):** verify that a fired
EvtParam can be re-armed by rewriting `time` (+`value`) while the
calculation is active, i.e. it fires again at the new time. If PF refuses,
fall back to an alternating double-buffer pool (2 objects per target) or
pre-created per-interval pools.

**Validation:** re-run 600 s closed loop (expect DSO tracking ≤5 Mvar,
matching profiles-off) and re-run u→y for the true plant floor.

## Assumptions / limits

- Event-admission behaviour measured on PF 2025 SP4, engine mode,
  single-session, events created while ComSim paused between Executes.
- The ~90/Execute figure is inferred from one run's arithmetic (177
  created vs ~88.5 executed per interval); treat as order-of-magnitude.
- Only qset was monitored per park; EvtLod/Pref_in starvation is inferred
  from the shared queue, not observed directly (their targets are not in
  ElmRes).
