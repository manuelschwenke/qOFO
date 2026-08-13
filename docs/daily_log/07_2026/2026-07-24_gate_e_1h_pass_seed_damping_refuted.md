# 2026-07-24 — Gate E 1 h PASS (deadband=0), seed & damping levers refuted, init anomaly

## Context / reason
Continuation of the DER Q(V) deadband-edge multi-equilibrium investigation. Goals this
session: (1) answer whether the pandapower(static)-vs-RMS equilibrium gap near the deadband
can be closed by a better `seed_qv_equilibrium` (a) or by tighter `run_control` damping (b);
(2) run and evaluate the 1 h validated-config replay (run 0058).

## What was done

### (a) Seed-basin experiment — `scratchpad/seed_basin_probe.py` (PF-free)
At a profiled operating point (post-init net + raw profile jump to t=300, ±1.0 override,
deadband 0.01), ran `run_control` from three DER-Q seeds: strong (`seed_qv_equilibrium`),
zero, and the RMS's own settled Q (from run 0051). **All three converge to the identical
pandapower fixed point** `{DSO_1:26.7, DSO_2:13.3, DSO_3:7.5, DSO_4:22.8}` — including the
seed set exactly to the RMS point `{60.7,48.9,39.0,40.2}`. Pre-solve values confirmed the
seeds genuinely differed going in.
**⇒ pandapower's equilibrium is SEED-INDEPENDENT**; `seed_qv_equilibrium` is a convergence
warm-start, not a fixed-point selector. Improving the seed cannot steer pp to the RMS basin.

### (b) Damping sweep — `scratchpad/damping_sweep.py` (PF-free)
The QVLocalLoop update is `Q←Q+λ·(target(V)−Q)`, whose fixed points are λ-independent.
Swept λ∈{0.5,0.1,0.02} from both strong and RMS seeds, driving `run_control` to convergence
(catching `ControllerNotConverged`, probing limit-cycle amplitude).
- λ=0.10, 0.02 (both seeds): converge to the SAME single fixed point (26.6/13.3/7.5/22.8),
  resid≈0.09, oscillation ≈0.01.
- λ=0.50 (both seeds): does NOT converge — LIMIT-CYCLES ~17 Mvar pk-pk (identical amplitude
  both seeds → same cycle, not a 2nd attractor) around the same point.
**⇒ damping does NOT move the attractor toward the RMS point.** Aggressive damping only
destabilises convergence into a limit cycle; convergent damping (incl. the operational
0.03 TSO / 0.10 DSO) lands on the one fixed point.
NOTE: the script's built-in `split` verdict is a FALSE POSITIVE — it compared the
non-converged λ=0.50 iterates; gate on `conv=True` before comparing.

**Combined conclusion:** the pp-vs-RMS deadband-edge gap is a genuine solver-vs-solver
fixed-point difference of the non-smooth Q(V)+network map — closable by neither seed nor
damping. `deadband=0` (or off-edge operation) remains the only lever. Sharpens P2:
near the edge `y_qss(u)` is solver-defined (run_control's attractor), and the RMS is a
different solver. Saved to memory `rms-deadband-p2-followups`.

### 1 h run 0058 evaluation (`results/rms_phase6_replay/0058_2026-07-24_091530`)
Config: `--duration 3600 --profiles --profile-delivery elmfile --dso-oltc-switch-cost 200
--der-deadband 0` + G 01 AVR + ±1.0 override, figures on. **Gate E PASS + settling PASS.**
- Static-vs-RMS endpoint: interface_q RMSE **1.03** / MAE 0.59 / max 4.21 Mvar;
  zone_voltage RMSE **0.00085** / MAE 0.00057 / max 0.00201 pu; **0 unsettled** (2160+540).
- Figure `interface_q_static_vs_rms.png`: 9/12 interfaces track near-perfectly; **t9/t10/t11
  (Zone 3 / DSO_4) drift monotonically to ~4 Mvar by t=3600 s** + a transient spike ~t=1250 s.
  Residual DSO_4 divergence, much reduced (4 vs 14–26 Mvar); no runaway.
- DSO Q-tracking (RMS internal): final |err| 0.03–1.98, mean 2.3–3.9, max ≤14.6 Mvar.

## Findings / anomalies to fix (no code changed for these yet)
1. **RMS init took 4.9 h** — `run_1h_gui.log:397` `[T] TOTAL init after [9]: 17665.40 s`.
   The ComSim stepping itself is normal (~25–30 s/step, verified real-time). Run 0051's init
   was negligible. Suspects: the G 01 AVR addition (ComInc struggling for consistent initial
   conditions in the "Rest of USA/Canada" composite) or deadband=0. **Investigate before any
   long run** — a ~5 h init tax per run is prohibitive.
2. **Stale hardcoded caveats in the Gate E summary generator** —
   `experiments/run_rms_phase6_replay.py` ~305–311 unconditionally emit "DER Q(V) … not yet
   implemented … Gate E remains blocked" and "0 … writes skipped because G 01 has no AVR
   block". Both are now FALSE (QVPRE plant law implemented; G 01 AVR added; 0-skipped is
   consistent with the AVR working). They contradict the "PASS" verdict in the same document.
   **Fix/remove for thesis use.**

## Follow-up fixes applied (2026-07-24, later)

### Init anomaly BISECTED from existing logs (no new run) — cause is deadband=0, NOT the AVR
Every run logs init twice (static ~5–7 s, then RMS). Correlating RMS init with the
event-folder size printed in each log:
- deadband 0.01: init ∝ event-folder size (~0.07 s/event) — H2+AVR 45 steps/4701 evts/360 s;
  0051 540 steps/51233 evts/3184 s (8.8× init for 10.9× folder). **0051's init was 53 min,
  never "negligible" — earlier note corrected.** The G 01 AVR is exonerated (H2 sat on the line).
- deadband 0: adds a large super-linear excess over folder-creation — predicted ~217 s at
  30 steps / ~1316 s at 180 steps, actual 825 s (+3.8×) / 17665 s (+13×).
**Mechanism (hypothesis):** the db=0 excess is the RMS ComInc initial-condition solve
thrashing under the steep no-dead-zone droop of all 44 parks, per-iteration cost amplified
by the pre-created event folder. **Reframe: the 4.9 h is a cost of the db=0 DIAGNOSTIC, not
of the physical model** — a physical-deadband (0.01) run pays only the ~linear folder cost.
Potential speedup for db=0 diagnostics: create the event pool AFTER ComInc (delicate — the
pre-created-before-ComInc slots are the ones that fire reliably). NOT applied.

### Stale Gate E summary caveats FIXED (code + artifact)
`experiments/run_rms_phase6_replay.py`: the "Known model limitation" section was two
UNCONDITIONAL hardcoded bullets ("DER Q(V) not yet implemented … Gate E remains blocked",
"G 01 has no AVR block") that went stale when the QVPRE law and the G 01 AVR landed — a PASS
run still printed "blocked". Replaced with `limitations` built from run state: the Q(V)
bullet only when `not qv_equivalent`; the skipped-writes bullet only when `len(skipped_writes)>0`
with a generic reason (no false G 01 naming); the `## Known model limitation` header emitted
only if a bullet applies. For 0058 (qv_equivalent, 0 skipped) the section now vanishes.
Compiles clean; existing `0058/gate_e_summary.md` corrected in place.

## Anchor-mismatch (interval-1) hypothesis — TESTED and REFUTED (2026-07-25)

User pushed the "it's only initialisation" reading and proposed a pre-settlement / common
anchor. Investigation (guided by the user's "re-anchored every 20 s in RMS too?" question):
- CONFIRMED: both plants re-anchor `Vanchor` to the local terminal voltage every 20 s
  (RMS `apply_u` plant.py:595-601; static `core/plant.py:122-127`). Per-interval ΔV (~0.0002 pu)
  ≪ deadband (0.01) ⇒ after re-anchor the droop is DORMANT, Q≈qset ⇒ the persistent gap is a
  closed-loop qset divergence, NOT a continuous edge effect. (My "sitting at the edge" framing
  was wrong.)
- Found a candidate seed in code: the static `QVLocalLoop` cold-starts its anchor at
  `qv_vref_pu`=1.03 on interval 1 (der_qv_local_loop.py:290-296 + runner comment lines
  1781-1785: *"qv_vref_anchor_pu left as the apply-step's responsibility"*), while the RMS was
  anchored to local `v_lf`≈1.02 at init. Hypothesis: this interval-1 anchor mismatch seeds DSO_4.
- **Implemented `seed_der_anchor_to_local_v` (config) + `--seed-der-anchor` (CLI):** initialise
  every DER's `qv_vref_anchor_pu` to local `res_bus.vm_pu` at init on both plants (runner
  ~line 1786). Compiles clean; default off.
- **A/B TEST (db=0.01, oltc=200, 600 s): REFUTED.** Baseline 0059 vs fix 0060 — DSO_4 gap
  −18.3 (baseline) vs **−19.5 (fix, slightly worse)**; interface_q RMSE 2.49 vs 2.76; the t20
  kick persists (DSO_4 −4.5 → −5.6). The anchor mismatch is NOT the seed. **3 h run NOT
  launched** (user's condition "if all works out" not met).
- **Conclusion:** the t20 split persists with identical anchors ⇒ the seed is the two SOLVERS
  (static run_control fixed-point vs RMS ComSim) responding differently to the FIRST profiled
  deadband crossing — the solver-vs-solver fixed-point difference from the seed/damping proof,
  localised to the first profiled step. DSO_4 (voltage sweeps the deadband) amplifies it;
  DSO_1/2/3 self-correct. Init-matching cannot fix a solver-level difference. **deadband=0
  remains the only lever.** Flag kept as a documented diagnostic (default off). Probes:
  `scratchpad/ab_compare.py`; runs 0059 (no-fix) / 0060 (fix).

## Risks / open
- ±1.0 pu DER capability override still active (diagnostic) — revert before publishing.
- t9–t11 Zone-3/DSO_4 residual drift unexplained (why the RMS TSO commands DSO_4 differently
  over the horizon) — the operating-point-robust discrepancy left.
- All static-vs-RMS agreement numbers are within-run (same u not enforced open-loop); the
  open-loop u→y gate remains the cleaner instrument.
