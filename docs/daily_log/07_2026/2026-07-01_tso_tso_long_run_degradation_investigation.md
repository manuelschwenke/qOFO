# Long-run tie-coordination degradation — root-cause investigation

**Date:** 2026-07-01
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Investigate the open problem Manu reported (2026-06-30/07-01): on longer
simulations, the horizontal TSO-TSO tie coordinator sometimes *degrades* control
performance, for both same-reference and different-reference zones. This is a
read-only diagnostic investigation — no changes to `controller/tie_coordinator.py`
or the runner. Diagnostic scripts and pickled logs are scratch artifacts, not
committed to the repo (`results/_long_run_degradation_probe/` is local-only).

## Method

Reused `007_TIE_COORDINATION.py`'s validated `_base_config()` + `TIE_KW` tuning
exactly (via importlib), varying only the zone voltage schedule, contingency, and
horizon. Instrumented with: windowed (60-min bin) mean zone V-RMS error and
`Σ|Q_tie|`; per-tie mean `|tie_dvref|` and the fraction of time saturated at
`±tie_dvref_max`; per-tie mean `|tie_grad_combined|`; total OLTC tap-change count.
Four runs (each OFF + COORD):

1. **Same reference** (1.03/1.03/1.03), clean, 12 h (720 min).
2. **Divergent reference** (007's own 1.05/1.03/1.02), clean, 12 h then extended to
   24 h (1440 min) after a transient signal appeared near the boundary of the
   first test.
3. **Same reference + contingency**: reproduces the *original* 008 scenario per
   `2026-06-30_mutual_gradient_notebook.md` (1.05/1.05/1.05, 400 Mvar reactive
   load connected at bus 9 in zone 2 at minute 20 — before 008's defaults were
   later changed to a divergent scenario), 300 min, matching 008's own horizon
   convention.

## Findings

### Clean (no-contingency) runs: no progressive degradation found

- **Same reference, 12 h:** no clear monotonic decline. COORD vs OFF is noisy
  window-to-window (roughly ±0.3 to ±1.5 mpu either direction), with a mild
  tilt toward COORD being worse in the final 3 hours. COORD does 14 OLTC tap
  changes vs OFF's 0, front-loaded in the first 3 hours (not accelerating).
  Coordinator saturation fraction ≈0 after the first hour.
- **Divergent reference, 24 h:** COORD beats OFF in V-RMS tracking in *every
  single* 60-min window across the full day, and cuts `Σ|Q_tie|` in every
  window too (15-35% in the first half, improving further to the best
  values of the whole run in hours 20-23). One transient episode — ties L5/L18
  hit the hard `±tie_dvref_max` clip for 2 consecutive windows (hours 11-13,
  frac_sat 0.10 then 0.08) — resolved on its own; COORD was still clearly
  ahead of OFF even during that episode. No lasting harm.

**Conclusion for the clean case:** across up to 24 h, with the network's own
validated tuning and no external disturbance, the gradient-exchange mechanism
itself does not show the reported degradation. Whatever mechanism is
responsible needs a disturbance to trigger it (confirmed with Manu — see below).

### Same reference + contingency, 300 min: degradation reproduced, mechanism identified

Steady per-window comparison (OFF stays flat at `Σ|Q_tie|` ≈105-114 Mvar
throughout; COORD starts at 125.6, peaks at 150.6, and settles back only to
~135 Mvar by the last hour — **never returning to OFF's level**). V-RMS is
actually a little *better* under COORD in 4 of 5 windows here — the
degradation shows up specifically in tie flow and switching activity, not
primarily in voltage tracking.

**Exact OLTC tap trace (both runs, from `zone_oltc_taps`):**

| time | run | zone | tap change |
|---|---|---|---|
| 21 min | OFF **and** COORD | 3 | 1 → 2 (identical in both — the expected direct response to the contingency) |
| 27 min | **COORD only** | 1 | -1 → -2 (never happens in OFF — permanent for the rest of the run) |
| 132 min | **COORD only** | 3 | 2 → 1 |
| 252 min | **COORD only** | 3 | 1 → 0 (COORD ends at tap 0; OFF stays at tap 2 for the whole remaining run) |

**The key diagnostic:** tie L2's `tie_dvref` — the coordinator's own agreed
boundary-voltage offset — spikes to -0.077 p.u. right after the contingency
(saturation fraction 0.23 in that window, the largest of any run tested today)
but **relaxes back to -0.007 p.u. by the last hour**, essentially fully
recovered. Yet L2's realised `|Q_tie|` in that same last hour is still **5.2×
higher under COORD than OFF** (27.7 vs 5.3 Mvar). The continuous coordination
state recovered; the realised flow did not.

## Hypothesis (well-supported, not exhaustively proven)

The degradation is not a runaway failure of the continuous gradient-exchange
law itself — every clean long run tested today (up to 24 h) shows that
mechanism behaving well, including recovering from its one transient
saturation episode. Instead, this looks like an **emergent interaction
between the coordinator's transient response and the discrete, sticky nature
of OLTC decisions**:

1. A contingency drives the coordinator's `ΔV_ref` hard toward its clip for
   the affected ties — a bigger, faster reactive-redistribution push than
   the decentralised (OFF) baseline has access to, since OFF has no
   cross-zone assistance to draw on.
2. That extra, faster push is enough to carry zone 1's OLTC across a
   discrete tap-changing threshold that OFF's slower, purely-local response
   never crosses.
3. Once made, that tap step is sticky (`oltc_cooldown_s` / `int_cooldown`
   wall-clock and iteration cooldowns, no mechanism pulls a "wrong" tap back)
   — so it permanently shifts COORD's plant into a different discrete
   operating point than OFF's, even after the coordinator's own continuous
   state has fully relaxed back near zero.
4. Zone 3's OLTC additionally walks to a *different final resting tap*
   (0 vs OFF's 2) over the following hours — a second, independent
   discrete divergence from the same triggering event.

This would explain why short validation runs (007's 70 min, 008's 300-min
sweep grid before this specific reproduction) look clean — the tap-crossing
event may not always occur, and a single validated snapshot doesn't
necessarily hit it — while longer runs, or runs with a contingency, give the
mechanism more chances to trigger and for the resulting discrete divergence
to compound. It is **not** primarily a voltage-tracking problem in this
reproduction (COORD's V-RMS is fine, even slightly better) — it shows up as
persistently elevated inter-zone reactive exchange and extra switching, which
is still a real, quantifiable cost a researcher would rightly call
"degraded performance."

## Assumptions / constraints

- All four runs use 007's own validated `TIE_KW` (`tie_grad_step=0.5,
  tie_anchor=0.5, tie_deadband_v_pu=0.002, tie_dvref_max=0.08`) and
  `tie_grad_eps` at its dataclass default (1e-3) — tuning itself was not
  varied in this investigation.
- The `same_ref_contingency` scenario is a single deterministic trial (fixed
  contingency timing/magnitude) — not yet tested across varied contingency
  size/timing to see how reliably the extra tap-crossing event recurs.
- OLTC discrete-switching parameters (`oltc_cooldown_s_mt=180s`,
  `oltc_cooldown_s_nc=60s`, `int_max_step=1`, `int_cooldown=6`) were left at
  `make_cigre_config()`'s defaults throughout.

## Risks / unresolved points

1. **Causal link asserted here (transient coordinator push → extra tap →
   persistent divergence) is a strong correlation with a plausible
   mechanism, not a proven causal test.** A clean confirmation would rerun
   the same contingency with OLTC action disabled or with a wider cooldown,
   to check whether the extra tap event (and the resulting `Q_tie`
   divergence) disappears.
2. **Not yet tested:** whether this same mechanism reproduces under the
   *divergent*-reference scenario with a contingency (only same-reference
   was tested with a contingency here, per Manu's recollection of "similar
   or shorter than 24h" and a contingency being involved — the specific
   scenario/duration combination was not narrowed further than that).
3. **Not yet tested:** sensitivity to contingency magnitude/timing — does a
   smaller load step avoid crossing the tap threshold? Does the effect
   scale with contingency size?
4. If confirmed, the fix is not obvious: the coordinator's transient
   aggressiveness could be damped (e.g. `tie_coord_period_s > 0` to slow the
   outer loop, or a smaller `tie_grad_step` during/after a detected
   disturbance), or the OLTC cooldown/hysteresis could be tuned to be more
   robust to transient pushes — either is a design change to discuss before
   implementing, not something to change unilaterally here.

## Next note to update in Obsidian

`[[todo]]` — add: (a) confirm the OLTC-tap causal mechanism directly (rerun
with tap action disabled or cooldown widened); (b) test contingency + divergent
reference together; (c) sweep contingency magnitude to see if the extra tap
event is threshold-sensitive; (d) if confirmed, discuss damping options
(`tie_coord_period_s`, smaller `tie_grad_step` post-disturbance, or OLTC-side
hysteresis) before implementing any fix.
