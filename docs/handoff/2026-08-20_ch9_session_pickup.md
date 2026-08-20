# Pickup prompt — Ch 9 §9.1 experiments, session of 2026-08-19/20

Paste everything between the rules as the first message of the new session.

---

You are my research assistant on a PhD project on hierarchical multi-zone
reactive-power control (see `.claude/CLAUDE.md`). A previous session ran the
Ch 9 §9.1 experiments from
`00_daily_log/2026-08-19_handoff_ch9_experiments_A_B.md` and left two
long-running jobs going. **Your job is to pick them up, finish the analysis,
and write it up.** Work in `Z:\Python_Projekte\qOFO_GH`.

**Python on this machine is `F:\python_environments\qOFO_clean\python.exe`** —
NOT the workstation path in `CLAUDE.md`. Cap parallel work at 12 workers; the
server is shared and three colleagues hold PowerFactory sessions
(`dmaihoefner`, `dnickel`, `ahebing` — never touch their processes).

Everything is committed on `main` at `e5d1602`, working tree clean. Nothing
pushed. Read these first:

- `docs/handoff/2026-08-19_ch9_experiments_A_B_results.md` — results so far
  (its Task B1 section is now SUPERSEDED, see below)
- `docs/daily_log/08_2026/2026-08-19_ch9_settling_table_emitter_rework.md` — Task A
- `docs/daily_log/08_2026/2026-08-19_ch9_ts_period_sweep_and_ninner.md` — Task B

## What is running right now

**1. Task A — the open-loop settling battery (PowerFactory).**

```
results/timescale/full_t0_wecc/20260820-135439/
```
Launched as
`python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --label full_t0_wecc --save-trajectories --pre-settle-s 300`.
Preflight passed at `2.46e-08` pu. 16 cases (11 dispatches, 5 disturbances);
expect ~3-5 h total. Watch `run.log`.

**Acceptance is exit 0.** Exit 2 means a case failed, a table row is unfilled,
or a settling time is censored. **A `T_s = 0.00 s` row means the event did not
fire** — that is a silent failure, not a fast plant; see the param-fold fix
below before believing any zero.

**Case 1 already validated the fix.** `der_q_+60Mvar_WP_TSO_s0_b18` returned
`worst u_TN_bus18 T_s = 11.77 s` against `11.23 s` in the 2026-08-07 run of
record — same worst signal, within 5 %. Events fire correctly at the
pre-settled clock, and the small difference is expected: this run measures
from the RMS steady state of the ZIP plant with a frozen profile, not from the
load-flow point. So the remaining cases can be taken at face value unless one
comes back at exactly 0.00 s.

When it finishes: report `timescale_table.tex`, `timescale_summary.md`, the
binding row, the margin at `T_STS = 20 s`, and `T_mech`/`T_elec` for BOTH tap
classes (coupler and machine transformer — each now has its own two-step
instrument case).

**2. Task B1 — isolated-STS `N_inner`, with convergence correctly defined.**

```
results/ch9_ninner/converged/20260820-140643/
```
Launched as
`python experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --workers 12 --label converged`.
240 steps (10 windows x 4 DSOs x 3 interfaces x 2 directions), ~1-2 h.

Analyse with `steps.csv` + `n_inner_summary.csv`. Report, over the flatness
sweep (`n_inner_flat_*` columns, 0.02-1.0 Mvar): median, p95, unconverged
fraction, and `steady_state_offset_mvar`, per DSO and pooled.

## The one thing to get right

**Convergence means the iteration goes flat, NOT that `q_pcc` reaches
`q_set`.** The subordinate controller is multi-objective — it trades
interface-Q tracking against voltage — so its optimum sits at a non-zero Q
offset by design. `N_inner` = first iteration from which
`|q_pcc(k) - q_pcc(k-1)|` stays below the flatness tolerance. The distance
from `q_set` is a separate reported quantity (`steady_state_offset_mvar`), not
a failure.

An earlier definition used arrival-in-band and reported 91 % of steps
"censored" while the areas held 20-31 Mvar of reported headroom, zero voltage
slack and (DSO_3) zero tap moves — converged loops the criterion could not
see. `n_inner_tracking` keeps the old view for comparison; never conflate them.

## Weights: use the FINAL tuned candidate

`experiments/ch_9_parameter_selection/_ch9_selected_design.py` rebuilds the
design through the campaign's own recipe and asserts it against the archive.
It now points at **`results/tuning_mc/stage1`, key `fe010aa3ead1`**
(`rho_emp_p95 = 1.3788`) — the final optimum.

An earlier version used `campaign_0815 / aa4f6d4a8654` (`rho = 1.4480`), the
candidate matching the §9.3 chapter text as drafted. **That is superseded.**
Anything measured at it answers for a controller that will not be used.

The final candidate carries a seventh coordinate `dso_v_authority = 20.0`,
which `build_config` turns into a per-area relief on DSO_2/DSO_4 via
`apply_dso_v_relief` — scaling `dso_g_v` AND `g_w_dso_oltc` together so the
OLTC loop gain is preserved. That relief is **generated, not stored** (the
archive holds `zone_g_w_class: null`), so the module asserts both halves:
`dso_g_v_per_area = 1682790.2832903902` and `dso_oltc = 7852.887279720442`.
Do not "simplify" those assertions away — an unapplied relief is invisible to
a weights-only check, and DSO_2/DSO_4 are the areas that fail without it.

## Still to do

1. **Re-run B2 at the final weights.** `results/ch9_ts_period_sweep/full/...`
   is complete (60/60) but was run at the SUPERSEDED candidate. Re-run:
   `python experiments\ch_9_parameter_selection\ch_9_1_ts_period_sweep.py --workers 12 --label final_weights`
2. **Relabel B2's `n_k`.** It uses the same arrival-in-band criterion that was
   wrong for B1. Its `rho_k` is fine (it IS a tracking measure, which is what
   that sweep is about), but `n_k` must be reported as tracking, or reworked to
   the flatness criterion.
3. **The staleness metric.** `rho_k` cannot show the expected U-shape by
   construction — it measures how well the subordinate layer executed the
   command, not whether the command was still right when it landed. The
   selection argument for `T_TS = 180 s` is therefore NOT yet measured. The
   missing experiment is the same sweep scored on the supervisory objective
   (`f_ts`, `f_q`).
4. **Record `dso_sigma_norm` per case in B1** — it separates "chose a zero
   step" from "was constrained", which nothing currently distinguishes.
5. **The capability filter is built on the REPORTED band**, which is the
   quantity known to over-report (voltage-blind capability message). It
   returned 0 of 240 limited while areas stopped well short of setpoint, so
   "not capability-limited" is not evidence of no limit. Needs a signal that
   is not the reported band.
6. **Update the results doc** `docs/handoff/2026-08-19_ch9_experiments_A_B_results.md`
   — its B1 section reports the superseded tracking definition AND the
   superseded weights.

## The author's method for `N_inner` (follow this)

Assume `N_inner = 9` -> tune (done) -> check by measurement (these runs) ->
if 9 is confirmed, stop; if not, set higher/lower, retune, iterate. The hope is
one iteration. **State the circularity**: `G_w` was calibrated with
`N_inner = 9` assumed, so this is a check on the guess surviving its own
consequence, not an independent measurement.

## PowerFactory study-case state — read before touching it

`02_RMS_CoSim` was repaired this session. Its 97 `ElmFile` profile sources
pointed at a deleted replay snapshot, so `ComLdf`/`ComInc` failed.
`pf/profile_playback.py:365` writes an ABSOLUTE path, so every replay run
repoints the shared case at its own `results/` snapshot and it dies at the next
prune. Current state: all 97 in service pointing at
`pf/profiles/rms_profile_t0_frozen.txt` (a constant-at-t0 profile, generated by
`pf/make_frozen_profile.py`, deliberately under `pf/` which is not pruned).

Restore files, in chronological order, under `results/pf_elmfile_restore/`
(**gitignored — copy them somewhere durable**):

| file | restores |
|---|---|
| `elmfile_outserv_20260819-185102.json` | original: in service, dead `0543` path |
| `elmfile_outserv_20260819-185508.json` | out of service, dead path |
| `elmfile_outserv_20260819-191003.json` | in service, pointing at `0566` |
| `elmfile_pin_20260820-091429.json` | load sources pinned to load-flow values (already reverted) |

Tools: `pf/deactivate_stale_elmfiles.py` (`--repoint` / `--out-of-service` /
`--restore` / `--all`), `pf/make_frozen_profile.py`,
`pf/pin_profile_sources_to_loadflow.py`.

**Do not modify the PF project further without asking the author.**

## Two traps that already cost hours

**The plant does not hold the load-flow point.** The anchored ZIP load model
(`load_model = "zip"`, `P = P_prof*(V/1.03)`, `Q = Q_prof*(V/1.03)^2`, see
`docs/daily_log/07_2026/2026-07-17_rms_phase1_scaffolding_zip_load_model.md`)
makes every load voltage-following, and at the DSO buses Q is capacitive, so
rising voltage increases injection quadratically. The plant leaves the
load-flow point and converges on the ZIP model's own equilibrium ~1.4e-2 pu
away. `--pre-settle-s` handles it, INSIDE every case — `run_case` calls
`ComInc`, which resets to the load-flow point every case, so a clean preflight
does not protect the cases. **With a non-zero pre-settle the operating point is
the RMS steady state of the ZIP plant, not the load-flow solution — the thesis
caption must say so.**

**Parameter events must fold into PF's 60 s event window.** `add_param_event`
did not, while `add_tap_event`/`add_outage_event` did. Invisible until the
battery armed events at a 900 s clock, at which point all six param cases
(2 DER Q + 4 AVR) returned `T_s = 0.00 s` — the event was due at 1805 s in a
965 s run. Fixed in `pf/screening.py`; at clock 0 the fold is the identity, so
existing callers are unchanged.

## Working rules

1. Distinguish measured facts, hypotheses and open questions. Label anything
   projected or carried over. No invented numbers.
2. Commit before any run that produces a number; record commit + dirty flag.
3. If something contradicts this brief, trust the code and say so.
4. Answer as: short answer; assumptions; details; risks / open points.
5. Do not edit the thesis repo (it lives on the author's laptop).
