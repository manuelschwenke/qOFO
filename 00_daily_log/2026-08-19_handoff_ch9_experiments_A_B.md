# Handoff prompt — Ch 9 §9.1 experiments A and B (paste into the server's Claude session)

Written 2026-08-19 from the thesis repo, after verifying every path, field and
line number below against `Z:\Python_Projekte\qOFO_GH` at commit `53865c3`.
Companion analysis: `00_daily_log/2026-08-19_ch9_sec91_experiment_inventory_and_ninner_design.md`.

Everything between the rules is the prompt. Paste it as the first message.

---

You are my research assistant on a PhD project on hierarchical multi-zone
reactive-power control. This session runs on the big server, which has the
PowerFactory seat and the compute. Your job is two experiments that fill open
values in §9.1 of the dissertation ("Timescale Separation of the Control
Layers"). Work in `Z:\Python_Projekte\qOFO_GH`. Python is
`C:\Users\Manuel Schwenke\.conda\envs\qOFO_clean\python.exe` (adjust if this
machine differs; the env name is `qOFO_clean`).

**Do not edit the thesis repo.** It lives on my laptop
(`...\12_Dissertation\latex_diss_ms`) and another session owns it. Produce
numbers, tables and a written summary; I will paste them in.

Document what you change in `docs/daily_log/08_2026/` following the existing
convention there (what changed, key method or structure of change, timestamp,
reason). Commit before every run that produces a number — two of the three
runs of record in this study were made from a dirty tree and are therefore not
reproducible, and I do not want a third.

## Background you need

Two OFO control layers, both MIQP:

- **TS-OFO** (transmission, EHV), period `T_TS`, dispatches AVR setpoints,
  OLTCs, MSC/MSR, TS-DER, and issues interface-Q setpoints down.
- **STS-OFO** (110 kV sub-transmission, one per EHV–HV interface), period
  `T_STS`, tracks that Q setpoint with local OLTCs and HV DER, and reports a
  capability interval (CAIR) plus tracking error upward.

§9.1 selects `T_STS = 20 s` from measured open-loop settling, and then
`T_TS = N_inner · T_STS` with `N_inner` supposed to be *measured*. Currently
`N_inner = 9` is an educated guess that has never been evaluated, and the
settling table has a defect. Hence two tasks.

- **Task A — the settling battery.** Exists, needs three catalogue additions,
  one structural fix, and a re-run. Needs the PowerFactory seat.
- **Task B — `N_inner`.** Does not exist. Needs writing and running. Pure QSS
  (pandapower), no seat, but compute-heavy.

Do A and B in either order; A needs the seat and B needs cores, so run them
concurrently if the machine allows.

---

# Task A — re-run the open-loop settling battery (Table 9.2)

## A.0 What exists

```
experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py   entry point
experiments/ch_9_parameter_selection/README.md                        status map
pf/screening.py                    StepDef, default_catalogue, disturbance_catalogue,
                                   settling_metrics, GEN_INDEX_TO_PF, ScreeningContext
pf/timescale_study.py              forwarding shim, still works
docs/handover_timescale_study.md   run instructions, §5 lists the traps
tests/experiments/test_ch9_timescale_seperation.py    12 tests
```

Run of record: `results/timescale/full_t0_wecc/20260807-085455` — 8/8 cases,
preflight drift 1.4e-10 pu, no failures, exit **2**, commit `b97badb` **dirty**.
The `20260807-082404` run of the same day is void: every tap returned 0.00 s
because the DER/tap handles were wrong (fix at
`ch_9_1_timescale_seperation.py:341-348`). Do not read numbers out of it.

Read `experiments/ch_9_parameter_selection/README.md` and
`docs/handover_timescale_study.md` before touching anything.

## A.1 The defect: the emitter and the thesis table have diverged

`TABLE_ROWS` at `ch_9_1_timescale_seperation.py:173` emits 8 rows:

| emitted row | matcher |
|---|---|
| Reactive-power step, TSO park | `der_q_` + `WP_TSO` |
| Reactive-power step, DSO DER | `der_q_` + `DER_` |
| AVR voltage-reference step | `avr_vref` |
| OLTC tap, one step | `tap_+1_NC3W` |
| OLTC tap, two steps (instrument only) | `tap_+2seq` |
| MSC switch-in | `shunt_+1` |
| Synchronous-machine outage | `outage_` |
| Load step | `load_` |

The thesis table has a *different* 7 rows: it splits AVR into G09 and G10, adds
a machine-transformer OLTC row, and drops the two-step instrument row. So the
table was filled by hand rather than pasted, and a hand-transcription error got
in:

| thesis row | thesis | measured 20260807-085455 |
|---|---|---|
| OLTC coupling transformer, one step | `11.13`, location "STS 1 B00" | **`16.28`** at `u_DSO_1_bus43` |

`11.13` appears nowhere in `results/timescale/`; "B00" is an unfilled
placeholder. Consequences: the binding row is the **coupler tap at 16.28 s**,
not the machine transformer at 15.13 s; the margin at `T_STS = 20 s` is
**3.72 s**, not the 4.87 s the chapter prints; and the chapter's ordering
sentence inverts.

**Fix the mechanism, not just the number.** Rework `TABLE_ROWS` so the emitter
produces exactly the thesis table, one row per emitted line, in this order:

1. Reactive-power step, +60 Mvar, TS DER  (`der_q_` + `WP_TSO`)
2. Reactive-power step, +20 Mvar, STS DER (`der_q_` + `DER_`)
3. AVR voltage-reference step, +0.02 pu, G09
4. AVR voltage-reference step, +0.02 pu, G10
5. AVR voltage-reference step, +0.001 pu, G09   ← new, see A.2
6. AVR voltage-reference step, +0.001 pu, G10   ← new
7. OLTC coupling transformer, one step   (`tap_+1_NC3W`)
8. OLTC machine transformer, one step    (`tap_+1_MT`)   ← new row for a case that already runs
9. MSC switch-in                         (`shunt_+1`)
10. Synchronous-machine outage
11. Load step

Keep the two-step instrument cases out of the table but keep them running —
they are what splits `T_mech` from `T_elec`. Emit the location column from the
worst signal name, never by hand.

Matching is by **literal substring, never regex** — `StepDef` names contain
`+`, which a regex reads as a quantifier, so `tap_+1` silently matches nothing
and the row is emitted as `[not run]`, which in the thesis reads as a
deliberate omission. That trap is documented at `:160-172`; do not reintroduce
it. Extend `--self-test` and
`tests/experiments/test_ch9_timescale_seperation.py` to cover the new rows;
note `:703` asserts `len(tex) == len(TABLE_ROWS) + 3`.

## A.2 Catalogue additions

All in `pf/screening.py::default_catalogue` (`:1117`); existing case
constructors at `:1163` (DER Q), `:1187` (AVR), `:1201`/`:1207` (coupler tap
×1 / ×2 sequential), `:1219` (machine-trafo tap), `:1232` (MSC).

1. **`avr_vref_+0.001` on G09 and G10.** The chapter says the tuned TS-OFO
   moves AVR references by less than 0.001 pu per iteration, so the +0.02 rows
   are informative worst cases but do not represent tuned controller dynamics.
   A row at the realistic magnitude is what makes the table usable.
2. **`tap_+2seq_MT_*`** — the two-step sequential case for the *machine*
   transformer, instrument only, mirroring the coupler case at `:1207`. Without
   it `T_mech`/`T_elec` cannot be separated for that class.
3. **Nothing else.** Do not add a combined multi-device tap dispatch in this
   pass; it is a separate open item.

## A.3 The `QVPRE` question — decide it before running

`default_catalogue` steps DER Q through `QVPRE.qset`, not `REEC_D.Qext`,
because the Q(V) layer overwrites `Qext` every solver step (`:1150-1160`). That
means the local re-anchored Q(V) droop layer is **inside** every measured
settling time. `Vanchor` is not re-anchored during the battery (no dispatch
occurs), so a step that pushes voltage past the dead band triggers a local
droop response within the measurement — plausible for the +60 Mvar TS-park row.

The thesis caption says "measured open loop", which overstates this. Two
options; tell me which you took and why:

- **(a)** Keep the physics, reword: "no secondary dispatch; primary control
  (AVR, governors, local Q(V)) active". Cheapest, and arguably the more honest
  characterisation of what a dispatch actually excites.
- **(b)** Neutralise `QVPRE` for the battery so the number is the plant
  response alone.

The OLTC `TAPCTRL_qOFO` block is *not* a concern — command-follower with a
mechanical time constant, not a voltage regulator.

## A.4 Run

PF 2025 SP4, project `IEEE39_qOFO`, study case `02_RMS_CoSim`, **GUI closed**,
free licence seat. Seat-free checks first:

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --self-test
```

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --dry-run
```

`--dry-run` must print `gen[1] -> G 03` and `gen[7] -> G 09`. If it prints
anything else, stop and tell me: the PF and pandapower generator numberings are
offset and have collided before. `G 01` is the 10 GVA interconnection
equivalent and the angle reference; it is refused by `OUTAGE_FORBIDDEN` and
that is correct.

Then the run:

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --label full_t0_wecc --save-trajectories
```

**Do not pass `--no-disturbances`.** The disturbance rows have never been run,
and the chapter's summary box claims they establish for how many dispatch
periods after an event a controller still samples a transient — a claim with no
measurement behind it. They cost 600 s of horizon each and bound nothing;
that is intended.

**Acceptance is exit 0.** Exit 2 means a case failed, a table row is unfilled,
or a settling time is censored (still outside the band at the last sample, so
the horizon set the number and not the plant). Re-run censored cases with a
longer `--dispatch-horizon` / `--disturbance-horizon` rather than reporting
them.

## A.5 Report back

- `timescale_table.tex`, `timescale_summary.md`, the run stamp, exit code.
- The binding row, the margin at `T_STS = 20 s`, and `T_s^cont` / `T_s^tap`.
- `T_mech` and `T_elec` for **both** tap classes, from the two-step minus
  one-step difference. For the coupler the existing run already gives
  `T_mech = 6.49 s`, `T_elec = 9.79 s` — i.e. the electrical transient
  dominates and the measured mechanical travel is not the 5 s block parameter
  the caption asserts. Confirm or correct that, and supply the same split for
  the machine transformer.
- Whether the +0.001 pu AVR rows change which row binds.
- Which `QVPRE` option you took.

---

# Task B — measure `N_inner`, and test whether `T_TS = 180 s` is defensible

Nothing exists. `experiments/ch_9_parameter_selection/README.md` lists
`9.1 | N_inner (eq. 9.2) | — | no script yet`. It used to come from a case
study that was cut from the thesis. The open-loop battery explicitly refuses to
report it (`timescale_summary.md`: "`T_TS`/`T_STS` = 9 is the **configured**
ratio ... do not quote this line as evidence for `N_inner`") — keep that
discipline.

Two experiments, because eq. (9.2) and the selection of `T_TS = 180 s` ask
different questions.

**Do B2 first.** It needs no controller or runner change, and its recorded CAIR
band widths are what size B1's step.

## B.1 What licenses running this in QSS at all

`T_STS = 20 s` exceeds the binding open-loop settling of Task A (16.28 s on the
current numbers), so the plant is settled at every STS sample by construction
and the quasi-steady-state model is admissible — not merely convenient. Say so
in your write-up; it is what connects the two halves of §9.1. It would stop
being true if `T_STS` were also swept downward, so **do not sweep `T_STS`**.
`dso_period_s = dt_s = 20 s` is fixed throughout.

## B.2 Use the selected weights, not the file defaults

This matters and is easy to get wrong. `experiments/run_multi_system_ofo.py`
carries the *old in-service* weights (`g_w_der=10`, `g_w_dso_der=1000`, …).
Those were never the outcome of the weight campaign and must not be used —
`N_inner` is a joint property of the period ratio and of `G_w`.

The §9.3 selection is campaign `campaign_0815`, candidate key **`aa4f6d4a8654`**
(verified: it is the row whose six coordinates match the chapter exactly —
`engage_tso_pu 0.018, lambda_tso 0.371, lambda_dso 1.6, tau 1.0,
engage_dso_pu 0.025, dso_g_v_ratio 1.5` — and whose `rho_emp_p95 = 1.4480351`
is the 1.448 the chapter prints):

```
results/tuning_mc/campaign_0815/evals/spec_aa4f6d4a8654.json     the six coordinates
results/tuning_mc/campaign_0815/evals/tier1_aa4f6d4a8654.json    knobs, weights, zone_g_w_class, window_meta
results/tuning_mc/campaign_0815/FINAL/01_search/all_candidates_design_bank.csv
```

Its designed weights: `g_w_der 5.8342850940171225`, `g_w_pcc 22.147759220095267`,
`g_w_dso_der 617.1516305056754`, `g_w_tso_oltc 4740.103355311716`,
`g_w_dso_oltc 183.1121290126729`, `dso_g_v 150000.0`, and
`zone_g_w_class = null` (global scalars, no per-area block for this candidate).

**Do not retype these.** Reproduce the config the way the campaign does, so a
later correction propagates: `tuning_mc/stage_1_search.py::evaluate_one` (`:524`)
is the canonical recipe — `load_config_yaml(tuning/scripts/configs/baseline_ieee39_thevenin.yaml)`
→ `design_payload(knobs, ...)` → `build_config(knobs, weights, baseline_cfg)`
(`:427`). Assert the resulting weights equal the JSON above before running, and
fail loudly if not.

## B.3 Windows and network

The 12 design windows are `tuning_mc/scenarios_mc_v2.py::tier1_design_set()`
(`:277`) — 90 min each (15 min stabilise + 75 min event), roles quiet /
gen_trip / ramp_up / ramp_down / reversal across four seasons, odd ISO weeks.
`WINDOW_META` carries role, season, stratum and start. Use these, so that
"distribution over the design windows" means the same bank §9.3 designed on.

`ScenarioSpec` (`tuning/scenarios.py:42`) already carries `tso_period_s`,
`dso_period_s`, `dt_s` and overlays them onto the config, so the sweep is
`dataclasses.replace(spec, tso_period_s=X)` and nothing else. `_run_scenario`
(`tuning/objectives_v2.py:426`) runs one scenario and returns
`(RunResult, records)` — reuse it rather than calling the runner directly.

⚠ **Check and tell me:** the design bank runs `MC_NETWORK = "rural_700"`
(`scenarios_mc_v2.py:87`), while `ScenarioSpec` defaults to `base_410`. The
weights were designed on `rural_700`, so B must use it too — but confirm this
is the same benchmark Ch 8 describes (IEEE 39-bus with four 110 kV
sub-networks). If the weights were designed on a network the thesis does not
describe as the benchmark, that is a finding I need to hear about, not a
detail to smooth over.

## B.4 — B2: the `T_TS` sweep (the selection evidence)

New: `experiments/ch_9_parameter_selection/ch_9_1_ts_period_sweep.py`, same
conventions as the settling battery — `run_meta.json` with commit **and dirty
flag**, `cases.csv` written *before* the runs, `_latest.txt`, `run.log`,
exit 2 if any point failed or any interval is censored.

Sweep `tso_period_s ∈ {60, 120, 180, 240, 300}` at `dso_period_s = dt_s = 20`,
i.e. `N_inner ∈ {3, 6, 9, 12, 15}`. All are integer multiples of 20 s, so the
GCD requirement at `configs/config.py:188` holds. Identical profile,
contingency schedule and seed at every point; nothing but `tso_period_s`
changes.

**Pilot first**, ~1 h: three points `{60, 180, 300}` on two or three windows.
Its purpose is not results — it is to read the actual CAIR band widths out of
`dso_trafo_q_cap_min_mvar` / `dso_trafo_q_cap_max_mvar` and check the settling
band is reachable at all. See the risk in B.6.

### Post-processor — this is where the work is

Everything needed is already logged in `MultiTSOIterationRecord`
(`experiments/helpers/records.py:260-271`); no runner change:

| field | use |
|---|---|
| `dso_trafo_q_set_mvar` | Q_PCC issued by the TS controller |
| `dso_trafo_q_actual_mvar` / `dso_trafo_q_meas_mvar` | achieved (plant truth) / metered |
| `dso_trafo_q_cap_min_mvar` / `_max_mvar` | CAIR band, absolute Mvar at HV side |
| `dso_trafo_tap_pos` | tap movement, lockout occupancy |
| `dso_z_slack_max` | soft-constraint violation |
| `dso_sigma_norm` | separates "chose a zero step" from "failed / constrained" |
| `dso_group_v_min_pu` / `_max_pu` | voltage bounds |
| `tso_active`, `dso_active`, `time_s` | interval segmentation |

Per (STS, TS interval *k*), segmenting on `tso_active` (the runner fires on
`time_s % period_s < 1`):

- `Δ_k = |q_set(k) − q_set(k−1)|` — how large a correction the TS asked for;
- `r_k = |q_set − q_actual|` at the **last sample before the next TS dispatch**
  — the residual the TS controller inherits;
- `ρ_k = r_k / Δ_k`, discarding intervals with `Δ_k` below ~1 Mvar (without the
  floor the ratio explodes on intervals where the TS barely moved);
- `n_k` = first STS iteration in the interval at which the error enters the band
  **and stays** → empirical `N_inner` *in situ*, right-censored at
  `T_TS/T_STS`. Report the censoring fraction; never drop censored intervals
  silently;
- per interval also: max `dso_z_slack_max`, voltage min/max, tap movements, and
  **tap-lockout occupancy**.

**Do not aggregate over time.** At a fixed horizon `T_TS = 60 s` yields five
times as many dispatches as 300 s, so any time-aggregated RMS residual mixes
"residual per dispatch" with "dispatch frequency". Report per `T_TS` as a
distribution — median / p95 / max of `ρ_k`, censoring fraction, violation
counts — per STS, never one scalar.

## B.5 — B1: the clean measurement (fills eq. 9.2)

New: `experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py`.

Isolated STS-OFO, parent silent, interface-Q setpoint stepped across the CAIR
band, weights frozen at the §9.3 selection. `N_inner` = the first STS iteration
at which the interface flow enters the settling band and stays.

**Good news on the hook:** an exogenous Q_PCC injection path already exists.
`config.q_pcc_setpoints_mvar_per_dso` (`configs/config.py:478`) is synthesised
into a `SetpointMessage` and delivered every step at
`experiments/runners/multi_tso_dso.py:4054-4081`. It is gated on
`tso_mode == 'local'`.

That gate forces a design decision — make it deliberately and tell me which you
took:

- **(i) `tso_mode='local'`, zero code change.** The TS layer runs local Q(V)
  instead of OFO. Physically the realistic "parent silent, primary control
  active" case, but the TS plant then *moves in response* to the STS, which is
  a confound for a measurement meant to isolate the subordinate loop.
- **(ii) Frozen OFO parent: `tso_period_s > n_total_s`,** so the TS-OFO solves
  once at t = 0 to establish the operating point and never revises. Cleaner —
  TS actuators hold. Needs the injection gate relaxed by one condition. This is
  the variant eq. (9.2) actually describes, and my preference; do (i) as a
  cross-check if cheap.

Do **not** reach for `tso_mode='local'` merely because it is the zero-change
path — it swaps in a different baseline controller, and that difference is the
thing being measured.

Other requirements:

- Step **to the CAIR band edge** — the hardest admissible traversal, as the
  chapter promises. Both directions, and from both an importing and an
  exporting initial operating point.
- Let the loop settle at setpoint A before stepping to B; if the current config
  field only supports a constant setpoint, add an optional time-varying
  schedule rather than starting mid-transient at t = 0.
- Repeat over the 12 design windows so the output is a distribution
  (median / p95 / max) plus a censoring fraction, not a single count.

### State the circularity, do not hide it

`G_w` was calibrated with `N_inner = 9` *assumed*; B1 measures `N_inner` with
those weights in place. That is a fixed-point argument evaluated at one
iteration, and it is defensible **only if stated**: the guess is used, the
weights follow, and B1 tests whether the guess survives its own consequence.
Report it that way — as a check on the guess, not as an independent
measurement. Decide in advance, and tell me, what happens if B1 returns
`N_inner > 9`: raise `T_TS`, or recalibrate the weights and repeat B1.

## B.6 Risks — read before writing the post-processor

- **The settling band is a real decision, not a detail.** §9.1 points at the
  thesis's convergence metric, which is still an unwritten stub, so there is no
  authoritative definition. The RMS battery used **1 Mvar absolute** on
  interface flows; reusing it keeps §9.1 internally consistent and is my
  default — but at a PCC with a ~200 Mvar CAIR width that is 0.5 %, which a
  MIQP with discrete taps may never reach, and then *every* interval comes back
  censored and the sweep says nothing. Make it a CLI flag, measure the actual
  band widths on the pilot, and bring me a recommendation with the pilot
  numbers behind it.
- **`T_TS = 60 s` is confounded by construction.** `N_inner = 3`, and the
  coupler OLTC cooldown is 60 s = 3 STS iterations (machine transformer 180 s =
  9 iterations). The changer is locked out for essentially the whole interval.
  Lockout occupancy is therefore not optional instrumentation — it is what
  separates "the STS cannot converge in 3 iterations" from "the changer was
  unavailable". Report it per interval.
- **Expect a U-shape, not monotone decay.** Longer `T_TS` also means a staler
  TS setpoint against the profile. If it appears, that *is* the selection
  argument for 180 s, not a defect — do not treat it as a bug.
- **QSS hides the intra-interval transient by construction.** "Converged" here
  means the QSS iteration converged, not that the plant settled. Say so; the
  closed-loop RMS verification chapter is where that is tested.
- **90-minute windows are thin at long periods.** At `T_TS = 300 s` a window
  yields only 18 TS intervals. If the distributions look starved, propose
  longer windows rather than quietly reporting a p95 over a handful of points.
- **One operating point / one benchmark** carries the whole timescale
  selection. That is a stated scope limit in the chapter; do not try to fix it
  here, but do say if something you see contradicts it.

## B.7 Report back

- Per `T_TS`: distribution of `ρ_k` (median / p95 / max), censoring fraction,
  voltage violations, tap-lockout occupancy — per STS, and pooled.
- Whether the result is monotone or U-shaped, and what it says about 180 s.
- From B1: `N_inner` as a distribution over windows and step directions, the
  band used, the censoring fraction, and which parent-silent variant you ran.
- A one-paragraph verdict I can adapt for the chapter: is `N_inner = 9`
  supported, and if not, what should `T_TS` be?

---

# Working rules

1. Distinguish clearly between measured facts, hypotheses and open questions.
   If a number is projected, inferred or carried over from an older run, label
   it. No invented numbers — the chapter's own guard.
2. Commit before any run that produces a number. Record commit + dirty flag in
   `run_meta.json` as the existing battery does.
3. If something contradicts what this brief says, trust the code and tell me.
   This brief was written from another machine.
4. Answer in the format: short answer; assumptions used; details; risks /
   unresolved points; concrete next steps.
5. Do not edit the thesis repo.
