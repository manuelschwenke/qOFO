# 2026-09-10 — Weight reset, and Stage I of the new tuning campaign

**Branch:** `weights-reset-2026-09-10`
**Reason:** the definition of controller weights had no single answer. Eight
competing sources plus three layers of silent defaults, and the two entry points
that were supposed to be authoritative both failed to run.

---

## 1. What was wrong

| source | example | kind |
|---|---|---|
| `configs/config.py` dataclass defaults | `g_v=5e4, g_w_gen=1e7, g_w_der=2` | silent fallback |
| `experiments/run_multi_system_ofo.py::make_config` | `g_v=1e7, g_w_der=14.4` | de-facto current |
| `run_multi_system_ofo.py::main_comparison` | `g_v=1.5e5, g_w_der=10` | second set, same file |
| `configs/paramsets/{tuned,per_area}.json` | JSON overlays | 2026-08 campaign |
| six YAML baselines under `tuning/scripts/configs/`, `configs/` | `g_v` 1e5…1e7 | BO campaign |
| `experiments/CIGRE_2026/005,006` | `g_v=1e7, g_w_gen=5e9` | frozen paper set |
| `_ch9_selected_design.py` | 7 knobs → generative rule → 2 patches | thesis §9.3 |
| `tuning/metrics.py::CostWeights` | `w_v_track_ts=35, …` | *scoring* weights, also hand-set |

Three findings beyond the clutter:

1. **A hidden per-area multiplier was on by default.** `configs/config.py:836`
   defaulted `dso_v_relief_factors` to `{"DSO_2": 10.0, "DSO_4": 10.0}` and
   applied it in `__post_init__` on **every** construction and every
   `dataclasses.replace`. Any run not overriding it silently carried a ×10
   voltage authority on two of four DSO areas.
2. **Both authoritative entry points were broken.**
   `run_multi_system_ofo.py:359` applied a parameter set `ch9_frozen` that does
   not exist (`SystemExit`); `_ch9_selected_design.py` asserted its rebuilt
   weights against `results/tuning_mc/stage1/evals/*.json`, and `results/` is
   gitignored and absent from this clone.
3. **λ had two incompatible definitions in tracked code.**
   `tuning/reparam.py:50` computed it over **continuous columns only**; the
   thesis (`Chapter09.tex:346`) and `stage_0_coupling_decomposition.py` include
   the tap and shunt columns, which is where the irreducible floor comes from.
   Same symbol, two numbers, differing ~50× on TS zone 1.

---

## 2. What changed

### Deleted — parameter values (17 files)
`configs/paramsets/*.json`, `configs/tuned_params_reparam*.yaml`,
`configs/baseline_002.yaml`, `tuning/scripts/configs/*.yaml`,
`tuning_mc/configs/limits_mc*.json`.

### Deleted — retired campaign machinery (24 modules)
`tuning/`: `reparam`, `rawspace`, `tune`, `objective`, `objectives_v2`,
`ceilings`, `compat`, `holdout_metrics`, `validate`, `scripts/`,
`stability_certificate/`, `reports/`.
`tuning_mc/`: `stage_0_preconditioning`, `stage_0_lambda_curve`,
`stage_0_coupling_decomposition`, `stage_1_search`, `stage_1a_excitation`,
`report_0815`, `export_final`, `preflight_rerun`, `metrics`, `scenarios_mc`,
`select_windows_v2`.
Plus six `experiments/CIGRE_2026/007{k,m,n,o,p,q}_*.py` — diagnostics *of the BO
objective*, whose subject matter goes with it — and eight test modules that
tested deleted code.

**Kept and reused:** `tuning/{metrics,runner,scenarios,parameters,
bisect_switching,_io,_sim_loader,_types}`, `tuning_mc/{scenarios_mc_v2,
coupled_spectral_radius}`, `controller/gw_precondition.py`.

### Closed the default holes
* The nine weights in `configs/config.py` are now `Optional[float] = None`.
  New `MultiTSOConfig.REQUIRED_WEIGHTS` / `.missing_weights()` /
  `.require_weights()`; the last is called by `run_multi_tso_dso` **before any
  controller is built**. Not from `__post_init__`, because a bare
  `MultiTSOConfig()` is still legitimate for structural reads —
  `export.make_snapshots` does exactly that for `scenario`.
* `dso_v_relief_factors` now defaults to `None`.
* `tuning/parameters.py` rewritten: `WEIGHT_FIELDS` (9), `GAUGE_FIELD` (`g_v`),
  `SEARCHED_FIELDS` (8). `apply_to_config` demands a **complete** weight vector
  and rejects unknown keys, non-positive and non-finite values.
  `FIXED_OVERRIDES` pins **no weight** — it previously re-pinned
  `g_w_tso_shunt` and `dso_gamma_oltc_q` on every trial — and now forces
  `precondition_g_w=False`, since the runtime preconditioner *regenerates*
  `g_w` at controller construction and would overwrite the selected weights.
* `make_config()` is now the **structural** factory: horizon, cadence, network,
  events. It carries no weights and returns a weight-incomplete config by
  design. Cadence corrected from `dso_period_s=15 s` to **20 s**, the
  `T_STS` Experiment A fixed (`Chapter09.tex:146`); the file had disagreed with
  every Chapter-9 script.
* `_ch9_selected_design.py` reduced from 366 lines of regenerate-and-patch to a
  parameter-set load.

### New — `tuning/stage1_screen.py`
Stage I: the analytic contraction screen, no simulation.

`B = Q^{1/2} H G_w^{-1} H^T Q^{1/2}` (`Chapter09.tex` eq.
`param:weights:recurrence`). Because `H` depends on the **operating point** and
not on the weights, and

```
H^T Q H = g_V · S_V + g_Q · S_Q ,   S_V = H_V^T H_V ,  S_Q = H_Q^T H_Q
```

the two Gram matrices are captured **once per condition** and `λ_max(B)` is then
one symmetric eigenproblem of size `n_u` (6–14 here) per candidate. Six
controller builds screen an arbitrarily large design; only survivors are
simulated.

Integer actuators are **in** `B` (relaxed unit increments), and additionally get
the exact commit statement `2|aᵢᵀz| > ‖aᵢ‖² + g_{w,i}`, inverted to price a tap
class from a stated engagement deviation.

---

## 3. Measured today

**Layer composition (W1, `rural_700`).** No shunt columns appear in any MIQP —
the integrator dispatches them outside it, so the shunt is correctly out of this
analysis, and `g_w_tso_shunt` never needs a value in the screen.

| layer | n_u | classes |
|---|---|---|
| TS_1 | 6 | 1 der, 3 gen, 2 oltc |
| TS_2 | 14 | 1 der, 9 pcc, 1 gen, 3 oltc |
| TS_3 | 9 | 2 der, 3 pcc, 2 gen, 2 oltc |
| STS_DSO_1..4 | 13 each | 10 dso_der, 3 dso_oltc |

**The engagement requirement and the 1.6 screen are compatible.** At
`π_q = π_vDS = 1` on W1:

| Δv_eng TS / STS | g_w_tso_oltc | g_w_dso_oltc | floor TS | floor STS |
|---|---|---|---|---|
| **0.010 / 0.020** | 20.7 | 17.1 | **0.5177** | **0.3514** |
| 0.015 / 0.025 | 34.9 | 22.4 | 0.3065 | 0.2677 |
| 0.020 / 0.030 | 49.2 | 27.8 | 0.2177 | 0.2162 |

Against a selection ceiling of `1.6/1.035 = 1.546`, the requested thresholds
leave ≈1.03 (TS) and ≈1.19 (STS) of headroom for the continuous loop gain. The
floor falls as taps are priced more dearly, as the thesis states.

**Screen ladder** — changed from the thesis on the author instruction:
`λ_max(B) ≤ 0.8·2 = 1.6` (was `0.75·2 = 1.5`), select at `1.6/1.035 = 1.546`.
`Chapter09.tex:359`, `AppendixD_BO_Hyperparams.tex` (`λ̄` 1.449 → 1.546) and
`docs/ch9_weight_selection_method.tex` must be updated to match. **Caution for
the text:** 1.6 is also the fit-validity range quoted in the 2026-08-17 handoff;
the two are different quantities that now share a value and must be
distinguished explicitly.

**The self-test found a real modelling fact.** `λ_max(B)` from the controller
sensitivity and `lambda_max_Mii` from `MultiTSOCoordinator.check_contraction`
disagree by **9 %–80 %** per zone (ratio 0.905 / 0.205 / 0.599 on W1). Not a
defect: the coordinator assembles `H_ii` column-by-column from raw Jacobian
sensitivities for its cross-zone criterion, while the controller acts on its own
post-transform matrix. The Chapter-9 recurrence describes the update the
*controller* performs, so the screen uses the controller sensitivity; the
coordinator blocks belong to the coupling diagnostic, where the off-diagonals
are the point. **This needs a sentence in the thesis** — the coupled figures and
the local screen are not formed on the same model.

`--self-test` now asserts what is actually true: the Gram factorisation is exact
to machine precision, `λ_max(B)` is **gauge-invariant to ~1e-16** under a common
×7 rescaling of all nine weights (the empirical licence for pinning `g_v`), and
the screen reads the controller own curvature rows.

---

## 4. Test status

`pytest tests/` — **742 passed, 29 failed, 15 skipped, 6 errors** (41 min).

**28 of the 29 failures are pre-existing and unrelated.** Verified by running
the same modules against a clean `git archive` of `HEAD` (pre-reset, weights
still defaulted): identical 16 failures on
`test_msc_msr_banks.py` + `test_tso_tertiary_shunt.py`. Cause is
`ValueError: Unknown scenario: 'base'` from `network/ieee39/build.py:383` — the
`base` scenario was removed earlier, and
`test_dso_generation_scenarios.py::test_removed_scenarios_are_rejected[base]`
*passes*, asserting that removal. Those test modules were never updated. I
touched none of the affected files.

**One failure was mine and is fixed:** `test_run_one_invalid_param_keys_raises`
matched the old `"Unknown BO params"` message.

Rewritten: `tests/test_dso_v_relief_pairing.py` (was built on
`make_config()` + every JSON paramset + `stage_0_preconditioning`, and asserted
the relief is on by default — the behaviour deliberately removed; the invariant
itself is kept and now tested against an explicit config),
`tests/tuning/test_parameters.py`, `tests/tuning/test_io.py`.

---

## 5. Open

* Pre-existing `base`-scenario test rot (28 failures) — not touched, not mine to
  decide.
* `CostWeights` / `cost_J` kept in `tuning/metrics.py` as a **reported
  diagnostic**. Selection is by constraints plus a lexicographic order, so no
  scoring weights enter the campaign; ripping the class out would ripple through
  `extract_metrics` and its tests for cosmetic gain.
* W2–W6 PowerFactory snapshots are absent from this clone. Not needed: the
  campaign is entirely quasi-static and builds each condition from
  `start_time` + `use_profiles` against `data/profiles.csv` (full year 2016,
  15-min, present).
* Environment rebuilt at `F:/python_environments/qOFO_clean` (py 3.12.14).
  `requirements/requirements-lock.txt` is **self-contradictory** as pip resolves
  it today — it pins `numpy==2.4.6` and `pandapower==3.4.0`, and pandapower
  3.4.0 metadata caps `numpy<2.4`. Installed with `--no-deps`, which is the
  correct way to replay a full freeze; verified pandapower power flow and a
  cvxpy MIQP via SCIP both work. GUROBI, HIGHS and SCIP all present.

---

## 6. Stage II-a — the screening sweep (run 2026-09-10 19:06, 4.84 h)

512 scrambled-Sobol draws, seed 20260910, log-space, box centred on the Stage-I
point (`configs/paramsets/stage1_start.json`). Spans: ±1.5 decade on the
objective priorities, ±1 on the continuous step prices and `g_w_gen`, ±0.5 on
the tap prices.

**The analytic screen paid for the campaign.** 512 draws → **115 survivors
(22.5 %)** in seconds; 77.5 % of the box was rejected without a single power
flow. Simulated 115 + the Stage-I baseline on 6 conditions × 3 h, 28 workers:
**116 evaluated, 0 failed, 4.84 h wall.**

Feasible after the constraint filter (wear NOT judged on the short bank):
**89 / 116**. Rejections: 15 on the relative interface-Q bound, 12 on hard
voltage-corridor violations.

**The baseline ranks 42 / 89.** Stage I is an analytic starting point, not an
optimum, and the search finds real improvement — best `e_v(TS)` is **−41.8 %**
against the Stage-I point.

### The trade-off is real and the frontier is not a point

| key | λ | e_v(TS) | e_v(DS) | Δ TS | Δ DS |
|---|---|---|---|---|---|
| `3bca92731a6e` | 1.344 | 0.00181 | 0.00950 | **−41.8 %** | **+51.8 %** |
| `a9555cdf78ba` | 1.216 | 0.00197 | 0.00582 | −36.7 % | −7.0 % |
| `13eef598aad0` | 0.837 | 0.00206 | 0.00545 | −33.8 % | −13.0 % |
| `399e30d3b60b` | 1.311 | 0.00222 | 0.00458 | −28.7 % | −26.8 % |
| `0f2ada9d2efa` | 0.977 | 0.00233 | 0.00380 | −25.2 % | −39.3 % |
| `1536fc132bde` | 1.353 | 0.00418 | 0.00196 | +34.3 % | −68.6 % |
| baseline | 1.000 | 0.00311 | 0.00626 | — | — |

11 Pareto-frontier points. The **strict lexicographic winner sits at the
extreme**: it buys 41.8 % of TS voltage by giving away 51.8 % of DS voltage.
Four frontier points dominate the baseline on *both* criteria. The 2 %
indifference band is too tight to reach them, so the band is a live design
choice, not a formality — it is the only thing standing between the stated
priority order and a candidate that spends one controlled output entirely.

### Coordinate sensitivity (Spearman ρ, feasible set)

| coordinate | e_v(TS) | e_v(DS) | e_q |
|---|---|---|---|
| **`g_w_tso_oltc`** | **+0.801** | −0.125 | +0.019 |
| **`g_q`** | −0.247 | +0.292 | **−0.779** |
| **`dso_g_v`** | +0.140 | **−0.589** | +0.116 |
| `g_w_pcc` | +0.222 | −0.373 | −0.171 |
| `g_w_der` | +0.241 | +0.238 | +0.027 |
| `g_w_dso_der` | −0.041 | +0.238 | −0.096 |
| `g_w_gen` | +0.094 | −0.060 | +0.068 |
| `g_w_dso_oltc` | −0.007 | +0.020 | −0.058 |

Three coordinates carry the sweep: `g_w_tso_oltc`, `g_q`, `dso_g_v`.

**`g_w_gen` is inert over [1e7, 1e9]** (|ρ| ≤ 0.09 on every criterion), and AVR
step sizes came out at 1.7e-5 … 3.6e-4 pu against a 5e-3 limit — two orders of
margin. The smoothness requirement is satisfied everywhere in the range, so the
AVR price is not a tuning coordinate on this plant. That is a reportable result,
not an absence of one.

**`g_w_dso_oltc` is inert on the short bank** (|ρ| ≤ 0.06), as expected: it
prices switching, and 3 h does not exercise switching. It is a Stage II-c
coordinate.

### The short bank is biased, in two known directions

**Cheap TS taps.** Mean box position of the top 10 on `g_w_tso_oltc` is **0.20**
(0 = cheapest), and they run 16–24 tap ops/day against the baseline 8. With wear
unjudged on a 3 h window the search simply buys TS voltage with tap operations.

**Cheap DS voltage.** Mean box position on `dso_g_v` is **0.29** — the search
discounts the subordinate voltage schedule, which is
`METHOD_weight_selection.md` §5.3 reproduced exactly.

Both are corrected on the 12 h bank, where the per-day budget is readable
(1 tap = 0.083/h at 12 h against 0.33/h at 3 h). **No Stage II-a ranking may be
quoted as a result.** Its outputs are a feasibility map, the sensitivity table
above, and a shortlist.

## 7. Stage II-b — launched 2026-09-11 00:00

19 candidates (11 Pareto-frontier + top-12 lexicographic + baseline,
deduplicated) on the **12 h** bank, all six conditions, 28 workers, ~3.4 h
estimated. W5 is run but excluded from the ranking aggregate and reported
separately. Shortlist at `results/THESIS_ch9_3_stage2/refine/shortlist.json`.

### Stage II-b, first attempt — total loss, and the design fault behind it

Launched 2026-09-11 00:00, 20 candidates × 6 windows × 12 h, 28 workers.
**All 20 futures failed at 1.81 h with nothing written.** The simultaneity is
diagnostic: `ProcessPoolExecutor` marks *every* pending future
`BrokenProcessPool` when one worker dies, so a single death destroyed the run.

**Not memory.** Measured afterwards on a standalone 12 h window: 91 min wall,
2160 records, **RSS 0.50 GiB**, pickled record list 13 MiB. 161 GiB was free
throughout and swap sat at 1.5 %. The original trigger is unidentified; this is
a shared server (python processes in other RDP sessions), so an external
interruption is plausible and is recorded as unexplained rather than guessed at.

**The fault that turned one death into total loss was the work unit.** A whole
candidate is 6 × 12 h ≈ 3.4 h, atomic, written only on completion — so at
1.81 h nothing had finished and nothing was saved. Stage II-a survived the same
design only because its units were short enough.

Rebuilt in `ch_9_3_stage2_screen.py`:

* work unit is one **(candidate, window)** pair, ~30-45 min, persisted to
  `windows/<key>__<W>.json` the moment it finishes;
* `BrokenProcessPool` is caught, the pool rebuilt, and outstanding units resumed
  from disk (`--max-restarts`, default 5) — the pattern
  `ch_10_6_ladder_mc.py` already used;
* candidate records are assembled from window files afterwards, and a candidate
  missing any window is **not** written rather than silently aggregated over a
  partial bank;
* jobs deduplicated by weight fingerprint, so an explicit shortlist that already
  contains the baseline cannot have two workers writing one path.

Relaunched 02:10 at 24 workers: 19 candidates × 6 = **114 units**.

**Incidental:** at the Stage-I point on W1 over 12 h, switching is very low —
0.0 tap ops/day TSO, 2.0 DSO. The 0.010 pu engagement threshold is rarely
crossed in a quiet window, so the baseline is inert on taps and the wear budget
will not bind for it. W5 carries `e_v(TS)` ~5x every other window, which is the
zero-capability condition behaving as the thesis describes and the reason it
stays out of the ranking aggregate.

---

## 8. Stage II-b — the 12 h bank (114 units, 0 failures, 3.31 h)

**The short-bank ranking inverted, as predicted.** Stage II-a's lexicographic
winner `3bca92731a6e` is rejected at 12 h on 79 DS corridor violations and 3
TSO reversals. The binding constraint across the bank is **hunting, not the tap
budget**: tap operations came out at 0-14/day against a budget of 30, while
reversals decided 7 of the 7 wear rejections.

### A constraint bug, found and fixed

The first pass reported 4/19 feasible with twelve candidates rejected as
"TSO reversals/day 2.0 > 2.0". Reversal counts are **integers per window**
(measured: 0, 1, 3 or 5 per 12 h). One reversal in 12 h is stored as
`0.0834/h`, which multiplied by 24 gives **2.0016/day** — and a strict
comparison against a 2.0/day budget rejected it. Twelve of nineteen candidates
were discarded by that rounding artefact alone.

`check_limits` now compares **integer counts over the window** against
`floor(budget/24 * hours)`, and takes `window_hours`. Corrected result:
**12/19 feasible**, 7 rejected on genuine multi-reversal hunting or DS corridor
violations.

### Result at the declared rule (2 % indifference band)

| # | key | λ | e_v(TS) | e_v(DS) | taps/d TS | rev/12h |
|---|---|---|---|---|---|---|
| 1 | `a9555cdf78ba` | 1.216 | 0.00148 (**−42.6 %**) | 0.00596 (−3.9 %) | 4 | 1 |
| 2 | `13eef598aad0` | 0.837 | 0.00159 (−38.0 %) | 0.00584 (−5.8 %) | 4 | 1 |
| 4 | `399e30d3b60b` | 1.311 | 0.00166 (−35.3 %) | 0.00480 (**−22.6 %**) | **2** | **0** |
| 11 | `874818ae70a4` | 1.000 | 0.00257 | 0.00620 | 2 | 0 | ← Stage-I baseline |

**Six candidates dominate the baseline on both voltage criteria.** The winner is
stable for bands 0-8 %; 10-12 % selects `13eef598aad0`; 15-30 % selects
`399e30d3b60b`; 50 % selects `5f017f940cc3`.

## 9. The coupled diagnostic disagrees with the screen — and it matters

Stage I-d run at four points on W1. `rho(I-M)` and `||I-M||_2` on the
observable subspace:

| point | local ρ(M) | local ρ(I−M) | coupled ρ(M) | coupled ρ(I−M) |
|---|---|---|---|---|
| `stage1_start` (analytic) | 1.3675 | 1.0000 ok | 1.6204 | 1.0000 fail |
| `1536fc132bde` | 1.7389 | 1.0000 ok | 2.3869 | 1.3869 fail |
| `399e30d3b60b` | 1.8510 | 1.0000 ok | 2.5170 | 1.5170 fail |
| **`a9555cdf78ba`** (selected) | **2.2127** | **1.2127 FAIL** | 2.8897 | 1.8897 fail |

Two separate findings, and they pull in different directions.

**(a) The coupled test fails everything, including the analytic baseline.** It
therefore cannot serve as a selection criterion as currently formed — it does
not discriminate between designs. That is consistent with the architecture:
`local_sensitivities_tso = True` means the controllers never see the coupling,
so a criterion that retains the off-diagonal blocks is measuring a model the
controller does not act on. It stays a **reported limitation**, which is what
Chapter 9 already says it is.

**(b) The LOCAL coordinator-model test does discriminate, and the selected
point is the only one of the four that breaches it.** `rho(M) = 2.2127 > 2` at
`a9555cdf78ba`, against 1.37-1.85 for the other three.

This is the two-model discrepancy of §3 becoming load-bearing. The Stage-II
screen is formed on the **controller** sensitivity (which the Chapter-9
recurrence describes) and put `a9555cdf78ba` at λ = 1.2164, comfortably inside
1.6. The coordinator forms its criterion on raw Jacobian blocks and puts the
same point at 2.21, outside the OFO bound of 2.

Neither model is obviously wrong, and the empirical evidence does not condemn
the point: over 6 × 12 h it produced the best TS tracking in the bank with no
power-flow failures, no corridor violations and one reversal. A linear
diagnostic saying a mode fails to decay is not the same as the constrained,
mixed-integer, multi-rate loop diverging — which Chapter 9 states explicitly.

**Open decision for the author, not to be resolved silently.** Either
(i) screen on both models and take the worse, which rejects `a9555cdf78ba` and
selects `399e30d3b60b` — 35.3 % better TS *and* 22.6 % better DS than baseline,
at 2 taps/day and zero reversals, the operationally gentlest point on the
frontier; or (ii) keep the controller-model screen as the criterion and report
the coordinator-model figure alongside it as a stated discrepancy.

`configs/paramsets/ch9_selected.json` currently holds `a9555cdf78ba` — the
point the **declared** rule produces. It is revisable and should be revisited
with this decision.

---

## 10. The wear metric was the wrong quantity (2026-09-11)

### What the 5 h disturbed comparison showed

Three parameter sets (`stage1_start`, `ch9_alt_lowwear` = candidate
`399e30d3b60b`, `ch9_selected` = `a9555cdf78ba`) run for 5 h with the
`make_config` contingency schedule -- generator trip, 250 Mvar load step, line
outage. **The first disturbance test the weights have had**: the whole tuning
bank was profile-driven with `contingencies=0`.

Tracking held up and kept its rank order: `e_v(TS)` 0.00141 / 0.00143 / 0.00183
(selected / alternative / baseline), no power-flow failures, AVR steps
2.5-2.9e-4 pu against a 5e-3 limit.

Wear did not. Tap operations rose ~7x against the quiet bank, and raw
reversals extrapolated to **9.6-14.4/day for all three**, including the
analytic baseline.

### That figure was wrong, twice over

1. It extrapolated a 5 h count to a day -- the quantisation error
   `METHOD_weight_selection.md` §5.2 warns against, and which this log had
   already quoted.
2. It counted *every* reversal. A tap that moves on a generator trip and moves
   back when the unit is restored has responded to two distinct events and is
   behaving correctly.

**A per-day reversal budget is not the requirement.** What the requirement
forbids is a tap reversing within a short dwell with nothing having happened to
explain it.

### The metric now used

`tuning.stage2_search.hunting_reversals`: an opposite-direction pair on one tap
counts as **hunting** when the dwell is under `HUNT_DWELL_S` (2 h) **and** no
disturbance onset falls between the two moves.

The second clause is the whole content. A first attempt exempted any reversal
whose *first* move landed near an event, and reported **zero hunting on a run
containing three 21-minute limit cycles** -- because each cycle began exactly
at an event tick. Attribution must be to a *new* event arriving between the
moves, not to proximity.

Measured on the 5 h runs:

| | hunting (2 h) | hunting (1 h) | event-explained | moves |
|---|---|---|---|---|
| `stage1_start` | **0** | 0 | 2 | 22 |
| `ch9_alt_lowwear` | **0** | 0 | 1 | 15 |
| `ch9_selected` | **3** | 3 | 3 | 21 |

All three of `ch9_selected`'s cycles have a **21 min dwell** = seven
supervisory periods: the integer cooldown expires and the tap immediately
reverses. Consistent with it carrying the cheapest taps of the three
(`g_w_tso_oltc` 10.2 against 17.7 and 20.7).

So the corrected reading is the opposite of the alarming one: the baseline and
the alternative are clean, and only the *selected* point hunts. This is the
third independent line of evidence against it, after the coordinator-model
`rho(M) = 2.2127 > 2` and the DS voltage give-away.

### Changes made

* `Limits.tap_reversals_per_day` **removed**, replaced by `max_hunting = 0` and
  `hunt_dwell_h = 2.0`. Hunting is counted directly, never derived from a rate.
* `tap_moves()` extracts every committed move as `(t, transformer, direction)`
  across both layers, and the list is **persisted in each window record**. The
  first Stage II-b discarded its records after extracting metrics, which is why
  the old bank could not be re-scored under the new definition and a full
  re-simulation was needed. A few hundred bytes per window prevents that
  recurring.
* `condition_set_evented()` gives each of W1-W6 **one** disturbance -- N-1
  generator outage cleared after an hour (W1/W3/W5), sustained load ramp
  (W2/W4), or load shedding at the voltage-rise corner (W6). One event, not an
  event-dense window, so a taps/day figure stays extrapolatable and the
  event-explained branch of the metric is exercised.

### Stage II-b rerun

27 candidates (baseline + 11 Pareto-frontier + top-14 on TS + top-8 on DS,
deduplicated) drawn from the 89 Stage II-a survivors -- a wider and
deliberately two-sided shortlist, since the previous one was ranked on a quiet
bank. Stage II-a itself is **not** re-run: its rejections were on interface-Q
ratio and voltage-corridor violations, neither touched by the wear
redefinition. 162 units on the evented 12 h bank, ~3.8 h at 24 workers.
