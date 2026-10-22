# Chapter 9: the batch network carried through B, and F1 instrumented

Timestamp: 2026-09-15T15:30:00+02:00
Branch: `weights-reset-2026-09-10`
Reason: follow-on to `2026-09-15_ch9_paramset_unification.md`. The author decided the
Chapter-9 batch runs the **doubled-DSO_3** network and the **W1–W6** bank, ruled bulk
shunt switching **out of the selection's scope**, and asked for the F1 instrumentation to
be built. Acting on the network decision exposed two further defects and one open result.

## 1. Verified: the whole Stage-I/II chain ran symmetric

Asked to confirm before recommending a D rerun. Code path, not inference:

- `tuning/stage1_screen.py:822` — `base = apply_to_config(make_config(), weights)`.
- `experiments/.../ch_9_3_stage2_screen.py::_run_one_window` — `apply_paramset(make_config(), ...)`.
- `make_config()` sets no `dso_der_scale` / `dso_load_p_scale` (defaults are empty dicts).
- `ScenarioSpec.overlay_on` does not touch either field.
- The only network override anywhere in Stage I is `--network`, which replaces
  `cfg.scenario` — the benchmark name `rural_700` — not the per-DSO multipliers.

So the operating-point cache, the analytic screen, and every simulated Stage-II window are
symmetric. **D has to be re-run on the batch network.** Its aggregate metrics (e_v, taps,
hunting) are network-dependent in a way the analytic screen is not.

## 1b. Regression introduced and fixed: live plotting in every batch worker

**Author-reported, same day.** Moving the Chapter-9 lineage onto `make_config()`
(previous entry, §1.1) carried two fields I had dismissed in the diff as cosmetic:
`live_plot_controller` and `live_plot_cascade`, which `make_config()` sets **True** because
it is the *interactive* runner configuration, and which the bare-dataclass lineage had
False.

They are not cosmetic in a batch. `experiments/runners/multi_tso_dso.py:2649` constructs a
matplotlib `TSOControllerLivePlotter` per scenario, with a redraw every step — inside each
of 16 pool workers, 120 times, for nobody. The first C run carried that overhead.

It cannot have changed the result: the plotters only read state and draw, and the
simulation is deterministic and not wall-clock dependent. It cost wall time, and it left
the C leg's `resolved_config.json` differing from a future E leg in a field that is not
the weights — which the experiment plan explicitly forbids.

Fixed in two places, because there are two ways to inherit it:

- `_ch9_selected_design.build_selected_config(..., live_plots=False)` forces all six
  `live_plot_*` switches off (`_LIVE_PLOT_FIELDS`), with an opt-in for interactive
  debugging. `live_plot_show_reserves` / `_show_tie_flows` are left alone — they are
  display options of a window that will not exist.
- `tuning/runner.py::run_scenario_with_records` does the same defensively, whatever the
  caller hands in. That is the shared batch harness, so it also covers the Stage-II search
  (22 workers), which inherits the same interactive factory.

C was re-run with plotting off rather than patched in the notes, so both legs' resolved
configs differ in `--paramset` alone.

## 2. A cache key that would have silently corrupted that rerun

`tuning/stage1_screen.py::_cache_path` keyed the operating points on
`(condition, cfg.scenario)` → `op_rural_700_W1.npz`. The per-DSO multipliers are **not**
part of `cfg.scenario`. Re-running Stage I on the doubled network with the existing cache
present would have loaded the **symmetric** Gram matrices and reported `lambda_max(B)` for
a network it was not screening — and printed `cached (...)` while doing it.

Fixed: new `_network_key(cfg)` hashes the three per-DSO multiplier maps into the filename.
The symmetric case keeps the bare key, so the existing cache — which is symmetric, per §1 —
stays valid and is not needlessly rebuilt. The doubled network now lands on
`op_rural_700_net190b46e8_W*.npz`.

## 2b. `python -m tuning.stage1_screen` wrote caches only it could read

Found immediately after §2, by trying to read the new cache from an analysis script.

The operating points are pickled, and `pickle` records a class as `module.qualname`. Run
as `python -m tuning.stage1_screen`, the module *is* `__main__`, so the cache was written
with `__main__.OperatingPoint` and every other consumer — including
`ch_9_3_stage2_screen.py --ops-dir` — fails with

```
AttributeError: Can't get attribute 'OperatingPoint' on <module '__main__' ...>
```

Verified by `pickletools` on both files: the pre-existing symmetric cache references
`tuning.stage1_screen` (it was written by an importing script), the one I had just written
via `-m` referenced `__main__`.

Fixed by delegating from the guard to the module imported under its real name
(`from tuning.stage1_screen import main as _main`), which costs one extra execution of the
module's definitions and nothing else. The poisoned files were deleted and rebuilt;
`pickletools` now reports `tuning.stage1_screen` for both.

This was latent before today — nothing had run this module as `__main__` and then tried to
reuse its cache — but the new `--design` CLI would have made it routine.

## 3. `design_stage1` had no driver, and `python -m` could not reach it

`stage1_start.json` names `tuning/stage1_screen.py::design_stage1` as its producer, but
nothing in the repo called it — the analytic point was generated interactively and the
recipe was not reproducible. Worse, the module's `if __name__ == "__main__"` guard sat
*above* the Stage I-c section, so `main()` ran before `design_stage1` was defined and
`--design` raised `NameError` on a name the file plainly contains.

Fixed: the guard moved to the end of the module, and `main()` gained `--design`,
`--frozen-network`, `--write-paramset` and `--shunt-int-g-w`. `--write-paramset` refuses to
overwrite an existing file: a parameter set is the record of what a controller was, and
replacing one detaches every run that names it from the weights it actually ran.

## 4. Result: the analytic design point barely moves with the network

```
python -m tuning.stage1_screen --design --frozen-network \
    --write-paramset stage1_start_dso3x2 --out results/stage1/stage1_design_dso3x2.json
```

| weight | symmetric | doubled DSO_3 | change |
|---|---|---|---|
| `g_v`, `g_q`, `dso_g_v`, `g_w_gen` | — | — | 0 (gauge/priority-pinned) |
| `g_w_der`, `g_w_pcc` | 0.0504872682 | 0.0527090589 | **+4.40 %** |
| `g_w_dso_der` | 0.4941725967 | 0.4937369147 | −0.09 % |
| `g_w_dso_oltc` | 17.5876198285 | 17.5857790395 | −0.01 % |
| `g_w_tso_oltc` | 20.6785280181 | 20.6780293983 | −0.002 % |

42 of 42 area-condition rows at or below the screen; worst `lambda_max(B) = 1.0000`
(TS_2 at W5), the bisection target, against a screen of 1.600.

Only the TS continuous scale `kappa_TS` moves, by 4.4 %. **The Stage-II search box, which
is centred on this point, is therefore essentially unchanged by the network.** D still has
to be re-run — the selection is made on simulated 12 h metrics, not on the analytic point —
but the box does not need re-deriving and the existing shortlist is a sound starting set.

## 5. Experiment C — COMPLETE. `N_inner` is 20.5, not 9

`ch_9_1_ninner_isolated_sts.py --paramset stage1_start_dso3x2` on the conditions bank and
the frozen network. 120 cases = 5 windows (W5 excluded, zero capability) x 4 DSO x
3 interfaces x 2 directions. Run `results/THESIS_ch9_1_ninner/C_initial_weights/20260915-150812`.

| DSO | dir | steps | cap-limited | N_inner med | p95 | max | censored | taps | V viol |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_1 | down | 15 | 0 | 19.0 | 32.9 | 35 | 0.00 | 0 | 0 |
| DSO_1 | up | 15 | 0 | 18.0 | 26.6 | 28 | 0.00 | 0 | 0 |
| DSO_2 | down | 15 | 0 | 22.0 | 28.3 | 29 | 0.00 | 0 | 0 |
| DSO_2 | up | 15 | 0 | 17.0 | 27.0 | 27 | 0.00 | 0 | 0 |
| DSO_3 | down | 15 | 0 | 16.0 | 34.3 | 35 | 0.00 | 0 | 0 |
| DSO_3 | up | 15 | 0 | 21.0 | 28.5 | 32 | 0.00 | 0 | 0 |
| DSO_4 | down | 15 | 0 | 23.0 | 35.4 | 41 | 0.00 | 0 | 0 |
| DSO_4 | up | 15 | 0 | 24.0 | 26.3 | 27 | 0.00 | 0 | 0 |
| **pooled** | both | **120** | **0** | **20.5** | **33.0** | **41** | **0.00** | 0 | 0 |

**Zero censored and zero capability-limited across all 120.** These are converged loops
measured by flatness, not right-censored ones, so the result is not an artefact of the
900 s observation window or of headroom. Median headroom 46.4 Mvar against a 10 Mvar step.

At `T_STS = 20 s`, a median `N_inner` of 20.5 implies `T_TS = N_inner * T_STS` ~ 410 s and
a p95 of 660 s, against the configured 180 s. **The chapter's `N_inner = 9` is not
supported at the analytic design point.**

### E probe: the tuned weights are ~1.8x faster, and still not 9

Same case (W1/DSO_1, both directions), `--paramset ch9_selected`:

| leg | weights | N_inner |
|---|---|---|
| C | `stage1_start_dso3x2` (analytic) | 28, 29 |
| E probe | `ch9_selected` (Stage-II selection) | 15, 17 |

So tuning nearly halves the settling count — a real and reportable effect of the Stage-II
search — but does not reach 9. Full E grid running.

### Why N_inner is 20.5: it is the FLATNESS TOLERANCE, not a slow loop

Asked directly by the author ("we wanted N_inner to be 9 — why is it that much bigger?").
Answered from the run's own flatness sweep, which the script records precisely so this
choice is "visible rather than decisive".

|N_inner| is the first iteration from which `|q(k) - q(k-1)|` stays at or below
`DEFAULT_FLAT_MVAR`. Over the same 120 cases, varying only that tolerance:

| tolerance [Mvar] | % of the 10 Mvar step | % of the 1.0 Mvar settling band | median `N_inner` | p95 | unconverged |
|--:|--:|--:|--:|--:|--:|
| 0.02 | 0.2 % | 2 % | 46.0 | 46.0 | 76.7 % |
| 0.05 | 0.5 % | 5 % | 37.0 | 46.0 | 15.0 % |
| **0.10** (default) | **1.0 %** | **10 %** | **20.5** | **33.0** | 0.0 % |
| 0.25 | 2.5 % | 25 % | **6.0** | 14.0 | 0.0 % |
| 1.00 | 10 % | 100 % | 0.0 | 1.0 | 0.0 % |

**At 0.25 Mvar the answer is 6 — better than 9.** Inverting per case, `N_inner = 9`
corresponds to a tolerance of **0.211 Mvar** (IQR 0.180–0.244), i.e. 2.1 % of the step or
21 % of the settling band. All 120 cases bracket 9, so this is interpolation, not
extrapolation.

So the loop is **not** two-and-a-half times slower than the design intended. It delivers
the step quickly and then has a long geometric tail of tiny increments, and the default
tolerance sits deep in that tail. The chapter's 9 and this 20.5 are **the same loop
measured with different rulers.**

The default is also internally inconsistent with the experiment's own declarations:

- `DEFAULT_BAND_MVAR = 1.0` Mvar is the declared *tracking* settling band, "what the
  open-loop RMS battery used" — so the flatness tolerance is 10x tighter than the error
  the same experiment is willing to call settled;
- the loop converges to a **steady-state offset of 2.27 Mvar** (median) from the setpoint,
  by design, because the subordinate controller is multi-objective. Demanding
  per-iteration motion below 0.1 Mvar therefore asks the loop to resolve movement ~23x
  smaller than the offset it legitimately settles at.

#### Tested: is the tail measurement NOISE? No.

The obvious candidate for "why does it take 20 iterations to stop moving" is the
instrument: `make_config()` leaves `measurement_noise.enabled = True`
(profile `minimum`, 10 % per-sample), so the C run had noise on, and a criterion on
`|q(k) - q(k-1)|` cannot resolve below the noise floor. Added
`--no-measurement-noise` and re-ran the same case both ways:

| quantity (W1/DSO_1/trafo_0/up) | noise ON | noise OFF | ratio |
|---|--:|--:|--:|
| `N_inner` | 28 | **28** | 1.000 |
| final per-iteration change [Mvar] | 0.0499 | 0.0453 | 0.908 |
| steady-state offset [Mvar] | 2.191 | 1.818 | 0.830 |

Flatness ladder for that case, noise ON vs OFF: 46/46, 44/42, 28/28, 8/9, 0/0.

**The instrument contributes ~9 % of the residual motion and nothing to `N_inner`.** The
slow tail is a genuine property of the closed loop. Hypothesis rejected; the tolerance
explanation is not rescued or undermined by it, it simply stands on its own.

Two hypotheses tested and rejected today — `lambda_min` (predicts 10^5 iterations) and the
noise floor (changes nothing) — which is worth recording so neither is re-proposed.

**The decision, and it is the author's.** `N_inner` is not a property of the controller
alone; it is a property of the controller *and* a declared convergence tolerance, and the
tolerance was never declared alongside the 9. The noise floor cannot supply it, so it has
to come from a stated engineering basis. The natural candidates are both already declared
elsewhere in the same experiment:

| basis | tolerance | `N_inner` median | implied `T_TS` |
|---|--:|--:|--:|
| tracking band / 4 | 0.25 Mvar | 6.0 | 120 s |
| **~ band / 5** | **0.211 Mvar** | **9** | **180 s** |
| tracking band / 10 (current default) | 0.10 Mvar | 20.5 | 410 s |

Note the middle row: the tolerance that reproduces the chapter's `N_inner = 9` is
0.211 Mvar, a fifth of the declared 1.0 Mvar tracking band. That is not an implausible
engineering choice — but it must be **declared as the definition and defended on its own
terms**, not selected because it returns 9. The steady-state offset the loop converges to
is ~1.8-2.2 Mvar, so both 0.1 and 0.25 are motion far below anything that matters for the
controlled output; the choice between them is a convention, and the chapter currently has
an undeclared one.

### Secondary: the spectral reading, which the tolerance finding supersedes

First hypothesis: `z^{k+1} = (I - B) z^k`, mode `r` contracts by `|1 - lambda_r|`, the
Stage-I screen bounds only `lambda_MAX`, so the slow mode is `lambda_MIN` and settling is
unconstrained by the design.

**Measured, and it fails.** On the doubled-network operating points, the worst layer-
condition is `TS_2@W1` with `lambda_min = 7.1e-05` (analytic) and `1.5e-05` (selected),
giving `rho = max|1 - lambda| = 0.99993` / `0.99999` and predicted settling of **64 758**
and **312 062** iterations. The loop converges in 15–41. The raw spectrum is not the
predictor.

The reason is in `lambda_max_B`'s own docstring: a near-zero eigenvalue is an actuator
direction that produces **no weighted output change** — `lambda_r` *is* that direction's
output-energy gain. `N_inner` is measured on the interface flow going flat, i.e. on the
**output**, so those modes are invisible to it. They persist in state and contribute
nothing to `q_pcc`.

Restricting the spectrum to modes carrying a fraction of the peak output energy:

| threshold (of `lambda_max`) | analytic, n to 1 % | selected, n to 1 % | ratio |
|---|--:|--:|--:|
| 1e-3 | 7236.7 | 3700.9 | 1.96 |
| 1e-2 | 549.9 | 387.3 | 1.42 |
| 0.05 | 143.4 | 80.8 | 1.77 |
| 0.1 | 73.8 | 45.3 | 1.63 |
| **0.2** | **25.8** | **15.5** | **1.67** |
| 0.5 | 7.2 | 3.9 | 1.83 |
| *measured* | *28, 29* | *15, 17* | *~1.8* |

At a 0.2 cut-off the predicted counts are 25.8 and 15.5 against measured 28–29 and 15–17.

**Be careful how this is quoted.** The threshold was chosen to match, so the absolute
agreement is one fitted parameter against two numbers and is NOT evidence. What *is*
evidence is the **ratio**: the same threshold applies to both legs, the predicted ratio
sits at 1.4–2.0 across two decades of threshold, and the measured ratio is ~1.8. So the
mechanism — tuning speeds the slowest *observable* mode by roughly 1.7x — is supported
independently of where the cut-off is placed, while the absolute count is not yet
predicted from first principles.

The consequence for the chapter stands either way: `N_inner = 9` was never a consequence
of the weight design. The Stage-I screen constrains the fastest mode for stability and
says nothing about the slowest observable one.

**But this is the secondary effect.** The tolerance ladder above explains the bulk of the
gap between 9 and 20.5 directly and without a fitted parameter, and it should be the
chapter's explanation. The spectral reading adds only that the *ratio* between the
analytic and tuned legs (~1.8x) is a real property of the weights rather than of the
ruler — the tolerance is identical in both legs, so it cannot produce that ratio.

## 5a. Experiment E -- COMPLETE. Tuning tightens the TAIL, not the median

`--paramset ch9_selected`, conditions bank, frozen network, 120 cases,
`results/THESIS_ch9_1_ninner/E_final_weights/20260915-174510`.
0 capability-limited, 0 censored at the default tolerance.

| tolerance [Mvar] | C median | **E median** | C p95 | **E p95** | C max | **E max** |
|--:|--:|--:|--:|--:|--:|--:|
| 0.02 | 46.0 | 35.0 | 46.0 | 46.0 | 46 | 46 |
| 0.05 | 37.0 | 18.0 | 46.0 | 32.1 | 46 | 46 |
| 0.10 | 20.5 | **13.0** | 33.0 | **20.0** | 41 | 29 |
| 0.25 | 6.0 | 7.0 | 14.0 | **11.0** | 16 | 12 |
| 1.00 | 0.0 | 1.0 | 1.0 | 2.0 | 1 | 3 |

Two things worth reporting:

* **At the default 0.1 Mvar the selected weights are clearly faster** -- median 13.0 vs
  20.5, p95 20.0 vs 33.0, max 29 vs 41.
* **At 0.25 Mvar E's median is slightly WORSE than C's** (7.0 vs 6.0) while its p95 is
  markedly better (11.0 vs 14.0) and its max much better (12 vs 16). So Stage-II tuning did
  not make the typical step faster at a loose criterion -- it **compressed the
  distribution**, which is the property a dispatch cadence actually needs. That is a
  better result than "it is faster", and it is the honest way to state it.

### Consequence for the cadence, and for launching D

`T_TS = 180 s` is `N_inner = 9`. Against E:

* at 0.10 Mvar, p95 = 20.0 -> 400 s. Not supported.
* at 0.25 Mvar, p95 = 11.0 -> 220 s. Not supported.
* at the adopted **0.5 Mvar**, the value is not measured -- it falls in the
  0.25-1.0 gap of the old ladder, between p95 11.0 and 2.0.

The pre-declared one-sided test (`p95(0.25) <= 9`, which would imply `p95(0.5) <= 9` by
monotonicity without interpolating) **fails at 11.0**. Log-interpolation suggests
`p95(0.5) ~ 6.5`, and the segment slopes (-9.8 per ln-unit over 0.1-0.25 against -6.5 over
0.25-1.0) indicate saturation, so 6.5 is if anything an over-estimate. But that is the
exact gap shown unreliable earlier, so D was **not** launched on it.

Instead E was re-run as `E_final_ladder` with the extended `FLAT_SWEEP_MVAR`
(0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0) **and `--save-trajectories`**, which
measures 0.5 directly and makes every future tolerance question answerable offline without
re-simulating. ~2 h, against the 9 h a wrongly-cadenced D would cost.

## 5a3. The measured ladder: `T_TS = 180 s` is supported, and D launched

`E_final_ladder` -- the same 120 cases at `ch9_selected`, re-run with the extended
`FLAT_SWEEP_MVAR` and `--save-trajectories`. Run
`results/THESIS_ch9_1_ninner/E_final_ladder/20260915-213349`.

| tolerance [Mvar] | median | **p95** | max | unconverged | `T_TS` at p95 |
|--:|--:|--:|--:|--:|--:|
| 0.02 | 35.0 | 46.0 | 46 | 21.7 % | 920 s |
| 0.05 | 18.0 | 32.1 | 46 | 2.5 % | 642 s |
| 0.10 | 13.0 | 20.0 | 29 | 0 % | 400 s |
| 0.15 | 10.5 | 16.0 | 17 | 0 % | 320 s |
| 0.25 | 7.0 | 11.0 | 12 | 0 % | 220 s |
| 0.35 | 5.0 | 8.0 | 8 | 0 % | 160 s |
| **0.50** | **3.0** | **6.0** | 6 | 0 % | **120 s** |
| 0.70 | 2.0 | 4.0 | 4 | 0 % | 80 s |
| 1.00 | 1.0 | 2.0 | 3 | 0 % | 40 s |

**p95 = 6.0 at the adopted 0.5 Mvar**, i.e. `T_TS` at p95 = 120 s against the configured
180 s -- supported with 50 % margin. `p95 = 9` is crossed near 0.30 Mvar.

The earlier log-interpolation across the old 0.25-1.0 gap predicted 6.5 against a measured
6.0, so it was slightly conservative and would not have misled here. The re-run was still
the right call: 2 h to remove the doubt, against 9 h of D at a cadence that might not have
held. And the extended ladder plus saved trajectories mean no future tolerance question
needs a re-simulation.

### Cadence, settled

Both metrics now point the same way at `T_TS = 180 s`:

* `N_inner` p95 = 6.0 at the declared 0.5 Mvar tolerance -> 120 s, so 180 s covers it;
* `rho` is flat across 180-360 s, so no longer period earns its latency, and longer
  periods cost censoring (0.11 -> 0.15) and tap wear.

**D launched** on that basis: the archived 27-candidate shortlist re-simulated on the
batch network, new output tree `results/THESIS_ch9_3_stage2_dso3x2/refine_evented`, so the
symmetric archive stays intact as the historical record.

## 5a2. Cadence sweep -- COMPLETE. Lengthening `T_TS` buys nothing

`ch_9_1_ts_period_sweep --paramset ch9_selected`, conditions bank, frozen network,
4 periods x 6 windows, **12 h per window** (the conditions bank defaults to `REFINE_H = 12`
and this script, unlike the ninner one, does not override `duration_s` -- so the bank
switch also changed the window length 8x, from the 90 min `tier1_design_set` windows).
Run `results/THESIS_ch9_1_ts_period_sweep/D_cadence/20260915-210400`.

`rho_k = r_k / delta_k` is the fraction of the commanded correction still outstanding when
the next dispatch lands, scored only on intervals where the supervisor moved at least
1 Mvar.

| `T_TS` | `N_cfg` | scored | rho med | rho p95 | n_k med | n_k p95 | censored | taps/ival | V viol |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **180** | 9 | 1021 | **0.813** | 3.110 | 0.0 | 9.0 | **0.11** | 0.003 | 0 |
| 200 | 10 | 975 | 0.806 | 2.466 | 0.0 | 10.0 | 0.12 | 0.003 | 0 |
| 240 | 12 | 981 | 0.802 | 2.335 | 0.0 | 12.0 | 0.12 | 0.004 | 0 |
| 360 | 18 | 879 | 0.787 | 2.250 | 0.0 | 18.0 | **0.15** | 0.006 | 0 |

Three readings, and the first is the one that matters:

1. **rho is flat.** Doubling the period from 180 s to 360 s moves the median from 0.813 to
   0.787 -- a 3 % improvement for twice the latency. **There is no case for raising
   `T_TS` on this evidence.** The "N_inner p95 says go to 400 s" argument is not supported
   in closed loop.
2. **Censoring gets WORSE with a longer period** (0.11 -> 0.15), and `n_k` p95 sits exactly
   on the censoring limit (9, 10, 12, 18) at every period -- the p95 interval never settles
   at any cadence. A longer interval accumulates a larger supervisory command, so the extra
   time does not buy convergence.
3. **DSO_2 is markedly better than its siblings** (rho med 0.51-0.65 against 0.77-0.85),
   i.e. the residual is not uniform across areas.

### Why rho ~ 0.8 does not contradict `N_inner` ~ 13

They measure different things and the gap has a mechanism. The subordinate controller is
multi-objective, so it converges to an optimum displaced from the commanded setpoint --
measured in C as a **2.27 Mvar median steady-state offset**. The sweep scores intervals
whose command is only just above the 1 Mvar floor, so a roughly fixed ~2 Mvar offset is
~100 % of a small command: hence `rho` medians near 0.8 and a p95 above 1 (residual larger
than the command). `rho` is therefore dominated by the **offset-to-command ratio**, not by
incomplete settling -- which is exactly why more time does not reduce it.

So this is not the contradiction that would block D. It does not show poor delivery *at
180 s specifically*; it shows the same residual at every period, and it removes "raise
`T_TS`" from the options rather than casting doubt on 180 s. The script's own caveat is
kept in view: `rho` cannot see whether a correction was still the right one when it landed,
so the absence of a U-shape is not evidence against one.

**Net cadence position:** `T_TS = 180 s` stands, selected because no longer period improves
subordinate delivery and longer periods cost censoring and tap wear -- with the `N_inner`
criterion resting on a declared 0.5 Mvar tolerance, pending the measured p95.

## 5d. Experiment D re-run on the batch network -- COMPLETE, selection UNCHANGED

`ch_9_3_stage2_screen --paramset stage1_start_dso3x2 --frozen-network --hours 12 --evented
--candidates-json <archived shortlist> --workers 22`, output tree
`results/THESIS_ch9_3_stage2_dso3x2/refine_evented`. **168/168 units, 0 failed, 5.42 h**
(against an 8.6 h estimate). The symmetric archive is untouched.

**The selection rule picks `399e30d3b60b` again -- the same candidate as `ch9_selected`.**
7 of 28 feasible; 21 rejected, all but three for hunting.

| rank | key | lambda | e_v(TS) | e_v(DS) | e_q | taps/d TS | taps/d DS | hunt |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| **1** | **399e30d3b60b** | 1.310 | **0.00146** | 0.00518 | 120277 | 4.0 | 4.0 | 0 |
| 2 | 874818ae70a4 (symmetric analytic) | 1.043 | 0.00200 | 0.00547 | 914951 | 4.0 | 6.0 | 0 |
| 3 | 859b46e0ea11 (**batch analytic**, baseline) | 1.000 | 0.00204 | 0.00547 | 904709 | 4.0 | 6.0 | 0 |

Selected: e_v(TS) -28.6 % and e_v(DS) -5.4 % against the baseline, `lambda_max(B)` 1.3104.

Two things worth taking from the table:

* **The two analytic points score almost identically** (e_v(TS) 0.00200 symmetric vs
  0.00204 batch), confirming from simulation what section 4 showed analytically: the 4.4 %
  move in `kappa_TS` is immaterial.
* **The network does change the metrics, in the expected direction.** Same weights,
  symmetric vs batch: e_v(TS) 0.001651 -> 0.00146 (better), e_v(DS) 0.004851 -> 0.00518
  (worse). Doubling DSO_3's DER and active load stresses the distribution side. So the
  rerun was necessary even though its verdict did not change -- the *numbers* the chapter
  quotes are different, only the *choice* is the same.

`selection_report.json` written to the new tree.

### Shunt activity, now recorded (section 6's scope statement, with numbers)

| window | commits | reversals | max level | taps/h TS | taps/h DS | n_hunting |
|---|--:|--:|--:|--:|--:|--:|
| W1 | 8 | 2 | 2 | 0.167 | 0.083 | 0 |
| W2 | 12 | 6 | 2 | 0.083 | 0.000 | 0 |
| W3 | 17 | 10 | 2 | 0.167 | 0.083 | 0 |
| **W4** (ranked) | **63** | **53** | 2 | 0.000 | 0.167 | 0 |
| W5 (excluded) | 96 | 66 | 2 | 1.084 | 1.251 | 19 |
| W6 | 14 | 7 | 2 | 0.083 | 0.000 | 0 |
| **total / 72 h** | **210** | **144** | | | | |

**70 commits/day and 48 reversals/day** pooled, against a 30 ops/day tap budget. Compare
the symmetric `shunt_live` measurement at the same weights: 213 / 139 -- essentially
identical, so shunt behaviour is as network-insensitive as the screen.

W4 is a *ranked* window and worsened on the batch network: 63/53 against 49/32 symmetric,
i.e. 126 commits/day and 106 reversals/day. The scope decision of section 6 stands, but
the chapter must quote these numbers when it states that bulk shunt switching was reported
and not constrained.

## 5b. Experiment A re-scored: a 0.5 Mvar interface-Q floor is admissible

Prompted by the author's proposal to declare **one** resolution for interface reactive
power and use it both as the Experiment-A settling floor and as the C/E flatness
tolerance. Archive found at
`Z:\Python_Projekte\qOFO_GH\results\THESIS_Ch9_A_plant_settling\phase1` (campaign
`THESIS_ch9_PF_batch_2026-09-08`, commit `e2b82a3`): `signal_metrics.csv` over W1-W6 plus
the 70-command W1_full sweep, 10 962 per-signal rows. **No PowerFactory needed** --
`settling_metrics` is post-processing.

**Scope correction from the author, and it decides the answer.** Interface reactive power
means the **coupling transformers** (`Q_DSO_NC3W_*`), not the TS tie lines
(`Q_TSO_Z1_Z2_TN_line*`). Scoring the tie lines as interface Q -- which I did first --
produces a spurious alarm.

Realisable dispatches only (the +-0.02 pu AVR cases excluded: they are ~90x the measured
`worst_avr_max_step_pu` of 0.000226 and 4x the 0.005 limit):

| signal set | tolerance | max settling |
|---|--:|--:|
| coupling-trafo Q | 1.0 Mvar | **8.61 s** |
| coupling-trafo Q | 0.5 Mvar (estimated) | 10.56 s |
| voltage, TN **and** DSO buses | 1e-3 pu | **12.96 s** -- unaffected by the Q floor |
| tie-line Q (out of scope) | 1.0 Mvar | 17.71 s |

So the binding row is a **voltage** at 12.96 s and does not move with the Q floor. Adopting
0.5 Mvar leaves the binding row at 12.96 s and the margin at `T_STS = 20 s` at **7.04 s**.
For coupler-Q to become binding it would have to rise 4.35 s, needing `tau > 6.3 s` against
2.82 s in the data; to breach 20 s, `tau > 16.4 s`. Neither is close.

**Verdict: a 0.5 Mvar interface-Q resolution is admissible in A.** Justified independently
of any desired `N_inner` as ~10x the measured measurement-chain resolution (0.045-0.050
Mvar, from the noise-on/off comparison in section 5).

Write it as one declared *resolution* with two roles, not one tolerance: in A it bounds the
**error envelope**, in C/E the **per-iteration increment**, and at the measured
`rho ~ 0.92` the same number is a far weaker condition on the increment than on the error.

### Two numbers the chapter must reconcile

1. Chapter 9.1 prints binding 16.28 s / margin 3.72 s at `u_DSO_1_bus43`, from the
   2026-08-19 handoff. This campaign gives **12.96 s / 7.04 s** at `V_TN_bus1`. Both are
   voltage rows; different signals, different campaigns. The table must name its run.
2. The tie-line row (17.71 s) is in `signal_metrics.csv`. Since it is excluded from the
   interface definition, the chapter has to say so -- otherwise a reader recomputing the
   binding row from the archive gets 17.71 s and a 2.29 s margin.

### What could not be computed

The binding case's coupler-Q trajectory is not archived: the run saves **one** trajectory
per case and for `tap_p1_MT_g0_t0` it saved the worst *voltage* signal, `V_TN_bus1`. The
10.56 s figure is therefore an estimate from a population fit whose correlation is
`r = -0.071` -- it must not be quoted as measured. It does not affect the verdict, because
the margin to the 12.96 s binding row is large. If A is ever re-run, save every controlled
signal's trajectory.

## 5c. The D rerun: prepared, and much cheaper than feared

`ch_9_3_stage2_screen.py` does not go through `build_selected_config`, so it was still
symmetric-only after the 2026-09-15 unification. Two changes make the D rerun possible:

1. **New `--frozen-network`**, applied to *both* the analytic screen and every simulated
   window. The second half matters: `ScenarioSpec.overlay_on` does not touch the per-DSO
   multipliers, so without it a window would run symmetric while the screen that admitted
   the candidate ran doubled.
2. **The operating-point loader no longer globs.** `ops = [pickle.load(...) for p_ in
   sorted(Path(args.ops_dir).glob("op_*.npz"))]` was correct only while one network
   existed on disk. Since section 2 the same directory holds symmetric *and*
   doubled-DSO_3 points, and that glob would have screened a **12-point mixture of two
   networks**. It now calls `load_or_build_operating_points`, reusing the network keying
   rather than restating it. This is a trap I created in section 2 and closed here.

### Measured: the analytic screen is network-invariant

Same 512 Sobol draws, same seed, both networks:

| | survivors of 512 | `lambda_max(B)` median | max |
|---|--:|--:|--:|
| symmetric | **116** (22.5 %) | 4.5445 | 175.0007 |
| doubled DSO_3 | **116** (22.5 %) | 4.5439 | 175.0267 |

Per-draw difference: median 0.00064, max 0.0687 (3.48 % relative). **Zero draws change
side of the 1.5459 selection ceiling.**

Consequence: **the shortlist transfers.** The D rerun does not need to re-screen or
re-shortlist -- what depends on the network is the simulated 12 h wear and tracking
metrics, not the contraction screen. So the rerun is the archived 27-candidate shortlist
re-simulated on the batch network (~8-9 h at 22 workers), not a fresh 116-candidate sweep
(~28 h).

This is consistent with section 4: the analytic design point moved only 4.4 % on
`kappa_TS` and under 0.1 % elsewhere. The curvature the screen reads is nearly unchanged
by doubling one DSO's DER and active load.

**Not launched.** The gate holds: `T_TS` is unsettled until the cadence sweep and the
tolerance decision land, and D re-run at a cadence that then changes is 9 h wasted.

## 6. Shunt switching: out of the selection's scope (author's decision)

Recorded in `tuning/stage2_search.py::Limits`, where the omission lives, with the measured
rates alongside it so the scope statement cannot be read as an oversight: at the selected
weights over 12 h, W4 (a *ranked* window) shows 49 commits / 32 reversals and W5 shows
112 / 80, against a 30 ops/day tap budget. The bulk banks are dispatched by an integrator
outside the MIQP on sustained need, with their own dwell guard and daily budget, so their
switching is not what the nine MIQP weights are selected to govern. The thesis must state
that it was reported and not constrained.

## 7. `ch9_alt_lowwear` deleted

`fields` was byte-identical to `ch9_selected.json`; verified by diff before removal. No
code loaded it. The remaining references are historical text (`ch9_oltc_led.json`'s
`derived_from`, two daily logs, one archived `run_meta.json`) and stay valid as records.

## 8. F1 instrumentation — built

### 8.1 Everything needed was already inside the DSL block

`pf/wecc_apply.py::QVPRE_EQUATIONS` computes, per park:

```
x1    = Tf-filtered terminal voltage        <- the dead-zone INPUT
veff  = x1 - Vanchor                        <- |U_filtered - U_anchor|, exactly
qcorr = deadzone(veff, db)                  <- zero iff inside the band
qraw  = qset - Kdroop*qcorr                 <- UNSATURATED droop command
Qext  = clip(qraw, qmin, qmax)              <- SATURATED reference
```

So F1's measurement was an **export** gap, not a modelling one — `veff` is literally the
quantity the experiment asks for, and the plant has been computing it all along.

### 8.2 Changes

| file | change |
|---|---|
| `pf/screening.py` | `QV_INTERNAL_SIGNALS` (six signals) and `QV_INTERNAL_LABEL_PREFIXES`; `monitored_outputs(..., include_qv_internals=False)` registers them per park. |
| `configs/config.py` | `rms_export_qv_internals: bool = False`. |
| `pf/plant.py` | passes it to `monitored_outputs`; **raises** if the flag is set and no QVPRE channel was registered, rather than producing a successful-looking run that recorded nothing. |
| `pf/replay.py` | `export_qv_internals` on the factory. |
| `experiments/helpers/rms_cosim_config.py` | `--qv-internals`. |
| `run_rms_cosim.py`, `run_comparison_rms_cosim_qss.py` | harvest and write `csv/rms_qv_internals.csv`; raise if the flag was set and the export came back empty. |
| `analysis/deadband_anchor.py` | **new** — the F1 measurement itself. |

The channels go to their **own** file. `analysis.deadband_n1`, `deadband_activity`,
`deadband_threshold` and `deadband_2d` all read `csv/rms_der_raw.csv` and select on its
label set; widening it would silently change what every one of them measures.

### 8.3 A monitor that was registered and never exported

`s:qset` has been a registered monitor since the DER-Q plot page needed it, but no entry
point ever wrote it out: both harvest filters admit `qDER_`, and
`"qsetDER_x".startswith("qDER_")` is `False`. It went into the PF result file and out of
every CSV. It now travels with the dead-band group, because the incremental droop command
is `qraw - qset` and without the setpoint the command cannot be separated from the
operating point it was issued around.

### 8.4 What `analysis/deadband_anchor.py` measures

`--mode reference` — per park, the actual anchor-update instants (changes in `Vanchor`,
which is written by pooled `EvtParam` events, so the trace shows when the anchor *moved*,
not when a nominal 20 s / 180 s grid says it should have); the maximum `|veff|` within
each inter-update interval; the pooled maximum over parks and runs; and
`select_half_width()` = that maximum plus a **declared** margin frozen before candidates
are compared. Parks whose anchor never moved are flagged rather than silently treated as
one interval.

`--mode confirm` — dead-zone activation count, `max|qcorr|`, and the incremental command
**twice**: `qraw - qset` (what the droop asked for) and `qext - qset` (what survived the
capability clip). A park at a limit has the second at zero with the first large, and
reporting only the second would score an actively-commanding dead zone as quiet. Such
parks are listed as `zero_headroom` separately.

`--startup-exclude-s` is applied before any reduction, and is recorded in the summary.

Self-test: 15/15 PASS, including the three that matter — a saturated park still fails the
confirmation, it is reported as zero-headroom rather than quiet, and the startup exclusion
actually moves the answer (the fixture is built with decaying per-interval drift, because
with a flat sawtooth the assertion would pass for the wrong reason).

## 9. Verification

- `analysis.deadband_anchor --self-test` — 15/15 PASS.
- `tuning.stage1_screen --paramset stage1_start --self-test` — OK (Gram factorisation
  exact, gauge-invariant, reads the coordinator's own curvature rows).
- `py_compile` on all nine touched modules — clean.
- `run_rms_cosim --help` shows `--qv-internals`.
- Stage-I design run on the doubled network: 42/42 rows under the screen.

**Not verified:** the six `s:<name>` monitor strings against a live PowerFactory. `s:qset`
on the same block is long-standing precedent, so the convention is right, but the first
run on machine P must confirm that `csv/rms_qv_internals.csv` is non-empty and that
`veffDER_* == vfiltDER_* - vanchorDER_*` holds sample-wise. The code raises rather than
returning empty at three points, so a wrong name fails loudly instead of yielding a run
that looks fine and measured nothing.

## 10. Risks / unresolved

1. **D must be re-run on the doubled network** (§1). The box centre is essentially
   unchanged (§4), so the shortlist is reusable, but the 12 h metrics are not.
2. **`N_inner` is 15–41, not 9** (§5), on 24 of 120 cases at the analytic point. If the
   full grid holds, the chapter's cadence is not supported by its own criterion and
   `T_TS` or the weights have to move. Measure `lambda_min` before theorising further.
3. **The `s:` monitor names are unverified against a live PF** (§9).
4. **W5's exclusion still needs restating** for the shunt-live configuration — carried
   over from the previous entry, unaffected by today's decisions.
5. The coverage statistic changes population with the bank. 227/240 = 94.6 % belongs to
   the twelve-window design bank; a figure from the 120-case W1–W6 bank is a different
   measurement and the two must not be quoted as a trend.
