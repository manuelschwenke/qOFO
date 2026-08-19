# Stage-1 re-run runbook (prepared 2026-08-18)

Why this re-run exists, what is already wired, what still needs a decision, and
the exact commands. Background and all measurements:
`docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md`.

## 0. Why

The tuned weights were found under a filter whose criteria are `f_ts` (TS
voltage) and `f_q` (interface Q). The subordinate layer's own voltage is a
*reported diagnostic*, "never optimised directly"
(`tuning_mc/metrics.py`). Consequence, measured: DSO 4 rides the 1.10 p.u.
bound for 58 % of a 6 h day and DSO 2 for 7.3 %, while every hard constraint is
satisfied — g2 corridor excess is 4.26e-5 against a 1e-4 limit, i.e. **feasible**.
The fix (per-area voltage authority) is *rejected* by the current filter: the
baseline dominates it on both criteria.

So the objective has to change before a re-run means anything.

## 1. Already implemented and tested

| piece | where | test |
| --- | --- | --- |
| guard-band DS metric | `tuning/metrics.py` — `DS_GUARD_HEADROOM_PU`, `TrajectoryMetrics.guard_deficit_ds_pu`, `.ds_headroom_min_pu` | `tests/tuning/test_stage1_archive_rescorable.py` |
| criterion switch | `tuning_mc/metrics.py` — `score_candidate(..., ds_criterion=)` | same |
| full metric vector archived | `tuning_mc/metrics.py` — `per_scenario[...]["metrics"]` | same (coverage-based) |
| `--ds-criterion` CLI, forwarded to workers | `tuning_mc/stage_1_search.py` | — |
| **scoring fingerprint** | `tuning_mc/stage_1_search.scoring_fingerprint`, stamped + checked in the cache branch | `tests/tuning/test_stage1_rerun_guards.py` |
| per-DSO relief, loop-gain safe under a searched `dso_g_v` | `configs/config.apply_dso_v_relief`, called last in `build_config` | same |
| pre-flight | `tuning_mc/preflight_rerun.py` | — |

Two traps were found and closed while preparing this; both are the same shape as
the ones `bank_fingerprint` / `stage0_fingerprint` already document.

* **The objective change would not have invalidated the cache.** It touches only
  the two metric modules; the cache validates bank and design weights, neither
  of which notices. Archived rows would have been replayed with their old
  `f_ds` mixed into a new front. Closed by `scoring_fingerprint`, which hashes
  both metric sources *and* the criterion, and treats an unstamped row as stale.
* **`--limits` is mandatory, and its omission was invisible.** Found by
  launching without it, 2026-08-18: `load_limits(None)` returns
  `ConstraintLimits()` defaults with `rho_emp_p95 = 1.0`, against the tier-1
  file's 1.5. Measured rho at the design point is **1.4357**, so every candidate
  came back `feasible=False` on g3 -- and `filter_accepts` rejects infeasible
  candidates outright, so the filter would have stayed empty for the entire ~9 h
  campaign while every individual number looked plausible. Worse, the limits sat
  in *no* fingerprint, so relaunching with the right file would have replayed
  the cached rows and kept their stale `feasible`. Closed twice over:
  `scoring_fingerprint` now hashes the resolved limits (verified: the 9 rows
  from the bad launch were detected and moved to `*.json.scoring_changed`), and
  pre-flight check 1b **blocks** when `--limits` is absent.
* **A fixed relief would have drifted the OLTC loop gain.** `dso_g_v_ratio` is a
  search coordinate, so `dso_g_v` moves per trial; absolute per-area numbers
  would change `dso_g_v / g_w_dso_oltc` as the search walked, and that ratio is
  what keeps the integer tap out of a limit cycle (50.5 reversals/h when broken,
  0.00 at baseline). Closed by applying the relief **last** in `build_config`,
  derived from the searched `dso_g_v` and the designed `dso_oltc`. Verified
  invariant to 0.00e+00 across `dso_g_v_ratio` in [0.25, 4].

## 2. Still needs a decision — how the relief enters the search

`tuning_mc.stage_1_search.DSO_V_RELIEF_FACTORS` defaults to `{}`, which
reproduces every earlier campaign. Three options:

**A — fixed relief, guard criterion as a ratchet.** Set
`DSO_V_RELIEF_FACTORS = {"DSO_2": 20.0, "DSO_4": 20.0}` (or pass
`--relief`). Stage 1 tunes the shared weights around a plant that already has
the relief. The guard criterion is not vacuous here: it still discriminates on
DSO 1 / DSO 3 (unrelieved, headroom +0.048 / +0.039) and on any candidate whose
other knobs erode margin. Cheapest; no new search direction. **Recommended for
this re-run.**

**B — relief as a search coordinate. IMPLEMENTED.** `dso_v_authority`, **one
shared coordinate** across `DSO_V_RELIEF_AREAS = ("DSO_2", "DSO_4")`, bounds
`(1.0, 100.0)`. Lower bound 1.0 is "no relief", so the incumbent can always walk
back to the unrelieved plant; the upper bound is where the coordinate stops
meaning what it says (at x20 the DER's voltage gradient is still ~18:1 below its
Q-tracking gradient, so the DSO remains a Q tracker that also shapes voltage; at
x100 that margin is ~3.6:1 and the roles start to invert).

Gated exactly like `lambda_tso_z*`: in `BOUNDS` so it is addressable, **not** in
`X0`, so every earlier campaign reproduces byte-for-byte. Enable with
`--search-dso-v-authority [START]` (default 20, the measured operating point).

*One* coordinate, not one per area, because **phase B is a compass search**:
each live direction costs **two evaluations (+/- delta) on every poll**, so a
per-area split doubles the marginal cost to separate two areas that both
measured a factor near 20. If they need to differ, derive the ratio from the
`reach x |Q_PCC|` predictor (daily log section 8) rather than searching it.

Cost: phase A probes it at `--probe-multipliers` (default 4 values) = **+4
evaluations**; phase B adds **+2 per poll** while it stays live. At ~1 h/trial
serial over 6 workers that is roughly +40 min in A and +20 min per poll in B.

**C — derive it in Stage 0.** The factor is predictable from
`reach x |Q_PCC|` (r = +0.996 over the four areas, correct ranking); see §8 of
the daily log. Most defensible for the thesis and needs no Stage-1 dimension,
but the predictor is fitted on one topology at four scalings and the duty is an
outcome, so it needs the two-pass or admissible-duty treatment first.

Chosen: **B**. A remains available (leave `--search-dso-v-authority` off and set
`DSO_V_RELIEF_FACTORS`); C stays the eventual thesis answer, and B's result is
the thing to check it against — if the search settles near the factor the
predictor implies, that is evidence for the Stage-0 rule rather than a
coincidence.

## 3. Pre-flight (always)

```bash
F:/python_environments/qOFO_clean/python.exe -m tuning_mc.preflight_rerun --ds-criterion guard --filter-ds --relief DSO_2=20,DSO_4=20 --scenario-set tier1 --limits tuning_mc/configs/limits_mc_v2_tier1.json
```

Exit 0 = clear to launch, 1 = blocking. It checks the objective is plumbed *and
in the dominance test*, the cache will invalidate correctly, the loop-gain
invariant holds across the search range, and the archive is re-scorable.

## 4. Launch

Phase A (identifiability probe), then Phase B (compass search). `--workers`
defaults to 6 — measured throughput peaks there and regresses past 8
(memory-bandwidth bound); the 12-core budget is a *machine* cap, not a worker
count.

```bash
F:/python_environments/qOFO_clean/python.exe -m tuning_mc.stage_1_search --phase a --scenario-set tier1 --ds-criterion guard --filter-ds --limits tuning_mc/configs/limits_mc_v2_tier1.json --search-dso-v-authority 20 --workers 18
```

```bash
F:/python_environments/qOFO_clean/python.exe -m tuning_mc.stage_1_search --phase b --scenario-set tier1 --ds-criterion guard --filter-ds --limits tuning_mc/configs/limits_mc_v2_tier1.json --search-dso-v-authority 20 --workers 18
```

Phase A will report `dso_v_authority` as live or dead. If it comes back **dead**
that is a result, not a failure: it would mean the guard criterion does not
respond to the factor on this bank, and the honest reading is that the areas
need the relief but the *search* cannot see the difference — fall back to A or C
rather than promoting a direction with no signal.

Set the BLAS thread vars before launching, and do not background the stages —
`TaskStop` orphans the child (see the server-core-budget note).

## 5. Cost

| bank | windows | sim-h/trial | serial h/trial | ~110 trials, 12 cores |
| --- | --- | --- | --- | --- |
| `tier1` (design) | 12 | 18.0 | ~1.0 | **~9 h wall** |
| `confirm` | 9 | 13.5 | ~0.7 | ~6 h wall |
| `audit` (tier 2) | 4 | 48.0 | ~2.6 | shortlist only |

At the measured 3.2 min/sim-h.

## 6. After

1. `--scenario-set confirm` on the shortlist.
2. `--scenario-set audit` (tier 2) on the finalists — the Tier-2 budget is what
   kills Tier-1 candidates, so this is not optional.
3. Re-check transfer: a single-window optimum does not transfer
   (`docs/tuning/METHOD_weight_selection.md`). Compare tier1 vs confirm ranking
   before pinning anything.
4. Re-pin the baseline and regenerate `make_config_per_area`'s per-area block if
   Stage 0 moved.

## 7. Open items carried into the re-run

* `DS_GUARD_HEADROOM_PU = 0.02` is design intent, not a calibration — it cannot
  be derived via `ConstraintLimits.from_reference` because the reference itself
  has negative headroom on two of four areas. Justify it as a planning margin.
* The factor 20 is a round number from two measurements (6.7, 20) on one 6 h
  window. Under option A it is an input, not a result — say so in the write-up.
* A `g6_ds_headroom` **hard** constraint was discussed but not added; the guard
  currently enters as a filter criterion only. Adding it would need a limit set
  from design intent for the same reason as above.
* **Pre-existing test failure, unrelated to this work but visible in any full
  run:** `tests/tuning/stability_certificate/test_hierarchy.py::
  test_default_factory_reads_run_multi_system_ofo_parameters` asserts the
  *pre-fold* weights of `make_config()`. It has been failing since the
  2026-08-13 change that folded the uniform `zone_g_w_scale = 0.3` into the
  weight literals (the file's own comments read `# 50 x 0.3` etc.). Verified
  identical at `HEAD` and in the working tree, so it is not caused by the
  re-run preparation.

  | field | test wants | `make_config()` has | ratio |
  | --- | --- | --- | --- |
  | `g_w_der` | 50 | 15 | 0.30 |
  | `g_w_pcc` | 200 | 50 | 0.25 |
  | `g_w_gen` | 5e9 | 1e9 | 0.20 |
  | `g_w_dso_der` | 1000 | 800 | 0.80 |
  | `g_w_dso_oltc` | 150 | 200 | 1.33 |

  Note the ratios are **not** a uniform 0.3, so this is not a pure gauge
  re-expression and the test cannot be fixed by scaling its expectations by a
  constant — someone has to decide whether the certificate should track
  `make_config()` at all. Left alone deliberately; not in scope here.
