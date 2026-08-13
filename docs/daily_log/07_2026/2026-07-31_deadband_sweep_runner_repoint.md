# 2026-07-31 — Dead-band sweep repointed after the RMS runner split

**Timestamp:** 2026-07-31, 12:55–13:00
**Scope:** one-line entry-point fix in `experiments/run_deadband_sweep.ps1`.
No controller, plant or analysis behaviour changed.

---

## 1. Symptom

The dead-band selection sweep (`docs/deadband_selection_rerun_handover.md`) was
launched and aborted on run 1 after 19 s:

```
F:\python_environments\qOFO_clean\python.exe: No module named
experiments.run_rms_phase6_replay
```

The script's fail-fast worked as designed — the abort cost 19 s, not the
7 h the full matrix would have taken.

## 2. Cause

`experiments/run_rms_phase6_replay.py` was split earlier the same day
(recorded as an observation in
`docs/daily_log/07_2026/2026-07-31_remove_dso_q_integral_and_pf_probes_subpackage.md`
§4.4, made by a concurrent session in the same working tree):

| new module | what it does | run dir | `runner_static` |
|---|---|---|---|
| `run_comparison_rms_cosim_qss.py` | QSS static leg **and** RMS leg, then compares | `results/rms_phase6_replay/` | populated |
| `run_rms_cosim.py` | RMS leg only, no QSS reference | `results/rms_cosim/` | `None` |
| `run_openloop_qss_to_rms.py` | the genuine open-loop `u → y` replay | — | — |
| `archived/run_rms_phase6_replay.py` | deprecation shim → comparison script | — | — |

The old name was a misnomer: it never replayed anything.

## 3. Method of change

`experiments/run_deadband_sweep.ps1`: the `-m` target became
`experiments.run_comparison_rms_cosim_qss`, with a comment recording why the
other entry point is wrong.

**`run_rms_cosim` would have silently produced an unusable matrix.** It is the
tempting choice — same flags, roughly half the wall time — but two of its
properties are disqualifying, and neither raises an error:

1. It writes to `results/rms_cosim/`, while `analysis/deadband_selection.py`
   collects from `results/rms_phase6_replay/`.
2. It stores `runner_static = None`, and the study's admission filter reads the
   `runner_static` block and rejects a run whose block is absent.

The sweep would have run 15 cells over ~7 h, exited 0, and the analysis would
have reported zero admitted runs.

The comparison script is also what the deprecation shim delegates to, so it is
the same code path that produced run `0080` — the one already-valid cell.

## 4. Verification

- `--help` on the comparison runner: every flag the sweep passes exists
  (`--duration`, `--profiles`, `--profile-delivery`, `--dso-oltc-switch-cost`,
  `--physical-capability`, `--der-deadband`, `--start-time`, `--scenario`,
  `--no-pdf`, `--verbose`).
- Exit-code propagation re-checked (`SystemExit(0)` → 0, `SystemExit(1)` → 1);
  the sweep's fail-fast depends on it. An apparent exit −1 in a first check was
  an artefact of truncating the pipeline with `Select-Object -First 40`.
- Confirmed `new_run_dir("rms_phase6_replay", ...)` in the comparison script, so
  the analysis root is unchanged.
- Sweep restarted 12:59:42; run 1 (δ = 0.0025) launched with the correct
  argument vector, including `--physical-capability` and `--scenario rural_700`.

## 5. PowerFactory session check

Requested before launch. No `PowerFactory.exe` or run-related `python.exe` in
session 195 (`mschwenke`) — no orphan from the run cancelled at 10:52. The three
live `PowerFactory.exe` processes belong to other users' disconnected sessions
(197 `ahebing`, 205 `ms_admin`, 206 `jleide`), all started 2026-07-30.

## 6. Follow-up: run `0080` is unscaled and had to be excluded

Prompted by the question "is DSO_3 ×2 applied?", checked against the run records
rather than the code default.

The sweep **does** apply it — run `0098` logs
`der_scale={'DSO_3': 2.0} load_p_scale={'DSO_3': 2.0}`, from the defaults in
`experiments/helpers/rms_cosim_config.py` (the sweep passes neither
`--symmetric-dso` nor explicit scales). But **run `0080` records neither key**: it
predates the default and is unscaled. The runner prints the consequence itself —
*"results are NOT comparable with an unscaled run"*.

`ADMIT` did not check either key, so `0080` was admitted into a study otherwise
built from ×2 runs. It escaped notice only by accident: `collect()` keys on
(window, δ) and the sweep re-runs `0080`'s exact cell, so the scaled run would have
overwritten it — silently, last-writer-wins.

### Method of change — `analysis/deadband_selection.py`

| Change | Detail |
|---|---|
| `ADMIT` + 2 keys | `dso_der_scale` and `dso_load_p_scale`, both `{"DSO_3": 2.0}` |
| `_admit` dict branch | `ADMIT` compared only `None`/`float`/`bool`. A **missing** key is rejected, not read as an empty dict — a run predating the multiplier records nothing, and treating that as "no scaling" is exactly the silent admission being fixed. |
| `collect` duplicate note | prints when two admitted runs share a (window, δ) cell, instead of overwriting silently |

Stale `deadband_metrics.csv` / `deadband_optima.csv` (the single `0080` row) were
renamed `*.VOID_run0080_unscaled.csv` so the void number cannot be picked up.

### Verification

- `python -m analysis.deadband_selection` → `0 run(s) admitted, 59 skipped`
  (was 1 admitted, i.e. `0080`).
- Direct `_admit` probe: `0080` → `None`, `0087`/`0096` (±1.0 pu stub) → `None`,
  in-flight `0098` → `('2016-01-05T08:00', 0.0025)`. The filter admits the sweep
  and rejects the rest, so it is not simply rejecting everything.

## 7. Result of the 15-run matrix (runs `0098`–`0112`)

All 15 runs `[Gate E] PASS`, 12:59:42 → 16:08:43, ~13 min each.

| δ [pu] | w1 ifQ | w2 ifQ | w3 ifQ |
|---|---|---|---|
| 0.0025 | 0.463 | 2.824 | 2.167 |
| 0.0050 | **0.441** | 2.363 | 2.167 |
| 0.0075 | 0.466 | 2.082 | 2.167 |
| 0.0100 | 0.510 | 1.951 | 2.167 |
| 0.0150 | 0.533 | 1.644 (edge) | 2.167 |

**δ\* is operating-point dependent.** Window 1 has a genuine interior minimum at
δ = 0.005. Window 2 falls monotonically to the top of the swept range, so its
argmin (0.015) is an *edge bound* — the true optimum is at or beyond it. The
spread 0.005 → ≥0.015 is therefore a lower bound on the movement, against only a
1.9× difference in the (void, see §9) screening figure.

The voltage metrics never show an interior minimum: DS voltage degrades
monotonically with δ in both live windows, and TS voltage is flat to within
±2%. So the two-sided U-curve argument holds for **interface-Q tracking only**;
for DS voltage, narrower is better as far as the sweep can see.

### The metric set does not measure what the argument claims

The stated reason a too-narrow dead band is bad is DER chatter and repeated OFO
re-anchoring, but none of the three metrics measures actuator activity.
Interface Q registers it only indirectly, and it is the sole metric showing any
narrow-side penalty. If the two-sided claim is load-bearing in the thesis, a
direct activity measure (DER Q traverse per interval, tap operations) should be
derived — `rms_der_raw.csv` already records per-park Q and V for every run, so
this is post-processing, not a re-run.

## 8. Window 3 is degenerate: zero DER Q capability

Runs `0108`–`0112` returned **bit-identical results for all five dead bands**
(ifQ 2.167265, TS V 0.00591785, DS V 0.01569772). Not a duplicate-file artefact:
`rms_der_raw.csv` (12.4 MB), `rms_controlled_outputs.csv`, `rms_monitors_raw.csv`
and `settling_per_interval.csv` are byte-identical between `0108` and `0109`,
while `static_records.pkl` differs. The same comparison on windows 1 and 2 shows
the RMS outputs *differing* with δ, so the mechanism works generally.

Cause, from the runs' own logs:

| window | zero-capability parks | initial PCC capability |
|---|---|---|
| 2016-01-05 08:00 | 16 / 44 | [−45.2, 41.6], [−49.0, 40.9] Mvar |
| 2016-01-15 03:00 | 16 / 44 | [−44.3, 41.8], [−52.0, 36.4] Mvar |
| 2016-07-15 03:00 | **44 / 44** | **[0.0, 0.0], [0.0, 0.0] Mvar** |

`sgen[N] (VDE-AR-N-4120-v2, P=0.0 MW, S_n=80.0 MVA): zero Q capability -- the
park cannot act as a Q actuator`. Under `--physical-capability` the VDE diagram
makes Q capability contingent on P. Aggregate DER infeed is **29.2 MW** at this
window against 2605.7 MW at window 2, so every park is below the threshold and
all 44 parks mean Q = 0.000 Mvar for the whole run (window 1 spans −43.0 to
+55.6 Mvar). With no DER able to move, the Q(V) characteristic never binds and
its dead zone cannot matter.

**The profile data is genuine, not corrupt.** 2016-07-15 03:00 sits inside a real
multi-day wind lull (13–15 July at 0.002–0.05, recovering to 0.4–0.7 by the
16th–17th); both wind profiles below 0.02 occurs for 11% of the year with lulls
up to 32 h; `corr(WP7, WP10) = 0.936` year-round, so their near-identical values
at this instant are unremarkable.

**Reportable finding:** dead-band selection presupposes DER reactive headroom.
Under the physical capability diagram that headroom vanishes at low-infeed
hours, and the dead band becomes inert. Kept as a documented null result
(Manuel's decision), not deleted.

## 9. The screening excursion column is void, and was misused

Window 3 entered the matrix labelled "0.02051 pu, annual maximum" — from the
Tier-1 season screening on the **older topology**. On the current topology it is
the *least* stressed window of the three, not the most. The label selected the
window; it does not describe it. It was nevertheless quoted throughout the day's
reporting as though it characterised the runs ("1.9× the excursion", "2.5×, the
annual maximum").

Changes so that it cannot be quoted again:

- CSV column renamed `excursion_pu` → `screening_excursion_pu_old_topology`.
- Per-window header now reads `old-topology screening value X pu -- selection
  only, NOT a property of this run`.
- The verdict block prints a provenance warning naming 2016-07-15T03:00 as the
  counter-example, and the closing advice changed from "check whether it scales
  with the excursion column" to an explicit instruction *not* to regress against
  it, pointing at DER Q headroom or a fresh screening instead.
- No excursion figure is invented for the replacement window; the table prints
  `n/a`, because mixing an old- and a new-topology measurement in one column is
  the defect itself.

## 10. Two further reporting defects fixed in `analysis/deadband_selection.py`

| Defect | Consequence | Fix |
|---|---|---|
| An argmin at the edge of the swept range was reported as `optimum` | Window 2's `0.015` entered the cross-window comparison as if measured, though the metric is still falling there | `_at_edge`; `*` marker + explanatory note; `at_range_edge` column in `deadband_optima.csv`; verdict caveat stating the spread is a lower bound |
| An argmin over identical values was reported as `optimum` | Window 3 reported `δ* = 0.0025` — merely the first dict key — giving the headline verdict `MOVES: [0.005, 0.015, 0.0025]` | `_flat`; metric prints `flat`; degenerate windows excluded from the verdict with a `DEGENERATE WINDOW` note pointing at DER Q capability; guard when <2 live windows remain |

Verdict now reads `interface-Q optimum MOVES: [0.005, 0.015]` with
`EXCLUDED ...: 2016-07-15T03:00`.

## 11. Follow-up matrix (`experiments/run_deadband_followup.ps1`, new)

Phase A replacement window base sweep (5 runs) → phase B wide extension
δ = 0.02, 0.03 on window 2 + replacement (4 runs) → phase C zero anchor δ = 0 on
windows 1, 2 and the replacement (3 runs, ~4.9 h each). Each phase gated on the
previous one's exit code. Window 3 is excluded from every phase: a δ = 0 run
there would cost ~4.9 h to reproduce numbers already known to be identical.

### Replacement window: `2016-12-18 14:00`

`2016-07-25 13:00` was tried first and **rejected empirically** — at +3259.4 MW
net infeed the static leg's Jacobian probe power flow raised
`LoadflowNotConverged` after 200 iterations, aborting in 32 s.

Offline screening could not have predicted this. A pandapower probe with
`init="flat"` marked demonstrably feasible hours as diverging, while with
runner-like settings (`distributed_slack=True`) even the failing window
converges — neither reproduces the runner. The reliable evidence is empirical:
+805 MW (window 2) runs, +3259 MW does not. `2016-12-18 14:00` (+1367.1 MW,
1.7× window 2) was therefore chosen for **margin, not maximum stress**.

**No capability confound** (corrected from run `0114`'s own log). An earlier
draft of this section claimed the replacement had 44/44 parks live against 28/44
for windows 1 and 2, and warned of a stress/capability confound. That came from
the screening proxy, which counts a park live at P > 0.5 MW — not the VDE
capability threshold. Measured:

| window | zero-capability parks | initial PCC capability |
|---|---|---|
| 2016-01-05 08:00 | 16/44 | [−45.2, 41.6] Mvar |
| 2016-01-15 03:00 | 16/44 | [−44.3, 41.8] Mvar |
| 2016-12-18 14:00 | 16/44 | [−42.0, 46.1] Mvar |

The three live windows span a factor of 3.3 in net infeed (+409 → +1367 MW) at
**essentially constant DER reactive capability**, which is what makes a δ\* shift
attributable to the operating point rather than to a change in the actuator set.

### `powershell -File` cannot pass arrays

The first follow-up launch died in one second with exit 2. `-File` passes every
argument as a separate literal token, so `-Deltas @('0.0025',…)` collapsed to its
first element and the leftovers bound positionally — `$Python` was bound to
`'0.0075'`, so the sweep script hit its "python not found" branch. Verified with
a probe script: via `-File`, `Deltas = 1 : [0.0025]`, `Python = '0.0075'`; via the
call operator, `Deltas = 5` and `Windows = 2` with the embedded spaces intact.
Fixed by invoking the sweep with `&`. `exit` inside a script called that way
returns control to the caller and sets `$LASTEXITCODE` (verified: parent
survives, code 7 propagates).

`powershell.exe` is also **not on PATH** in this shell; it is resolved from
`$PSHOME`.

## 12. Final grid and result (29 admitted runs)

Phases A–D completed 2026-07-31 19:1x. Live windows carry all eight dead bands;
2016-07-15 03:00 keeps its five null cells.

| δ [pu] | w1 ifQ | w2 ifQ | repl. ifQ |
|---|---|---|---|
| 0 | 1.395 | 4.367 | 3.568 |
| 0.0025 | 0.463 | 2.824 | 1.838 |
| 0.005 | **0.441** | 2.363 | 1.615 |
| 0.0075 | 0.466 | 2.082 | 1.544 |
| 0.01 | 0.510 | 1.951 | 1.512 |
| 0.015 | 0.533 | 1.644 | 1.494 |
| 0.02 | 0.527 | **1.524** | 1.210 |
| 0.03 | 0.640 | 1.572 | **1.126** (edge) |

### The δ = 0 cost estimate was void, by a factor of ~23

δ = 0 was excluded from the original sweep on a recorded cost of ~4.9 h per run.
Measured: **12.3 / 12.8 / 13.2 min** — the same as every other cell. The
exclusion withheld the single most informative point in the study for no reason.
Corrected in `docs/deadband_selection_rerun_handover.md` and in the sweep
script's header.

### Three findings

1. **The dead zone is genuinely two-sided, and δ = 0 is what establishes it.**
   Interface-Q error at δ = 0 is 3.2× / 2.9× / 3.2× its optimum in the three live
   windows. Before that point the narrow-side branch rested on a ~5 % difference
   (0.463 vs 0.441) not separable from run-to-run variation.
2. **δ\* is strongly operating-point dependent:** 0.005 → 0.02 → ≥0.03, a factor
   of ≥6. The first two are bracketed interior minima; the replacement window is
   still falling at δ = 0.03, so the separation is larger than measured.
3. **The controlled quantities select opposite ends.** DS voltage is best at
   δ = 0 in *every* live window while interface Q is worst there; moving from
   δ = 0 to the interface-Q optimum costs 2.0–2.8× in DS voltage error. There is
   no single δ\* across the controlled outputs — it is an exchange, and the
   chapter should present it as one. TS voltage does not discriminate (optima at
   0.0025 / 0.03 / 0.02, spreads of 1–3 %).

Method write-up with these results: `docs/ch8_deadband_window_selection.tex`.

## 13. Risks / unresolved points

1. **The 2026-07-31 runner split cannot be validated empirically.** The whole RMS
   runner family (`run_comparison_rms_cosim_qss.py`, `run_rms_cosim.py`,
   `helpers/rms_cosim_config.py`, `helpers/rms_replay.py`) is **untracked**, and
   the pre-split file no longer exists on disk, so no diff proves the split was
   behaviour-preserving. `0080` cannot serve as the reproduction check because it
   differs in scaling as well as code version. The claim rests on the daily-log
   entry and on reading the current source.
2. **The study now stands entirely on the 15 sweep runs** — there is no admitted
   run predating them. A failed cell is a hole in the matrix, not a fallback.
2. The untracked state of `pf/` and the RMS runners remains the standing risk
   already raised in the 2026-07-31 refactor log §4.3 — nothing here is
   recoverable from git.
3. Runs `0087`–`0097` are `rural_700` but carry
   `der_q_capability_override_pu = 1.0`; they are a different experiment and are
   correctly excluded by the admission filter. Run `0097` was cancelled manually
   at 10:52 and left no `rms_records.pkl`.
