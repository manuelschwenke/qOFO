# 2026-08-07 — E3: tracking accuracy versus dead-band half-width

**Timestamp:** 2026-08-07 13:17 (campaign start), ms_admin account
**Scope:** new `experiments/run_e3_tracking.ps1`; new
`results/deadband_droop_e3_tracking/` (README, `finalize_e3_runs.py`,
`analysis_e3_tracking.py`); new counter marker
`results/rms_phase6_replay/0539_RESERVED_numbering_marker`. No existing code
path changed — the sweep drives
`experiments.run_comparison_rms_cosim_qss` through its existing CLI.
**Follows:** `2026-08-06_qv_switch_at_contingency_and_adaptive_step.md`;
companion to the E1 false-activation battery in `results/deadband_droop/`.

---

## 1. What E3 is, and why it exists

E1 measures the *proxy*: how often ordinary profile drift pushes a park out of
its dead zone and makes the local Q(V) layer act. E3 measures the *harm*: how
much the droop layer degrades the OFO's tracking accuracy as the band narrows.
A narrow band lets the droop answer voltage movement the OFO is itself
producing, so the two layers compete. The expected shape is tracking accuracy
flat above some half-width and degrading below it; that knee is the lower
bound on the dead band, obtained from the damage rather than from an
activation rate.

32 undisturbed RMS runs, 900 s: 8 half-widths (0, 0.0025, 0.005, 0.0075, 0.01,
0.02, 0.05, 0.2 pu — the last being the droop-out-of-loop reference) × 2 droop
slopes (0.05, 0.10) × 2 operating windows (−117 MW import, +409 MW export).
Runs 0540–0579, moved to `results/deadband_droop_e3_tracking/data`.

## 2. Method

**The dead band is live from t = 0**, set through
`--tso-deadband`/`--dso-deadband`, and `--qv-deadband-at-contingency` is *not*
used — the opposite of the load-step sweep and the same as E1. There is no
disturbance here, so a band installed at a contingency would never be
installed at all.

**Landmine, accepted deliberately.** Those flags also feed the controllers and
the static plant, so the legs do **not** share a common run-up — each is its
own closed-loop system from t = 0. That is correct for this experiment, but it
means the legs may be compared through **aggregate metrics only**. Pointwise
trajectory differencing against a shared baseline is valid in
`results/deadband_droop/` and invalid here; both the README and the sweep
script header say so at the top.

**Sampling.** Every metric is taken at a dispatch instant k−, the pre-apply
measurement (achieved-value tracking). The whole first TSO period is discarded
as initialisation transient: 36 of 45 DS instants and 4 of 5 TSO instants
survive. Voltage comes from the RMS trace at the last sample strictly before
the instant (the pickled records carry only zone/group aggregates, and an RMS
over (bus, dispatch) needs every bus); interface Q and the objective come from
`rms_records.pkl`.

**Two asymmetries worth recording.**

* *The interface setpoint is lagged one record.* `dso_trafo_q_set_mvar` on
  record *k* is the setpoint in force **after** the dispatch at *k* and changes
  exactly at TSO instants, while `dso_trafo_q_meas_mvar` at *k* is the
  pre-control measurement — verified equal to record *k−1*'s
  `dso_trafo_q_actual_mvar`. The tracking error is therefore
  `|q_meas(k) − q_set(k−1)|`, band from *k−1* too. Pairing at the same index
  would manufacture an error at every TSO instant.
* *Activation counting reuses E1's rule verbatim*, including its
  `t >= tso_period_s` window start, whereas the tracking metrics use
  `t > tso_period_s`. Comparability with E1's published numbers beats internal
  tidiness. Verified: the implementation reproduces E1's published counts
  exactly on run 0506 (TS 0/12, DS 0/840), which is why the E3 activation axis
  can be cross-plotted against E1 directly.

**Weighting.** `g_v` (1e7) and `dso_g_v` (1e5) are scalars here (`zone_g_v` is
null), so a weighted RMS is exactly `sqrt(g)` × the unweighted one and adds no
information. Both are emitted as asked, alongside the quantity that genuinely
is weighted — the objective's voltage penalty `g · Σ (V_i − V_ref)²`.

**Fixed 10 ms step, no adaptive stepping**, matching E1: there is no switching
transient to resolve. 900 s rather than E1's 600 s, as specified.

## 3. Run numbering — a shared counter with two writers

`new_run_dir` takes the next counter from the directory names present in
`results/rms_phase6_replay`. Runs 0437–0536 were moved out to
`results/deadband_droop/data`, which dropped the visible maximum back to 0436
and would have made the next run re-issue 0437.

A `0536_counter_anchor` already existed (created 13:09 today by the other
account for its E1 profile-drift runs 0537–0538). E3 adds
`0539_RESERVED_numbering_marker` so its own runs start at 0540 as specified.
**Neither may be deleted**; and the E3 marker must be advanced to
`0579_RESERVED_numbering_marker` once these runs are moved out, or the counter
regresses into numbers already in use.

Because a second campaign was allocating from the same counter concurrently,
E3's numbers can interleave with a neighbour's. Both `finalize_e3_runs.py` and
`analysis_e3_tracking.py` therefore select runs on six `config.json` fields at
once (900 s horizon, no contingencies, `qv_deadband_at_contingency` null, both
levels' half-width equal and in the swept set, slope in {0.05, 0.10}, one of
the two windows) and **ignore run numbers entirely**. Nothing else in
`results/` satisfies all six.

## 4. Concurrency

The other account (mschwenke, Z:) started a 3600 s E1 profile-drift run at
13:10 and was still in its RMS leg when E3 started at 13:17. Two concurrent
PowerFactory sessions across the two accounts are established practice here —
`run_qstep_sweep.ps1` was written for exactly that split, and the 2026-08-06
campaign shows two run directories created in the same second (0482/0483),
which the atomic `mkdir` allocation in `new_run_dir` handles. The cost is CPU
contention, so per-run wall time during the overlap is not representative.

## 5. Deviations from the specification

* **`--der-slope` sets both levels.** The spec named
  `runner_rms.tso_qv_slope_pu`; the CLI has no TS-only path that also reaches
  the RMS plant's per-sgen droop map, and the existing campaign swept
  `tso_qv_slope_pu = dso_qv_slope_pu` throughout. Both levels therefore carry
  the swept slope, as in `results/deadband_droop/`. Same for the half-width.
* **Exit code 1 is not treated as a failure.** `main()` returns 1 whenever
  Gate E's verdict is not PASS, which includes "a dispatch interval did not
  settle inside its window" — at narrow dead bands that is the effect being
  measured. The sweep script logs it distinctly and continues; health is
  judged on `csv/rms_comres_full.csv` reaching 900 s and a non-empty
  `rms_records.pkl`.

## 6. The PowerFactory exit-segfault has a second exit code, and every sweep script checks the wrong one

Run 0540 finished with **Gate E PASS**, all 45 dispatch steps, a complete
193 MB `rms_comres_full.csv`, `rms_records.pkl` and every figure — and exit
code **`-1073741819`**. That is `0xC0000005`, `STATUS_ACCESS_VIOLATION`: the
harmless PowerFactory segfault at process exit, the same event the older
scripts describe as "exit 139 (PF exit-segfault, results written)".

`139` is the **POSIX** encoding (128 + SIGSEGV). Windows never produces it.
`run_qstep_sweep.ps1` and `run_qstep_falseactivation.ps1` both test only
`$rc -eq 139`, so on this platform that branch is dead: the segfault falls
through to the generic `$rc -ne 0` arm and a perfectly healthy run is recorded
as FAILED. Last night's campaigns happened to exit 0 and never exposed it.

`run_e3_tracking.ps1` now accepts both codes. The other two scripts still
carry the defect and would mislabel a whole campaign; that is logged
separately as out of scope for E3.

**Consequence for the E3 campaign in flight:** the fix landed after the sweep
had already been parsed by PowerShell, so the *running* instance keeps
printing `!!! FAILED` once per run and will end with a wrong failure tally.
Nothing is lost — the loop calls `continue` either way, and run 0540 is
complete and analysed. Judge the campaign on `MANIFEST_runs.csv` coverage,
not on that tally.

## 7. Timing, measured

Run 0540 (900 s, **contended** with the other account's 3600 s run):
**31.9 min** wall — 14.1 min setup (static leg 1.7 + PowerFactory build,
ComInc and Q(V) anchoring 12.4), 15.2 min RMS integration at ~19 s/step over
45 steps, 2.5 min ComRes export and post-processing.

Reference: E1 run 0506 (600 s, uncontended) took **11.5 min** total at
**13.4 s/step**. So contention costs ~1.4× on integration and ~4× on the
PowerFactory build.

## 8. Status

Campaign in flight from 13:17; run 0540 complete and analysed, 0541 started
13:49. Results section of
`results/deadband_droop_e3_tracking/README.md` is filled when the 32 cells are
complete.
