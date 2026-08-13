# E1 — dead-band lower bound / profile-drift instrument

**Date:** 2026-08-07, 13:10–15:30 (runs), analysis to 15:40
**Runs:** 0537 (import), 0538 (export) → `results/deadband_droop_e1_drift/data`
**Reason:** Chapter 9 needs a *lower* bound on the local Q(V) dead-band
half-width that is independent of any disturbance. The existing campaign
(`results/deadband_droop`, runs 0498–0536) measures false activation by
installing each candidate band live and re-running, which makes every rung its
own closed loop. E1 removes that coupling: pin the band open, record the
open-loop drift once, and threshold offline.

## What was changed

Nothing in the controller or plant code. New artefacts only:

| file | what |
|---|---|
| `experiments/run_e1_drift.ps1` | driver; the FA battery's invocation with `--duration 3600` and `--tso-deadband 0.5 --dso-deadband 0.5` |
| `results/deadband_droop_e1_drift/analysis_e1_drift.py` | per-(level, park, window) drift extraction |
| `results/deadband_droop_e1_drift/make_falseactivation_figure.py` | TikZ generator: standalone figure + shared-axis panel (a) |
| `results/rms_phase6_replay/0536_counter_anchor/` | numbering anchor (see "Numbering" below) |

## Method

Two undisturbed RMS co-simulations, `experiments.run_comparison_rms_cosim_qss`,
3600 s, fixed 10 ms step, stride 10. Scenario `rural_700`, DSO_3 ×2, physical
VDE-AR-N-4120 capability, rev-2 sensitivities, `g_w_dso_oltc = 200`, OFO on with
the campaign's tuned weights — untouched. No contingency;
`qv_deadband_at_contingency` deliberately **not** set, so the band is constant
for the whole run. One droop slope (0.05): with the band at 0.5 pu the law never
leaves its dead zone, so a second slope would reproduce the first bit for bit.
Log confirms: `deadbands applied: [0.5] pu; droops applied: [0.05] pu`.

Metric, per level × park × inter-dispatch window:

    d = max_{t in [t_k, t_k+1)} |V(t) - V_anchor|

`V_anchor` is the terminal voltage at the dispatch instant — literally what
`PowerFactoryPlant.apply_u` writes into the QVPRE block's `Vanchor`, read from
`m:u` in the paused RMS state. **The maximum, not the endpoint difference**: the
dead band is a running test on the instantaneous deviation, so a mid-window
excursion that returns before the next dispatch would still have fired it.

TS parks re-anchor on TSO instants (180 s), DS parks on DSO instants (20 s) —
`multi_tso_dso.py:2997` guards the TSO branch with `_is_period_hit`, the DSO
branch runs every step. First TSO period discarded as initialisation.

Two details resolved rather than inherited:

1. **Anchor sample.** The ComRes grid does not land on the dispatch instants
   (179.9717 then 180.0700 s around t = 180) and the `qset`/`Vanchor` events
   fire at t+0.01 s, so the first sample *inside* a window is already
   post-write. The anchor is read as the last sample at or before t_k.
   Checked against the campaign's first-in-window convention: max difference
   6.0e-6 pu, ~400× below the narrowest candidate. Immaterial, but correct.
2. **`drift_max_late_pu`** added — the same maximum over the second half of the
   window — after the relative-position diagnostic showed most DS maxima are
   post-dispatch settling rather than profile movement.

## Result

Park counts and windows (all four cells identical): **4 TS parks × 19 windows =
76**, **40 DS parks × 171 windows = 6840** per operating window. The 40 DS parks
are 28 `DER_DSO_*` + 12 `WPC_DSO_*`; the campaign's 840 DS rows were 40 × 21,
not 28 × 30.

Drift maxima [pu]:

| window | TS | DS |
|---|---|---|
| −117 MW (import) | 0.004846 | 0.011152 |
| +409 MW (export) | 0.002698 | 0.010920 |

False activation is 0 at both levels and both windows from **0.0125 pu**
upward. The binding sample is one DS window — `WPC_DSO_4_s43_b111`, window 81,
maximum at rel = 0.996 — so the bound rests on end-of-window profile drift, not
on a transient.

**Relative position is bimodal and splits the levels.** TS maxima sit at
rel ≈ 0.89 (90.8% import / 69.7% export in the 9th decile): genuine drift
accumulated across a 180 s window. DS maxima cluster in the *first* decile
(96.3% import, 81.9% export): post-dispatch electrical settling inside a 20 s
window, not profile movement. The DS curve at the narrow rungs is therefore
substantially a transient statistic, and the panel title "false activation on
profile drift" is strictly accurate only for TS. The `drift_max_late_pu` column
exists to separate the two; note that both DS maxima survive it unchanged
(0.01115 / 0.01092), i.e. the *bound* is profile-driven even though the *bulk*
of the distribution is not.

## Deviations from the specification

- **DS parks are 40, not the expected ~28**, and usable DS windows are 171, not
  179 (windows 9–179 after discarding the first TSO period; window 180 holds
  the single end-of-run sample).
- **The existing campaign is a reactive load step**, +200 Mvar at TS bus 7 / 11
  via `--q-step-bus`, not a load step in MW.
- **Run 2 was allocated 0543, not 0538** — see below. Renumbered on move-out.
- Horizon was **not** shortened: `data/profiles.csv` is resampled to 20 s across
  all of 2016 and both start instants resolve 181 profile rows for 3600 s.

## Numbering

`new_run_dir` takes `max(existing NNNN_ dirs) + 1`, and runs 0437–0536 had been
moved out of `results/rms_phase6_replay`, dropping the visible maximum to 0436.
`0536_counter_anchor/` holds it so E1 starts at 0537. A concurrent E3 campaign
reserved 0540–0579 while run 1 was in flight, so run 2 was allocated 0543; it
was renumbered to 0538 on move-out and 0543 released back. `meta.json` records
both (`counter`, `counter_as_allocated`).

The allocation is **not atomic** — two sessions starting within the same
directory-listing window both read the same max and both succeed, because the
`FileExistsError` retry only catches a same-second name clash. Flagged as a
follow-up; the two anchor directories are a workaround, not a fix.

## Open points

- Two operating windows of one scenario, no measurement noise, δ_TS = δ_DS.
- Whether the DS bound should be read off `drift_max_pu` or `drift_max_late_pu`
  is a modelling choice, not a data question. They agree at the maximum here.
- The thesis figure `deadband_detector_bounds_min.tex` was updated on the user's
  Desktop (backup `.bak_2026-08-07_1500`); panel (a) had carried its own
  `xmin`/`xmax`/`xtick`, which broke the groupplot's shared x axis. Removed.
  **Not compiled** — no LaTeX toolchain on this server.
- The empty directory `results/rms_phase6_replay/0537_2026-08-07_131025/` could
  not be deleted (directory handle held by something outside this session); its
  contents were moved out in full (35 files, 1068 MB verified).
