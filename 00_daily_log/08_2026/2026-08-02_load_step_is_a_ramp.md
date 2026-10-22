# 2026-08-02 — The exogenous "load step" is a 20 s ramp, and it starts one dispatch interval early

**Reason.** Selecting the disturbance amplitude for the 2D dead-band experiment.
Three candidate runs failed Gate E, and the first diagnosis of why was wrong.

## Measurement

`load_step_time_s = 100` does **not** produce a step at t = 100 s. The runner
applies the step to the interpolated profile frame, which has `dt_s = 20 s`
rows; the RMS ElmFile playback then interpolates between successive rows. The
load therefore ramps over `[step_t − dt_s, step_t]`.

Measured on run 0280 (44 parks, RMS traces at dt = 0.1 s), first DER activity is
at **t = 80.53 s** for a step configured at t = 100 s, and the largest single
jump in the whole run is there. An otherwise identical uniform-step run (0239)
is bit-identical up to t = 80 and quiet through the same window:

```
bin        0239 (x1.10)   0280 (bus41 +1100 MW)
 0- 20s        54.83            54.83
20- 40s        10.78            10.78
40- 60s         4.27             4.27
60- 80s         2.07             2.07
80-100s        90.93         11575.70     <- ramp
```

This is a known resolution limit, not a new defect: the runner's own comment at
`experiments/runners/multi_tso_dso.py:1670` says the step is applied to the
interpolated frame so it is "exact at dt_s resolution", i.e. it reduced smearing
from 15 min to 20 s rather than to zero.

## What it invalidated

A window of `t < step_t` labelled "pre-step" contains the entire ramp. Using it
produced a **spurious finding** that was reported before being checked against
the time series:

- claimed: at `delta_DS = 0` the parks chatter pre-step at 313 Mvar/park/interval
  against 1.07 at `delta_DS = 0.02`, i.e. DS chatter propagating into the TS
  parks;
- actual: the two runs are identical before the ramp. The 313 was the ramp.

Retracted. The correct quiet window ends at `step_t − dt_s`.

## Settling criterion

Gate E is the wrong instrument for judging an amplitude — it validates QSS/RMS
equivalence. The criterion is whether the cascade returns to its quiet level,
which is how the ×1.5 uniform amplitude was rejected in design E. Defined as
`tail / floor ≤ 2`, where floor is the mean summed |dQ| per 20 s interval over
the second half of the pre-ramp window and tail is the mean over the last three
intervals, it reproduces the earlier judgement independently:

| run | disturbance | δ | peak | tail/floor | verdict |
|---|---|---|---|---|---|
| 0272 | ×1.01 | 0.01 | 12 | 0.4 | settled |
| 0239 | ×1.10 | 0.01 | 91 | 0.5 | settled |
| 0257 | ×1.25 | 0.01 | 316 | 0.5 | settled |
| 0275 | ×1.50 | 0.0025 | 9 309 | 18.5 | **not settled** (rejected in design E) |
| 0281 | bus 41 +400 MW | 0.01 | 390 | 0.5 | settled |
| 0279 | bus 41 +1100 MW | 0.02/0.0 | 12 473 | 5.2 | **not settled** |
| 0280 | bus 41 +1100 MW | 0.01 | 11 558 | 16.2 | **not settled** |

**+1100 MW is rejected.** At the same dead band it drives a peak 37× that of the
largest accepted uniform amplitude and is still 16× its quiet level at the end
of the horizon. +400 MW sits alongside the accepted ×1.25 step and passes
Gate E.

Also corrected: Gate E does **not** fail across the `delta = 0` row. All 22
uniform-step runs at this window pass across the full δ range including δ = 0.
The only failures are the ×1.5 amplitude and the three +1100 MW runs.

## Change

`analysis/deadband_2d.py` — the pre-window of the `dv_pre_*` noise floor now
ends at `step_t − dt_s`. Without this the floor captures the ramp and the
`dv_snr` guard would report every cell as noise-dominated.

## Open

The **post**-step windows in `analysis/deadband_disturbance.py` and
`analysis/deadband_threshold.py` start at `step_t`, so they exclude the ramp
interval `[step_t − dt_s, step_t]` and therefore part of the response. Those
modules produced results already written into
`docs/ch8_deadband_selection.tex`. Changing the window would move published
numbers, so it is recorded here rather than changed silently — the measurements
are self-consistent, they simply begin one dispatch interval after the
disturbance starts. Worth deciding before the chapter is final.
