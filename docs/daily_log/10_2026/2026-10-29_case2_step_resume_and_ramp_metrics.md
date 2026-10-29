# 2026-10-29: STEP campaign resumed; ramp metrics decided

## Pool deadlock and resume (07:35)

- **Symptom:** the STEP campaign `step/0002_2026-10-28_193301_run_step` stopped at 21:07 on
  28 Oct after 120 of 1350 runs. No worker was left, no run was in flight, and the main
  process was idle.
- **Cause:** `ProcessPoolExecutor(max_tasks_per_child=4)` on Python 3.12. After 30 workers × 4
  tasks the executor did not replace the recycled workers.
- **Fix (63814c2):** the pool runs without recycling, as the MC driver's pool does.
- **Resume:** only the job's own two processes (queue cmd 10716, main 13524) were stopped; four
  Python jobs of another user on e5power were left alone. Resumed with `--run-dir` at 07:37,
  the MC again queued after it (`launch_logs/20261029_073719_step_resume_then_mc.cmd`).
- **After the resume:** 162 runs done by 07:56 with 30 workers, past the old stall point.
- **Failures:** six P runs at OP-H after the G4 trip (gen 2, phases 20–160 s) failed with
  `LoadflowNotConverged`. These are outcomes; whether L2 fails too shows in the E case.

## Ramp metrics: three versions compared (126 R runs), the defined one kept

Scripts in the scratchpad (`ramp_eval.py`, `ramp_rate.py`); outputs `ramp_eval.json`,
`ramp_rate.json` and `ramp_traces.csv` in the run folder on e5power. Values are means over
3 OPs × 3 areas, the ramped area, in mpu.

| Version | Content |
|---|---|
| A, as defined (plan §2) | deviation from the pre-ramp state over the ramp-up (RMS, peak), hold deviation, raw in-period sawtooth, settling |
| B, split | offset at the end of the hold, lag relative to shape × offset, detrended sawtooth |
| C | fast minus slow ramp at equal load level; settling after the ramp |

Version A, RMS during the ramp (slow): L2 11.9, S1 10.6, S2 8.2, O2 6.2, O3 4.6, M 4.1.
It is dominated by the offset; the fast/slow ratio is 1.0–1.3.

Version B:
- offset: L2 21.7, S1 16.5, S2 12.8, O2 7.3, M 7.0, O3 5.3;
- detrended sawtooth during the ramp: O2/O3 2.8–4.6, others ≤ 1.4, every variant ≤ 0.3 without
  a ramp;
- the lag is contaminated by partial saturation (negative early lag for S2 and O2) and by the
  plant's nonlinearity (L2 shows 1.4).

Version C:
- fast-minus-slow peak: L2 0.2, M 1.3, S1 2.7, S2 3.2, O3 4.8, O2 5.2;
- settling after the ramp: M 0.5–3, S2 8–10, S1 12, O3 11–15, O2 16–18 min.

**Author decision:** keep the metrics as defined (A). Caveat recorded: during a ramp the raw
sawtooth includes the ramp's own rise within a period.

**Open:** S1/S2 in area 1 at OP-M/OP-L end 0–9 mpu worse than L2 after the ramp, and the pilot
error does not explain it. To be analysed at bus level.
