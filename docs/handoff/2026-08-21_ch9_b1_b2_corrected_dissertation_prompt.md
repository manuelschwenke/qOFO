# Prompt for dissertation-access Claude session

Use the following prompt verbatim:

> You have access to my dissertation and to the `qOFO_GH` repository. Update
> the Chapter 9 timescale/parameter-selection text using **only the final
> corrected B1 and B2 results** below. Inspect the dissertation first, preserve
> its notation and writing style, then edit the relevant source files directly.
>
> Source-of-truth artifacts:
>
> - B1:
>   `docs/data/2026-08-21_ch9_b1_b2_corrected/B1_ninner_10mvar/20260821-180548/`
> - B2:
>   `docs/data/2026-08-21_ch9_b1_b2_corrected/B2_ts_period_sweep/20260821-183927/`
> - Technical record:
>   `docs/daily_log/08_2026/2026-08-21_dso4_damping_b1_b2_corrected.md`
>
> Do not use or discuss superseded capability-edge runs, stopped runs, or the
> earlier non-convergent diagnostic cases. Do not invent a formal TSO setpoint
> bound. State that 10 Mvar is an empirical supervisory envelope: 3597/3600
> archived command increments (99.9167%) were at most 10 Mvar; only three
> exceeded 10 Mvar and two exceeded 20 Mvar.
>
> For B1, describe the final protocol accurately: ten admissible operating
> windows, four DSOs, three interfaces per DSO, balanced half-fraction direction
> design (120 cases), (T_\mathrm{STS}=20) s, applied step capped at 10 Mvar,
> frozen supervisory parent, and convergence defined by the interface-flow
> flatness condition
> (|q_\mathrm{PCC}(k)-q_\mathrm{PCC}(k-1)|\le1) Mvar holding thereafter.
> The two VDE dead-zone windows are outside the population because no reactive
> capability band exists to traverse.
>
> Report the final B1 numbers exactly: 120/120 converged, zero censoring,
> median (N_\mathrm{inner}=2), p90 (=6.2), p95 (=10), maximum (=18),
> and 113/120 cases (94.2%) at (N_\mathrm{inner}\le9). There were zero tap
> moves, voltage violations, capability-limited flags, and command-delivery
> errors. Per-area p90 values are DSO 1: 6.0, DSO 2: 11.1, DSO 3: 4.0,
> DSO 4: 3.0. Retain (N_\mathrm{inner}=9) as the pooled approximately-90th-
> percentile engineering choice and explicitly say it is not a worst-case
> guarantee; a strict pooled p95 choice would be 10. Preserve the circularity
> statement: (G_w) was calibrated assuming nine inner iterations, so B1 checks
> whether that assumption survives its own tuned controller rather than
> providing an independent measurement.
>
> If the controller-parameter table is in scope, use the final corrected
> Chapter-9 setting without narrating the debugging history: voltage-relief
> factor 10 for DSO 2/4 with (g_v/g_{w,\mathrm{OLTC}}) preserved, and the
> DSO-4-only continuous-DER weight
> (g_{w,\mathrm{DSO4-DER}}=828.3682). DSO 1?3 retain the global
> (g_{w,\mathrm{DSO-DER}}=549.8818).
>
> For B2, describe 60/60 completed closed-loop runs over 12 windows with
> (T_\mathrm{STS}=20) s and
> (T_\mathrm{TS}\in\{60,120,180,240,300\}) s. Use this pooled table:
>
> | (T_\mathrm{TS}) [s] | ratio | (ho) median | (ho) p95 | censored |
> |---:|---:|---:|---:|---:|
> | 60 | 3 | 0.4407 | 2.5140 | 0.1574 |
> | 120 | 6 | 0.3095 | 2.7782 | 0.1552 |
> | 180 | 9 | 0.2572 | 3.5164 | 0.1597 |
> | 240 | 12 | 0.2221 | 2.9677 | 0.1621 |
> | 300 | 15 | 0.1909 | 2.5977 | 0.1659 |
>
> Interpret B2 carefully: median (ho) decreases monotonically because it
> measures subordinate tracking of the received correction, not the
> supervisory error caused by a stale correction. Therefore B2 supports the
> timescale/tracking discussion but does **not** uniquely prove that 180 s is
> optimal; that would require the supervisory objective or an explicit
> staleness metric. Do not use B2's `n_k` as the dissertation
> (N_\mathrm{inner}); B1 is the authoritative flatness-based measurement.
>
> Constraint evidence: B2 recorded three tap moves in 30,240 interface
> intervals. All periods (\ge120) s had zero voltage violations. At 60 s,
> six duplicated interface-interval flags occurred in one
> `d_ramp_down_spring` window; the maximum voltage was 1.100187 p.u. and the
> maximum slack was only (8.6\times10^{-5}) p.u. Report this compactly if the
> surrounding dissertation text discusses constraint satisfaction; do not
> round it to zero.
>
> After editing, build or lint the dissertation if its workflow is available.
> Return: (1) the files changed, (2) the exact inserted/revised passages, and
> (3) any remaining claim that the available experiments do not support.

