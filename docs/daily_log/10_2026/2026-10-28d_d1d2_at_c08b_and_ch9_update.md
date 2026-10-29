# D1/D2 and the Ch 9 (now Ch 8) results at the selected set c08b1d3d33ab

Timestamp: 2026-10-28, ~14:50 – 2026-10-29 morning, Europe/Berlin.

**Reason:** procedure 2026-10-28 (STS engagement bound, run 0008) selected
c08b1d3d33ab; the D1/D2 runs and the thesis results quoted fde2e81af6b1.
Author: "yes, please, and set c08b in run_multi_system_ofo.py".

## Changes (qOFO)

| commit | change |
|---|---|
| 61cf608 | `run_multi_system_ofo.main()` comment names the selected set and prints it. Only this hunk is committed; the author's `make_config` edits stay uncommitted. `main()` already used `SELECTED_PARAMSET`. |
| 92577b1 | `ch_9_4_deadband`: the converter's interval after a discrete step (the controllers' correction against a droop reference set at the stepped voltage) counts with the event. Author decision 2026-10-28, "count as part of the event, so stay at 0.01". `excursions()` gains `after_discrete`; `selection.json` gains `n_correction_intervals`; the confirmation classifies those activations as discrete. |

## Runs

- **Run 0008** (procedure 2026-10-28): `report_w5/` added. The W5 report of
  c08b and the Stage-I point is 12 h on the frozen network; the Stage-I row
  reproduces run 0001 exactly. Recorded with `annotate_run`.
  - c08b in W5: e_v(TS) 0.00816, e_v(DS) 0.01092, AVR step 0.0083 pu,
    6 tap hunting events, 1 bank hunting event.
  - Stage-I point in W5: 0.01097 / 0.00973, 12 tap hunting events.
- **Run 0009** (c08b, full): stage 1 complete.
  - The rule then in force gave a drift max of 0.0087 pu, i.e. design
    0.015 pu. That maximum is the STS controller's correction in the interval
    after the W2 capability step at 7170 s.
  - Stage 2 was stopped after the author's decision. The run is marked
    failed with the reason; its stage 1 is reused.
- **Run 0010** (run of record for D1/D2; stage 1 from 0009):
  - **D1 selection:** drift max 0.00277 pu (W4, STS), so Δ\* = 0.0075 and
    design 0.010 pu. The rule as written would give 0.025 pu. 264 discrete
    and 264 correction intervals of 65 784. dt check W1: 0.0014953 pu at
    10 s and at 20 s.
  - **D1 confirmation at 0.010:** 0 drift activations. Activations at
    discrete events by STS converters only, each episode ≤ 20 s: W2 10
    converters (≤ 2.7 Mvar), W4 8 (≤ 1.2 Mvar), W5 12 (0 Mvar).
    Deviation ≤ 0.0084 pu and 0.81 Mvar; same tap and bank operations;
    W1, W3, W5 and W6 identical to the reference.
  - **D1 confirmation at 0.0075:** 14 / 8 / 22 converters; ≤ 0.0106 pu and
    1.25 Mvar.
  - **D2 at 0.010 (W1–W3, W6):**

    | Disturbance | Support | Reduction of the voltage step |
    |---|---|---|
    | Bus 8 | 78–85 Mvar | 31–33 % |
    | Bus 12 | 69–81 Mvar | 15–22 % |

    - At 0.020: 33–39 and 16–28 Mvar, reduction 14–16 and 6–12 %.
    - W4: 47–52 Mvar. W5: 0.
    - Hand-over ≤ 360 s.
    - STS > 1.10 pu only with the droop off, for one 10-s step at a TS
      dispatch: W2, all four events, up to 1.110 pu; W6, bus 12 inductive.
    - TS outside W5: 0.964–1.099 pu at the design width. At 0.020 and
      with the droop off, the bus 12 capacitive step in W2 gives 1.101 pu
      for 40 s.
    - W5, bus 12 capacitive: 1.106 pu for 170 s in all legs, since no
      converter has capability.
    - The supervisory bank and tap actions differ between the legs at
      c08b. For fde2 they had been identical, and that sentence was dropped.
- **Archived:** run 0007 (fde2), superseded by 0010.

## Thesis (`Z:\latex_diss_clone`)

The timescales chapter is now `Chapters/Chapter08.tex` after the author's
renumbering; graphics are under `graphics/Ch8`. The author's 2026-10-29
morning commit includes the c08b edits made here on 10-28:
- B/C: desired-behaviour item 2 (engagement bound), selection rule 1,
  `tab:param:weights:resolved`, the B result, W5, the C result,
  `tab:param:timescales:ninner`, the summary box;
- settling CDF `graphics/Ch8/data/2026-10-28_settling`;
- the provenance blocks of runs 0008 and 0001;
- the D1 method sentence on the correction interval, in the author's wording.

This entry adds the D1/D2 results (the author's text, numbers to c08b),
the D2 table, and the provenance block 0007 → 0010. `thesis_runs check`:
6 blocks, 0 failing.

## Open

- The W5 sentence in the Experiment B assumptions ("no weight can change
  its outcome") still contradicts the W5 numbers.
- The D2 trajectory figure is still a `\todo`.
