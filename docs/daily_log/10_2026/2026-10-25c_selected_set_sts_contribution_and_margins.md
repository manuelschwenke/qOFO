# Selected weights (c08b1d3d33ab): STS contribution, wear and voltage margins

Timestamp: 2026-10-25, ~16:30 Europe/Berlin.

Reason: author question — how much does the STS contribute with the selected
set, how often do taps and banks switch, and does the STS keep a margin to
0.90 / 1.10 pu. The primary criterion stays TS voltage tracking; STS demand
counts only through it; the STS may be somewhat worse than the starting point
as long as it keeps a margin to the absolute limits.

## Method

The Stage-II rows keep no interface or DER series, so the selected set and the
Stage-I point were re-simulated in W1–W6 with the records kept
(`results/THESIS_ch8_weight_selection_20261024/records_unit.py`, BLAS pinned
to one thread as in Stage II; `sts_contribution.py`, `.log`, `.json`). All 12
re-runs reproduce the stored e_v(TS) exactly. No code changed.

## Results (ranked windows W1–W4, W6; selected vs Stage-I point)

- Wear (stored rows): TS taps 11 vs 10, STS taps 9 vs 19, bank commits 12 vs
  11 over 5 × 12 h; ≤ 2 operations per device and 12 h; no hunting.
- TS demand on the STS (mean |Σ Q_set − Q_set(t0)| per STS): 12.7–23.1 vs
  24.5–51.4 Mvar — roughly half. Delivered flow likewise roughly half. Interface
  tracking RMS 1.0–2.1 vs 6.2–23.9 Mvar. STS DER utilisation unchanged
  (0.11–0.24, > 90 % of capability at most 5 % of the time).
- e_v(TS) 0.00449 vs 0.00545: the gain comes with less STS flow, more precise
  STS tracking and the cheaper TS-side actuators (g_w_gen 2 decades lower).
- e_v(DS) 0.0067–0.0129 vs 0.0052–0.0104 per STS.
- **Margin to 0.90 / 1.10 pu: DSO_2 0.004 pu (W1: up to 1.096 pu, above
  1.08 pu for 11.6 % of the 12 h), DSO_4 0.011 pu (W1: 1.089 pu, 3.1 % above
  1.08 pu)**; DSO_1 0.035, DSO_3 0.043. Stage-I point: ≥ 0.031 pu everywhere
  (DSO_2 max 1.055, DSO_4 max 1.069). TS margin unchanged (0.051 vs 0.050).
- Inference from the weights (not measured): STS taps engage at a ~27 %
  larger systematic deviation than designed (g_w_dso_oltc / dso_g_v 0.00212 vs
  0.00166), consistent with fewer STS tap operations and the higher STS
  voltages.

## Open, for the author

- The procedure's filter has no margin requirement (only the 0.90–1.10
  corridor itself). If a margin m is required, c08b1d3d33ab fails it for any
  m > 0.004 pu, and the procedure has to be applied again with that limit;
  the Stage-II rows would need the per-window STS voltage extremes.
