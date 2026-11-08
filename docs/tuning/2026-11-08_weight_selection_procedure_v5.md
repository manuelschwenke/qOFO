# Controller weight selection, procedure v5: study network north+reserved, STS reactive engagement

**Status:** FROZEN 2026-11-08, before the first simulation of its campaign. Applied once by
`experiments/ch_09_parameter_selection/ch_9_17_procedure_v5.py`. Its output is
`configs/paramsets/ch9_selected_nr_2026_11_v5.json`, whatever it is.

**Why v5.**
- The study network is now `rural_700_north_res` (author 2026-11-08; daily log
  `2026-11-08b_network_scarcity_screen.md`). Every operating point, λ and θ value and the
  weight sets v1–v4 belong to `rural_700`.
- With the v4 weights the TS controllers leave the STS nearly idle. Per Mvar the PCC setpoint
  was priced 61× the TS IBG. The v4 ranking put all 27 feasible candidates inside the 2 %
  e_v(TS) band, so the e_v(DS) tie-break chose the largest g_w_pcc in the box (Spearman
  −0.76). Diagnosis: daily log `2026-11-08_sts_underuse_diagnosis.md`.
- On the new network zone 2 has no TS IBG. With G3 decommitted (the scarce state, ≈ 15 % of
  hours) the PCC setpoints are its only continuous actuator. g_w_pcc then sets the zone's loop
  gain directly: at v4 (0.873) λ_pcc ≈ 0.16, i.e. ≈ 14 TS periods to remove 90 % of an error.

**Author decisions 2026-11-08:** "Bound g_w_pcc"; G3-off states "in the λ bank, plus a
report"; experiments B, C, D1, D2 with 30 workers on e5power and 20 on e5wimi.

## What is unchanged

Procedure v4 (`2026-11-04_weight_selection_procedure_v4.md`) with all its amendments:
- Stage I of v3: STS weights from π_q = 10, π_v,DS = 1, λ_STS 1.4, θ_STS ladder 0.01 → 0.015 →
  0.02; bank price from Δv_eng,shunt = 0.005; g_w_gen = 5·10⁶; g_v, g_w_der, g_w_tso_oltc from
  `stage1_v2_uc_2026_11_pq10`; Stage-I verification by the settling test.
- Stage II-a (512 Sobol draws, seed 20260910), II-b (≤ 4 rounds), II-c (bank price), completion
  on W2, ranking e_v(TS) with the 2 % band and e_v(DS) tie-break, settling confirmation p95 ≤ 27,
  W5 report, go criteria (e_v(TS) ≤ 0.01, couplers act in every search window, banks after
  events in ≥ 2 windows).
- Hunting tolerance 0, dwell < 1 h, start-up exemption 900 s, early stop.
- Search windows W1, W3, W4, W6 (12 h, evented); λ without the bank columns, ceiling 1.546;
  θ_TS,min ≤ 0.009 pu.
- `--config-set sts_coupler_hold_after_bank=1`, `shunt_ff_ledger = True`, DSO band 0.92–1.08.

## Changes

1. **Network.** `rural_700_north_res`, DSO_3 ×2 as before. W1–W6 are captured anew (12 h); all
   six commit G3.
2. **λ bank.** W1–W6, plus the grid of `ch_9_13` (months 1–12 × days 8, 22 × 02/08/14/20 h, 96
   starts) captured on the new network twice: with a 12-h horizon (as v4) and with a 3-h
   horizon. The 3-h horizon is the MC/REF episode length, so the unit commitment then
   decommits G3 as it does in those runs. **Both G3 states enter.** v4's envelope "G3
   committed" is dropped. The banks stay out of B.
3. **STS reactive engagement: bounds on g_w_pcc.** On the G3-off points 𝒫_off of the bank, with
   κ_p = λ_max(g_V · H_V,pcc,pᵀ H_V,pcc,p) for TS zone 2 (the PCC columns of zone 2 alone):
   - **responsiveness (upper bound):** min_{p∈𝒫_off} κ_p / g_w_pcc ≥ Λ_STS = 0.5, i.e. in the
     scarce state the STS removes at least half of a PCC-direction voltage error per TS period;
   - **contraction (lower bound):** the λ screen on the whole bank (it contains 𝒫_off);
     necessary for it, max_p κ_p / g_w_pcc ≤ 1.546.
   - The II-a box of g_w_pcc is this interval, log-uniform:
     [max_p κ_p / 1.546, min_p κ_p / 0.5].
   - The Stage-I g_w_pcc is its geometric centre.
   - II-b draws keep the v4 rule (f × the II-a span around the current best). Every draw must
     pass the responsiveness bound as well as the v4 screen.
   - If the interval is empty, the campaign stops before Stage II.
4. **G3-off report (not ranked).** After a GO, the winner runs in the mini-MC G3-off episodes of
   `rural_700_north_res` (the 21 seeds of e5power run 0007, 180 min) with OFO and OFO+. Failures
   and OFO+ − OFO area-2 e_v are reported. This is the follow-up "G3-off report", next to C
   (settling figure) and D1 + D2.

## Follow-ups (after GO, `ch_9_18_followups_v5.py`)

1. Record the config fields (`sts_coupler_hold_after_bank = 1`) and λ_max(B) in the result;
   SELECTED_PARAMSET → v5 (committed on `res/tuning-v5-2026-11-08`).
2. **C:** the settling figure from the Stage-I verification and the winner's confirmation
   (`make_settling_cdf.py`, copied to a dated folder in the thesis clone; nothing committed there).
3. **D1 + D2** (`ch_9_4_deadband`) on e5power with 30 workers.
4. **G3-off report** on e5wimi with 20 workers, in parallel with 3.

## Hosts

e5wimi: 20 workers (mask 0x5555555555, one logical CPU per physical core). e5power: 30 workers,
worktree `F:\qOFO_runs\qOFO_v5` (branch `res/tuning-v5-2026-11-08`).
