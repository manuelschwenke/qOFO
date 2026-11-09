# 2026-11-09 (e): Night plan for the Ch 9/10 reruns on v5, and whether R2/R3 still represent their purpose on the new network

**Session:** "REF rerun plan" (Claude Code), 16:40 to ...
**Branch / worktree:** `res/ref-v5-2026-11-09` in `F:\qOFO_ref`. It was created from
`res/tuning-v5-2026-11-08` at d2ef395.
**Author, 2026-11-09:**
- "I agree with the proposed sequence."
- "We should ensure the two REF runs still represent what they should in the new base of the
  network."
- Hosts tonight: 30 workers on e5power and 30 on e5wimi.
- Gate: the final v5 weights come from the session "STS capabilities tuning analysis".

## 1. Plan (agreed)

| run | thesis target | scenario | variants | wall on the old base (S1) |
|---|---|---|---|---|
| R2 | `ch:case3:nominal:scarce` | E2 | S1, L2, O2, M, S2, O3 | 2.35 h (ref/0029) |
| R3 | `ch:case3:nominal:drift` | P2R | S1, L2, O2, Msel, S2, O3 | 2.53 h (ref/0028) |
| R3, two ramps | `tab:case3:r3:b03b05` | P2R_B03B05 | same | 1.91 h (laptop) |
| R1 / Case 1 | `ch:case_svr` | A | L2, S1m, S1, O0, O1, O2 | 1.93 h (case1/0004) |

- Each variant runs as one process, so the four runs need 24 processes.
- The wall time is set by S1 (SVR). On the new network it is expected at 2–5 h, which is an
  estimate.
- Gates:
  - **G1:** the final paramset, with λ_max(B) recomputed if the weights change, because M reads
    it.
  - **G2:** the H fix merged. Done on this branch, see § 2.
  - **G3:** E2 still feasible and representative. See § 3.

## 2. Code changes on `res/ref-v5-2026-11-09`

- **13951ba:** merge of `res/h-sn-mva-fix-2026-11-09` (5597ce4).
  - Why: the night plan merges it before thesis results (memory `h-sn-mva-slip`, Ch 11 plan B2).
    The author agreed to the sequence.
  - `F:\qOFO_v5` is not touched; its follow-ups (D1+D2 on e5power, the G3-off report) keep
    running there.
- **bfae475:** `experiments/ch_11_case2_multisystem/ref/scenarios.py` gets the
  `P2R_B03B05` entry.
  - It is a verbatim copy of the author's entry in `ref/ref_from_server/ref/scenarios.py`, where
    run 0001 of 2026-11-08 used it.
  - Why: the canonical `run_ref` must reproduce the two-ramp R3 table.
- No other code change.

## 3. Representativeness on `rural_700_north_res`

### 3.1 Scenario indices
- Unchanged on both networks: gen 1 = G3 (IEEE 32), gen 2 = G4.
- Lines 4/7/14/25 = 3-4, 4-14, 9-39, 17-27.
- Buses 2/3/4/7/11 = IEEE 3/4/5/8/12.

### 3.2 R3 (Period II): selection rule re-applied (fact)
- Ran the declared score script unchanged (thesis `00_daily_log/2026-11-07_ch10_period2_score.py.txt`)
  on the 15-min year series of each network. The series were built with
  `characterise_reference_period.build/classify_buses/exogenous_year` from F:\qOFO_ref.
- TS IBG:
  - `rural_700`: 540 + 500 + 508 + 650 MW (W1, W2, W3, W4).
  - north_res: 508 + 1500 MW (W3 and the park at IEEE 26).
  - All parks use profile WP10. The TS wind series therefore only changes scale: 2198 → 2008 MW.
    Of the score's terms, only the residual load changes.

| network | 6 Nov 08:00 | 11 Dec 07:00 | 8 Sep 10:00 | 21 Jun 09:00 | 13 Dec 10:00 |
|---|---|---|---|---|---|
| rural_700 | **1** (0.9615) | 2 (0.9545) | 3 (0.9541) | 11 | 15 |
| north_res | **1** (0.9614) | 2 (0.9545) | 3 (0.9541) | 11 | 14 |

- The `rural_700` ranks reproduce the declaration's table (1 / 3 / 11 / 15).
- **Period II stays the rule's first window.**
- TS wind up/down in the window: 879 / 957 MW, against 962 / 1048 before. The STS quantities
  are unchanged: export → import → export, −841 → +148 MW.

### 3.3 R2 (E2): unit commitment at the period start (fact)
- north_res, 5 Jan 2016 08:00, with the 360-min configuration of the REF.
  - Committed: G10, G3, G4, G7 and G1. G9 is off.
  - **G3 is committed, so E2's G3 trip is not void.**
- Area-2 STS: DSO_1, DSO_2, DSO_3, with 30 IBG. Area 2 has no TS IBG.

### 3.4 R2 (E2): plant-only bounds per stage (fact; plant model only, no controller result)
The declaration's own scripts are not available: `candE.py` and `cand_mismatch.py` are lost, the
"shared notes" of the 11-06 session. Rebuilt as `2026-11-09e_ref_repr_screen.py`, following
`scarcity_E_bounds{,2}.py`, which do survive in the scratchpad of session 29f1fa99.

- **Start point:** the runner's plant at t = 0, taken with `pre_loop_hook` from the 360-min REF
  configuration, so the unit commitment is the REF's.
- **frozen:** the events are applied; only the IBG Q(V) loops react.
- **TS-a2:**
  - G3's AVR at 1.05 pu if G3 is in service;
  - the area-2 MSCs in and the MSRs out;
  - the area-2 TS IBG at Q_max;
  - the STS IBG frozen at their t = 0 Q.
- **TS-all:** TS-a2, plus every AVR at 1.05 pu and every TS IBG at Q_max.
- **STS share:** the common fraction of area-2 STS IBG upward capability, on top of TS-a2, that
  returns the area-2 level (IEEE 3–14, mean − 1.03 pu) to 0.
- The script was also run on `rural_700` with today's code, as a check against the declaration:

| stage | declaration (rural_700) | rural_700, this script | **north_res** |
|---|---|---|---|
| | frozen (min V) / TS-a2 / TS-all / STS share | same | same |
| 0 t = 0 | – | −8.9 (1.016) / +36.0 / +62.0 / – | −5.5 (1.019) / +18.1 / +56.4 / – |
| 1 G3 out | −4.9 (1.019) / +56.0 / +97.7 / – | −4.0 (1.020) / +62.0 / +107.6 / – | −2.5 (1.021) / +26.8 / +92.1 / – |
| 2 G3 out + 600 | −58.2 (0.937) / −28.3 / +10.3 / 25 % | −32.1 (0.964) / −30.1 / +13.3 / 23 % | −39.6 (0.957) / **−68.5** / **−7.2** / **50 %** |
| 3 G3 back + 600 | −41.8 (0.952) / −18.0 / +6.0 / 29 % | −30.8 (0.965) / −18.0 / +8.0 / 26 % | −35.2 (0.960) / **−34.6** / +0.6 / **51 %** |
| 4 + 3-4 open | −59.2 (0.915) / −30.5 / −10.4 / 42 % | −34.5 (0.947) / −31.5 / −9.6 / 38 % | −42.6 (0.937) / **−53.9** / **−24.9** / **64 %** |

**Reading:**
- The script reproduces the declaration's TS-a2, TS-all and STS share on `rural_700` to within
  2–4 mpu or 4 percentage points. It is therefore a valid replacement for candE.py.
- The frozen level differs: −32 against −58 in stage 2. This is plausibly the Q(V) re-solve fix
  47e0668 (11-07), which came after the declaration. This is a hypothesis.
- **Against the old network, north_res makes E2 scarcer.** Area 2 lost its 500 MW TS IBG at IEEE 6
  (W2), so its own TS resources are G3 and the MSCs alone. Changes against the declaration:
  - the TS-a2 deficit doubles in stages 2–4;
  - TS-all already falls short in stage 2, not only in stage 4;
  - the STS share needed rises from 25–42 % to 50–64 %.
- **Status of the declaration's criteria and hypotheses on north_res:**
  - Stage 1 is not scarce: holds.
  - Stages 2–4 are scarce, and the STS can restore the level (share < 100 %): holds.
  - Frozen minimum ≥ 0.90 pu: holds, at 0.937 in stage 4.
  - Changed: the largest **TS-a2** deficit is now stage 2 (G3 out: area 2 has no synchronous
    machine and no TS IBG), not stage 4. Stage 4 still has the largest TS-all deficit and STS share.
  - Hypothesis 2's wording "only in stage 4 does TS-all fall short" no longer holds. Both stage 2
    and stage 4 need the STS beyond all TS resources.

### 3.5 R2: the line rule re-applied (fact)
End states with G3 back and 600 Mvar at IEEE 4, on north_res:

| open line | TS-a2 | TS-all | STS share | frozen min V |
|---|---|---|---|---|
| **3-4** | **−53.9** | −24.9 | 64 % | 0.937 |
| 2-3 | −47.4 | −12.6 | 59 % | 0.950 |
| 4-14 | −39.8 | −3.7 | 58 % | 0.934 |
| 17-27 | −38.9 | −8.8 | 55 % | 0.958 |
| 14-15 | −38.5 | −10.0 | 47 % | 0.958 |
| 3-18 | −37.0 | −4.2 | 53 % | 0.958 |

- **Line 3-4 is still the rule's choice:** it leaves the largest area-2 deficit at the TS-a2 bound.
- Every candidate meets the frozen bound (≥ 0.90 pu).
- 9-39 stays excluded, and 4-5 stays excluded as degenerate, as declared.

### 3.6 R2: mismatch loop gain (fact, v5 weights `ch9_selected_nr_2026_11_v5`)
μ = max Re eig(G_w⁻¹ H_mᵀ D H_t) over the continuous columns (der, pcc, gen).
- H_m is the controller's cached model.
- H_t is the same sensitivity code on the post-event plant, with frozen controls.
- D weights the V rows with g_v; the Q_PCC rows carry tso_g_q_pcc = 0.
- A machine out of service is a zero column of H_t. It is dropped from both matrices, because a
  zero column only adds an eigenvalue 0. Two cases: G3 after its trip, and G9, which the unit
  commitment keeps off.
- Script: `2026-11-09e_e2_mismatch.py`. A first version read H_t from the memoised `_H_cache`
  (dH = 0); the cache is now cleared.

| state | O2 area 1 / 2 / 3 | O3 area 1 / 2 / 3 |
|---|---|---|
| t = 0 | 0.75 / 0.43 / 0.30 | 0.75 / 0.43 / 0.30 |
| G3 out | 0.81 / 0.39 / 0.34 | 0.81 / 0.39 / 0.34 |
| G3 out + 600 | 0.84 / 0.49 / 0.36 | 0.81 / 0.43 / 0.35 |
| G3 back + 600 | 0.76 / 0.47 / 0.31 | 0.75 / 0.45 / 0.31 |
| + 3-4 open | 0.79 / 0.59 / 0.31 | 0.79 / 0.55 / 0.31 |

- **All TS-OFO states are below 2, at 0.84 at most.**
- Side finding: the cached model differs from the t = 0 plant by about 14 % (max |ΔH| / max |H|)
  even before any event. The controllers' H is taken at the bootstrap operating point, not at the
  period start.
- **Not evaluated:** CEN (variant M at λ_sel = 1.530). In the declaration Msel was marginal while
  G3 was out (2.17 / 2.26 at λ 1.405), which is hypothesis 5. Open.
- The values must be recomputed if the final weights differ from ae17a8cfdcce. It is one run of
  about 3 min.

### 3.7 R3 (Period II, P2R, P2R_B03B05): scarcity at the ramp hold (fact)
The same screen at 6 Nov 08:00 and at the hold of the ramps (10:35, i.e. 155 min). The
commitment is the same in both states: G10, G3, G4, G7 and G1. **G9 is off on north_res**; it was
on on `rural_700`.

| state | rural_700: frozen (min) / TS-a2 / TS-all / STS share | north_res: same |
|---|---|---|
| P2 t = 0 | −16.1 (1.003) / +27.0 / +56.3 / – | −9.3 (1.013) / +15.5 / +56.3 / – |
| P2R hold, +200/+100 at IEEE 12 | −12.6 (0.990) / +16.0 / +31.0 / – | −9.1 (0.994) / +11.0 / +29.6 / – |
| P2R_B03B05 hold, 2 × 300/300 at IEEE 3/5 | −27.8 (0.985) / −19.0 / −6.9 / 27 % | −31.8 (0.982) / −29.7 / −13.7 / 31 % |

- **P2/P2R stay non-scarce on north_res, as Period II is meant to be.**
- **P2R_B03B05 was already an area-2 scarcity case on `rural_700`.** It stays one on north_res, a
  little more strongly.
  - This was not stated in the thesis paragraph of 11-08, which compares it to R3 as a "second
    ramp".
  - It is a fact about the scenario, not a change from the network.

### 3.8 LOC screens (L2, 360 min, `run_ref`, runs 0001–0003 of F:\qOFO_ref)
- Launched 17:03 on CPUs 20 and 22 (cap 12 until 18:00); one core each from 18:03.
- Results: see § 6.

## 4. Finding: the REF dead-band gate failed (v5 follow-ups, 18:18)
- D1 on north_res with v5 (`F:\qOFO_v5` run 0003, `d1/selection.json`): d_max = 0.0027 pu, so
  **Δ* = 0.0075 pu**. On `rural_700` with v4 it was d_max 0.0052, so Δ* = 0.010.
- REF and Case 1 run with the configuration default `tso_qv_deadband_pu` = `dso_qv_deadband_pu` =
  0.010. The v5 tuning ran with that value too.
- By the author's rule of 2026-11-05 ("we leave 0.01 and we drop the one design width step more
  argument", DESIGN_STEPS = 0), the REF dead band equals D1's design width.
- **Author decision needed:**
  - keep 0.010 (as tuned; it is ≥ Δ*), or
  - run REF and Case 1 at 0.0075 (D1's design width, not the tuning condition).

## 5. e5power
- D1+D2 ended 17:14, so e5power is free.
- `F:\qOFO_runs\qOFO_ref` was cloned from a bundle of `res/ref-v5-2026-11-09` at bfae475.
- The final paramset still has to be fetched there before Case 1 starts.

## 6. Open / pending
- G1: final weights from "STS capabilities tuning analysis". Nothing new committed as of 18:40.
- The dead-band decision (§ 4).
- The LOC screens (§ 3.8).
- CEN's mismatch gain (§ 3.6).
- The thesis text of R2 (stages, hypotheses 2/4) and R3 (commitment without G9, TS wind
  1.29 → 0.25 GW becomes ×0.914) changes with the rerun.
