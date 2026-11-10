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
- ~~Side finding: the controllers' H is taken at the bootstrap operating point, not at the period
  start.~~ **Corrected 18:55:**
  - The runner rebuilds the models after initialisation, at t = 0 of the period, with the run's
    own unit commitment (`multi_tso_dso.py` ≈ l. 2519). Checked with `model_vs_start` (scratchpad):
    `shared_jac.net` equals the loop-start plant (|ΔV| ≤ 4e-10 pu; commitment, taps, banks, IBG and
    loads identical).
  - The models are never refreshed in the run: `sensitivity_update_interval` = 10⁶ and
    `refresh_shared_jac_on_tso` = False. Banks enter as rank-1 updates.
  - What differs from the plant is the **structure**. `local_sensitivities_tso` = True, so each
    TSO-OFO uses a reduced zone net: Ward/Thevenin tie sources (`WARD_TIE_TH_SRC_*`), with the STS
    as PQ loads at the MV bus.
  - On common buses that net differs from the plant by up to 0.011 / 0.043 / 0.015 pu (zones
    1/2/3).
  - The t = 0 row of μ (0.75 against the model's own λ of 0.68 in zone 1) is therefore the
    reduced-model error. It is exactly the mismatch the criterion is meant to measure.
  - The author asked (18:5x) whether the model should always be built from the REF start point. It
    already is, and the v5 tuning ran the same way, so nothing changes.
  - A second variant with the runner's PF settings for the post-event plants (max_iteration 100)
    failed with ControllerNotConverged in the event states. The table above uses the declaration's
    settings (max_iter 300).
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
- **Decision (author, 2026-11-09 ~18:50):** "We will use 0.01 width and argue again with one value
  higher for margin reasons."
  - REF and Case 1 run at `tso_qv_deadband_pu` = `dso_qv_deadband_pu` = 0.010, the configuration
    default. No code change is needed.
  - Argument in the thesis: the design width is D1's Δ* = 0.0075 plus one candidate step for
    margin, which gives 0.010.
    - This reinstates the design-margin step that was dropped on 2026-11-05 (DESIGN_STEPS = 0,
      97e5a12).
    - On `rural_700` the step was dropped because Δ* already equalled 0.010.
  - The D1 run of record, `F:\qOFO_v5` 0003, records `design_margin_steps = 0` with
    Δ* = 0.0075. The "+1 step" is an argument on top of that record. D1 is not re-run.
  - Ch 8's D1 paragraph (thesis) has to state it; that is not done here.

## 5. e5power
- D1+D2 ended 17:14, so e5power is free.
- `F:\qOFO_runs\qOFO_ref` was cloned from a bundle of `res/ref-v5-2026-11-09` at bfae475.
- The final paramset still has to be fetched there before Case 1 starts.

## 5b. Launch (author 18:5x: "tuning is through, start the REF runs and the chapter 9 run")
- **Weights:** the tuning session confirms GO 16:37 with ae17a8cfdcce, i.e.
  `ch9_selected_nr_2026_11_v5`, with λ_max(B) 1.530 in its provenance. `SELECTED_PARAMSET` on this
  branch points to it.
- The tuning session's latest note: in the G3-off report, v5 shows "more oscillation aborts".
  That is relevant to R2's 61 min with G3 out, and is reported, not a gate.
- **e5wimi, F:\qOFO_ref, code 27f0651.** Launched with `launch_night.ps1` (pythonw,
  Win32_Process.Create), 18 processes on CPUs 0–22, 30–38 even, plus CPU 1:

| run | scenario | variants | started |
|---|---|---|---|
| ref/0004_2026-11-09_185815_run_ref_v5_E2 | E2 | S1, L2, O2, M, S2, O3 | 18:58 |
| ref/0005_2026-11-09_185843_run_ref_v5_P2R | P2R | S1, L2, O2, Msel, S2, O3 | 18:58 |
| ref/0006_2026-11-09_185912_run_ref_v5_P2R_B03B05 | P2R_B03B05 | S1, L2, O2, Msel, S2, O3 | 18:59 |

  - Checked in `config_repr.txt`: scenario `rural_700_north_res`; dead band 0.01/0.01;
    `local_sensitivities_tso` True; M `central_lambda_target` 1.5301.
- **e5power, F:\qOFO_runs\qOFO_ref, code bfae475.** The later commits on the branch are docs only.
  - `run_case1 --only L2,S1m,S1,O0,O1,O2 --jobs 6 --scenario A --tag v5_A`.
  - Run `ch_10_case1_svr/0001_2026-11-09_185947_run_case1_v5_A`, started 18:59.
  - Driver console: `results\_case1_v5_A_console.log` there.
  - The detached watcher `pull_case1.py` (scratchpad) pulls the run into F:\qOFO_ref under the same
    path when it ends.
- **Expected end** (S1 is the bottleneck; new-network factor 1.0–1.6):
  - Case 1 ≈ 21:00–22:00;
  - R3 and R3-two-ramp ≈ 21:30–23:00;
  - R2 ≈ 22:00–00:30, worst case ≈ 02:00.
- **Supersedes, after the result** (`thesis_runs archive`, never delete): the v4 runs on rural_700.
  - ref/0028 and ref/0029 (F:\qOFO_res);
  - case1/0004 (F:\qOFO_res);
  - the server-mirror run ref_from_server/0001 (two-ramp).

## 5c. LOC screens: result (19:12)
- All three runs are `ok`: 1080 records = 360 min. Every step converges under LOC on north_res.
  - E2 took 2.14 h, P2 1.80 h and P2R_B03B05 1.82 h. They ran on 2 shared CPUs until 18:03.
- Area-2 minimum per stage under L2:

| E2 stage | north_res min / median | old ref/0029 L2 min / median |
|---|---|---|
| G3 out + 600, 91:40–100 | 0.906 / 0.920 | 0.929 / 0.935 |
| same, 100–152:20 | **0.893** / 0.905 | 0.928 / 0.942 |
| 3-4 open, 241–250 | 0.917 / 0.917 | 0.945 / 0.945 |
| same, 250–360 | 0.906 / 0.912 | 0.935 / 0.943 |

- **Under LOC, area 2 falls below 0.90 pu for about 10 min.** That is 29 samples, between 101:40
  and 110:40, plus one at 114:40, while G3 is out with the 600 Mvar load.
  - This is not a run failure: the MC rule does not count a voltage outside 0.90 as a failure
    either.
  - It is the uncoordinated baseline, 25–35 mpu lower than on `rural_700`. That is consistent with
    the scarcer bounds of § 3.4.
  - Hypothesis 6 ("LOC near 0.92 pu after the line trip") becomes 0.906–0.917 after the trip, and
    0.893 at the low point in stage 2.
  - The declared static frozen-control criterion (≥ 0.90) still holds at 0.957. That check
    includes the relays and IBG Q(V) loops at equilibrium, not the transient after the step.
- P2: area-2 min 1.010 pu, no excursion. P2R_B03B05: area-2 min 0.966 pu at the ramp hold.

## 5d. Ch 10 MC queued behind the reference runs (author ~19:20)
- **Author:** "(a), then only queue 50 mc runs tonight. I can add another 50 later, correct? Queue
  them, to only start after the ref runs."
  - (a) means the declared protocol of 2026-11-07, with G3-off episodes in their drawn
    proportion. No G3-on filter: the 11-04 scope filter was tied to v4's design envelope, and v5's
    λ bank contains G3-off states.
  - **50 + 50: yes.**
    - `run_mc` continues the newest campaign of the same `--tag` and skips every run that has its
      JSON. Re-running with `--n 100` later adds exactly the missing episodes. The OFO+ extension
      uses `--n-o3 300`.
    - Statistically this is execution in two batches, not an extension: N = 100 stays fixed in
      advance, and the second 50 run whatever the first 50 show.
    - Tests and intervals are computed only on the full window. Looking at the first 50 is
      allowed, but not a decision on whether to continue.
- **Queue** `mc_queue.py`, detached pythonw in this session's scratchpad (pid in `mc_queue.pid`,
  log `mc_queue.log`). No repository code change.
  - **e5power: S1, O2.** Starts when the Ch 9 run there has ended, through `mc_remote_wrapper.py`.
    - The wrapper sets the remote root to `F:\qOFO_runs\qOFO_ref` and adds `--n 50`; the module
      default root is `qOFO_res`, and its remote call has no `--n`.
    - Call: `run_mc_remote --variants S1,O2 --jobs 30 --buffer 6 --tag v5nr_e5power`, pulled back
      into `results/ch_11_case2_multisystem/mc/`.
    - Dry run 19:25: e5power at bfae475; no Python file differs.
  - **e5wimi: L2, M, S2, O3 and the prescreen.** Starts when REF 0004–0006 have all 18 status
    files and their launchers have exited.
    - Call: `run_mc --new --variants L2,M,S2,O3 --n 50 --n-o3 50 --buffer 6 --jobs 30 --tag v5nr`.
    - 30 CPUs at night, 10 from 08:00 (e5wimi day rule), 20 from 18:00. Only its own process tree
      is re-pinned.
  - The split follows amendment A1 of 11-07.
- **Estimate**, old-base minutes per episode × 56 episodes, × 1.0–1.6 for the new network:
  - e5power: ≈ 90–140 CPU-h at ≈ 18 core-equivalents (another user runs ≈ 34 Python processes
    there). From ≈ 22:00 it ends ≈ 03:00–06:00.
  - e5wimi: ≈ 95–150 CPU-h at ≈ 19.5. From the REF end (≈ 00:30–02:00) it ends ≈ 05:30–09:30.
    After 08:00 it runs at 10 CPUs.
- **Later, the second 50:** the same two calls with `--n 100` (local `--n-o3 300`, no `--new`).
  Then `merge_mc.py` and the final invocation with all six variants, as on 11-07/08.
  Superseded by § 5e.

## 5e. Status 21:35 and the full N = 100 tonight (author ~21:40)
- **Status 21:35.**
  - REF 0004 (E2): 6/6 ok, 1080 records each.
  - REF 0005 and 0006: 5/6 ok each; the CEN (Msel) runs at t = 272 and 327 min.
  - Case 1 (`ch_10_case1_svr/0001_2026-11-09_185947_run_case1_v5_A`): 6/6 ok, pulled 21:00.
  - e5power MC (S1, O2): started 21:03, prescreen done 21:17.
  - Console zone minima, first look only (the 3-min console samples; not the evaluation):
    - E2: M 0.970, O3 0.936, S2 0.925, O2 0.912, L2 0.895, S1 0.894.
    - P2R_B03B05: O3 1.001, S2 0.991, O2 0.987, S1 0.975, L2 0.967.
    - Case 1: O2 0.965; O0/O1/S1/S1m 0.939; L2 0.931.
- **Correction of the 21:35 speed statement.** "e5power ends before midnight" rested on the first
  S1 results (405–1368 s). Those were the earliest finishers of a pool of 30, so the sample is
  biased towards short episodes. At 21:42 only 8 of the ≈ 30 S1 episodes started at 21:17 had
  ended.
  - Calibration on the 11-07 campaign (rural_700, v4):
    - e5power share S1 + O2, 130 seeds: 143 CPU-h in 6.5 h, ≈ 22 effective workers.
    - e5wimi share L2 + M + S2 + O3: 344 CPU-h in 13.6 h, ≈ 25 effective workers.
  - Medians per episode (s): L2 2605, M 2564, O2 1284, O3 836, S1 2834, S2 1065.
- **Author ~21:40:** "If it is faster then expected, we can run 50 additional MC runs overnight in
  queue, mainly on e5power."
  - The second 50 complete the N = 100 fixed in advance. The timing is decided on throughput
    only; no result of the first 50 was evaluated.
  - The e5wimi share had not started, so the split was changed by variant. The declared protocol
    allows that (A1 of 11-07; `merge_mc` copies per (seed, variant)).
  - A split by seed would rerun episodes 1–56 of a moved variant, since `run_mc` has no seed
    offset. So L2 moves to e5power for all 106 episodes.
- **Queue v2** `mc_queue2.py` (copy alongside: `2026-11-09e_mc_queue2.py`). Detached pythonw,
  pid in `mc_queue2.pid`, same log `mc_queue.log`.
  - `mc_queue.py` was stopped at 21:46. Its running child, the first e5power call (wrapper pid
    103212), was left alone.
  - **e5power** (tag `v5nr_e5power`, the same campaign): after the first call has ended and been
    pulled back, it runs `mc_remote_wrapper.py 100 --variants S1,L2,O2 --jobs 30 --buffer 6`.
    - S1 and O2 get seeds 57–106; L2 gets all 106.
  - **e5wimi** (tag `v5nr`, new campaign): after REF 0004–0006, it runs
    `run_mc --new --variants M,S2,O3 --n 100 --n-o3 100 --buffer 6 --jobs 30`, with the same
    pinning rule.
    - OFO+'s further 200 episodes (`--n-o3 300`) stay for later.
  - **Run number.** The e5power campaign is `0001_2026-11-09_210315_run_mc_v5nr_e5power` there.
    An empty local folder of that exact name reserves 0001 here, so the local campaign becomes
    0002 and the pull later lands in the reserved folder. `run_mc_remote`'s `reserve_number`
    only guards the reverse order.
- **Estimate at the 11-07 rates:**
  - e5power: 62 CPU-h (first call) + 121 CPU-h (second) at ≈ 22 effective workers, ending
    ≈ 05:30 (range 04:30–07:00).
  - e5wimi: 173 CPU-h at ≈ 25, from ≈ 22:20, ending ≈ 05:15 (range 04:00–07:30; M's tail reaches
    2.5 h per episode).
- **Morning:**
  - `merge_mc.py` from the e5power folder into the local `v5nr` campaign.
  - Then the final invocation with all six variants for the replacements under rule (b).
  - Then `--report` and `evaluate_mc`.

## 5f. Night result and the closing pass (2026-11-10, 07:26–07:36)
- **Night.**
  - REF 0005 and 0006 Msel ok; queue v2 started the e5wimi share at 22:16.
  - e5power first call ended 23:47, pulled 23:52. Second call (S1, L2, O2, --n 100) 23:55–04:05,
    pulled 04:08 into the reserved `mc/0001_2026-11-09_210315_run_mc_v5nr_e5power`.
  - e5wimi `run_mc` (M, S2, O3) exit 0 at 05:56 after 27 606 s, as `mc/0002_2026-11-09_221649_run_mc_v5nr`.
  - Every share ran 106 episodes per variant. Wall medians (s): M 4697, S1 3055, L2 1724,
    O2 1355, S2 813, O3 594.
  - e5power side note: `reserve_number` created an empty `0002_..._run_mc_v5nr` there and logged
    "numbers the run 0003". The remote run continued 0001, so the empty folder is only a number
    placeholder.
- **Merge 07:28.** `merge_mc` 0001 → 0002: 318 runs (S1, L2, O2) copied, no prescreen mismatch
  (dry run first).
- **`run_mc --report` 07:28** (all six variants, rule (b)):
  - 14 episodes replaced. The window reaches seed 20261124; the 8 seeds 20261117–20261124 have not
    yet been run by all six.
  - Infeasible: 20261079.
  - Failures per variant (105 episodes each; cause):

    | variant | failed | cause |
    |---|---|---|
    | O2 | 9 | tap hunting |
    | S1 | 6 | tap hunting |
    | O3 | 4 | 3 tap hunting, 1 PF |
    | S2 | 3 | 2 tap hunting, 1 PF |
    | M | 2 | PF |
    | L2 | 1 | tap hunting |

  - The paired comparisons in this report are on 92 complete episodes, not the final window; not
    to be used.
- **Closing pass, launched 07:33** (detached via `launch_ref.py mcFinal`, CPUs 0,2,…,18,
  `--jobs 10`, the e5wimi day rule):
  `run_mc --variants M,S1,L2,O2,S2,O3 --n 100 --n-o3 100 --buffer 6 --tag v5nr`.
  - It prescreened 10 further draws (the batch overshoot of `_prescreen_until`; 114 valid of 143)
    and scheduled 48 runs.
  - A failure among the 8 triggers a further replacement round in the same call.
  - Estimate: 27 CPU-h at 10 workers, ≈ 10:30–11:00. With one more round, ≈ 12:00–14:00.

## 5g. Copy to Z: for the thesis session (2026-11-10, author: "copy them into the dissertation text then")
- Robocopy (full path `C:\Windows\System32\Robocopy.exe`; not on PATH in this shell). Same
  relative path under `Z:\Python_Projekte\qOFO\results\`, none existed before. File count and
  bytes are equal to F: for every run.

  | run | files | MB |
  |---|---|---|
  | `ch_11_case2_multisystem/ref/0004_2026-11-09_185815_run_ref_v5_E2` | 32 | 62.6 |
  | `ch_11_case2_multisystem/ref/0005_2026-11-09_185843_run_ref_v5_P2R` | 31 | 62.4 |
  | `ch_11_case2_multisystem/ref/0006_2026-11-09_185912_run_ref_v5_P2R_B03B05` | 31 | 62.4 |
  | `ch_10_case1_svr/0001_2026-11-09_185947_run_case1_v5_A` | 29 | 60.3 |

- Not copied:
  - the LOC screens ref/0001–0003 (screens, not thesis results);
  - the MC, which is still in its closing pass.
- **Run numbers.** `qOFO_ref` is a fresh tree, so its counters start at 0001. On Z:,
  `ch_10_case1_svr/0001_...` now exists twice (v4_A of 11-05 and v5_A of 11-09), and ref `0006`
  repeats the counter of the archived `0006_2026-10-27_181840_run_ref`. The full folder names
  are unique; cite runs by full name.
- **State of the runs:**
  - The REF runs carry `metrics.json` and `report.md`.
  - The Case 1 run has no run-level report yet. `report_case1.py` is needed for
    `tab:case_svr:reference` and the equivalence and Pareto figures.
  - The superseded v4 runs on Z: and F:\qOFO_res are not yet archived.

## 6. Open / pending
- G1, the weights: resolved (§ 5b).
- The dead band: decided (§ 4).
- The LOC screens (§ 3.8).
- CEN's mismatch gain (§ 3.6).
- The thesis text of R2 (stages, hypotheses 2/4) and R3 (commitment without G9, TS wind
  1.29 → 0.25 GW becomes ×0.914) changes with the rerun.
