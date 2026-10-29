# Tap engagement at the selected controller weights (`ch9_selected_band_2026_10`)

**Type:** decision note for the author. Analysis of existing records only.
**Nothing changed:** no weight, parameter set, procedure or producer was
modified, and no simulation was run. One controller build at W1 was made to
read the sensitivity columns; it did not step the plant.
**Date:** 2026-10-28.
**Trigger:** the author observed, running `experiments/run_multi_system_ofo.py`,
that the STS coupler taps barely move although DSO_4 sits at about
1.05–1.055 pu (median) and up to about 1.075 pu, and that the TS taps also
engage rarely. The author suspects the engagement rule is set too high.

**Sources:**
- Run of record
  `results/ch_09_parameter_selection/0001_2026-10-25_215433_weight_selection_driver_band/`:
  - `margin_check/*.pkl`: fde2e81af6b1 = selected set, 3cb66225e1a7 =
    Stage-I point; W1–W4, W6; 12 h; 20 s records; 10/10 reproduce the
    stored e_v(TS).
  - `selection/union_2b_round4/` (242 candidates) and
    `selection/selection_2b_round4.json`.
  - `expC/criterion_results.json`.
- Operating points `results/stage1/operating_points/op_rural_700_net82afed5b_W*.npz`,
  the cache that run loaded (`stage2a.log`).

**Scripts:** `analysis/tap_engagement/{thresholds,usage,candidates,capture_H,followups}.py`.
**Outputs:** `results/analysis/2026-10-28_tap_engagement/` (`*.json`, `*.log`).
`thresholds.py` first reproduces `stage1_start_10_2026_banks` from the cache
with `design_stage1(d_engage_ts=0.0075, d_engage_sts=0.015, d_engage_shunt=0.010)`.
All ten weights match exactly (relative error 0).

---

## 1. Short answer

**Commit rule** (Stage-I machinery, `tuning/stage1_screen.py`). The MIQP step
objective is $w^\top G_w w + c^\top w$ with diagonal $G_w$ and no curvature
term. With no output limit active, a free integer column $i$ therefore
commits one step iff $|c_i| > g_{w,i}$. For a tap or bank column the
gradient is

$$
c_i = 2\,g_V \sum_j e_j\,H_V[j,i],
\qquad e_j = V_j - V_\mathrm{ref},\quad V_\mathrm{ref} = 1.03\ \mathrm{pu}.
$$

This holds for STS and TS columns:
- STS: `dso_gamma_oltc_q = 0` (`make_config`), so the coupler columns carry
  **no** interface-Q gradient.
- TS: `tso_g_q_pcc = 0`, so the TS integer columns carry none either.

The interface-Q term is therefore **excluded by construction**, not by
approximation. Under a uniform offset the commit threshold of column $i$ is

$$
\theta_i = \frac{g_{w,i}}{2\,g_V\,\big|\sum_j H_V[j,i]\big|}.
$$

All 12 STS coupler columns have one sign on all 10 observed buses (negative;
checked on the captured $H$ at W1). Hence
$|c_i| \le 2 g_V \max_j|e_j|\,|\sum_j H_V[j,i]|$, and
$\max_j |e_j| > \theta_i$ is a **necessary** condition for a
gradient-driven commit, not only an approximation.

**Effective thresholds.** Because $\theta_i \propto g_{w,i}/g_V$, the
selected set rescales the Stage-I thresholds per class:

| class | Stage-I → selected | factor | strongest column: Stage I → selected [pu] | all columns, selected [pu] |
|---|---|---|---|---|
| STS couplers | g_w 16.64 → 5.834, dso_g_v 10⁴ → 659.9 | (0.3506 × 15.15) = **5.31** | 0.0150 → **0.0797** | 0.0797–0.1478 |
| TS taps | g_w 21.37 → 25.59, g_v 4·10⁴ fixed | 1.20 | 0.0075 → 0.0090 | 0.0090–0.0629, two columns ∞ |
| MSC/MSR banks | g_w 27.21 → 23.81 | 0.875 | 0.0100 → 0.0087 | 0.0087–0.0135 |

**Consequence for the STS.** The upward headroom of the DSO operating band
is $1.08 - 1.03 = 0.05$ pu, which is below 0.0797 pu. While all STS buses
are inside the band, **no coupler can commit on the voltage gradient**. A
coupler can then move only through an active output constraint (the soft
band, slack price 10⁹).

Measured over 5 × 12 h:
- STS operations: 19 at the Stage-I point, **1** at the selected weights.
- The single move (DSO_4|trafo_10, W1, t = 10.87 h) occurred at
  v_max = 1.0818 pu, above the band.
- At that move, max|e| = 0.052 pu, below its θ = 0.097 pu. The move is
  therefore **not** gradient-driven.

**The parent session's hypothesis is confirmed for the STS:** 0.08 pu for the
strongest coupler, up to 0.148 pu for the weakest.

**It does not explain the TS.** Their thresholds rose by only 20 %. Low TS
tap use has two causes:
- TS voltages stay within ±0.012 pu (area mean) of 1.03 pu.
- Only 2 of 7 TS tap columns are within reach (θ 0.009–0.010 pu); two have
  an identically zero voltage column in the TS model.

**Why the search moved there.** θ_STS ∝ g_w_dso_oltc / dso_g_v. Stage II
spans ±1.5 decades on dso_g_v but ±0.5 on g_w_dso_oltc, so 76 % of
var(log θ_STS) over the 242 candidates comes from dso_g_v. The filter bounds
θ_STS from below only, through tap hunting; nothing bounds it from above. The
ranking is lexicographic on e_v(TS). Along the incumbent path, dso_g_v fell
from 10⁴ to 660 and θ_STS rose from 0.015 to 0.080. e_v(TS) improved by
20 % and e_v(DS) worsened by 74 %. In the last step alone, 2.2 % of e_v(TS)
was bought with +32 % of e_v(DS). The same mechanism parked the DSO taps on
2026-08-20 (`docs/daily_log/08_2026/2026-08-20_dso_oltc_inactivity_at_the_tuned_point.md`).

**Recommendation: option (c1).** Amend the filter with
θ_STS,min ≤ V_max,band − V_ref = 0.05 pu and re-select over the 242
existing candidates. This needs no new Stage-II simulation. The winner is the
same for every bound tested: **301c285ee52b**. Its settling check is the one
missing measurement; it was aborted, not failed. The fallback is
**c08b1d3d33ab**, already PASS. Details and costs are in §5.

---

## 2. Assumptions

1. **Threshold model.**
   - θ is the systematic-offset reading of the Stage-I rule.
   - The strongest column of a class is the one with the largest
     $|\sum_j H_V[j,i]|$ over W1–W6, including W5, as in `design_stage1`.
   - Thresholds are computed on the cached operating points; the plant moves
     within a window, so they are point values.
   - Over W1–W6 each coupler's θ varies by ±2.5 %.
2. **No active output limit.** The commit rule holds when no output limit is
   active. With the band, a current limit or an interface capability limit
   active, a column can commit below its θ. This is what happened at the one
   STS move at the selected weights.
3. **Device names.** Device names follow the controller column order.
   - TS taps: `z<zone>[k]`, order of `oltc_trafo_indices`.
   - TS banks: `z<zone>b[k]`, order of `shunt_bus_indices`.
   - STS couplers: `DSO_n|trafo_m`, order of `coupling_trafo_indices`.

   The order was verified by the captured controller configuration (W1) and
   matches the keys of the records and of `moves`.
4. **Voltage proxy.** The records hold only group minimum, mean and maximum
   voltages per STS/zone, not per-bus values.
   - The exact $\sum_j e_j H_V[j,i]$ at a move cannot be reconstructed.
   - The arithmetic mean deviation is reported as a proxy.
   - The maximum deviation is used for the rigorous necessary condition.
5. **Event coincidence.** A move counts as event-coincident if it occurs
   0–30 min after an event of `_condition_events` (12 h windows: W1/W3 gen-2
   trip 17 280 s, restore 20 880 s; W2/W4 ramp at bus 5 from 19 440 s; W6
   shedding at bus 27 from 19 440 s). Such a move cannot be assessed: the
   record before the move predates the event.
6. **Hunting attribution.** Hunting is attributed to STS or TS from the
   transformer names in `hunting_detail` of W1–W4, W6 (W5 is not filtered).

---

## 3. Findings (established from the records and the code)

### 3.1 Threshold per device (systematic reading; range over W1–W6) and operations

Operations are counted over W1–W4, W6 (5 × 12 h) as Stage I / selected.

| device | element | θ Stage I [pu] | θ selected [pu] | ops S1 / sel |
|---|---|---|---|---|
| DSO_1\|trafo_0 | Coupler3W_TN6_HV0 | 0.0254–0.0264 | 0.1350–0.1400 | 0 / 0 |
| DSO_1\|trafo_1 | Coupler3W_TN7_HV3 | 0.0185–0.0193 | 0.0984–0.1023 | 3 / 0 |
| DSO_1\|trafo_2 | Coupler3W_TN4_HV8 | 0.0159–0.0169 | 0.0847–0.0897 | 2 / 0 |
| DSO_2\|trafo_3 | Coupler3W_TN11_HV0 | 0.0264–0.0270 | 0.1400–0.1433 | 0 / 0 |
| DSO_2\|trafo_4 | Coupler3W_TN13_HV3 | 0.0180–0.0189 | 0.0955–0.1005 | 1 / 0 |
| DSO_2\|trafo_5 | Coupler3W_TN3_HV8 | 0.0156–0.0164 | 0.0828–0.0869 | 3 / 0 |
| DSO_3\|trafo_6 | Coupler3W_TN10_HV0 | 0.0242–0.0248 | 0.1284–0.1316 | 0 / 0 |
| DSO_3\|trafo_7 | Coupler3W_TN9_HV3 | 0.0186–0.0192 | 0.0990–0.1020 | 1 / 0 |
| DSO_3\|trafo_8 | Coupler3W_TN12_HV8 | 0.0162–0.0170 | 0.0862–0.0901 | 1 / 0 |
| DSO_4\|trafo_9 | Coupler3W_TN23_HV0 | 0.0274–0.0278 | 0.1458–0.1478 | 0 / 0 |
| DSO_4\|trafo_10 | Coupler3W_TN20_HV3 | 0.0179–0.0186 | 0.0949–0.0988 | 3 / 1 |
| DSO_4\|trafo_11 | Coupler3W_TN22_HV8 | **0.0150**–0.0157 | **0.0797**–0.0835 | 5 / 0 |
| z1[0] | MachineTrf gen0 → bus 1 | 0.0114–0.0119 | 0.0136–0.0143 | 0 / 0 |
| z1[1] | MachineTrf gen7 → bus 28 | 0.0083–0.0084 | 0.0100–0.0101 | 6 / 1 |
| z2[0] | MachineTrf gen1 → bus 9 | ∞ (H_V column = 0) | ∞ | 0 / 0 |
| z2[1] | NetworkOLTC bus 11–10 | 0.0503–0.0525 | 0.0602–0.0629 | 0 / 0 |
| z2[2] | NetworkOLTC bus 11–12 | 0.0357–0.0366 | 0.0427–0.0438 | 0 / 0 |
| z3[0] | MachineTrf gen2 → bus 18 | ∞ (H_V column = 0) | ∞ | 0 / 0 |
| z3[1] | MachineTrf gen5 → bus 35 | **0.0075**–0.0077 | **0.0090**–0.0092 | 4 / 4 |
| z2b[0…5] | banks, buses 54/53/74/73/94/93 | 0.0100–0.0140 | 0.0087–0.0122 | 9 / 13 |
| z3b[0…1] | banks, buses 114/113 | 0.0133–0.0154 | 0.0116–0.0135 | 2 / 4 |

**Per window.** The strongest STS coupler, Stage I → selected:
- W1 0.0156 → 0.0828;
- W2 0.0150 → 0.0797;
- W3 0.0157 → 0.0835;
- W4 0.0156 → 0.0830;
- W5 0.0156 → 0.0828;
- W6 0.0153 → 0.0815.

The weakest coupler is 0.0274–0.0278 → 0.1458–0.1478 in every window.

**Structure.**
- The first coupler of each STS (trafo_0/3/6/9) has the smallest column sum
  ($|\sum_j H_V| \approx 0.030$–$0.034$, against $0.050$–$0.055$ for the
  third). It never moved, even at the Stage-I point.
- The two TS_2 taps are network OLTCs with mixed-sign voltage columns and
  small column sums. They are effectively voltage-inert at either weight set.
- z2[0] and z3[0] have an identically zero voltage column in the TS model, at
  W1 and in all six cached operating points.

### 3.2 STS voltage deviation from V_ref = 1.03 pu

Columns: area-mean deviation (median / maximum) and maximum bus voltage.

| window | DSO | Stage I: mean dev p50 / max, v_max | selected: mean dev p50 / max, v_max | min > 1.08 (sel) |
|---|---|---|---|---|
| W1 | DSO_1 | −0.0024 / +0.0056, 1.0418 | −0.0033 / +0.0069, 1.0439 | 0 |
| W1 | DSO_2 | +0.0023 / +0.0063, 1.0462 | +0.0107 / +0.0178, 1.0778 | 0 |
| W1 | DSO_3 | +0.0037 / +0.0075, 1.0414 | +0.0037 / +0.0072, 1.0421 | 0 |
| W1 | DSO_4 | +0.0056 / +0.0140, 1.0693 | +0.0185 / +0.0289, **1.0823** | **176** |
| W2 | DSO_1 | +0.0024 / +0.0076, 1.0427 | +0.0035 / +0.0095, 1.0495 | 0 |
| W2 | DSO_2 | +0.0085 / +0.0115, 1.0548 | +0.0147 / +0.0187, 1.0686 | 0 |
| W2 | DSO_3 | +0.0063 / +0.0087, 1.0457 | +0.0039 / +0.0062, 1.0404 | 0 |
| W2 | DSO_4 | +0.0097 / +0.0124, 1.0564 | +0.0192 / +0.0241, 1.0750 | 0 |
| W3 | DSO_1 | −0.0045 / +0.0106, 1.0481 | +0.0011 / +0.0134, 1.0514 | 0 |
| W3 | DSO_2 | +0.0019 / +0.0071, 1.0432 | +0.0103 / +0.0141, 1.0579 | 0 |
| W3 | DSO_3 | +0.0024 / +0.0095, 1.0412 | +0.0028 / +0.0128, 1.0468 | 0 |
| W3 | DSO_4 | +0.0067 / +0.0146, 1.0587 | +0.0201 / +0.0270, 1.0745 | 0 |
| W4 | DSO_1 | +0.0038 / +0.0156, 1.0596 | +0.0119 / +0.0193, 1.0677 | 0 |
| W4 | DSO_2 | +0.0059 / +0.0142, 1.0526 | +0.0114 / +0.0234, 1.0710 | 0 |
| W4 | DSO_3 | +0.0077 / +0.0163, 1.0580 | +0.0105 / +0.0225, 1.0617 | 0 |
| W4 | DSO_4 | +0.0063 / +0.0145, 1.0611 | +0.0154 / +0.0233, 1.0756 | 0 |
| W6 | DSO_1 | −0.0008 / +0.0035, 1.0416 | −0.0068 / +0.0055, 1.0480 | 0 |
| W6 | DSO_2 | +0.0034 / +0.0067, 1.0453 | +0.0111 / +0.0151, 1.0697 | 0 |
| W6 | DSO_3 | +0.0000 / +0.0065, 1.0400 | −0.0036 / +0.0063, 1.0408 | 0 |
| W6 | DSO_4 | +0.0084 / +0.0147, 1.0631 | +0.0115 / +0.0148, 1.0686 | 0 |

At the selected weights:
- The largest mean deviation is 0.029 pu and the largest bus deviation is
  0.052 pu (DSO_4, W1).
- Both are below the smallest threshold of 0.080 pu, so the coupler
  thresholds are never reached in any area or window.

At the Stage-I point the mean deviations reach 0.0163 pu, the same order as
the thresholds of 0.015–0.019 pu.

The per-DSO e_v(DS) from `margin_report.txt`, selected (Stage I):
- DSO_1 0.0079 (0.0052);
- DSO_2 0.0129 (0.0061);
- DSO_3 0.0067 (0.0058);
- DSO_4 0.0192 (0.0104).

The author's live-plot observation is consistent with W1 at the selected
weights: the DSO_4 area mean has a median of 1.0485 pu, with v_max at
1.075–1.082 pu.

### 3.3 Validation of the rule against the recorded moves

| run | class | moves | event-coincident | other | mean dev / θ (other moves) | max\|e\| ≥ θ (necessary) |
|---|---|---|---|---|---|---|
| Stage I | STS | 19 | 9 | 10 | 0.63–0.96 (median 0.90) | 10 / 10 |
| Stage I | TS | 10 | 7 | 3 | 0.75, 0.82, 0.94 | 3 / 3 |
| selected | STS | 1 | 0 | 1 | 0.23 | **0 / 1** |
| selected | TS | 5 | 5 | 0 | — | — |

**At the Stage-I point** every assessable move satisfies the necessary
condition. Taps commit when the arithmetic mean deviation reaches 63–96 % of
θ, so the rule predicts engagement to within the proxy error of
Assumption 4.

**At the selected weights** the one STS move violates the necessary
condition. It is therefore driven by an output constraint, not by the
gradient.

**The DSO_4 episode in W1** (selected weights):
- v_max exceeds 1.08 pu in three intervals: 4.82–5.24 h, 9.49–10.87 h and
  10.88–12.0 h. Within them v_max holds at 1.0815–1.0820 pu.
- Over 9.5–12 h the STS DER of DSO_4 inject **+52 to +118 Mvar** (capability
  −152/+189 Mvar). They follow the interface-Q setpoints: coupler trafo_9,
  for example, carries −31.4 Mvar against a setpoint of −39.4 Mvar at 10 h.
- Injection raises the STS voltages, so the DER act against the band.
- The coupler that moved (trafo_10, +1 at 10.87 h) lowered the area mean by
  0.0022 pu but not v_max. The far-end bus has the smallest coupler
  sensitivity: 7.2·10⁻⁴ pu per step for trafo_10.

**All five TS moves** at the selected weights (z3[1] × 4, z1[1] × 1)
coincide with the generator trip/restore or the load ramp. At the Stage-I
point z1[1] also moved three times without an event, at 75–94 % of its θ.
The TS zone-mean deviation stays within ±0.012 pu at the selected weights
(e_v(TS) 0.0044), close to the 0.009–0.010 pu of the only two TS taps in
reach.

### 3.4 Stage-II candidates (242 evaluated, 94 feasible)

θ_STS,min is the strongest coupler over W1–W6, computed with
`engage_deviation` for every candidate.

| θ_STS,min [pu] | n | feasible | rejected for tap hunting | … of which STS taps hunt | median e_v(TS) / e_v(DS), feasible |
|---|---|---|---|---|---|
| < 0.010 | 37 | **0** | 36 | 36 | — |
| 0.010–0.015 | 27 | 8 | 19 | 14 | 0.0051 / 0.0065 |
| 0.015–0.020 | 22 | 10 | 9 | 1 | 0.0056 / 0.0070 |
| 0.020–0.030 | 17 | 11 | 5 | 0 | 0.0053 / 0.0086 |
| 0.030–0.050 | 37 | 22 | 11 | 0 | 0.0051 / 0.0091 |
| 0.050–0.080 | 37 | 17 | 19 | 0 | 0.0052 / 0.0105 |
| ≥ 0.080 | 63 | 25 | 26 | 0 | 0.0060 / 0.0105 |

**Hunting.**
- Tap hunting removed 126 of 148 rejected candidates.
- The hunting device is a TS tap in 95 of these 126, an STS coupler in 51,
  and both in 20.
- **STS hunting occurs only at θ_STS,min ≤ 0.0169 pu.** All 51 candidates
  with STS hunting lie at or below it; none lies above.
- The remaining tap-hunting rejections at higher θ_STS are TS taps, and they
  are independent of θ_STS.

**Effect of the filter.**
- The filter removed all low-threshold candidates below 0.010 pu.
- It removed 19 of 27 at 0.010–0.015 pu (14 with STS hunting) and 12 of 22
  at 0.015–0.020 pu (9 for tap hunting, 1 of them STS).
- It imposes **no upper bound**.

**Feasible candidates.**
- θ_STS,min ranges from 0.0106 to 1.22 (median 0.047).
- Spearman ρ(θ_STS, e_v(DS)) = +0.58 (p 9·10⁻¹⁰): a higher threshold goes
  with worse STS voltages.
- Spearman ρ(θ_STS, e_v(TS)) = +0.31: a higher threshold does **not** buy
  TS voltage on average.
- Spearman ρ(θ_STS, STS operations) = −0.62.
- Feasible median STS operations per 5 × 12 h: 20 at θ < 0.015, 9 at
  0.015–0.02, 1 at 0.02–0.03, 0 above.

**Search geometry.**
- var(log θ_STS) = 0.313. Of this, var(log dso_g_v) contributes 0.240 and
  var(log g_w_dso_oltc) 0.077 (covariance ≈ 0), so dso_g_v carries 76 %.
- The 72 candidates with dso_g_v < 1500 all have θ_STS ≥ 0.038 pu.
- The combination "low STS voltage priority and low tap threshold" was
  therefore never evaluated.

**Incumbent path** (centre of each Stage II-b box):

| key | stage | dso_g_v | g_w_dso_oltc | g_q | θ_STS,min | e_v(TS) | e_v(DS) | e_q | STS ops |
|---|---|---|---|---|---|---|---|---|---|
| 3cb66225e1a7 | Stage I | 10 000 | 16.64 | 0.040 | 0.0150 | 0.00547 | 0.00749 | 1 039 596 | 19 |
| faf088d2570e | II-a best | 4 240 | 16.22 | 0.356 | 0.0345 | 0.00507 | 0.00844 | 115 393 | 0 |
| e552b328fd76 | II-b r1 | 8 623 | 12.82 | 0.093 | 0.0134 | 0.00507 | 0.00607 | 296 351 | 18 |
| ff3f747d3f96 | II-b r2 | 4 621 | 7.68 | 0.115 | 0.0150 | 0.00471 | 0.00796 | 426 919 | 18 |
| c08b1d3d33ab | II-b r3 | 2 186 | 4.63 | 0.635 | 0.0191 | 0.00449 | 0.00991 | 138 244 | 8 |
| **fde2e81af6b1** | II-b r4 (result) | **660** | 5.83 | 0.174 | **0.0797** | 0.00440 | 0.01305 | 299 762 | 1 |

### 3.5 Contraction screen under a tap-price change

λ_max(B) of the selected set is 1.467, bound by TS_2 at W5; the STS layer
is at 1.130 (DSO_4, W6). Lowering only g_w_dso_oltc so that the strongest
coupler engages at θ gives:

| θ strongest (weakest) [pu] | g_w_dso_oltc | factor | STS λ_max(B) | STS integer floor | overall λ_max(B) |
|---|---|---|---|---|---|
| 0.0797 (0.148), selected | 5.834 | 1 | 1.130 | 0.071 | 1.467 |
| 0.015 (0.028) | 1.098 | 0.188 | 1.136 | 0.378 | 1.467 |
| 0.020 (0.037) | 1.464 | 0.251 | 1.134 | 0.283 | 1.467 |
| 0.030 (0.056) | 2.196 | 0.376 | 1.132 | 0.189 | 1.467 |

The screen does not constrain such an adjustment: 1.467 is below the
selection ceiling of 1.546 in every row.

---

## 4. Reasoning and options

### 4.1 Diagnosis

The tap price is anchored at the Stage-I point:

$$
g_w = 2 g_V |h| \Delta v_\mathrm{eng}.
$$

The engagement threshold is the ratio $g_w / g_V$, not the price itself.
Stage II searches the two factors independently, with the voltage weight
over a span three times wider than the price. The procedure therefore holds
neither the stated threshold nor any bound on it:
- The **filter** limits the threshold from below, through STS hunting at
  θ ≤ 0.017 pu.
- The **ranking** on e_v(TS) rewards a low STS voltage priority. That
  priority frees the STS DER for interface-Q tracking, which improves TS
  voltage. It also drags the coupler thresholds out of the band.

The engagement rule itself (0.015 pu STS, 0.0075 pu TS, 0.010 pu banks) is
not too high. At the Stage-I point the couplers engage where the rule says
they should (§3.3). What is too high is the threshold the selected set
**implies**: 0.080–0.148 pu, versus a band headroom of 0.05 pu.

For the TS the picture is different:
- The class-wide price is set by the strongest column, z3[1], at 0.0075 pu.
- Five of seven columns are therefore at 1.5× to ∞ of the stated threshold.
- The TS voltage itself is well regulated by the continuous actuators.

Low TS tap use is a property of the Stage-I design and the plant, not of the
selection.

### 4.2 Options

Common to all options: any change of `SELECTED_PARAMSET` invalidates the
selection result of Experiment B, Experiment C for the new set, and D1/D2.
It also invalidates the producers that use the set:
- `ch_9_1_ninner_isolated_sts.py`;
- `ch_9_1_ts_period_sweep.py`;
- `ch_9_4_deadband.py`;
- the Case-2 result runs, which wait for the set.

| option | what changes | status of the procedure | compute | invalidated | main risk |
|---|---|---|---|---|---|
| **(a) accept** | nothing; Part III text states that the selected weights exclude the STS couplers from voltage control (θ 0.080–0.148 pu > 0.05 pu headroom) and that they act only as backup to the soft band | as frozen | 0 h | nothing | Part III shows nearly inert STS OLTCs (1 operation in 60 h); DSO_4 exceeds the band for 176 min in W1. A reader can ask why the tap prices were designed from an engagement threshold that the search then voided. |
| **(b) search the threshold** | parametrise `g_w_dso_oltc := 2·dso_g_v·h_max·θ_STS` (likewise TS taps and banks if wanted) and search θ_STS in a **bounded** box, e.g. [0.012, 0.03] pu; code change in `tuning/stage2_search.py` and the driver | new frozen version, full re-application | ≈ 38 h on 30–38 workers + C (in the driver) + D1/D2 (≈ 2 × 35 min) + `ch_9_1_*` | B, C, D1/D2, `ch_9_1_*`, Case-2 waits | Without the upper bound, θ drifts upward again: on 2026-08-20 `engage_dso_pu` was a coordinate and walked 0.025 → 0.050. Different winner, unknown in advance. |
| **(c1) filter, offline re-selection** | add to the filter θ_STS,min ≤ V_max,band − V_ref = 0.05 pu (a coupler must be able to engage inside the band) and re-select over the 242 candidates with the unchanged ranking | amendment after the result; must be declared, with the as-frozen result reported alongside | minutes for the re-selection + one Experiment-C run for 301c285ee52b (≈ 6 h on 7 workers in the run of record; fewer hours on 30, by scaling) + D1/D2 + `ch_9_1_*` | the selection row of B (its simulations stay valid), C for one candidate, D1/D2, `ch_9_1_*` | Post-hoc filter (see 4.3). 301c285ee52b sits at θ 0.0147, just below the 0.017 pu hunting-free boundary seen elsewhere, but it showed no hunting in W1–W4, W6. |
| (c2) filter, full re-application | as (c1), applied before Stage II-b so the box centres follow the new rule | new frozen version | ≈ 38 h + as (b) | as (b) | Cost of (b) without its parametrisation fix. |
| (d) post-selection tap price | `g_w_dso_oltc` 5.834 → 1.10–1.46 (θ strongest 0.015–0.020); λ_max(B) unchanged at 1.467 | manual change of a procedure output | 5 × 12 h filter windows (≈ 0.5 h wall, 5 workers) + C + D1/D2 + `ch_9_1_*` | B's result, C, D1/D2, `ch_9_1_*` | Untested regime: no evaluated candidate combines dso_g_v < 1500 with θ < 0.038. With DER weakly tied to STS voltage, the taps become the main STS voltage actuator; hunting cannot be excluded. Hardest to defend. |

Re-selection over the feasible set, ranking unchanged (e_v(TS), then
e_v(DS) within 2 %):

| added rule | pool | winner | θ_STS | e_v(TS) | e_v(DS) | e_q | STS ops | settling |
|---|---|---|---|---|---|---|---|---|
| none (as run) | 94 | fde2e81af6b1 | 0.0797 | 0.00440 | 0.01305 | 299 762 | 1 | PASS (p95 25.0) |
| θ_STS ≤ 0.020 | 18 | **301c285ee52b** | 0.0147 | 0.00456 | 0.00861 | 247 923 | 22 | aborted at 47/120 (not measured) |
| θ_STS ≤ 0.030 | 29 | 301c285ee52b | | | | | | |
| θ_STS ≤ 0.050 | 51 | 301c285ee52b | | | | | | |
| e_v(DS) ≤ 1.5 × Stage I | 76 | 301c285ee52b | | | | | | |
| e_v(DS) ≤ 1.25 × Stage I | 49 | 301c285ee52b | | | | | | |
| next in rank in all θ rules | | c08b1d3d33ab | 0.0191 | 0.00449 | 0.00991 | 138 244 | 8 | PASS (p95 22.05) |

301c285ee52b against the selected set:
- e_v(TS) +3.8 %, e_v(DS) −34 %, e_q −17 %;
- λ_max(B) 1.345 (selected 1.467);
- TS tap operations 5 (5);
- per STS coupler at most 2 per 12 h, i.e. 4/day against the limit of 30.

c08b1d3d33ab against the selected set:
- e_v(TS) +2.2 %, just outside the 2 % band; e_v(DS) −24 %; e_q −54 %;
- λ_max(B) 1.414.

### 4.3 Recommendation

**(c1)**, with the bound θ_STS,min ≤ V_max,band − V_ref (0.05 pu). Reasons:
1. **The bound states a design requirement, not a preference.** Without it
   the integer actuator the Stage-I rule designed can be priced out of the
   operating band. The bound follows from two quantities fixed before the
   campaign: the band of 2026-10-25 and the reference of 1.03 pu.
2. **The winner does not depend on where the bound is placed.** Every
   tested rule (θ ≤ 0.02, 0.03, 0.05; e_v(DS) ≤ 1.25× or 1.5× Stage I)
   selects 301c285ee52b, so the amendment does not tune the answer.
3. **The cost is about one settling run plus the D1/D2 re-run**, against
   ≈ 38 h for (b) or (c2).
4. **The primary criterion barely moves.** The price is +3.8 % in e_v(TS),
   about the size of the 2 % indifference band. The gain is −34 % in e_v(DS)
   and STS couplers that take part in voltage control.

If the author holds to "applied once, whatever it is", the consistent choice
is **(a)**. In that case the thesis must report the implied thresholds
(§3.1), the band episode (§3.3) and the incumbent path (§3.4) as a finding
about the procedure. (b) is the methodologically cleanest repair if a
re-application is acceptable anyway, for example for other reasons.

---

## 5. Risks and unresolved points

**Hypotheses (not verified here):**
- H1. 301c285ee52b passes the settling requirement. Basis: ranks 2–3 pass at
  p95 22.05, and its λ_max(B) of 1.345 is below the selected 1.467. Its run
  was aborted at 47/120 steps and left no `steps.csv`.
- H2. The STS hunting boundary near θ ≈ 0.017 pu arises because the three
  couplers of an STS commit in the same iteration. $G_w$ is diagonal and
  there is no curvature coupling, so the joint mean step is
  ≈ 0.012–0.013 pu, together with the DER response. This is not checked
  against the hunting traces.
- H3. The ratio mean-deviation/θ of 0.63–0.96 at Stage-I moves reflects
  measurement noise and the per-bus weighting of the sensitivities. The
  records do not allow the H-weighted deviation to be reconstructed
  (Assumption 4).

**Open questions:**
- O1. **TS model.** Why are the voltage columns of z2[0] (gen1, bus 31 → 9)
  and z3[0] (gen2, bus 32 → 18) identically zero in the TS sensitivity?
  Candidate explanations, unverified: how these units' voltage control is
  represented, or how their buses are classified in the sensitivity model.
  Not needed for the STS decision, but it removes two of the seven TS taps
  from the TS controller.
- O2. **Soft band residual.** Why does DSO_4 hold v_max 0.0015–0.002 pu
  above the soft band for about 2.5 h despite a slack price of 10⁹?
  Candidate explanations: measurement noise on the constrained buses,
  linearisation error, or a recorded maximum outside the controller's
  observed set. This bears on option (a), where the band is the couplers'
  only trigger.
- O3. **Amendment form.** If (c1) is chosen, does the thesis present it as an
  amendment to the frozen procedure (with the as-frozen result reported), or
  as part of the procedure? The latter is not accurate for this run.
- O4. **TS taps and banks.** Should the threshold bound of (b)/(c) also cover
  them? Not indicated by this analysis: their thresholds moved by
  +20 % / −12.5 %.

**Decision requested from the author:** amend the frozen procedure with the
filter θ_STS,min ≤ 0.05 pu and re-select over the existing candidates (c1),
or keep the as-frozen result (a)?
