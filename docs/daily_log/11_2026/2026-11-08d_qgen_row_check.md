# 2026-11-08d — Q_gen rows of the TS sensitivity matrix H versus finite differences

Timestamp: 2026-11-08 22:47. Branch **`res/qgen-row-check-2026-11-08`** (from
`res/tuning-v5-2026-11-08` @ 86f9b20), worktree `F:\qOFO_qgen`. No controller or sensitivity
code changed. Script `2026-11-08d_qgen_row_check.py`, output `2026-11-08d_qgen_row_check.out`
(132 power flows on the reduced zone nets, 66 on the plant; about 80 s single-threaded).

## Question
`2026-11-08c_procedure_v5_launch.md` recorded dQ_G3/dV_set,G3 = 22 "Mvar/pu" in H (zone 2, W1)
as implausibly small for a transmission machine. Are the Q_gen rows of H right, and if not, why?

## Method
- Operating points: **W1** (2016-01-05 08:00, 12 h) and **MC20261108** (2016-01-12 09:00, 3 h,
  G3 committed), built up to the runner's pre-loop hook as in `2026-11-08c_nr_capture.py`
  (ch_9_6 `base_config`, network `rural_700_north_res`, `local_sensitivities_tso=True`,
  `tie_boundary_equivalent="thevenin"`, weights v4; the weights do not enter H). The recomputed H
  equals the captured H bit for bit.
- H as the controller takes it: `ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix())`,
  rows `[V | Q_PCC | I | Q_gen]`, columns `[der | pcc | gen | oltc | shunt]`.
- Central finite differences (FD) of a pandapower power flow (run_control off, single slack,
  `enforce_q_lims=False`, tolerance 1e-9 MVA) on three nets:
  1. **model-PQ**: the controller's reduced zone net (`ctrl.sensitivities.net`) with every load
     frozen to constant P/Q at its solved value. This is the like-for-like reference for the
     analytical formulas, because pandapower's Newton-Raphson Jacobian (the J that
     `JacobianSensitivities` inverts) does not contain dS_load/dV (`pypower/newtonpf.py` only
     re-evaluates `Sbus` between iterations);
  2. **model (ZIP)**: the same reduced net with its voltage-dependent loads
     (`network/ieee39/load_model.py`: P constant current, Q constant impedance);
  3. **plant**: the full 118-bus plant net, prepared as in `JacobianSensitivities.__init__`
     (non-slack P frozen at `res_gen`; max |dV| to the runner's solution 6e-10 pu).
- Steps: V_set of every in-service machine ±0.005 pu (all other PV setpoints held); one Q_PCC
  set point as a Q load at the coupler HV bus, ±10 Mvar (load convention); the TS IBG
  (`sgen.q_mvar`, generator convention) ±10 Mvar; one in-service machine/interconnector OLTC ±1 step.
  Recorded: `res_gen.q_mvar` of every machine (generator convention), the slack Q,
  `res_trafo3w.q_hv_mvar` (load convention), |V| at the monitored buses.
- Additionally: the DSO controllers' interface-Q × OLTC entries (same builder family) against FD
  on the DSO's reduced net and the plant; and the stage-1 λ screen recomputed from the 35 captures
  with those entries corrected (no power flow).

## Results

### Q_gen rows, W1 [Mvar per unit input: per pu V_set, per Mvar, per tap step]

| zone | column | row | H as built | H × sn_mva | FD model-PQ | FD model (ZIP) | FD plant |
|---|---|---|---|---|---|---|---|
| 1 | V_set G10 (gen 0) | Q_G10 | 23.446 | 2344.6 | 2344.6 | 2336.9 | 2114.0 |
| 1 | V_set G10 | Q_G1 (gen 9, slack) | −9.488 | −948.8 | −948.8 | −1031.7 | −1538.3 |
| 1 | IBG sgen 1 | Q_G10 | −0.364 | (dimensionless) | −0.364 | −0.374 | −0.477 |
| 1 | tap trafo 0 (G10 machine trafo) | Q_G10 | 0.327 | 32.7 | 32.7 | 32.6 | 29.9 |
| 1 | tap trafo 0 | Q_G1 | −0.119 | −11.9 | −11.9 | −13.0 | −19.4 |
| 2 | **V_set G3 (gen 1)** | **Q_G3** | **22.030** | **2203.0** | **2203.0** | **2302.3** | **2035.9** |
| 2 | Q_PCC,set trafo3w 7 | Q_G3 | 0.572 | (dimensionless) | 0.572 | 0.553 | 0.604 |
| 2 | tap trafo 3 (bus 11–10) | Q_G3 | 0.00087 | 0.087 | 0.086 | 0.269 | 0.432 |
| 3 | V_set G4 (gen 2) | Q_G4 | 20.223 | 2022.3 | 2022.3 | 2061.7 | 1873.6 |
| 3 | V_set G4 | Q_G7 | −5.743 | −574.3 | −574.3 | −542.8 | −678.1 |
| 3 | V_set G7 (gen 5) | Q_G4 | −5.801 | −580.1 | −580.1 | −550.6 | −672.6 |
| 3 | V_set G7 | Q_G7 | 11.922 | 1192.2 | 1192.2 | 1222.3 | 1162.9 |
| 3 | Q_PCC,set trafo3w 9 | Q_G4 / Q_G7 | 0.316 / 0.227 | (dimensionless) | 0.316 / 0.227 | 0.311 / 0.222 | 0.371 / 0.262 |
| 3 | IBG sgen 0 | Q_G4 / Q_G7 | −0.717 / −0.079 | (dimensionless) | −0.717 / −0.079 | −0.711 / −0.075 | −0.748 / −0.094 |
| 3 | tap trafo 11 (G7 machine trafo) | Q_G4 / Q_G7 | −0.071 / 0.160 | −7.08 / 16.04 | −7.08 / 16.04 | −6.72 / 16.40 | −8.18 / 15.63 |

MC20261108 (self terms): G10 24.226 → FD 2422.6 / 2390.6 / 2115.3; G3 22.016 → 2201.6 / 2305.6 /
1958.5; G4 20.238 → 2023.8 / 2066.5 / 1831.1; G7 12.208 → 1220.8 / 1252.1 / 1177.9
(model-PQ / model ZIP / plant). Taps and cross terms as at W1.

Ratio FD/H over all Q_gen entries of both points (`.out`, SUMMARY):

| column kind | FD model-PQ / H | FD model (ZIP) / H | FD plant / H |
|---|---|---|---|
| V_set (14) | **100.000** (100.000–100.000) | 102 (94–111) | 97 (87–179) |
| tap (10) | **99.99** (98.7–100.0) | 102 (95–311) | 118 (89–498) |
| Q_PCC,set (6) | 1.000 | 0.975 | 1.16 |
| IBG (8) | 1.000 | 1.01 | 1.28 |

### Units and conventions (established)
- `net.sn_mva = 100` on the plant and on every reduced net.
- V_set column: per pu of the setpoint of the machine's own 10.5-kV terminal (PV bus behind the
  machine transformer); dV_terminal/dV_set = 1.0000 in both nets. Q_gen in generator
  convention on both sides; signs agree everywhere. The Q_PCC,set column is load convention and
  agrees in sign with FD.
- The controller compares the Q_gen rows with `measurement.gen_q_mvar` and with capability bounds
  in Mvar (`tso_controller.py:1345-1371`), so the rows must be in Mvar per unit input.

### Physical plausibility of the true value
G3's machine transformer: 900 MVA, vk 18 %, x = 0.020 pu on 100 MVA; an infinite bus behind it
would give V/x ≈ 5150 Mvar/pu. The FD value 2036–2302 Mvar/pu implies a grid-side reactance of
≈ 0.025–0.029 pu (≈ 3.4–3.9 GVA short-circuit power at bus 31's HV node). 22 Mvar/pu is
therefore not physical; it is the per-unit number.

## Root cause (established)
`JacobianSensitivities` works on pandapower's per-unit Jacobian (system base `sn_mva`).
`compute_dQgen_dVgen_matrix` and `compute_dQgen_ds_2w_matrix` return ∂Q[pu]/∂V[pu] and
∂Q[pu]/∂step, but are documented, and consumed, as Mvar/pu and Mvar/step. The `sn_mva` factor is
missing:
- `sensitivity/jacobian.py:3003-3009` (`compute_dQgen_dVgen_matrix`: indirect term
  `dQgen_dx_cache @ dx_dVl` and direct term `dQl_dVl` from `_compute_dg_dVk`, both pu; return
  without `* sn_mva`);
- `sensitivity/jacobian.py:3045-3054` (`compute_dQgen_ds_2w_matrix`: indirect and direct term pu);
- `sensitivity/jacobian.py:3091-3093` (`compute_dQgen_ds_3w_matrix`, same; not used by the TSO here).
`compute_dQgen_dQder_matrix` (Mvar/Mvar, base cancels) and `compute_dQgen_dQ_shunt_matrix`
(dimensionless × `q_step_mvar`) are correct. The controller assembly (`tso_controller.py:2234-2311`)
copies the blocks without rescaling; row/column indices, the gen→ppc bus mapping, the direct term
∂Q_k/∂V_k (`jacobian.py:2753`, `:2807`) and the PV/slack treatment are correct: with constant-PQ
loads, H × 100 reproduces FD to five digits, including G3 as the promoted slack of the zone-2
model net.

Why it survived: the tests encode the slip. `tests/test_jacobian_qgen.py:150` and
`tests/test_jacobian_tap.py:202` multiply the analytical value by `net.sn_mva` before comparing
with FD. The development net (`network/build_tuda_net.py:575`, `pp.create_empty_network()`) has
`sn_mva = 1`, where pu ≡ Mvar. The 2026-06-22 fix of the same class for ∂V/∂Q
(`2026-06-22_shunt_integrator.md`) declared the Q_gen rows "dimensionless ratios … correct",
which holds for their Q columns only.

## Secondary findings (established by FD unless marked)
1. **Same slip in the interface-Q rows.** `compute_dQtrafo3w_hv_ds` (`jacobian.py:2403-2466`)
   returns pu per step: the DSO controllers' Q_iface × OLTC entries are 100× too small (24
   entries, FD DSO-model/H = 95–101, FD plant/H = 87–101; e.g. DSO_3 coupler 6: H −0.090, FD
   −8.9 Mvar/step on its own interface, +5.3 and +3.8 on the two parallel couplers, i.e. the
   tap mainly shifts Q between parallel couplers). `tests/test_jacobian.py:1444` passes only
   because the fixture's tap step (1.222 %) is close to 1/sn_mva.
2. **Q_PCC × V_set** (`compute_dQtrafo3w_hv_dVgen`, `jacobian.py:2541-2663`): unit slip plus
   omitted angle terms; the two kept branch-voltage terms nearly cancel, and the result has the
   wrong sign (W1, G3 → trafo3w 7: H × 100 = +1.75, full chain ∂Q_HV/∂x·dx/dV_k × 100 = −2.48,
   FD model-PQ −2.43 Mvar/pu). Magnitude in the model ≤ 6 Mvar/pu. In the plant the open-loop value
   is −49…+20 Mvar/pu, dominated by the voltage dependence of the HV networks that the reduced
   model replaces by constant PQ loads (structural, by design under DSO tracking).
3. **Load model.** The analytical H is the sensitivity of a constant-power-load network; the
   network's ZIP loads change Q_gen self terms by −1…+5 % and V rows by up to 8 % of max |H|
   (FD model vs model-PQ). Small cross terms can be off by a factor 3 (G3 vs tap trafo 3:
   0.087 vs 0.269 Mvar/step).
4. **Reduction mismatch** (plant vs model, fact; cause = Thevenin tie boundaries and the
   HV-network equivalent, hypothesis): self terms −3…−13 %; cross terms to the external
   equivalent G1 up to +79 %.

## Consumers of the Q_gen rows and consequence
Config at both points: `g_z_q_gen = 100`, `tso_g_res_sg = 0`, `zone_tso_g_res_sg = None`,
`enable_avr_saturation_mode = False`, `g_q_tso = 0`.
- **Active:** the Q_gen soft output constraint (`tso_controller.py:1345-1371`, slack weight
  `g_z_q_gen`, MIQP `miqp_solver.py:597-606`). It acts only when the predicted Q_gen leaves the
  capability band. At W1 and MC20261108 all dispatchable machines are well inside (Q = 36–121
  Mvar for G3, G4, G7, G10; lower capability bounds −207…−331 Mvar, upper 354…687 Mvar). For G3
  the plant's table limit (+400 Mvar,
  `enforce_q_lims_plant=True`) binds before the capability bound (+668 Mvar at 440 MW). On the
  under-excited side the capability bound (−273) is the tighter one.
- **Inactive in this config:** SG reserve term (`tso_controller.py:1759-1802`, `g_res_sg = 0`), the
  Q_gen block of the output gradient, `objective_terms` (diagnostic). Not read by the stage-1 λ
  screen (`tuning/stage1_screen.py:478-481` reads V and Q_PCC rows only), the coordinator
  (weight 0 on Q_gen rows), or the SVR controller (own perturbation estimate).
- **Qualitative effect of a correct row:** (i) the MIQP stops a V_set move that would take a
  machine across its capability bound; as built it predicts 1 % of the true ΔQ_gen (G3, AVR step
  0.002 pu: 0.044 instead of 4.4 Mvar); (ii) once a bound is violated, the relief shifts from
  Q_PCC set points and TS IBG towards the machine's own AVR. Effective price per Mvar of
  G3-Q relief, g_w/h²: AVR 3500 (v4) / 10 300 (v5 start) as built, 0.35 / 1.03 corrected;
  Q_PCC,set via trafo3w 7: 2.7 (v4) / 0.54 (v5 start).
- **Stylised quantitative check** (`.out`, "[z0 = 10 Mvar over Q_max …]"): minimiser of
  w'G_w w + g_z(z0 + h'w)² over the continuous columns, one violated row, α = 1, no voltage
  gradient. G3 at W1: v4 as built 9.95 Mvar of the relief from Q_PCC and 0.13 Mvar from the AVR
  (ΔV_set −6e-5 pu); corrected 4.4 Mvar Q_PCC, 5.6 Mvar AVR (ΔV_set −0.0025 pu). v5 start: as built
  10.0 / 0.0; corrected 9.2 / 0.8. G7: v4 as built IBG 5.1, Q_PCC 4.7, AVR 0.1; corrected
  2.3 / 2.2 / 5.5. Total relief is about 10 Mvar in every case.
- **Interface-Q slip (finding 1) affects the running campaign.** With the DSO Q_iface × OLTC
  entries scaled by `sn_mva`, the STS λ of the stage-1 screen (banks excluded, dso_oltc included,
  35 captures) changes from medians 0.97–1.33 and maxima up to 1.59 to medians 2.0–12.5 and
  maxima up to 16.4 (v4: DSO_3 median 0.99 → 12.5; v5 start: DSO_4 max 1.55 → 2.43). 69–97 % of
  the corrected OLTC-column curvature comes from g_q S_Q. TS λ is
  unaffected (TS S_Q enters with `tso_g_q_pcc = 0`). How λ applies to an integer column (commit
  rule) is an open question.

## Proposed fix (NOT applied)
A. Q_gen rows, low risk, no tuning dependency:
   - `compute_dQgen_dVgen_matrix`, `compute_dQgen_ds_2w_matrix`, `compute_dQgen_ds_3w_matrix`:
     return `matrix * float(self.net.sn_mva)`. The PQ-mode wrapper inherits the change.
   - Tests: drop the compensating `* sn_mva` (`test_jacobian_qgen.py:150`,
     `test_jacobian_tap.py:202`), tighten the 0.3–3.0 band to ±2 %, and add a test on a net with
     `sn_mva ≠ 1`.
   - Expected effect: Q_gen rows ×100 in the V_set and tap columns. Against FD on the model with
     constant-power loads: within 1e-4 relative for V_set and machine-trafo taps, 1.3 % on the
     small interconnector-tap cross term. Against the ZIP model: −1…+5 % on self terms. Against
     the plant: −13…+79 %. Behaviour
     changes only when a capability bound is predicted to be violated (see above).
B. Interface-Q rows (needs an author decision, because it moves the STS design): `compute_dQtrafo3w_hv_ds`
   × `sn_mva` (FD-confirmed) and `compute_dQtrafo_2w_ds` × `sn_mva` (`jacobian.py:1164`, same
   pattern by inspection, not exercised on this network); replace `compute_dQtrafo3w_hv_dVgen` by the full
   chain `_compute_dQhv_dx_3w(d) @ (-J_inv @ dg_dVk)` × `sn_mva`; fix `test_jacobian.py:1444`.
   Consequence: the DSO MIQP sees 1–9 Mvar of interface-Q change per tap step instead of
   0.01–0.09 Mvar, and the STS λ of the weight-selection procedure changes as above.
C. Optional: add dS_load/dV of the ZIP loads to J (finding 3). This is a model-fidelity question,
   not a bug.

## Open points
- Whether a run with fix A changes any campaign metric. Expected: only in hours with a machine at
  its capability bound (G3-on scarcity hours; under-excited side). Not simulated.
- Whether fix B changes the DSO OLTC hunting seen in v4 (DSO taps mainly circulate Q between
  parallel couplers, which the as-built model does not see). Hypothesis, not tested.
- How the λ criterion should treat dso_oltc columns once their Q rows are correct.
- `enable_avr_saturation_mode=False`: when the plant switches a machine to PQ at its table
  limit, H still has the PV column. Outside the scope of this check.
