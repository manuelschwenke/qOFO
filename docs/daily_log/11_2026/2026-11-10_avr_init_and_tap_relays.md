# 2026-11-10 – Machine terminals start at 1.00 pu; set-point and 345 kV tap relays (handover)

**Status:** implemented, tested, committed on `res/avr-init-relay-2026-11-10` (worktree
`.claude/worktrees/avr-init-relay`, base cc09027 = tip of `res/ref-v5-2026-11-09`, Python code identical to bfae475
of the v5 runs) and cherry-picked onto `res/trial-v5-2026-11-09` (the branch checked out in `Z:\Python_Projekte\qOFO`).
Not pushed.  **No production run has been made with this code yet.**

**Why:** Ch 9 v5 Case-1 analysis (thesis repo `00_daily_log/2026-11-10_ch9_pilot_error_analysis.md`).  The pilot
error of O2 was lower than that of the classical variants because, after the events, the AVR set-points of G3 (area 2)
and G7 (area 3) sat at the IEC 60034-1 zone-A bound 1.05 pu under S1/S1m/O1.  They had only +0.02 pu of range, since
the machines started at 1.03 pu.  O2 escaped the bound through machine taps and banks.  Author decisions 2026-11-10:

1. Start the machine terminals at 1.00 pu (centre of [0.95, 1.05]); the machine-transformer taps still put the 345 kV
   side at 1.03 pu at the start.
2. A local set-point re-centring rule on the machine transformers of the variants whose machine taps no MIQP dispatches.
3. LOC (L2): a 345 kV-side relay on its machine transformers, 1.03 +/- 0.02 pu (AVR set-points pinned there).
4. The two network OLTCs at IEEE 12 (12-11, 12-13): O2, O3 and M dispatch them in the MIQP (zone tap lists, also
   collected by the central controller, `multi_tso_dso.py`, `c_oltc2w`), so every other variant gets a local relay
   holding bus 12 at 1.03 +/- 0.02 pu.
5. Keep the AVR set-point range [0.95, 1.05] (IEC 60034-1 zone A).

## Code changes

| File | Change |
|---|---|
| `configs/config.py` | new fields `gen_vm_init_pu` (None = old behaviour), `mt_setpoint_relay`, `mt_relay_v_low_pu` 0.96, `mt_relay_v_high_pu` 1.04, `mt_relay_confirm_steps` 3, `mt_relay_q_util_max` 0.8, `mt_hv_relay`, `mt_hv_relay_tol_pu` 0.02, `network_oltc_relay`, `network_oltc_relay_tol_pu` 0.02; `fix_machine_taps` docstring corrected (see finding below) |
| `experiments/run_multi_system_ofo.py` | `make_config`: `gen_vm_init_pu=1.00` (Part III base: Ch 8 frozen config, Case 1, Case 2, Case 3 builders that start from it) |
| `experiments/runners/multi_tso_dso.py` | Phase 1b `[10.1b]`: terminal ramp; LOC/SVR pin uses `gen_vm_init_pu`; construction of `SetpointTapRelay`s and of the 345 kV `TapRelay`s (`mt_hv_relay`, `network_oltc_relay`) with consistency checks; log lines |
| `controller/local_rules.py` | new `SetpointTapRelay`; `LocalDiscreteRules(..., setpoint_taps=...)` and a check that no trafo has two rules; local actions of the new rule are logged in `rec.local_tap_moves` with `"rule": "setpoint"`, `gen_idx`, `v_set_pu` |
| `experiments/ch_10_case1_svr/run_case1.py` | `_SVR` (S1m, S1, O0) and O1: `mt_setpoint_relay=True`, `network_oltc_relay=True`; L2: `mt_hv_relay=True`, `network_oltc_relay=True`; guards: `_SVR_FIELDS` += `mt_setpoint_relay`, `mt_hv_relay`, `network_oltc_relay`; (O1, O2) += `mt_setpoint_relay`, `network_oltc_relay` |
| `tests/test_setpoint_tap_relay.py` | 7 tests of the new rule |

Case 2 (`experiments/ch_11_case2_multisystem/variants.py`) is not edited: its L2, S1 and S2 inherit from `run_case1`,
and its guards reuse `run_case1._SVR_FIELDS`.

### Resulting device rules per variant

| Variant | Machine taps | Bus-12 network taps | AVR set-points |
|---|---|---|---|
| L2 (LOC) | 345 kV relay 1.03 +/- 0.02 | local relay 1.03 +/- 0.02 | pinned at 1.00 |
| S1m, S1, O0 (SVR^-, SVR, OFO form) | set-point rule 0.96 / 1.04 | local relay 1.03 +/- 0.02 | RPR, start 1.00 |
| O1 (OFO-c) | set-point rule 0.96 / 1.04 | local relay 1.03 +/- 0.02 | TS-OFO, start 1.00 |
| O2, O3, M (OFO, OFO+, CEN) | MIQP | MIQP | MIQP, start 1.00 |
| Case 2 S1, S2 (SVR, SVR+) | set-point rule | local relay | RPR, start 1.00 |

All local tap rules: one step per action, holding time `oltc_cooldown_s_mt` = 900 s (= TS-OFO lock).  Banks unchanged
(local rule below O2, MIQP in O2/O3/M).

### Method details
- **Terminal ramp (Phase 1b):** the old Phase 1 runs first (terminals at 1.03 pu, DiscreteTapControl on the machine
  transformers holds the 345 kV side at 1.03 +/- 0.01).  Then the terminals go to 1.00 pu in 0.005 pu steps, each
  solved with `init="results"` and the tap controllers active.  Before each step the machine taps move at least to
  the ratio 1.03 / V_terminal.
  - Why a ramp: a direct start at 1.00 (taps 0 or preset +2) did not converge from pandapower's DC start.
    Any single machine back at 1.03 converged, and a warm start converged in 7 iterations.
  - Why the ratio preset: in Phase 1, W3's temporary PV unit holds G4's 345 kV bus (IEEE 19) at exactly 1.03, so G4's
    tap controller never leaves its band and G4 would absorb 211 Mvar against W3.
  - Result (scenario A, 05 Jan 2016 08:00):
    - taps: G10 +2, G3 +2, G4 +2, G7 +3, G1eq +2;
    - 345 kV sides: 1.0255 / 1.0229 / 1.0300 / 1.0211 / 1.0238 pu;
    - machine Q: G10 65, G3 29, G7 109, G1eq 94 Mvar (old start 67 / 31 / 112 / 94), G4 −35 Mvar (old −2).
- **Set-point rule:** reads the AVR set-point `net.gen.vm_pu` (the controller output), not a voltage.
  - If the set-point stays ≥ 1.04 for 3 steps, one tap step raises the grid side against the terminal, so the RPR or
    TS-OFO can lower the set-point again. If it stays ≤ 0.96, the opposite step.
  - Blocked while Q ≥ 0.8 Q_max (raising step) or Q ≤ 0.8 Q_min (lowering step), on the controllers' capability
    curve (`actuator_bounds.compute_gen_q_bounds`).
  - It has no voltage target, so unlike the old HV relay on machine transformers (divergence 2026-09-01, wind-down
    2026-11-05) it does not fight the AVR loop.
  - Applies to dispatchable machines only: not the network equivalent G1eq and not the slack terminal.

### Finding: `fix_machine_taps` also held the bus-12 network OLTCs
Its docstring said "couplers, network OLTCs and the banks keep their local rules".  The runner's relay loop runs over
`meta.machine_trafo_indices`, which includes the two network OLTCs (gen map -1), and skips all of them under
`fix_machine_taps`.  So in every v5 run of L2, S1m, S1, O0, O1 and Case 2 S1/S2 the bus-12 taps were held
(console: "0 TS tap relay(s)"), while O2/O3/M moved them.  In the Ch 9 v5 run these taps (11 operations under O2) lift
bus 12 from 0.95 to 1.00–1.02 pu after the 250 Mvar connection.  Without bus 12 the area-2 error would be S1 13.8,
O1 12.5, O2 9.0 instead of 22.3, 21.5, 10.5 (1e-3 pu).  So most of O2's area-2 advantage came from this asymmetry.

## Checks done (scratchpad, not runs of record)
- `pytest tests/test_setpoint_tap_relay.py tests/test_local_rules.py`: 10 passed.
- `run_case1.assert_case1_controlled()` and `variants.assert_case2_controlled()` pass.
- Smoke runs (scenario A, 6–12 min, `unit_commitment_window_s=21600` so the commitment equals the 360-min run): S1, O2,
  L2 run.
  - L2: 7 tap relays (machine transformers 0, 2, 6, 11, 10 and network 3, 4, all ±0.02).
  - S1: 2 tap relays (3, 4) and 5 set-point rules (gens 0, 1, 2, 5, 7; gen 7 = G9 is off line and stays idle).
- TS-OFO λ_max at step 1 (O2): 1.334 / 1.446 / 1.224 against v5 1.409 / 1.379 / 1.263 (zone 2 +4.9 %).
- Host reproducibility (ManuL16 vs e5power, old code): S1 and O2 over 360 min match the v5 run to 1.8e-9 pu.
  Pitfall: a SHORT run changes the unit commitment, because its window is the run horizon (G3 is kept off in a
  12-min run). Use `unit_commitment_window_s=21600` for short checks.

## Tuning
The weights (`ch9_selected_nr_2026_11_v5`) are not changed by any of this, and none of the new rules acts inside an
OFO.  But the new start changes the operating point at which they were validated.  The machine-transformer ratio scales
the AVR columns of H, and the reactive capability depends on the terminal voltage.  So the loop gain moves a little
(see λ_max at step 1 above).  **To do:** re-evaluate λ_max(B) of the selected weights on the Ch 8 design points with
the new start, without re-selecting (design value 1.5301, paramset provenance).  If it stays clearly below 2, keep the
weights and report the new value in Ch 8.

## What has to be rerun with this code
Everything closed-loop that builds on `make_config`:
- Ch 9 / Case 1: `experiments/ch_10_case1_svr/run_case1.py --only K1-REF --scenario A`.
  Also K1-PAR, if `fig:case_svr:pareto` stays.
- Ch 10 / Case 2: REF scenarios, ladder, and the MC ensemble (N = 100) for L2, S1, S2, O2, O3, M.
- Ch 11 / Case 3, if its runs start from the same base.
- Ch 8: the weight-procedure and settling runs were made at the old start.  Rerun them, or at least add the
  λ_max(B) re-evaluation above.

Run from the branch that holds this commit.  Cite runs by their full folder name (thesis provenance scheme).

## Related, still open
- **0.92/1.08 sensitivity** (old code with the bounds widened by a wrapper;
  `results/ch_10_case1_svr/0005_2026-11-10_102828_run_case1_v5_A_avr092_108`):
  - S1m and O2 completed. S1 and O1 diverge at the 250 Mvar connection (121:40).
  - Pre-event Q/Q_max of G3: S1 0.85, O1 0.80, S1m 0.82 (survives), O2 0.69, so reserve alone does not explain it.
  - Not diagnosed; probably superseded by the new start (same +0.05 pu of upward set-point range).
- **Plant Q limit:** the network limits machine Q by `net.gen.max_q_mvar` (G3 400 Mvar), while the controllers use
  the P-dependent capability curve (about 315 Mvar). In the 0.92/1.08 run, O2's G3 reached 341 Mvar.
- **Wrapper and analysis scripts** (thesis repo, not qOFO):
  - `graphics/Ch9/data/2026-11-10_case1_ref_v5/avr_bounds_092_108/` (wrapper, comparison script);
  - `graphics/Ch9/data/2026-11-10_case1_ref_v5/pilot_analysis/` (pilot-error analysis of the v5 run).
