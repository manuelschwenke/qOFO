# Harmonised device limits, local rules, scenario D, Ch 9 re-validation producer

Timestamp: 2026-10-30, ~14:30–17:00 Europe/Berlin.
Branch `weekend/harmonised-limits-2026-10-30`, from `case2/ref` at 400a037. Worktree
`F:\qOFO_weekend`.
Session: side session of the Case-2 main session, at the author's direct instruction.

## Why

Author decisions of 30 Oct, after the STS access-gain analysis
(`docs/analysis/2026-10-30_sts_access_gain.md` in the `case2-ref` worktree):
- **AVR range:** one range for every variant, [0.95, 1.05] pu.
- **OLTC holding time:** one minimum time between two taps for every OLTC: 120 s for STS couplers,
  15 min for TS taps.
- **TS taps of the classical variants:** a local relay at 1.03 ± 0.01 pu, like the coupler
  relays, instead of being held.
- **Banks of the classical variants:** a local rule — MSC in below 1.01 pu, MSR in above 1.05 pu
  on the primary side of the hosting coupler.
- **Weekend runs:**
  - a re-validation of the selected weights;
  - REF A plus the declared stress scenarios B and D;
  - STEP, MC and Case 1.

## What changed (key structures)

| file | change |
|---|---|
| `configs/config.py` | **new fields:** `gen_vm_min_pu` / `gen_vm_max_pu` = 0.95 / 1.05; `tso_int_cooldown` = 4, i.e. (1+4)·180 s = 15 min, with STS `int_cooldown` 5 → 120 s; `shunt_local_v_ref_pu` = 1.03 and `shunt_local_dv_pu` = 0.02; `shunt_dispatch='local'`. **Changed values:** `oltc_cooldown_s_mt` 180 → 900 s; `oltc_cooldown_s_nc` 60 → 120 s |
| `controller/tso_controller.py` | class default `gen_vm_max_pu` 1.07 → 1.05 |
| `controller/local_rules.py` (new) | `TapRelay`: explicit one step per action, HV side, holding time. `BankRelay`: thresholds; holding time and MSC/MSR exclusion via `ShuntBank`. `LocalDiscreteRules`: releases before engagements |
| `experiments/runners/multi_tso_dso.py` | **Range and lock:** the AVR range goes to the TSO zones, the central controller and SVR (`SVRZoneConfig.v_set_*`); the TS lock goes to the TSO `OFOParameters`. **Relays:** built wherever TS taps or banks are not MIQP-dispatched (the slack machine's transformer stays held). **Per step:** the relays are evaluated before the end-of-step PF and force it when they act; moves are recorded in `rec.local_tap_moves` / `rec.shunt_commits` (`zone='local'`) |
| `experiments/helpers/records.py` | `local_tap_moves` |
| `experiments/ch_10_case1_svr/run_case1.py` | `_NO_SHUNTS` → `_LOCAL_SHUNTS` (banks installed, local rule); `--jobs` (variants in parallel); `K1-ALL` |
| `experiments/ch_11_case2_multisystem/variants.py` | `ts_equivalent_cooldown` uses the TS lock: CEN 53 → 44 central steps = 15 min |
| `experiments/ch_11_case2_multisystem/ref/scenarios.py` | scenario **D** (B with the load at IEEE 4) |
| `experiments/ch_09_parameter_selection/ch_9_5_avr_revalidation.py` (new) | **Re-simulation:** the declared re-validation re-simulates 13 candidates plus the Stage-I point on W1–W4 and W6. **Selection:** the unchanged `ch_9_3_select.py`; settling verdicts from runs 0001 and 0008 are reused. **Output:** `DECISION.json` |
| `docs/analysis/2026-10-30_scenarios_B_D_declaration.md` (new) | declaration of B and D before any run |
| `tests/test_local_rules.py` (new) | relay band and holding time; bank thresholds, exclusion, release order |
| `_weekend/chain.py` | per-host chains (e5wimi: re-validation → STEP + Case 1; e5power: REF A/B/D → MC after DECISION_OK) |

**Why the TS relay is not a `DiscreteTapControl`.** The runner records a divergence after the
gen-2 trip under SVR, measured 2026-09-01. Its cause: the pandapower controller iterates many taps
inside one `runpp(run_control=True)`, while the AVR holds the LV side of the same transformer. The
explicit rule applies the same law — band 1.03 ± 0.01, one step per action, holding time — once
per step, outside the power flow.

**Amendment ~15:05 (author):** the bank rule acts only after its condition has held for 3
consecutive 20-s steps (`shunt_local_confirm_steps` = 3). It is applied to the release as well;
the counter restarts after a commit. The e5power REF launched at 15:02 without it was stopped and
restarted.

## Checks

- **Unit tests:** `test_local_rules`, `test_shunt_banks`, `test_oltc_cooldown`,
  `test_svr_controller`, `test_tso_tertiary_shunt` (71 passed); `test_tso_saturation`,
  `test_svr_sts_access`, `test_central_updater_wrapper`, `test_shunt_banks_integration` (20
  passed).
- **Tap direction:** a +1 tap raises the HV-side voltage on machine transformers 0 and 2 and on
  network OLTCs 3 and 4 (IEEE 12 is their HV bus).
- **Smoke runs:** see the addendum below.
