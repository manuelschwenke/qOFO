# 2026-11-10f -- Common rerun base: AVR-init / tap-relay patch + coupler relay band +-0.02

**Status:** implemented and tested on branch `res/rerun-base-2026-11-10` (worktree `F:\qOFO_rerun`, e5wimi), cut from
`res/avr-init-relay-2026-11-10` = 22d8922 = cc09027 (tip of `res/ref-v5-2026-11-09`, the code line of the v5 runs of
record) + the AVR-init / tap-relay patch of `2026-11-10_avr_init_and_tap_relays.md`. Not pushed. **No production run
has been made with this code yet.**

**Why:** author decisions 2026-11-10:
- "then we go for the +- 0.02 band": local coupler relays at 1.03 +- 0.02 pu, after the relay-band screen
  (`2026-11-10e_relay_band_screen_result.md`). With +- 0.01 the relay band equals the IBG Q(V) dead band and the
  relays withdraw the droop support.
- Respect the AVR-init / tap-relay changes of the other session and use one common base for the REF, Ch 9 and MC
  reruns. Author confirmed the base 22d8922 ("Yes, use that base").
- Not the trial branch: `res/trial-v5-2026-11-09` (main checkout) carries the same patch as a14dd13 but lacks the v5
  REF line. Missing there: `ch9_selected_nr_2026_11_v5` and SELECTED_PARAMSET, the H-fix merge 13951ba, and
  P2R_B03B05 (bfae475). See 2026-11-10e, section "Common base".

## Code changes (on top of 22d8922)

| File | Change |
|---|---|
| `configs/config.py` | new field `coupler_relay_tol_pu: Optional[float] = None` next to `dso_oltc_init_tol_pu`; `None` keeps the old behaviour (run-time band = initialisation band) |
| `experiments/runners/_multi_tso_helpers.py` | new `set_coupler_relay_band(net, coupler_indices, v_target_pu, tol_pu)`: sets `vm_lower_pu` / `vm_upper_pu` of the coupler `DiscreteTapControl`s (trafo3w) only |
| `experiments/runners/multi_tso_dso.py` | after the Phase-2 keep/drop of the coupler controllers: in local DSO mode (not central) and with `coupler_relay_tol_pu` set, the kept relays get `oltc_init_v_target_pu` +- `coupler_relay_tol_pu`; raises if the number of relays differs from the number of couplers; log line `[local DSO] coupler relay band 1.030 +-0.020 p.u. on 12 couplers (start taps from +-0.010)` |
| `experiments/run_multi_system_ofo.py` | `make_config`: `coupler_relay_tol_pu=0.02` (Part III base, so Case 1 and Case 2 both get it) |
| `tests/test_coupler_relay_band.py` | 4 tests: only the listed couplers change; machine-transformer relay untouched; a relay at +-0.01 steps where +-0.02 does not; config default None |

Kept unchanged on purpose:
- Initialisation band `dso_oltc_init_tol_pu` = 0.01, so all six variants start from the same coupler taps; only the
  run-time band of the kept relays changes. Same procedure as the screen.
- Rate limit of the coupler relays (`oltc_cooldown_s_nc` 120 s), the Q(V) droop (v_ref 1.03, dead band 0.01,
  slope 0.06).
- Variants with STS-OFO (S2, O3) and CEN (M) drop the coupler relays after the initialisation as before, so the band
  does not act there.

### Resulting coupler rule per variant
| Variant | Coupler OLTCs at run time |
|---|---|
| Case 2 L2, S1, O2 | local relay 1.03 +- 0.02, 120 s |
| Case 2 S2, O3 | STS-OFO |
| Case 2 M | CEN MIQP |
| Case 1, all variants (STS in local mode) | local relay 1.03 +- 0.02, 120 s |

## Checks
- `pytest tests/test_coupler_relay_band.py tests/test_setpoint_tap_relay.py tests/test_local_rules.py`: 14 passed.
- `run_case1.assert_case1_controlled()` and `variants.assert_case2_controlled()` pass; resolved configs carry
  `coupler_relay_tol_pu = 0.02` in every Case 1 and Case 2 variant (the field is common, so no guard changes).
- Smoke runs (scenario E2, 8 min, `unit_commitment_window_s=21600`, noise seed of REF; scratchpad, not runs of
  record): L2, O2 and O3 complete (24 records each).
  - L2 and O2 log `[local DSO] coupler relay band 1.030 +-0.020 p.u. on 12 couplers (start taps from +-0.010)`.
  - O3 drops the coupler relays as before (no band line).
  - L2 also shows the handover's 345 kV relays (machine transformers and IEEE-12 network OLTCs, +-0.020) and
    8 bank relays.
  - All three start from the same Phase 1b terminal ramp (1.030 -> 1.000 pu, machine taps {0: 2, 2: 2, 6: 2,
    11: 3, 10: 0, 12: 2}) and the same Phase 2 coupler initialisation (+-0.010).

## What has to be rerun with this base
As listed in `2026-11-10_avr_init_and_tap_relays.md`: Ch 9 Case 1 (K1-REF, K1-PAR if kept), Ch 10 REF (E2, P2R,
P2R_B03B05) for all six variants and the MC ensemble (N = 100), Ch 11 if it builds on the same base, Ch 8 or at least
the lambda_max(B) re-evaluation.

Gate before the reruns (from the handover): re-evaluate lambda_max(B) of `ch9_selected_nr_2026_11_v5` on the Ch 8
design points with the new 1.00 pu start (first check: zone 2 +4.9 %); keep the weights if clearly below 2.

## Open
- CEN rule (session cen-tuning-2): its variant change is to branch from this base.
- Commit of this branch (author).
