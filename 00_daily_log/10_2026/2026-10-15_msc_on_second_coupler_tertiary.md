# MSC moved to the second coupling transformer's tertiary

- Timestamp: 2026-10-15 (CEST).
- Reason: Author decision. Two compensation devices on one tertiary are uncommon in practice. Until now each DSO's MSC and MSR shared the tertiary of the first coupling transformer, where their independent integrators energised both together and moved as a 50 Mvar pair (`2026-10-14_shunt_gw_and_msc_msr_sibling_cycle.md`). Now the MSC sits on the second coupler's tertiary and the MSR stays on the first: one device per tertiary.
- The per-tertiary dispatcher draft (`docs/architecture/2026-10-14_tertiary_shunt_dispatcher_DRAFT.md`) is **on hold**. With one device per tertiary it is identical to the existing per-bank integrator. Author decision: validate the new placement first, and group MSC + MSR per DSO only if pairing and both-on persist across the two tertiaries.

## What changed

| File | Change |
|---|---|
| `network/ieee39/hv_networks.py` | `add_hv_networks(..., msc_coupling_pos=1, msr_coupling_pos=0)`: index into each DSO's coupling-transformer list for the MSC / MSR tertiary; fail-fast `ValueError` when out of range. Build order per DSO stays MSC, MSR (the recorder relies on it). Legacy `'bipolar'` build unchanged (first tertiary). Section-4 comment and docstring updated. |
| `configs/config.py` | `tso_shunt_msc_coupling_pos: int = 1`, `tso_shunt_msr_coupling_pos: int = 0`; `0 / 0` reproduces runs before 2026-10-15. Validated `>= 0` in `validate_integrator_mode`. |
| `experiments/runners/multi_tso_dso.py` | Passes both positions to `add_hv_networks`. `shunt_bus_to_dso_id` now maps **every** coupler tertiary, not only the first. Bank interface transformer = the coupler whose tertiary hosts the bank (was hard-coded `coupling_trafo_indices[0]`); `ValueError` if the bank bus is not a tertiary of its DSO. This fixes the feed-forward `dQ_itf`, the Q_PCC gradient term and the DSO capability shift for the moved MSC. |
| `network/ieee39/meta.py` | `coupling_lv_bus_indices` docstring. |
| `tests/test_msc_msr_banks.py` | `test_msc_raises_v_msr_lowers_v` checks each bank against its own bus. New `test_one_device_per_tertiary_by_default`, `test_shared_tertiary_placement_reproducible` (0/0 restores the shared bus; position 9 raises). |
| `tests/test_shunt_integrator_integration.py` | Interface transformer taken from the bank's hosting coupler (`_host_trafo`) instead of `coupling_trafo_indices[0]`. |

Not changed: `export/make_snapshots.py` calls `add_hv_networks` without positions and therefore gets the new default as well. The PowerFactory project is **not** updated.

## Validation

- `test_msc_msr_banks.py`, `test_shunt_integrator_integration.py`, `test_shunt_integrator.py`, `test_tso_tertiary_shunt.py`, `test_shunt_voltage_rescaling.py`, `test_plant_static.py`: 46 passed.
- Smoke run (scratch `smoke_msc_placement.py`; `ch9_selected_shunt_gw006`, 20 min, `shunt_int_g_w = 1e-4` and dwell 0 to force commits; NOT representative): exit 0, 60 records, 49 commits. MSC commits on buses 54/74/94/114 (second coupler tertiaries), MSR on 53/73/93/113. `dQ_itf` 18–31 Mvar per 25 Mvar step, sign MSC engage < 0, MSR engage > 0 as before. Under forcing the MSC and MSR of one DSO still commit in the same instant (release one, engage the other), and DSO_4 ends with both energised. The validation runs below measure whether this persists in representative operation.
- Running: W4, W5 (12 h, evented, DSO_3 ×2) and the 6 h `make_config` scenario, each at `shunt_int_g_w` 0.03 and 0.06, new placement → `results/THESIS_shunt_placement_20261015/`. Reference (old placement): `results/THESIS_shunt_gw_compare_20261014/`.

## Validation results (new placement vs 2026-10-14 shared tertiary)

`results/THESIS_shunt_placement_20261015/`; reference `results/THESIS_shunt_gw_compare_20261014/`. "Paired" = MSC and MSR of one DSO commit in the same TS instant; "both on" = tertiary-pair minutes with capacitor and reactor energised.

| run | placement, g_w | commits | rev ≤ 2 h | paired | both on [min] | e_v_ts | e_v_ds | e_q |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| W4 | old, 0.03 | 65 | 52 | 62 % | 309 | 0.00221 | 0.00706 | 319 865 |
| W4 | **new, 0.03** | 28 | 16 | 0 % | 0 | 0.00195 | 0.00528 | 166 923 |
| W4 | old, 0.06 | 41 | 28 | 49 % | 423 | 0.00209 | 0.00701 | 200 521 |
| W4 | **new, 0.06** | **19** | **8** | 0 % | 0 | 0.00190 | 0.00533 | 165 486 |
| W5 | old, 0.03 | 112 | 82 | 96 % | 0 | 0.00655 | 0.00946 | 209 786 |
| W5 | **new, 0.03** | 90 | 65 | 58 % | 57 | 0.00650 | 0.00863 | 178 247 |
| W5 | old, 0.06 | 57 | 33 | 91 % | 153 | 0.00593 | 0.00820 | 163 127 |
| W5 | **new, 0.06** | 77 | 47 | 26 % | 210 | 0.00669 | 0.00843 | 184 851 |
| 6 h | old, 0.03 | 14 | 1 | 43 % | 51 | zone 2: 7.83 mpu | | |
| 6 h | **new, 0.03** | 13 | 1 | 0 % | 27 | zone 2: 8.87 mpu | | |
| 6 h | old, 0.06 | 11 | 1 | 18 % | 24 | zone 2: 7.95 mpu | | |
| 6 h | **new, 0.06** | 10 | 1 | 0 % | 9 | zone 2: 8.99 mpu | | |

No corridor violations in any run.

- **W4 and 6 h:** pairing removed (0 %); W4 commits −57 % (0.03) / −54 % (0.06), all three quality metrics better than the old placement; W4 both-on 0.
- **W5 (zero-capability window, excluded from the Ch-9 ranking):** pairing persists across the two tertiaries (58 % / 26 %), both-on 57 / 210 min, median reversal gap still = 30 min dwell. W5 at 0.06 is now worse than at 0.03 in e_v_ts (+3 %) and e_q (+4 %); DSO taps 1.58/h vs 1.25/h.
- **6 h:** TS zone-2 RMS error +13 % vs old placement; DSO_2 T3 interface RMSE 3.79 → 5.69 Mvar (T3 already lagged its setpoint in the DSO_2 ramp before). Not seen in W4 (e_v_ts better). Cause not analysed.
- **g_w (Q4):** at the new placement 0.06 gives fewer commits in all three (−32 % W4, −14 % W5, −23 % 6 h) at equal quality in W4 and 6 h, slightly worse in W5. Recommendation: keep 0.06 (`run_multi_system_ofo.py` unchanged).

## Consequences / open

- All Part III results before 2026-10-15 used the shared-tertiary placement.
- Per-DSO grouping (draft on hold): needed only for W5-like operation, where both banks of a DSO still pair and sit energised together. Author decision pending.
- Dissertation updated in `Z:\latex_diss_clone` (re-cloned `manuelschwenke/dissertation`): Ch 7 plant description (two places), Ch 3 shunt subsection, Appendix B table B.5, figure `fig:ieee39_hv_underlay`. Log: `00_daily_log/2026-10-15_shunt_one_device_per_tertiary.md` there. Not committed.
