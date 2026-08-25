# 2026-08-25 — Ch. 9.1 experiment A: report worst V and worst Q, not one number

**Timestamp:** 2026-08-25 09:40 CEST

**What changed:** `experiments/ch_9_parameter_selection/ch_9_1_actuator_location_sweep.py`,
`tests/experiments/test_ch9_actuator_location_sweep.py`,
`docs/data/2026-08-25_ch9_actuator_location_by_output_family/` (renamed from
`..._2026-08-24_..._voltage_only`).

**Reason:** yesterday's change reduced the reported worst over nodal voltages
alone and carried the pooled all-output max beside it as `t_settle_all_s`. The
dissertation table needs **both families explicitly** — worst nodal voltage and
worst interface Q — because they are the controlled outputs of the two layers
and are not commensurable. A pooled max is worse than either: it hides which
family bound the result, and over this run it was a Q row in 53 of 70 cases, so
a lone column read as a voltage claim and was not one.

## Method

`V_OUTPUT_KINDS` and `Q_OUTPUT_KINDS` partition every kind
`build_output_specs` produces (asserted at import and in the tests, so a new
output kind cannot fall outside both columns and vanish). `_family_columns`
emits one `worst_*` group per family, suffixed: `t_settle_s` / `worst_signal`
for voltage, `t_settle_q_s` / `worst_signal_q` for Q. The `_all` family is gone
— it is `max` of the two and was the thing being objected to.

`results.csv` additionally splits Q by the layer that tracks it,
`t_settle_dso_q_s` and `t_settle_tso_q_s`. Not in the table, but the two are
tracked at different periods (DSO 20 s, TSO 180 s), so the pooled Q worst is the
wrong quantity to compare against either one alone — see the result below.

`summary.md` prints the two halves side by side. They are reduced
independently, so **the worst-V case and the worst-Q case in a class/direction
group need not be the same case**; both case names are printed. Censoring and
sub-band are marked inline as `(c)` / `(sb)`.

## Defect found and fixed while re-aggregating

`reduce_worst` and `_family_columns` read the censoring flags with `bool()`.
That is correct for live metric dicts (real booleans) but wrong for rows re-read
from `signal_metrics.csv`, where they are the strings `"True"` / `"False"` —
and `bool("False")` is `True`. The first re-aggregation therefore reported
**70 of 70 cases censored** against the run's true 0. Both now use `_truth`,
with a regression test on string-valued rows. This mattered: the archived
per-output rows are the only path from a finished run to the table, so every
number in Ch. 9.1 would have carried a censored flag.

## Result — no RMS was re-run

Run `actuator_location_sweep_t0/20260822-014209`, 70/70 complete, re-aggregated
from its own 6090 per-output rows. 0 censored in both families.

| Family | Case | Worst settling | Output |
|---|---|---:|---|
| nodal voltage | `avr_m0p02pu_G10` | **12.472 s** | `V_TN_bus1` |
| interface Q (pooled) | `avr_p0p02pu_G10` | **17.172 s** | `Q_TSO_Z1_Z2_TN_line2` |
| — DSO interface Q | `tap_p1_MT_g1_t2` | 8.412 s | `Q_DSO_NC3W_DSO_3_t7` |
| — TSO boundary Q | `avr_p0p02pu_G10` | 17.172 s | `Q_TSO_Z1_Z2_TN_line2` |

Q settles later than V in 50 of 70 cases. The binding actuator class is AVR
`v_ref` in both families, with the TSO gen-transformer OLTC next (11.812 s V,
15.262 s Q).

## Effect on the admissible rates — the layer split is the argument

Against `dso_period_s = dt_s = 20 s`, the quantities the DSO layer must clear
are its own: 12.472 s of nodal voltage and **8.412 s of DSO interface Q**,
i.e. margins of 1.60x and 2.38x. The 17.172 s figure is a **TSO boundary**
corridor, tracked at `tso_period_s = 180 s`, a margin of 10.5x.

So the selected rates do not change, and the pooled 17.172 s was never the
right number to compare against 20 s in the first place — it belongs to the
slower layer. Stating V and Q separately, and Q split by layer, is what makes
that visible.

## Assumptions

- Bands unchanged: 1 mpu voltage, 1 Mvar Q, 5 s confirmation, final-window mean
  as the reference value.
- Settling measured from dispatch, including the OLTC's 5 s command-to-contact
  delay.
- Snapshot `full_t0_20160105-0800.json`; disturbances and currents excluded.
- Each family's worst is over that family's rows only; the flag columns
  (`censored`, `subband`) are likewise per family.

## Risks / unresolved

1. **The Q band is absolute, the voltage band is not comparable to it.** 1 Mvar
   is applied to flows whose own step sizes differ by more than an order of
   magnitude across the 12 DSO interfaces and 5 TSO corridors, so a Q worst can
   be set by a corridor with a small step rather than by a slow one. This is why
   the two are reported separately rather than pooled — but it also means the Q
   column is not internally comparable across corridors. A per-corridor relative
   band (a fraction of that corridor's own step) would fix that and is **not**
   implemented. Until then, the Q column bounds settling but does not rank
   corridors.
2. The layer split is reported in `results.csv` only, not in the table. If the
   timescale-separation argument in the text leans on the 8.412 s / 17.172 s
   distinction — and per the section above it should — that split needs to be
   visible in the chapter, not only in the data.
3. `q_minus_v_s` in `results_by_family.csv` is a per-case difference of two
   independent reductions, not a paired quantity: it can be driven by which
   signal won each family, not by a physical lag between them.
