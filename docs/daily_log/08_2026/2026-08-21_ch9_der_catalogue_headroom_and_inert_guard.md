# 2026-08-21 — the DER settling row measured a park with no capability

**Timestamp:** 2026-08-21, morning session (server, `Z:\Python_Projekte\qOFO_GH`).
**Touches:** `pf/screening.py`, `experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py`.
**Follows:** `docs/handoff/2026-08-20_ch9_session_pickup.md`, section "OPEN — case 2
returned 0.00 s and the actuator did not move".

## What was wrong

The 2026-08-20 battery (`results/timescale/full_t0_wecc/20260820-135439`, commit
`6126898`, clean, 16/16 cases, **exit 0**) wrote three `0.00 s` rows, two of which
reached `timescale_table.tex` as numbers. Exit 0 did not catch them because the
acceptance guard only tests *censoring* — a settling time at or beyond the horizon
— and never the mirror failure, a settling time of zero because nothing moved.

Diagnosed from the saved trajectories (seat-free, anchored at `t_event = 305 s`)
and then confirmed against the plant:

| case | max post-event deviation | verdict |
|---|---|---|
| `der_q_+20Mvar_DER_DSO_1_s10_b50` | 3.0e-08 pu / 1.6e-06 Mvar | **inert** — the solver noise floor (cf. preflight drift 2.46e-08 pu) |
| `avr_vref_+0.001_G09` | 1.093e-03 pu | event applied; response genuinely sub-band |
| `avr_vref_+0.001_G10` | 8.733e-04 pu | event applied; response genuinely sub-band |

The two AVR rows are **not** a fault. Their responses scale linearly against the
`+0.02` reference to within 2 % (`1.093e-3 × 20 = 2.19e-2` vs `2.14e-2` measured),
so the event fired correctly and the plant response is simply smaller than the
`1e-3` pu band. That is the informative content of those rows — at the magnitude
the tuned TS-OFO actually issues, no settling time is measurable at this band —
but `0.00` is the wrong way to print it.

## Root cause of the inert row

Read-only query of all 32 `ElmGenstat` parks at the frozen t0 operating point
(`ComLdf`, then the `QVPRE` parameter vector, `QVPRE_PARAM_ORDER`):

```
park                        S_n      P     qset    qmin    qmax   head_up [Mvar]
DER_DSO_1_s10_b50          40.0   0.00   0.0000  0.0000  0.0000            0.00
DER_DSO_1_s4_b47           70.0  58.31   0.0000 -0.3300  0.4100           28.70
DER_DSO_1_s5_b48          100.0  45.35   0.0000 -0.3300  0.4100           41.00
DER_DSO_1_s6_b49           80.0  66.65   0.0000 -0.3300  0.4100           32.80
WP_TSO_s0_b18             508.0 230.38  -0.0000 -0.3300  0.4100          208.28
```

16 of the 32 parks sit at `P = 0` and therefore carry `qmin = qmax = 0` under the
VDE capability rule. `QVPRE` clips `qset` to `[qmin, qmax]` every solver step, so
the +20 Mvar dispatch was clipped to zero and the plant never saw it.

`default_catalogue` selected that park by `sorted(dso)[0]` — **a lexicographic
accident**: `"s10"` sorts before `"s4"`. The park was chosen for its name, never
for whether it could execute a dispatch. The TSO side was unaffected only by luck:
`sorted(tso)[0]` is `WP_TSO_s0_b18`, which has 208 Mvar of headroom, and its
commanded 60 Mvar is 0.118 pu against a 0.41 pu rail.

A second defect surfaced in the same query: the step rule `min(60 Mvar, 0.5·S_n)`
ignores the operating-diagram limit. `0.5 pu` exceeds `qmax = 0.41 pu` for **every**
park in this model, so any DSO-side case would have been silently clipped and the
row would have carried a magnitude the plant never saw.

## What changed

**`pf/screening.py`**

- `_qvpre_headroom_mvar(app, park)` — upward reactive headroom in Mvar, read as
  `(qmax − qset)·S_n` from the block that actually enforces it. Returns `0.0` when
  the park, its REEC or its QVPRE cannot be resolved, so an unreadable capability
  is treated as no capability and the caller skips loudly.
- `_first_park_with_headroom(app, gens, prefix)` — replaces `sorted(...)[0]`.
  Deliberately still alphabetical rather than "largest headroom", so the selection
  stays reproducible and comparable with the runs of record. On this model it
  changes only the DSO pick, and keeps it inside the same sub-network:
  `DER_DSO_1_s10_b50` → `DER_DSO_1_s4_b47`. The TSO pick is unchanged.
- Step magnitude is now `min(60 Mvar, 0.5·S_n, headroom)`, so the commanded step
  is realisable. The cap binds on the DSO side (29 Mvar) and never on the TSO side.
- If no park of a class has capability, the row is emitted as `[not run]` with a
  warning rather than as a silent zero.

**`ch_9_1_timescale_seperation.py`**

- `run_case` now carries the variable per monitored signal and computes
  `inert = not any(step > band)` across the controlled outputs. Reported alongside
  `censored`, in `run_meta`, `timescale_summary.csv` and the markdown.
- `build_table` emits `[--] & [$<$band]` for an inert row and never a number.
- The two DER captions carry `{mvar}`, filled from the case name by `_case_mvar`
  (literal `partition`, never regex — the names contain `+`). Same principle as the
  location column: a magnitude that depends on the operating point must not be
  typed ahead of the run.
- `main` exits 2 on any inert row, naming the two possible causes, because which
  one applies is an author judgement and not something the script should guess.
- `--only SUBSTR...` runs a subset by literal substring. Re-running one row costs
  minutes against the ~4.5 h full battery. A partial run always exits 2 and says
  the table is incomplete by construction.

## Verification

- `--self-test`: 30/30 PASS, including two new checks (an inert row never prints a
  settling time; the DER magnitude is filled from the case name).
- `pytest tests/experiments/test_ch9_timescale_seperation.py`: 17 passed.
- `--dry-run`: still prints `gen[1] -> G 03` and `gen[7] -> G 09`.
- Catalogue against the live project: `der_q_+60Mvar_WP_TSO_s0_b18` at 0.1181 pu
  (unchanged, headroom 208.28 Mvar) and `der_q_+29Mvar_DER_DSO_1_s4_b47` at
  0.41 pu (headroom 28.70 Mvar).
- `derive` / `write_outputs` exercised on a two-case subset: tap splits degrade to
  `NaN` rather than raising, and the unrun rows emit `[not run]`.

## Consequences for Table 9.1

- The binding row is unaffected: it is the coupler tap at 16.42 s, margin 3.58 s at
  `T_STS = 20 s`. Neither DER row was ever near binding.
- The DSO DER row changes park, magnitude and caption: `+20 Mvar` on
  `DER_DSO_1_s10_b50` → `+29 Mvar` on `DER_DSO_1_s4_b47`, same sub-network.
- The two `+0.001` pu AVR rows will now render as `[$<$band]` rather than `0.00`,
  and the run will exit 2 until the caption states what that means. The chapter
  should say it explicitly: at the magnitude the tuned TS-OFO issues, the plant
  response does not leave the 1e-3 pu band, so those rows bound nothing.

## Open

- The commanded DSO step lands exactly on the rail (`qset = qmax = 0.41 pu`), so
  the park has no remaining headroom for a Q(V) droop response on top. That is the
  intended worst case, but the caption should not describe it as a generic step.
- The PowerFactory project was **not** modified. The only seat use was a load flow
  and attribute reads.

## Result — `results/timescale/der_headroom_fix/20260821-090759`

`--label der_headroom_fix --save-trajectories --pre-settle-s 300 --only der_q_`,
commit `29522e5`, exit 2 (partial by construction), preflight 2.46e-08 pu — bit
for bit the same preflight drift as the 2026-08-20 run, so the operating point is
identical and the two runs are comparable.

| case | worst signal | `T_s` | inert | censored |
|---|---|--:|:--|:--|
| `der_q_+60Mvar_WP_TSO_s0_b18` | `u_TN_bus18` | **11.77 s** | False | False |
| `der_q_+29Mvar_DER_DSO_1_s4_b47` | `u_TN_bus27` | **4.02 s** | False | False |

- The TSO row **reproduces exactly** — 11.77 s at the same worst signal. The
  catalogue change left it untouched, as intended.
- The DSO row is now a measurement rather than a zero. `u_TN_bus27` is also the
  worst signal the 2026-08-07 run of record found for this class (1.78 s there,
  at a different park, magnitude and operating point), so the row is consistent
  in kind with the pre-defect history.
- Neither case flags `inert`, so the guard is not firing on a healthy case.
- The caption magnitude was filled from the case name: the emitted row reads
  `$+29$\,Mvar` without anything being typed.

At 4.02 s the DSO DER row is far from binding, so **Table 9.1's binding row and
margin are unchanged**: coupler tap 16.42 s, margin 3.58 s at `T_STS = 20 s`.

A full battery re-run is still needed before the table can be pasted — this run
covers 2 of 16 cases and the other 14 rows carry the 2026-08-20 values, which
were measured before the `{mvar}` caption and the `inert` flag existed.
