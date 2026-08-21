# Handoff prompt — fill Table 9.1 (settling times) in the dissertation

> ## ⚠ HOLD — do not paste the tap rows yet (2026-08-21, later the same day)
>
> The two OLTC rows, the binding row, the margin and both `T_mech`/`T_elec`
> splits are **under review** and may change. Everything else in this brief
> stands.
>
> The battery applies **two** 5 s lags to a tap and only one of them is
> physical: the command `ntapcmd` is scheduled at `t_event + 5 s`
> (`TAP_MECH_DELAY_S`, a leftover from the retired `EvtTap` path), and the
> `TAPCTRL` DSL block *then* slides the tap with its own first-order lag
> `Tmech = 5 s`. Measured on the saved trajectories: the coupler tap reaches
> 63 % of its excursion at 9.97 s and 95 % at 19.97 s after `t_event`, i.e.
> exactly 5 s of dead time followed by one and three time constants of a 5 s
> lag. The MSC shunt, scheduled with no offset, reaches 95 % in 0.57 s.
>
> If the scheduling offset is removed, `T_s` for the coupler tap falls from
> 16.42 s to ~11.4 s, the **binding row moves from the coupler tap to the
> AVR $+0.02$ pu G 10 row at 15.22 s**, and the margin becomes ~4.78 s.
>
> Awaiting the author's decision on whether a command-to-motion delay on top
> of the mechanical travel is intended. Fill the non-tap rows if you like;
> leave the tap rows, the binding sentence and the margin until this clears.

Written 2026-08-21 on the server session that produced the runs. The target
session owns the thesis repo **and** has `Z:\` mounted, so it can read the run
directories directly rather than working from pasted numbers.

Everything between the rules is the prompt.

---

You are my research assistant on a PhD project on hierarchical multi-zone
reactive-power control. This session owns the dissertation LaTeX repo
(`...\12_Dissertation\latex_diss_ms`) and can read `Z:\Python_Projekte\qOFO_GH`.

**Your job is to fill `tab:param:timescales:settling` in §9.1 and fix the
sentences around it.** The measurements are done; nothing needs re-running.

**Do not invent numbers.** Every value is in the run directories named below.
Where I give a number here, it is so you can check you read the right file — if
the file disagrees with this brief, **trust the file and tell me**.

## Read these first

```
Z:\Python_Projekte\qOFO_GH\results\timescale\full_t0_wecc\20260820-135439\
    timescale_table.tex      timescale_summary.md      timescale_summary.csv
    run_meta.json

Z:\Python_Projekte\qOFO_GH\results\timescale\der_headroom_fix\20260821-090759\
    timescale_summary.csv    run_meta.json

Z:\Python_Projekte\qOFO_GH\docs\daily_log\08_2026\
    2026-08-19_ch9_settling_table_emitter_rework.md
    2026-08-21_ch9_der_catalogue_headroom_and_inert_guard.md
```

## Provenance is SPLIT across two runs — this is the one thing to get right

Fourteen rows come from the full battery `20260820-135439` (commit `6126898`,
working tree **clean**, 16/16 cases, preflight drift 2.46e-08 pu). Two rows —
both reactive-power steps — come from `20260821-090759` (commit `29522e5`,
tree dirty, exit 2 because it ran `--only der_q_` and is partial by
construction), because the DSO DER case in the first run measured a park with no
reactive capability and returned a meaningless `0.00 s`.

The two runs share the same operating point: preflight drift is `2.46e-08` pu in
both, bit for bit. The TSO reactive-power row reproduces exactly (11.77 s at
`u_TN_bus18` in both), which is the evidence that they are comparable.

**Say so in the caption or a footnote.** A table silently assembled from two
runs is exactly the failure that put `11.13 s` and the placeholder "STS 1 B00"
into the current draft. Documented, it is fine; silent, it is the same defect
again.

**This is decided — do not propose re-running.** A full battery re-run (~4.5 h)
would collapse the table to one commit and one stamp, and the author has weighed
that against the evidence of comparability above and chosen to use the data as
it stands. Your job is to document the split, not to remove it.

Suggested footnote, adapt as you like:

> Rows for the reactive-power steps were measured in a separate run of the same
> battery (2026-08-21) after the subordinate DER case was found to target a park
> with no reactive capability at this operating point. Both runs share the same
> pre-settled operating point (preflight drift 2.46e-08 pu in each) and the
> transmission-side reactive-power row reproduces identically, so the rows are
> directly comparable.

## The table

| # | row | location column | `T_s` [s] | from |
|--:|:--|:--|--:|:--|
| 1 | Reactive-power step, $+60$ Mvar, TS DER | `u_TN_bus18` | 11.77 | both |
| 2 | Reactive-power step, $+29$ Mvar, STS DER | `u_TN_bus27` | 4.02 | 0821 |
| 3 | AVR voltage-reference step, $+0.02$ pu, G 09 | `u_TN_bus38` | 6.02 | 0820 |
| 4 | AVR voltage-reference step, $+0.02$ pu, G 10 | `u_TN_bus18` | 15.22 | 0820 |
| 5 | AVR voltage-reference step, $+0.001$ pu, G 09 | — | **`[<band]`** | 0820 |
| 6 | AVR voltage-reference step, $+0.001$ pu, G 10 | — | **`[<band]`** | 0820 |
| 7 | OLTC coupling transformer, one step | `u_DSO_1_bus43` | **16.42** | 0820 |
| 8 | OLTC machine transformer, one step | `u_TN_bus1` | 15.17 | 0820 |
| 9 | MSC switch-in | `u_TN_bus28` | 4.02 | 0820 |
| 10 | Synchronous-machine outage | `u_TN_bus0` | 25.02 | 0820 |
| 11 | Load step | `u_TN_bus38` | 305.82 | 0820 |

The location column is the measured worst controlled output. **Never type it.**

### Row 2 changed park and magnitude — the caption must reflect it

The draft says `+20 Mvar`. It is now **`+29 Mvar` on `DER_DSO_1_s4_b47`**, same
110 kV sub-network as before. The previous target, `DER_DSO_1_s10_b50`, sits at
`P = 0` at this operating point and therefore carries `qmin = qmax = 0` under the
VDE capability rule, so the commanded step was clipped to nothing and the plant
moved by 3e-08 pu — the solver's own noise floor. The magnitude is now capped at
the park's realisable headroom, which is why it is 29 and not a round number.

### Rows 5 and 6 are a result, not a gap

Do **not** print `0.00`. Both events fired correctly — their responses scale
against the `+0.02` pu rows linearly to within 2 % (`1.093e-03 × 20 = 2.19e-02`
against `2.14e-02` measured) — but the plant response never leaves the `1e-3` pu
measurement band. The chapter should say this explicitly, because it is the most
policy-relevant statement in the table:

> at the magnitude the tuned TS-OFO actually issues (< 0.001 pu per iteration),
> the plant response does not leave the measurement band, so these dispatches
> bound nothing.

### Rows 10 and 11 report the FIRST matching case, not the worst

Flagging this because it is a judgement call I should not make for you. Both
disturbance classes ran several cases, and the emitter fills the single table row
from the first match:

| class | tabulated | also measured |
|---|---|---|
| outage | G 03, 25.02 s | **G 09, 30.17 s** |
| load step | $+10\%$, 305.82 s | $-10\%$ 305.82 s, **$+25\%$ 314.52 s** |

So the table under-reports both. Either name the case in the row label, or take
the maximum — but the current unlabelled row is ambiguous. The derived "worst
disturbance" quantity in the summary already uses the maximum, 314.52 s, so the
table and the text currently disagree with each other.

## The sentences that have to change

1. **The binding row.** It is the **coupling-transformer tap at 16.42 s**, not
   the machine transformer. The draft prints `11.13 s` with location "STS 1 B00";
   `11.13` appears nowhere in any run and "B00" is an unfilled placeholder.
2. **The margin.** At `T_STS = 20 s` it is **3.58 s**, not 4.87 s.
3. **The ordering sentence inverts.** The coupler tap (16.42 s) is slower than
   the machine transformer (15.17 s) and slower than the worst continuous
   dispatch (15.22 s). Check the surrounding prose for any claim that depends on
   the old ordering.
4. **`T_s^cont` = 15.22 s**, from `avr_vref_+0.02_G10`.

## `T_mech` / `T_elec`, both classes

From the sequential two-step instrument cases, `T_mech = T_s(2) − T_s(1)` and
`T_elec = 2·T_s(1) − T_s(2)`:

| class | `T_s(1)` | `T_s(2)` | `T_mech` | `T_elec` |
|:--|--:|--:|--:|--:|
| coupler | 16.42 | 22.86 | **6.44** | **9.98** |
| machine transformer | 15.17 | 21.36 | **6.19** | **8.98** |

Two consequences for the caption:

- The **electrical transient dominates** in both classes — a completed tap is a
  step change in ratio like any other, so it *adds* to the mechanical travel
  rather than being covered by it.
- Measured mechanical travel is **6.2–6.4 s**, not the **5 s** block parameter
  the current caption asserts. Correct that claim.

The two-step cases are instruments only and must **not** get a table row; they
are what separates `T_mech` from `T_elec`. The controller caps taps at one step
per iteration and then locks the changer out (60 s coupler, 180 s machine
transformer), so a two-step command cannot arise as a dispatch.

## Caption obligations

- **"Measured open loop" overstates it.** No secondary dispatch occurs, but
  primary control — AVR, governors, and the local re-anchored Q(V) droop — is
  active throughout and is inside every number. The agreed wording is *"no
  secondary dispatch; primary control (AVR, governors, local Q(V)) active"*.
  This was a deliberate choice (option (a) of the 2026-08-19 analysis): keeping
  the Q(V) layer measures the plant a dispatch actually excites, whereas
  neutralising it would measure a plant that never operates.
- **The operating point is not the load-flow solution.** Each case is pre-settled
  for 300 s, so the measurement starts from the RMS steady state of the anchored
  ZIP load model, roughly 1.4e-02 pu from the load-flow point. The caption must
  say so.
- Bands: `1e-3` pu on voltages, `1 Mvar` on interface flows. RMS step 10 ms
  fixed, read stride 5, so every `T_s` is resolved to 50 ms.
- Disturbance rows do **not** enter the bound. The worst is 314.52 s = 15.7
  dispatch intervals, i.e. where a disturbance exceeds `T_STS` the
  quasi-steady-state premise is violated for that window, and the closed-loop RMS
  chapter evaluates the consequence.

## Do not write

`T_TS / T_STS = 9` is the **configured** ratio. It is **not** the measured
`N_inner` of eq. (9.2), and this open-loop battery does not measure `N_inner` at
all. That is a separate experiment with its own figure (see
`2026-08-21_ch9_ninner_figure_prompt.md`). The summary file carries this warning
in its own text; keep the discipline.

## Working rules

1. Distinguish measured facts, hypotheses and open questions. Label anything
   projected or carried over. No invented numbers.
2. If something contradicts this brief, trust the run directories and say so.
3. Answer as: short answer; assumptions; details; risks / open points.
4. Table, caption and the surrounding sentences are the deliverable. Do not
   restructure §9.1.
