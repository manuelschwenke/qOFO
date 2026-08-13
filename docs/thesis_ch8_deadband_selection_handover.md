# Handover: dead-band selection → thesis Ch. 8 §2

> ## ⚠ THE NUMBERS IN THIS DOCUMENT ARE VOID (2026-07-31)
>
> The **argument** below is still the one to make; every **number** is superseded.
> The U-curve it quotes (runs `0067`–`0075`, δ\* = 0.005) is invalid twice over:
>
> 1. it is on scenario `wind_replace` (410 MW/DSO) and on the topology **before**
>    the 2026-07-29 change; and
> 2. the 2026-07-30 fixes changed the **Jacobian for every scenario** —
>    `JacobianSensitivities` had been linearising about a point up to **0.058 pu**
>    away from the plant solution and is now exact. See
>    `docs/daily_log/2026-07-29_rural700_infeasible_tn_statcom_q.md`.
>
> To regenerate the figures and CSVs, follow
> **`docs/deadband_selection_rerun_handover.md`**. Do not mix pre- and post-fix
> runs in one curve: everything from run `0080` onward is post-fix.

**To:** session with access to both `Z:\Python_Projekte\qOFO_GH` and `latex_diss_ms`
**From:** qOFO_GH code session, 2026-07-29
**Target:** Chapter 8 *"Parameterisation of the Control Hierarchy"*, §2 — selection
of the DER Q(V) dead-zone half-width δ. Chapter structure: see
`docs/thesis_restructure_handover.md`.

**This supersedes `docs/thesis_ch8_handover_results_handover.md` entirely.**
That document framed the result as a droop-vs-OFO "share", which is not a
meaningful quantity here — see §6.

---

## 1. The argument to make

The dead band is a **two-sided design choice**:

* **too narrow** → the droop responds to every small variation. It chatters, and
  it pre-empts deviations the OFO would have resolved optimally at the next
  dispatch.
* **too wide** → the droop does nothing between dispatches, so the local layer
  contributes no fast support at all.

Both extremes are measurable in the same quantity — the interface-Q tracking
error of the cascade — and it has a clear interior minimum at **δ ≈ 0.005 pu**.

This is a **control-performance** justification. Do **not** justify δ by
quasi-steady-state validity; that belongs in Ch. 11 as a consequence only.
Selecting a plant parameter to suit the simulation method would be circular.

---

## 2. Primary result — all three controlled quantities are U-shaped

**These are PROFILED-OPERATION runs, not an injected-disturbance scenario.**
Physical VDE capability, 300 s, all starting 2016-01-05 08:00, δ the only
variable (`rms_phase6_replay` 0067–0073). The cascade follows the load/wind
profile; the metrics are its own controlled outputs.

| δ [pu] | interface-Q mean \|e\| [Mvar] | TS zone V, RMS err [pu] | DS group V, RMS dev [pu] | DER-Q travel [Mvar] | tap sw. |
|---|---|---|---|---|---|
| 0      | **2.846** | 0.00904 | 0.00466 | 100.8 | 0 |
| 0.0025 | 1.242 | 0.00796 | **0.00377** | 104.7 | 0 |
| **0.005** | **1.104** | **0.00761** | 0.00434 | **100.1** | 0 |
| 0.0075 | 1.496 | 0.00990 | 0.00927 | 110.6 | 0 |
| 0.01   | 1.652 | 0.01087 | 0.01323 | 126.4 | 1 |
| 0.015  | **3.013** | **0.01175** | **0.01489** | 156.2 | 1 |
| 0.02   | 2.912 | 0.01109 | 0.01394 | 145.2 | 1 |

Relative to each metric's own optimum:

| δ | interface-Q | TS V | DS V |
|---|---|---|---|
| 0 | 2.58 | 1.19 | 1.23 |
| 0.0025 | 1.13 | 1.05 | **1.00** |
| 0.005 | **1.00** | **1.00** | 1.15 |
| 0.0075 | 1.36 | 1.30 | 2.46 |
| 0.01 | 1.50 | 1.43 | 3.51 |
| 0.015 | 2.73 | 1.54 | 3.94 |
| 0.02 | 2.64 | 1.46 | 3.69 |

**Statements to make:**

1. **All three quantities degrade at both extremes**, with an interior optimum.
   Interface-Q and TS voltage both minimise at **δ = 0.005**; DS voltage at
   0.0025, with 0.005 only 15 % off its best. Three independent controlled
   quantities agreeing is the strength of the argument — a dead band that
   improved reactive-power tracking while degrading voltage quality would be a
   bad trade, and that is not what happens.
2. **δ = 0 is the worst configuration for interface-Q tracking** (2.846, 2.6×
   the optimum). Removing the dead band does not improve control, it degrades
   it: all 44 parks then respond to arbitrarily small deviations, including one
   another's. Useful supporting point for Ch. 5 — the dead band has a
   control-theoretic purpose beyond avoiding converter wear.
3. **The distribution level is hurt most by a wide dead band.** DS voltage
   degrades 3.9× at δ = 0.015 against 1.5× for TS voltage. The dead band lives in
   a distribution-level control, so widening it removes support where that
   support is local. Worth one sentence.
4. Above δ = 0.01, tap switching begins and DER-Q travel rises (100 → 156 Mvar),
   i.e. the discrete actuators start compensating for the absent droop.

---

## 3. Supporting result — droop activity, quiet vs disturbed

From the separate disturbance runs (profiles off, load-step ladder on DSO_4).
Droop contribution measured directly as `|Q − q^set|`, which is the droop term
itself since `Q = q^set − K_droop·dz(V − V_anchor)`:

| δ [pu] | quiescent RMS [Mvar] | peak, +40 % disturbance | peak, +100 % disturbance |
|---|---|---|---|
| 0.0025 | **5.10** | 23.61 | 50.84 |
| 0.005  | 3.40 | 23.67 | 52.35 |
| 0.0075 | 2.72 | 22.32 | 56.10 |
| 0.01   | 3.19 | 19.38 | 65.36 |
| 0.015  | **2.28** | 19.23 | 50.30 |

Use this only as the *mechanism* behind the narrow side of the U-curve:
quiescent activity falls from 5.10 to 2.28 Mvar as δ widens, while at δ = 0.005
the droop still engages fully for real disturbances (23.7 / 52.4 Mvar peak).
It is optional for the chapter; §2 carries the argument on its own.

---

## 4. Figure

```
Z:\Python_Projekte\qOFO_GH\results\handover_study\figures\deadband_selection.pdf
Z:\Python_Projekte\qOFO_GH\results\handover_study\figures\deadband_selection.png
```

Single panel: interface-Q on the left axis, TS and DS voltage errors on the right,
each metric's optimum ringed, both failure modes annotated, decision range
0.005–0.01 shaded. **Use the PDF in LaTeX.**
Regenerate with `scratchpad/deadband_fig2.py`; table with `scratchpad/deadband_vq.py`.

---

## 5. Method

* **Benchmark:** IEEE 39 `wind_replace`, 3 TSO zones + 4 DSO underlays, physical
  VDE capability (the ±1.0 pu diagnostic override is OFF), `g_w_dso_oltc = 200`.
* **U-curve (§2):** profiled operation, tracking measured over 15 dispatch
  intervals on the RMS plant.
* **Mechanism (§3):** dedicated disturbance runs, profiles OFF so the injected
  events are the only excitation. Area-wide load steps on all 20 DSO_4 loads at
  +5 / +15 / +40 / +100 %, each reverted after 40 s, events placed mid-interval.
  Quiescent windows are the settled periods between events.
* **Baseline control:** a no-disturbance run at δ = 0.01 established that the TSO
  layer's 180 s dispatch cadence injects up to 8 Mvar of activity into some
  analysis windows. Disturbance figures in §3 are peak values inside the event
  window; the quiescent column is measured in windows free of that cadence.
* Runs: `results/handover_study/0014…0018` (δ = 0.0025…0.015),
  control `results/handover_control/0001`.
  **Runs 0001–0013 are smoke/verification and superseded ladder-v1 runs — do not cite.**
* Scripts: `scratchpad/handover_study.py` (driver),
  `deadband_selectivity.py` (§3 table), `deadband_fig.py` (figure).

---

## 6. What NOT to write

* **Do not present a droop-vs-OFO "share" of the response.** It is not a
  meaningful quantity under this control law: at every dispatch the OFO
  re-anchors and absorbs whatever the droop delivered into the new `q^set`, so
  the steady-state split is ~100 % OFO *by construction*. Any share number is an
  artefact of where in the dispatch cycle it is measured. Earlier tables
  circulating a "droop share" (29 %, 53 %, 78 %, …) must not be used.
* **Do not cite the machine-outage anchor.** `gen[0]` is electrically remote from
  DSO_4 and moved that group's Q by only 0.2–1.6 Mvar net; N-1 needs its own
  study with a zone-3 machine.
* **Do not present δ = 0 as a candidate** — not deployable, and measured as the
  worst configuration. It appears only as a limiting reference.
* **Do not claim the sub-dead-band regime is characterised.** The +5 % rung's
  droop and OFO components are each 1–3 Mvar, too small to resolve; two ladder
  designs failed to separate it. Report as open.

---

## 7. Caveats that must appear

1. **Single operating point and a short horizon.** The §2 result is 300 s of
   profiled operation starting 2016-01-05 08:00 — one five-minute window on one
   winter morning. δ is the only variable across the seven runs, so the
   *comparison* is controlled, but the optimum is not established across seasons,
   loading levels or operating points. An earlier screening found the system's
   per-interval voltage excursion varies by a factor of 2.5 over the year, so the
   position of the optimum could well move. **State this explicitly** rather than
   presenting δ = 0.005 as a universal value; the defensible claim is that the
   criterion is two-sided and that 0.005 optimises it at this operating point.
   Repeating the seven runs at two or three further profile windows would settle
   it and is the obvious next step if a stronger claim is wanted.
2. The §3 disturbance runs are DSO_4 only, profiles off.
2. The baseline control was run at **δ = 0.01 only**, so δ-independence of the
   background activity is an assumption.
3. Load-step reverts leave ~1.2 % residual base drift across the full ladder.
4. δ is a free design parameter **within regulatory limits**; 0.005 and 0.01 are
   both inside them. The study selects between admissible values, it does not
   establish the limits themselves.
