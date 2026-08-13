# 2026-06-23 — Zigzag investigation in 005_cigre V3/V4/V5 (Q(V)-deadband × OFO)

## What was added (investigation only; no controller code changed)

- `experiments/diag_zigzag.py` — loads `results/005_cigre/{V3,V4,V5}/log.pkl`,
  computes per-signal oscillation metrics: reversal-rate (`rev_rate`, fraction of
  consecutive non-zero increments that flip sign; ~1.0 = period-2 chatter),
  total-variation ratio (`tv_ratio = sum|dx| / range`; >>1 = fails to settle),
  step_rms/step_max. Writes `results/005_cigre/_zigzag/{report.txt,*_zigzag.csv}`.
- `experiments/diag_zigzag2.py` — windowed `rev_rate`/`step_rms` vs sim time with
  the contingency schedule overlaid, to separate intrinsic chatter (present at
  t=0) from stale-frozen-H bursts (grow after a contingency). Read-only.

## Log facts (so future reads don't re-derive)

- All three pickles are dt=60 s. V3 = 300 records (300 min). V4/V5 = 600 records
  (600 min) — V4/V5 were run under an EARLIER, longer-horizon config than the
  current `make_cigre_config()` (which is now 300 min, dt_s=20, tso_period 180 s).
  In the pickles the TSO fires ~every 6 min, DSO every minute.
- Contingencies (current driver): gen2 trip@60, load@b11 +300/150 @120,
  gen2 restore@180, line25 trip@260 (V4/V5's older run also has the @360 events
  inside its 600-min window).

## Findings

The "zigzag" is a CONTINUOUS phenomenon. The OLTCs do NOT hunt (corrected below).

0. **OLTCs do NOT limit-cycle (audited, `diag_oltc.py`).** Earlier `rev_rate≈1.0`
   on `oltcTap_DSO_*` was a METRIC ARTIFACT — `rev_rate` discards flat segments, so
   a tap that moves only a few times in alternating directions scores ~1.0. Real
   counts over 300 min: each V3 interface OLTC moves 4–7 times with 2–5 reversals,
   and the move-times are SYNCHRONISED across all 12 OLTCs (min 60/90/120/150/180/
   186) — i.e. they step at the contingencies/profile changes, then hold (worst
   trace `DSO_1|trafo_2`: −3→−2→−3→−2→0→−1, settles). That is TRACKING, not hunting.
   V4 OFO OLTCs move 0–2 times with ZERO reversals; V5 0–8 with ≤2. No OLTC fix
   needed; OLTCs are not a zigzag source.

1. **Continuous interface-Q / inter-zone tie-flow churn — all variants.**
   - V3 (local DSO): present from t=0 (before any contingency), `tv_ratio` of the
     interface-Q up to ~6.5 — i.e. the realised interface Q travels 6–7× its own
     range over the run. Intrinsic to local Q(V)+OLTC reacting to the slow TS-OFO.
   - V4 (OFO DSO): OLTC hunting gone and interface-Q `tv_ratio` drops to ~2; residual
     oscillation concentrates in inter-zone TIE FLOWS (`tieQ_*`, `rev_rate` ~0.2–0.35).
     Two regimes: larger-amplitude bursts (`step_rms` up to ~6) around disturbances,
     plus a LOW-amplitude settled-state dither late in the run (`rev_rate` ~0.6 but
     `step_rms` ~0.03 Mvar — cosmetic).
   - V5 (central OFO): same as V4 minus the cascade gap — tie/interface dominate.

   Net: moving local→OFO REDUCES the zigzag (kills OLTC hunting, halves interface
   churn). The residual is inter-zone tie coordination + disturbance-response bursts.

## Root cause (verified in code) — NB: frozen H is INTENTIONAL (OFO premise)

The fixed approximate sensitivity is by design: OFO relies on the measurement loop,
not a re-derived model, to correct steady-state error. So "stale H" is NOT the
diagnosis. Config facts (for the record): analytical reduced-Jacobian H
(`numerical_h=False`), bare DER columns (`apply_qv_h_transform=False` ⇒ ∂y/∂q_set =
∂y/∂Q, gain I), held for the horizon (`local_sensitivities_*=True`,
`sensitivity_update_interval=1e6`, `refresh_shared_jac_on_tso=False`, shunts off).
The relevant structural consequence is that under `local_sensitivities_tso=True` the
coordinator ZEROES the off-diagonal cross-zone H_ij blocks
(`multi_tso_coordinator.py:548`, `zero_offdiag`) — each zone's OFO is blind to
inter-zone coupling by construction.

Why the oscillation is NOT a DER-gain (T') problem — note the DER channels are the
quiet ones: `dsoQder_*`/`zoneQder_*` have `rev_rate` 0.04–0.13. The oscillators are
(a) inter-zone TIE FLOWS (V4/V5) and (b) LOCAL OLTC taps (V3). T' acts only on DER
columns ⇒ it targets the wrong variables.

Scalar OFO contraction (per output channel, with step penalty w=g_w, weight q):
  Δu = (g_m² q + w)^{-1} g_m q (y*−y_k),  Δy = g_t Δu  ⇒  e_{k+1}=(1−L)e_k,
  L = g_t g_m q / (g_m² q + w),  overshoot/oscillation when L>1.
  • bare H, deadband: g_m=g_t=1      ⇒ L=q/(q+w) < 1     (matched, non-overshooting)
  • bare H, slope:    g_m=1, g_t=T'<1 ⇒ L=T'q/(q+w) < 1   (under-steps, safe/slow)
  • T',  deadband:    g_m=T'<1, g_t=1 ⇒ L=T'q/(T'²q+w); L>1 if w small  → OVERSHOOT
  • T',  slope:       g_m=g_t=T'      ⇒ L<1               (matched)
So BARE H is non-overshooting in BOTH segments; reviving T' would RISK overshoot in
the deadband — where the system mostly sits (V≈1.03, db±0.01). This validates the
thesis's bare-H choice and means T' would likely make the deadband interaction WORSE.

## On T' (the user's question) — answer: NO, do not bring it back

T' is the correct gain only on the Q(V) SLOPE; in the deadband the correct gain is
identity (bare H), and reanchoring (V_anchor:=V_meas) keeps OFO DERs at the deadband
centre. Per the scalar analysis above, a frozen T' over-models the in-deadband gain
and can drive the DER channel to overshoot when g_w is small. It also only touches
DER columns, which are not the channels that oscillate. A deadband-aware active-set
T' (K=0 in deadband/saturated, K=R on slope) would be "correct" but requires per-step
state ⇒ effectively un-freezing H, against the OFO design.

## Where the oscillation actually comes from + levers (not yet tested)

1. Inter-zone TIE-FLOW hunting (V4/V5): cross-zone coupling is zeroed by
   `zero_offdiag` (intentional). Lever = horizontal coordination
   (`multi_tso_coordinator`, `zone_contraction_lhs` diagnostic) and/or raising g_w on
   the PCC/gen columns that drive the tie, or lowering g_v. NOT T'.
2. V3 LOCAL interface churn (higher tv_ratio than V4): driven by local Q(V) DERs +
   event-synchronised OLTC tracking. Reduced simply by moving to OFO DSO (V4). NOT an
   OLTC-hunting problem — taps are well-behaved (see finding 0).
3. Late-run low-amplitude DER dither (V4, step_rms ~0.03 Mvar): deadband-edge
   active-set flutter. Cosmetic; if wanted, a small OFO step deadband/hysteresis.

## Confirmatory runs (diag_confirm.py, dt=60, 300 min; results/_diag_confirm/)

Round 1:
- **T' is a no-op in V3**: `apply_qv_h_transform` False vs True give BIT-IDENTICAL
  metrics (tieQ 0.33/2.4, ifaceQact 0.29/4.2, dsoQder 0.40/5.1). Expected — V3 DSO is
  local Q(V), so there are no DSO OFO DER columns for T' to touch where the churn is.
- **g_w_pcc x10 does NOT damp the tie flows** (V4): tieQ rev 0.24→0.27, step_rms
  2.83→3.02 (slightly worse); it only shrank PCC/DER steps (zoneQpcc step_rms
  0.97→0.18). ⇒ the inter-zone tie oscillation is NOT driven by the PCC setpoint;
  my g_w_pcc hypothesis is FALSIFIED. Tie lever is horizontal (gen/AVR, g_v, or the
  coordinator), tested in round 2 (diag_confirm2.py: V4 T' on/off, V4 g_v 1e7→1e6).

Round 2 (diag_confirm2.py):
- **T' is a no-op in V4 too** — `apply_qv_h_transform` False vs True BIT-IDENTICAL,
  even though V4 has OFO DER columns at both layers for T' to act on. DEFINITIVE:
  reviving T' changes nothing in either V3 or V4. (Likely S_VQ·R small at the DER
  buses / DERs in deadband, so T'≈I.)
- g_v 1e7→1e6: tieQ rev 0.24→0.16 but step_rms 2.83→3.57 (bigger steps); a tracking
  trade-off, not a clean fix.

## THE ACTUAL ZIGZAG (user clarified): V3 e_v SAWTOOTH at the TSO cadence

The zigzag the user means is in Fig3a — the system-wide TS voltage tracking error
e_v, V3 (blue) after the 60-min gen trip. `diag_voltage.py` extracts e_v(t) from
`zone_v_rms_err_pu` and overlays TSO steps. It is a CLEAN SAWTOOTH, not chatter:

  60min* 11.9 → 61–65 ramps 12.2→13.3 → 66min* 11.9 → 67–71 ramps →13.3 → 72min* …

Each ~6-min TSO step drops e_v; between steps it ramps monotonically back UP. Teeth
spacing = TSO period (6 min); V3 peak-to-trough = 1.86 mp.u. vs **V4 = 0.20, V5 =
0.39** (≈10× smaller). V1/V2 (no OFO) have no teeth.

Mechanism = inter-sample drift of the SLOW outer loop with NO fast inner loop:
- The TSO-OFO corrects the EHV voltage only every 6 min.
- In V3 the DSO is LOCAL Q(V) and ONE-SIDED (g_w_pcc=1e10 pins the interface), so the
  DSO does NOT track the PCC setpoint. Between TSO ticks nothing absorbs the (morning
  load-ramp + local-Q(V)) drift, so e_v grows back until the next TSO correction.
- V4's DSO-OFO IS that fast inner loop (fires every minute, tracks the interface), so
  the ripple collapses → smooth e_v. This is a SELLING POINT for the cascaded scheme.

So: NOT an instability, NOT a deadband limit cycle, NOT a T' problem. It is sample-
and-hold ripple of the 6-min TSO loop. Local Q(V) contributes to the between-step
drift (the HV DERs regulate their own bus, not the EHV target) but the amplitude is
set by (drift rate)×(TSO period). Decisive test pending (diag_confirm3.py): V3 at
tso_period ∈ {360,120,60}s — if p2t scales ~linearly with the period, it is confirmed
inter-sample drift. Levers: faster TSO cadence, or the cascade (V4) — NOT T'/deadband.
Artifact: results/005_cigre/_zigzag/v3_ev_zoom.png.

## CORRECTION: stale data + wrong cadence; DECISIVE cosphi test (2026-06-24)

User flagged (correctly) that all the above used STALE pickles: results/005_cigre is
TSO@360s/dt60 (V4/V5 even 600-min), and diag_confirm* forced dt60 for speed. The
TRUE config is TSO 180s / DSO 20s / dt 20s / 300 min. Re-ran from make_cigre_config()
UNMODIFIED (diag_fresh.py -> results/_diag_fresh/), 900 records each.

**Decisive Q(V) isolation (diag_fresh_pair.py).** Ran Vdbg = TS-OFO + STS **cosphi**
(Q(V) OFF at HV), identical to V3 otherwise:

  e_v sawtooth peak-to-trough, window 65-175 min (morning load ramp):
     Vdbg (Q(V) OFF) = 0.89 mp.u.    V3 (Q(V) ON) = 0.76 mp.u.
  window 200-255 min (load flat):
     Vdbg = 0.10                     V3 = 0.10

⇒ **Turning Q(V) off does NOT remove the sawtooth — it is identical (Vdbg even
slightly larger).** The user's Q(V)×OFO hypothesis is FALSIFIED by direct experiment.
The teeth lock to the 3-min TSO ticks and the amplitude tracks the LOAD-RAMP slope
(0.8-0.9 mp.u. during the 8-10 a.m. ramp; 0.10 once load flattens). Q(V) at HV, if
anything, slightly DAMPS it. Artifact: results/_diag_fresh/ev_vdbg_vs_v3.png.

**Mechanism (confirmed): TSO inter-sample drift / sample-and-hold ripple.** The
TS-OFO corrects EHV voltage only every 180 s; between ticks the operating point
drifts with the load profile and the tick resets it. Amplitude ≈ (load-drift rate) ×
(TSO period): halving the period 360→180 s roughly halved p2t (1.86 → ~0.8 mp.u.), and
the teeth vanish (0.10) when the load is flat. NOT a Q(V) effect, NOT a deadband limit
cycle, NOT a T' problem (T' already shown bit-identical).

At the CORRECT cadence the V3 ripple is much milder than the stale figure implied
(0.76 vs 1.86 mp.u.). Levers: faster TSO cadence (linear reduction), or accept it
(small). V4 (DSO-OFO inner loop) and V5 (central) amplitudes from the same fresh run:
pending (diag_fresh.py still running V4/V5); user notes the ripple is visible there
too — expected, since both have a 180 s outer loop sampling the same drift.
