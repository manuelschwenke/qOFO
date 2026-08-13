# G 01 modelling asymmetry, DSO_4/zone-3 gap diagnosis, and the H1/H2/H3 tests

**Date:** 2026-07-23 (evening)
**Area:** Gate E static-vs-RMS parity — plant modelling (G 01) + DSO coupler OLTC weight
**Runs:** 0047 (oltc=150, event profiles), **0049 = H1** (oltc=1000), **0050 = H2** (oltc=1000 + G 01 AVR), 3 h overnight run launched
**Code added:** `pf/probe_g01_avr_plan.py`, `pf/add_g01_avr.py`, `pf/add_g01_gov.py`

---

## Question

Run 0047's figures showed the RMS plant sitting **below** the static plant in
DSO_4 and in TS zone 3, and the user asked (a) why, and (b) whether G 01 — the
10 GVA "Rest of U.S.A. / Canada" equivalent, which has no AVR in RMS — is
modelled consistently in both plants.

## Finding 1: G 01 was modelled incompatibly (now partly fixed)

| | static (pandapower) | RMS (PowerFactory), before fix |
|---|---|---|
| role | `gen[9]` @ bus 40, `slack=True`, `vm_pu=1.03` | bare `ElmSym`, composite `Rest of U.S.A. / Canada` with **every controller slot empty** |
| voltage | **pinned 1.03** | **floats** — bus 38 drifts 1.0292 → 1.0343 pu over 900 s |
| reactive | **unlimited** | network-determined, 58.4 → 23.8 Mvar |
| active / frequency | algebraic `distributed_slack`, weight = `sn_mva`, **no frequency state** | real inertia, **no governor** → speed 1.0000 → 0.9980 pu (−0.2 %) |

Structurally confirmed: the ComRes export contains `AVR 03/04/07/09/10` — there
is **no AVR 01**. The +0.0043 pu drift at bus 38 matches the +0.0044 pu measured
static-vs-RMS gap at that bus and the +0.0043 pu TS zone-1 gap.

**Note:** the 2026-07-21 "G 01 actuator asymmetry removed" fix only stopped the
OFO from *dispatching* its V-ref. It never made the two plants equivalent — the
plant-level asymmetry above survived it.

## Finding 2: the DSO_4 sag was mostly the coupler taps (H1 fixed it)

Run 0047 used `g_w_dso_oltc=150`, i.e. **without** the 2026-07-22 runaway fix.
Its coupler taps diverged again (DSO_4·trafo_11 −4 vs −1 static, trafo_9 −3 vs −1).

**H1 = re-apply `g_w_dso_oltc=1000`** (run 0049):

| | 0047 (150) | 0049 (1000) |
|---|---|---|
| coupler tap divergences | 5 | **0** |
| DSO_1 / 2 / 3 gap | −0.0066 / −0.0071 / −0.0064 | −0.0009 / −0.0014 / −0.0006 |
| **DSO_4 gap** | **−0.0091** | **+0.0011** |
| zone 1 / 3 gap | +0.0043 / −0.0022 | +0.0043 / −0.0023 (**unchanged**) |
| machine Q excess | +95.3 Mvar | +95.6 Mvar (**unchanged**) |

So the DSO-area discrepancy was **predominantly the discrete tap divergence**,
and H1 removes it. Critically it left the TS-zone dipole untouched, proving
zone 3 is a **separate mechanism**.

## Finding 3: H2 (AVR for G 01) helps but does not explain the rest

`pf/add_g01_avr.py` copies a working `avr_IEEET1` into G 01's existing composite
(same `SYM Frame_no droop` frame as every other plant) and binds it to the empty
`Avr Slot`; Gov/Pss are deliberately left empty so the test isolates *voltage*.
Slots are bound BY NAME via `ElmComp.pblk` (index writes can evict the machine).
Smoke test: bus 38 held 1.02921 → 1.02921 over 60 s (drift 0.00000), `s:usetp`
initialised to 1.03 by ComInc, `c:Ka=6.2` live (no dead-parameter table).

Run 0050 vs 0049 (only difference = the AVR):

| metric | 0049 (no AVR) | 0050 (+AVR) |
|---|---|---|
| zone 1 gap | +0.0043 | **+0.0032** (−26 %) |
| zone 3 gap | −0.0023 | −0.0022 (unchanged) |
| machine Q excess | +95.6 | **+82.1 Mvar** (−14 %) |
| DSO_1/2/3 gaps | −0.0009/−0.0014/−0.0006 | −0.0002/−0.0005/+0.0002 |
| DSO_4 setpoint delta | −24.18 | −24.25 (unchanged) |

**The AVR is a correct fix and helps, but ~74 % of the zone-1 gap, the whole
zone-3 sag and the entire −24 Mvar DSO_4 setpoint divergence remain unexplained.**

## Finding 4: H3 (governor) BLOCKED — the droops are not uniform

Parity requirement: static uses `distributed_slack` with `slack_weight = sn_mva`
(`network/ieee39/build.py:171`) on every in-service machine, so
`dP_i = dP_tot * S_n,i / sum S_n` — G 01 absorbs **69.93 %** of any imbalance
with **zero** frequency deviation. In RMS `dP_i = -(df/R_pu,i) * S_n,i`, so the
static law holds **iff every machine has the same per-unit droop**.

`pf/add_g01_gov.py --report` shows they do **not**:

* G 03, G 04, G 07, G 09: `gov_IEEEG1`, `K = 5.0` → R ≈ 0.20 pu (20 %)
* **G 10: `gov_IEEEG3`, `Sigma = 0.04`** → permanent droop **4 %**
* G 01: no governor at all

Implied current RMS sharing (∝ S_n/R): **G 10 ≈ 60 %**, G 09 ≈ 12 %,
G 03/G 04 ≈ 9.6 %, G 07 ≈ 8.4 %, **G 01 = 0 %** — versus the static target of
G 01 69.93 %, G 09/G 10 6.99 %, G 03/G 04 5.59 %, G 07 4.90 %. The P-imbalance
sharing between the two plants is therefore **completely different**.

**H3 was NOT applied.** Satisfying the user's constraint ("steady state in
accordance with the steady-state plant droops") would require not just adding a
governor to G 01 but **re-tuning G 10 (and harmonising all droops)** — a change
to already-working machines, i.e. an architectural decision for the user.
`add_g01_gov.py` is ready (`--report` / `--apply` / `--test-sharing` /
`--revert`); `--test-sharing` does a load-step test and prints realised ΔP share
per machine against the static target, which is the acceptance criterion.

**Corrects an earlier memory note** claiming "GOV 02..10, uniform K=5" — only the
IEEEG1 subset is uniform; G 10 is IEEEG3 at 4 %.

## Correction to an earlier claim in this session

An intermediate analysis reported a "−97 Mvar DSO DER deficit balancing a
+95 Mvar machine excess". That was an **artifact**: the ElmRes DER sum matched
only the 7 `DER_DSO_*` parks per area and missed the **3 coupling
WP_STATCOM parks** (10 sgens per DSO). Per the records (all 10, same source in
both plants) DER Q is essentially equal (220.8 static vs 222.0 RMS). The real
asymmetry is the **machine Q excess alone** (+95.6, → +82.1 after the AVR).

## Overnight run launched

`--duration 10800 --profiles --profile-delivery elmfile --dso-oltc-switch-cost 1000
--stride 100` — i.e. the two validated fixes (oltc=1000, G 01 AVR), governor
untouched. Still under the ±1.0 pu DER capability override (diagnostic, not
physical).

## Open / next

1. **Largest open item:** what drives the residual zone-1 (+0.0032) / zone-3
   (−0.0022) dipole, the +82 Mvar machine Q excess and the −24 Mvar DSO_4
   setpoint divergence. Neither the taps nor G 01's AVR explain it.
2. **H3 decision** (user): harmonise all governor droops for P-sharing parity,
   or accept and document the mismatch. Note it is probably second-order for the
   Q/V question (ZIP loads here are voltage-dependent, not frequency-dependent).
3. Revert `der_q_capability_override_pu` before any published result.
4. Event-pool size still scales with duration (~545 slots/target for 3 h);
   ElmFile playback removed the profile half only.

---

## Overnight 3 h run RESULT (run 0051, for morning review)

Completed cleanly: 540/540 RMS steps, no error, **no per-interval accumulation**
(D-step wall time held ~23 s then *eased* to ~13 s — the 51 k-slot event folder
costs a constant overhead, it does not compound). ElmFile profile playback +
pre-created pool worked over the full horizon. Config: oltc=1000, G 01 AVR,
±1.0 pu DER override, elmfile profiles.

**Endpoint (t = 10800 s) vs static:**

| quantity | value |
|---|---|
| coupler tap divergences | **0** |
| DSO_1 / 2 / 3 interface-Q \|err\| | **0.13 / 0.09 / 0.01 Mvar** (near-perfect) |
| **DSO_4 interface-Q \|err\|** | **9.75 Mvar** (Q_set −76.60, Q_act −66.85) |
| DSO area V gap | DSO_1 +0.0006, DSO_2 −0.0002, DSO_3 −0.0001, **DSO_4 +0.0047** |
| TS zone V gap | zone 1 **+0.0008**, zone 2 +0.0001, zone 3 **+0.0001** |
| machine Q excess (rms − static) | **+104.9 Mvar** |

**Key change vs the 900 s runs — the TS-zone dipole is essentially GONE at this
endpoint** (zone 1 +0.0008 / zone 3 +0.0001, vs +0.0032 / −0.0022 at t=900 s in
run 0050). So the zone-1/zone-3 dipole is **operating-point dependent, not a
fixed structural asymmetry** — it appeared at the 900 s profile state and has
closed by the 3 h state. (These are different operating points: profiles have
evolved 3 h; comparing endpoints across durations compares different states.)

**What persists:**
1. **DSO_4** is the lone outlier — commanded to *strongly absorb* (Q_set
   −76.6 Mvar), tracks only to −66.9 (9.75 Mvar short), +0.0047 pu voltage gap.
   The RMS TSO keeps commanding DSO_4 very differently from the static TSO.
   This is the persistent, operating-point-robust open item.
2. **Machine Q excess +104.9 Mvar** (even larger than the 900 s +82–95) — the
   RMS plant still sources materially more reactive support from the EHV
   machines. Yet at this endpoint it does NOT produce a zone dipole, which means
   the machine-Q excess and the zone-voltage gap are only loosely coupled.

**Caveat:** these are single-endpoint numbers. The trajectory figures (being
written) will show whether the dipole appears *transiently* during the 3 h and
whether DSO_4's lag is steady or intermittent.

**Recommended next (morning):** focus on DSO_4's setpoint divergence — why the
RMS TSO commands DSO_4 to absorb ~77 Mvar. That is now the single robust
discrepancy; the zone dipole is operating-point transient and the taps + G 01
AVR are handled.

---

## DSO_4 root cause found (2026-07-23, late) — DER Q(V) re-droop divergence under profiles

User proposed starting both plants from the identical operating point. Chased it:

1. **Corrected t=0 test** (`scratchpad/t0_compare_v2.py`, builds RMS + `read_y`):
   the two plants **start matched to 1e-5 pu** everywhere incl. DSO_4 (RMS ComInc
   reproduces the static LDF exactly). So the same-start handshake is already
   satisfied and would not fix DSO_4. (v1 wrongly showed 0.00000 — it compared
   the pandapower net to itself; v2 harvests the actual RMS ComInc state.)

2. **Profiles-OFF run 0052:** DSO_4 and DSO_1 setpoints are **identical**
   static-vs-RMS (Δset = 0.00), q_act within ~1 Mvar. ⇒ the whole DSO_4
   divergence is caused by **profiles**, from dispatch step 1.

3. **`--profile-settle 10` (run 0053): NO effect** (DSO_4 Δset = −7.45, same as
   settle=0). So it is *not* a read-timing lag.

4. **Step-1 pre-control measurement diff** (`scratchpad/step1_meas_diff.py`):
   profiles-OFF everything matches to 1e-4; profiles-ON the **DER reactive power
   diverges at step 1, before any OFO dispatch**:
   - DSO_4 group DER Q: static 21.82 vs rms 3.37 (Δ −18.4 Mvar)
   - zone-3 DER Q: static **90.4** vs rms **−1.4** (Δ −90 Mvar)
   - zone-3 gen Q: static 188.7 vs rms 266.3 (Δ +78 — machines cover the gap)
   - DSO_4 mean V: Δ −0.015 pu

**Mechanism:** when the profile shifts the operating point each interval, the
**static plant's DERs autonomously re-droop** (QVLocalLoop, refined by
`run_control` re-solve + `seed_qv_equilibrium`) and inject ~90 Mvar in zone 3;
the **RMS plant's DERs do not reach that same Q(V) equilibrium** at the moment
the OFO reads — they sit near their ComInc Q. The OFO reads an 18–90 Mvar
different DER-Q state and commands DSO_4 differently. This is exactly the
divergence documented in `pf/plant.py` ("static re-droops between dispatches,
RMS doesn't") — **dormant without profiles, dominant with them.**

**This is a genuine plant-model difference, not a runtime knob.** The RMS QVPRE
does not converge to the static QVLocalLoop's re-droop fixed point under a
moving operating point. Options (all need investigation + validation):
- diagnose why the RMS QVPRE doesn't re-droop at read time (Vanchor
  re-anchoring cadence? Kdroop? insufficient dynamic settle?);
- seed the RMS DER Q to the droop equilibrium each interval (mirror the static
  `seed_qv_equilibrium`), off-clock, before the OFO reads;
- or decide the static plant's per-dispatch full re-droop is the unphysical
  side and make it hold constant Q between dispatches like the RMS (changes the
  reference).

**3 h re-run WITHHELD** — no complete fix, so an overnight run would not be
meaningful. Model left with the G 01 AVR applied (run 0051 config); no code
changed for this investigation (`--profile-settle` is a CLI flag only).
