# 2026-11-07b — STS q(v) frozen by the OLTC-limiter re-solve (L2, S1, O2)

Morning: read-only diagnosis. Afternoon (author's approval to test the fix):
fix applied in the working tree and tested at E2 to 160 min (section "Fix and
E2 test" at the end). Not committed, no full rerun.

## Question

In E2 (run `ch_11_case2_multisystem/ref/0025_..._v4_E2`), at 91.67–100 min the
STS DERs under L2 and O2 deliver about 0 Mvar while the STS voltages are
0.94–0.99 p.u., far below the local q(v) dead band (V_set 1.03, db 0.01,
slope 0.06). At 91.67 min under O2 the STS draw about +104 Mvar from the
TS; under O3 their DERs inject 88–103 Mvar each. Why is the droop silent?

## Answer

The q(v) loops work. What the log shows is the state that the OLTC rate
limiter leaves behind when it reverts the coupler relays and re-solves with
`run_control=False`.

In local STS mode (`dso_mode="local"`) the 12 coupler OLTCs keep their
pandapower `DiscreteTapControl` relays (band 1.03 ± 0.01, runner l. 2226).
Every plant solve in a step follows the same three stages:

1. `plant.advance` → `runpp_holding_taps(run_control=True)`. Inside this one
   solve the relays tap without limit (e.g. −1 → −6) and bring the STS
   voltage back into their band. The DER `QVLocalLoop`s settle at that
   recovered voltage. The relay band equals the droop's dead band
   (1.03 ± 0.01), so most DERs end at Q ≈ 0. A few end slightly negative
   because of relay overshoot.
2. `_OLTCRateLimiter.clamp` reverts the taps to the step's snapshot
   ± 1 tap (cooldown 120 s for the couplers).
3. `pp.runpp(run_control=False)`. No controller runs, so DER Q stays at its
   stage-1 value while the voltage falls back to 0.94–0.98 p.u. This state
   is recorded.

The same pattern runs at all three re-solve sites in
`experiments/runners/multi_tso_dso.py`: post-profile (l. 3791/3799/3811),
post-contingency (l. 3887/3935/3950) and end-of-step (l. 5233/5236/5248).
`_multi_tso_helpers._OLTCRateLimiter` (l. 191) prescribes that re-solve in
its docstring.

After a disturbance the coupler taps walk at 1 tap per 120 s. For as long as
the relays want more taps than the limiter grants, the droop is held
**outside its own law** and does not act. This is exactly the
post-disturbance window that the Case-2 event metrics measure. Once the taps
have caught up, no clamp is needed and the recorded state is consistent
again.

Probable origin: the pattern dates from local STS DERs at cos φ = 1
(`local_der_mode` default `"cos_phi_1"`, `configs/config.py:1595`). With
constant Q, skipping the control loop costs nothing. With `local_der_mode="qv"`
it freezes the droop.

## Evidence

**Log (O2, 91.67 min).** STS 1–3 have 150–380 Mvar of Q capability
(`dso_der_q_max_mvar`) at V 0.94–0.96 p.u., and sum(Q_DER) = 0.06 / 0.53 /
0.06 Mvar. The console logs post-profile clamps of 9–11 couplers in every
step from step 276 to step 289 and beyond (e.g. `trafo3w#5 -9->-2`).

**Scratch reproduction.** O2 config, E2 start, pre-loop hook, G3 out plus
600 Mvar at bus 3, one solve (script:
`2026-11-07b_sts_qv_limiter_repro.py.txt`). The operating point is the
08:00 profile, not 91:40, so the numbers show the mechanism, not the size of
the effect in E2.

| state | STS 1 | STS 2 | STS 3 | STS 4 |
|---|---|---|---|---|
| (A) run_control, relays free: V_DER min / ΣQ_DER | 1.038 / −13.3 | 1.029 / −10.9 | 1.035 / −3.5 | 1.029 / −17.8 |
| (B) taps clamped, run_control=False (= log) | 0.982 / −13.3 | 0.953 / −10.9 | 0.979 / −3.5 | 1.015 / −17.8 |
| (C) taps clamped and held, run_control | 1.013 / +26.1 | 0.993 / +68.8 | 1.014 / +39.8 | 1.022 / −15.0 |

In (B), Q is identical to (A) at a voltage 5–8 points lower. That state is
not a fixed point of the droop law. Settled (C), the droops inject 26–69 Mvar
per STS and raise the STS voltages by 3–4 points.

**Log-wide check.** I compared the logged ΣQ_DER with the q(v) law
evaluated at the logged DER-bus voltages (open loop; it overstates the
amount). Mismatches above +20 Mvar (the droop should inject more) occur
only in **91.67–99.33 min**, in L2, S1 and O2 alike (54 / 63 / 59
step–STS pairs, up to the full capability). Mismatches below −20 Mvar
(the droop should absorb more) occur after the G3 trip (30.7 min) and the
G3 restore (152.3–154 min), in S1 up to −174 Mvar.

**Ruled out:**
- Anchor following the voltage (the lead from
  `2026-11-07_svr_calibration_qv_der.patch`): in local mode no write path
  touches the STS DERs. `qv_vref_anchor_pu` stays NaN and `q_set_mvar`
  stays 0 throughout, so the droop is centred on the nominal 1.03,
  as configured.
- Loops not installed: all 40 STS `QVLocalLoop`s are installed and in
  service.
- Logged field being a command: `dso_der_q_mvar` and
  `dso_group_q_der_mvar` are `res_sgen.q_mvar`, the plant's Q. The plant
  state itself is inconsistent.

## Verdict: bug

The thesis gives the no-access STS a working local characteristic: "q(v)
on IBGs" (Ch 10, variants table). Chapter 7 (l. 32) says that at each
evaluation instant "the droop laws of the DAD layer are settled to their
algebraic fixed point". The recorded states in the clamp windows violate
both. `ch:setup:references:lower` is still a stub, so it cannot be the
source of an intended exception.

Side effects of the same re-solve:
- The 4 TS STATCOM q(v) loops are frozen in the same way. In L2 that is
  their local q(v); in S1 and O2 it is their response between dispatches.
- `SlackQLimit` (the ±1200 Mvar box of the network equivalent) is not
  enforced in these states.

## What is affected

The path is active wherever the coupler relays are kept: `dso_mode="local"`
and not central (runner l. 3169–3187).

- **Case 2** (`ch_11_case2_multisystem`, ref and step): **L2, S1, O2**.
  Not S2 or O3 (STS-OFO, relays dropped), and not M (central; limiter
  disabled, relays dropped). In E2 this means the access gain O2→O3 and
  S1→S2 is overstated in the post-disturbance windows (Ch 10 l. 282: "whose
  STS droops respond at once"). Runs 0027 and 0028 (running now) are
  affected if they contain these variants.
- **Case 1** (`ch_10_case1_svr`, runs 0001 and 0003): **every variant**
  (L2, S1m, S1, O0, O1, O2 and the K1-PAR derivatives), since all of them use
  `_LOCAL_STS`. The limiter re-solve is logged in each of them; the size of
  the effect is not quantified. The bias is common to all Case-1 variants,
  but the STS behaviour is wrong in all of them.
- **Monte-Carlo ensemble** (`ch_11_case2_multisystem/mc`): **L2, S1, O2**
  of `ALL_VARIANTS`. S2, O3 and M are not affected.
- Also the local-mode rungs of `ladder/variant_ladder.py`. Chapter 9 runs
  default to `dso_mode="ofo"` and do not go through this path (not checked
  further).

## Fix options (morning proposal)

At the three sites, after a clamp, re-solve with `run_control=True` and the
`DiscreteTapControl` relays out of service, as `runpp_holding_taps`
already does on hunting. The Q(V) loops and `SlackQLimit` then settle at the
clamped taps. This changes every L2/S1/O2 result in Case 1, Case 2 and the
MC ensemble. Before any rerun, a short test at the E2 load step would show
how much of the O2→O3 gap remains.

## Fix and E2 test (afternoon, uncommitted)

**Change.**
- `core/plant.py`: new `runpp_relays_held(net, **kw)`, which runs
  `pp.runpp(run_control=True)` with the in-service `DiscreteTapControl`
  relays out of service and restores them afterwards. `runpp_holding_taps`
  now uses it for its own hunting fallback (same behaviour as before).
- `experiments/runners/multi_tso_dso.py`: all three post-clamp re-solves
  call it, with `max_iter=300` and the other arguments unchanged.
- `_multi_tso_helpers._OLTCRateLimiter` docstring updated.
- New test `tests/test_relays_held_resolve.py` (5 tests).

The 12 test files that touch the changed modules all pass (80 tests).
Variants without relays are untouched by construction: S2 has only
`QVLocalLoop` and `SlackQLimit` in its plant (checked), and the limiter is
off in O3 and M.

**E2 test.** O2 and S1 with the fix, same config and seed as run 0025, to
160 min. Output in `results/diagnostics/2026-11-07_qv_relays_held_E2/`
(`run_test.py`, `compare.py`). Wall time: O2 2152 s, S1 5090 s.

The fix works. The logged STS Q now follows the Q(V) law: the largest
mismatch is 7 Mvar, the noise level of the droop's voltage reading, against
up to 377 Mvar in 0025. At the load step, STS 1–3 inject 232 Mvar (O2) and
274 Mvar (S1) instead of about 0. Their interface flow goes from +104 Mvar
drawn from the TS to −120 / −162 Mvar exported. The O3 value is −123 Mvar.
The support then fades: the coupler relays walk their taps (1 per 120 s),
the STS voltage returns into the relay band, which is also the droop's dead
band, and by 98–100 min the fixed run and 0025 coincide. Over the same time,
O3 and S2 keep building support, to −317 / −460 Mvar by 130 min.

Area-2 error [mpu]. The reference is O3 for O2 and S2 for S1. "Gap left" is
(fix − ref) / (0025 − ref).

| | O2 0025 | O2 fix | O3 | gap left | S1 0025 | S1 fix | S2 | gap left |
|---|---|---|---|---|---|---|---|---|
| e at 91:40 | 80.7 | 45.0 | 44.8 | 1 % | 97.7 | 58.4 | 57.5 | 2 % |
| e at 96:40 | 54.8 | 50.8 | 29.1 | 84 % | 72.0 | 64.9 | 26.6 | 84 % |
| e at 121:40 | 50.6 | 50.7 | 19.3 | 100 % | 62.8 | 63.0 | 17.3 | 100 % |
| RMS 91:40–100:00 | 61.7 | 48.7 | 37.3 | 47 % | 76.7 | 63.1 | 46.3 | 55 % |
| RMS 91:40–152:20 | 45.7 | 43.8 | 21.7 | 92 % | 57.7 | 55.9 | 23.1 | 95 % |
| RMS 0–160 min | 29.1 | 27.8 | 14.4 | 91 % | 36.8 | 35.5 | 16.0 | 94 % |

So the fault inflated the access gain only in the first minutes after a
disturbance. Directly after the step, it is almost entirely an artefact:
fixed O2 equals O3. From about 5 minutes on, the gap is genuine; it comes from
the coupler relays restoring the STS voltage into the dead band. Over
0–160 min, 91–94 % of the gap remains. Ch 10 l. 282 ("55–74 mpu under LOC,
SVR and OFO … 35–36 under the variants with access") no longer holds for the
instant directly after the step.

**Other runs, Q(V)-law check of the logs** (mismatches above 20 Mvar):
- Period II 0027 (L2) and 0028 (L2, O2): none. 0028 S1: 4 step–STS pairs
  in 0–2 min (start-up), otherwise ≤ 7.5 Mvar.
- Case 1 0003: 9–17 step–STS pairs per variant in 121:40–125:20 (load step),
  up to 30–74 Mvar, plus single steps at 30–42 min.

## Proposed reruns (not started; host and workers to be agreed)

Per-variant wall times are from 0025 / 0003 and scaled from the test
(O2 +8 %, S1 +10 %).

| run | variants | why | wall per variant |
|---|---|---|---|
| E2 0025 | L2, S1, O2 | load-step window, `fig:case3:event`, `tab:case3:ref`, epost | L2 ≈ 2.3 h, S1 ≈ 3.2 h, O2 ≈ 80 min |
| Case 1 0003 (K1-REF) | L2, S1m, S1, O0, O1, O2 | load-step window 121:40–125:20 | S1/S1m ≈ 3 h, O0 ≈ 2.6 h, L2 ≈ 1.5 h, O1/O2 ≈ 55 min |
| MC ensemble | L2, S1, O2 on the 100 valid episodes | every drawn disturbance | ≈ 160 CPU-h + ~10 % (S1 ≈ 80 min per episode) |
| Period II 0027, 0028 | none needed | no fault above noise after 2 min | (0028 S1 optional, ≈ 2.8 h) |

- E2 and Case 1 together: 9 workers, about 3.3 h wall (bounded by S1).
- MC: about 9–10 h on 18 workers on e5wimi, or 5–6 h on 30 on e5power. If
  the fixed L2/S1/O2 change which episodes are valid (L2 prescreen) or fully
  paired ("all six ok"), S2/O3/M also need the newly entering episodes.
- Commit the fix before the reruns so that the README records a clean
  commit.
