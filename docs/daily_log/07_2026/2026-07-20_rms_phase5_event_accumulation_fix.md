# 2026-07-20 — Event-accumulation bug: diagnosis, fix, corrected Gate D (CLOSED)

**Headline.** The step battery had been contaminated: PF silently refuses
`Delete()` on simulation events while a calculation is active, so
`purge_events()` never actually purged and **every battery entry after the
first replayed all events of the earlier entries in the same PF session**.
After the fix (ResetCalculation before purging + verified-empty,
Fail-Fast), the re-run gives the *true* per-actuator responses:
**Gate D PASSES the 20 s window for every actuator class**, worst case
13.2 s (6.8 s margin) — and the responses are far more benign than the
contaminated battery suggested. **Phase 5 / Gate D is closed.**

## How the bug was caught

Three different disturbances (AVR V-ref, coupler tap ×1, coupler tap ×2)
reported near-identical worst-output metrics (17.96–17.98 s, overshoot
1.895–1.899, step 0.0068 pu at `u_TN_bus18`) — physically implausible.
The per-signal CSVs sealed it: the machine-trafo-tap run showed the DSO_1
coupler at the *2-tap-sequence* operating point (q_STS −2.52 vs −2.50 Mvar,
PCC −2.2 %), which a G1-trafo tap cannot cause; and the `u_TN_bus18` final
offsets grew monotonically down the battery (0.0059 → 0.0061 → 0.00675 →
0.00676 → 0.00676 → 0.00768 → 0.00809) — each run superimposing all
previous events plus its own.

Consequences for earlier records (all corrected today):

- The previously reported "+10 Mvar DSO park: 17.4 s" (also in the
  2026-07-20 screening-build log, now banner-annotated) was in reality
  "+60 Mvar TSO **and** +10 Mvar DSO together". The clean +10 Mvar DSO
  response never leaves the settled bands at all (T_s = 0.0 s).
- The +60 Mvar TSO-park step (13.18 s) was always clean (first entry of a
  fresh PF process) — it reproduces exactly across three sessions.
- `pf/probe_tap_avr.py` verification runs were cross-contaminated too;
  the *mechanism* conclusions stand (each event visibly produced its
  distinctive response), the response magnitudes do not.

## The fix (`pf/screening.py::ScreeningContext.purge_events`)

```
app.ResetCalculation()          # deletion is refused while calc active
delete all events in p_event
verify folder empty, raise otherwise   # Fail-Fast beats silent corruption
```

`PowerFactoryPlant.advance` deliberately does **not** delete mid-run
(same silent no-op; within one continuous run past events cannot
re-fire); cross-run hygiene is the construct-time purge.

## Corrected Gate-D step battery (full_t0_wecc, window 20 s, run 143633)

Settling to absolute bands (voltage 1e-3 pu, interface Q 1 Mvar), measured
from the dispatch instant; taps move 5 s after dispatch (mechanical delay).

| step | worst ctrl output | T_s [s] | margin | overshoot | note |
|:--|:--|--:|--:|--:|:--|
| DER Q +60 Mvar (508 MVA TSO park) | u_TN_bus18 | **13.18** | 6.8 s | 2.08 | the one true ring case |
| DER Q +10 Mvar (20 MVA DSO park) | q_STS DSO_1 | **0.00** | 20 s | — | never leaves the bands |
| AVR V-ref +0.02 pu (G 09, 1 GVA) | u_TN_bus18 | **4.68** | 15.3 s | 2.9¹ | fast AVR tracking |
| Coupler tap +1 (NC3W_DSO_1) | q_STS DSO_1 (7.8 Mvar) | **5.00** | 15 s | 0.00 | = the mech delay itself |
| Coupler tap +2 sequential | q_STS DSO_1 (15.2 Mvar) | **9.98** | 10 s | 0.00 | = delay of the 2nd tap |
| Machine-trafo tap +1 (MT_g0, G1) | u_TN_bus1 | **12.48** | 7.5 s | 0.21 | G1 has no AVR → slow area recovery |
| MSC switch-in (DSO_1 tertiary) | u_TN_bus9 | **1.48** | 18.5 s | 1.11 | fast |

¹ overshoot ratio on a near-band-floor step (0.0007 pu) — not meaningful.

## Corrected interpretation

- **Tap responses are delay-dominated, not ring-dominated**: the
  controlled interface flow re-enters its band essentially the moment the
  tap physically moves; a k-tap sequential move settles at ≈ 5k s. Within
  a 20 s window up to 3 sequential taps fit with margin — a direct design
  number for the OFO's `int_max_step`.
- The **only double-digit electromechanical ring** among realistic
  dispatches is the large TSO-park Q step (13.2 s; ζ ≈ 0.04 modes,
  PSS-off) and the G1 machine-trafo tap (12.5 s; the 10 GVA equivalent
  has no AVR, so the bus-39 area recovers on network stiffness alone).
- Small DSO dispatches (≤ 10 Mvar) are effectively instantaneous at the
  controlled outputs.
- PSS tuning remains optional (margin, not feasibility).

## Gate D checklist (plan Phase 5)

- [x] Flat run green (6.8e-12 pu drift).
- [x] Modal table archived (837 modes, 0 unstable; results/screening).
- [x] Step battery verdict vs the **20 s** window documented, **including
  the OLTC sequential-tap case**: PASS, worst case 13.2 s.

*Files*: `pf/screening.py` (purge fix, tap/AVR/shunt catalogue),
`pf/probe_tap_avr.py` (handle discovery + verification),
`results/screening/full_t0_wecc/20260720-143633/` (clean battery).
