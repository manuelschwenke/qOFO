# 2026-07-21 (late) — Time-series profiles wired into the PowerFactory plant

**Context.** The RMS co-simulation could not apply profiles: the runner
raised `NotImplementedError("time-series profiles (PF-side load events not
wired yet)")` for any non-static plant, so Gate E ran with
`cfg.use_profiles = False`.  That is not a cosmetic gap — it pins every DER
park at `P = S_n` (all 44 parks at `P/S_n = 1.0`), which is what made the
circular STATCOM capability diagram collapse to zero earlier the same day,
and it means primary control is never exercised.

## Mechanisms found (both verified live)

PowerFactory reads element **input data only at initialisation**, so neither
`ElmLod.plini` nor `ElmGenstat.pgini` can be written mid-RMS — the same
reason taps need `EvtTap`.  Two different event types are required:

| quantity | mechanism | semantics |
|---|---|---|
| load P/Q (53 of 113 loads) | `EvtLod` | `iopt_type=0` "Incremental Change", `dP`/`dQ` in **percent of present value** |
| DER active power (44 parks) | `EvtParam` on `WTGWGO_A.Pref_in` | **absolute, pu of park rating** |

Verification:

* `EvtLod` +50 % on `TN_load0_const_b0`: `m:P:bus1` 38.129 → 57.181 MW at the
  event instant; PF logged `Load Event: 'Incremental Change' - Active Power
  changed by 50,000 %`.
* `EvtParam` `Pref_in = 0.5` on `WECC_WP_TSO_s0_b18`: park P 508.000 →
  254.088 MW = exactly 0.5 pu of the 508 MVA rating (≈2 s converter ramp).

**The 2026-07-20 finding "EvtLod does NOT fire (silently ignored)" was
wrong.**  That probe left `p_target` unset, so the event had no object to act
on and PF did nothing.  It was recorded as a dead end only because the probe
never read the output window.  **Always set `p_target`; always read
`GetOutputWindow()` before concluding an event mechanism does not work.**

The WECC active-power chain is `WTGWGO_A(Pref_in) → Pref_out →
REEC_D(Pref)`; `Pref_in` is the free end.

## Implementation

* `core/plant.py`: `Plant.apply_exogenous(profiles, t)` added to the
  protocol.  `PandapowerStaticPlant` delegates to
  `core.profiles.apply_profiles` — bit-identical to the previous direct
  `apply_profiles(net, …)` call.
* `pf/screening.py`: `ScreeningContext.add_load_event()` (`EvtLod`), with the
  `p_target` landmine documented at the call site.
* `pf/plant.py`: `_loads` and `_wgo` handle tables; `apply_exogenous()` scales
  the mirror net first, then converts each load's absolute factor change into
  the incremental percentage `EvtLod` needs, and writes each park's `Pref_in`
  absolutely.  `_percent_delta()` returns 0 for a load starting at ~0, since a
  percentage cannot express a step away from zero.
* `experiments/runners/multi_tso_dso.py`: the loop calls
  `plant.apply_exogenous(...)` instead of writing `net`; profiles removed from
  the non-static `NotImplementedError` list.

## Still unsupported: synchronous-machine dispatch

`use_zonal_gen_dispatch` writes `net.gen.p_mw` directly (`apply_gen_dispatch`)
and is now the remaining entry in the non-static guard.  Driving machine P in
RMS needs a **third** mechanism — an `EvtParam` on the governor's power
reference.  Note there is a design question attached: forcing a machine
schedule partly bypasses primary control, whereas letting the governors absorb
the load variation is arguably more physical for an RMS study, and would
finally exercise the pandapower-distributed-slack vs RMS-governor asymmetry.

## Status

End-to-end run with `use_profiles=True` and `use_zonal_gen_dispatch=False`
completes against the PF plant with no error, and the mirror net follows the
profile.  **PF-side following is not yet conclusively demonstrated**: over 60 s
the profile moves only ~0.09 %, and the PF/mirror ratio is dominated by the
per-bus ZIP voltage factor, so the measurement cannot separate "followed" from
"ignored" at that amplitude.  The individual event mechanisms are proven
exactly (above); what remains is an end-to-end validation at a *steep* profile
segment or over a longer horizon, plus a parity check at a profile-scaled
operating point to confirm ZIP anchoring stays consistent between the plants.
