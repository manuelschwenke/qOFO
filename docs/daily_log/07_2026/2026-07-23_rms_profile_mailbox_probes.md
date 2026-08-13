# RMS profile playback and live-mailbox probes

**Timestamp:** 2026-07-23 14:48 CEST  
**Reason:** Determine whether the per-element PowerFactory event list can be
replaced by (a) a Python-updated live DSL mailbox for online OFO commands and
(b) precomputed file-backed sources for known exogenous profiles.

## Scope and controller boundary

The controller implementation was not changed.  The TSO/DSO MIQPs continue to
receive only RMS measurements and cached sensitivities; they never receive the
RMS plant equations.  Controlled outputs remain TS/DS interface reactive-power
flows and bus voltages.  Actuators remain DER reactive-power references,
synchronous-machine AVR references, OLTCs, and shunts.

All probes used PowerFactory 2025 SP4 and the `02_RMS_CoSim` study case.  Each
probe substituted a temporary empty `IntEvt` and a temporary `ElmRes`, then
restored the original study-case pointers and deleted its temporary database
objects in `finally`.

## Authoritative model facts

The installed PowerFactory manuals were used:

- `Help/TechRef/TechRef_MeasurementFile.pdf`: `ElmFile` reads a time column and
  up to 24 selected data columns during RMS simulation, exposes `y1..y24`,
  supports constant or linear approximation, and holds its first/last value
  outside the file time range.
- `Help/TechRef/TechRef_GeneralLoad.pdf`: a three-phase dynamic `ElmLod` accepts
  `Pext` and `Qext` in MW/Mvar.  Its normal voltage/frequency dependence remains
  part of the physical load response.
- `Help/UserManual_en.pdf`, chapter 30: a `BlkSig` routed line needs source and
  destination slot endpoints, variable indices, and output/input connection
  types; slot declarations use the assigned models' native signal names.

The technical reference also states that an `ElmFile` source file remains in
use until the calculation is reset.  It is therefore a pre-initialisation
trajectory source, not a file that Python can rewrite online.

## Probe 1: direct live DSL parameter write

File: `pf/probe_live_dsl_parameter.py`

The probe changed `QVPRE.params:0` twice between `ComSim.Execute` calls in one
active calculation.

- Database writes were accepted and the paused object's `qset` read back as
  `0.00 -> 0.03 -> 0.06`.
- The same 0--2.5 s RMS time axis continued.
- Each edit changed the already-recorded `s:qset` result prefix by exactly
  `0.03 pu`; the final history contained the last value throughout.

**Finding:** direct parameter writes invalidate/re-evaluate prior dynamic
results.  They are not a causally valid online command channel.

## Probe 2: live IntMat mailbox

File: `pf/probe_live_intmat.py`

The probe changed all output ordinates of an active REEC_D `vdlp.IntMat` from
the original table to `0.40 pu`, then `0.30 pu`.

- Both database matrices read back exactly.
- The 0--2.5 s RMS result prefixes remained bitwise unchanged.
- The running DSL model retained its initial `vdlp=1.10 pu`.
- `Pord` remained `0.833063 pu` and the physical park remained
  `33.322521 MW`.

**Finding:** PowerFactory copies this IntMat into the dynamic model at
`ComInc`; later database edits are not visible to the active solver.  IntMat is
valid for preloaded data but not for a Python-updated live mailbox.

## Probe 3: event-free ElmFile RMS profile

Files:

- `pf/probe_rms_elmfile_profile.py`
- `pf/probe_data/live_elmfile_profile.txt`

The probe built the documented
`ElmFile.y1/y2 -> ElmLod.Pext/Qext` composite model around
`TN_load20_const_b38` and replayed three piecewise-constant profile levels:

| simulation time | file P factor | file Q factor | physical P factor | physical Q factor |
|---:|---:|---:|---:|---:|
| 0.25 s | 1.0 | 1.0 | 1.0000 | 1.0000 |
| 1.00 s | 1.2 | 0.8 | 1.2004 | 0.8005 |
| 2.00 s | 0.8 | 1.2 | 0.7995 | 1.1985 |

- The `ElmFile` output values matched their expected MW/Mvar values.
- The small physical-factor deviations are the intended ZIP
  voltage/frequency response, not a profile error.
- The RMS trace was monotone from 0 to 2.5 s with 252 rows.
- The isolated event folder remained empty.
- A separate post-probe check confirmed restoration of the original
  9,511-event folder, the normal `All calculations.ElmRes`, the original
  IntMat, and deletion of all temporary frame/composite/source objects.

**Finding:** known load trajectories can be transferred once before `ComInc`
and applied in RMS with zero simulation events.

## Architectural conclusion

The originally proposed architecture must be split:

1. **Supported and proven:** precompute known load and DER-active-power
   trajectories in Python, store them in one or more PowerFactory measurement
   files, and connect `ElmFile` outputs through composite frames.
2. **Rejected for PF 2025 SP4:** update IntMat or ordinary DSL parameters as a
   live mailbox between `ComSim.Execute` calls.
3. **Still required:** deliver online OFO outputs with one-shot simulation
   events, because their values depend on the preceding RMS measurement.

For the current profiles-on topology this removes approximately 53 `EvtLod`
and 44 `Pref_in` events per 20 s interval: 97 of about 177 interval events.
The remaining approximately 80 interval events are principally DSO DER
`qset/Vanchor` commands, plus slower TSO/discrete actions.

## Risks and next validation

- Production integration requires persistent, idempotently built composite
  sources for 53 loads and 44 WECC DER active-power inputs.  It should be
  introduced behind an explicit replay option until equivalence is measured.
- File timestamps must reproduce the existing dispatch convention: the
  profile for controller step `k` currently lands at
  `(k-1) * 20 s + 0.5 s`, not at `k * 20 s`.
- `ElmFile.approx=constant` is required to retain stepwise profile semantics.
- A 40 s profiles-on equivalence run should compare old-event and new-file
  terminal P/Q trajectories before a 900 s benchmark.
- Online event-pool construction still scales with the number of OFO command
  firings; eliminating profile events reduces but does not remove that cost.
