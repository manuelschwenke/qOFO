# SBX-H demo live plot starts at time zero

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** `014_SBX_SINGLE_DEMO.py` opened its figure immediately but kept the
SBX-H plot in the uninitialized placeholder state until minute 30.

## Established cause

The plotter itself was constructed before the simulation loop. However, the
runner constructs the SBX-H adapter only at the first TSO tick satisfying
`time_s >= config.sbx_warmup_s`. Experiment 014 set this value to 1800 s, so no
corridor registry, scheduled references, terminal history, or reactive-flow
baseline was available to the plotter during minutes 0--30.

## Revision

- Experiment 014 now sets `sbx_warmup_s = 0.0` through
  `SBX_START_MIN = 0.0`.
- The contracts are initialized from the already-converged initial operating
  point at the first TSO tick.
- The SBX mechanism plot therefore builds and begins recording at minute 0.
- Scenario text and the generated experiment summary now call this contract
  initialization rather than a controller/plant warm-up.
- A regression test guards the zero-start configuration.

The generic runner warm-up option is unchanged. Other experiments can still
delay contract initialization when they explicitly need a closed-loop settling
interval.

## Assumptions and controlled quantities

- The initial network power flow supplied by the shared 005 configuration is
  converged and is accepted as the scheduled-contract snapshot.
- Actuators, MIQP constraints, cached sensitivities, scheduled terminal-voltage
  control, hold/sag classification and settlement are unchanged.
- The reactive disturbance remains scheduled for minute 60 in the default
  300-minute run.
