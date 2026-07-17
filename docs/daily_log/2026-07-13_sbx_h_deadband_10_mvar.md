# SBX-H reactive-flow deadband set to 10 Mvar

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Manuel selected a minimum settlement deadband of 10 Mvar;
the previous 5 Mvar placeholder was considered too small.

## Change

- Changed `SBXConfig.q_band_mvar` from 5 Mvar to 10 Mvar.
- Pinned experiment 014 to the same 10 Mvar value.
- Experiment 015 inherits 10 Mvar through the SBX-H default unless an
  explicit planning-derived band schedule is supplied.

## Settlement meaning

The change affects settlement volume only. Physical reactive flow is
not clipped. Under the current deductible rule, eligible signed support
is the relieving part of

`Q_sup = max(0, |Q_meas - Q_0| - 10 Mvar)`.

Voltage holding/violation roles and the requirement for a relieving Q
direction remain unchanged. The 10 Mvar value is a selected contract
parameter, not yet an empirically calibrated measurement/model-error
bound or a corridor capability.

## Verification planned

- Run the complete `tests/sbx_h` suite.
- Confirm experiment 014 constructs `SBXConfig(q_band_mvar=10.0)`.
