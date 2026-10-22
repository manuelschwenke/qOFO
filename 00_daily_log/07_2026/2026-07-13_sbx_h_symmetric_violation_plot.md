# SBX-H symmetric voltage violations and clearer live plot

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Manuel requested that terminal overvoltage be treated like
undervoltage for hold/violation settlement, and that experiment 014 make
the meaning of `Q_0`, `B_Q`, and signed `Q_sup` visually unambiguous.

## Architectural decision

The agreed terminal voltage is a two-sided schedule. For each side and
settlement window, with terminal error `e_i = V_meas,i - V_sched,i`:

- hold: `max_i |e_i| <= epsilon_hold`;
- violation (`a_sags` / `b_sags` compatibility fields): at least one
  `e_i < -epsilon_sag` or `e_i > +epsilon_sag`;
- transition: neither hold nor violation;
- a simultaneous under- and overvoltage on different terminals of the
  same corridor side is classified as `mixed` and is not assigned a
  remunerated Q direction.

Payment still requires exactly one violating side, a holding opposite
side, and beyond-deadband Q with the relieving sign. Undervoltage needs
Q toward the violating side; overvoltage needs Q away from it. The
violating side pays the holding side.

## Code changes

- `sbx_h/settlement.py`
  - symmetric hold and violation classification;
  - explicit `under`, `over`, and `mixed` violation kinds;
  - reversed relieving-flow direction for overvoltage;
  - min/max voltage errors and violation kinds added to the settlement
    ledger.
- `sbx_h/scheduler.py`
  - violation kinds propagated into cycle records for diagnostics and
    plotting.
- `sbx_h/adapter.py`
  - initial hold pre-check now uses the same symmetric absolute-error
    definition.
- `visualisation/plot_sbx.py`
  - absolute-flow overlay replaced by the directly interpretable
    residual `dQ = Q_meas - Q_0`;
  - the shaded/dashed `+/- B_Q` region is labelled as a no-payment
    deadband;
  - signed paid `Q_sup` is plotted on the same residual origin;
  - corridor titles report current `Q_meas`, `Q_0`, `B_Q`, and current
    measured/scheduled terminal voltages;
  - voltage traces use the terminal with the largest absolute schedule
    error, state strips say hold/violation/transition, repeated legends
    and repeated escalation lines were removed, and payments span one
    compact bottom row.
- `experiments/014_SBX_SINGLE_DEMO.py`, package and configuration
  documentation updated to use symmetric voltage-violation language.

## Verification

- `pytest tests/sbx_h -q`: **47 passed**.
- Added settlement tests for:
  - A-side overvoltage paid for relieving A-to-B Q;
  - B-side overvoltage paid for relieving B-to-A Q;
  - voltage above the hold tolerance but below the violation threshold
    is transition, not hold.
- 15-minute headless experiment-014 smoke run completed normally and
  generated the redesigned figure and settlement outputs.

## Open point

`B_Q = 10 Mvar` and the support-energy price remain placeholder contract
parameters. A large distance between `Q_meas` and `Q_0` is not itself a
plotting or model error: `Q_0` is the counterfactual flow at scheduled
terminal voltages and measured P. It does, however, reveal that the
realised boundary-voltage pattern is far from the agreed schedule and
therefore motivates calibration/prequalification of the schedule and
deadband before economic interpretation.
