# Reserve trigger margin derived, and why the old proxy never fired

Timestamp: 2026-10-21, 09:40 CEST.

Reason: the `sts_reserve` memory has never become non-zero in any recorded run.
The author decided to report the binding unit rather than a fleet mean, to
extend the upward report with directional DER margins in Mvar when the
interface is changed, to start with the variant that needs no interface change,
and to derive the threshold from the recordings.

Changes: none to code. Added
`docs/tuning/2026-10-21_reserve_trigger_margin_derivation.md` with the
derivation, the distribution and the limits.

Method: the one-sided interface margins are already recorded, as
`dso_trafo_q_cap_min_mvar` and `dso_trafo_q_cap_max_mvar` against
`dso_trafo_q_set_mvar`, so the threshold could be derived offline from the
held-bank baseline of the 20 October hour without a new run. Margins are
expressed in bank steps, a scale the actuator defines, rather than as a
normalised position in the band.

Findings:

- Over 21 TS decision instants and 12 interfaces, the binding margin has
  minimum 0.725, p5 1.023, median 1.845 and maximum 2.392 bank steps.
- **`k = 1.0` is proposed.** It fires on 9 of 252 TS-interface-samples on the
  two tightest interfaces, enough to confirm a 360 s persistence and far from
  permanent, and it reads physically as "less than one full bank step of
  headroom left in the binding direction". `k ≤ 0.5` never fires; `k ≥ 1.25`
  makes DSO_4 demand a step at 21 of 21 instants, which is the ratchet failure
  mode already diagnosed for the regular path.
- `margin_up` is the binding direction at every one of the 252 samples, so the
  one-sided form is necessary rather than a refinement.
- The old proxy needed a margin below about 0.28 bank steps (7 Mvar) to reach
  `r > 0.8`, while the margin never fell below 0.725 steps (18.1 Mvar). That is
  the quantitative reason it never fired, and it exposes the structural defect
  of the normalised form: its threshold in Mvar moves with the band width.
- **Neither reference convention relieves the DER**, and they differ, so an
  earlier reading of the margin table as "engagement tightens the margin" was
  too coarse. At an objective commit the dispatched setpoints move with the
  bank (DSO_1 `trafo_1` −6.6 → −26.6 Mvar) and the band shifts with the
  measured flow, so the margin is nearly unchanged and the DER stay where they
  were: the offset convention tells the subordinate controller not to fight the
  bank. At a tracking support commit the dispatched setpoint is invariant by
  construction (`trafo_10` stays −4.0) while the measured flow jumps +7.8 →
  −16.1, so the subordinate controller **works against the bank** — the flow
  recovers to −13.8 over six minutes while aggregate DER Q moves −47.9 →
  −50.9, and that interface's `margin_up` falls from 51.7 to 28.5 Mvar.
  Asymptotically the subordinate loop cancels the bank, having spent DER
  headroom to do it. Reserve restoration through bank switching is therefore
  not demonstrated, and its mechanism is not yet established.
- The per-DER maximum is dropped as a trigger, on the author's objection: the
  reserve that matters is interface-referred and deliverable, which requires
  the sensitivity weighting the reported band already has. Option B is
  therefore not a trigger and option C as written is not the right quantity;
  option A is the correct signal and needs no interface extension. The per-DER
  distribution retains one use, qualifying the linear band's validity as units
  saturate, which is not a trigger.

Open: the −0.037 excursion at `g_w = 5` means the dispatched setpoint fell
marginally outside the reported band, about 0.9 Mvar. The paired
capability-shift and `q_adj` logic exists to prevent that, so it is either
measurement noise and voltage rescaling or a residual in that pairing, and it
should be checked before the reserve path treats the band as a bound. The
derivation is one hour of one scenario and should be repeated over campaign
windows as `tuning/shunt_weight_calibration.py` does for the movement weight.
