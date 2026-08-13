# WECC DER model — one-park GUI build spec (hand back to Claude)

**Goal.** Build and verify **one** WECC composite dynamic model on a single
wind park in the PF GUI (where ComInc errors are visible), so Claude can
replicate it across the other parks by API copy + re-point + rescale.

Do this once, on one park. Everything after is scripted.

## Target

- **Park:** `WP_TSO_s0_b18` — a static generator (`ElmGenstat`) at
  `TN_bus18`, `sgn = 508 MVA`. (Any TSO wind park is fine; this one is
  representative.)
- **Study case:** `02_RMS_CoSim`, with variations **`wind_replace` + `full`**
  active and the four `DSO_*` grids active (Data Manager → each DSO grid →
  Activate). Load flow = the parity options (balanced; load voltage
  dependency ON; automatic taps/shunts/limits OFF).

## Build

Attach a composite dynamic model to the **existing** `WP_TSO_s0_b18`
(keep the generator — it is part of the validated parity model):

- **Frame:** `WECC Large-scale PV Plant`
  (`Library → Dynamics → IBR → WECC → Frm`).
  *Easiest route:* insert the ready template
  `Library → Templ → TemplPv → WECC Large-scale PV Plant 110MVA 60Hz`,
  then re-point its **Generator** slot to `WP_TSO_s0_b18`.
- **Slots:** Generator = `WP_TSO_s0_b18`; Gen-Con Model = `REGC_A`;
  Electrical Control = `REEC_A`; Plant Control = `REPC_A`; measurement
  slots as the template fills them.
- **Rating:** the template is 110 MVA — set the converter/plant MVA base to
  the park's `sgn = 508 MVA`. The pu control gains carry over unchanged
  (that is why one park is enough for Claude to rescale the rest).
- **Control mode:** set REEC/REPC to **local reactive-power control**
  (Q-priority, *not* voltage-droop) so the plant **Q-reference is an
  external setpoint**. That reference becomes the OFO write handle
  (replacing the load-flow-only `qsetp`).

## Verify (the go/no-go)

1. **Calculate Initial Conditions** (RMS, balanced) → must be **green** (no
   init errors). This is the whole point of doing it in the GUI.
2. Quick response check (optional): add a parameter event stepping the plant
   **Q-reference +60 Mvar at t = 5 s**, run 20 s → the park's Q and the
   `TN_bus18` voltage should move and settle (this is exactly what did *not*
   happen when stepping `qsetp`).

## Hand back — just tell Claude

1. "Built + ComInc green on `WP_TSO_s0_b18`."
2. The **composite model's name** (e.g. `WECC_WP_TSO_s0_b18`).

That's all. Claude then runs `python pf\wecc_introspect.py` to read the
frame, every block and parameter, and the Q-reference signal, and scripts
the rollout to the other 15 parks + the step-battery rebuild. No need to
type out parameters — the API reads them from your verified composite.
