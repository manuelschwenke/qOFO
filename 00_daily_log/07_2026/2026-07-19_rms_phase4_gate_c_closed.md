# 2026-07-19 - RMS Phase 4: layered full rebuild and Gate C closed

- **Timestamp:** 2026-07-19 19:58 CEST
- **Reason:** implement the user-approved layered PowerFactory architecture,
  extend the auxiliary-load-bus workaround to all four DSO underlays, delete
  the incomplete `full` artefact, rebuild it deterministically, and complete
  Phase 4 / Gate C.

## Outcome

Phase 4 is complete. The derived PowerFactory model now has the ordered state

1. template/base;
2. `wind_replace` variation (TN wind replacement and TN auxiliary load buses);
3. `full` variation on top (four DSO underlays and their internal auxiliary
   load buses).

Both required full snapshots pass the formal voltage, angle, flow, injection,
and 12-coupler interface gates. The PF project was returned to the designated
full t0 state, and a final dry run reported zero changes.

## Model and architecture revision

### Layered variations

`full` no longer duplicates the wind-replacement delta. The synchronizer and
parity runner activate the exact ordered stack `base -> wind_replace -> full`.
This respects PowerFactory's cross-variation name reservation and preserves
one deterministic `loc_name` per derived object. The first incomplete `full`
variation was deleted with exact-name/class guards after explicit user
approval, then rebuilt from the frozen snapshot.

### Auxiliary load-bus representation

The pandapower ZIP implementation aggregates a load and a fixed static
generator on the same internal ppc bus before applying the load-voltage
factor. Its public result tables, however, scale only the load. In the old
full snapshot this produced an internally inconsistent active-power balance
of `-15.7533066 MW`; the analytically predicted co-location term was
`-15.7533051 MW`.

The Phase-3 auxiliary-bus pattern was therefore generalized and applied to
the DSO layer. For every DSO, seven loads that share a physical parent with a
static generator are placed on seven `DN_AUX` buses. Each child is connected
to its parent by an explicit 1 km line with `r = x = 0.01 ohm/km` and
`max_i = 10 kA`. These are numerical separation elements, not physical grid
assets. A literal zero-impedance branch/switch was deliberately avoided to
retain a nonsingular, explicit pandapower admittance model.

Established full-snapshot effects:

- 28 DSO auxiliary buses and 28 links; no active load/static-generator bus
  overlap remains;
- maximum parent-to-child voltage drop about `2.95e-5 pu` over both full
  snapshots;
- maximum single-link active loss about `6.08e-4 MW`;
- public active-power balance residual reduced to `-3.02e-6 MW` at t0 and
  `-8.11e-6 MW` at peak residual.

Physical DSO buses remain first in the per-zone ordering; auxiliary children
are internal representation nodes and are not added as controller voltage
outputs.

### Three-winding transformer mapping

The persistent PF 2025 fields were verified as pairwise short-circuit data:

- `uktr3_h` / `pcut3_h`: HV-MV pair on a 300 MVA pair base;
- `uktr3_m` / `pcut3_m`: MV-LV pair on a 75 MVA pair base;
- `uktr3_l` / `pcut3_l`: LV-HV pair on a 75 MVA pair base.

Copper losses are written as

`Pcu_kW = vkr_percent / 100 * min(Sn_pair_MVA) * 1000`,

giving 900, 150, and 187.5 kW for the three specified pairs. The derived
`uktrr3_*`, `r1pu_*`, and `x1pu_*` values then match the snapshot conversion.

The last Gate-C mismatch was not a tap-ratio error: a reversible sweep showed
that `ElmTr3.m:t_h = 1 + n * 0.0125` exactly. It was a tap-placement enum
error. In PowerFactory 2025 SP4, `TypTr3.itapos = 1` means terminal-side tap
placement and `0` means star-point placement. Thus pandapower
`tap_at_star_point=False` maps to `itapos=1`. Reversible evidence:

| mapping | max abs dV [pu] | max abs angle [deg] | max non-bus deviation |
|---|---:|---:|---:|
| incorrect `itapos=0` | `6.532e-4` | `1.827e-2` | `3.859e-1` MW/Mvar |
| corrected `itapos=1` | `1.545e-5` | `4.059e-4` | `1.686e-2` MW/Mvar |

The correction is encoded in `pf/pf_sync.py` and pinned by
`tests/pf/test_pf_sync_trafo3w.py`.

## Code and data changed

- `network/ieee39/aux_load_buses.py`: shared finite-impedance auxiliary-bus
  construction and constants.
- `network/ieee39/scenarios/wind_replace.py`: delegates the TN workaround to
  the shared helper.
- `network/ieee39/hv_networks.py`: creates seven DSO auxiliary buses/links per
  underlay and moves only the co-located loads.
- `network/ieee39/meta.py`: generic and per-DSO auxiliary ownership metadata.
- `export/make_snapshots.py` and
  `experiments/runners/multi_tso_dso.py`: deterministic physical-parent zone
  ownership and physical-before-auxiliary bus ordering.
- `pf/naming.py`: deterministic `DN_AUX` names and corrected membership return
  annotation.
- `pf/session.py`, `pf/pf_sync.py`, and `pf/pf_parity.py`: layered variation
  activation, DSO-grid routing, full construction, 3W/shunt/controller sync,
  terminal-side tap placement, and full parity collection.
- `tests/test_full_dso_aux_load_buses.py` and
  `tests/pf/test_pf_sync_trafo3w.py`: topology, power-balance, and enum
  regressions.
- `export/snapshots/full_t0_20160105-0800.json` and
  `export/snapshots/full_peakres_20160413-0900.json`: regenerated full oracles
  after the explicit representation change. The old files were backed up at
  `C:/Users/mschwenke/.codex/visualizations/2026/07/18/019f76f0-b56c-7043-801b-8e2f03304d15/pre_dso_aux_full_snapshots`.

## Live PowerFactory rebuild evidence

The incomplete `full` variation, its exact DSO grids, 44 DSO line types, and
12 three-winding types were removed with verified targets. The subsequent
full rebuild created 728 objects. Twelve disconnected cubicle shells left by
the earlier interrupted wind-layer attempt were separately verified to have
no object pointer, deleted, and their connected replacements renamed to the
canonical names.

Final structural audit:

- calculation-relevant grids: template Grid plus `DSO_1` through `DSO_4`;
- per DSO: 20 `ElmTerm`, 18 `ElmLne`, 20 `ElmLod`, 10 `ElmGenstat`,
  10 `ElmStactrl`, 3 `ElmTr3`, 2 `ElmShnt`, and 74 `StaCubic`;
- project types: 72 DSO `TypLne` (including 28 auxiliary-link types) and 12
  `TypTr3`;
- no decorated duplicate names and complete deterministic snapshot lookup;
- immediate second sync: zero creates/updates/deletes.

The deletion is not recoverable through PF undo after the engine process
closes, but all deleted objects are deterministic derived artefacts and were
successfully regenerated from the backed-up/frozen snapshot inputs.

## Gate C evidence

Formal defaults: `1e-4 pu` voltage magnitude, `0.01 deg` voltage angle, and
`1 MW/Mvar` per flow or injection. Automatic taps, shunts, and limits were
off; discrete positions and continuous setpoints came from each snapshot.

| full snapshot | max abs dV [pu] | max abs angle [deg] | max 3W P/Q deviation | verdict |
|---|---:|---:|---:|---|
| t0, 2016-01-05 08:00 | `1.545e-5` | `4.059e-4` | `2.673e-3` MW/Mvar | PASS |
| peak residual, 2016-04-13 09:00 | `1.609e-5` | `5.377e-4` | `2.579e-3` MW / `1.424e-3` Mvar | PASS |

The `--interfaces` report explicitly covered HV/MV P and Q for all 12
couplers at both points. Every interface quantity is more than two orders of
magnitude inside the 1 MW/Mvar gate.

## Verification

- focused network/export/PF suite: `66 passed, 2 skipped`;
- additional focused correction run: `5 passed`;
- exact JSON round-trip verification for both regenerated full snapshots;
- full t0 and peak-residual live PF parity: PASS;
- final restored t0 full sync dry run: created 0, updated 0, deleted 0.

## Assumptions, constraints, actuators, and controlled outputs

- **Established model facts:** the pandapower snapshot is the source of truth;
  PF is a generated artefact; `itapos=1` is required for terminal-side taps in
  the installed PF 2025 SP4 release; auxiliary links are explicitly finite.
- **Assumptions:** the quantified auxiliary-link drop/loss is negligible for
  the intended hierarchical-control study and is preferable to corrupting
  ZIP/fixed-injection semantics.
- **Constraints:** variations must be activated in order; automatic LDF tap
  and shunt action remains disabled for parity; all future writes address
  canonical names and cached model/controller mappings, not GUI positions.
- **Actuators represented:** 12 DSO-coupler HV OLTCs, 8 tertiary MSC/MSR
  banks, 44 static-generator Q controllers, retained synchronous-machine AVR
  setpoints, and the existing TN OLTCs.
- **Controlled outputs:** physical TN/DSO bus voltages, physical branch
  currents/flows, all coupler HV/MV interface P/Q variables (`q_STS`), and
  source/load injections. Auxiliary child-bus voltages are diagnostic only.

## Risks and unresolved points

- Phase 4 validates steady-state translation only. Dynamic models, RMS
  initialization drift, modal damping, event semantics, and 10 s controller
  exchange are Phase 5/6 work and remain open.
- The `itapos` integer meaning is release-specific empirical/API-schema
  knowledge; the code comment and unit regression prevent silent reversal on
  this installation. Re-verify after a PF version upgrade.
- Publications must describe the auxiliary buses as numerical representation
  nodes and disclose their finite impedance and bounded losses.
- No PowerFactory manual extract is needed to close Phase 4; the installed API
  schema, solved-ratio sweep, controlled enum experiment, and two parity
  operating points provide direct evidence. The manual may still be useful
  when Phase 5 reaches dynamic block/model parameterization.
