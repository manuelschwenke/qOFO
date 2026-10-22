# 2026-07-19 — RMS Phase 3: Gate B resolved with auxiliary load buses

- **Timestamp:** 2026-07-19 01:03 CEST
- **Reason:** implement the user-selected auxiliary-bus workaround for the
  pandapower ZIP/static-generator aggregation defect, close Gate B, and add
  the missing PowerFactory station-controller write handles.

## Code and data changed

- `network/ieee39/meta.py`: added optional metadata for internal auxiliary
  buses, their physical parents, and their linking lines.
- `network/ieee39/scenarios/wind_replace.py`: moved loads co-located with TSO
  wind parks at buses 18 and 24 to two internal `TN_AUX` buses connected by
  explicit 1 km links with `r=x=0.01 ohm/km` and `max_i=10 kA`.
- `network/ieee39/helpers.py` and `network/ieee39/hv_networks.py`: propagated
  the new optional metadata through manual metadata reconstructions.
- `export/dynamic_snapshot.py`: made optional dataclass fields backward
  compatible when reading older snapshots.
- `export/make_snapshots.py` and
  `experiments/runners/multi_tso_dso.py`: assigned each auxiliary bus to its
  physical parent zone for dispatch while excluding it from the TSO voltage
  controlled-output set.
- `pf/naming.py`: added deterministic `AUX_TN_bus*` / `AUX_TN_line*` names.
- `pf/pf_sync.py`: created the auxiliary buses/links/load copies inside the
  wind variation, switched off the superseded physical-bus load copies, and
  added one operational constant-Q `ElmStactrl` per TSO wind park.
- `tests/test_wind_replace_aux_load_buses.py`: verified exact topology,
  metadata, monitoring exclusion, and constant-PQ wind injection in the ppc.
- Regenerated the wind-replacement and full t0/peak-residual reference
  snapshots; the base snapshots were not changed.

## Established results

The impedance sweep covered `r=x` from `1e-5` to `0.1 ohm` for both operating
points. `0.01 ohm` was selected: the near-ideal `1e-5 ohm` case failed at the
peak point, while the selected value retained a finite conditioning margin.

- maximum auxiliary-bus voltage drop: approximately `6.06e-5 pu`;
- maximum physical-TN voltage change relative to the near-ideal reference:
  approximately `2e-6 pu`;
- peak auxiliary-link losses: approximately `0.0379 MW` and `0.0379 Mvar`;
- estimated peak ppc condition number: approximately `1.27e6`.

PowerFactory Gate B with the station controllers active:

| snapshot | max `|d vm|` | max `|d va|` | result |
|---|---:|---:|---|
| t0 (`2016-01-05 08:00`) | `3.118e-8 pu` | `5.407e-6 deg` | PASS |
| peak residual (`2016-04-13 11:00`) | `2.861e-8 pu` | `1.951e-6 deg` | PASS |

All reported flow/injection deviations are below `2e-4 MW/Mvar`, far below
the informational `1 MW/Mvar` threshold. A second wind sync dry run reported
zero creates and zero updates.

The live PF 2025 SP4 probe established that `ElmStactrl.i_ctrl=1` and
`qu_char=0` implement constant-Q control. Changing `qsetp` by `+/-10 Mvar`
changed the connected wind-park injection by the same amount to float32
precision. The earlier `ElmGenstat.q_min/q_max=+/-sn` assignment was corrected
to `+/-1.0`, because these attributes are per-unit on `sgn`.

## Assumptions, constraints, actuators, and controlled outputs

- **Assumption:** the auxiliary links are internal numerical representation
  elements, not physical transmission assets.
- **Constraint:** their impedance is explicit and finite; no zero-impedance
  branch or closed switch is used in the pandapower oracle.
- **Actuators:** four TSO `ElmGenstat` parks with Q commands written through
  their attached `ElmStactrl.qsetp` handles. Automatic tap/shunt actions and
  reactive limits remain disabled for parity load flows.
- **Controlled outputs:** physical TN bus voltage magnitudes/angles and
  physical branch flows. The two auxiliary voltages are deliberately not
  controller outputs or voltage-constraint rows.

## Verification

- focused auxiliary-bus tests: `2 passed`;
- naming and snapshot round-trip tests: `29 passed, 2 skipped`;
- all modified Python modules compile successfully;
- all regenerated snapshots passed exact JSON round-trip verification;
- live PF wind sync is idempotent and both Gate B snapshots pass.

## Risks / unresolved points

- The workaround introduces two internal nodes and small, explicitly
  quantified losses. Any publication/model description must label them as a
  numerical separation device for ZIP/PQ semantics.
- Gate B is closed. The remaining uncertainty has moved to the Phase 4
  representation of multiple active PowerFactory variations; see the
  separate Phase 4 start log.
