# PowerFactory naming convention (`loc_name`)

Executable form: `pf/naming.py` (`build_name_map(snapshot_doc)`), proven
total + collision-free on the reference snapshots by `tests/pf/test_naming.py`.
The sync script's find-or-create logic and the parity/plant lookups depend on
these names — **never** rename script-owned objects in the GUI.

Principles

1. Every script-owned object embeds its **pandapower index** (the snapshot is
   keyed identically, so lookups never guess).
2. Template-owned objects (the ten `ElmSym` machines) are **not** renamed;
   they are addressed via `TEMPLATE_MACHINE_NAMES` (pandapower label →
   template `loc_name`, verified once in the Phase-1 hello-world).
3. Charset `[A-Za-z0-9_]` only.

## Script-owned objects

| Element (pandapower) | PF class | `loc_name` pattern | Example |
|---|---|---|---|
| 345 kV TN bus | `ElmTerm` | `TN_bus{i}` | `TN_bus15` |
| 10.5 kV machine terminal | `ElmTerm` | `GT_bus{i}` | `GT_bus29` |
| 110 kV HV bus | `ElmTerm` | `{net_id}_bus{i}` | `DSO_1_bus45` |
| 20 kV tertiary bus | `ElmTerm` | `{net_id}_tert{i}` | `DSO_1_tert48` |
| TN line | `ElmLne` | `TN_line{i}` | `TN_line7` |
| HV line | `ElmLne` | `{net_id}_line{i}` | `DSO_2_line60` |
| Machine 2W trafo (OLTC) | `ElmTr2` | `MT_g{gen}_t{i}` | `MT_g1_t2` |
| Network 2W OLTC (bus 12 pair) | `ElmTr2` | `NT_t{i}` | `NT_t3` |
| 3W coupler (345/110/20) | `ElmTr3` | `NC3W_{net_id}_t{i}` | `NC3W_DSO_1_t0` |
| Load (const half) | `ElmLod` | `{TN\|net_id}_load{i}_const_b{bus}` | `TN_load4_const_b3` |
| Load (profile half) | `ElmLod` | `{TN\|net_id}_load{i}_var_b{bus}` | `DSO_3_load88_var_b71` |
| TSO wind park (STATCOM) | `ElmGenstat` | `WP_TSO_s{i}_b{bus}` | `WP_TSO_s0_b4` |
| DSO coupling-bus WP | `ElmGenstat` | `WPC_{net_id}_s{i}_b{bus}` | `WPC_DSO_1_s11_b47` |
| DSO DER (wind/PV) | `ElmGenstat` | `DER_{net_id}_s{i}_b{bus}` | `DER_DSO_4_s38_b80` |
| Q controller of any sgen | `ElmStactrl` | `CTRL_{sgen_loc_name}` | `CTRL_WP_TSO_s0_b4` |
| TSO tertiary shunt | `ElmShnt` | `SH_{MSC\|MSR\|BIPOLAR}_{net_id}_s{i}` | `SH_MSC_DSO_1_s0` |

Notes

- `{i}` / `{bus}` / `{gen}` are pandapower indices from the snapshot; `net_id`
  is the sub-network id (`DSO_1` … `DSO_4`) resolved via `meta.hv_networks`.
- Bus classification: tertiary/HV membership from `meta.hv_networks`;
  `GT_bus` for `subnet == "GEN_TERM"` **or** `subnet == "TN"` with
  `vn_kv < 100` (the pre-existing case39 machine terminals keep subnet TN at
  10.5 kV); remaining TN buses are `TN_bus`.
- Anything unclassifiable raises `NamingError` at map-build time (Fail-Fast),
  so a schema drift can never produce silently mis-named objects.

## Template-owned objects

| pandapower gen label | template `ElmSym` (⚠ placeholder, verify in Gate 1) |
|---|---|
| `G1` (slack anchor, bus 39) | `G 01` |
| `G2` … `G9` | `G 02` … `G 09` |
| `G10` (Hydro, bus 30) | `G 10` |

- `pf/hello_pf.py` prints the actual machine names; correct
  `TEMPLATE_MACHINE_NAMES` in `pf/naming.py` and set
  `TEMPLATE_NAMES_VERIFIED = True` afterwards (pf_sync refuses to run
  before that).
- Machine step-up trafos that already exist in the template are *adopted*
  (renamed to `MT_g{gen}_t{i}`) by the sync script on first run; G1's
  step-up does not exist in the template and is created outright (the
  builder collapses the bus 19→20→34 chain and creates G1's terminal —
  see the build-plan appendix).
