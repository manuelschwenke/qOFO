"""
export/dynamic_snapshot.py
==========================
Dump the fully built and profile-scaled IEEE 39-bus network (plus optional
110 kV underlays) together with its converged power-flow solution into one
self-contained JSON snapshot, and rebuild a runnable pandapower network from
such a snapshot.

Role in the RMS build plan (Phase 0)
------------------------------------
The snapshot is the single hand-over artefact between the pandapower builder
(single source of truth) and the PowerFactory sync script (``pf/pf_sync.py``,
Phase 2 of ``docs/RMS_IEEE39_PowerFactory_Build_Plan.md``):

* ``model``    -- every electrically relevant parameter of every element,
                  keyed by its pandapower index.
* ``solution`` -- the converged power-flow state (bus voltages, branch
                  flows, machine/DER reactive power).  This is the parity
                  target for Gates A-C and the RMS initialisation oracle.
* ``meta``     -- the :class:`~network.ieee39.meta.IEEE39NetworkMeta` index
                  catalogue, the control-zone map, the removed-generator
                  list and a flat OFO actuator inventory.

Completeness contract (Fail-Fast)
---------------------------------
Every element table is serialised from an *explicit* field list.  Columns
known to be irrelevant to the load flow and the RMS build (geodata, OPF
limits, short-circuit data) are listed in per-table ignore sets; columns
that may exist but must be *unused* (e.g. a second tap changer) are asserted
to hold only their unused sentinel.  Any column that falls in none of these
categories raises :class:`SnapshotSchemaError` at dump time -- the schema
must be extended deliberately, never silently.

The round-trip test (``tests/export/test_dynamic_snapshot_roundtrip.py``)
proves the field lists are complete: a network rebuilt from the JSON alone
must reproduce the stored power-flow solution to 1e-8 pu.

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import MISSING, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pandapower as pp

from network.ieee39.constants import GEN_NAMEPLATE
from network.ieee39.meta import HVNetworkInfo, IEEE39NetworkMeta

SCHEMA_VERSION = 1

# Sentinel for "column may exist but every entry must be NaN/None".
_ALL_NAN = object()


# =====================================================================
#  Errors
# =====================================================================

class SnapshotError(RuntimeError):
    """Base class for snapshot dump/load failures."""


class SnapshotSchemaError(SnapshotError):
    """A table carries columns the schema does not know about."""


class SnapshotValueError(SnapshotError):
    """A required field is missing or NaN, or a reference dangles."""


# =====================================================================
#  Per-table field schema
# =====================================================================

@dataclass(frozen=True)
class TableSchema:
    """Explicit serialisation contract for one pandapower element table.

    ``required``      -- serialised; NaN/missing raises.
    ``nullable``      -- serialised; NaN becomes JSON ``null``.  Columns in
                         this set that do not exist in the table are simply
                         omitted from the records.
    ``ignored``       -- known-irrelevant to the load flow and the RMS
                         build; dropped without inspection.
    ``assert_unused`` -- column -> unused sentinel.  ``_ALL_NAN`` demands
                         every entry be NaN/None; any other value demands
                         every entry equal that value.  Violations raise.
    """
    required: Tuple[str, ...]
    nullable: Tuple[str, ...] = ()
    ignored: Tuple[str, ...] = ()
    assert_unused: Mapping[str, Any] = field(default_factory=dict)


#: Order matters for the loader: buses first, then branches, then injections.
TABLE_SCHEMAS: Dict[str, TableSchema] = {
    "bus": TableSchema(
        required=("vn_kv", "in_service"),
        nullable=("name", "type", "subnet"),
        # 'zone' is pandapower's free-text zone tag (unused here -- the
        # control zone comes from zone_map); geodata and OPF voltage limits
        # do not enter the load flow.
        ignored=("zone", "geo", "min_vm_pu", "max_vm_pu"),
    ),
    "line": TableSchema(
        required=("from_bus", "to_bus", "length_km",
                  "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km",
                  "g_us_per_km", "max_i_ka", "df", "parallel", "in_service"),
        nullable=("name", "std_type", "type", "subnet"),
        ignored=("geo", "max_loading_percent"),
    ),
    "trafo": TableSchema(
        required=("hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv",
                  "vk_percent", "vkr_percent", "pfe_kw", "i0_percent",
                  "shift_degree", "df", "parallel", "in_service"),
        nullable=("name", "std_type", "tap_side", "tap_neutral", "tap_min",
                  "tap_max", "tap_pos", "tap_step_percent",
                  "tap_step_degree", "tap_changer_type"),
        ignored=("max_loading_percent",),
        # A second tap changer or tap-dependency tables would change the
        # model -- the builder never uses them, so they must stay unused.
        assert_unused={
            "tap2_side": _ALL_NAN, "tap2_neutral": _ALL_NAN,
            "tap2_min": _ALL_NAN, "tap2_max": _ALL_NAN,
            "tap2_pos": _ALL_NAN, "tap2_step_percent": _ALL_NAN,
            "tap2_step_degree": _ALL_NAN, "tap2_changer_type": _ALL_NAN,
            "leakage_resistance_ratio_hv": _ALL_NAN,
            "leakage_reactance_ratio_hv": _ALL_NAN,
            "id_characteristic_table": _ALL_NAN,
            "tap_dependency_table": False,
            "xn_ohm": _ALL_NAN,
            "oltc": False,
            "power_station_unit": _ALL_NAN,
        },
    ),
    "trafo3w": TableSchema(
        required=("hv_bus", "mv_bus", "lv_bus",
                  "sn_hv_mva", "sn_mv_mva", "sn_lv_mva",
                  "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                  "vk_hv_percent", "vk_mv_percent", "vk_lv_percent",
                  "vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent",
                  "pfe_kw", "i0_percent", "shift_mv_degree",
                  "shift_lv_degree", "in_service"),
        nullable=("name", "std_type", "tap_side", "tap_neutral", "tap_min",
                  "tap_max", "tap_pos", "tap_step_percent",
                  "tap_step_degree", "tap_changer_type", "tap_at_star_point"),
        ignored=("max_loading_percent",),
        assert_unused={
            "id_characteristic_table": _ALL_NAN,
            "tap_dependency_table": False,
            # Zero-sequence data only enters asymmetric power flow /
            # short-circuit; the builder never sets it.
            "vk0_hv_percent": _ALL_NAN, "vk0_mv_percent": _ALL_NAN,
            "vk0_lv_percent": _ALL_NAN,
            "vkr0_hv_percent": _ALL_NAN, "vkr0_mv_percent": _ALL_NAN,
            "vkr0_lv_percent": _ALL_NAN,
            "vector_group": _ALL_NAN,
        },
    ),
    "load": TableSchema(
        # pandapower >= 3.x carries separate ZIP shares for P and Q; all
        # four are 0.0 in this project (constant-power oracle convention).
        required=("bus", "p_mw", "q_mvar",
                  "const_z_p_percent", "const_z_q_percent",
                  "const_i_p_percent", "const_i_q_percent",
                  "scaling", "in_service"),
        nullable=("name", "sn_mva", "type", "subnet",
                  "profile_p", "profile_q", "base_p_mw", "base_q_mvar",
                  # Anchored-ZIP bookkeeping (network/ieee39/load_model.py):
                  # the voltage at which the profile powers are served.
                  "zip_anchor_vm_pu"),
        # OPF dispatch limits and the controllable flag do not enter runpp.
        ignored=("controllable", "min_p_mw", "max_p_mw",
                 "min_q_mvar", "max_q_mvar"),
    ),
    "sgen": TableSchema(
        required=("bus", "p_mw", "q_mvar", "sn_mva", "scaling", "in_service"),
        nullable=("name", "type", "subnet", "profile", "op_diagram",
                  "base_p_mw",
                  # Q(V) droop layer (tag_der_q_modes) -- needed by the
                  # PowerFactory ElmStactrl / DSL droop replication.
                  "q_mode", "qv_slope_pu", "qv_vref_pu", "qv_deadband_pu",
                  "cosphi", "cosphi_sign", "q_set_mvar", "qv_vref_anchor_pu"),
        # current_source & friends parameterise short-circuit calculations
        # only; OPF columns do not enter runpp.
        ignored=("controllable", "current_source", "generator_type",
                 "k", "rx", "lrc_pu", "max_ik_ka", "kappa",
                 "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"),
        # An active reactive-capability curve would change the Q limits the
        # snapshot reports -- must stay unused.
        assert_unused={
            "reactive_capability_curve": False,
            "curve_style": _ALL_NAN,
            "id_q_capability_characteristic": _ALL_NAN,
        },
    ),
    "gen": TableSchema(
        required=("bus", "p_mw", "vm_pu", "sn_mva",
                  "min_q_mvar", "max_q_mvar", "min_p_mw", "max_p_mw",
                  "scaling", "slack", "slack_weight", "in_service"),
        nullable=("name", "type", "profile", "base_p_mw"),
        # Short-circuit machine data and OPF flags are irrelevant to runpp.
        ignored=("controllable", "vn_kv", "xdss_pu", "rdss_ohm", "rdss_pu",
                 "cos_phi", "pg_percent", "power_station_trafo"),
        # An active reactive-capability curve would change the Q limits
        # that enforce_q_lims applies -- must stay unused.
        assert_unused={
            "reactive_capability_curve": False,
            "curve_style": _ALL_NAN,
            "id_q_capability_characteristic": _ALL_NAN,
        },
    ),
    "shunt": TableSchema(
        required=("bus", "q_mvar", "p_mw", "vn_kv", "step", "max_step",
                  "in_service"),
        nullable=("name",),
        assert_unused={
            "step_dependency_table": False,
            "id_characteristic_table": _ALL_NAN,
        },
    ),
}

#: Tables that must be empty -- the schema does not cover them, and the
#: IEEE 39 builder never creates them.  ``poly_cost`` / ``pwl_cost`` are
#: deliberately absent from this list: case39() ships OPF cost data that
#: does not affect runpp; it is not serialised.
_MUST_BE_EMPTY = (
    "ext_grid", "switch", "motor", "asymmetric_load", "asymmetric_sgen",
    "storage", "ward", "xward", "impedance", "dcline", "svc", "tcsc",
    "ssc", "vsc", "measurement", "characteristic",
)

#: Solution (result-table) fields serialised per element table.
_SOLUTION_FIELDS: Dict[str, Tuple[str, ...]] = {
    "bus": ("vm_pu", "va_degree", "p_mw", "q_mvar"),
    "line": ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar",
             "i_from_ka", "i_to_ka"),
    "trafo": ("p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar",
              "i_hv_ka", "i_lv_ka"),
    "trafo3w": ("p_hv_mw", "q_hv_mvar", "p_mv_mw", "q_mv_mvar",
                "p_lv_mw", "q_lv_mvar"),
    "load": ("p_mw", "q_mvar"),
    "sgen": ("p_mw", "q_mvar"),
    "gen": ("p_mw", "q_mvar", "vm_pu", "va_degree"),
    "shunt": ("p_mw", "q_mvar", "vm_pu"),
}

#: Keys every ``solver_options`` dict must define so a stored solution is
#: reproducible.  ``run_control`` must be False: snapshots are dumped with
#: no controllers attached (the dump refuses otherwise).
_SOLVER_OPTION_KEYS = ("run_control", "calculate_voltage_angles", "init",
                       "max_iteration", "distributed_slack",
                       "enforce_q_lims")


# =====================================================================
#  JSON conversion helpers
# =====================================================================

def _is_nanlike(value: Any) -> bool:
    """True for None / NaN / pandas NA scalars."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_jsonable(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a plain-Python JSON value."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, str):
        return value
    raise SnapshotValueError(
        f"Cannot serialise value {value!r} of type {type(value).__name__}"
    )


def _cell(df: pd.DataFrame, idx: Any, col: str, *, table: str,
          nullable: bool) -> Any:
    """Read one cell with Fail-Fast NaN policy."""
    value = df.at[idx, col]
    if _is_nanlike(value):
        if nullable:
            return None
        raise SnapshotValueError(
            f"{table}[{idx}].{col} is NaN/None but the field is required"
        )
    return _to_jsonable(value)


def _check_table_schema(df: pd.DataFrame, table: str,
                        schema: TableSchema,
                        problems: List[str]) -> None:
    """Collect schema violations for one table (aggregated raise later)."""
    cols = set(df.columns)

    for col in schema.required:
        if col not in cols:
            problems.append(f"{table}: required column '{col}' is missing")

    known = (set(schema.required) | set(schema.nullable)
             | set(schema.ignored) | set(schema.assert_unused))
    unknown = sorted(cols - known)
    if unknown:
        problems.append(
            f"{table}: unknown column(s) {unknown} -- extend the schema in "
            f"export/dynamic_snapshot.py deliberately (serialise, ignore, "
            f"or assert-unused)"
        )

    if len(df) == 0:
        return
    for col, sentinel in schema.assert_unused.items():
        if col not in cols:
            continue
        series = df[col]
        if sentinel is _ALL_NAN:
            bad = series[~series.isna()]
        else:
            bad = series[~(series.isna() | (series == sentinel))]
        if len(bad):
            problems.append(
                f"{table}.{col}: expected unused "
                f"({'all NaN' if sentinel is _ALL_NAN else sentinel!r}) but "
                f"found values {bad.to_dict()}"
            )


def _dump_table(df: pd.DataFrame, table: str,
                schema: TableSchema) -> Dict[str, Dict[str, Any]]:
    """Serialise one element table keyed by stringified pandapower index."""
    # Only nullable columns that actually exist are emitted, so optional
    # builder columns (e.g. gen 'profile') never appear as fabricated nulls.
    nullable_present = [c for c in schema.nullable if c in df.columns]
    records: Dict[str, Dict[str, Any]] = {}
    for idx in df.index:
        rec: Dict[str, Any] = {}
        for col in schema.required:
            rec[col] = _cell(df, idx, col, table=table, nullable=False)
        for col in nullable_present:
            rec[col] = _cell(df, idx, col, table=table, nullable=True)
        records[str(int(idx))] = rec
    return records


def _dump_solution_table(res: pd.DataFrame, table: str,
                         fields: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for idx in res.index:
        rec = {}
        for col in fields:
            if col not in res.columns:
                raise SnapshotValueError(
                    f"res_{table} lacks column '{col}'"
                )
            rec[col] = _cell(res, idx, col, table=f"res_{table}",
                             nullable=False)
        records[str(int(idx))] = rec
    return records


# =====================================================================
#  Meta / zone / inventory blocks
# =====================================================================

def _meta_to_dict(meta: IEEE39NetworkMeta) -> Dict[str, Any]:
    """Serialise the meta dataclass (tuples -> lists, HVNetworkInfo -> dict)."""
    out: Dict[str, Any] = {}
    for name in meta.__dataclass_fields__:
        value = getattr(meta, name)
        if name == "hv_networks":
            out[name] = [
                {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in hv.__dict__.items()}
                for hv in value
            ]
        elif isinstance(value, tuple):
            out[name] = list(value)
        else:
            out[name] = value
    return out


def meta_from_dict(data: Mapping[str, Any]) -> IEEE39NetworkMeta:
    """Rebuild the meta dataclass from its JSON form."""
    kwargs: Dict[str, Any] = {}
    for name in IEEE39NetworkMeta.__dataclass_fields__:
        if name not in data:
            field_def = IEEE39NetworkMeta.__dataclass_fields__[name]
            if field_def.default_factory is not MISSING:
                kwargs[name] = field_def.default_factory()
                continue
            if field_def.default is not MISSING:
                kwargs[name] = field_def.default
                continue
            raise SnapshotValueError(f"meta block lacks field '{name}'")
        value = data[name]
        if name == "hv_networks":
            kwargs[name] = tuple(
                HVNetworkInfo(**{
                    k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in hv.items()
                })
                for hv in value
            )
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return IEEE39NetworkMeta(**kwargs)


_GEN_NAME_RE = re.compile(r"^(G\d+)_bus(\d+)$")


def _removed_generators(net: pp.pandapowerNet) -> List[Dict[str, Any]]:
    """Machines of the 10-unit IEEE 39 fleet absent from ``net.gen``.

    The nameplate loop in ``build_ieee39_net`` names every synchronous
    machine ``{label}_bus{ieee_term_0idx}``; labels present in ``net.gen``
    are retained, the complement of :data:`GEN_NAMEPLATE` was removed by
    the scenario (wind_replace).  PowerFactory uses this list to set the
    template machines (and their step-up transformers) out of service.
    """
    retained: set = set()
    for gi in net.gen.index:
        name = str(net.gen.at[gi, "name"])
        m = _GEN_NAME_RE.match(name)
        if not m:
            raise SnapshotValueError(
                f"gen[{gi}] name {name!r} does not follow the "
                f"'{{label}}_bus{{term}}' convention; cannot derive the "
                f"removed-generator list"
            )
        retained.add(m.group(1))

    removed = []
    for term_bus, (label, sn_mva, gen_type) in sorted(GEN_NAMEPLATE.items()):
        if label in retained:
            continue
        removed.append({
            "label": label,
            "ieee_bus_1idx": term_bus + 1,
            "term_bus_0idx": term_bus,
            "sn_mva": float(sn_mva),
            "gen_type": gen_type,
        })
    return removed


#: Name marker stamped by ``network/ieee39/hv_networks.py`` on the parks that
#: sit at an EHV-HV coupling bus.  Role is an intrinsic property of what the
#: builder created, so it must not be re-derived from the capability model:
#: this used to key off ``op_diagram == "STATCOM"``, which silently reclassified
#: all 12 coupling parks (and renamed their PF objects ``WPC_* -> DER_*``) the
#: moment their operating diagram was corrected on 2026-07-21.
_COUPLING_WP_NAME_MARKER = "WP_STATCOM_HV"


def _sgen_role(idx: int, net: pp.pandapowerNet,
               meta: IEEE39NetworkMeta) -> str:
    """Classify a static generator for the PowerFactory naming convention."""
    if idx in set(int(s) for s in meta.tso_der_indices):
        return "TSO-WP"
    subnet = str(net.sgen.at[idx, "subnet"])
    name = str(net.sgen.at[idx, "name"])
    op_diagram = str(net.sgen.at[idx, "op_diagram"]) \
        if "op_diagram" in net.sgen.columns else ""
    if idx in set(int(s) for s in meta.dso_der_indices):
        if subnet == "DN" and (
            _COUPLING_WP_NAME_MARKER in name
            # Legacy fallback: snapshots written before the 2026-07-21
            # op_diagram correction identified these parks by diagram alone.
            or op_diagram == "STATCOM"
        ):
            return "DSO-COUPLING-WP"
        return "DSO-DER"
    raise SnapshotValueError(
        f"sgen[{idx}] ({net.sgen.at[idx, 'name']!r}) is neither a TSO nor a "
        f"DSO DER per meta -- unknown role"
    )


def _actuator_inventory(meta: IEEE39NetworkMeta) -> Dict[str, Any]:
    """Flat OFO actuator inventory (convenience view over ``meta``)."""
    return {
        "machine_oltc_trafo_indices": [
            int(t) for t, g in zip(meta.machine_trafo_indices,
                                   meta.machine_trafo_gen_map) if g >= 0
        ],
        "network_oltc_trafo_indices": [
            int(t) for t, g in zip(meta.machine_trafo_indices,
                                   meta.machine_trafo_gen_map) if g < 0
        ],
        "coupler_oltc_trafo3w_indices": [
            int(t) for hv in meta.hv_networks
            for t in hv.coupling_trafo_indices
        ],
        "avr_gen_indices": [int(g) for g in meta.gen_indices],
        "tso_der_sgen_indices": [int(s) for s in meta.tso_der_indices],
        "dso_der_sgen_indices": [int(s) for s in meta.dso_der_indices],
        "tso_tertiary_shunt_indices": [
            int(s) for s in meta.tso_tertiary_shunt_indices
        ],
        "tso_tertiary_shunt_kinds": list(meta.tso_tertiary_shunt_kinds),
    }


# =====================================================================
#  Dump
# =====================================================================

def dump_dynamic_snapshot(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    zone_map: Mapping[int, Sequence[int]],
    label: str,
    out_dir: Union[str, Path],
    *,
    solver_options: Mapping[str, Any],
    snapshot_time: Optional[datetime] = None,
    phase: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """Serialise ``net`` + its converged solution to ``out_dir/label.json``.

    Parameters
    ----------
    net : pp.pandapowerNet
        Fully built, profile-scaled network whose **last** power flow was
        run with exactly ``solver_options`` and converged.  ``net.res_*``
        tables are serialised as the parity target.
    meta : IEEE39NetworkMeta
        Index catalogue matching ``net``.
    zone_map : Mapping[int, Sequence[int]]
        Control-zone partition (zone id -> bus indices), including the
        dispatch extension (HV buses, machine terminal buses).
    label : str
        File stem; also stored in the provenance block.
    out_dir : path-like
        Target directory (created if absent).
    solver_options : Mapping
        The exact ``pp.runpp`` keyword set that produced ``net.res_*``.
        Must define all of ``run_control, calculate_voltage_angles, init,
        max_iteration, distributed_slack, enforce_q_lims`` with
        ``run_control=False``.
    snapshot_time : datetime, optional
        Profile timestamp the network was scaled to.
    phase : str, optional
        Build phase tag (``base`` / ``wind_replace`` / ``full``).
    notes : str, optional
        Free-text provenance note.

    Returns
    -------
    Path
        The written JSON file.

    Raises
    ------
    SnapshotSchemaError, SnapshotValueError
        On any schema gap, NaN in a required field, attached controllers,
        non-empty unsupported tables, or a non-converged solution.
    """
    # ── Pre-flight validation ────────────────────────────────────────────
    missing_opts = [k for k in _SOLVER_OPTION_KEYS if k not in solver_options]
    if missing_opts:
        raise SnapshotValueError(
            f"solver_options lacks required key(s) {missing_opts}"
        )
    if solver_options["run_control"]:
        raise SnapshotValueError(
            "solver_options['run_control'] must be False: snapshots are "
            "controller-free states"
        )
    if hasattr(net, "controller") and len(net.controller) > 0:
        raise SnapshotValueError(
            f"net.controller holds {len(net.controller)} controller(s); "
            f"drop them before dumping (the stored solution must be "
            f"reproducible by a plain runpp)"
        )
    for tbl in _MUST_BE_EMPTY:
        if tbl in net and isinstance(net[tbl], pd.DataFrame) and len(net[tbl]):
            raise SnapshotSchemaError(
                f"net.{tbl} is non-empty ({len(net[tbl])} rows) but the "
                f"snapshot schema does not cover '{tbl}'"
            )
    if not bool(net.get("converged", False)):
        raise SnapshotValueError(
            "net.converged is False -- run a converging pp.runpp with "
            "solver_options before dumping"
        )
    if len(net.res_bus) != len(net.bus):
        raise SnapshotValueError(
            f"res_bus has {len(net.res_bus)} rows but bus has "
            f"{len(net.bus)} -- stale results?"
        )

    problems: List[str] = []
    for table, schema in TABLE_SCHEMAS.items():
        _check_table_schema(net[table], table, schema, problems)
    if problems:
        raise SnapshotSchemaError(
            "Snapshot schema violations:\n  - " + "\n  - ".join(problems)
        )

    # ── Model section ────────────────────────────────────────────────────
    model: Dict[str, Any] = {
        "net": {"f_hz": float(net.f_hz), "sn_mva": float(net.sn_mva)},
    }
    for table, schema in TABLE_SCHEMAS.items():
        model[table] = _dump_table(net[table], table, schema)

    # Attach the control zone to each bus record (None outside the map).
    bus_zone: Dict[int, int] = {}
    for z, buses in zone_map.items():
        for b in buses:
            if b in bus_zone:
                raise SnapshotValueError(
                    f"bus {b} appears in zones {bus_zone[b]} and {z}"
                )
            bus_zone[int(b)] = int(z)
    for key, rec in model["bus"].items():
        rec["zone"] = bus_zone.get(int(key))

    # Attach the derived role to each sgen record.
    for key, rec in model["sgen"].items():
        rec["role"] = _sgen_role(int(key), net, meta)

    # Referential integrity: every branch/injection endpoint must exist.
    bus_keys = set(model["bus"].keys())
    _refs = {
        "line": ("from_bus", "to_bus"),
        "trafo": ("hv_bus", "lv_bus"),
        "trafo3w": ("hv_bus", "mv_bus", "lv_bus"),
        "load": ("bus",), "sgen": ("bus",), "gen": ("bus",),
        "shunt": ("bus",),
    }
    for table, cols in _refs.items():
        for key, rec in model[table].items():
            for col in cols:
                if str(rec[col]) not in bus_keys:
                    raise SnapshotValueError(
                        f"{table}[{key}].{col} = {rec[col]} references a "
                        f"bus that is not serialised"
                    )

    # ── Solution section ─────────────────────────────────────────────────
    solution: Dict[str, Any] = {}
    for table, fields in _SOLUTION_FIELDS.items():
        res = net[f"res_{table}"]
        if len(res) != len(net[table]):
            raise SnapshotValueError(
                f"res_{table} has {len(res)} rows but {table} has "
                f"{len(net[table])}"
            )
        solution[table] = _dump_solution_table(res, table, fields)

    # ── Assemble document ────────────────────────────────────────────────
    doc: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "label": label,
            "phase": phase,
            "snapshot_time": (snapshot_time.isoformat()
                              if snapshot_time else None),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "pandapower": pp.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "notes": notes,
        },
        "solver_options": dict(solver_options),
        "zone_map": {str(int(z)): [int(b) for b in buses]
                     for z, buses in zone_map.items()},
        "meta": _meta_to_dict(meta),
        "removed_generators": _removed_generators(net),
        "actuators": _actuator_inventory(meta),
        "model": model,
        "solution": solution,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        # allow_nan=False: any NaN that slipped past the field policy is a
        # bug and must raise here rather than emit invalid JSON.
        json.dump(doc, handle, indent=1, allow_nan=False)
    return out_path


# =====================================================================
#  Load / rebuild
# =====================================================================

def load_snapshot(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and structurally validate a snapshot JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotValueError(
            f"{path.name}: schema_version {doc.get('schema_version')!r} "
            f"!= supported {SCHEMA_VERSION}"
        )
    for section in ("solver_options", "zone_map", "meta", "model",
                    "solution"):
        if section not in doc:
            raise SnapshotValueError(f"{path.name}: section '{section}' missing")
    return doc


def _rec_get(rec: Mapping[str, Any], key: str, *, table: str,
             idx: str) -> Any:
    if key not in rec:
        raise SnapshotValueError(f"{table}[{idx}] lacks field '{key}'")
    return rec[key]


def _sorted_items(table_dict: Mapping[str, Mapping[str, Any]]):
    return sorted(table_dict.items(), key=lambda kv: int(kv[0]))


def _set_extra_columns(df: pd.DataFrame, table_dict: Mapping[str, Any],
                       columns: Sequence[str]) -> None:
    """Assign nullable/custom columns present in the records to ``df``."""
    present = [c for c in columns
               if any(c in rec for rec in table_dict.values())]
    for col in present:
        values = {}
        for key, rec in table_dict.items():
            v = rec.get(col)
            values[int(key)] = np.nan if v is None else v
        series = pd.Series(values)
        df[col] = series.reindex(df.index)


def load_snapshot_to_pandapower(
    source: Union[str, Path, Mapping[str, Any]],
) -> Tuple[pp.pandapowerNet, Dict[str, Any]]:
    """Rebuild a runnable pandapower network from a snapshot.

    Only the ``model`` section is consumed; the caller may then execute
    ``pp.runpp(net, **doc['solver_options'])`` and compare against
    ``doc['solution']`` (see :func:`verify_roundtrip`).

    Returns
    -------
    (net, doc)
        The rebuilt network and the parsed snapshot document.
    """
    doc = load_snapshot(source) if not isinstance(source, Mapping) else dict(source)
    model = doc["model"]

    net = pp.create_empty_network(
        f_hz=float(model["net"]["f_hz"]),
        sn_mva=float(model["net"]["sn_mva"]),
    )

    # ── Buses ────────────────────────────────────────────────────────────
    for key, rec in _sorted_items(model["bus"]):
        pp.create_bus(
            net,
            index=int(key),
            vn_kv=_rec_get(rec, "vn_kv", table="bus", idx=key),
            name=rec.get("name"),
            type=rec.get("type") or "b",
            in_service=_rec_get(rec, "in_service", table="bus", idx=key),
        )
    _set_extra_columns(net.bus, model["bus"], ("subnet", "zone"))

    # ── Lines ────────────────────────────────────────────────────────────
    for key, rec in _sorted_items(model["line"]):
        pp.create_line_from_parameters(
            net,
            index=int(key),
            from_bus=_rec_get(rec, "from_bus", table="line", idx=key),
            to_bus=_rec_get(rec, "to_bus", table="line", idx=key),
            length_km=_rec_get(rec, "length_km", table="line", idx=key),
            r_ohm_per_km=_rec_get(rec, "r_ohm_per_km", table="line", idx=key),
            x_ohm_per_km=_rec_get(rec, "x_ohm_per_km", table="line", idx=key),
            c_nf_per_km=_rec_get(rec, "c_nf_per_km", table="line", idx=key),
            g_us_per_km=_rec_get(rec, "g_us_per_km", table="line", idx=key),
            max_i_ka=_rec_get(rec, "max_i_ka", table="line", idx=key),
            df=_rec_get(rec, "df", table="line", idx=key),
            parallel=_rec_get(rec, "parallel", table="line", idx=key),
            in_service=_rec_get(rec, "in_service", table="line", idx=key),
            name=rec.get("name"),
            type=rec.get("type"),
        )
    _set_extra_columns(net.line, model["line"], ("subnet", "std_type"))

    # ── Two-winding transformers ─────────────────────────────────────────
    for key, rec in _sorted_items(model["trafo"]):
        kwargs: Dict[str, Any] = {}
        for opt in ("tap_side", "tap_neutral", "tap_min", "tap_max",
                    "tap_pos", "tap_step_percent", "tap_step_degree",
                    "tap_changer_type"):
            if rec.get(opt) is not None:
                kwargs[opt] = rec[opt]
        pp.create_transformer_from_parameters(
            net,
            index=int(key),
            hv_bus=_rec_get(rec, "hv_bus", table="trafo", idx=key),
            lv_bus=_rec_get(rec, "lv_bus", table="trafo", idx=key),
            sn_mva=_rec_get(rec, "sn_mva", table="trafo", idx=key),
            vn_hv_kv=_rec_get(rec, "vn_hv_kv", table="trafo", idx=key),
            vn_lv_kv=_rec_get(rec, "vn_lv_kv", table="trafo", idx=key),
            vk_percent=_rec_get(rec, "vk_percent", table="trafo", idx=key),
            vkr_percent=_rec_get(rec, "vkr_percent", table="trafo", idx=key),
            pfe_kw=_rec_get(rec, "pfe_kw", table="trafo", idx=key),
            i0_percent=_rec_get(rec, "i0_percent", table="trafo", idx=key),
            shift_degree=_rec_get(rec, "shift_degree", table="trafo", idx=key),
            df=_rec_get(rec, "df", table="trafo", idx=key),
            parallel=_rec_get(rec, "parallel", table="trafo", idx=key),
            in_service=_rec_get(rec, "in_service", table="trafo", idx=key),
            name=rec.get("name"),
            **kwargs,
        )
    _set_extra_columns(net.trafo, model["trafo"], ("std_type",))

    # ── Three-winding transformers ───────────────────────────────────────
    for key, rec in _sorted_items(model["trafo3w"]):
        kwargs = {}
        for opt in ("tap_side", "tap_neutral", "tap_min", "tap_max",
                    "tap_pos", "tap_step_percent", "tap_step_degree",
                    "tap_changer_type", "tap_at_star_point"):
            if rec.get(opt) is not None:
                kwargs[opt] = rec[opt]
        pp.create_transformer3w_from_parameters(
            net,
            index=int(key),
            hv_bus=_rec_get(rec, "hv_bus", table="trafo3w", idx=key),
            mv_bus=_rec_get(rec, "mv_bus", table="trafo3w", idx=key),
            lv_bus=_rec_get(rec, "lv_bus", table="trafo3w", idx=key),
            sn_hv_mva=_rec_get(rec, "sn_hv_mva", table="trafo3w", idx=key),
            sn_mv_mva=_rec_get(rec, "sn_mv_mva", table="trafo3w", idx=key),
            sn_lv_mva=_rec_get(rec, "sn_lv_mva", table="trafo3w", idx=key),
            vn_hv_kv=_rec_get(rec, "vn_hv_kv", table="trafo3w", idx=key),
            vn_mv_kv=_rec_get(rec, "vn_mv_kv", table="trafo3w", idx=key),
            vn_lv_kv=_rec_get(rec, "vn_lv_kv", table="trafo3w", idx=key),
            vk_hv_percent=_rec_get(rec, "vk_hv_percent", table="trafo3w", idx=key),
            vk_mv_percent=_rec_get(rec, "vk_mv_percent", table="trafo3w", idx=key),
            vk_lv_percent=_rec_get(rec, "vk_lv_percent", table="trafo3w", idx=key),
            vkr_hv_percent=_rec_get(rec, "vkr_hv_percent", table="trafo3w", idx=key),
            vkr_mv_percent=_rec_get(rec, "vkr_mv_percent", table="trafo3w", idx=key),
            vkr_lv_percent=_rec_get(rec, "vkr_lv_percent", table="trafo3w", idx=key),
            pfe_kw=_rec_get(rec, "pfe_kw", table="trafo3w", idx=key),
            i0_percent=_rec_get(rec, "i0_percent", table="trafo3w", idx=key),
            shift_mv_degree=_rec_get(rec, "shift_mv_degree", table="trafo3w", idx=key),
            shift_lv_degree=_rec_get(rec, "shift_lv_degree", table="trafo3w", idx=key),
            in_service=_rec_get(rec, "in_service", table="trafo3w", idx=key),
            name=rec.get("name"),
            **kwargs,
        )
    _set_extra_columns(net.trafo3w, model["trafo3w"], ("std_type",))

    # ── Loads ────────────────────────────────────────────────────────────
    for key, rec in _sorted_items(model["load"]):
        pp.create_load(
            net,
            index=int(key),
            bus=_rec_get(rec, "bus", table="load", idx=key),
            p_mw=_rec_get(rec, "p_mw", table="load", idx=key),
            q_mvar=_rec_get(rec, "q_mvar", table="load", idx=key),
            const_z_p_percent=_rec_get(rec, "const_z_p_percent", table="load", idx=key),
            const_z_q_percent=_rec_get(rec, "const_z_q_percent", table="load", idx=key),
            const_i_p_percent=_rec_get(rec, "const_i_p_percent", table="load", idx=key),
            const_i_q_percent=_rec_get(rec, "const_i_q_percent", table="load", idx=key),
            sn_mva=rec.get("sn_mva") if rec.get("sn_mva") is not None else np.nan,
            scaling=_rec_get(rec, "scaling", table="load", idx=key),
            in_service=_rec_get(rec, "in_service", table="load", idx=key),
            name=rec.get("name"),
            type=rec.get("type"),
        )
    _set_extra_columns(net.load, model["load"],
                       ("subnet", "profile_p", "profile_q",
                        "base_p_mw", "base_q_mvar", "zip_anchor_vm_pu"))

    # ── Static generators ────────────────────────────────────────────────
    for key, rec in _sorted_items(model["sgen"]):
        pp.create_sgen(
            net,
            index=int(key),
            bus=_rec_get(rec, "bus", table="sgen", idx=key),
            p_mw=_rec_get(rec, "p_mw", table="sgen", idx=key),
            q_mvar=_rec_get(rec, "q_mvar", table="sgen", idx=key),
            sn_mva=_rec_get(rec, "sn_mva", table="sgen", idx=key),
            scaling=_rec_get(rec, "scaling", table="sgen", idx=key),
            in_service=_rec_get(rec, "in_service", table="sgen", idx=key),
            name=rec.get("name"),
            type=rec.get("type"),
        )
    _set_extra_columns(net.sgen, model["sgen"],
                       ("subnet", "profile", "op_diagram", "base_p_mw",
                        "q_mode", "qv_slope_pu", "qv_vref_pu",
                        "qv_deadband_pu", "cosphi", "cosphi_sign",
                        "q_set_mvar", "qv_vref_anchor_pu", "role"))

    # ── Synchronous machines ─────────────────────────────────────────────
    for key, rec in _sorted_items(model["gen"]):
        pp.create_gen(
            net,
            index=int(key),
            bus=_rec_get(rec, "bus", table="gen", idx=key),
            p_mw=_rec_get(rec, "p_mw", table="gen", idx=key),
            vm_pu=_rec_get(rec, "vm_pu", table="gen", idx=key),
            sn_mva=_rec_get(rec, "sn_mva", table="gen", idx=key),
            min_q_mvar=_rec_get(rec, "min_q_mvar", table="gen", idx=key),
            max_q_mvar=_rec_get(rec, "max_q_mvar", table="gen", idx=key),
            min_p_mw=_rec_get(rec, "min_p_mw", table="gen", idx=key),
            max_p_mw=_rec_get(rec, "max_p_mw", table="gen", idx=key),
            scaling=_rec_get(rec, "scaling", table="gen", idx=key),
            slack=_rec_get(rec, "slack", table="gen", idx=key),
            in_service=_rec_get(rec, "in_service", table="gen", idx=key),
            name=rec.get("name"),
            type=rec.get("type"),
        )
        net.gen.at[int(key), "slack_weight"] = float(
            _rec_get(rec, "slack_weight", table="gen", idx=key)
        )
    _set_extra_columns(net.gen, model["gen"], ("profile", "base_p_mw"))

    # ── Shunts ───────────────────────────────────────────────────────────
    for key, rec in _sorted_items(model["shunt"]):
        pp.create_shunt(
            net,
            index=int(key),
            bus=_rec_get(rec, "bus", table="shunt", idx=key),
            q_mvar=_rec_get(rec, "q_mvar", table="shunt", idx=key),
            p_mw=_rec_get(rec, "p_mw", table="shunt", idx=key),
            vn_kv=_rec_get(rec, "vn_kv", table="shunt", idx=key),
            step=_rec_get(rec, "step", table="shunt", idx=key),
            max_step=_rec_get(rec, "max_step", table="shunt", idx=key),
            in_service=_rec_get(rec, "in_service", table="shunt", idx=key),
            name=rec.get("name"),
        )
    if "step" in net.shunt.columns and len(net.shunt):
        # Mirror add_hv_networks: bipolar step = -1 must survive the dtype.
        net.shunt["step"] = net.shunt["step"].astype("int64")

    net["name"] = str(doc.get("provenance", {}).get("label", "snapshot"))
    return net, doc


# =====================================================================
#  Round-trip verification
# =====================================================================

@dataclass
class RoundTripReport:
    """Deviation summary between a stored and a recomputed solution."""
    label: str
    ok: bool
    max_dev: Dict[str, float]
    worst: List[Tuple[str, int, str, float, float, float]]
    """Rows ``(table, index, field, stored, recomputed, |dev|)`` sorted by
    descending deviation (top 10)."""

    def summary(self) -> str:
        lines = [f"Round-trip report for '{self.label}': "
                 f"{'OK' if self.ok else 'FAILED'}"]
        for quantity, dev in sorted(self.max_dev.items()):
            lines.append(f"  max |d {quantity}| = {dev:.3e}")
        if not self.ok:
            lines.append("  worst deviations:")
            for table, idx, fld, stored, new, dev in self.worst:
                lines.append(
                    f"    {table}[{idx}].{fld}: stored={stored:.10g} "
                    f"recomputed={new:.10g} |dev|={dev:.3e}"
                )
        return "\n".join(lines)


#: Comparison tolerances per solution field family.
_ROUNDTRIP_TOL = {
    "vm_pu": 1e-8, "va_degree": 1e-8,
    "p": 1e-6, "q": 1e-6, "i": 1e-9,
}


def _tol_for(fieldname: str) -> float:
    if fieldname == "vm_pu":
        return _ROUNDTRIP_TOL["vm_pu"]
    if fieldname == "va_degree":
        return _ROUNDTRIP_TOL["va_degree"]
    if fieldname.startswith("i_"):
        return _ROUNDTRIP_TOL["i"]
    if fieldname.startswith("p"):
        return _ROUNDTRIP_TOL["p"]
    if fieldname.startswith("q"):
        return _ROUNDTRIP_TOL["q"]
    raise SnapshotValueError(f"No tolerance rule for field {fieldname!r}")


def verify_roundtrip(path: Union[str, Path]) -> RoundTripReport:
    """Rebuild the snapshot, rerun the stored solver options, compare.

    Returns a :class:`RoundTripReport`; ``report.ok`` is True when every
    recomputed solution entry matches the stored one within
    :data:`_ROUNDTRIP_TOL`.
    """
    net, doc = load_snapshot_to_pandapower(path)
    pp.runpp(net, **doc["solver_options"])

    devs: List[Tuple[str, int, str, float, float, float]] = []
    max_dev: Dict[str, float] = {}
    ok = True
    for table, fields in _SOLUTION_FIELDS.items():
        stored_tbl = doc["solution"][table]
        res = net[f"res_{table}"]
        if len(stored_tbl) != len(res):
            raise SnapshotValueError(
                f"res_{table}: stored {len(stored_tbl)} rows, "
                f"recomputed {len(res)}"
            )
        for key, rec in stored_tbl.items():
            idx = int(key)
            for fld in fields:
                stored = float(rec[fld])
                new = float(res.at[idx, fld])
                dev = abs(new - stored)
                quantity = f"{table}.{fld}"
                max_dev[quantity] = max(max_dev.get(quantity, 0.0), dev)
                if dev > _tol_for(fld):
                    ok = False
                    devs.append((table, idx, fld, stored, new, dev))

    devs.sort(key=lambda row: row[-1], reverse=True)
    label = doc.get("provenance", {}).get("label", str(path))
    return RoundTripReport(label=label, ok=ok, max_dev=max_dev,
                           worst=devs[:10])
