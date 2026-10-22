"""
pf/naming.py
============
The PowerFactory ``loc_name`` convention for all script-owned objects,
derived deterministically from a dynamic snapshot (export/dynamic_snapshot).

Single source of truth for the names that ``pf_sync`` (Phase 2+) creates
and that ``pf_parity`` / ``PowerFactoryPlant`` look up.  The human-readable
specification lives in docs/pf_naming.md; this module is the executable
form.  ``tests/pf/test_naming.py`` proves the map is total (every model
element receives a name) and collision-free on the shipped reference
snapshots.

Principles
----------
* Every script-owned object embeds its **pandapower index**; the snapshot
  is keyed the same way, so sync and parity never guess.
* Template-owned objects (the ten ElmSym machines) are **not** renamed;
  they are addressed via :data:`TEMPLATE_MACHINE_NAMES`.
* Only ``[A-Za-z0-9_]`` in generated names.

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple

from export.dynamic_snapshot import meta_from_dict

_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

#: pandapower gen label -> machine loc_name in the DIgSILENT 39-bus
#: template.  Verified against the live project on 2026-07-17 (PF 2025 SP4,
#: hello_pf run: ElmSym objects 'G 01' ... 'G 10', all in service).
TEMPLATE_MACHINE_NAMES: Dict[str, str] = {
    "G1": "G 01", "G2": "G 02", "G3": "G 03", "G4": "G 04", "G5": "G 05",
    "G6": "G 06", "G7": "G 07", "G8": "G 08", "G9": "G 09", "G10": "G 10",
}
TEMPLATE_NAMES_VERIFIED: bool = True


class NamingError(RuntimeError):
    """The snapshot contains an element the convention cannot classify."""


def _hv_membership(meta) -> Tuple[Dict[int, str], Dict[int, str],
                                  Dict[int, str], Dict[int, str],
                                  Dict[int, str], Dict[int, str],
                                  Dict[int, str]]:
    """Per-subnet membership maps: bus/tertiary/line/sgen/trafo3w -> net_id."""
    hv_bus: Dict[int, str] = {}
    tert_bus: Dict[int, str] = {}
    hv_line: Dict[int, str] = {}
    hv_sgen: Dict[int, str] = {}
    coupler: Dict[int, str] = {}
    aux_bus: Dict[int, str] = {}
    aux_line: Dict[int, str] = {}
    for hv in meta.hv_networks:
        for b in hv.bus_indices:
            hv_bus[int(b)] = hv.net_id
        for b in hv.coupling_lv_bus_indices:
            tert_bus[int(b)] = hv.net_id
        for li in hv.line_indices:
            hv_line[int(li)] = hv.net_id
        for s in hv.sgen_indices:
            hv_sgen[int(s)] = hv.net_id
        for t in hv.coupling_trafo_indices:
            coupler[int(t)] = hv.net_id
        for b in hv.internal_aux_bus_indices:
            aux_bus[int(b)] = hv.net_id
        for li in hv.internal_aux_line_indices:
            aux_line[int(li)] = hv.net_id
    return hv_bus, tert_bus, hv_line, hv_sgen, coupler, aux_bus, aux_line


def build_name_map(doc: Mapping[str, Any]) -> Dict[Tuple[str, int], str]:
    """loc_name for every script-owned element of a snapshot document.

    Parameters
    ----------
    doc : Mapping
        Parsed snapshot JSON (see :func:`export.dynamic_snapshot.load_snapshot`).

    Returns
    -------
    dict
        ``(table, pandapower_index) -> loc_name`` covering every element of
        the ``model`` section **except** ``gen`` (template-owned machines,
        see :data:`TEMPLATE_MACHINE_NAMES`).  Sgen controller objects
        (ElmStactrl) are named ``CTRL_`` + the sgen name and are not part
        of this map.

    Raises
    ------
    NamingError
        If any element cannot be classified or two names collide.
    """
    meta = meta_from_dict(doc["meta"])
    model = doc["model"]
    (hv_bus, tert_bus, hv_line, hv_sgen, coupler,
     aux_bus, aux_line) = _hv_membership(meta)

    machine_trafo_gen = {
        int(t): int(g) for t, g in zip(meta.machine_trafo_indices,
                                       meta.machine_trafo_gen_map)
    }
    shunt_kind = {
        int(s): str(k) for s, k in zip(meta.tso_tertiary_shunt_indices,
                                       meta.tso_tertiary_shunt_kinds)
    }

    names: Dict[Tuple[str, int], str] = {}

    def _put(table: str, idx: int, name: str) -> None:
        if not _NAME_RE.match(name):
            raise NamingError(f"{table}[{idx}]: generated name {name!r} "
                              f"contains characters outside [A-Za-z0-9_]")
        names[(table, idx)] = name

    # ── Buses ────────────────────────────────────────────────────────────
    for key, rec in model["bus"].items():
        idx = int(key)
        subnet = rec.get("subnet")
        vn = float(rec["vn_kv"])
        if idx in tert_bus:
            _put("bus", idx, f"{tert_bus[idx]}_tert{idx}")
        elif idx in hv_bus:
            _put("bus", idx, f"{hv_bus[idx]}_bus{idx}")
        elif subnet == "GEN_TERM" or (subnet == "TN" and vn < 100.0):
            # 10.5 kV machine terminals: created ones carry subnet GEN_TERM;
            # pre-existing case39 gen buses keep subnet TN with vn = 10.5.
            _put("bus", idx, f"GT_bus{idx}")
        elif subnet == "TN_AUX":
            _put("bus", idx, f"AUX_TN_bus{idx}")
        elif subnet == "DN_AUX":
            net_id = aux_bus.get(idx)
            if net_id is None:
                raise NamingError(
                    f"bus[{idx}]: DN_AUX bus has no DSO owner in metadata"
                )
            _put("bus", idx, f"AUX_{net_id}_bus{idx}")
        elif subnet == "TN":
            _put("bus", idx, f"TN_bus{idx}")
        else:
            raise NamingError(
                f"bus[{idx}]: unclassifiable (subnet={subnet!r}, "
                f"vn_kv={vn}, not in any HV sub-network)"
            )

    # ── Lines ────────────────────────────────────────────────────────────
    for key, rec in model["line"].items():
        idx = int(key)
        if idx in hv_line:
            _put("line", idx, f"{hv_line[idx]}_line{idx}")
        elif rec.get("subnet") == "TN_AUX":
            _put("line", idx, f"AUX_TN_line{idx}")
        elif rec.get("subnet") == "DN_AUX":
            net_id = aux_line.get(idx)
            if net_id is None:
                raise NamingError(
                    f"line[{idx}]: DN_AUX line has no DSO owner in metadata"
                )
            _put("line", idx, f"AUX_{net_id}_line{idx}")
        elif rec.get("subnet") == "TN":
            _put("line", idx, f"TN_line{idx}")
        else:
            raise NamingError(f"line[{idx}]: unclassifiable "
                              f"(subnet={rec.get('subnet')!r})")

    # ── Two-winding transformers ─────────────────────────────────────────
    for key in model["trafo"]:
        idx = int(key)
        if idx not in machine_trafo_gen:
            raise NamingError(
                f"trafo[{idx}] is not in meta.machine_trafo_indices -- "
                f"unknown 2W transformer class"
            )
        g = machine_trafo_gen[idx]
        if g >= 0:
            _put("trafo", idx, f"MT_g{g}_t{idx}")     # machine step-up OLTC
        else:
            _put("trafo", idx, f"NT_t{idx}")          # network OLTC (bus 12)

    # ── Three-winding coupling transformers ──────────────────────────────
    for key in model["trafo3w"]:
        idx = int(key)
        if idx not in coupler:
            raise NamingError(
                f"trafo3w[{idx}] is not a coupling transformer of any "
                f"HV sub-network"
            )
        _put("trafo3w", idx, f"NC3W_{coupler[idx]}_t{idx}")

    # ── Loads ────────────────────────────────────────────────────────────
    for key, rec in model["load"].items():
        idx = int(key)
        bus = int(rec["bus"])
        kind = "var" if rec.get("profile_p") else "const"
        if rec.get("subnet") == "TN":
            _put("load", idx, f"TN_load{idx}_{kind}_b{bus}")
        elif rec.get("subnet") == "DN":
            net_id = hv_bus.get(bus) or tert_bus.get(bus) or aux_bus.get(bus)
            if net_id is None:
                raise NamingError(f"load[{idx}]: DN load at bus {bus} "
                                  f"outside every HV sub-network")
            _put("load", idx, f"{net_id}_load{idx}_{kind}_b{bus}")
        else:
            raise NamingError(f"load[{idx}]: unclassifiable "
                              f"(subnet={rec.get('subnet')!r})")

    # ── Static generators (DER / wind parks) ─────────────────────────────
    for key, rec in model["sgen"].items():
        idx = int(key)
        bus = int(rec["bus"])
        role = rec["role"]
        if role == "TSO-WP":
            _put("sgen", idx, f"WP_TSO_s{idx}_b{bus}")
        elif role == "DSO-COUPLING-WP":
            _put("sgen", idx, f"WPC_{hv_sgen[idx]}_s{idx}_b{bus}")
        elif role == "DSO-DER":
            _put("sgen", idx, f"DER_{hv_sgen[idx]}_s{idx}_b{bus}")
        else:
            raise NamingError(f"sgen[{idx}]: unknown role {role!r}")

    # ── Shunts ───────────────────────────────────────────────────────────
    for key, rec in model["shunt"].items():
        idx = int(key)
        if idx not in shunt_kind:
            raise NamingError(
                f"shunt[{idx}] is not a TSO tertiary shunt -- unknown class"
            )
        bus = int(rec["bus"])
        net_id = tert_bus.get(bus)
        if net_id is None:
            raise NamingError(f"shunt[{idx}]: bus {bus} is not a tertiary "
                              f"bus of any HV sub-network")
        _put("shunt", idx, f"SH_{shunt_kind[idx]}_{net_id}_s{idx}")

    # ── Uniqueness across everything ─────────────────────────────────────
    seen: Dict[str, Tuple[str, int]] = {}
    for elm_key, name in names.items():
        if name in seen:
            raise NamingError(
                f"loc_name collision: {name!r} for {seen[name]} and {elm_key}"
            )
        seen[name] = elm_key
    return names


def controller_name(sgen_loc_name: str) -> str:
    """ElmStactrl loc_name for a static generator's Q controller."""
    return f"CTRL_{sgen_loc_name}"


def machine_template_name(gen_record: Mapping[str, Any]) -> str:
    """Template ElmSym loc_name for a snapshot ``model.gen`` record.

    The pandapower gen names follow ``{label}_bus{term}`` (nameplate loop in
    build_ieee39_net); the label indexes :data:`TEMPLATE_MACHINE_NAMES`.
    """
    name = str(gen_record.get("name"))
    m = re.match(r"^(G\d+)_bus\d+$", name)
    if not m:
        raise NamingError(
            f"gen name {name!r} does not follow '{{label}}_bus{{term}}'"
        )
    label = m.group(1)
    if label not in TEMPLATE_MACHINE_NAMES:
        raise NamingError(f"no template machine mapping for label {label!r}")
    return TEMPLATE_MACHINE_NAMES[label]
