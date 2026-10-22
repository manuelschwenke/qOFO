"""
pf/pf_sync.py
=============
Synchronise the IEEE39_qOFO PowerFactory project with a dynamic snapshot
(export/dynamic_snapshot.py).  Phase ``base`` (this file's first cut)
aligns the 345 kV grid: template adoption (rename to the pf/naming.py
convention), line/trafo parameter push, load replacement, machine dispatch.

Ownership rules (docs/pf_naming.md)
-----------------------------------
* ``ElmSym`` machines and their types are template-owned: never created or
  deleted; the script sets dispatch (``pgini``/``usetp``), the reference-
  machine flag, ``outserv`` and the rated voltage (``TypSym.ugn``) only.
* Everything else is script-owned after adoption: template buses, lines
  and transformers are renamed to the convention on first run and their
  parameters are overwritten from the snapshot; template loads are deleted
  and recreated per snapshot row; template objects without a snapshot
  counterpart (the IEEE bus 20 chain) are deleted.

Behaviour
---------
* ``--dry-run`` prints every intended operation without touching the
  project.
* Idempotent: a second run reports only ``unchanged`` entries.
* Fail-Fast: any snapshot element that cannot be mapped, and any template
  object that cannot be classified, raises.

Usage (PF machine)::

    python pf\\pf_sync.py export\\snapshots\\base_t0_20160105-0800.json --phase base --dry-run
    python pf\\pf_sync.py export\\snapshots\\base_t0_20160105-0800.json --phase base

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.dynamic_snapshot import load_snapshot, meta_from_dict  # noqa: E402
from pf.naming import (  # noqa: E402
    TEMPLATE_NAMES_VERIFIED,
    build_name_map,
    controller_name,
    machine_template_name,
)
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    deactivate_variations_except,
    ensure_variation,
    get_all,
    run_ldf,
    set_variation_active,
)

PARITY_STUDY_CASE = "01_LDF_Parity"

#: Layered variation names: ``base -> wind_replace -> full``.  The full
#: variation stores only the Phase-4 DSO/coupling delta and therefore requires
#: the wind replacement layer to be active underneath it.
WIND_REPLACE_VARIATION = "wind_replace"
FULL_MODEL_VARIATION = "full"

#: The adopted DIgSILENT IEEE39 template grid.  Phase 4 adds four further
#: ElmNet objects under Network Data, so the former "exactly one grid" lookup
#: must identify the base grid by its verified template name.
BASE_GRID_NAME = "Grid"

#: Tolerances for the "unchanged" float comparison.  PowerFactory stores
#: most element attributes in single precision (float32, rel. eps 1.2e-7),
#: so written float64 values read back slightly off; anything tighter than
#: ~1e-6 relative would re-report every float attribute on every run.
_REL_EPS = 2e-6
_ABS_EPS = 1e-9

#: loc_name prefix of the RMS composite model that ``pf/wecc_apply.py``
#: builds per static generator (``WECC_<park loc_name>``).  Repeated here
#: rather than imported so this module keeps its load-flow-only dependency
#: set; :func:`delete_stale_sgens` needs it to take a deleted park's
#: dependants with it.
RMS_COMPOSITE_PREFIX = "WECC_"


# =====================================================================
#  Change report
# =====================================================================

@dataclass
class ChangeReport:
    created: List[str] = field(default_factory=list)
    renamed: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Sync change report:"]
        for kind in ("created", "renamed", "updated", "deleted"):
            entries = getattr(self, kind)
            lines.append(f"  {kind}: {len(entries)}")
            for e in entries:
                lines.append(f"    - {e}")
        lines.append(f"  unchanged: {len(self.unchanged)}")
        return "\n".join(lines)


# =====================================================================
#  Sync context (execution + dry-run indirection)
# =====================================================================

class SyncContext:
    """Wraps every mutating PF call so --dry-run can intercept it."""

    def __init__(self, app, doc: Mapping[str, Any], *, dry_run: bool):
        self.app = app
        self.doc = doc
        self.dry_run = dry_run
        self.report = ChangeReport()
        self.meta = meta_from_dict(doc["meta"])
        self.names = build_name_map(doc)
        #: target loc_name -> ElmTerm as it will exist after sync_buses.
        #: In a dry run, entries for to-be-created terminals hold None.
        self.term_alias: Dict[str, Any] = {}

        # Phase 4 adds one ElmNet per DSO.  The original DIgSILENT template
        # grid remains the unique loc_name='Grid' object under Network Data.
        network_data = app.GetProjectFolder("netdat")
        if network_data is None:
            raise PFSessionError("PowerFactory project has no Network Data folder")
        # Direct folder contents are required here: GetCalcRelevantObjects
        # omits a newly created but still empty ElmNet, which would make an
        # interrupted sync create a decorated duplicate on resume.
        grids = list(network_data.GetContents("*.ElmNet", False) or [])
        base_grids = [g for g in grids if g.loc_name == BASE_GRID_NAME]
        if len(base_grids) != 1:
            raise PFSessionError(
                f"Expected exactly one base grid {BASE_GRID_NAME!r} under "
                f"'Network Data', found {[g.loc_name for g in grids]}"
            )
        self.grid = base_grids[0]
        self.network_data_folder = self.grid.fold_id

        expected_dso = {str(hv.net_id) for hv in self.meta.hv_networks}
        self.dso_grids: Dict[str, Any] = {
            g.loc_name: g for g in grids if g.loc_name in expected_dso
        }
        self.bus_grid_names: Dict[int, str] = {}
        self.line_grid_names: Dict[int, str] = {}
        self.trafo3w_grid_names: Dict[int, str] = {}
        for hv in self.meta.hv_networks:
            net_id = str(hv.net_id)
            for bus_idx in (
                *hv.bus_indices,
                *hv.coupling_lv_bus_indices,
                *hv.internal_aux_bus_indices,
            ):
                self.bus_grid_names[int(bus_idx)] = net_id
            for line_idx in (*hv.line_indices, *hv.internal_aux_line_indices):
                self.line_grid_names[int(line_idx)] = net_id
            for trafo_idx in hv.coupling_trafo_indices:
                self.trafo3w_grid_names[int(trafo_idx)] = net_id

    # ── primitive operations ─────────────────────────────────────────────
    def set_attr(self, obj, attr: str, value, *, label: str) -> bool:
        """Set one attribute if it differs; returns True when changed."""
        try:
            old = obj.GetAttribute(attr)
        except Exception as exc:
            raise PFSessionError(
                f"{label}: cannot read attribute {attr!r}: {exc}"
            ) from exc
        def _pf_key(item):
            if hasattr(item, "GetFullName"):
                return (item.GetClassName(), item.GetFullName())
            return item

        if isinstance(old, float) and isinstance(value, (int, float)):
            same = math.isclose(old, float(value), rel_tol=_REL_EPS,
                                abs_tol=_ABS_EPS)
        elif isinstance(old, (list, tuple)) \
                and isinstance(value, (list, tuple)):
            same = [_pf_key(v) for v in old] == [_pf_key(v) for v in value]
        elif hasattr(old, "GetFullName") and hasattr(value, "GetFullName"):
            same = _pf_key(old) == _pf_key(value)
        else:
            same = old == value
        if same:
            return False
        if not self.dry_run:
            obj.SetAttribute(attr, value)
        self.report.updated.append(
            f"{label}.{attr}: {old!r} -> {value!r}"
        )
        return True

    def rename(self, obj, new_name: str) -> None:
        old = obj.loc_name
        if old == new_name:
            return
        if not self.dry_run:
            obj.loc_name = new_name
        self.report.renamed.append(f"{obj.GetClassName()} {old!r} -> {new_name!r}")

    def create(self, parent, class_name: str, name: str):
        self.report.created.append(f"{class_name} {name!r}")
        if self.dry_run:
            return None
        obj = parent.CreateObject(class_name, name)
        if obj is None:
            raise PFSessionError(
                f"CreateObject({class_name!r}, {name!r}) in "
                f"{parent.loc_name!r} returned None"
            )
        # CreateObject may decorate the name; enforce exactly.
        if obj.loc_name != name:
            obj.loc_name = name
        return obj

    def delete(self, obj, *, label: str) -> None:
        self.report.deleted.append(label)
        if not self.dry_run:
            ierr = obj.Delete()
            if ierr:
                raise PFSessionError(f"Delete() on {label} returned {ierr}")

    def cubicle(self, term, tag: str):
        """New connection cubicle in ``term`` (None in dry-run)."""
        return self.create(term, "StaCubic", f"Cub_qofo_{tag}")


# =====================================================================
#  Helpers
# =====================================================================

def _network_all(ctx: SyncContext, class_name: str) -> List[Any]:
    """All Network Data objects, including disconnected/inactive islands.

    ``GetCalcRelevantObjects`` intentionally omits an unconnected new DSO.
    Direct recursive folder discovery makes a failed full sync resumable
    without producing decorated duplicate elements.
    """
    return list(
        ctx.network_data_folder.GetContents(f"*.{class_name}", True) or []
    )


def _term_index_maps(ctx: SyncContext):
    """Current ElmTerm lookup maps: by loc_name and by template IEEE no."""
    by_name: Dict[str, Any] = {}
    by_ieee: Dict[int, Any] = {}
    for t in _network_all(ctx, "ElmTerm"):
        by_name[t.loc_name] = t
        if t.loc_name.startswith("Bus ") and t.loc_name[4:].isdigit():
            by_ieee[int(t.loc_name[4:])] = t
    return by_name, by_ieee


def _endpoints_ieee(ctx: SyncContext, obj,
                    attrs: Tuple[str, ...]) -> Tuple[int, ...]:
    """IEEE bus numbers a branch connects to (via cubicles).

    Resolves both template names ('Bus NN') and adopted convention names
    (via the reverse of the naming map), so matching works regardless of
    whether the bus adoption has already renamed the terminals -- the
    2026-07-17 first-run bug was exactly this order dependence.
    """
    if not hasattr(ctx, "_busname_to_ieee"):
        rev = {}
        for (tbl, idx), name in ctx.names.items():
            if tbl == "bus" and int(idx) <= 38:
                rev[name] = int(idx) + 1
        ctx._busname_to_ieee = rev
    out = []
    for a in attrs:
        cub = obj.GetAttribute(a)
        if cub is None:
            out.append(-1)
            continue
        name = cub.cterm.loc_name
        if name.startswith("Bus ") and name[4:].isdigit():
            out.append(int(name[4:]))
        elif name in ctx._busname_to_ieee:
            out.append(ctx._busname_to_ieee[name])
        else:
            out.append(-1)  # created GT terminal etc.
    return tuple(out)


def _endpoint_term_names(obj, attrs: Tuple[str, ...]) -> Tuple[Optional[str], ...]:
    """loc_names of the terminals a branch's cubicles currently sit in."""
    out: List[Optional[str]] = []
    for a in attrs:
        cub = obj.GetAttribute(a)
        term = cub.cterm if cub is not None else None
        out.append(term.loc_name if term is not None else None)
    return tuple(out)


def _reconnect(ctx: SyncContext, obj, attr: str, bus_idx: int, tag: str,
               label: str) -> bool:
    """Re-point one connection attribute at the snapshot's terminal.

    Connection cubicles are created exactly once, when the element itself is
    created.  For the element classes whose ``loc_name`` does not encode
    their bus -- ``ElmLne``, ``ElmTr3``, ``ElmShnt`` -- a later bus change in
    the pandapower model therefore leaves the PF object silently attached to
    its original terminal: find-or-create by name matches, every attribute
    compares equal, and the sync reports "unchanged".

    Found 2026-07-29: reordering ``SUBNET_DEFS[*].hv_buses`` from ``(3,0,8)``
    to ``(0,3,8)`` swaps couplers 0 and 1 of every DSO onto each other's
    110 kV bus, i.e. eight ``ElmTr3.busmv`` moves that :func:`sync_trafo3w`
    did not carry over.

    Returns True when the connection was (or, in a dry run, would be) moved.
    """
    target = ctx.names[("bus", int(bus_idx))]
    cub = obj.GetAttribute(attr)
    current = cub.cterm if cub is not None else None
    if current is not None and current.loc_name == target:
        return False
    ctx.report.updated.append(
        f"{label}.{attr}: "
        f"{current.loc_name if current is not None else None!r} -> {target!r}"
    )
    if ctx.dry_run:
        return True
    term = _target_term(ctx, int(bus_idx))
    obj.SetAttribute(attr, ctx.cubicle(term, tag))
    # Drop the vacated cubicle: PF clears its back-reference when the element
    # moves, and leaving it behind would accumulate one dead StaCubic per move
    # and collide on the name if the element ever moved back.  The obj_id
    # guard makes this a no-op should PF keep the reference instead.
    if cub is not None and cub.GetAttribute("obj_id") is None:
        ctx.delete(cub, label=f"StaCubic {cub.loc_name!r} (vacated by {label})")
    return True


def _target_term(ctx: SyncContext, bus_idx: int):
    """ElmTerm for a pandapower bus index (post-bus-sync view).

    Resolves via the sync's alias map (base phase) or, when that has not
    been populated (wind_replace/full deltas run against the already-synced
    base grid), a live lookup by convention name.  Returns None only in a
    dry run for terminals that would be created.
    """
    name = ctx.names[("bus", int(bus_idx))]
    if name in ctx.term_alias:
        term = ctx.term_alias[name]
        if term is None and not ctx.dry_run:
            raise PFSessionError(
                f"Terminal {name!r} recorded as dry-run-created in a real "
                f"run -- internal inconsistency"
            )
        return term
    live = [t for t in _network_all(ctx, "ElmTerm") if t.loc_name == name]
    if live:
        ctx.term_alias[name] = live[0]
        return live[0]
    raise PFSessionError(
        f"Terminal {name!r} (pandapower bus {bus_idx}) not found -- "
        f"base sync must run first"
    )


def ensure_dso_grids(ctx: SyncContext) -> None:
    """Create, parameterise, and activate one ``ElmNet`` per DSO.

    A newly created PF 2025 ``ElmNet`` defaults to 50 Hz and is not part of
    the active study-case network.  Both facts must be corrected before its
    contents become calculation-relevant and before connection references can
    be verified on a subsequent engine session.
    """
    frequency = float(ctx.doc["model"]["net"]["f_hz"])
    for hv in ctx.meta.hv_networks:
        net_id = str(hv.net_id)
        grid = ctx.dso_grids.get(net_id)
        if grid is None:
            grid = ctx.create(ctx.network_data_folder, "ElmNet", net_id)
            ctx.dso_grids[net_id] = grid
        if grid is None:                       # dry-run creation placeholder
            continue
        ctx.set_attr(grid, "frnom", frequency, label=net_id)
        if ctx.dry_run:
            continue
        if not bool(grid.IsCalcRelevant()):
            ierr = grid.Activate()
            if ierr:
                raise PFSessionError(
                    f"Activate() on DSO grid {net_id!r} returned {ierr}"
                )
            ctx.report.updated.append(f"{net_id}.Activate()")


def _grid_for_name(ctx: SyncContext, net_id: Optional[str]):
    if net_id is None:
        return ctx.grid
    grid = ctx.dso_grids.get(net_id)
    if grid is None and not ctx.dry_run:
        raise PFSessionError(
            f"DSO grid {net_id!r} is missing; call ensure_dso_grids first"
        )
    return grid


def _grid_for_bus(ctx: SyncContext, bus_idx: int):
    return _grid_for_name(ctx, ctx.bus_grid_names.get(int(bus_idx)))


def _grid_for_line(ctx: SyncContext, line_idx: int):
    return _grid_for_name(ctx, ctx.line_grid_names.get(int(line_idx)))


def _grid_for_trafo3w(ctx: SyncContext, trafo_idx: int):
    return _grid_for_name(ctx, ctx.trafo3w_grid_names.get(int(trafo_idx)))


# =====================================================================
#  Phase 'base' sync steps
# =====================================================================

def sync_buses(ctx: SyncContext) -> None:
    """Adopt/rename template terminals; create missing; delete stale."""
    model = ctx.doc["model"]
    by_name, by_ieee = _term_index_maps(ctx)

    # PF API queries return fresh wrapper objects per call, so claim
    # tracking must key on (template) loc_names, never on object identity.
    claimed_template_names = set()
    for key, rec in sorted(model["bus"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        target = ctx.names[("bus", idx)]
        vn = float(rec["vn_kv"])

        obj = by_name.get(target)
        if obj is None and idx <= 38 and (idx + 1) in by_ieee:
            obj = by_ieee[idx + 1]           # template 'Bus {idx+1:02d}'
            claimed_template_names.add(obj.loc_name)
            ctx.rename(obj, target)
        if obj is None:
            obj = ctx.create(_grid_for_bus(ctx, idx), "ElmTerm", target)
        ctx.term_alias[target] = obj
        if obj is None:                       # dry-run creation
            continue
        ctx.set_attr(obj, "uknom", vn, label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)

    # Stale template terminals (e.g. 'Bus 20', chain-collapsed) are
    # deleted in sync_finalise_deletions, after the branches that still
    # reference them are gone.  After a real run the claimed ones no
    # longer match the 'Bus NN' pattern (renamed); in a dry run the
    # claimed-name set provides the same information.
    ctx._stale_terms = [
        t for t in _network_all(ctx, "ElmTerm")
        if t.loc_name.startswith("Bus ")
        and t.loc_name not in claimed_template_names
    ]


def sync_lines(ctx: SyncContext) -> None:
    model = ctx.doc["model"]
    by_name, _ = _term_index_maps(ctx)

    # Template lines by IEEE endpoint pair (unordered).
    tmpl_by_pair: Dict[frozenset, Any] = {}
    lines_by_name: Dict[str, Any] = {}
    claimed_template_names = set()
    for ln in _network_all(ctx, "ElmLne"):
        lines_by_name[ln.loc_name] = ln
        if not ln.loc_name.startswith("Line "):
            continue                       # only template lines are matchable
        pair = frozenset(_endpoints_ieee(ctx, ln, ("bus1", "bus2")))
        if -1 not in pair:
            tmpl_by_pair[pair] = ln

    for key, rec in sorted(model["line"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        target = ctx.names[("line", idx)]
        pair = frozenset((int(rec["from_bus"]) + 1, int(rec["to_bus"]) + 1))

        obj = lines_by_name.get(target) or tmpl_by_pair.get(pair)
        if obj is not None and obj.loc_name.startswith("Line "):
            claimed_template_names.add(obj.loc_name)
        if obj is None:
            # pandapower keeps ratio-1.0 case39 branches as lines where the
            # template models a transformer (IEEE 23-36, G 07's step-up);
            # such lines are created outright and the template trafo falls
            # out as stale.
            obj = ctx.create(_grid_for_line(ctx, idx), "ElmLne", target)
            if obj is not None:
                f_term = _target_term(ctx, int(rec["from_bus"]))
                t_term = _target_term(ctx, int(rec["to_bus"]))
                cub_f = ctx.cubicle(f_term, f"{target}_f")
                cub_t = ctx.cubicle(t_term, f"{target}_t")
                obj.SetAttribute("bus1", cub_f)
                obj.SetAttribute("bus2", cub_t)
                typ = ctx.create(_types_folder(ctx), "TypLne",
                                 f"TYP_{target}")
                typ.SetAttribute("frnom", float(ctx.doc["model"]["net"]["f_hz"]))
                typ.SetAttribute("nlnph", 3)
                obj.SetAttribute("typ_id", typ)
        if obj is None:                        # dry-run creation
            continue
        ctx.rename(obj, target)
        _reconnect(ctx, obj, "bus1", int(rec["from_bus"]),
                   f"{target}_f", target)
        _reconnect(ctx, obj, "bus2", int(rec["to_bus"]),
                   f"{target}_t", target)

        ctx.set_attr(obj, "dline", float(rec["length_km"]), label=target)
        ctx.set_attr(obj, "nlnum", int(rec["parallel"]), label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)

        typ = obj.typ_id
        if typ is None:
            raise PFSessionError(f"{target}: template line has no TypLne")
        tlabel = f"{target}(typ)"
        from_bus = int(rec["from_bus"])
        to_bus = int(rec["to_bus"])
        vn_from = float(model["bus"][str(from_bus)]["vn_kv"])
        vn_to = float(model["bus"][str(to_bus)]["vn_kv"])
        if not math.isclose(vn_from, vn_to, rel_tol=0.0, abs_tol=1e-9):
            raise PFSessionError(
                f"{target}: line endpoints have unequal nominal voltages "
                f"{vn_from} and {vn_to} kV"
            )
        ctx.set_attr(typ, "uline", vn_from, label=tlabel)
        ctx.set_attr(typ, "sline", float(rec["max_i_ka"]), label=tlabel)
        ctx.set_attr(typ, "rline", float(rec["r_ohm_per_km"]), label=tlabel)
        ctx.set_attr(typ, "xline", float(rec["x_ohm_per_km"]), label=tlabel)
        c_uf = float(rec["c_nf_per_km"]) / 1000.0
        ctx.set_attr(typ, "cline", c_uf, label=tlabel)
        f_hz = float(ctx.doc["model"]["net"]["f_hz"])
        ctx.set_attr(typ, "bline", 2.0 * math.pi * f_hz * c_uf,
                     label=tlabel)
        ctx.set_attr(typ, "gline", float(rec["g_us_per_km"]), label=tlabel)

    ctx._stale_lines = [
        ln for ln in _network_all(ctx, "ElmLne")
        if ln.loc_name.startswith("Line ")
        and ln.loc_name not in claimed_template_names
    ]


def _types_folder(ctx: SyncContext):
    equip = ctx.app.GetProjectFolder("equip")
    for sub in equip.GetContents("qOFO Types.IntPrjfolder", False) or []:
        return sub
    return ctx.create(equip, "IntPrjfolder", "qOFO Types")


def _push_trafo2_type(ctx: SyncContext, typ, rec: Mapping[str, Any],
                      label: str) -> None:
    ctx.set_attr(typ, "strn", float(rec["sn_mva"]), label=label)
    ctx.set_attr(typ, "utrn_h", float(rec["vn_hv_kv"]), label=label)
    ctx.set_attr(typ, "utrn_l", float(rec["vn_lv_kv"]), label=label)
    # PF 2025 stores the positive-sequence impedance as the pu pair
    # (r1pu, x1pu); the percent fields uktr/uktrr are derived read-only
    # views -- direct writes to them are silently ignored (probed
    # 2026-07-17).  vk/vkr [%] convert exactly:
    vk = float(rec["vk_percent"])
    vkr = float(rec["vkr_percent"])
    if vkr > vk:
        raise PFSessionError(f"{label}: vkr {vkr} > vk {vk}")
    r1 = vkr / 100.0
    x1 = math.sqrt((vk / 100.0) ** 2 - r1 ** 2)
    ctx.set_attr(typ, "r1pu", r1, label=label)
    ctx.set_attr(typ, "x1pu", x1, label=label)
    ctx.set_attr(typ, "pfe", float(rec["pfe_kw"]), label=label)
    ctx.set_attr(typ, "curmg", float(rec["i0_percent"]), label=label)
    ctx.set_attr(typ, "nt2ag", float(rec["shift_degree"]) / 30.0,
                 label=label)
    # Tap changer: pandapower tap_side 'hv' -> PF tap_side 0 (HV).
    if rec.get("tap_side") is not None:
        if rec["tap_side"] != "hv":
            raise PFSessionError(
                f"{label}: tap_side {rec['tap_side']!r} not supported "
                f"(base-phase machine trafos are HV-side)"
            )
        ctx.set_attr(typ, "tap_side", 0, label=label)
        ctx.set_attr(typ, "dutap", float(rec["tap_step_percent"]),
                     label=label)
        ctx.set_attr(typ, "phitr", 0.0, label=label)
        ctx.set_attr(typ, "nntap0", int(rec["tap_neutral"]), label=label)
        ctx.set_attr(typ, "ntpmn", int(rec["tap_min"]), label=label)
        ctx.set_attr(typ, "ntpmx", int(rec["tap_max"]), label=label)


def sync_trafos(ctx: SyncContext) -> None:
    model = ctx.doc["model"]
    by_name, _ = _term_index_maps(ctx)

    tmpl_by_pair: Dict[Tuple[int, int], Any] = {}
    tr_by_name: Dict[str, Any] = {}
    for tr in _network_all(ctx, "ElmTr2"):
        tr_by_name[tr.loc_name] = tr
        if not tr.loc_name.startswith("Trf "):
            continue                       # only template trafos are matchable
        hv, lv = _endpoints_ieee(ctx, tr, ("bushv", "buslv"))
        if hv != -1 and lv != -1:
            tmpl_by_pair[(hv, lv)] = tr

    claimed_template_names = set()
    for key, rec in sorted(model["trafo"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        target = ctx.names[("trafo", idx)]
        pair = (int(rec["hv_bus"]) + 1, int(rec["lv_bus"]) + 1)

        obj = tr_by_name.get(target) or tmpl_by_pair.get(pair) \
            or tmpl_by_pair.get((pair[1], pair[0]))
        if obj is not None and obj.loc_name.startswith("Trf "):
            claimed_template_names.add(obj.loc_name)
        if obj is None:
            # New transformer (chain-collapse replacement, G1 step-up).
            obj = ctx.create(ctx.grid, "ElmTr2", target)
            if obj is not None:
                hv_term = _target_term(ctx, int(rec["hv_bus"]))
                lv_term = _target_term(ctx, int(rec["lv_bus"]))
                cub_h = ctx.cubicle(hv_term, f"{target}_h")
                cub_l = ctx.cubicle(lv_term, f"{target}_l")
                obj.SetAttribute("bushv", cub_h)
                obj.SetAttribute("buslv", cub_l)
                typ = ctx.create(_types_folder(ctx), "TypTr2",
                                 f"TYP_{target}")
                obj.SetAttribute("typ_id", typ)
        if obj is None:                        # dry-run creation
            continue
        ctx.rename(obj, target)

        # ElmTr2 endpoints are checked, not re-pointed.  Template adoption
        # matches on the unordered endpoint pair, so an adopted unit may carry
        # the HV/LV orientation reversed with respect to pandapower; blindly
        # reconnecting would move the tap changer to the other winding.  The
        # unordered comparison still catches a genuine bus reassignment.
        have = set(_endpoint_term_names(obj, ("bushv", "buslv")))
        want = {ctx.names[("bus", int(rec["hv_bus"]))],
                ctx.names[("bus", int(rec["lv_bus"]))]}
        if have != want:
            raise PFSessionError(
                f"{target}: connected to {sorted(map(str, have))} but the "
                f"snapshot places it between {sorted(want)}; 2W endpoints are "
                f"not re-pointed automatically (orientation ambiguity) -- "
                f"delete the transformer and re-run the sync"
            )

        typ = obj.typ_id
        if typ is None:
            raise PFSessionError(f"{target}: no TypTr2 attached")
        _push_trafo2_type(ctx, typ, rec, f"{target}(typ)")

        tap_pos = rec.get("tap_pos")
        ctx.set_attr(obj, "nntap",
                     int(tap_pos) if tap_pos is not None else 0,
                     label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)
        if int(rec["parallel"]) != 1:
            raise PFSessionError(f"{target}: parallel != 1 unsupported")

    ctx._stale_trafos = [
        t for t in _network_all(ctx, "ElmTr2")
        if t.loc_name.startswith("Trf ")
        and t.loc_name not in claimed_template_names
    ]


def sync_loads(ctx: SyncContext) -> None:
    model = ctx.doc["model"]
    by_name, _ = _term_index_maps(ctx)

    # The template's shared load type is exactly the anchored-ZIP model
    # (P ~ u^1 via bP/kpu1, Q ~ u^2 via cQ/kqu): reuse it for every load.
    lodtypes = [t for t in get_all(ctx.app, "TypLod")
                if t.loc_name == "General Load Type"]
    if len(lodtypes) != 1:
        raise PFSessionError(
            f"Expected exactly one 'General Load Type', found "
            f"{[t.loc_name for t in lodtypes]}"
        )
    zip_type = lodtypes[0]
    for attr, want in (("aP", 0.0), ("bP", 1.0), ("cP", 0.0),
                       ("kpu1", 1.0), ("aQ", 0.0), ("bQ", 0.0),
                       ("cQ", 1.0), ("kqu", 2.0)):
        have = zip_type.GetAttribute(attr)
        if not math.isclose(float(have), want, abs_tol=1e-9):
            raise PFSessionError(
                f"'General Load Type'.{attr} = {have!r}, expected {want} "
                f"-- template load model no longer matches the anchored "
                f"ZIP convention"
            )

    lod_by_name = {ld.loc_name: ld for ld in _network_all(ctx, "ElmLod")}

    for key, rec in sorted(model["load"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        target = ctx.names[("load", idx)]
        if float(rec["scaling"]) != 1.0:
            raise PFSessionError(f"{target}: scaling != 1 unsupported")

        obj = lod_by_name.get(target)
        if obj is None:
            obj = ctx.create(
                _grid_for_bus(ctx, int(rec["bus"])), "ElmLod", target
            )
            if obj is not None:
                term = _target_term(ctx, int(rec["bus"]))
                cub = ctx.cubicle(term, target)
                obj.SetAttribute("bus1", cub)
        if obj is None:                        # dry-run
            continue
        obj.SetAttribute("typ_id", zip_type)
        ctx.set_attr(obj, "plini", float(rec["p_mw"]), label=target)
        ctx.set_attr(obj, "qlini", float(rec["q_mvar"]), label=target)
        ctx.set_attr(obj, "scale0", 1.0, label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)

    # Template loads ('Load NN') are superseded by the snapshot rows.
    ctx._stale_loads = [
        ld for ld in _network_all(ctx, "ElmLod")
        if ld.loc_name.startswith("Load ")
    ]


def sync_gen_dispatch(ctx: SyncContext, *,
                      rebase_terminals: bool = True) -> None:
    """Template-owned machines: dispatch, setpoints, reference flag, and
    (base phase only) terminal rebase.

    ``rebase_terminals=False`` (wind_replace/full deltas) skips the rated-
    voltage and reconnection edits -- the base sync already placed every
    retained machine on its 10.5 kV terminal, and repeating those inside a
    variation would record spurious cubicle deltas.
    """
    if not TEMPLATE_NAMES_VERIFIED:
        raise PFSessionError(
            "pf/naming.py TEMPLATE_NAMES_VERIFIED is False -- verify the "
            "machine names first (Gate 1)"
        )
    model = ctx.doc["model"]
    solution = ctx.doc["solution"]
    sym_by_name = {m.loc_name: m for m in _network_all(ctx, "ElmSym")}

    for key, rec in sorted(model["gen"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        tpl_name = machine_template_name(rec)
        mach = sym_by_name.get(tpl_name)
        if mach is None:
            raise PFSessionError(
                f"gen[{idx}]: template machine {tpl_name!r} not found"
            )
        label = tpl_name
        ngnum = int(mach.GetAttribute("ngnum"))

        # Dispatch: converged post-distributed-slack P (solution), per
        # parallel machine.  The PF reference machine then only covers the
        # PF-vs-pandapower loss mismatch.
        p_sol = float(solution["gen"][key]["p_mw"])
        ctx.set_attr(mach, "pgini", p_sol / ngnum, label=label)
        ctx.set_attr(mach, "usetp", float(rec["vm_pu"]), label=label)
        ctx.set_attr(mach, "av_mode", "constv", label=label)
        ctx.set_attr(mach, "outserv", 0 if rec["in_service"] else 1,
                     label=label)
        ctx.set_attr(mach, "ip_ctrl", 1 if rec["slack"] else 0, label=label)

        if not rebase_terminals:
            continue

        # Terminal rebase: machine sits at the snapshot terminal bus with
        # the snapshot rated voltage (10.5 kV).
        term_idx = int(rec["bus"])
        term_name = ctx.names[("bus", term_idx)]
        vn = float(model["bus"][str(term_idx)]["vn_kv"])
        typ = mach.typ_id
        if typ is None:
            raise PFSessionError(f"{label}: no TypSym attached")
        ctx.set_attr(typ, "ugn", vn, label=f"{label}(typ)")

        cub = mach.GetAttribute("bus1")
        current_term = cub.cterm if cub is not None else None
        if current_term is None or current_term.loc_name != term_name:
            target_term = _target_term(ctx, term_idx)
            ctx.report.updated.append(
                f"{label}.bus1: {current_term.loc_name if current_term else None!r} "
                f"-> {term_name!r}"
            )
            if not ctx.dry_run:
                new_cub = ctx.cubicle(target_term, f"{label.replace(' ', '')}")
                mach.SetAttribute("bus1", new_cub)


def delete_stale_sgens(ctx: SyncContext) -> None:
    """Delete static-generator rename orphans found by the sgen sync.

    Split out of :func:`sync_finalise_deletions` because that finaliser runs
    only in the ``base`` phase, while static generators are synced in
    ``wind_replace`` and ``full``.

    A park's dependants are named after it, so they have to go with it: the
    ``ElmStactrl`` write handle (load-flow relevant -- an orphan keeps
    controlling its old cubicle) and, once ``pf/wecc_apply.py`` has run, the
    ``ElmComp`` whose frame slots point at the deleted generator.  Both were
    left behind before 2026-07-29, when the ``hv_buses`` reorder renamed the
    eight coupling-bus STATCOM parks for the first time.
    """
    stale = getattr(ctx, "_stale_sgens", [])
    if stale:
        ctrls = {c.loc_name: c for c in _network_all(ctx, "ElmStactrl")}
        comps = {c.loc_name: c for c in _network_all(ctx, "ElmComp")}
    for g, term_name, owner in stale:
        park = g.loc_name
        ctrl = ctrls.get(controller_name(park))
        if ctrl is not None:
            ctx.delete(ctrl, label=(f"ElmStactrl {ctrl.loc_name!r} "
                                    f"(controller of deleted {park!r})"))
        comp = comps.get(f"{RMS_COMPOSITE_PREFIX}{park}")
        if comp is not None:
            ctx.delete(comp, label=(f"ElmComp {comp.loc_name!r} "
                                    f"(RMS model of deleted {park!r})"))
        ctx.delete(g, label=(f"ElmGenstat {park!r} (rename orphan: "
                             f"shares terminal {term_name!r} with {owner!r})"))
    ctx._stale_sgens = []


def _psym_has_live_member(ctrl) -> bool:
    """True unless every machine in ``ctrl.psym`` is gone.

    A PF handle to a deleted object survives in the owning list but raises on
    attribute access, so liveness is probed rather than tested.  Anything that
    cannot be read is reported live: this predicate only ever authorises a
    deletion, so it must fail safe.
    """
    try:
        members = ctrl.GetAttribute("psym") or []
    except Exception:                                          # noqa: BLE001
        return True
    for m in members:
        if m is None:
            continue
        try:
            if hasattr(m, "IsDeleted") and m.IsDeleted():
                continue
            _ = m.loc_name                     # dead handles raise here
            return True
        except Exception:                                      # noqa: BLE001
            continue
    return False


def delete_orphan_station_controllers(ctx: SyncContext) -> None:
    """Delete ``ElmStactrl`` objects left behind by an earlier park rename.

    Found 2026-07-29 while auditing the DSO geometry re-sync: twelve
    ``CTRL_DER_DSO_*`` controllers from the 2026-07-21 ``WPC_* -> DER_*`` role
    reclassification were still in service (``i_ctrl = 1``, non-zero
    ``qsetp``) with a ``psym`` list holding nothing but a dead handle.  They
    do not disturb the load flow -- a station controller with no machine has
    nothing to actuate, and Gate C passed with them present -- but they are
    live objects referencing deleted ones.

    Two conditions must hold, so the sweep cannot misfire during the
    ``wind_replace`` phase (where the snapshot names only the TSO parks while
    the DSO controllers and their parks are both alive):

    1. the name is ``CTRL_<park>`` for a park this snapshot does not contain;
    2. no machine in ``psym`` still exists.
    """
    expected = {
        controller_name(ctx.names[("sgen", int(key))])
        for key in ctx.doc["model"]["sgen"]
    }
    for ctrl in _network_all(ctx, "ElmStactrl"):
        name = ctrl.loc_name
        if not name.startswith("CTRL_") or name in expected:
            continue
        if _psym_has_live_member(ctrl):
            continue
        ctx.delete(ctrl, label=(f"ElmStactrl {name!r} (orphan: every machine "
                                f"in psym has been deleted)"))


def sync_finalise_deletions(ctx: SyncContext) -> None:
    """Delete superseded template objects (branches first, then buses)."""
    for ln in getattr(ctx, "_stale_lines", []):
        ctx.delete(ln, label=f"ElmLne {ln.loc_name!r} (superseded)")
    for tr in getattr(ctx, "_stale_trafos", []):
        ctx.delete(tr, label=f"ElmTr2 {tr.loc_name!r} (superseded)")
    for ld in getattr(ctx, "_stale_loads", []):
        ctx.delete(ld, label=f"ElmLod {ld.loc_name!r} (template load)")
    delete_stale_sgens(ctx)
    for t in getattr(ctx, "_stale_terms", []):
        ctx.delete(t, label=f"ElmTerm {t.loc_name!r} (not in snapshot)")


def sync_base(ctx: SyncContext) -> None:
    model = ctx.doc["model"]
    for tbl in ("sgen", "shunt", "trafo3w"):
        if model[tbl]:
            raise PFSessionError(
                f"Phase 'base' expects an empty {tbl!r} table; snapshot has "
                f"{len(model[tbl])} rows -- wrong snapshot for this phase?"
            )
    sync_buses(ctx)
    sync_lines(ctx)
    sync_trafos(ctx)
    sync_loads(ctx)
    sync_gen_dispatch(ctx)
    sync_finalise_deletions(ctx)


# =====================================================================
#  Phase 'wind_replace' delta (recorded inside the variation)
# =====================================================================

def _outserv_parent_load_copies(ctx: SyncContext) -> None:
    """Disable lower-layer load copies superseded by TN_AUX load rows.

    DSO auxiliary loads are first created in the ``full`` variation and have
    no inherited parent-bus copy; they are deliberately skipped here.

    A wind snapshot keeps each pandapower load index but moves the affected
    row to an internal auxiliary bus.  Its deterministic PF name therefore
    changes from ``..._b{parent}`` to ``..._b{aux}``.  ``sync_loads`` creates
    the auxiliary copy in the active variation; this helper switches off the
    still-present base copy so exactly one electrical load remains active.
    """
    aux_parent = dict(zip(
        (int(b) for b in ctx.meta.internal_aux_bus_indices),
        (int(b) for b in ctx.meta.internal_aux_parent_buses),
    ))
    if len(aux_parent) != len(ctx.meta.internal_aux_bus_indices):
        raise PFSessionError("Duplicate internal auxiliary bus in metadata")

    loads_by_name = {ld.loc_name: ld for ld in _network_all(ctx, "ElmLod")}
    for key, rec in sorted(
        ctx.doc["model"]["load"].items(), key=lambda kv: int(kv[0])
    ):
        aux_bus = int(rec["bus"])
        if aux_bus not in aux_parent:
            continue
        aux_rec = ctx.doc["model"]["bus"][str(aux_bus)]
        if aux_rec.get("subnet") != "TN_AUX":
            continue
        idx = int(key)
        parent = aux_parent[aux_bus]
        kind = "var" if rec.get("profile_p") else "const"
        base_name = f"TN_load{idx}_{kind}_b{parent}"
        base_load = loads_by_name.get(base_name)
        if base_load is None:
            raise PFSessionError(
                f"Auxiliary load {ctx.names[('load', idx)]!r}: base copy "
                f"{base_name!r} not found; run the base sync first"
            )
        ctx.set_attr(base_load, "outserv", 1, label=base_name)

def _outserv_absent_snapshot_loads(ctx: SyncContext) -> None:
    """Disable inherited script-owned loads absent from this snapshot.

    The full builder removes seven TN profile-half rows whose demand is
    represented by the DSO underlays.  In the layered PF architecture those
    wind-layer objects still exist and must be switched off in ``full``;
    deleting them would prevent restoration of the validated wind state.
    """
    expected = {
        ctx.names[("load", int(key))]
        for key in ctx.doc["model"]["load"]
    }
    for load in _network_all(ctx, "ElmLod"):
        script_owned = load.loc_name.startswith(("TN_load", "DSO_"))
        if script_owned and load.loc_name not in expected:
            ctx.set_attr(load, "outserv", 1, label=load.loc_name)


def _outserv_removed_machines(ctx: SyncContext) -> None:
    """Out-of-service the removed machines and their step-up chains.

    The removed generators (snapshot ``removed_generators``) are template
    ElmSym objects; each connects via its ``bus1`` cubicle to a 10.5 kV
    GT terminal fed by one machine ElmTr2.  All three (machine, trafo,
    terminal) and any loads stranded on the terminal are set out of service
    so the wind_replace state matches the pandapower model, where they are
    deleted.  Navigation is topological, so no base snapshot is needed.
    """
    from pf.naming import TEMPLATE_MACHINE_NAMES

    sym_by_name = {m.loc_name: m for m in _network_all(ctx, "ElmSym")}
    tr2 = _network_all(ctx, "ElmTr2")
    loads = _network_all(ctx, "ElmLod")

    for entry in ctx.doc["removed_generators"]:
        label = entry["label"]
        tpl = TEMPLATE_MACHINE_NAMES.get(label)
        if tpl is None:
            raise PFSessionError(f"no template machine mapping for {label!r}")
        mach = sym_by_name.get(tpl)
        if mach is None:
            raise PFSessionError(f"removed machine {tpl!r} not found in PF")

        cub = mach.GetAttribute("bus1")
        term = cub.cterm if cub is not None else None
        tname = term.loc_name if term is not None else None

        ctx.set_attr(mach, "outserv", 1, label=tpl)

        # Machine trafo(s) feeding the terminal.
        for tr in tr2:
            for side in ("bushv", "buslv"):
                c = tr.GetAttribute(side)
                if c is not None and c.cterm is not None \
                        and c.cterm.loc_name == tname:
                    ctx.set_attr(tr, "outserv", 1, label=tr.loc_name)

        # Loads stranded on the terminal (case39 bus-31 load on G 02).
        for ld in loads:
            c = ld.GetAttribute("bus1")
            if c is not None and c.cterm is not None \
                    and c.cterm.loc_name == tname:
                ctx.set_attr(ld, "outserv", 1, label=ld.loc_name)

        # The terminal itself (silences the isolated-node warning).
        if term is not None:
            ctx.set_attr(term, "outserv", 1, label=tname)


def _sync_station_controller(
    ctx: SyncContext,
    obj,
    rec: Mapping[str, Any],
    target: str,
    ctrl_by_name: Dict[str, Any],
) -> None:
    """Attach one operational constant-Q ``ElmStactrl`` to a static source.

    PF 2025 SP4 was live-probed on 2026-07-19: ``i_ctrl=1`` selects reactive
    power control and ``qu_char=0`` selects Const. Q.  With ``p_cub`` set to
    the generator cubicle, changes of ``qsetp`` track the generator injection
    to float32 precision.  This is the later OFO write handle.
    """
    name = controller_name(target)
    ctrl = ctrl_by_name.get(name)
    if ctrl is None:
        ctrl = ctx.create(
            _grid_for_bus(ctx, int(rec["bus"])), "ElmStactrl", name
        )
        if ctrl is not None:
            ctrl_by_name[name] = ctrl
    if ctrl is None:                           # dry-run
        return

    cub = obj.GetAttribute("bus1")
    if cub is None:
        raise PFSessionError(f"{target}: no connection cubicle for controller")
    ctx.set_attr(ctrl, "i_ctrl", 1, label=name)
    ctx.set_attr(ctrl, "qu_char", 0, label=name)
    ctx.set_attr(ctrl, "qsetp", float(rec["q_mvar"]), label=name)
    ctx.set_attr(ctrl, "p_cub", cub, label=name)
    ctx.set_attr(ctrl, "psym", [obj], label=name)
    ctx.set_attr(ctrl, "Srated", float(rec["sn_mva"]), label=name)
    ctx.set_attr(ctrl, "outserv", 0 if rec["in_service"] else 1, label=name)
    ctx.set_attr(obj, "c_pstac", ctrl, label=target)


def _sync_static_generators(
    ctx: SyncContext, *, allowed_roles: Tuple[str, ...]
) -> None:
    """Create/adopt static sources and their constant-Q write handles."""
    model = ctx.doc["model"]
    allowed = set(allowed_roles)
    gs_by_name = {g.loc_name: g for g in _network_all(ctx, "ElmGenstat")}
    ctrl_by_name = {c.loc_name: c for c in _network_all(ctx, "ElmStactrl")}
    claimed: set = set()
    claimed_terms: Dict[str, str] = {}          # terminal name -> park name

    for key, rec in sorted(model["sgen"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        if rec["role"] not in allowed:
            raise PFSessionError(
                f"sgen[{idx}] role {rec['role']!r} not in allowed roles "
                f"{sorted(allowed)!r}"
            )
        if float(rec["scaling"]) != 1.0:
            raise PFSessionError(f"sgen[{idx}]: scaling != 1 unsupported")
        target = ctx.names[("sgen", idx)]
        obj = gs_by_name.get(target)
        if obj is None:
            obj = ctx.create(
                _grid_for_bus(ctx, int(rec["bus"])), "ElmGenstat", target
            )
            if obj is not None:
                term = _target_term(ctx, int(rec["bus"]))
                cub = ctx.cubicle(term, target)
                obj.SetAttribute("bus1", cub)
                gs_by_name[target] = obj
        if obj is None:                        # dry-run
            continue
        sn = float(rec["sn_mva"])
        ctx.set_attr(obj, "sgn", sn, label=target)
        ctx.set_attr(obj, "cosn", 1.0, label=target)
        ctx.set_attr(obj, "ngnum", 1, label=target)
        ctx.set_attr(obj, "av_mode", "constq", label=target)
        ctx.set_attr(obj, "pgini", float(rec["p_mw"]), label=target)
        ctx.set_attr(obj, "qgini", float(rec["q_mvar"]), label=target)
        # q_min/q_max are per-unit on ElmGenstat.sgn (confirmed from the
        # PF 2025 class schema), not Mvar values.
        ctx.set_attr(obj, "q_min", -1.0, label=target)
        ctx.set_attr(obj, "q_max", 1.0, label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)
        _sync_station_controller(ctx, obj, rec, target, ctrl_by_name)
        claimed.add(target)
        cub = obj.GetAttribute("bus1")
        if cub is not None and cub.cterm is not None:
            claimed_terms[cub.cterm.loc_name] = target

    # Rename orphans: a genstat that the snapshot does not name, sharing a
    # terminal with one that it does.  Without this the park is injected
    # TWICE -- on 2026-07-21 a role reclassification renamed 12 coupling
    # parks WPC_* -> DER_*, and the orphans silently added a phantom
    # 480 MW / 78 Mvar to every RMS run.
    #
    # Deliberately NOT "every genstat missing from the snapshot":
    # ``_network_all`` walks all Network Data including inactive islands, so
    # during the wind_replace phase (allowed_roles=('TSO-WP',)) that rule
    # would delete all 40 not-yet-synced DSO parks.  Sharing a terminal with
    # a claimed park is the precise signature of a rename orphan and cannot
    # match a legitimately separate one.
    stale = []
    for name, g in gs_by_name.items():
        if name in claimed:
            continue
        cub = g.GetAttribute("bus1")
        term = cub.cterm if cub is not None else None
        if term is None:
            continue
        owner = claimed_terms.get(term.loc_name)
        if owner is not None:
            stale.append((g, term.loc_name, owner))
    ctx._stale_sgens = stale


def _sync_retained_taps(ctx: SyncContext) -> None:
    """Push the wind_replace OLTC tap positions onto the retained trafos.

    The removed machines shift the operating point, so the Phase-10.1/10.2
    OLTC init lands several machine 2W taps at different positions than in
    the base snapshot (e.g. trafos 2 and 6).  Only the tap position
    (``nntap``, an ElmTr2 attribute) changes; the type impedance is
    identical, so this records a minimal variation delta.
    """
    model = ctx.doc["model"]
    tr_by_name = {t.loc_name: t for t in _network_all(ctx, "ElmTr2")}
    for key, rec in sorted(model["trafo"].items(), key=lambda kv: int(kv[0])):
        target = ctx.names[("trafo", int(key))]
        obj = tr_by_name.get(target)
        if obj is None:
            raise PFSessionError(f"{target}: retained trafo missing in PF")
        tap_pos = rec.get("tap_pos")
        ctx.set_attr(obj, "nntap",
                     int(tap_pos) if tap_pos is not None else 0,
                     label=target)


def sync_wind_replace(ctx: SyncContext) -> None:
    """Apply the wind_replace delta.  The caller must have activated the
    variation so every change below is recorded there, not on the base."""
    model = ctx.doc["model"]
    if not model["sgen"]:
        raise PFSessionError(
            "wind_replace snapshot has no sgen rows -- wrong snapshot?"
        )
    if model["trafo3w"] or model["shunt"]:
        raise PFSessionError(
            "wind_replace snapshot has trafo3w/shunt rows -- that is the "
            "'full' phase, not wind_replace"
        )
    aux_lengths = {
        len(ctx.meta.internal_aux_bus_indices),
        len(ctx.meta.internal_aux_parent_buses),
        len(ctx.meta.internal_aux_line_indices),
    }
    if len(aux_lengths) != 1:
        raise PFSessionError(
            "Internal auxiliary metadata lists have different lengths"
        )
    if ctx.meta.internal_aux_bus_indices:
        # Reuse the base synchronisers inside the active variation.  Existing
        # TN objects compare unchanged; only the auxiliary terminals, links,
        # and moved load copies produce variation deltas.
        sync_buses(ctx)
        sync_lines(ctx)
        sync_loads(ctx)
        _outserv_parent_load_copies(ctx)
    # Retained machines carry a different (6-machine) dispatch; push it.
    sync_gen_dispatch(ctx, rebase_terminals=False)
    _sync_retained_taps(ctx)
    _outserv_removed_machines(ctx)
    _sync_static_generators(ctx, allowed_roles=("TSO-WP",))
    delete_stale_sgens(ctx)
    delete_orphan_station_controllers(ctx)


# =====================================================================
#  Phase 'full' delta: wind replacement plus four DSO underlays
# =====================================================================

def _push_trafo3w_type(
    ctx: SyncContext, typ, rec: Mapping[str, Any], label: str
) -> None:
    impedances = {}
    for side in ("h", "m", "l"):
        side_key = {"h": "hv", "m": "mv", "l": "lv"}[side]
        vk = float(rec[f"vk_{side_key}_percent"])
        vkr = float(rec[f"vkr_{side_key}_percent"])
        if vkr > vk:
            raise PFSessionError(f"{label}: {side_key} vkr {vkr} > vk {vk}")
        sn_mva = float(rec[f"sn_{side_key}_mva"])
        pair_sides = {
            "h": ("hv", "mv"),
            "m": ("mv", "lv"),
            "l": ("lv", "hv"),
        }[side]
        pair_base_mva = min(
            float(rec[f"sn_{pair_side}_mva"])
            for pair_side in pair_sides
        )
        pcu_kw = vkr / 100.0 * pair_base_mva * 1000.0
        impedances[side] = (vk, pcu_kw)
        ctx.set_attr(
            typ, f"strn3_{side}", sn_mva,
            label=label,
        )
        ctx.set_attr(
            typ, f"utrn3_{side}", float(rec[f"vn_{side_key}_kv"]),
            label=label,
        )

    ctx.set_attr(typ, "pfe", float(rec["pfe_kw"]), label=label)
    ctx.set_attr(typ, "curm3", float(rec["i0_percent"]), label=label)
    for side in ("h", "m", "l"):
        ctx.set_attr(typ, f"tr3cn_{side}", "YN", label=label)
    ctx.set_attr(typ, "nt3ag_h", 0.0, label=label)
    ctx.set_attr(
        typ, "nt3ag_m", float(rec["shift_mv_degree"]) / 30.0,
        label=label,
    )
    ctx.set_attr(
        typ, "nt3ag_l", float(rec["shift_lv_degree"]) / 30.0,
        label=label,
    )

    if rec.get("tap_side") != "hv":
        raise PFSessionError(
            f"{label}: only the specified HV-side 3W OLTC is supported"
        )
    if rec.get("tap_changer_type") != "Ratio":
        raise PFSessionError(f"{label}: only Ratio tap changers are supported")
    if bool(rec.get("tap_at_star_point")):
        raise PFSessionError(f"{label}: star-point tap modelling is unsupported")
    # PowerFactory 2025 enum: 1 = tap modelled at the winding terminal,
    # 0 = tap modelled at the star point.  The winding itself is selected by
    # the separate _h/_m/_l fields.  pandapower tap_at_star_point=False
    # therefore maps to PF itapos=1.  This placement matters because it
    # determines which side of the winding impedance sees the tap ratio.
    ctx.set_attr(typ, "itapos", 1, label=label)
    ctx.set_attr(typ, "n3tmn_h", int(rec["tap_min"]), label=label)
    ctx.set_attr(typ, "n3tmx_h", int(rec["tap_max"]), label=label)
    ctx.set_attr(typ, "n3tp0_h", int(rec["tap_neutral"]), label=label)
    ctx.set_attr(typ, "du3tp_h", float(rec["tap_step_percent"]),
                 label=label)

    # ``uktrr3_*`` and ``r1pu_*``/``x1pu_*`` are derived fields.  PF 2025
    # persists uk in percent and copper loss in kW; the latter reproduces the
    # snapshot vkr via Pcu = vkr/100 * min(Sn_pair).
    for side, (vk, pcu_kw) in impedances.items():
        ctx.set_attr(typ, f"uktr3_{side}", vk, label=label)
        ctx.set_attr(typ, f"pcut3_{side}", pcu_kw, label=label)


def sync_trafo3w(ctx: SyncContext) -> None:
    model = ctx.doc["model"]
    by_name = {tr.loc_name: tr for tr in _network_all(ctx, "ElmTr3")}
    for key, rec in sorted(model["trafo3w"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        target = ctx.names[("trafo3w", idx)]
        obj = by_name.get(target)
        if obj is None:
            obj = ctx.create(_grid_for_trafo3w(ctx, idx), "ElmTr3", target)
            if obj is not None:
                for attr, bus_field, tag in (
                    ("bushv", "hv_bus", "h"),
                    ("busmv", "mv_bus", "m"),
                    ("buslv", "lv_bus", "l"),
                ):
                    term = _target_term(ctx, int(rec[bus_field]))
                    cub = ctx.cubicle(term, f"{target}_{tag}")
                    obj.SetAttribute(attr, cub)
                typ = ctx.create(_types_folder(ctx), "TypTr3", f"TYP_{target}")
                obj.SetAttribute("typ_id", typ)
                by_name[target] = obj
        if obj is None:                        # dry-run
            continue
        # The 3W name carries no bus, so a coupler that changes winding bus
        # must be re-pointed explicitly (hv_buses reorder, 2026-07-29).
        for attr, bus_field, tag in (
            ("bushv", "hv_bus", "h"),
            ("busmv", "mv_bus", "m"),
            ("buslv", "lv_bus", "l"),
        ):
            _reconnect(ctx, obj, attr, int(rec[bus_field]),
                       f"{target}_{tag}", target)
        typ = obj.GetAttribute("typ_id")
        if typ is None:
            raise PFSessionError(f"{target}: no TypTr3 attached")
        _push_trafo3w_type(ctx, typ, rec, f"{target}(typ)")
        ctx.set_attr(obj, "n3tap_h", int(rec["tap_pos"]), label=target)
        ctx.set_attr(obj, "ntrcn2", 0, label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)


def sync_shunts(ctx: SyncContext) -> None:
    model = ctx.doc["model"]
    indices = [int(v) for v in ctx.meta.tso_tertiary_shunt_indices]
    kinds = [str(v) for v in ctx.meta.tso_tertiary_shunt_kinds]
    q_steps = [float(v) for v in ctx.meta.tso_tertiary_shunt_q_steps_mvar]
    if not (len(indices) == len(kinds) == len(q_steps)):
        raise PFSessionError("TSO tertiary shunt metadata lengths differ")
    kind_by_idx = dict(zip(indices, kinds))
    step_by_idx = dict(zip(indices, q_steps))
    by_name = {sh.loc_name: sh for sh in _network_all(ctx, "ElmShnt")}

    for key, rec in sorted(model["shunt"].items(), key=lambda kv: int(kv[0])):
        idx = int(key)
        target = ctx.names[("shunt", idx)]
        kind = kind_by_idx.get(idx)
        if kind not in {"MSC", "MSR"}:
            raise PFSessionError(f"{target}: unsupported shunt kind {kind!r}")
        q_step = step_by_idx[idx]
        if not math.isclose(abs(float(rec["q_mvar"])), q_step,
                            rel_tol=0.0, abs_tol=1e-9):
            raise PFSessionError(
                f"{target}: |q_mvar| does not equal metadata step {q_step}"
            )
        if float(rec["p_mw"]) != 0.0:
            raise PFSessionError(f"{target}: non-zero shunt p_mw unsupported")

        obj = by_name.get(target)
        if obj is None:
            obj = ctx.create(
                _grid_for_bus(ctx, int(rec["bus"])), "ElmShnt", target
            )
            if obj is not None:
                term = _target_term(ctx, int(rec["bus"]))
                cub = ctx.cubicle(term, target)
                obj.SetAttribute("bus1", cub)
                by_name[target] = obj
        if obj is None:                        # dry-run
            continue
        # ``SH_<kind>_<net>_s<idx>`` carries no bus either.
        _reconnect(ctx, obj, "bus1", int(rec["bus"]), target, target)

        # Live PF 2025 SP4 probe: shtype 2 = pure capacitor, 1 = reactor.
        # mode_inp='Q' exposes qcapn/qrean as the per-step rated Mvar.
        if kind == "MSC":
            if float(rec["q_mvar"]) >= 0.0:
                raise PFSessionError(f"{target}: MSC must have q_mvar < 0")
            ctx.set_attr(obj, "shtype", 2, label=target)
            rating_attr = "qcapn"
        else:
            if float(rec["q_mvar"]) <= 0.0:
                raise PFSessionError(f"{target}: MSR must have q_mvar > 0")
            ctx.set_attr(obj, "shtype", 1, label=target)
            rating_attr = "qrean"
        ctx.set_attr(obj, "mode_inp", "Q", label=target)
        ctx.set_attr(obj, "ushnm", float(rec["vn_kv"]), label=target)
        ctx.set_attr(obj, rating_attr, q_step, label=target)
        ctx.set_attr(obj, "ncapx", int(rec["max_step"]), label=target)
        ctx.set_attr(obj, "ncapa", int(rec["step"]), label=target)
        ctx.set_attr(obj, "iswitch", 1, label=target)
        ctx.set_attr(obj, "outserv", 0 if rec["in_service"] else 1,
                     label=target)


def sync_full(ctx: SyncContext) -> None:
    """Apply the Phase-4 DSO/coupling delta above ``wind_replace``."""
    model = ctx.doc["model"]
    if len(ctx.meta.hv_networks) != 4:
        raise PFSessionError(
            f"Phase 'full' expects four HV networks, got "
            f"{len(ctx.meta.hv_networks)}"
        )
    if not model["trafo3w"] or not model["shunt"]:
        raise PFSessionError("Phase 'full' requires trafo3w and shunt rows")

    ensure_dso_grids(ctx)
    sync_buses(ctx)
    sync_lines(ctx)
    sync_loads(ctx)
    _outserv_absent_snapshot_loads(ctx)
    _outserv_parent_load_copies(ctx)
    sync_gen_dispatch(ctx, rebase_terminals=False)
    _sync_retained_taps(ctx)
    _outserv_removed_machines(ctx)
    _sync_static_generators(
        ctx,
        allowed_roles=("TSO-WP", "DSO-DER", "DSO-COUPLING-WP"),
    )
    delete_stale_sgens(ctx)
    delete_orphan_station_controllers(ctx)
    sync_trafo3w(ctx)
    sync_shunts(ctx)


# =====================================================================
#  CLI
# =====================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync the PowerFactory model from a dynamic snapshot.")
    parser.add_argument("snapshot", help="Path to the snapshot JSON")
    parser.add_argument(
        "--phase", choices=("base", "wind_replace", "full"),
        default="base", help="Build phase",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--study-case", default=PARITY_STUDY_CASE)
    args = parser.parse_args(argv)

    doc = load_snapshot(args.snapshot)
    snap_phase = doc["provenance"].get("phase")
    if snap_phase != args.phase:
        raise SystemExit(
            f"Snapshot phase is {snap_phase!r}, expected {args.phase!r}"
        )

    app = connect(args.project, study_case=args.study_case)

    # Variation topology is deliberately layered:
    # base -> wind_replace -> full.  PowerFactory reserves object names across
    # inactive variations, so duplicating the wind delta in ``full`` would
    # create decorated names and make deterministic lookup impossible.
    if args.phase == "base":
        if not args.dry_run:
            deactivate_variations_except(app, keep=None)
        ctx = SyncContext(app, doc, dry_run=args.dry_run)
        sync_base(ctx)
    elif args.phase == "wind_replace":
        if not args.dry_run:
            deactivate_variations_except(app, keep=WIND_REPLACE_VARIATION)
            ensure_variation(app, WIND_REPLACE_VARIATION)
            set_variation_active(app, WIND_REPLACE_VARIATION, True)
        ctx = SyncContext(app, doc, dry_run=args.dry_run)
        sync_wind_replace(ctx)
    else:
        if not args.dry_run:
            deactivate_variations_except(app, keep=None)
            ensure_variation(app, WIND_REPLACE_VARIATION)
            set_variation_active(app, WIND_REPLACE_VARIATION, True)
            ensure_variation(app, FULL_MODEL_VARIATION)
            set_variation_active(app, FULL_MODEL_VARIATION, True)
        ctx = SyncContext(app, doc, dry_run=args.dry_run)
        sync_full(ctx)

    print(ctx.report.summary())
    if args.dry_run:
        print("(dry run -- nothing was modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
