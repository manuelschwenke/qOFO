"""
tests/pf/test_naming.py
=======================
The loc_name convention (pf/naming.py) must be total and collision-free on
the shipped reference snapshots: every model element (except template-owned
machines) receives exactly one PF-safe name.

Runs entirely without PowerFactory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from export.dynamic_snapshot import load_snapshot
from pf.naming import (
    TEMPLATE_MACHINE_NAMES,
    build_name_map,
    controller_name,
    machine_template_name,
)

_SNAP_DIR = Path(__file__).resolve().parents[2] / "export" / "snapshots"
_SNAPSHOTS = sorted(p.name for p in _SNAP_DIR.glob("*.json"))


@pytest.mark.parametrize("snapshot_name", _SNAPSHOTS)
def test_name_map_total_and_unique(snapshot_name):
    doc = load_snapshot(_SNAP_DIR / snapshot_name)
    names = build_name_map(doc)

    # Total: every element of every scripted table is named.
    for table in ("bus", "line", "trafo", "trafo3w", "load", "sgen", "shunt"):
        for key in doc["model"][table]:
            assert (table, int(key)) in names, (
                f"{snapshot_name}: {table}[{key}] received no loc_name"
            )

    # Unique (build_name_map raises on collision; double-check here) and
    # PF-safe charset, including the derived controller names.
    all_names = list(names.values())
    all_names += [controller_name(n) for (t, _i), n in names.items()
                  if t == "sgen"]
    assert len(all_names) == len(set(all_names))
    for name in all_names:
        assert re.match(r"^[A-Za-z0-9_]+$", name), name


@pytest.mark.parametrize("snapshot_name", _SNAPSHOTS)
def test_machine_template_mapping_resolves(snapshot_name):
    doc = load_snapshot(_SNAP_DIR / snapshot_name)
    seen = set()
    for rec in doc["model"]["gen"].values():
        tpl = machine_template_name(rec)
        assert tpl in TEMPLATE_MACHINE_NAMES.values()
        assert tpl not in seen, f"duplicate template machine {tpl!r}"
        seen.add(tpl)


def test_snapshots_present():
    """The reference snapshots must exist (they anchor these tests)."""
    assert _SNAPSHOTS, f"no snapshots found in {_SNAP_DIR}"
