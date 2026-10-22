"""
tests/export/test_dynamic_snapshot_roundtrip.py
===============================================
Round-trip validation of the dynamic snapshot exporter (RMS build plan,
Phase 0, Gate 0).

For each build phase the test dumps a snapshot at the default experiment
start time, rebuilds a pandapower network from the JSON alone, reruns the
stored solver options, and requires the recomputed solution to match the
stored one (vm/va to 1e-8, flows to 1e-6 -- see
``export.dynamic_snapshot._ROUNDTRIP_TOL``).  Passing proves the explicit
JSON schema captures every electrically relevant parameter, i.e. the
snapshot is a complete hand-over artefact for the PowerFactory sync.
"""

from __future__ import annotations

import json

import pytest

from export.dynamic_snapshot import (
    SnapshotValueError,
    dump_dynamic_snapshot,
    load_snapshot_to_pandapower,
    meta_from_dict,
    verify_roundtrip,
)
from export.make_snapshots import DEFAULT_T0, PHASES, build_snapshot_state

# One dump per phase, shared across the tests of this module.
_STATE_CACHE = {}


@pytest.fixture(scope="module", params=PHASES)
def dumped(request, tmp_path_factory):
    """Build phase ``request.param`` at t0, dump it, return (state, path)."""
    phase = request.param
    if phase not in _STATE_CACHE:
        state = build_snapshot_state(phase, DEFAULT_T0, verbose=0)
        out_dir = tmp_path_factory.mktemp(f"snap_{phase}")
        path = dump_dynamic_snapshot(
            state.net, state.meta, state.zone_map,
            label=f"roundtrip_{phase}", out_dir=out_dir,
            solver_options=state.solver_options,
            snapshot_time=state.snapshot_time, phase=phase,
        )
        _STATE_CACHE[phase] = (state, path)
    return _STATE_CACHE[phase]


def test_roundtrip_power_flow_matches(dumped):
    """Rebuilt net + stored solver options must reproduce the solution."""
    _state, path = dumped
    report = verify_roundtrip(path)
    assert report.ok, report.summary()


def test_element_counts_survive(dumped):
    """Every serialised table must rebuild with identical row sets."""
    state, path = dumped
    net2, _doc = load_snapshot_to_pandapower(path)
    for table in ("bus", "line", "trafo", "trafo3w", "load", "sgen",
                  "gen", "shunt"):
        orig_idx = sorted(int(i) for i in state.net[table].index)
        new_idx = sorted(int(i) for i in net2[table].index)
        assert orig_idx == new_idx, (
            f"{table}: index mismatch after round-trip "
            f"(orig {len(orig_idx)}, rebuilt {len(new_idx)})"
        )


def test_meta_roundtrip(dumped):
    """The meta block must reconstruct the exact IEEE39NetworkMeta."""
    state, path = dumped
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    meta2 = meta_from_dict(doc["meta"])
    assert meta2 == state.meta


def test_zone_map_roundtrip(dumped):
    state, path = dumped
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    zone_map2 = {int(z): sorted(b) for z, b in doc["zone_map"].items()}
    assert zone_map2 == {z: sorted(b) for z, b in state.zone_map.items()}


def test_removed_generators_match_scenario(dumped):
    """base retains the full 10-machine fleet; wind_replace removes
    G2, G5, G6, G8 (G10 Hydro retained per the 2026-07-17 decision)."""
    state, path = dumped
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    removed = sorted(r["label"] for r in doc["removed_generators"])
    if state.phase == "base":
        assert removed == []
    else:
        assert removed == ["G2", "G5", "G6", "G8"]
        assert "G10" not in removed


def test_missing_required_field_raises(dumped):
    """Fail-Fast: a snapshot with a deleted required field must not load."""
    state, path = dumped
    if state.phase != "base":
        pytest.skip("tamper check only needs one phase")
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    first_bus = next(iter(doc["model"]["bus"]))
    del doc["model"]["bus"][first_bus]["vn_kv"]
    with pytest.raises((SnapshotValueError, KeyError)):
        load_snapshot_to_pandapower(doc)
