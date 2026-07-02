"""
BME Phase 1 — tests for :class:`network.boundary_topology.BoundaryTopology`.

Spec references (docs/BME_STATUS.md; BME spec §3.1–§3.3, §5 Phase 1):
* boundary registry B and per-pair subsets B_ij on the IEEE 39 3-area case
  (audit A3 ground truth: 5 tie lines, 9 boundary buses);
* separator assertion passes on the intact partition and RAISES on a
  deliberately broken partition;
* closure-based ownership (D1): every in-service bus owned by exactly one
  zone, machine-trafo terminal buses inherit their grid bus's zone;
* tie-line loss shares 50/50 (D1);
* cross-zone non-line branches raise.
"""
from __future__ import annotations

import copy

import pandapower as pp
import pytest

from network.boundary_topology import BoundaryTopology
from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39


# Audit A3 ground truth (0-indexed pandapower bus ids).
EXPECTED_REGISTRY = [1, 2, 8, 13, 14, 16, 17, 26, 38]
EXPECTED_TIES = {
    2: (1, 2, 1, 2),     # line 2:  z1 bus 1  — z2 bus 2
    14: (1, 2, 38, 8),   # line 14: z1 bus 38 — z2 bus 8
    25: (1, 3, 26, 16),  # line 25: z1 bus 26 — z3 bus 16
    5: (2, 3, 2, 17),    # line 5:  z2 bus 2  — z3 bus 17
    18: (2, 3, 13, 14),  # line 18: z2 bus 13 — z3 bus 14
}


@pytest.fixture(scope="module")
def ieee39_case():
    net, meta = build_ieee39_net()
    zone_map, _ = fixed_zone_partition_ieee39(net)
    return net, meta, zone_map


@pytest.fixture(scope="module")
def topology(ieee39_case):
    net, _, zone_map = ieee39_case
    return BoundaryTopology(net, zone_map)


def test_registry_matches_audit_a3(topology):
    assert topology.registry == EXPECTED_REGISTRY
    assert topology.registry_pos == {
        b: k for k, b in enumerate(EXPECTED_REGISTRY)
    }


def test_ties_match_audit_a3(topology):
    assert {t.line_idx for t in topology.ties} == set(EXPECTED_TIES)
    for t in topology.ties:
        zi, zj, bi, bj = EXPECTED_TIES[t.line_idx]
        assert (t.zone_i, t.zone_j, t.bus_i, t.bus_j) == (zi, zj, bi, bj)
        assert t.zone_i < t.zone_j  # fixed orientation


def test_b_pairs(topology):
    assert topology.B_pairs[(1, 2)] == [1, 2, 8, 38]
    assert topology.B_pairs[(1, 3)] == [16, 26]
    assert topology.B_pairs[(2, 3)] == [2, 13, 14, 17]
    # Shared boundary bus: IEEE 3 (0-idx 2) serves ties of two zone pairs.
    assert 2 in topology.B_pairs[(1, 2)] and 2 in topology.B_pairs[(2, 3)]


def test_own_and_adjacent_boundary(topology):
    assert topology.own_boundary(1) == [1, 26, 38]
    assert topology.own_boundary(2) == [2, 8, 13]
    assert topology.own_boundary(3) == [14, 16, 17]
    # Adjacent = own ∪ far endpoints of own ties (support of μ, §3.4).
    assert topology.adjacent_boundary(1) == [1, 2, 8, 16, 26, 38]
    assert topology.adjacent_boundary(2) == [1, 2, 8, 13, 14, 17, 38]
    assert topology.adjacent_boundary(3) == [2, 13, 14, 16, 17, 26]


def test_ownership_covers_every_bus(ieee39_case, topology):
    net, _, zone_map = ieee39_case
    for b in net.bus.index:
        if not bool(net.bus.at[b, "in_service"]):
            continue
        owner = topology.bus_owner(int(b))
        assert owner in (1, 2, 3)
    # Partition buses keep their partition zone.
    for z, buses in zone_map.items():
        for b in buses:
            assert topology.bus_owner(int(b)) == z


def test_machine_terminal_buses_inherit_grid_zone(ieee39_case, topology):
    """Generator terminal buses (10.5 kV, absent from the TN partition)
    must be owned by the zone of their grid-side bus (closure, D1)."""
    net, meta, _ = ieee39_case
    assert len(meta.machine_trafo_indices) > 0
    for t in meta.machine_trafo_indices:
        hv = int(net.trafo.at[t, "hv_bus"])
        lv = int(net.trafo.at[t, "lv_bus"])
        assert topology.bus_owner(lv) == topology.bus_owner(hv)


def test_bus_ieee20_removed_by_build(ieee39_case):
    """Audit A3 follow-up: 0-idx bus 19 (IEEE 20) is REMOVED by
    build_ieee39_net's two-trafo-chain collapse (build.py l. 289), which
    is why it appears in no zone list."""
    net, _, _ = ieee39_case
    assert 19 not in net.bus.index


def test_tie_loss_shares_50_50(topology):
    shares = topology.tie_loss_shares()
    assert set(shares) == set(EXPECTED_TIES)
    for line_idx, per_zone in shares.items():
        zi, zj, _, _ = EXPECTED_TIES[line_idx]
        assert per_zone == {zi: 0.5, zj: 0.5}


def test_interior_excludes_own_boundary(topology):
    for z in (1, 2, 3):
        interior = set(topology.interior_buses(z))
        assert interior.isdisjoint(topology.own_boundary(z))
        assert interior  # non-empty
        # Interior plus own boundary = all owned buses.
        assert interior | set(topology.own_boundary(z)) == set(
            topology.zone_buses(z)
        )


def test_broken_partition_raises(ieee39_case):
    """Deliberately broken partition: dropping boundary bus 2 (IEEE 3)
    from zone 2 removes its ties from detection, so removing the
    (reduced) B leaves a component spanning several zones — the
    separator assertion must raise."""
    net, _, zone_map = ieee39_case
    broken = {z: list(buses) for z, buses in zone_map.items()}
    broken[2] = [b for b in broken[2] if b != 2]
    with pytest.raises(ValueError, match="[Ss]eparator"):
        BoundaryTopology(net, broken)


def test_cross_zone_trafo_raises():
    """A cross-zone 2W transformer is a coupling path outside B and must
    raise (§3.2: enlarge B, never weaken the check)."""
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=345.0)
    b1 = pp.create_bus(net, vn_kv=345.0)
    b2 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, length_km=10.0,
        r_ohm_per_km=0.05, x_ohm_per_km=0.3, c_nf_per_km=10.0,
        max_i_ka=1.0,
    )
    pp.create_transformer_from_parameters(
        net, hv_bus=b1, lv_bus=b2, sn_mva=100.0,
        vn_hv_kv=345.0, vn_lv_kv=110.0, vk_percent=12.0,
        vkr_percent=0.3, pfe_kw=0.0, i0_percent=0.0,
    )
    pp.create_load(net, bus=b2, p_mw=10.0, q_mvar=2.0)
    with pytest.raises(ValueError, match="non-line"):
        BoundaryTopology(net, {1: [b0, b1], 2: [b2]})


def test_unknown_zone_raises(topology):
    with pytest.raises(ValueError, match="unknown zone"):
        topology.own_boundary(99)


def test_zone_map_with_unknown_bus_raises(ieee39_case):
    net, _, zone_map = ieee39_case
    bad = {z: list(b) for z, b in zone_map.items()}
    bad[1] = bad[1] + [99999]
    with pytest.raises(ValueError, match="does not exist"):
        BoundaryTopology(net, bad)
